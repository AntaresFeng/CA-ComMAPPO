"""Legacy CA-ComMAPPO integration with XuanCe's categorical MAPPO.

Communication is embedded in the actor representation. Policy and learner
subclasses provide the active-agent mask context while reusing PPO updates.
The intent/TTC research design is retired; keeping this implementation does not
prescribe the next algorithm. See ``docs/current_experiment_status.md``.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager

import numpy as np
import torch
from gymnasium import Space
from xuance.environment import (
    DummyVecMultiAgentEnv,
    SubprocVecMultiAgentEnv,
    space2shape,
)
from xuance.torch import Module, ModuleDict
from xuance.torch.agents.multi_agent_rl.mappo_agents import MAPPO_Agents
from xuance.torch.learners.multi_agent_rl.mappo_learner import MAPPO_Learner
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.policies.categorical_marl import MAAC_Policy
from xuance.torch.utils import ActivationFunctions, NormalizeFunctions

from ca_commappo.algorithms.conflict_aware_comm import TTC_MAX, ConflictAwareComm
from ca_commappo.algorithms.representations import (
    _EGO_INDEX,
    _N_FEATURES,
    _VEHICLES_COUNT,
    PermutationInvariantAttention,
)

CA_COMMUNICATION_CONTRACT_VERSION = 2


class CAComMAPPOActorRep(Module):
    """Actor representation = PI social encoder + conflict-aware comm.

    ``forward`` returns ``{'state': concat(z_i, c_i)}`` of dim ``H + comm_dim``,
    so ``MAAC_Policy`` builds the actor with input ``H + comm_dim + n_agents``
    automatically. The critic uses the plain ``PermutationInvariantAttention``
    (no comm), built separately by the agent.

    The ego row carries normalized world position/velocity even with
    ``absolute: false``. The legacy proxy head uses this fixed observation
    convention; it does not infer route intent or pairwise collision time.
    """

    def __init__(
        self,
        input_shape,
        hidden_sizes,
        normalize=None,
        initialize=None,
        activation=None,
        device=None,
        n_agents: int = 1,
        comm_dim: int | None = None,
        recede_penalty: float = 10.0,
        ttc_max: float = TTC_MAX,
        conflict_lambda_init: float = 1.0,
        use_conflict_prior: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_sizes = list(hidden_sizes)
        self.device = device
        self.n_agents = int(n_agents)
        self.ttc_max = float(ttc_max)
        self._runtime_agent_mask: torch.Tensor | np.ndarray | None = None
        self._runtime_trace_sink: Callable[[dict], None] | None = None

        hidden = self.hidden_sizes[-1]
        self.comm_dim = int(comm_dim) if comm_dim is not None else hidden

        self.encoder = PermutationInvariantAttention(
            input_shape=input_shape,
            hidden_sizes=self.hidden_sizes,
            normalize=normalize,
            initialize=initialize,
            activation=activation,
            device=device,
        )
        # message m_i = [z_i (H), ttc_i (1), intent_i (1)] -> msg_dim = H + 2.
        self.comm = ConflictAwareComm(
            msg_dim=hidden + 2,
            comm_dim=self.comm_dim,
            n_agents=self.n_agents,
            ttc_slot=hidden,
            intent_slot=hidden + 1,
            recede_penalty=recede_penalty,
            lam_init=conflict_lambda_init,
            use_conflict_prior=use_conflict_prior,
            device=device,
        )
        self.output_shapes = {"state": (hidden + self.comm_dim,)}

    @contextmanager
    def communication_mask(self, agent_mask) -> Iterator[None]:
        previous = self._runtime_agent_mask
        self._runtime_agent_mask = agent_mask
        try:
            yield
        finally:
            self._runtime_agent_mask = previous

    @contextmanager
    def communication_trace(
        self, sink: Callable[[dict], None] | None
    ) -> Iterator[None]:
        previous = self._runtime_trace_sink
        self._runtime_trace_sink = sink
        try:
            yield
        finally:
            self._runtime_trace_sink = previous

    def _agent_mask(self, agent_mask, n_env: int) -> torch.Tensor:
        source = self._runtime_agent_mask if agent_mask is None else agent_mask
        if source is None:
            return torch.ones(n_env, self.n_agents, device=self.device)
        mask = torch.as_tensor(source, dtype=torch.float32, device=self.device)
        if mask.numel() != n_env * self.n_agents:
            raise ValueError(
                "CA-ComMAPPO agent mask must contain one value per "
                f"environment/agent pair, got shape {tuple(mask.shape)}."
            )
        return mask.reshape(n_env, self.n_agents)

    def _ttc_intent(self, ego: torch.Tensor):
        """ego: (..., 7) = [presence, x, y, vx, vy, cos_h, sin_h] (normalized)."""
        pos = ego[..., 1:3]
        vel = ego[..., 3:5]
        heading = ego[..., 5:7]
        eps = 1e-6
        d = torch.linalg.norm(pos, dim=-1)
        speed = torch.linalg.norm(vel, dim=-1)
        dir_to_c = -pos / (d.unsqueeze(-1) + eps)  # unit vector i -> C(0,0)
        approach = (heading * dir_to_c).sum(dim=-1)  # >0 => heading toward C
        ttc_raw = d / (speed + eps)
        not_approaching = approach <= 0
        ttc = torch.where(
            not_approaching, torch.full_like(ttc_raw, self.ttc_max), ttc_raw
        )
        ttc = torch.clamp(ttc, min=0.0, max=self.ttc_max)
        ttc = ttc / (ttc + 1.0)
        intent = (~not_approaching).float()
        return ttc, intent

    def forward(self, observations, agent_mask=None):
        x = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch = x.shape[0]  # n_env * n_agents
        n_agents = self.n_agents
        if batch % n_agents != 0:
            raise ValueError(
                f"CAComMAPPOActorRep batch {batch} not divisible by n_agents {n_agents}."
            )
        n_env = batch // n_agents

        z = self.encoder(x)["state"]  # (B, H)
        views = x.reshape(batch, _VEHICLES_COUNT, _N_FEATURES)
        ego = views[:, _EGO_INDEX, :]  # (B, 7)
        ttc, intent = self._ttc_intent(ego)  # (B,), (B,)

        z = z.reshape(n_env, n_agents, -1)
        ttc = ttc.reshape(n_env, n_agents)
        intent = intent.reshape(n_env, n_agents)
        m = torch.cat(
            [z, ttc.unsqueeze(-1), intent.unsqueeze(-1)], dim=-1
        )  # (E, A, H+2)

        alive = self._agent_mask(agent_mask, n_env)
        c = self.comm(m, alive)  # (E, A, comm_dim)
        if self._runtime_trace_sink is not None:
            with torch.no_grad():
                alpha = self.comm.attention(m.detach(), alive.detach())
            self._runtime_trace_sink(
                {
                    "active_mask": alive.detach().cpu().clone(),
                    "scaled_ttc": ttc.detach().cpu().clone(),
                    "intent": intent.detach().cpu().clone(),
                    "attention": alpha.detach().cpu().clone(),
                    "effective_lambda": self.comm.lambda_value.detach().cpu().clone(),
                    "use_conflict_prior": self.comm.use_conflict_prior,
                }
            )

        state = torch.cat([z, c], dim=-1)  # (E, A, H+comm_dim)
        state = state.reshape(batch, -1)  # (B, H+comm_dim)
        return {"state": state}


class CA_ComMAPPO_Policy(MAAC_Policy):
    """Categorical CA-ComMAPPO policy; reuses MAAC_Policy.forward (comm lives in
    the actor representation). Exists for the ``policy: "CA_ComMAPPO_Policy"``
    config string and as the registration target."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._communication_agent_mask = None
        self._communication_trace_sink = None
        self.register_buffer(
            "_communication_contract_version",
            torch.tensor(CA_COMMUNICATION_CONTRACT_VERSION, dtype=torch.int64),
            persistent=True,
        )

    @contextmanager
    def communication_mask(self, agent_mask) -> Iterator[None]:
        previous = self._communication_agent_mask
        self._communication_agent_mask = agent_mask
        try:
            yield
        finally:
            self._communication_agent_mask = previous

    @contextmanager
    def communication_trace(
        self, sink: Callable[[dict], None] | None
    ) -> Iterator[None]:
        previous = self._communication_trace_sink
        self._communication_trace_sink = sink
        try:
            yield
        finally:
            self._communication_trace_sink = previous

    def forward(self, *args, **kwargs):
        with ExitStack() as stack:
            for representation in self.actor_representation.values():
                stack.enter_context(
                    representation.communication_mask(self._communication_agent_mask)
                )
                stack.enter_context(
                    representation.communication_trace(self._communication_trace_sink)
                )
            return super().forward(*args, **kwargs)

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        version_key = "_communication_contract_version"
        if version_key not in state_dict:
            raise RuntimeError(
                "Legacy CA-ComMAPPO checkpoint is incompatible with the scaled-TTC "
                "and non-negative-lambda communication contract; retrain it."
            )
        version = int(torch.as_tensor(state_dict[version_key]).item())
        if version != CA_COMMUNICATION_CONTRACT_VERSION:
            raise RuntimeError(
                "Unsupported CA-ComMAPPO communication contract version "
                f"{version}; expected {CA_COMMUNICATION_CONTRACT_VERSION}."
            )
        if any(key.endswith(".comm.lam") for key in state_dict):
            raise RuntimeError(
                "Legacy CA-ComMAPPO checkpoint contains the unconstrained lam "
                "parameter and cannot be loaded; retrain it."
            )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


class CA_ComMAPPO_Learner(MAPPO_Learner):
    """CA-ComMAPPO learner; reuses MAPPO's clipped PPO loss (comm not in critic
    yet, per design doc §9 future work)."""

    def update(self, sample):
        sample_tensor = self.build_training_data(
            sample=sample,
            use_parameter_sharing=self.use_parameter_sharing,
            use_actions_mask=self.use_actions_mask,
            use_global_state=self.use_global_state,
        )
        key = self.model_keys[0]
        communication_mask = sample_tensor["agent_mask"][key].reshape(
            sample_tensor["batch_size"], self.n_agents
        )
        with self.policy.communication_mask(communication_mask):
            info = super().update(sample)
        comm_modules = [
            representation.comm
            for representation in self.policy.actor_representation.values()
        ]
        lambda_values = torch.stack(
            [comm.lambda_value.detach() for comm in comm_modules]
        )
        lambda_grads = [
            comm.raw_lambda.grad.detach()
            for comm in comm_modules
            if comm.raw_lambda.grad is not None
        ]
        info["diagnostics/conflict_lambda"] = float(lambda_values.mean().item())
        info["diagnostics/conflict_lambda_raw_grad"] = (
            float(torch.stack(lambda_grads).abs().mean().item())
            if lambda_grads
            else 0.0
        )
        return info


class CA_ComMAPPO_Agents(MAPPO_Agents):
    """MAPPO agents with a CA-ComMAPPO policy build.

    Overrides ``_build_policy`` to:
    * build the actor representation manually (``CAComMAPPOActorRep``) with
      ``n_agents`` (``_build_representation`` does not pass n_agents);
    * build the critic representation via the plain ``PermutationInvariantAttention``;
    * bypass ``MAPPO_Agents._build_policy``'s policy-string whitelist by looking
      up ``CA_ComMAPPO_Policy`` directly.
    """

    def __init__(
        self,
        config: Namespace,
        envs: DummyVecMultiAgentEnv | SubprocVecMultiAgentEnv | None = None,
        num_agents: int | None = None,
        agent_keys: list[str] | None = None,
        state_space: Space | None = None,
        observation_space: Space | None = None,
        action_space: Space | None = None,
        callback=None,
    ):
        super().__init__(
            config,
            envs,
            num_agents,
            agent_keys,
            state_space,
            observation_space,
            action_space,
            callback,
        )
        self._rollout_agent_mask: np.ndarray | None = None

    def _all_active_mask(self, n_env: int) -> np.ndarray:
        return np.ones((n_env, self.n_agents), dtype=np.float32)

    def _mask_from_infos(
        self,
        infos,
        n_env: int,
        *,
        use_next_mask: bool = False,
    ) -> np.ndarray:
        if infos is None:
            if self._rollout_agent_mask is None or self._rollout_agent_mask.shape != (
                n_env,
                self.n_agents,
            ):
                return self._all_active_mask(n_env)
            return self._rollout_agent_mask.copy()
        if len(infos) != n_env:
            raise ValueError(
                f"Expected {n_env} info entries for CA-ComMAPPO, got {len(infos)}."
            )
        mask = self._all_active_mask(n_env)
        for i_env, info in enumerate(infos):
            if "reset_obs" in info:
                continue
            if use_next_mask:
                values = info.get("next_agent_mask", info.get("agent_mask"))
            else:
                values = info.get("agent_mask", info.get("next_agent_mask"))
            if values is None:
                continue
            if isinstance(values, Mapping):
                values = [values[key] for key in self.agent_keys]
            values = np.asarray(values, dtype=np.float32).reshape(-1)
            if values.size != self.n_agents:
                raise ValueError(
                    "CA-ComMAPPO info mask must contain one value per agent, "
                    f"got {values.size}."
                )
            mask[i_env] = values
        return mask

    def action(
        self,
        obs_dict,
        state=None,
        avail_actions_dict=None,
        rnn_hidden_actor=None,
        rnn_hidden_critic=None,
        test_mode=False,
        info=None,
        **kwargs,
    ):
        communication_mask = self._mask_from_infos(info, len(obs_dict))
        with self.policy.communication_mask(communication_mask):
            return super().action(
                obs_dict=obs_dict,
                state=state,
                avail_actions_dict=avail_actions_dict,
                rnn_hidden_actor=rnn_hidden_actor,
                rnn_hidden_critic=rnn_hidden_critic,
                test_mode=test_mode,
                **kwargs,
            )

    @contextmanager
    def communication_trace(self, sink: Callable[[dict], None]) -> Iterator[None]:
        with self.policy.communication_trace(sink):
            yield

    def store_experience(
        self,
        obs_dict,
        avail_actions,
        actions_dict,
        log_pi_a,
        rewards_dict,
        values_dict,
        terminals_dict,
        info,
        **kwargs,
    ) -> None:
        super().store_experience(
            obs_dict,
            avail_actions,
            actions_dict,
            log_pi_a,
            rewards_dict,
            values_dict,
            terminals_dict,
            info,
            **kwargs,
        )
        self._rollout_agent_mask = self._mask_from_infos(
            info,
            len(obs_dict),
            use_next_mask=True,
        )

    def _build_policy(self) -> Module:
        normalize_fn = (
            NormalizeFunctions[self.config.normalize]
            if hasattr(self.config, "normalize")
            else None
        )
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device

        # Actor representation: PI encoder + conflict-aware comm (built manually).
        a_representation = ModuleDict()
        for key in self.model_keys:
            a_representation[key] = CAComMAPPOActorRep(
                input_shape=space2shape(self.observation_space[key]),
                hidden_sizes=self.config.representation_hidden_size,
                normalize=normalize_fn,
                initialize=initializer,
                activation=activation,
                device=device,
                n_agents=self.n_agents,
                comm_dim=self.config.representation_hidden_size[-1],
                recede_penalty=getattr(self.config, "recede_penalty", 10.0),
                ttc_max=getattr(self.config, "ttc_max", TTC_MAX),
                conflict_lambda_init=getattr(self.config, "conflict_lambda_init", 1.0),
                use_conflict_prior=getattr(self.config, "use_conflict_prior", True),
            )

        # Critic representation: plain PI encoder (no comm).
        if self.use_global_state:
            space_critic_in = {
                k: (sum(self.state_space.shape),) for k in self.agent_keys
            }
        else:
            dim_obs_all = sum(
                sum(self.observation_space[k].shape) for k in self.agent_keys
            )
            space_critic_in = {k: (dim_obs_all,) for k in self.agent_keys}
        c_representation = self._build_representation(
            "PermutationInvariantAttention", space_critic_in, self.config
        )

        policy_settings = {
            "action_space": self.action_space,
            "n_agents": self.n_agents,
            "representation_actor": a_representation,
            "representation_critic": c_representation,
            "actor_hidden_size": self.config.actor_hidden_size,
            "critic_hidden_size": self.config.critic_hidden_size,
            "normalize": normalize_fn,
            "initialize": initializer,
            "activation": activation,
            "device": device,
            "use_distributed_training": self.distributed_training,
            "use_parameter_sharing": self.use_parameter_sharing,
            "model_keys": self.model_keys,
            "use_rnn": self.use_rnn,
            "rnn": self.config.rnn if self.use_rnn else None,
        }
        policy = REGISTRY_Policy["CA_ComMAPPO_Policy"](**policy_settings)
        self.continuous_control = False
        return policy
