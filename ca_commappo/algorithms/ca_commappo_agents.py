"""CA-ComMAPPO: Conflict-Aware Communication MAPPO (the paper's innovation).

Builds on MAPPO with:
* the permutation-invariant social encoder (Phase 1, reused) for ``z_i``;
* a conflict-aware communication module (``ConflictAwareComm``) that replaces
  CommNet's mean aggregation with ``c_i = sum_{j!=i} alpha_ij W_v m_j`` where
  ``alpha_ij = softmax(q_i^T k_j + b_ij)`` carries a kinematic conflict prior.

Design frozen in ``docs/ca_commappo_design.md``. Implementation choice: the comm
is folded into the actor representation (``CAComMAPPOActorRep``) so that
``MAAC_Policy.forward`` and ``MAPPO_Learner`` are reused unchanged (the actor
input dim auto-includes the comm context). ``CA_ComMAPPO_Policy`` /
``CA_ComMAPPO_Learner`` are thin subclasses for config-string naming.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping

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

CA_COMMUNICATION_CONTRACT_VERSION = 1


class CAComMAPPOActorRep(Module):
    """Actor representation = PI social encoder + conflict-aware comm.

    ``forward`` returns ``{'state': concat(z_i, c_i)}`` of dim ``H + comm_dim``,
    so ``MAAC_Policy`` builds the actor with input ``H + comm_dim + n_agents``
    automatically. The critic uses the plain ``PermutationInvariantAttention``
    (no comm), built separately by the agent.

    The ego vehicle (obs index 0) carries *absolute* position/velocity under
    ``absolute: false`` (see design doc §2), so each agent computes its own
    ``ttc_i`` / ``intent_i`` locally (CTDE-compliant).
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

    def forward(self, observations):
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

        alive = torch.ones(n_env, n_agents, device=self.device)
        c = self.comm(m, alive)  # (E, A, comm_dim)

        state = torch.cat([z, c], dim=-1)  # (E, A, H+comm_dim)
        state = state.reshape(batch, -1)  # (B, H+comm_dim)
        return {"state": state}


class CA_ComMAPPO_Policy(MAAC_Policy):
    """Categorical CA-ComMAPPO policy; reuses MAAC_Policy.forward (comm lives in
    the actor representation). Exists for the ``policy: "CA_ComMAPPO_Policy"``
    config string and as the registration target."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "_communication_contract_version",
            torch.tensor(CA_COMMUNICATION_CONTRACT_VERSION, dtype=torch.int64),
            persistent=True,
        )

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
