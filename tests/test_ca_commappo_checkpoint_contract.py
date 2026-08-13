from __future__ import annotations

import pytest
import torch
from gymnasium.spaces import Discrete
from xuance.torch import Module, ModuleDict

from ca_commappo.algorithms.ca_commappo_agents import CA_ComMAPPO_Policy


class _IdentityRepresentation(Module):
    def __init__(self, width: int):
        super().__init__()
        self.output_shapes = {"state": (width,)}

    def forward(self, observations):
        return {"state": torch.as_tensor(observations, dtype=torch.float32)}


def _policy() -> CA_ComMAPPO_Policy:
    actor = ModuleDict({"agent_0": _IdentityRepresentation(3)})
    critic = ModuleDict({"agent_0": _IdentityRepresentation(3)})
    return CA_ComMAPPO_Policy(
        action_space={"agent_0": Discrete(3)},
        n_agents=1,
        representation_actor=actor,
        representation_critic=critic,
        actor_hidden_size=[4],
        critic_hidden_size=[4],
        normalize=None,
        initialize=None,
        activation=torch.nn.ReLU,
        device="cpu",
        use_distributed_training=False,
        use_parameter_sharing=True,
        model_keys=["agent_0"],
        use_rnn=False,
        rnn=None,
    )


def test_current_communication_contract_state_dict_round_trips():
    source = _policy()
    target = _policy()

    target.load_state_dict(source.state_dict())

    assert torch.equal(
        target._communication_contract_version,
        source._communication_contract_version,
    )


def test_legacy_checkpoint_without_contract_version_is_rejected():
    policy = _policy()
    legacy = {
        key: value
        for key, value in policy.state_dict().items()
        if key != "_communication_contract_version"
    }

    with pytest.raises(RuntimeError, match="Legacy CA-ComMAPPO checkpoint"):
        policy.load_state_dict(legacy, strict=False)


def test_legacy_unconstrained_lambda_key_is_rejected():
    policy = _policy()
    legacy = dict(policy.state_dict())
    legacy["actor_representation.agent_0.comm.lam"] = torch.tensor(1.0)

    with pytest.raises(RuntimeError, match="unconstrained lam"):
        policy.load_state_dict(legacy, strict=False)
