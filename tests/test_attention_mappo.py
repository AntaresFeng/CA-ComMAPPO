from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces
from torch import nn
from xuance.torch.agents import MAPPO_Agents
from xuance.torch.policies import Categorical_MAAC_Policy
from xuance.torch.representations import Basic_MLP

from ca_commappo.networks.attention_mappo import (
    VEHICLE_ATTENTION_REPRESENTATION,
    AttentionCategoricalMAACPolicy,
    AttentionEncoderConfig,
    AttentionMAPPOAgents,
    VehicleAttentionActorRepresentation,
    VehicleAttentionCriticRepresentation,
    _validate_attention_runtime_config,
    resolve_mappo_agent_class,
)
from ca_commappo.training.mappo_highway_intersection import load_configs


ROOT = Path(__file__).resolve().parents[1]
ATTENTION_FORMAL_CONFIG = (
    ROOT / "configs" / "mappo" / "intersection-multi-agent-v1-attention.yaml"
)
ATTENTION_SMOKE_CONFIG = (
    ROOT / "configs" / "mappo" / "intersection-multi-agent-v1-attention-smoke.yaml"
)


def _settings(embed_dim: int = 16) -> AttentionEncoderConfig:
    return AttentionEncoderConfig(
        embed_dim=embed_dim,
        num_heads=4,
        num_layers=2,
        ffn_dim=embed_dim * 2,
        dropout=0.0,
        activation="gelu",
    )


def _actor_observations() -> torch.Tensor:
    observations = torch.zeros(2, 5, 7)
    observations[:, 0] = torch.tensor([1.0, 0.0, 0.0, 0.2, 0.0, 1.0, 0.0])
    observations[:, 1] = torch.tensor([1.0, 0.3, 0.1, -0.2, 0.0, 0.0, 1.0])
    observations[:, 2] = torch.tensor([1.0, -0.4, 0.2, 0.1, 0.0, 0.0, -1.0])
    observations[:, 3] = torch.tensor([1.0, 0.2, -0.5, 0.0, 0.3, -1.0, 0.0])
    return observations


def _critic_state(
    controlled: torch.Tensor | None = None,
    npcs: torch.Tensor | None = None,
    npc_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if controlled is None:
        controlled = torch.tensor(
            [
                [1.0, -0.2, 0.0, 0.3, 0.0, 1.0, 0.0],
                [1.0, 0.4, 0.1, -0.1, 0.2, 0.0, 1.0],
            ]
        )
    if npcs is None:
        npcs = torch.tensor(
            [
                [1.0, 0.1, 0.2, 0.0, -0.2, 1.0, 0.0],
                [1.0, -0.3, 0.4, 0.2, 0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    if npc_mask is None:
        npc_mask = torch.tensor([1.0, 1.0, 0.0])
    return torch.cat([controlled.reshape(-1), npcs.reshape(-1), npc_mask])


def _attention_policy(embed_dim: int = 16) -> AttentionCategoricalMAACPolicy:
    actor_representation = nn.ModuleDict(
        {"agent_0": VehicleAttentionActorRepresentation((35,), _settings(embed_dim))}
    )
    critic_representation = nn.ModuleDict(
        {
            "agent_0": VehicleAttentionCriticRepresentation(
                (38,), 2, _settings(embed_dim)
            )
        }
    )
    return AttentionCategoricalMAACPolicy(
        action_space={"agent_0": spaces.Discrete(3)},
        n_agents=2,
        representation_actor=actor_representation,
        representation_critic=critic_representation,
        actor_hidden_size=[16],
        critic_hidden_size=[16],
        normalize=nn.LayerNorm,
        initialize=torch.nn.init.orthogonal_,
        activation=nn.ReLU,
        device="cpu",
        use_distributed_training=False,
        use_parameter_sharing=True,
        model_keys=["agent_0"],
        use_rnn=False,
        rnn=None,
    )


def _mlp_policy() -> Categorical_MAAC_Policy:
    actor_representation = nn.ModuleDict(
        {
            "agent_0": Basic_MLP(
                (35,), [16], normalize=None, initialize=None, activation=nn.ReLU
            )
        }
    )
    critic_representation = nn.ModuleDict(
        {
            "agent_0": Basic_MLP(
                (38,), [16], normalize=None, initialize=None, activation=nn.ReLU
            )
        }
    )
    return Categorical_MAAC_Policy(
        action_space={"agent_0": spaces.Discrete(3)},
        n_agents=2,
        representation_actor=actor_representation,
        representation_critic=critic_representation,
        actor_hidden_size=[16],
        critic_hidden_size=[16],
        normalize=None,
        initialize=None,
        activation=nn.ReLU,
        device="cpu",
        use_distributed_training=False,
        use_parameter_sharing=True,
        model_keys=["agent_0"],
        use_rnn=False,
        rnn=None,
    )


def _runtime_config(**overrides) -> Namespace:
    values = {
        "representation": VEHICLE_ATTENTION_REPRESENTATION,
        "policy": "Categorical_MAAC_Policy",
        "use_parameter_sharing": True,
        "use_global_state": True,
        "use_rnn": False,
        "flatten_observations": True,
        "attention": {
            "actor": {
                "embed_dim": 16,
                "num_heads": 4,
                "num_layers": 2,
                "ffn_dim": 32,
                "dropout": 0.0,
                "activation": "gelu",
            },
            "critic": {
                "embed_dim": 16,
                "num_heads": 4,
                "num_layers": 2,
                "ffn_dim": 32,
                "dropout": 0.0,
                "activation": "gelu",
            },
        },
        "highway_config": {
            "controlled_vehicles": 2,
            "observation": {
                "observation_config": {
                    "absolute": False,
                    "features": [
                        "presence",
                        "x",
                        "y",
                        "vx",
                        "vy",
                        "cos_h",
                        "sin_h",
                    ],
                }
            },
        },
    }
    values.update(overrides)
    return Namespace(**values)


def test_attention_encoder_config_validates_required_values():
    mapping = {
        "embed_dim": 32,
        "num_heads": 4,
        "num_layers": 2,
        "ffn_dim": 64,
        "dropout": 0.1,
        "activation": "relu",
    }

    assert AttentionEncoderConfig.from_mapping(
        mapping, section="attention.actor"
    ) == AttentionEncoderConfig(32, 4, 2, 64, 0.1, "relu")


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"embed_dim": 30}, "divisible"),
        ({"num_heads": 0}, "positive integer"),
        ({"num_layers": True}, "positive integer"),
        ({"dropout": 1.0}, r"\[0, 1\)"),
        ({"activation": "tanh"}, "relu.*gelu"),
        ({"activation": []}, "relu.*gelu"),
    ],
)
def test_attention_encoder_config_rejects_invalid_values(updates, message):
    mapping = {
        "embed_dim": 32,
        "num_heads": 4,
        "num_layers": 2,
        "ffn_dim": 64,
        "dropout": 0.0,
        "activation": "gelu",
    }
    mapping.update(updates)

    with pytest.raises(ValueError, match=message):
        AttentionEncoderConfig.from_mapping(mapping, section="attention.actor")


def test_attention_encoder_config_rejects_missing_setting():
    with pytest.raises(ValueError, match="attention.actor.ffn_dim"):
        AttentionEncoderConfig.from_mapping(
            {
                "embed_dim": 32,
                "num_heads": 4,
                "num_layers": 2,
                "dropout": 0.0,
                "activation": "gelu",
            },
            section="attention.actor",
        )


def test_actor_attention_is_permutation_and_padding_invariant():
    torch.manual_seed(7)
    representation = VehicleAttentionActorRepresentation((35,), _settings())
    representation.eval()
    observations = _actor_observations()

    baseline = representation(observations.reshape(2, -1))["state"]
    permuted = observations[:, [0, 3, 1, 2, 4]]
    permuted_state = representation(permuted.reshape(2, -1))["state"]
    padded_changed = observations.clone()
    padded_changed[:, 4, 1:] = 1000.0
    padded_state = representation(padded_changed.reshape(2, -1))["state"]

    assert baseline.shape == (2, 16)
    assert baseline.dtype == torch.float32
    assert torch.allclose(baseline, permuted_state, atol=1e-6, rtol=1e-6)
    assert torch.allclose(baseline, padded_state, atol=1e-6, rtol=1e-6)


def test_actor_attention_handles_all_zero_input_and_backpropagates():
    torch.manual_seed(11)
    representation = VehicleAttentionActorRepresentation((35,), _settings())
    zero_state = representation(torch.zeros(3, 35))["state"]
    assert torch.isfinite(zero_state).all()

    observations = _actor_observations().reshape(2, -1).requires_grad_()
    state = representation(observations)["state"]
    state[:, 0].sum().backward()
    assert observations.grad is not None
    assert torch.isfinite(observations.grad).all()
    assert observations.grad.abs().sum() > 0
    assert any(parameter.grad is not None for parameter in representation.parameters())


def test_critic_attention_is_agent_specific_and_npc_permutation_invariant():
    torch.manual_seed(13)
    representation = VehicleAttentionCriticRepresentation((38,), 2, _settings())
    representation.eval()
    state = _critic_state()
    duplicated_state = state.unsqueeze(0).repeat(2, 1)
    agent_ids = torch.eye(2)

    values = representation(duplicated_state, agent_ids=agent_ids)["state"]
    assert values.shape == (2, 16)
    assert torch.isfinite(values).all()
    assert not torch.allclose(values[0], values[1])

    controlled = state[:14].reshape(2, 7)
    npcs = state[14:35].reshape(3, 7)
    mask = state[35:]
    permutation = torch.tensor([1, 0, 2])
    permuted_state = _critic_state(
        controlled,
        npcs[permutation],
        mask[permutation],
    )
    baseline = representation(state.unsqueeze(0), agent_ids=agent_ids[:1])["state"]
    permuted = representation(permuted_state.unsqueeze(0), agent_ids=agent_ids[:1])[
        "state"
    ]
    assert torch.allclose(baseline, permuted, atol=1e-6, rtol=1e-6)


def test_critic_attention_uses_explicit_npc_mask_and_handles_zero_state():
    torch.manual_seed(17)
    representation = VehicleAttentionCriticRepresentation((38,), 2, _settings())
    representation.eval()
    state = _critic_state()
    changed = state.clone()
    changed[28:35] = torch.tensor([1.0, 900.0, -800.0, 700.0, 0.0, 0.0, 1.0])
    agent_id = torch.tensor([[1.0, 0.0]])

    baseline = representation(state.unsqueeze(0), agent_ids=agent_id)["state"]
    masked_changed = representation(changed.unsqueeze(0), agent_ids=agent_id)["state"]
    zero_state = representation(torch.zeros(1, 38), agent_ids=agent_id)["state"]

    assert torch.allclose(baseline, masked_changed, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(zero_state).all()


def test_critic_attention_rejects_invalid_state_layout():
    with pytest.raises(ValueError, match=r"state_dim = 7N \+ 8K"):
        VehicleAttentionCriticRepresentation((37,), 2, _settings())


def test_attention_policy_returns_actions_and_agent_specific_values():
    torch.manual_seed(19)
    policy = _attention_policy()
    observations = {"agent_0": _actor_observations().repeat(2, 1, 1).reshape(4, -1)}
    agent_ids = torch.eye(2).repeat(2, 1)
    critic_states = {"agent_0": _critic_state().unsqueeze(0).repeat(4, 1)}

    _, distributions = policy(observation=observations, agent_ids=agent_ids)
    _, values = policy.get_values(critic_states, agent_ids=agent_ids)

    assert distributions["agent_0"].stochastic_sample().shape == (4,)
    assert values["agent_0"].numel() == 4
    assert torch.isfinite(values["agent_0"]).all()
    actor_parameters = {
        parameter.data_ptr() for parameter in policy.actor_representation.parameters()
    }
    critic_parameters = {
        parameter.data_ptr() for parameter in policy.critic_representation.parameters()
    }
    assert actor_parameters.isdisjoint(critic_parameters)


def test_attention_policy_loads_matching_state_strictly_and_rejects_mlp_state():
    attention_policy = _attention_policy()
    attention_policy.load_state_dict(attention_policy.state_dict(), strict=False)

    with pytest.raises(
        RuntimeError, match="Attention checkpoint architecture mismatch"
    ):
        attention_policy.load_state_dict(_mlp_policy().state_dict(), strict=False)


@pytest.mark.parametrize(
    ("key", "invalid"),
    [
        ("use_parameter_sharing", False),
        ("use_global_state", False),
        ("use_rnn", True),
        ("flatten_observations", False),
        ("policy", "Gaussian_MAAC_Policy"),
    ],
)
def test_attention_runtime_config_rejects_incompatible_modes(key, invalid):
    with pytest.raises(ValueError, match=key):
        _validate_attention_runtime_config(_runtime_config(**{key: invalid}))


def test_attention_runtime_config_rejects_wrong_features_and_absolute_actor_obs():
    wrong_features = _runtime_config(
        highway_config={
            "controlled_vehicles": 2,
            "observation": {
                "observation_config": {
                    "features": [
                        "presence",
                        "y",
                        "x",
                        "vx",
                        "vy",
                        "cos_h",
                        "sin_h",
                    ]
                }
            },
        }
    )
    with pytest.raises(ValueError, match="observation features"):
        _validate_attention_runtime_config(wrong_features)

    absolute = _runtime_config(
        highway_config={
            "controlled_vehicles": 2,
            "observation": {"observation_config": {"absolute": True}},
        }
    )
    with pytest.raises(ValueError, match="ego-relative"):
        _validate_attention_runtime_config(absolute)


def test_attention_configs_define_expected_capacities_and_network_sizes():
    formal = load_configs(config_path=str(ATTENTION_FORMAL_CONFIG))
    smoke = load_configs(config_path=str(ATTENTION_SMOKE_CONFIG))
    formal_actor, formal_critic = _validate_attention_runtime_config(formal)
    smoke_actor, smoke_critic = _validate_attention_runtime_config(smoke)

    assert formal.representation == VEHICLE_ATTENTION_REPRESENTATION
    assert formal_actor.embed_dim == formal_critic.embed_dim == 128
    assert formal_actor.num_layers == formal_critic.num_layers == 2
    assert smoke_actor.embed_dim == smoke_critic.embed_dim == 64
    assert smoke_actor.num_layers == smoke_critic.num_layers == 2
    assert (
        VehicleAttentionCriticRepresentation((254,), 2, formal_critic).npc_capacity
        == 30
    )
    assert (
        VehicleAttentionCriticRepresentation((205,), 3, smoke_critic).npc_capacity == 23
    )


def test_mappo_agent_class_resolution_preserves_mlp_baseline():
    assert (
        resolve_mappo_agent_class(VEHICLE_ATTENTION_REPRESENTATION)
        is AttentionMAPPOAgents
    )
    assert resolve_mappo_agent_class("Basic_MLP") is MAPPO_Agents


def test_attention_representations_reject_wrong_last_dimensions():
    actor = VehicleAttentionActorRepresentation((35,), _settings())
    critic = VehicleAttentionCriticRepresentation((38,), 2, _settings())

    with pytest.raises(ValueError, match="Actor observation last dimension"):
        actor(np.zeros((2, 34), dtype=np.float32))
    with pytest.raises(ValueError, match="Critic state last dimension"):
        critic(np.zeros((2, 37), dtype=np.float32), agent_ids=torch.eye(2))
