"""Vehicle-token attention representations and XuanCe MAPPO integration."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import torch
from gymnasium import spaces
from torch import nn
from xuance.torch.agents import MAPPO_Agents
from xuance.torch.policies import Categorical_MAAC_Policy
from xuance.torch.utils import ActivationFunctions, NormalizeFunctions

from ca_commappo.envs.highway_intersection_wrapper import (
    GLOBAL_OBSERVATION_FEATURES,
    build_intersection_config,
)


VEHICLE_ATTENTION_REPRESENTATION = "VehicleAttention"
VEHICLE_FEATURE_DIM = len(GLOBAL_OBSERVATION_FEATURES)
PRESENCE_INDEX = GLOBAL_OBSERVATION_FEATURES.index("presence")


@dataclass(frozen=True)
class AttentionEncoderConfig:
    embed_dim: int
    num_heads: int
    num_layers: int
    ffn_dim: int
    dropout: float
    activation: str

    @classmethod
    def from_mapping(cls, value: Any, *, section: str) -> AttentionEncoderConfig:
        if not isinstance(value, Mapping):
            raise ValueError(f"{section} must be a mapping")

        embed_dim = _required_positive_int(value, "embed_dim", section)
        num_heads = _required_positive_int(value, "num_heads", section)
        num_layers = _required_positive_int(value, "num_layers", section)
        ffn_dim = _required_positive_int(value, "ffn_dim", section)
        if embed_dim % num_heads != 0:
            raise ValueError(f"{section}.embed_dim must be divisible by num_heads")

        dropout_value = _required_value(value, "dropout", section)
        if isinstance(dropout_value, bool) or not isinstance(dropout_value, Real):
            raise ValueError(f"{section}.dropout must be a number in [0, 1)")
        dropout = float(dropout_value)
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"{section}.dropout must be a number in [0, 1)")

        activation_value = _required_value(value, "activation", section)
        if not isinstance(activation_value, str) or activation_value not in {
            "relu",
            "gelu",
        }:
            raise ValueError(f"{section}.activation must be 'relu' or 'gelu'")

        return cls(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            activation=activation_value,
        )


def _required_value(value: Mapping[str, Any], key: str, section: str) -> Any:
    if key not in value:
        raise ValueError(f"Missing required attention setting: {section}.{key}")
    return value[key]


def _required_positive_int(value: Mapping[str, Any], key: str, section: str) -> int:
    configured = _required_value(value, key, section)
    if (
        isinstance(configured, bool)
        or not isinstance(configured, int)
        or configured <= 0
    ):
        raise ValueError(f"{section}.{key} must be a positive integer")
    return configured


def _attention_settings(
    config: Namespace,
) -> tuple[AttentionEncoderConfig, AttentionEncoderConfig]:
    attention = getattr(config, "attention", None)
    if not isinstance(attention, Mapping):
        raise ValueError("attention must be a mapping with actor and critic sections")
    actor = AttentionEncoderConfig.from_mapping(
        _required_value(attention, "actor", "attention"),
        section="attention.actor",
    )
    critic = AttentionEncoderConfig.from_mapping(
        _required_value(attention, "critic", "attention"),
        section="attention.critic",
    )
    return actor, critic


class _VehicleTransformerEncoder(nn.Module):
    def __init__(
        self,
        settings: AttentionEncoderConfig,
        *,
        device: str | int | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(
            VEHICLE_FEATURE_DIM,
            settings.embed_dim,
            device=device,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=settings.embed_dim,
            nhead=settings.num_heads,
            dim_feedforward=settings.ffn_dim,
            dropout=settings.dropout,
            activation=settings.activation,
            batch_first=True,
            norm_first=True,
            device=device,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=settings.num_layers,
            norm=nn.LayerNorm(settings.embed_dim, device=device),
            enable_nested_tensor=False,
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.modules():
            if isinstance(module, nn.MultiheadAttention):
                if module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                if module.bias_k is not None:
                    nn.init.zeros_(module.bias_k)
                if module.bias_v is not None:
                    nn.init.zeros_(module.bias_v)

    def forward(
        self,
        vehicle_features: torch.Tensor,
        role_embeddings: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.input_projection(vehicle_features) + role_embeddings
        return self.encoder(tokens, src_key_padding_mask=padding_mask)


class VehicleAttentionActorRepresentation(nn.Module):
    """Encode a flat ego-relative Kinematics observation as a vehicle set."""

    def __init__(
        self,
        input_shape: tuple[int, ...],
        settings: AttentionEncoderConfig,
        *,
        device: str | int | torch.device | None = None,
    ) -> None:
        super().__init__()
        if len(input_shape) != 1 or input_shape[0] % VEHICLE_FEATURE_DIM != 0:
            raise ValueError(
                "VehicleAttention actor input must be a flat sequence of 7-feature vehicles"
            )
        self.input_dim = input_shape[0]
        self.vehicle_count = self.input_dim // VEHICLE_FEATURE_DIM
        if self.vehicle_count <= 0:
            raise ValueError(
                "VehicleAttention actor requires at least one vehicle slot"
            )

        self.device = device
        self.output_shapes = {"state": (settings.embed_dim,)}
        self.transformer = _VehicleTransformerEncoder(settings, device=device)
        self.role_embeddings = nn.Embedding(2, settings.embed_dim, device=device)
        nn.init.normal_(self.role_embeddings.weight, mean=0.0, std=0.02)

    def forward(self, observations: Any) -> dict[str, torch.Tensor]:
        tensor = torch.as_tensor(
            observations,
            dtype=torch.float32,
            device=self.device,
        )
        if tensor.shape[-1] != self.input_dim:
            raise ValueError(
                f"Actor observation last dimension must be {self.input_dim}, "
                f"got {tensor.shape[-1]}"
            )
        leading_shape = tensor.shape[:-1]
        vehicles = tensor.reshape(-1, self.vehicle_count, VEHICLE_FEATURE_DIM)

        padding_mask = vehicles[..., PRESENCE_INDEX] <= 0.5
        padding_mask = padding_mask.clone()
        padding_mask[:, 0] = False

        role_ids = torch.ones(
            self.vehicle_count,
            dtype=torch.long,
            device=vehicles.device,
        )
        role_ids[0] = 0
        roles = self.role_embeddings(role_ids).unsqueeze(0)
        encoded = self.transformer(vehicles, roles, padding_mask)
        state = encoded[:, 0].reshape(*leading_shape, self.output_shapes["state"][0])
        return {"state": state}


class VehicleAttentionCriticRepresentation(nn.Module):
    """Encode centralized controlled/NPC vehicle tokens for an agent-specific value."""

    def __init__(
        self,
        input_shape: tuple[int, ...],
        n_agents: int,
        settings: AttentionEncoderConfig,
        *,
        device: str | int | torch.device | None = None,
    ) -> None:
        super().__init__()
        if len(input_shape) != 1:
            raise ValueError("VehicleAttention critic input must be one-dimensional")
        if isinstance(n_agents, bool) or not isinstance(n_agents, int) or n_agents <= 0:
            raise ValueError(
                "VehicleAttention critic n_agents must be a positive integer"
            )

        self.input_dim = input_shape[0]
        self.n_agents = n_agents
        npc_dimensions = self.input_dim - n_agents * VEHICLE_FEATURE_DIM
        npc_stride = VEHICLE_FEATURE_DIM + 1
        if npc_dimensions <= 0 or npc_dimensions % npc_stride != 0:
            raise ValueError(
                "VehicleAttention critic state must satisfy state_dim = 7N + 8K"
            )
        self.npc_capacity = npc_dimensions // npc_stride
        self.device = device
        self.output_shapes = {"state": (settings.embed_dim,)}

        self.transformer = _VehicleTransformerEncoder(settings, device=device)
        self.controlled_slot_embeddings = nn.Embedding(
            n_agents,
            settings.embed_dim,
            device=device,
        )
        self.npc_type_embedding = nn.Parameter(
            torch.empty(1, 1, settings.embed_dim, device=device)
        )
        nn.init.normal_(self.controlled_slot_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.npc_type_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        observations: Any,
        *,
        agent_ids: Any,
    ) -> dict[str, torch.Tensor]:
        tensor = torch.as_tensor(
            observations,
            dtype=torch.float32,
            device=self.device,
        )
        if tensor.shape[-1] != self.input_dim:
            raise ValueError(
                f"Critic state last dimension must be {self.input_dim}, "
                f"got {tensor.shape[-1]}"
            )
        ids = torch.as_tensor(agent_ids, dtype=torch.float32, device=self.device)
        if ids.shape[-1] != self.n_agents:
            raise ValueError(
                f"Critic agent_ids last dimension must be {self.n_agents}, "
                f"got {ids.shape[-1]}"
            )

        leading_shape = tensor.shape[:-1]
        flat_state = tensor.reshape(-1, self.input_dim)
        flat_ids = ids.reshape(-1, self.n_agents)
        if flat_state.shape[0] != flat_ids.shape[0]:
            raise ValueError("Critic state and agent_ids batch dimensions must match")

        controlled_end = self.n_agents * VEHICLE_FEATURE_DIM
        npc_end = controlled_end + self.npc_capacity * VEHICLE_FEATURE_DIM
        controlled = flat_state[:, :controlled_end].reshape(
            -1, self.n_agents, VEHICLE_FEATURE_DIM
        )
        npcs = flat_state[:, controlled_end:npc_end].reshape(
            -1, self.npc_capacity, VEHICLE_FEATURE_DIM
        )
        npc_mask = flat_state[:, npc_end:]
        vehicles = torch.cat([controlled, npcs], dim=1)

        controlled_padding = controlled[..., PRESENCE_INDEX] <= 0.5
        npc_padding = npc_mask <= 0.5
        padding_mask = torch.cat([controlled_padding, npc_padding], dim=1)
        query_indices = flat_ids.argmax(dim=-1)
        batch_indices = torch.arange(flat_state.shape[0], device=flat_state.device)
        padding_mask = padding_mask.clone()
        padding_mask[batch_indices, query_indices] = False

        controlled_roles = self.controlled_slot_embeddings.weight.unsqueeze(0).expand(
            flat_state.shape[0], -1, -1
        )
        npc_roles = self.npc_type_embedding.expand(
            flat_state.shape[0], self.npc_capacity, -1
        )
        roles = torch.cat([controlled_roles, npc_roles], dim=1)
        encoded = self.transformer(vehicles, roles, padding_mask)
        state = encoded[batch_indices, query_indices].reshape(
            *leading_shape, self.output_shapes["state"][0]
        )
        return {"state": state}


class AttentionCategoricalMAACPolicy(Categorical_MAAC_Policy):
    """Categorical MAAC policy that makes critic attention agent-specific."""

    def get_values(
        self,
        observation: dict[str, Any],
        agent_ids: torch.Tensor | None = None,
        agent_key: str | None = None,
        rnn_hidden: dict[str, Any] | None = None,
    ) -> tuple[dict[str, list[None]], dict[str, torch.Tensor]]:
        if agent_ids is None:
            raise ValueError("VehicleAttention critic requires agent_ids")
        if self.use_rnn:
            raise ValueError(
                "VehicleAttention does not support recurrent representations"
            )

        rnn_hidden_new: dict[str, list[None]] = {}
        values: dict[str, torch.Tensor] = {}
        agent_list = self.model_keys if agent_key is None else [agent_key]
        for key in agent_list:
            outputs = self.critic_representation[key](
                observation[key],
                agent_ids=agent_ids,
            )
            rnn_hidden_new[key] = [None, None]
            critic_input = torch.concat([outputs["state"], agent_ids], dim=-1)
            values[key] = self.critic[key](critic_input)
        return rnn_hidden_new, values

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        del strict
        try:
            return super().load_state_dict(state_dict, strict=True, assign=assign)
        except RuntimeError as exc:
            raise RuntimeError(
                "Attention checkpoint architecture mismatch; use matching N, K, "
                "feature order, and attention settings."
            ) from exc


class AttentionMAPPOAgents(MAPPO_Agents):
    """MAPPO agent that builds project-owned vehicle attention representations."""

    def _build_policy(self) -> nn.Module:
        actor_settings, critic_settings = _validate_attention_runtime_config(
            self.config
        )
        _validate_attention_spaces(
            observation_space=self.observation_space,
            action_space=self.action_space,
            state_space=self.state_space,
            model_keys=self.model_keys,
        )

        actor_representations = nn.ModuleDict()
        critic_representations = nn.ModuleDict()
        critic_input_shape = tuple(self.state_space.shape)
        for key in self.model_keys:
            actor_representations[key] = VehicleAttentionActorRepresentation(
                tuple(self.observation_space[key].shape),
                actor_settings,
                device=self.device,
            )
            critic_representations[key] = VehicleAttentionCriticRepresentation(
                critic_input_shape,
                self.n_agents,
                critic_settings,
                device=self.device,
            )

        normalize = (
            NormalizeFunctions[self.config.normalize]
            if hasattr(self.config, "normalize")
            else None
        )
        policy = AttentionCategoricalMAACPolicy(
            action_space=self.action_space,
            n_agents=self.n_agents,
            representation_actor=actor_representations,
            representation_critic=critic_representations,
            actor_hidden_size=self.config.actor_hidden_size,
            critic_hidden_size=self.config.critic_hidden_size,
            normalize=normalize,
            initialize=torch.nn.init.orthogonal_,
            activation=ActivationFunctions[self.config.activation],
            device=self.device,
            use_distributed_training=self.distributed_training,
            use_parameter_sharing=self.use_parameter_sharing,
            model_keys=self.model_keys,
            use_rnn=False,
            rnn=None,
        )
        self.continuous_control = False
        return policy


def _validate_attention_runtime_config(
    config: Namespace,
) -> tuple[AttentionEncoderConfig, AttentionEncoderConfig]:
    required_values = {
        "representation": VEHICLE_ATTENTION_REPRESENTATION,
        "policy": "Categorical_MAAC_Policy",
        "use_parameter_sharing": True,
        "use_global_state": True,
        "use_rnn": False,
        "flatten_observations": True,
    }
    for key, expected in required_values.items():
        actual = getattr(config, key, None)
        if actual != expected:
            raise ValueError(
                f"VehicleAttention requires {key}={expected!r}, got {actual!r}"
            )

    highway_config = build_intersection_config(
        getattr(config, "highway_config", None) or {}
    )
    observation_config = highway_config["observation"]["observation_config"]
    features = tuple(observation_config.get("features", ()))
    if features != GLOBAL_OBSERVATION_FEATURES:
        raise ValueError(
            "VehicleAttention requires observation features in the order "
            f"{GLOBAL_OBSERVATION_FEATURES!r}"
        )
    if bool(observation_config.get("absolute", False)):
        raise ValueError("VehicleAttention actor requires ego-relative observations")
    return _attention_settings(config)


def _validate_attention_spaces(
    *,
    observation_space: dict[str, spaces.Space],
    action_space: dict[str, spaces.Space],
    state_space: spaces.Space,
    model_keys: list[str],
) -> None:
    if not isinstance(state_space, spaces.Box) or len(state_space.shape) != 1:
        raise ValueError("VehicleAttention requires a one-dimensional Box state_space")
    for key in model_keys:
        observation = observation_space[key]
        if not isinstance(observation, spaces.Box) or len(observation.shape) != 1:
            raise ValueError(
                "VehicleAttention requires flat one-dimensional Box observations"
            )
        if not isinstance(action_space[key], spaces.Discrete):
            raise ValueError("VehicleAttention requires discrete action spaces")


def resolve_mappo_agent_class(representation: str) -> type[MAPPO_Agents]:
    if representation == VEHICLE_ATTENTION_REPRESENTATION:
        return AttentionMAPPOAgents
    return MAPPO_Agents
