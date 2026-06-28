"""Environment adapters for CA-ComMAPPO."""

from ca_commappo.envs.highway_intersection import (
    HighwayIntersectionMultiAgentEnv,
    build_intersection_config,
    register_highway_intersection_env,
)

__all__ = [
    "HighwayIntersectionMultiAgentEnv",
    "build_intersection_config",
    "register_highway_intersection_env",
]
