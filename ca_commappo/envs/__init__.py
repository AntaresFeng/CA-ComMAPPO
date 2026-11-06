"""Environment adapters for CA-ComMAPPO."""

from ca_commappo.envs.highway_intersection_wrapper import (
    DEFAULT_HIGHWAY_ENV_ID,
    HighwayIntersectionMultiAgentEnv,
    build_intersection_config,
    register_highway_intersection_env,
)
from ca_commappo.envs.intersection import ResearchIntersectionEnv

__all__ = [
    "DEFAULT_HIGHWAY_ENV_ID",
    "HighwayIntersectionMultiAgentEnv",
    "ResearchIntersectionEnv",
    "build_intersection_config",
    "register_highway_intersection_env",
]
