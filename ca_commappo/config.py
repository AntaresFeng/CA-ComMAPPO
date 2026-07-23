import argparse
from pathlib import Path
from typing import Any

from xuance.common import load_yaml, recursive_dict_update


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
COMMON_CONFIG_PATH = CONFIG_DIR / "common.yaml"

SMOKE_OVERRIDES = {
    "logger": "tensorboard",
    "vectorize": "DummyVecMultiAgentEnv",
    "representation_hidden_size": [64, 64],
    "actor_hidden_size": [64, 64],
    "critic_hidden_size": [64, 64],
    "parallels": 1,
    "buffer_size": 8,
    "n_epochs": 1,
    "n_minibatch": 1,
    "running_steps": 16,
    "eval_interval": 16,
    "test_episode": 1,
}


def config_path_for(env_id: str, agent: str) -> Path:
    directory = agent.lower()
    path = CONFIG_DIR / directory / f"{env_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown highway {agent} config: {path}")
    return path


def load_configs(
    env_id: str,
    *,
    config_path: str = "",
    agent: str = "mappo",
    smoke: bool = False,
    overrides: dict[str, Any] | None = None,
) -> argparse.Namespace:
    """Compose common, environment, algorithm, smoke, and CLI config layers."""
    algorithm_path = (
        Path(config_path) if config_path else config_path_for(env_id, agent)
    )
    if not algorithm_path.exists():
        raise FileNotFoundError(f"Unknown highway {agent} config: {algorithm_path}")

    common = load_yaml(file_dir=str(COMMON_CONFIG_PATH))
    algorithm = load_yaml(file_dir=str(algorithm_path))
    resolved_env_id = algorithm.get("env_id", env_id)
    environment_path = CONFIG_DIR / "highway" / f"{resolved_env_id}.yaml"
    if not environment_path.exists():
        raise FileNotFoundError(
            f"Unknown highway environment config: {environment_path}"
        )

    config = recursive_dict_update(
        common, {"highway_config": load_yaml(file_dir=str(environment_path))}
    )
    config = recursive_dict_update(config, algorithm)
    if smoke:
        config = recursive_dict_update(config, SMOKE_OVERRIDES)
    if overrides:
        config = recursive_dict_update(config, overrides)

    config["config_path"] = str(algorithm_path)
    config["highway_config_path"] = str(environment_path)
    config["smoke"] = smoke
    return argparse.Namespace(**config)
