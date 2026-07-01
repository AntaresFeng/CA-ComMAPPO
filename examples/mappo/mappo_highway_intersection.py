import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from xuance.common import load_yaml, recursive_dict_update
from xuance.environment import make_envs
from xuance.torch.agents import MAPPO_Agents
from xuance.torch.utils.operations import set_seed

from ca_commappo.envs.highway_intersection import register_highway_intersection_env


CONFIG_DIR = Path(__file__).resolve().parent / "mappo_highway_configs"
DEFAULT_ENV_ID = "intersection_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run XuanCe MAPPO on highway-env intersection-v1."
    )
    parser.add_argument("--env-id", type=str, default=DEFAULT_ENV_ID)
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional YAML config path. Empty value uses the built-in --env-id config.",
    )
    parser.add_argument(
        "--mode",
        choices=("train", "test", "benchmark"),
        default="train",
        help="train runs one training job, test loads a model, benchmark alternates train/test.",
    )
    parser.add_argument("--parallels", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--running-steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--test-episode", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--n-minibatch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--model-dir-load", type=str, default=None)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving final/best models; useful for very small smoke runs.",
    )
    return parser


def config_path_for(env_id: str) -> Path:
    path = CONFIG_DIR / f"{env_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown highway MAPPO config: {path}")
    return path


def load_configs(
    env_id: str = DEFAULT_ENV_ID,
    overrides: dict[str, Any] | None = None,
    config_path: str = "",
) -> argparse.Namespace:
    path = Path(config_path) if config_path else config_path_for(env_id)
    if config_path and not path.exists():
        raise FileNotFoundError(f"Unknown highway MAPPO config: {path}")
    config_dict = load_yaml(file_dir=str(path))
    if overrides:
        config_dict = recursive_dict_update(config_dict, overrides)
    return argparse.Namespace(**config_dict)


def cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    candidates = {
        "parallels": args.parallels,
        "buffer_size": args.buffer_size,
        "running_steps": args.running_steps,
        "eval_interval": args.eval_interval,
        "test_episode": args.test_episode,
        "n_epochs": args.n_epochs,
        "n_minibatch": args.n_minibatch,
        "seed": args.seed,
        "device": args.device,
        "log_dir": args.log_dir,
        "model_dir": args.model_dir,
        "model_dir_load": args.model_dir_load,
        "test_mode": args.mode == "test",
    }
    return {key: value for key, value in candidates.items() if value is not None}


def print_train_information(configs: argparse.Namespace) -> None:
    train_information = {
        "Deep learning toolbox": configs.dl_toolbox,
        "Calculating device": configs.device,
        "Algorithm": configs.agent,
        "Environment": configs.env_name,
        "Scenario": configs.env_id,
        "Vectorizer": configs.vectorize,
        "Parallels": configs.parallels,
        "Running steps": configs.running_steps,
    }
    for key, value in train_information.items():
        print(f"{key}: {value}")


def patch_xuance_marl_buffer_aliases() -> None:
    """Patch XuanCe 1.4.3 MARL buffer field-name drift in this process.

    `BaseBuffer` stores the newer names `num_envs`, `per_env_buffer_size`,
    `observation_space`, and `action_space`, but `MARL_OnPolicyBuffer` still
    reads the older names `n_envs`, `n_size`, `obs_space`, and `act_space`.
    Adding aliases after `BaseBuffer.__init__` lets MAPPO build its on-policy
    buffer without editing installed files under `.venv`.
    """
    from xuance.common.memory_tools_marl import BaseBuffer

    if getattr(BaseBuffer, "_ca_commappo_alias_patch", False):
        return

    original_init = BaseBuffer.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Keep the shim narrow: only expose the legacy names used downstream.
        self.n_envs = self.num_envs
        self.n_size = self.per_env_buffer_size
        self.obs_space = self.observation_space
        self.act_space = self.action_space

    BaseBuffer.__init__ = patched_init
    BaseBuffer._ca_commappo_alias_patch = True


def train(configs: argparse.Namespace, agents: MAPPO_Agents, save_model: bool) -> None:
    train_steps = max(1, configs.running_steps // configs.parallels)
    agents.train(train_steps)
    if save_model:
        agents.save_model("final_train_model.pth")
    print("Finish training.")


def test(configs: argparse.Namespace, agents: MAPPO_Agents, envs) -> None:
    model_path = getattr(configs, "model_dir_load", configs.model_dir)
    agents.load_model(path=model_path)
    scores = agents.test(
        test_episodes=configs.test_episode,
        test_envs=envs,
        close_envs=False,
    )
    print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
    print("Finish testing.")


def benchmark(
    configs: argparse.Namespace,
    agents: MAPPO_Agents,
    save_model: bool,
) -> None:
    configs_test = deepcopy(configs)
    configs_test.parallels = configs_test.test_episode
    test_envs = make_envs(configs_test)
    try:
        train_steps = max(1, configs.running_steps // configs.parallels)
        eval_interval = max(1, configs.eval_interval // configs.parallels)
        num_epoch = max(1, int(train_steps / eval_interval))

        test_scores = agents.test(
            test_episodes=configs.test_episode,
            test_envs=test_envs,
            close_envs=False,
        )
        best_scores_info = {
            "mean": np.mean(test_scores),
            "std": np.std(test_scores),
            "step": agents.current_step,
        }
        if save_model:
            agents.save_model(model_name="best_model.pth")

        for epoch in range(num_epoch):
            print(f"Epoch: {epoch + 1}/{num_epoch}:")
            agents.train(eval_interval)
            test_scores = agents.test(
                test_episodes=configs.test_episode,
                test_envs=test_envs,
                close_envs=False,
            )
            mean_score = np.mean(test_scores)
            if mean_score > best_scores_info["mean"]:
                best_scores_info = {
                    "mean": mean_score,
                    "std": np.std(test_scores),
                    "step": agents.current_step,
                }
                if save_model:
                    agents.save_model(model_name="best_model.pth")

        print(
            "Best Model Score: %.2f, std=%.2f"
            % (best_scores_info["mean"], best_scores_info["std"])
        )
    finally:
        test_envs.close()


def run(configs: argparse.Namespace, mode: str, save_model: bool = True) -> None:
    register_highway_intersection_env()
    patch_xuance_marl_buffer_aliases()
    set_seed(configs.seed)
    envs = make_envs(configs)
    agents = None
    try:
        agents = MAPPO_Agents(config=configs, envs=envs)
        print_train_information(configs)
        if mode == "benchmark":
            benchmark(configs, agents, save_model=save_model)
        elif mode == "test":
            test(configs, agents, envs)
        else:
            train(configs, agents, save_model=save_model)
    finally:
        if agents is not None:
            agents.finish()
        envs.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configs = load_configs(args.env_id, cli_overrides(args), config_path=args.config)
    run(configs, mode=args.mode, save_model=not args.no_save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
