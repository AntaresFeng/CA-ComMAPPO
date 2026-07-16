import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import wandb
from xuance.common import load_yaml, recursive_dict_update
from xuance.environment import make_envs
from xuance.torch.agents import MAPPO_Agents
from xuance.torch.utils.operations import set_seed

from ca_commappo.envs.highway_intersection_wrapper import DEFAULT_HIGHWAY_ENV_ID
from ca_commappo.evaluation.highway_metrics import print_highway_summary
from ca_commappo.evaluation.mappo_highway_metrics import (
    HighwayMetricsCallback,
    evaluate_highway_policy,
    is_better_highway_summary,
)
from ca_commappo.evaluation.mappo_eval_artifacts import (
    append_eval_record_jsonl,
    build_eval_metadata,
    build_eval_record,
    write_eval_metadata,
)
from ca_commappo.evaluation.mappo_video_recorder import (
    record_mappo_policy_videos,
    resolve_video_episode_count,
)
from ca_commappo.training.mappo_run_layout import (
    align_wandb_run_name,
    prepare_run_directory,
)
from ca_commappo.xuance_compat import patch_xuance_marl_buffer_aliases


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "mappo"
EVAL_ENV_SEED_OFFSET = 10_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run XuanCe MAPPO on highway-env multi-agent intersection."
    )
    parser.add_argument("--env-id", type=str, default=DEFAULT_HIGHWAY_ENV_ID)
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
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Root directory for isolated run folders, logs, models, eval JSON, and videos.",
    )
    parser.add_argument("--model-dir-load", type=str, default=None)
    parser.add_argument(
        "--record-video",
        action="store_true",
        default=None,
        help="Record representative MAPPO test episodes as MP4 videos.",
    )
    parser.add_argument(
        "--video-episodes",
        type=str,
        default=None,
        help="Number of episodes to record, or 'all'. Defaults to min(test_episode, 6).",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Directory for recorded MP4s and video summary artifacts.",
    )
    parser.add_argument(
        "--video-seed",
        type=int,
        default=None,
        help="Base seed for video rollout episodes. Defaults to config seed.",
    )
    parser.add_argument(
        "--no-video-contact-sheet",
        dest="video_contact_sheet",
        action="store_false",
        default=None,
        help="Skip the default contact-sheet JPEG for recorded videos.",
    )
    parser.add_argument(
        "--no-combined-video",
        dest="video_combined",
        action="store_false",
        default=None,
        help="Skip the default combined MP4 for recorded videos.",
    )
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
    env_id: str = DEFAULT_HIGHWAY_ENV_ID,
    overrides: dict[str, Any] | None = None,
    config_path: str = "",
) -> argparse.Namespace:
    path = Path(config_path) if config_path else config_path_for(env_id)
    if config_path and not path.exists():
        raise FileNotFoundError(f"Unknown highway MAPPO config: {path}")
    config_dict = load_yaml(file_dir=str(path))
    if overrides:
        config_dict = recursive_dict_update(config_dict, overrides)
    config_dict["config_path"] = str(path)
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
        "output_dir": args.output_dir,
        "model_dir_load": args.model_dir_load,
        "test_mode": args.mode == "test",
        "record_video": args.record_video,
        "video_episodes": args.video_episodes,
        "video_dir": args.video_dir,
        "video_seed": args.video_seed,
        "video_contact_sheet": args.video_contact_sheet,
        "video_combined": args.video_combined,
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
        "Run directory": configs.run_dir,
        "Model directory": configs.model_dir,
    }
    for key, value in train_information.items():
        print(f"{key}: {value}")


def write_eval_run_metadata(
    configs: argparse.Namespace,
    agents: MAPPO_Agents,
    mode: str,
) -> None:
    metadata = build_eval_metadata(configs=configs, agents=agents, mode=mode)
    metadata_path = write_eval_metadata(metadata, agents.log_dir)
    print(f"Eval metadata saved: {metadata_path}")


def append_eval_run_record(
    *,
    agents: MAPPO_Agents,
    mode: str,
    phase: str,
    epoch: int | None,
    is_initial_eval: bool,
    is_best: bool,
    eval_result: dict[str, Any],
) -> Path:
    record = build_eval_record(
        mode=mode,
        phase=phase,
        epoch=epoch,
        step=agents.current_step,
        is_initial_eval=is_initial_eval,
        is_best=is_best,
        eval_result=eval_result,
    )
    return append_eval_record_jsonl(record, agents.log_dir)


def train(configs: argparse.Namespace, agents: MAPPO_Agents, save_model: bool) -> None:
    train_steps = max(1, configs.running_steps // configs.parallels)
    agents.train(train_steps)
    if save_model:
        agents.save_model("final_train_model.pth", model_path=configs.model_dir)
    print("Finish training.")


def test(configs: argparse.Namespace, agents: MAPPO_Agents, envs) -> None:
    model_path = getattr(configs, "model_dir_load", configs.model_dir)
    agents.load_model(path=model_path)
    write_eval_run_metadata(configs, agents, mode="test")
    result = evaluate_highway_policy(
        agents=agents,
        envs=envs,
        test_episodes=configs.test_episode,
        phase="test",
        log_prefix="Test-Highway",
    )
    records_path = append_eval_run_record(
        agents=agents,
        mode="test",
        phase="test",
        epoch=None,
        is_initial_eval=False,
        is_best=True,
        eval_result=result,
    )
    scores = result["scores"]
    print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
    print_highway_summary(result["summary"])
    print(f"Eval records saved: {records_path}")
    if getattr(configs, "record_video", False):
        video_episode_count = resolve_video_episode_count(
            test_episode=configs.test_episode,
            requested=getattr(configs, "video_episodes", None),
        )
        video_dir = (
            getattr(configs, "video_dir", None) or Path(agents.log_dir) / "videos"
        )
        video_seed = getattr(configs, "video_seed", None)
        video_result = record_mappo_policy_videos(
            configs=configs,
            agents=agents,
            model_path=str(model_path),
            video_dir=video_dir,
            episode_count=video_episode_count,
            base_seed=configs.seed if video_seed is None else video_seed,
            make_contact_sheet=getattr(configs, "video_contact_sheet", True),
            make_combined_video=getattr(configs, "video_combined", True),
        )
        print(f"Video eval summary saved: {video_result['summary_path']}")
        print(f"Video files saved: {video_result['video_dir']}")
        if video_result.get("combined_video_path"):
            print(f"Combined video saved: {video_result['combined_video_path']}")
        if video_result.get("contact_sheet_path"):
            print(f"Contact sheet saved: {video_result['contact_sheet_path']}")
    print("Finish testing.")


def benchmark(
    configs: argparse.Namespace,
    agents: MAPPO_Agents,
    save_model: bool,
) -> None:
    configs_test = deepcopy(configs)
    configs_test.parallels = configs_test.test_episode
    configs_test.env_seed = configs_test.seed + EVAL_ENV_SEED_OFFSET
    test_envs = make_envs(configs_test)
    try:
        write_eval_run_metadata(configs, agents, mode="benchmark")
        train_steps = max(1, configs.running_steps // configs.parallels)
        eval_interval = max(1, configs.eval_interval // configs.parallels)
        num_epoch = max(1, int(train_steps / eval_interval))

        eval_result = evaluate_highway_policy(
            agents=agents,
            envs=test_envs,
            test_episodes=configs.test_episode,
            phase="benchmark",
            log_prefix="Eval-Highway",
        )
        test_scores = eval_result["scores"]
        best_scores_info = {
            "mean": np.mean(test_scores),
            "std": np.std(test_scores),
            "step": agents.current_step,
            "summary": eval_result["summary"],
        }
        print_highway_summary(eval_result["summary"])
        records_path = append_eval_run_record(
            agents=agents,
            mode="benchmark",
            phase="benchmark",
            epoch=0,
            is_initial_eval=True,
            is_best=True,
            eval_result=eval_result,
        )
        print(f"Eval records saved: {records_path}")
        if save_model:
            agents.save_model(model_name="best_model.pth", model_path=configs.model_dir)

        for epoch in range(num_epoch):
            print(f"Epoch: {epoch + 1}/{num_epoch}:")
            agents.train(eval_interval)
            eval_result = evaluate_highway_policy(
                agents=agents,
                envs=test_envs,
                test_episodes=configs.test_episode,
                phase="benchmark",
                log_prefix="Eval-Highway",
            )
            test_scores = eval_result["scores"]
            mean_score = np.mean(test_scores)
            print_highway_summary(eval_result["summary"])
            is_best = is_better_highway_summary(
                eval_result["summary"], best_scores_info["summary"]
            )
            records_path = append_eval_run_record(
                agents=agents,
                mode="benchmark",
                phase="benchmark",
                epoch=epoch + 1,
                is_initial_eval=False,
                is_best=is_best,
                eval_result=eval_result,
            )
            print(f"Eval records saved: {records_path}")
            if is_best:
                best_scores_info = {
                    "mean": mean_score,
                    "std": np.std(test_scores),
                    "step": agents.current_step,
                    "summary": eval_result["summary"],
                }
                if save_model:
                    agents.save_model(
                        model_name="best_model.pth", model_path=configs.model_dir
                    )

        print(
            "Best Model Score: %.2f, std=%.2f"
            % (best_scores_info["mean"], best_scores_info["std"])
        )
        print("Best Highway Metrics:")
        print_highway_summary(best_scores_info["summary"])
    finally:
        test_envs.close()


def run(configs: argparse.Namespace, mode: str, save_model: bool = True) -> None:
    prepare_run_directory(configs, mode)
    patch_xuance_marl_buffer_aliases()
    set_seed(configs.seed)
    envs = make_envs(configs)
    agents = None
    try:
        metrics_callback = HighwayMetricsCallback()
        agents = MAPPO_Agents(config=configs, envs=envs, callback=metrics_callback)
        align_wandb_run_name(configs, wandb.run)
        metrics_callback.set_logger(agents.log_infos)
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
    if args.record_video and args.mode != "test":
        parser.error("--record-video is only supported with --mode test")
    configs = load_configs(args.env_id, cli_overrides(args), config_path=args.config)
    run(configs, mode=args.mode, save_model=not args.no_save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
