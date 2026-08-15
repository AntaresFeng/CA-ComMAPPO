import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import wandb
from xuance.environment import make_envs
from xuance.torch.agents import CommNet_Agents, IC3Net_Agents, IPPO_Agents, MAPPO_Agents
from xuance.torch.utils.operations import set_seed

from ca_commappo.algorithms import register_components
from ca_commappo.algorithms.ca_commappo_agents import CA_ComMAPPO_Agents
from ca_commappo.algorithms.comm_runtime_patch import patch_comm_agents_info
from ca_commappo.algorithms.mappo_diagnostics import patch_mappo_diagnostics
from ca_commappo.config import load_configs as load_project_configs
from ca_commappo.envs.highway_intersection_wrapper import (
    DEFAULT_HIGHWAY_ENV_ID,
    register_highway_intersection_env,
)
from ca_commappo.evaluation.highway_metrics import (
    print_highway_summary,
    summarize_episode_records,
    summary_to_log_infos,
)
from ca_commappo.evaluation.mappo_eval_artifacts import (
    append_eval_record_jsonl,
    build_eval_metadata,
    build_eval_record,
    write_eval_metadata,
)
from ca_commappo.evaluation.mappo_highway_metrics import (
    HighwayMetricsCallback,
    evaluate_highway_policy,
    is_better_highway_summary,
)
from ca_commappo.evaluation.mappo_video_recorder import (
    record_mappo_policy_videos,
    resolve_video_episode_count,
)
from ca_commappo.evaluation.model_provenance import (
    BEST_MODEL_FILENAME,
    load_model_provenance,
    write_best_model_provenance,
)
from ca_commappo.training.mappo_run_layout import (
    align_wandb_run_name,
    prepare_run_directory,
)
from ca_commappo.xuance_compat import (
    patch_xuance_mappo_masked_statistics,
    patch_xuance_marl_buffer_aliases,
)

# Ensure environment is registered in main process and subprocesses.
register_highway_intersection_env()


AGENT_REGISTRY = {
    "IPPO": IPPO_Agents,
    "MAPPO": MAPPO_Agents,
    "COMMNET": CommNet_Agents,
    "IC3NET": IC3Net_Agents,
    "CA_COMMAPPO": CA_ComMAPPO_Agents,
}


def get_agent_class(agent_name: str):
    if agent_name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent: {agent_name}")
    return AGENT_REGISTRY[agent_name]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run XuanCe IPPO/MAPPO on highway-env multi-agent intersection."
    )
    parser.add_argument("--env-id", type=str, default=DEFAULT_HIGHWAY_ENV_ID)
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional YAML config path. Empty value uses the built-in --env-id config.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Apply the shared fast smoke-test overrides to the selected config.",
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
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=None)
    parser.add_argument("--checkpoint-eval-seed", type=int, default=None)
    parser.add_argument("--final-eval-seed", type=int, default=None)
    parser.add_argument("--eval-parallels", type=int, default=None)
    parser.add_argument("--test-episode", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--n-minibatch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--logger",
        choices=("wandb", "tensorboard"),
        default=None,
        help="Override the configured local experiment logger.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Root directory for isolated run folders, logs, models, eval JSON, and videos.",
    )
    parser.add_argument("--model-dir-load", type=str, default=None)
    parser.add_argument(
        "--allow-unverified-model",
        action="store_true",
        default=None,
        help=(
            "Allow diagnostic evaluation of a legacy best_model.pth without its "
            "provenance sidecar; such a run is ineligible for final aggregation."
        ),
    )
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
        help="Base seed for video episodes. Defaults to final_eval_seed.",
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


def load_configs(
    env_id: str = DEFAULT_HIGHWAY_ENV_ID,
    overrides: dict[str, Any] | None = None,
    config_path: str = "",
    agent: str = "MAPPO",
    smoke: bool = False,
) -> argparse.Namespace:
    return load_project_configs(
        env_id,
        config_path=config_path,
        agent=agent,
        smoke=smoke,
        overrides=overrides,
    )


def cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    candidates = {
        "parallels": args.parallels,
        "buffer_size": args.buffer_size,
        "running_steps": args.running_steps,
        "eval_interval": args.eval_interval,
        "checkpoint_eval_episodes": args.checkpoint_eval_episodes,
        "checkpoint_eval_seed": args.checkpoint_eval_seed,
        "final_eval_seed": args.final_eval_seed,
        "eval_parallels": args.eval_parallels,
        "test_episode": args.test_episode,
        "n_epochs": args.n_epochs,
        "n_minibatch": args.n_minibatch,
        "seed": args.seed,
        "device": args.device,
        "logger": args.logger,
        "output_dir": args.output_dir,
        "model_dir_load": args.model_dir_load,
        "allow_unverified_model": args.allow_unverified_model,
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
    agents,
    mode: str,
    evaluation_protocol: dict[str, Any] | None = None,
    model_provenance: dict[str, Any] | None = None,
) -> None:
    metadata = build_eval_metadata(
        configs=configs,
        agents=agents,
        mode=mode,
        evaluation_protocol=evaluation_protocol,
        model_provenance=model_provenance,
    )
    metadata_path = write_eval_metadata(metadata, agents.log_dir)
    print(f"Eval metadata saved: {metadata_path}")


def append_eval_run_record(
    *,
    agents,
    mode: str,
    phase: str,
    epoch: int | None,
    is_initial_eval: bool,
    is_best: bool,
    eval_result: dict[str, Any],
    evaluation_protocol: dict[str, Any] | None = None,
    model_provenance: dict[str, Any] | None = None,
) -> Path:
    record = build_eval_record(
        mode=mode,
        phase=phase,
        epoch=epoch,
        step=agents.current_step,
        is_initial_eval=is_initial_eval,
        is_best=is_best,
        eval_result=eval_result,
        evaluation_protocol=evaluation_protocol,
        model_provenance=model_provenance,
    )
    return append_eval_record_jsonl(record, agents.log_dir)


def train(configs: argparse.Namespace, agents, save_model: bool) -> None:
    train_steps = max(1, configs.running_steps // configs.parallels)
    agents.train(train_steps)
    if save_model:
        agents.save_model("final_train_model.pth", model_path=configs.model_dir)
    print("Finish training.")


def fixed_seed_evaluation_protocol(
    *,
    base_seed: int,
    episode_count: int,
    parallels: int,
    purpose: str,
) -> dict[str, Any]:
    if episode_count <= 0:
        raise ValueError("evaluation episode_count must be positive")
    if parallels <= 0:
        raise ValueError("evaluation parallels must be positive")
    return {
        "type": "fixed_explicit_episode_seeds",
        "purpose": purpose,
        "deterministic_actions": True,
        "reproducible_actions": True,
        "action_selection_mode": "deterministic_argmax",
        "base_seed": int(base_seed),
        "episode_count": int(episode_count),
        "parallels": min(int(parallels), int(episode_count)),
        "episode_seeds": list(
            range(int(base_seed), int(base_seed) + int(episode_count))
        ),
    }


def evaluate_fixed_seed_policy(
    *,
    configs: argparse.Namespace,
    agents,
    protocol: dict[str, Any],
    phase: str,
    log_prefix: str,
) -> dict[str, Any]:
    """Evaluate exactly one episode for every explicit seed in ``protocol``."""

    episode_seeds = [int(seed) for seed in protocol["episode_seeds"]]
    eval_parallels = int(protocol["parallels"])
    records: list[dict[str, Any]] = []

    for batch_start in range(0, len(episode_seeds), eval_parallels):
        batch_seeds = episode_seeds[batch_start : batch_start + eval_parallels]
        configs_eval = deepcopy(configs)
        configs_eval.env_seed = batch_seeds[0]
        configs_eval.parallels = len(batch_seeds)
        eval_envs = make_envs(configs_eval)
        try:
            batch_result = evaluate_highway_policy(
                agents=agents,
                envs=eval_envs,
                test_episodes=len(batch_seeds),
                phase=phase,
                log_prefix=None,
                deterministic=True,
                record_once_per_env=True,
                episode_seeds=batch_seeds,
            )
        finally:
            eval_envs.close()
        records.extend(batch_result["episodes"])

    records.sort(key=lambda record: int(record["seed"]))
    for episode_index, record in enumerate(records):
        record["episode_index"] = episode_index
    observed_seeds = [int(record["seed"]) for record in records]
    if observed_seeds != episode_seeds:
        raise RuntimeError(
            "fixed-seed evaluation did not produce exactly one episode per seed"
        )

    summary = summarize_episode_records(records)
    agents.log_infos(summary_to_log_infos(log_prefix, summary), agents.current_step)
    return {
        "scores": [float(record["score"]) for record in records],
        "episodes": records,
        "summary": summary,
    }


def benchmark_train_segments(configs: argparse.Namespace) -> list[int]:
    """Return exact vector-step segments without silently changing the budget."""

    running_steps = int(configs.running_steps)
    parallels = int(configs.parallels)
    eval_interval = int(configs.eval_interval)
    if running_steps <= 0 or parallels <= 0 or eval_interval <= 0:
        raise ValueError("running_steps, parallels, and eval_interval must be positive")
    if running_steps % parallels != 0:
        raise ValueError("running_steps must be divisible by parallels")
    if eval_interval % parallels != 0:
        raise ValueError("eval_interval must be divisible by parallels")

    remaining = running_steps // parallels
    interval = eval_interval // parallels
    segments: list[int] = []
    while remaining > 0:
        segment = min(interval, remaining)
        segments.append(segment)
        remaining -= segment
    return segments


def test(configs: argparse.Namespace, agents, envs) -> None:
    model_path = getattr(configs, "model_dir_load", configs.model_dir)
    model_provenance = load_model_provenance(
        model_path=model_path,
        expected_training_seed=configs.seed,
        expected_configs=configs,
        allow_unverified=bool(getattr(configs, "allow_unverified_model", False)),
    )
    agents.load_model(path=model_path)
    protocol = fixed_seed_evaluation_protocol(
        base_seed=configs.final_eval_seed,
        episode_count=configs.test_episode,
        parallels=configs.eval_parallels,
        purpose="final_test",
    )
    write_eval_run_metadata(
        configs,
        agents,
        mode="test",
        evaluation_protocol=protocol,
        model_provenance=model_provenance,
    )
    result = evaluate_fixed_seed_policy(
        configs=configs,
        agents=agents,
        protocol=protocol,
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
        evaluation_protocol=protocol,
        model_provenance=model_provenance,
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
            base_seed=(configs.final_eval_seed if video_seed is None else video_seed),
            make_contact_sheet=getattr(configs, "video_contact_sheet", True),
            make_combined_video=getattr(configs, "video_combined", True),
            model_provenance=model_provenance,
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
    agents,
    save_model: bool,
) -> None:
    protocol = fixed_seed_evaluation_protocol(
        base_seed=configs.checkpoint_eval_seed,
        episode_count=configs.checkpoint_eval_episodes,
        parallels=configs.eval_parallels,
        purpose="checkpoint_selection",
    )
    write_eval_run_metadata(
        configs, agents, mode="benchmark", evaluation_protocol=protocol
    )
    train_segments = benchmark_train_segments(configs)
    num_epoch = len(train_segments)

    eval_result = evaluate_fixed_seed_policy(
        configs=configs,
        agents=agents,
        protocol=protocol,
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
        evaluation_protocol=protocol,
    )
    print(f"Eval records saved: {records_path}")
    if save_model:
        _save_best_model_with_provenance(
            configs=configs,
            agents=agents,
            selected_epoch=0,
            selection_summary=eval_result["summary"],
            selection_protocol=protocol,
        )

    for epoch, train_segment in enumerate(train_segments, start=1):
        print(f"Epoch: {epoch}/{num_epoch}:")
        agents.train(train_segment)
        eval_result = evaluate_fixed_seed_policy(
            configs=configs,
            agents=agents,
            protocol=protocol,
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
            epoch=epoch,
            is_initial_eval=False,
            is_best=is_best,
            eval_result=eval_result,
            evaluation_protocol=protocol,
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
                _save_best_model_with_provenance(
                    configs=configs,
                    agents=agents,
                    selected_epoch=epoch,
                    selection_summary=eval_result["summary"],
                    selection_protocol=protocol,
                )

    if save_model:
        agents.save_model(model_name="final_model.pth", model_path=configs.model_dir)
    print(
        f"Best Model Score: {best_scores_info['mean']:.2f}, "
        f"std={best_scores_info['std']:.2f}"
    )
    print("Best Highway Metrics:")
    print_highway_summary(best_scores_info["summary"])


def _save_best_model_with_provenance(
    *,
    configs: argparse.Namespace,
    agents,
    selected_epoch: int,
    selection_summary: dict[str, Any],
    selection_protocol: dict[str, Any],
) -> dict[str, Any]:
    agents.save_model(model_name=BEST_MODEL_FILENAME, model_path=configs.model_dir)
    return write_best_model_provenance(
        model_path=Path(configs.model_dir) / BEST_MODEL_FILENAME,
        configs=configs,
        selected_epoch=selected_epoch,
        selected_step=agents.current_step,
        selection_summary=selection_summary,
        selection_protocol=selection_protocol,
    )


def run(configs: argparse.Namespace, mode: str, save_model: bool = True) -> None:
    prepare_run_directory(configs, mode)
    patch_xuance_marl_buffer_aliases()
    patch_xuance_mappo_masked_statistics()
    patch_mappo_diagnostics()
    register_components()
    patch_comm_agents_info()
    set_seed(configs.seed)
    envs = make_envs(configs)
    agents = None
    try:
        metrics_callback = HighwayMetricsCallback()
        agent_class = get_agent_class(configs.agent)
        agents = agent_class(config=configs, envs=envs, callback=metrics_callback)
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
    configs = load_configs(
        args.env_id,
        cli_overrides(args),
        config_path=args.config,
        smoke=args.smoke,
    )
    run(configs, mode=args.mode, save_model=not args.no_save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
