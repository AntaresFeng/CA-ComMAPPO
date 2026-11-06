"""Read-only CA-ComMAPPO communication traces and attention heatmaps."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from ca_commappo.envs.highway_intersection_wrapper import (
    IDLE_ACTION,
    HighwayIntersectionMultiAgentEnv,
)
from ca_commappo.evaluation.highway_metrics import (
    build_episode_record,
    summarize_episode_records,
)
from ca_commappo.evaluation.mappo_eval_artifacts import json_safe
from ca_commappo.evaluation.mappo_video_recorder import (
    _agent_action_for_single_env,
    _init_single_env_rnn_hidden,
)

CA_TRACE_SCHEMA_VERSION = 1
CA_TRACE_EPISODE_COUNT = 10
CA_TRACE_METADATA_FILENAME = "ca_trace_metadata.json"
CA_TRACE_RECORDS_FILENAME = "ca_trace_records.jsonl"
CA_TRACE_SUMMARY_FILENAME = "ca_trace_summary.json"


def ca_trace_protocol(*, base_seed: int, episode_count: int) -> dict[str, Any]:
    if base_seed < 0 or episode_count <= 0:
        raise ValueError(
            "trace base_seed must be non-negative and episode_count positive"
        )
    return {
        "schema_version": CA_TRACE_SCHEMA_VERSION,
        "type": "ca_commappo_read_only_communication_trace",
        "purpose": "attention_ttc_interpretability",
        "deterministic_actions": True,
        "action_selection_mode": "deterministic_argmax",
        "base_seed": int(base_seed),
        "episode_count": int(episode_count),
        "episode_seeds": list(range(base_seed, base_seed + episode_count)),
        "trajectory_source": "normalized_actor_local_ego_rows",
    }


def collect_ca_trace_episodes(
    *,
    configs: Any,
    agents: Any,
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    if not hasattr(agents, "communication_trace"):
        raise TypeError("CA trace evaluation requires CA_ComMAPPO_Agents")
    episode_seeds = [int(seed) for seed in protocol["episode_seeds"]]
    feature_names = [
        str(name)
        for name in configs.highway_config["observation"]["observation_config"][
            "features"
        ]
    ]
    vehicles_count = int(
        configs.highway_config["observation"]["observation_config"]["vehicles_count"]
    )
    env_configs = deepcopy(configs)
    env_configs.render_mode = None
    env = HighwayIntersectionMultiAgentEnv(env_configs)
    episodes = []
    try:
        for episode_index, episode_seed in enumerate(episode_seeds):
            episodes.append(
                _collect_single_trace_episode(
                    env=env,
                    agents=agents,
                    episode_index=episode_index,
                    episode_seed=episode_seed,
                    feature_names=feature_names,
                    vehicles_count=vehicles_count,
                )
            )
    finally:
        env.close()
    return episodes


def _collect_single_trace_episode(
    *,
    env: HighwayIntersectionMultiAgentEnv,
    agents: Any,
    episode_index: int,
    episode_seed: int,
    feature_names: list[str],
    vehicles_count: int,
) -> dict[str, Any]:
    rnn_hidden_actor, rnn_hidden_critic = _init_single_env_rnn_hidden(agents)
    obs, _info = env.reset(seed=episode_seed)
    current_info = {"agent_mask": env.agent_mask()}
    agent_rewards = {agent: 0.0 for agent in env.agents}
    latest_info: dict[str, Any] = {}
    truncated = False
    trace_steps = []

    for step_index in range(int(env.max_episode_steps)):
        captured = []
        with agents.communication_trace(captured.append):
            policy_out = _agent_action_for_single_env(
                agents,
                env,
                obs,
                rnn_hidden_actor,
                rnn_hidden_critic,
                current_info,
            )
        if len(captured) != 1:
            raise RuntimeError(
                "CA trace requires exactly one actor communication payload per action; "
                f"observed {len(captured)}"
            )
        trace = _serialize_and_validate_trace(captured[0], env.agents)
        policy_actions = {
            agent: int(policy_out["actions"][0][agent]) for agent in env.agents
        }
        environment_actions = {
            agent: policy_actions[agent] if trace["active_mask"][agent] else IDLE_ACTION
            for agent in env.agents
        }
        ego_observations = _ego_observations(
            obs,
            agents=env.agents,
            vehicles_count=vehicles_count,
            feature_names=feature_names,
        )
        rnn_hidden_actor = policy_out.get("rnn_hidden_actor")
        rnn_hidden_critic = policy_out.get("rnn_hidden_critic")
        next_obs, rewards, terminated, truncated, latest_info = env.step(
            policy_out["actions"][0]
        )
        for agent, reward in rewards.items():
            agent_rewards[agent] += float(reward)
        trace_steps.append(
            {
                "step_index": step_index,
                **trace,
                "ego_observations": ego_observations,
                "policy_actions": policy_actions,
                "environment_actions": environment_actions,
                "rewards": {agent: float(rewards[agent]) for agent in env.agents},
                "terminated": {agent: bool(terminated[agent]) for agent in env.agents},
                "truncated": bool(truncated),
                "crashed": [bool(value) for value in latest_info["crashed"]],
                "arrived": [bool(value) for value in latest_info["arrived"]],
                "next_agent_mask": {
                    agent: bool(latest_info["next_agent_mask"][agent])
                    for agent in env.agents
                },
            }
        )
        obs = next_obs
        current_info = latest_info
        if all(bool(value) for value in terminated.values()) or bool(truncated):
            break

    outcome = build_episode_record(
        episode_index=episode_index,
        steps=len(trace_steps),
        agent_rewards=agent_rewards,
        crashed_agents=latest_info["crashed"],
        arrived_agents=latest_info["arrived"],
        truncated=bool(truncated),
        phase="ca_trace",
        seed=episode_seed,
        score=sum(agent_rewards.values()) / len(agent_rewards),
        waiting_steps=latest_info.get("waiting_steps"),
        dt=latest_info.get("dt"),
        arrival_times=latest_info.get("arrival_times"),
    )
    return {
        "schema_version": CA_TRACE_SCHEMA_VERSION,
        "episode_seed": episode_seed,
        "agent_order": list(env.agents),
        "trace_steps": trace_steps,
        "outcome": outcome,
    }


def _serialize_and_validate_trace(
    payload: dict[str, Any], agent_order: list[str]
) -> dict[str, Any]:
    n_agents = len(agent_order)
    active = np.asarray(payload["active_mask"], dtype=np.float64).reshape(-1)
    scaled_ttc = np.asarray(payload["scaled_ttc"], dtype=np.float64).reshape(-1)
    intent = np.asarray(payload["intent"], dtype=np.float64).reshape(-1)
    attention = np.asarray(payload["attention"], dtype=np.float64).reshape(
        n_agents, n_agents
    )
    effective_lambda = float(np.asarray(payload["effective_lambda"]).item())
    if (
        active.size != n_agents
        or scaled_ttc.size != n_agents
        or intent.size != n_agents
    ):
        raise ValueError("CA trace payload does not match the configured agent count")
    if not np.isfinite(scaled_ttc).all() or not np.isfinite(attention).all():
        raise ValueError("CA trace contains non-finite TTC or attention values")
    if not math.isfinite(effective_lambda) or effective_lambda < 0.0:
        raise ValueError("CA trace effective lambda must be finite and non-negative")
    if np.any((scaled_ttc < 0.0) | (scaled_ttc >= 1.0)):
        raise ValueError("CA trace scaled TTC must lie in [0, 1)")
    if np.any((attention < -1e-7) | (attention > 1.0 + 1e-7)):
        raise ValueError("CA trace attention weights must lie in [0, 1]")
    active_bool = active > 0.0
    for receiver in range(n_agents):
        valid_senders = active_bool.copy()
        valid_senders[receiver] = False
        if not active_bool[receiver]:
            if not np.allclose(attention[receiver], 0.0, atol=1e-7):
                raise ValueError("inactive CA trace receiver has non-zero attention")
            continue
        if not np.allclose(attention[receiver, ~valid_senders], 0.0, atol=1e-7):
            raise ValueError("CA trace contains attention to an invalid sender")
        expected_sum = 1.0 if valid_senders.any() else 0.0
        if not math.isclose(
            float(attention[receiver].sum()), expected_sum, rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError("CA trace attention row does not satisfy its mask")
    return {
        "active_mask": {
            agent: bool(active[index] > 0.0) for index, agent in enumerate(agent_order)
        },
        "scaled_ttc": {
            agent: float(scaled_ttc[index]) for index, agent in enumerate(agent_order)
        },
        "intent": {
            agent: int(intent[index] > 0.5) for index, agent in enumerate(agent_order)
        },
        "attention": attention.tolist(),
        "effective_lambda": effective_lambda,
        "use_conflict_prior": bool(payload["use_conflict_prior"]),
    }


def _ego_observations(
    observations: dict[str, Any],
    *,
    agents: list[str],
    vehicles_count: int,
    feature_names: list[str],
) -> dict[str, dict[str, float]]:
    result = {}
    feature_count = len(feature_names)
    for agent in agents:
        rows = np.asarray(observations[agent], dtype=np.float64).reshape(
            vehicles_count, feature_count
        )
        result[agent] = {
            feature: float(rows[0, index])
            for index, feature in enumerate(feature_names)
        }
    return result


def write_ca_trace_artifacts(
    *,
    output_dir: str | Path,
    configs: Any,
    protocol: dict[str, Any],
    model_provenance: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    if not episodes:
        raise ValueError("CA trace artifacts require at least one episode")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "metadata": destination / CA_TRACE_METADATA_FILENAME,
        "records": destination / CA_TRACE_RECORDS_FILENAME,
        "summary": destination / CA_TRACE_SUMMARY_FILENAME,
    }
    plot_dir = destination / "ca_trace_plots"
    plot_targets = []
    for episode in episodes:
        stem = f"attention_seed_{int(episode['episode_seed'])}"
        plot_targets.extend([plot_dir / f"{stem}.png", plot_dir / f"{stem}.pdf"])
    plot_targets.extend(
        [plot_dir / "attention_all_cases.png", plot_dir / "attention_all_cases.pdf"]
    )
    existing = [path for path in [*paths.values(), *plot_targets] if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite CA trace artifacts: {existing}")
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = []
    episode_matrices = []
    for episode in episodes:
        matrix = mean_attention_matrix([episode])
        episode_matrices.append(matrix)
        stem = f"attention_seed_{int(episode['episode_seed'])}"
        plot_paths.extend(
            _write_attention_heatmap(
                matrix,
                episode["agent_order"],
                plot_dir / stem,
                title=f"CA-ComMAPPO mean attention — case {episode['episode_seed']}",
            )
        )
    overall_matrix = mean_attention_matrix(episodes)
    plot_paths.extend(
        _write_attention_heatmap(
            overall_matrix,
            episodes[0]["agent_order"],
            plot_dir / "attention_all_cases",
            title="CA-ComMAPPO mean attention — all trace cases",
        )
    )

    outcomes = [episode["outcome"] for episode in episodes]
    lambda_values = [
        float(step["effective_lambda"])
        for episode in episodes
        for step in episode["trace_steps"]
    ]
    summary = {
        "schema_version": CA_TRACE_SCHEMA_VERSION,
        "episode_summary": summarize_episode_records(outcomes),
        "trace_steps": sum(len(episode["trace_steps"]) for episode in episodes),
        "effective_lambda_min": min(lambda_values),
        "effective_lambda_max": max(lambda_values),
        "mean_attention_all_cases": _json_matrix(overall_matrix),
        "cases": [
            {
                "episode_seed": episode["episode_seed"],
                "steps": len(episode["trace_steps"]),
                "outcome": episode["outcome"],
                "mean_attention": _json_matrix(matrix),
            }
            for episode, matrix in zip(episodes, episode_matrices, strict=True)
        ],
        "plot_paths": [str(path) for path in plot_paths],
    }
    metadata = {
        "schema_version": CA_TRACE_SCHEMA_VERSION,
        "config": json_safe(configs),
        "protocol": json_safe(protocol),
        "model_provenance": json_safe(model_provenance),
        "agent_order": episodes[0]["agent_order"],
    }
    _write_json_exclusive(paths["metadata"], metadata)
    with paths["records"].open("x", encoding="utf-8") as output:
        for episode in episodes:
            output.write(json.dumps(json_safe(episode), sort_keys=True) + "\n")
    _write_json_exclusive(paths["summary"], summary)
    return {**paths, "plots": plot_paths, "summary_payload": summary}


def mean_attention_matrix(episodes: list[dict[str, Any]]) -> np.ndarray:
    if not episodes:
        raise ValueError("cannot average an empty trace episode list")
    agent_order = episodes[0]["agent_order"]
    n_agents = len(agent_order)
    totals = np.zeros((n_agents, n_agents), dtype=np.float64)
    counts = np.zeros((n_agents, n_agents), dtype=np.int64)
    for episode in episodes:
        if episode["agent_order"] != agent_order:
            raise ValueError("CA trace episodes use inconsistent agent order")
        for step in episode["trace_steps"]:
            attention = np.asarray(step["attention"], dtype=np.float64)
            active = np.asarray(
                [step["active_mask"][agent] for agent in agent_order], dtype=bool
            )
            valid = active[:, None] & active[None, :] & ~np.eye(n_agents, dtype=bool)
            totals[valid] += attention[valid]
            counts[valid] += 1
    result = np.full((n_agents, n_agents), np.nan, dtype=np.float64)
    np.divide(totals, counts, out=result, where=counts > 0)
    return result


def _write_attention_heatmap(
    matrix: np.ndarray,
    agent_order: list[str],
    output_stem: Path,
    *,
    title: str,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(5.5, 4.8))
    image = axis.imshow(
        np.ma.masked_invalid(matrix), cmap="viridis", vmin=0.0, vmax=1.0
    )
    axis.set_xticks(range(len(agent_order)), agent_order, rotation=35, ha="right")
    axis.set_yticks(range(len(agent_order)), agent_order)
    axis.set_xlabel("Sender")
    axis.set_ylabel("Receiver")
    axis.set_title(title)
    for receiver in range(matrix.shape[0]):
        for sender in range(matrix.shape[1]):
            value = matrix[receiver, sender]
            label = "—" if not np.isfinite(value) else f"{value:.2f}"
            axis.text(sender, receiver, label, ha="center", va="center", color="white")
    figure.colorbar(image, ax=axis, label="Mean attention weight")
    figure.tight_layout()
    paths = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    for path in paths:
        figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return paths


def _json_matrix(matrix: np.ndarray) -> list[list[float | None]]:
    return [
        [None if not np.isfinite(value) else float(value) for value in row]
        for row in matrix
    ]


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")


__all__ = [
    "CA_TRACE_EPISODE_COUNT",
    "CA_TRACE_METADATA_FILENAME",
    "CA_TRACE_RECORDS_FILENAME",
    "CA_TRACE_SCHEMA_VERSION",
    "CA_TRACE_SUMMARY_FILENAME",
    "ca_trace_protocol",
    "collect_ca_trace_episodes",
    "mean_attention_matrix",
    "write_ca_trace_artifacts",
]
