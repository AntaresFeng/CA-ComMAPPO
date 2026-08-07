from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import yaml
from gymnasium import spaces

from ca_commappo.cli.run_sanity_baseline import (
    _apply_protocol_overrides,
    parse_args,
)
from ca_commappo.evaluation.aggregate_results import aggregate
from ca_commappo.evaluation.highway_metrics import build_episode_record
from ca_commappo.evaluation.sanity_baseline_runner import (
    load_sanity_config,
    make_action_rng,
    run_episode,
    run_sanity_baseline,
    select_actions,
    strict_episode_seeds,
    strict_evaluation_protocol,
)
from ca_commappo.evaluation.sanity_eval_artifacts import (
    write_sanity_eval_artifacts,
)


STRICT_CONFIG = Path("configs/sanity/highway_intersection_final.yaml")


def test_strict_config_reuses_task_config_and_defines_exact_final_seeds():
    config = load_sanity_config(STRICT_CONFIG)
    canonical_highway_config = yaml.safe_load(
        Path("configs/highway/intersection-multi-agent-v1.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config.episode_seed_base == 20_000
    assert config.episode_count == 100
    assert config.eval_parallels == 1
    assert config.highway_config == canonical_highway_config
    assert config.highway_config["controlled_vehicles"] == 4
    assert config.highway_config["observation"]["observation_config"][
        "vehicles_count"
    ] == 15
    assert strict_episode_seeds(config) == list(range(20_000, 20_100))
    assert strict_evaluation_protocol(config, "random") == {
        "type": "fixed_explicit_episode_seeds",
        "purpose": "final_test",
        "deterministic_actions": False,
        "reproducible_actions": True,
        "action_selection_mode": "seeded_random",
        "base_seed": 20_000,
        "episode_count": 100,
        "parallels": 1,
        "episode_seeds": list(range(20_000, 20_100)),
    }
    assert strict_evaluation_protocol(config, "idle-only") == {
        "type": "fixed_explicit_episode_seeds",
        "purpose": "final_test",
        "deterministic_actions": True,
        "reproducible_actions": True,
        "action_selection_mode": "fixed_policy",
        "base_seed": 20_000,
        "episode_count": 100,
        "parallels": 1,
        "episode_seeds": list(range(20_000, 20_100)),
    }


def test_load_rejects_partial_or_nonsequential_strict_protocol(tmp_path):
    partial = tmp_path / "partial.yaml"
    partial.write_text("episode_seed_base: 20000\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must either both be set"):
        load_sanity_config(partial)

    parallel = tmp_path / "parallel.yaml"
    parallel.write_text(
        "episode_seed_base: 20000\nepisode_count: 100\neval_parallels: 10\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="executes sequentially"):
        load_sanity_config(parallel)


def test_random_policy_is_replayable_from_episode_seed():
    env = Namespace(
        agents=["agent_0", "agent_1"],
        action_space={
            "agent_0": spaces.Discrete(3),
            "agent_1": spaces.Discrete(3),
        },
    )

    first_rng = make_action_rng(20_037)
    second_rng = make_action_rng(20_037)
    direct_env_seed_rng = np.random.default_rng(20_037)
    first = [select_actions(env, "random", first_rng) for _ in range(20)]
    second = [select_actions(env, "random", second_rng) for _ in range(20)]
    direct_stream = [
        select_actions(env, "random", direct_env_seed_rng) for _ in range(20)
    ]

    assert first == second
    assert first != direct_stream


def test_run_episode_keeps_raw_env_seed_and_uses_child_action_stream(monkeypatch):
    config = load_sanity_config(STRICT_CONFIG)

    class FakeEnv:
        agents = ["agent_0", "agent_1"]
        action_space = {
            "agent_0": spaces.Discrete(3),
            "agent_1": spaces.Discrete(3),
        }

        def __init__(self):
            self.reset_seed = None
            self.actions = None
            self.closed = False

        def reset(self, *, seed):
            self.reset_seed = seed
            return {}, {}

        def step(self, actions):
            self.actions = actions
            return (
                {},
                {"agent_0": 1.0, "agent_1": 3.0},
                {"agent_0": True, "agent_1": True},
                False,
                {
                    "crashed": (False, False),
                    "arrived": (True, True),
                    "waiting_steps": (0, 0),
                    "dt": 0.2,
                },
            )

        def close(self):
            self.closed = True

    env = FakeEnv()
    monkeypatch.setattr(
        "ca_commappo.evaluation.sanity_baseline_runner._make_env", lambda config: env
    )
    expected_rng = make_action_rng(20_037)
    expected_actions = select_actions(env, "random", expected_rng)

    record = run_episode(config, "random", seed=20_037, episode_index=0)

    assert env.reset_seed == 20_037
    assert env.actions == expected_actions
    assert env.closed is True
    assert record["score"] == record["episode_reward"] == 2.0


def test_strict_runner_pairs_policies_on_same_ordered_cases(monkeypatch):
    config = load_sanity_config(STRICT_CONFIG)
    calls: list[tuple[str, int, int, str | None]] = []

    def fake_run_episode(config, policy, seed, episode_index, *, phase=None):
        calls.append((policy, seed, episode_index, phase))
        arrived = seed % 2 == 0
        return build_episode_record(
            phase=phase,
            policy=policy,
            seed=seed,
            episode_index=episode_index,
            steps=10,
            agent_rewards={f"agent_{index}": 1.0 for index in range(4)},
            crashed_agents=[not arrived] + [False] * 3,
            arrived_agents=[arrived] * 4,
            truncated=False,
            score=1.0,
            waiting_steps=[0, 1, 2, 3],
            dt=0.2,
        )

    monkeypatch.setattr(
        "ca_commappo.evaluation.sanity_baseline_runner.run_episode",
        fake_run_episode,
    )

    results = run_sanity_baseline(config)

    expected_seeds = list(range(20_000, 20_100))
    assert [record[1] for record in calls[:100]] == expected_seeds
    assert [record[1] for record in calls[100:]] == expected_seeds
    assert all(record[3] == "test" for record in calls)
    for policy in ("random", "idle-only"):
        episodes = results["policies"][policy]["episodes"]
        assert [episode["seed"] for episode in episodes] == expected_seeds
        assert results["policies"][policy]["summary"]["episodes"] == 100
        assert "mean_waiting_time" in results["policies"][policy]["summary"]
        assert (
            results["policies"][policy]["evaluation_protocol"]
            == results["evaluation_protocols"][policy]
        )
    assert results["evaluation_protocols"]["random"]["deterministic_actions"] is False
    assert (
        results["evaluation_protocols"]["random"]["action_selection_mode"]
        == "seeded_random"
    )
    assert (
        results["evaluation_protocols"]["idle-only"]["action_selection_mode"]
        == "fixed_policy"
    )


def test_legacy_config_keeps_existing_seed_derivation(monkeypatch):
    config = load_sanity_config("configs/sanity/highway_intersection.yaml")
    calls: list[tuple[int, int, str | None]] = []

    def fake_run_episode(config, policy, seed, episode_index, *, phase=None):
        calls.append((seed, episode_index, phase))
        return build_episode_record(
            policy=policy,
            seed=seed,
            episode_index=episode_index,
            steps=1,
            agent_rewards={"agent_0": 0.0},
            crashed_agents=[False],
            arrived_agents=[True],
            truncated=False,
        )

    monkeypatch.setattr(
        "ca_commappo.evaluation.sanity_baseline_runner.run_episode",
        fake_run_episode,
    )

    results = run_sanity_baseline(config, policy="random")

    assert calls == [(seed, index, None) for index, seed in enumerate(range(1, 6))]
    assert "evaluation_protocol" not in results
    assert "evaluation_protocols" not in results


def test_strict_artifacts_encode_policy_specific_action_semantics(
    monkeypatch, tmp_path
):
    config = load_sanity_config(STRICT_CONFIG)

    def fake_run_episode(config, policy, seed, episode_index, *, phase=None):
        return build_episode_record(
            phase=phase,
            policy=policy,
            seed=seed,
            episode_index=episode_index,
            steps=10,
            agent_rewards={f"agent_{index}": float(seed % 5) for index in range(4)},
            crashed_agents=[False] * 4,
            arrived_agents=[True] * 4,
            truncated=False,
            score=float(seed % 5),
            waiting_steps=[0] * 4,
            dt=0.2,
        )

    monkeypatch.setattr(
        "ca_commappo.evaluation.sanity_baseline_runner.run_episode",
        fake_run_episode,
    )
    results = run_sanity_baseline(config)

    records_paths = write_sanity_eval_artifacts(
        results=results,
        config=config,
        artifact_root=tmp_path / "runs",
        created_at="20260807_120000_000000",
    )

    assert records_paths["random"].parent.parent.name == "random_highway"
    assert records_paths["idle-only"].parent.parent.name == "idle_only_highway"
    random_record = json.loads(records_paths["random"].read_text(encoding="utf-8"))
    idle_record = json.loads(
        records_paths["idle-only"].read_text(encoding="utf-8")
    )
    random_metadata = json.loads(
        records_paths["random"]
        .with_name("eval_metadata.json")
        .read_text(encoding="utf-8")
    )
    assert random_record["evaluation_protocol"]["deterministic_actions"] is False
    assert random_record["evaluation_protocol"]["reproducible_actions"] is True
    assert (
        random_record["evaluation_protocol"]["action_selection_mode"]
        == "seeded_random"
    )
    assert random_metadata["evaluation_protocol"] == random_record[
        "evaluation_protocol"
    ]
    assert idle_record["evaluation_protocol"]["deterministic_actions"] is True
    assert (
        idle_record["evaluation_protocol"]["action_selection_mode"]
        == "fixed_policy"
    )
    rows = aggregate(records_paths.values(), expected_seeds=[1])
    assert [row["algorithm"] for row in rows] == ["idle_only", "random"]


def test_cli_protocol_overrides_are_explicit_and_keep_run_seed():
    config = load_sanity_config("configs/sanity/highway_intersection.yaml")
    args = parse_args(
        [
            "--episode-seed-base",
            "30000",
            "--episode-count",
            "12",
            "--run-seed",
            "7",
            "--artifact-root",
            "runs",
        ]
    )

    overridden = _apply_protocol_overrides(config, args)

    assert strict_episode_seeds(overridden) == list(range(30_000, 30_012))
    assert overridden.run_seed == 7
    assert args.artifact_root == Path("runs")
