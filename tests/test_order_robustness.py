import json
from argparse import Namespace

import numpy as np

from ca_commappo.evaluation.order_robustness import (
    DEFAULT_PERMUTATION_SEEDS,
    build_order_robustness_pairs,
    make_neighbour_row_permuter,
    observation_permutation_protocol,
    write_order_robustness_artifacts,
)


def test_neighbour_permuter_preserves_ego_rows_and_row_multisets():
    original = {
        "agent_1": np.arange(40, 50, dtype=np.float32),
        "agent_0": np.arange(10, dtype=np.float32),
    }
    snapshot = {agent: value.copy() for agent, value in original.items()}
    transform = make_neighbour_row_permuter(
        permutation_seed=30_000,
        vehicles_count=5,
        feature_count=2,
    )

    first = transform(original, 20_000, 0)
    second = transform(original, 20_000, 0)

    for agent, original_value in original.items():
        original_rows = original_value.reshape(5, 2)
        permuted_rows = first[agent].reshape(5, 2)
        assert np.array_equal(permuted_rows[0], original_rows[0])
        assert sorted(map(tuple, permuted_rows[1:])) == sorted(
            map(tuple, original_rows[1:])
        )
        assert np.array_equal(first[agent], second[agent])
        assert np.array_equal(original_value, snapshot[agent])


def test_permutation_protocol_freezes_actor_only_schedule():
    protocol = observation_permutation_protocol(
        permutation_seeds=DEFAULT_PERMUTATION_SEEDS,
        vehicles_count=4,
        feature_names=["presence", "x"],
    )

    assert protocol["target"] == "actor_local_observation_only"
    assert protocol["ego_row"] == 0
    assert protocol["permuted_rows"] == [1, 2, 3]
    assert protocol["global_state_unchanged"] is True
    assert protocol["agent_mask_unchanged"] is True


def test_paired_artifacts_match_cases_and_report_perturbed_minus_baseline(tmp_path):
    baseline = _result(
        [
            _episode(20_000, arrival=True, collision=False, reward=2.0),
            _episode(20_001, arrival=False, collision=True, reward=-1.0),
        ]
    )
    perturbed = {
        30_000: _result(
            [
                _episode(20_000, arrival=False, collision=True, reward=-2.0),
                _episode(20_001, arrival=False, collision=True, reward=1.0),
            ]
        )
    }

    pairs = build_order_robustness_pairs(
        baseline_result=baseline,
        perturbed_results=perturbed,
    )
    paths = write_order_robustness_artifacts(
        output_dir=tmp_path,
        config=Namespace(agent="MAPPO", seed=1),
        evaluation_protocol={"episode_seeds": [20_000, 20_001]},
        permutation_protocol={"permutation_seeds": [30_000]},
        model_provenance={"verification": "sidecar_sha256_verified"},
        baseline_result=baseline,
        perturbed_results=perturbed,
    )

    assert pairs[0]["episode_seed"] == 20_000
    assert pairs[0]["perturbations"][0]["deltas"]["arrival"] == -1.0
    assert pairs[0]["perturbations"][0]["deltas"]["collision"] == 1.0
    assert pairs[0]["perturbations"][0]["deltas"]["episode_reward"] == -4.0
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert (
        summary["variants"][0]["paired_case_deltas"]["episode_reward"]["mean"] == -1.0
    )
    assert len(paths["pairs"].read_text(encoding="utf-8").splitlines()) == 2


def _episode(seed, *, arrival, collision, reward):
    return {
        "seed": seed,
        "arrival": arrival,
        "collision": collision,
        "truncated": False,
        "episode_reward": reward,
        "steps": 10,
        "agent_rewards": {"agent_0": reward},
        "agent_collision_fraction": float(collision),
        "agent_arrival_fraction": float(arrival),
        "throughput": float(arrival) / 2,
        "waiting_time": 0.0,
        "travel_time": 2.0,
    }


def _result(episodes):
    return {
        "episodes": episodes,
        "summary": {
            "episodes": len(episodes),
            "arrival_rate": sum(episode["arrival"] for episode in episodes)
            / len(episodes),
            "collision_rate": sum(episode["collision"] for episode in episodes)
            / len(episodes),
            "mean_episode_reward": sum(
                episode["episode_reward"] for episode in episodes
            )
            / len(episodes),
        },
    }
