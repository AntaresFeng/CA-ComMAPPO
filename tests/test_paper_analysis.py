import csv
import json
from argparse import Namespace

import pytest

from ca_commappo.evaluation.aggregate_results import METRICS
from ca_commappo.evaluation.order_robustness import (
    DEFAULT_PERMUTATION_SEEDS,
    write_order_robustness_artifacts,
)
from ca_commappo.evaluation.paper_analysis import (
    PairedComparison,
    build_paired_statistics,
    exact_sign_flip_pvalue,
    load_order_summary,
    require_core_seed_matrix,
    write_paper_analysis,
)


def test_paired_statistics_match_training_seeds_and_keep_direction_explicit():
    rows = [
        _per_seed_row("reference", 2, 2.0),
        _per_seed_row("target", 1, 3.0),
        _per_seed_row("reference", 1, 1.0),
        _per_seed_row("target", 2, 5.0),
    ]
    comparison = PairedComparison("target_minus_reference", "target", "reference")

    raw, summary = build_paired_statistics(rows, comparisons=[comparison])

    arrival_rows = [row for row in raw if row["metric"] == "arrival_rate"]
    assert [row["seed"] for row in arrival_rows] == [1, 2]
    assert [row["delta_target_minus_reference"] for row in arrival_rows] == [2.0, 3.0]
    arrival_summary = next(row for row in summary if row["metric"] == "arrival_rate")
    assert arrival_summary["delta_mean"] == 2.5
    assert arrival_summary["delta_sample_std"] == pytest.approx(2**-0.5)
    assert arrival_summary["target"] == "target"
    assert arrival_summary["reference"] == "reference"


def test_paired_statistics_reject_mismatched_training_seed_sets():
    comparison = PairedComparison("target_minus_reference", "target", "reference")
    rows = [
        _per_seed_row("target", 1, 1.0),
        _per_seed_row("target", 2, 2.0),
        _per_seed_row("reference", 1, 0.0),
    ]

    with pytest.raises(ValueError, match="identical training seeds"):
        build_paired_statistics(rows, comparisons=[comparison])


def test_exact_sign_flip_test_is_two_sided_and_exact():
    assert exact_sign_flip_pvalue([1.0, 1.0, 1.0]) == 0.25
    assert exact_sign_flip_pvalue([0.0, 0.0, 0.0]) == 1.0


def test_final_analysis_requires_all_core_algorithms_at_seeds_one_to_three():
    expected = {
        "mappo": [1, 2, 3],
        "attention_mappo": [1, 2, 3],
        "ca_commappo": [1, 2, 3],
        "ca_commappo_no_prior": [1, 2, 3],
    }

    require_core_seed_matrix(expected)
    expected["ca_commappo_no_prior"] = [1]
    with pytest.raises(ValueError, match="ca_commappo_no_prior"):
        require_core_seed_matrix(expected)


def test_order_summary_is_recomputed_from_all_paired_cases(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "mappo_highway" / "order"
    episodes = [
        {
            "seed": seed,
            "arrival": seed % 2 == 0,
            "collision": seed % 2 != 0,
            "truncated": False,
            "episode_reward": float(seed % 3),
            "steps": 10,
            "agent_rewards": {"agent_0": float(seed % 3)},
            "agent_collision_fraction": float(seed % 2 != 0),
            "agent_arrival_fraction": float(seed % 2 == 0),
            "throughput": float(seed % 2 == 0) / 2,
            "waiting_time": 0.0,
            "travel_time": 2.0,
        }
        for seed in range(20_000, 20_100)
    ]
    result = {
        "episodes": episodes,
        "summary": {
            "episodes": 100,
            "arrival_rate": 0.5,
            "collision_rate": 0.5,
            "mean_episode_reward": 1.0,
        },
    }
    provenance = {
        "verification": "sidecar_sha256_verified",
        "checkpoint_path": str(tmp_path / "best_model.pth"),
        "training_seed": 1,
        "source_commit": "b" * 40,
        "runtime_source_sha256": "c" * 64,
    }
    paths = write_order_robustness_artifacts(
        output_dir=run_dir,
        config=Namespace(agent="MAPPO", seed=1),
        evaluation_protocol={
            "purpose": "observation_order_robustness",
            "base_seed": 20_000,
            "episode_count": 100,
            "episode_seeds": list(range(20_000, 20_100)),
        },
        permutation_protocol={"permutation_seeds": list(DEFAULT_PERMUTATION_SEEDS)},
        model_provenance=provenance,
        baseline_result=result,
        perturbed_results={seed: result for seed in DEFAULT_PERMUTATION_SEEDS},
    )
    monkeypatch.setattr(
        "ca_commappo.evaluation.paper_analysis.validate_model_provenance_payload",
        lambda *args, **kwargs: None,
    )

    rows, source = load_order_summary("mappo", paths["summary"])

    assert source == ("b" * 40, "c" * 64)
    assert len(rows) > 0
    assert {row["cases"] for row in rows} == {100}

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    summary["variants"][0]["paired_case_deltas"]["arrival"]["mean"] = 0.5
    paths["summary"].write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees with pairs"):
        load_order_summary("mappo", paths["summary"])


def test_paper_analysis_writes_immutable_tables_and_png_pdf_plots(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    aggregate_rows = [_aggregate_row("mappo", 1.0)]
    per_seed_rows = [_per_seed_row("mappo", 1, 1.0)]
    paired_rows = [
        {
            "comparison": "test",
            "target": "mappo",
            "reference": "reference",
            "seed": 1,
            "metric": metric,
            "target_value": 1.0,
            "reference_value": 0.0,
            "delta_target_minus_reference": 1.0,
        }
        for metric, _title in (
            ("arrival_rate", ""),
            ("collision_rate", ""),
            ("mean_episode_reward", ""),
        )
    ]
    paired_summary = [
        {
            "comparison": "test",
            "target": "mappo",
            "reference": "reference",
            "metric": metric,
            "seeds": 1,
            "delta_mean": 1.0,
            "delta_sample_std": 0.0,
            "exact_two_sided_sign_flip_p": 1.0,
        }
        for metric, _title in (
            ("arrival_rate", ""),
            ("collision_rate", ""),
            ("mean_episode_reward", ""),
        )
    ]
    order_rows = [
        {
            "algorithm": "mappo",
            "training_seed": 1,
            "permutation_seed": 30_000,
            "metric": metric,
            "cases": 100,
            "paired_case_delta_mean": 0.1,
            "paired_case_delta_sample_std": 0.2,
        }
        for metric, _title in (
            ("arrival", ""),
            ("collision", ""),
            ("episode_reward", ""),
        )
    ]
    destination = tmp_path / "paper"

    write_paper_analysis(
        output_dir=destination,
        manifest_path=manifest,
        aggregate_rows=aggregate_rows,
        per_seed_rows=per_seed_rows,
        paired_rows=paired_rows,
        paired_summary_rows=paired_summary,
        expected_seeds={"mappo": [1]},
        order_rows=order_rows,
        order_inputs={"mappo": "/tmp/order.json"},
    )

    expected_files = {
        "analysis_metadata.json",
        "main_results.csv",
        "main_results.md",
        "main_results.pdf",
        "main_results.png",
        "order_robustness.csv",
        "order_robustness.pdf",
        "order_robustness.png",
        "paired_deltas.csv",
        "paired_deltas.pdf",
        "paired_deltas.png",
        "paired_summary.csv",
        "per_seed_metrics.csv",
    }
    assert {path.name for path in destination.iterdir()} == expected_files
    assert all((destination / name).stat().st_size > 0 for name in expected_files)
    metadata = json.loads(
        (destination / "analysis_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["statistical_unit"] == "training_seed"
    assert metadata["paired_delta_direction"] == "target_minus_reference"
    assert metadata["files"] == sorted(expected_files)
    with (destination / "per_seed_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        assert next(iter(csv.DictReader(source)))["seed"] == "1"

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_paper_analysis(
            output_dir=destination,
            manifest_path=manifest,
            aggregate_rows=aggregate_rows,
            per_seed_rows=per_seed_rows,
            paired_rows=paired_rows,
            paired_summary_rows=paired_summary,
            expected_seeds={"mappo": [1]},
        )


def _per_seed_row(algorithm, seed, value):
    return {
        "algorithm": algorithm,
        "seed": seed,
        "checkpoint_sha256": "a" * 64,
        "source_commit": "b" * 40,
        "runtime_source_sha256": "c" * 64,
        "training_device": "cpu",
        **{metric: value for metric in METRICS},
    }


def _aggregate_row(algorithm, value):
    return {
        "algorithm": algorithm,
        "seeds": 1,
        **{
            key: item
            for metric in METRICS
            for key, item in (
                (f"{metric}_mean", value),
                (f"{metric}_std", 0.0),
            )
        },
    }
