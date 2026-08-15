"""Build strict multi-seed paper tables, paired statistics, and plots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ca_commappo.evaluation.aggregate_results import (
    METRICS,
    AggregationError,
    aggregate_with_per_seed,
    algo_from_path,
    load_runs_manifest,
)
from ca_commappo.evaluation.model_provenance import (
    VERIFIED_PROVENANCE,
    validate_model_provenance_payload,
)
from ca_commappo.evaluation.order_robustness import (
    DEFAULT_PERMUTATION_SEEDS,
    ORDER_ROBUSTNESS_METADATA_FILENAME,
    ORDER_ROBUSTNESS_PAIRS_FILENAME,
    ORDER_ROBUSTNESS_SCHEMA_VERSION,
)

PAPER_ANALYSIS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PairedComparison:
    name: str
    target: str
    reference: str


CORE_COMPARISONS = (
    PairedComparison(
        name="attention_mappo_minus_mappo",
        target="attention_mappo",
        reference="mappo",
    ),
    PairedComparison(
        name="ca_commappo_minus_attention_mappo",
        target="ca_commappo",
        reference="attention_mappo",
    ),
    PairedComparison(
        name="ca_commappo_minus_no_prior",
        target="ca_commappo",
        reference="ca_commappo_no_prior",
    ),
)
CORE_TRAINING_SEEDS = (1, 2, 3)

_DISPLAY_NAMES = {
    "random": "Random",
    "idle_only": "Idle-only",
    "ippo": "IPPO",
    "mappo": "MAPPO",
    "attention_mappo": "Attention-MAPPO",
    "commnet": "CommNet",
    "ic3net": "IC3Net",
    "ca_commappo": "CA-ComMAPPO",
    "ca_commappo_no_prior": "CA no-prior",
}
_ALGORITHM_ORDER = tuple(_DISPLAY_NAMES)
_MAIN_PLOT_METRICS = (
    ("arrival_rate", "Episode arrival rate"),
    ("collision_rate", "Episode collision rate"),
    ("mean_episode_reward", "Mean episode reward"),
    ("mean_throughput", "Mean throughput"),
)
_PAIRED_PLOT_METRICS = (
    ("arrival_rate", "Arrival-rate delta"),
    ("collision_rate", "Collision-rate delta"),
    ("mean_episode_reward", "Episode-reward delta"),
)
_ORDER_PLOT_METRICS = (
    ("arrival", "Arrival delta"),
    ("collision", "Collision delta"),
    ("episode_reward", "Episode-reward delta"),
)


def exact_sign_flip_pvalue(values: Sequence[float]) -> float:
    """Return the exact two-sided paired sign-flip p-value."""
    deltas = [float(value) for value in values]
    if not deltas:
        raise ValueError("paired sign-flip test requires at least one delta")
    if not all(math.isfinite(value) for value in deltas):
        raise ValueError("paired sign-flip deltas must be finite")
    observed = abs(statistics.mean(deltas))
    extreme = 0
    total = 2 ** len(deltas)
    tolerance = 1e-15
    for signs in itertools.product((-1.0, 1.0), repeat=len(deltas)):
        permuted = abs(
            math.fsum(sign * value for sign, value in zip(signs, deltas)) / len(deltas)
        )
        if permuted + tolerance >= observed:
            extreme += 1
    return extreme / total


def build_paired_statistics(
    per_seed_rows: Sequence[Mapping[str, Any]],
    *,
    comparisons: Sequence[PairedComparison] = CORE_COMPARISONS,
    require_all: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair algorithm outcomes by training seed and summarize their deltas."""
    by_algorithm: dict[str, dict[int, Mapping[str, Any]]] = {}
    for row in per_seed_rows:
        algorithm = str(row["algorithm"])
        seed = row["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("per-seed metric rows require integer training seeds")
        if seed in by_algorithm.setdefault(algorithm, {}):
            raise ValueError(
                f"duplicate per-seed row for algorithm={algorithm!r}, seed={seed}"
            )
        by_algorithm[algorithm][seed] = row

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        target_rows = by_algorithm.get(comparison.target)
        reference_rows = by_algorithm.get(comparison.reference)
        if target_rows is None or reference_rows is None:
            if require_all:
                missing = [
                    algorithm
                    for algorithm, rows in (
                        (comparison.target, target_rows),
                        (comparison.reference, reference_rows),
                    )
                    if rows is None
                ]
                raise ValueError(
                    f"comparison {comparison.name!r} is missing algorithms: {missing}"
                )
            continue
        target_seeds = set(target_rows)
        reference_seeds = set(reference_rows)
        if target_seeds != reference_seeds:
            raise ValueError(
                f"comparison {comparison.name!r} requires identical training seeds: "
                f"target={sorted(target_seeds)}, reference={sorted(reference_seeds)}"
            )
        for metric in METRICS:
            deltas = []
            for seed in sorted(target_seeds):
                target_value = float(target_rows[seed][metric])
                reference_value = float(reference_rows[seed][metric])
                delta = target_value - reference_value
                deltas.append(delta)
                raw_rows.append(
                    {
                        "comparison": comparison.name,
                        "target": comparison.target,
                        "reference": comparison.reference,
                        "seed": seed,
                        "metric": metric,
                        "target_value": target_value,
                        "reference_value": reference_value,
                        "delta_target_minus_reference": delta,
                    }
                )
            summary_rows.append(
                {
                    "comparison": comparison.name,
                    "target": comparison.target,
                    "reference": comparison.reference,
                    "metric": metric,
                    "seeds": len(deltas),
                    "delta_mean": statistics.mean(deltas),
                    "delta_sample_std": (
                        statistics.stdev(deltas) if len(deltas) > 1 else 0.0
                    ),
                    "exact_two_sided_sign_flip_p": exact_sign_flip_pvalue(deltas),
                }
            )
    return raw_rows, summary_rows


def require_core_seed_matrix(expected_seeds: Mapping[str, Sequence[int]]) -> None:
    """Require the pre-registered four-algorithm, three-training-seed matrix."""
    core_algorithms = {
        algorithm
        for comparison in CORE_COMPARISONS
        for algorithm in (comparison.target, comparison.reference)
    }
    for algorithm in sorted(core_algorithms):
        observed = expected_seeds.get(algorithm)
        if observed is None or tuple(observed) != CORE_TRAINING_SEEDS:
            raise ValueError(
                f"final paper analysis requires {algorithm!r} training seeds "
                f"{list(CORE_TRAINING_SEEDS)}; got {observed!r}"
            )


def load_order_summary(
    algorithm: str,
    path: str | Path,
) -> tuple[list[dict[str, Any]], tuple[str, str]]:
    """Validate one order-eval artifact and return long-form paired case summaries."""
    summary_path = Path(path)
    path_algorithm = algo_from_path(summary_path)
    if path_algorithm != algorithm:
        raise ValueError(
            f"order summary label {algorithm!r} does not match its run path "
            f"algorithm {path_algorithm!r}: {summary_path}"
        )
    summary = _load_json(summary_path)
    if summary.get("schema_version") != ORDER_ROBUSTNESS_SCHEMA_VERSION:
        raise ValueError(f"unsupported order-robustness schema: {summary_path}")
    metadata_path = summary_path.with_name(ORDER_ROBUSTNESS_METADATA_FILENAME)
    metadata = _load_json(metadata_path)
    if metadata.get("schema_version") != ORDER_ROBUSTNESS_SCHEMA_VERSION:
        raise ValueError(f"unsupported order-robustness metadata: {metadata_path}")
    provenance = metadata.get("model_provenance")
    if not isinstance(provenance, dict):
        raise TypeError(f"order-eval metadata lacks model provenance: {metadata_path}")
    if provenance.get("verification") != VERIFIED_PROVENANCE:
        raise ValueError(f"order-eval model is not strictly verified: {metadata_path}")
    checkpoint_path = provenance.get("checkpoint_path")
    training_seed = provenance.get("training_seed")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise ValueError(
            f"order-eval provenance lacks checkpoint path: {metadata_path}"
        )
    if isinstance(training_seed, bool) or not isinstance(training_seed, int):
        raise TypeError(f"order-eval provenance lacks training seed: {metadata_path}")
    if training_seed != 1:
        raise ValueError(f"formal order-eval requires training seed 1: {metadata_path}")
    validate_model_provenance_payload(
        provenance,
        checkpoint_path=Path(checkpoint_path),
        expected_training_seed=training_seed,
        expected_configs=None,
        verify_current_source=False,
        require_clean_source=True,
    )

    permutation_protocol = metadata.get("permutation_protocol")
    evaluation_protocol = metadata.get("evaluation_protocol")
    if not isinstance(permutation_protocol, dict) or not isinstance(
        evaluation_protocol, dict
    ):
        raise TypeError(f"order-eval metadata lacks protocols: {metadata_path}")
    expected_episode_seeds = list(range(20_000, 20_100))
    expected_evaluation = {
        "purpose": "observation_order_robustness",
        "base_seed": 20_000,
        "episode_count": 100,
        "episode_seeds": expected_episode_seeds,
    }
    for field, expected in expected_evaluation.items():
        if evaluation_protocol.get(field) != expected:
            raise ValueError(
                f"order-eval protocol requires {field}={expected!r}: {metadata_path}"
            )
    expected_permutation_seeds = permutation_protocol.get("permutation_seeds")
    if expected_permutation_seeds != list(DEFAULT_PERMUTATION_SEEDS):
        raise ValueError(
            f"order-eval requires permutation seeds "
            f"{list(DEFAULT_PERMUTATION_SEEDS)}: {metadata_path}"
        )
    variants = summary.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError(f"order-eval summary has no variants: {summary_path}")
    observed_permutation_seeds = [
        variant.get("permutation_seed") for variant in variants
    ]
    if observed_permutation_seeds != expected_permutation_seeds:
        raise ValueError(
            f"order-eval permutation seeds disagree with metadata: {summary_path}"
        )
    _validate_order_pairs(summary_path, variants)

    rows = []
    for variant in variants:
        paired = variant.get("paired_case_deltas")
        if not isinstance(paired, dict):
            raise TypeError(f"order-eval variant lacks paired deltas: {summary_path}")
        for metric, values in sorted(paired.items()):
            if not isinstance(values, dict):
                raise TypeError(
                    f"invalid paired order metric {metric!r}: {summary_path}"
                )
            cases = values.get("cases")
            mean = values.get("mean")
            sample_std = values.get("sample_std")
            if (
                isinstance(cases, bool)
                or not isinstance(cases, int)
                or cases != 100
                or isinstance(mean, bool)
                or not isinstance(mean, (int, float))
                or isinstance(sample_std, bool)
                or not isinstance(sample_std, (int, float))
                or not math.isfinite(float(mean))
                or not math.isfinite(float(sample_std))
            ):
                raise ValueError(
                    f"invalid paired order metric {metric!r}: {summary_path}"
                )
            rows.append(
                {
                    "algorithm": algorithm,
                    "training_seed": training_seed,
                    "permutation_seed": variant["permutation_seed"],
                    "metric": metric,
                    "cases": cases,
                    "paired_case_delta_mean": float(mean),
                    "paired_case_delta_sample_std": float(sample_std),
                }
            )
    return rows, (
        str(provenance["source_commit"]),
        str(provenance["runtime_source_sha256"]),
    )


def _validate_order_pairs(
    summary_path: Path, variants: Sequence[Mapping[str, Any]]
) -> None:
    pairs_path = summary_path.with_name(ORDER_ROBUSTNESS_PAIRS_FILENAME)
    pairs = []
    try:
        with pairs_path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    pair = json.loads(line)
                    if not isinstance(pair, dict):
                        raise TypeError("order-eval pair records must be objects")
                    pairs.append(pair)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid order-eval pairs {pairs_path}: {error}") from error
    expected_episode_seeds = list(range(20_000, 20_100))
    observed_episode_seeds = [pair.get("episode_seed") for pair in pairs]
    if observed_episode_seeds != expected_episode_seeds:
        raise ValueError(
            f"order-eval pairs require cases 20000..20099 in order: {pairs_path}"
        )

    paired_values: dict[int, dict[str, list[float]]] = {
        int(variant["permutation_seed"]): {} for variant in variants
    }
    expected_permutation_seeds = set(paired_values)
    for pair in pairs:
        if pair.get("schema_version") != ORDER_ROBUSTNESS_SCHEMA_VERSION:
            raise ValueError(f"unsupported order-eval pair schema: {pairs_path}")
        perturbations = pair.get("perturbations")
        if not isinstance(perturbations, list):
            raise TypeError(f"order-eval pair lacks perturbations: {pairs_path}")
        observed = {
            perturbation.get("permutation_seed") for perturbation in perturbations
        }
        if observed != expected_permutation_seeds or len(perturbations) != len(
            expected_permutation_seeds
        ):
            raise ValueError(
                f"order-eval pair has inconsistent permutation seeds: {pairs_path}"
            )
        for perturbation in perturbations:
            permutation_seed = int(perturbation["permutation_seed"])
            deltas = perturbation.get("deltas")
            if not isinstance(deltas, dict):
                raise TypeError(f"order-eval perturbation lacks deltas: {pairs_path}")
            for metric, value in deltas.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError(f"non-finite order-eval delta: {pairs_path}")
                paired_values[permutation_seed].setdefault(metric, []).append(numeric)

    for variant in variants:
        permutation_seed = int(variant["permutation_seed"])
        summaries = variant.get("paired_case_deltas")
        if not isinstance(summaries, dict):
            raise TypeError(f"order-eval variant lacks paired deltas: {summary_path}")
        if set(summaries) != set(paired_values[permutation_seed]):
            raise ValueError(
                f"order-eval summary metric set disagrees with pairs: {summary_path}"
            )
        for metric, stored in summaries.items():
            values = paired_values[permutation_seed][metric]
            if len(values) != 100 or not isinstance(stored, dict):
                raise ValueError(
                    f"order-eval paired metric {metric!r} requires 100 cases: "
                    f"{summary_path}"
                )
            expected = {
                "cases": 100,
                "mean": statistics.mean(values),
                "sample_std": statistics.stdev(values),
            }
            for field, expected_value in expected.items():
                stored_value = stored.get(field)
                matches = (
                    stored_value == expected_value
                    if field == "cases"
                    else isinstance(stored_value, (int, float))
                    and not isinstance(stored_value, bool)
                    and math.isclose(
                        float(stored_value),
                        float(expected_value),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                )
                if not matches:
                    raise ValueError(
                        f"order-eval summary {metric}.{field} disagrees with pairs: "
                        f"{summary_path}"
                    )


def write_paper_analysis(
    *,
    output_dir: str | Path,
    manifest_path: str | Path,
    aggregate_rows: Sequence[Mapping[str, Any]],
    per_seed_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
    paired_summary_rows: Sequence[Mapping[str, Any]],
    expected_seeds: Mapping[str, Sequence[int]],
    order_rows: Sequence[Mapping[str, Any]] = (),
    order_inputs: Mapping[str, str] | None = None,
) -> Path:
    """Write one immutable, self-contained analysis directory."""
    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite analysis directory: {destination}"
        )

    manifest = Path(manifest_path)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as temporary_parent:
        temporary = Path(temporary_parent) / "artifacts"
        temporary.mkdir()
        _write_csv(temporary / "main_results.csv", aggregate_rows)
        _write_csv(temporary / "per_seed_metrics.csv", per_seed_rows)
        _write_csv(temporary / "paired_deltas.csv", paired_rows)
        _write_csv(temporary / "paired_summary.csv", paired_summary_rows)
        (temporary / "main_results.md").write_text(
            _main_results_markdown(aggregate_rows), encoding="utf-8"
        )
        _plot_main_results(aggregate_rows, temporary / "main_results")
        _plot_paired_deltas(
            paired_rows,
            paired_summary_rows,
            temporary / "paired_deltas",
        )
        if order_rows:
            _write_csv(temporary / "order_robustness.csv", order_rows)
            _plot_order_robustness(order_rows, temporary / "order_robustness")

        metadata = {
            "schema_version": PAPER_ANALYSIS_SCHEMA_VERSION,
            "statistical_unit": "training_seed",
            "dispersion": "sample_standard_deviation",
            "paired_delta_direction": "target_minus_reference",
            "paired_test": "exact_two_sided_sign_flip",
            "paired_test_role": "auxiliary_exploratory_evidence",
            "fixed_case_statistics_role": "auxiliary_not_training_seed_variance",
            "runs_manifest": str(manifest.resolve()),
            "runs_manifest_sha256": _sha256_file(manifest),
            "expected_seeds": {
                algorithm: list(seeds)
                for algorithm, seeds in sorted(expected_seeds.items())
            },
            "comparisons": [
                {
                    "name": comparison.name,
                    "target": comparison.target,
                    "reference": comparison.reference,
                }
                for comparison in CORE_COMPARISONS
                if any(
                    row["comparison"] == comparison.name for row in paired_summary_rows
                )
            ],
            "order_summary_inputs": dict(sorted((order_inputs or {}).items())),
            "files": sorted(
                {
                    "analysis_metadata.json",
                    *(
                        str(path.relative_to(temporary))
                        for path in temporary.rglob("*")
                        if path.is_file()
                    ),
                }
            ),
        }
        (temporary / "analysis_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(destination)
    return destination


def _main_results_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    metrics = (
        "arrival_rate",
        "collision_rate",
        "mean_episode_reward",
        "mean_throughput",
        "mean_waiting_time",
        "mean_travel_time",
    )
    header = ["Algorithm", "Seeds", *metrics]
    lines = [
        "# Main final-test results",
        "",
        "Training seed is the statistical unit. Values are mean ± sample std; ",
        "fixed static policies are reported as point estimates.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---", "---:", *(["---:"] * len(metrics))]) + " |",
    ]
    for row in rows:
        values = [_display_name(str(row["algorithm"])), str(row["seeds"])]
        for metric in metrics:
            mean = float(row[f"{metric}_mean"])
            std = row[f"{metric}_std"]
            values.append(
                f"{mean:.4f}" if std is None else f"{mean:.4f} ± {float(std):.4f}"
            )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _plot_main_results(rows: Sequence[Mapping[str, Any]], output_stem: Path) -> None:
    ordered = sorted(rows, key=lambda row: _algorithm_sort_key(str(row["algorithm"])))
    labels = [_display_name(str(row["algorithm"])) for row in ordered]
    positions = np.arange(len(ordered))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, _MAIN_PLOT_METRICS):
        means = [float(row[f"{metric}_mean"]) for row in ordered]
        errors = [
            0.0 if row[f"{metric}_std"] is None else float(row[f"{metric}_std"])
            for row in ordered
        ]
        axis.bar(
            positions,
            means,
            yerr=errors,
            capsize=4,
            color="#4472C4",
            alpha=0.85,
        )
        axis.set_title(title)
        axis.set_xticks(positions, labels, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Final test across training seeds (mean ± sample std)")
    _save_figure(figure, output_stem)


def _plot_paired_deltas(
    rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    output_stem: Path,
) -> None:
    comparisons = list(dict.fromkeys(str(row["comparison"]) for row in summary_rows))
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    if not comparisons:
        for axis in axes:
            axis.text(0.5, 0.5, "No complete core comparisons", ha="center")
            axis.set_axis_off()
        _save_figure(figure, output_stem)
        return
    positions = np.arange(len(comparisons))
    for axis, (metric, title) in zip(axes, _PAIRED_PLOT_METRICS):
        for comparison_index, comparison in enumerate(comparisons):
            matching = [
                row
                for row in rows
                if row["comparison"] == comparison and row["metric"] == metric
            ]
            values = [float(row["delta_target_minus_reference"]) for row in matching]
            offsets = np.linspace(-0.08, 0.08, len(values)) if values else []
            axis.scatter(
                comparison_index + offsets,
                values,
                color="#4472C4",
                label="training seed" if comparison_index == 0 else None,
                zorder=3,
            )
            summary = next(
                row
                for row in summary_rows
                if row["comparison"] == comparison and row["metric"] == metric
            )
            axis.scatter(
                comparison_index,
                float(summary["delta_mean"]),
                marker="D",
                color="#C00000",
                label="mean" if comparison_index == 0 else None,
                zorder=4,
            )
            axis.annotate(
                f"p={float(summary['exact_two_sided_sign_flip_p']):.3g}",
                (comparison_index, float(summary["delta_mean"])),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        axis.axhline(0.0, color="black", linewidth=1, alpha=0.6)
        axis.set_title(title)
        axis.set_xticks(
            positions,
            [_comparison_display_name(value) for value in comparisons],
            rotation=25,
            ha="right",
        )
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    figure.suptitle("Paired training-seed deltas (target − reference)")
    _save_figure(figure, output_stem)


def _plot_order_robustness(
    rows: Sequence[Mapping[str, Any]], output_stem: Path
) -> None:
    algorithms = list(dict.fromkeys(str(row["algorithm"]) for row in rows))
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, (metric, title) in zip(axes, _ORDER_PLOT_METRICS):
        for algorithm_index, algorithm in enumerate(algorithms):
            matching = sorted(
                (
                    row
                    for row in rows
                    if row["algorithm"] == algorithm and row["metric"] == metric
                ),
                key=lambda row: int(row["permutation_seed"]),
            )
            x_values = np.arange(len(matching)) + algorithm_index * 0.08
            means = [float(row["paired_case_delta_mean"]) for row in matching]
            errors = [float(row["paired_case_delta_sample_std"]) for row in matching]
            axis.errorbar(
                x_values,
                means,
                yerr=errors,
                marker="o",
                capsize=3,
                label=_display_name(algorithm),
            )
        reference_rows = [row for row in rows if row["metric"] == metric]
        permutation_seeds = sorted(
            {int(row["permutation_seed"]) for row in reference_rows}
        )
        axis.axhline(0.0, color="black", linewidth=1, alpha=0.6)
        axis.set_title(title)
        axis.set_xticks(range(len(permutation_seeds)), permutation_seeds, rotation=25)
        axis.set_xlabel("Permutation seed")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    figure.suptitle("Observation-order perturbation: paired case deltas")
    _save_figure(figure, output_stem)


def _save_figure(figure: plt.Figure, output_stem: Path) -> None:
    for suffix in (".png", ".pdf"):
        figure.savefig(output_stem.with_suffix(suffix), dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["no_rows"]
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_order_inputs(values: Sequence[str]) -> dict[str, str]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                "--order-summary values must use ALGORITHM=/path/to/summary.json"
            )
        algorithm, path = value.split("=", 1)
        if not algorithm or not path:
            raise ValueError(
                "--order-summary values must use ALGORITHM=/path/to/summary.json"
            )
        if algorithm in parsed:
            raise ValueError(f"duplicate order summary algorithm: {algorithm}")
        parsed[algorithm] = path
    return parsed


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as source:
            payload = json.load(source)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"JSON artifact must be an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _learned_source(per_seed_rows: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    sources = {
        (str(row["source_commit"]), str(row["runtime_source_sha256"]))
        for row in per_seed_rows
        if row.get("checkpoint_sha256") is not None
    }
    if len(sources) != 1:
        raise ValueError(f"expected one learned-policy source identity, got {sources}")
    return next(iter(sources))


def _display_name(algorithm: str) -> str:
    return _DISPLAY_NAMES.get(algorithm, algorithm.replace("_", " ").title())


def _algorithm_sort_key(algorithm: str) -> tuple[int, str]:
    try:
        return _ALGORITHM_ORDER.index(algorithm), algorithm
    except ValueError:
        return len(_ALGORITHM_ORDER), algorithm


def _comparison_display_name(name: str) -> str:
    comparison = next(
        (candidate for candidate in CORE_COMPARISONS if candidate.name == name), None
    )
    if comparison is None:
        return name
    return f"{_display_name(comparison.target)} − {_display_name(comparison.reference)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-manifest",
        required=True,
        help="strict explicit final-test manifest accepted by aggregate_results",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new immutable output directory for tables and plots",
    )
    parser.add_argument(
        "--order-summary",
        action="append",
        default=[],
        metavar="ALGORITHM=PATH",
        help="optional validated order_robustness_summary.json (repeatable)",
    )
    parser.add_argument(
        "--allow-partial-comparisons",
        action="store_true",
        help="development-only: omit core comparisons whose algorithms are absent",
    )
    args = parser.parse_args(argv)

    try:
        paths, expected_seeds = load_runs_manifest(args.runs_manifest)
        if not args.allow_partial_comparisons:
            require_core_seed_matrix(expected_seeds)
        aggregate_rows, per_seed_rows = aggregate_with_per_seed(
            paths,
            expected_seeds_by_algorithm=expected_seeds,
        )
        paired_rows, paired_summary_rows = build_paired_statistics(
            per_seed_rows,
            require_all=not args.allow_partial_comparisons,
        )
        order_inputs = _parse_order_inputs(args.order_summary)
        source = _learned_source(per_seed_rows)
        order_rows = []
        for algorithm, path in order_inputs.items():
            rows, order_source = load_order_summary(algorithm, path)
            if order_source != source:
                raise ValueError(
                    f"order-eval source for {algorithm!r} differs from final results"
                )
            order_rows.extend(rows)
        output = write_paper_analysis(
            output_dir=args.output_dir,
            manifest_path=args.runs_manifest,
            aggregate_rows=aggregate_rows,
            per_seed_rows=per_seed_rows,
            paired_rows=paired_rows,
            paired_summary_rows=paired_summary_rows,
            expected_seeds=expected_seeds,
            order_rows=order_rows,
            order_inputs=order_inputs,
        )
    except (
        AggregationError,
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Paper analysis failed: {error}", file=sys.stderr)
        return 1
    print(f"Wrote paper analysis: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CORE_COMPARISONS",
    "PAPER_ANALYSIS_SCHEMA_VERSION",
    "PairedComparison",
    "build_paired_statistics",
    "exact_sign_flip_pvalue",
    "load_order_summary",
    "require_core_seed_matrix",
    "write_paper_analysis",
]
