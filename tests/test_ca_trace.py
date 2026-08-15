import json
from argparse import Namespace

import numpy as np

from ca_commappo.evaluation.ca_trace import (
    ca_trace_protocol,
    mean_attention_matrix,
    write_ca_trace_artifacts,
)
from ca_commappo.evaluation.highway_metrics import build_episode_record


def test_mean_attention_uses_only_steps_where_both_agents_are_active():
    episode = _episode(
        20_000,
        [
            _step([True, True], [[0.0, 1.0], [1.0, 0.0]]),
            _step([True, False], [[0.0, 0.0], [0.0, 0.0]]),
        ],
    )

    matrix = mean_attention_matrix([episode])

    assert np.isnan(matrix[0, 0])
    assert matrix[0, 1] == 1.0
    assert matrix[1, 0] == 1.0
    assert np.isnan(matrix[1, 1])


def test_trace_artifacts_include_machine_records_and_png_pdf_heatmaps(tmp_path):
    episode = _episode(
        20_000,
        [_step([True, True], [[0.0, 1.0], [1.0, 0.0]])],
    )
    protocol = ca_trace_protocol(base_seed=20_000, episode_count=1)

    paths = write_ca_trace_artifacts(
        output_dir=tmp_path,
        configs=Namespace(agent="CA_COMMAPPO", seed=1),
        protocol=protocol,
        model_provenance={"verification": "sidecar_sha256_verified"},
        episodes=[episode],
    )

    assert paths["metadata"].is_file()
    assert paths["records"].is_file()
    assert paths["summary"].is_file()
    assert len(paths["plots"]) == 4
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths["plots"])
    summary_text = paths["summary"].read_text(encoding="utf-8")
    assert "NaN" not in summary_text
    summary = json.loads(summary_text)
    assert summary["effective_lambda_min"] == 1.0
    assert summary["mean_attention_all_cases"] == [[None, 1.0], [1.0, None]]


def _step(active, attention):
    return {
        "active_mask": {
            "agent_0": active[0],
            "agent_1": active[1],
        },
        "attention": attention,
        "effective_lambda": 1.0,
    }


def _episode(seed, steps):
    return {
        "schema_version": 1,
        "episode_seed": seed,
        "agent_order": ["agent_0", "agent_1"],
        "trace_steps": steps,
        "outcome": build_episode_record(
            episode_index=0,
            steps=len(steps),
            agent_rewards={"agent_0": 1.0, "agent_1": 1.0},
            crashed_agents=(False, False),
            arrived_agents=(True, True),
            truncated=False,
            seed=seed,
            waiting_steps=(0, 0),
            dt=0.2,
        ),
    }
