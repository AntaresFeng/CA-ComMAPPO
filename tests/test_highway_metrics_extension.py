"""Tests for the Phase 4 metric extension (throughput / waiting / travel time).

The new metrics are *conditional* (backward-compatible): they appear only when
the episode record carries ``waiting_steps`` and ``dt`` (supplied by the adapter
in real runs). When absent, the summary keeps the original 9 keys. The best-model
sort key (``_summary_sort_key``) must NOT change.
"""

from __future__ import annotations


import pytest

from ca_commappo.evaluation.highway_metrics import (
    build_episode_record,
    summarize_episode_records,
)
from ca_commappo.evaluation.mappo_highway_metrics import (
    _summary_sort_key,
    is_better_highway_summary,
)


def _record(steps, arrived, waiting_steps=None, dt=None):
    return build_episode_record(
        episode_index=0,
        steps=steps,
        agent_rewards={"agent_0": 1.0, "agent_1": 1.0},
        crashed_agents=[False, False],
        arrived_agents=arrived,
        truncated=False,
        waiting_steps=waiting_steps,
        dt=dt,
    )


class TestNewMetricsConditional:
    def test_record_has_travel_throughput_waiting_when_dt_and_waiting_given(self):
        record = _record(steps=10, arrived=[True, True], waiting_steps=[2, 4], dt=0.2)
        # travel_time = steps * dt = 2.0
        assert record["travel_time"] == pytest.approx(2.0)
        # throughput = arrived_count / (steps*dt) = 2 / 2.0 = 1.0
        assert record["throughput"] == pytest.approx(1.0)
        # waiting_time = mean(waiting_steps) * dt = 3.0 * 0.2 = 0.6
        assert record["waiting_time"] == pytest.approx(0.6)

    def test_record_has_travel_and_throughput_when_only_dt_given(self):
        record = _record(steps=10, arrived=[True, False], dt=0.2)
        assert record["travel_time"] == pytest.approx(2.0)
        assert record["throughput"] == pytest.approx(0.5)  # 1 arrived / 2.0 s
        assert "waiting_time" not in record

    def test_record_omits_new_metrics_when_dt_absent(self):
        # Backward-compatible: no dt -> no new keys (original 9-key path).
        record = _record(steps=10, arrived=[True, True])
        for key in ("travel_time", "throughput", "waiting_time"):
            assert key not in record

    def test_throughput_zero_when_episode_duration_zero(self):
        record = _record(steps=0, arrived=[True, True], dt=0.2)
        # steps=0 -> duration 0 -> throughput defined as 0 (avoid div-by-zero).
        assert record["throughput"] == 0.0


class TestSummarizeNewMetrics:
    def test_summary_includes_mean_new_metrics_when_present(self):
        records = [
            _record(steps=10, arrived=[True, True], waiting_steps=[2, 4], dt=0.2),
            _record(steps=20, arrived=[True, False], waiting_steps=[0, 2], dt=0.2),
        ]
        summary = summarize_episode_records(records)
        assert "mean_travel_time" in summary
        assert "mean_throughput" in summary
        assert "mean_waiting_time" in summary
        # travel_time per episode: 2.0, 4.0 -> mean 3.0
        assert summary["mean_travel_time"] == pytest.approx(3.0)

    def test_summary_omits_new_metrics_when_absent(self):
        records = [
            _record(steps=10, arrived=[True, True]),
            _record(steps=20, arrived=[True, False]),
        ]
        summary = summarize_episode_records(records)
        for key in ("mean_travel_time", "mean_throughput", "mean_waiting_time"):
            assert key not in summary
        # Original 9 keys still present.
        assert set(summary) >= {
            "episodes",
            "mean_episode_reward",
            "mean_agent_reward",
            "mean_episode_length",
            "collision_rate",
            "arrival_rate",
            "truncation_rate",
            "mean_agent_collision_fraction",
            "mean_agent_arrival_fraction",
        }


class TestSortKeyUnchanged:
    def test_sort_key_ignores_new_metrics(self):
        """The new metrics must NOT enter the best-model sort key."""
        base = {
            "arrival_rate": 0.5,
            "collision_rate": 0.5,
            "mean_episode_reward": 2.0,
            "mean_episode_length": 8.0,
        }
        with_new = {
            **base,
            "mean_throughput": 99.0,
            "mean_waiting_time": 0.0,
            "mean_travel_time": 0.0,
        }
        assert _summary_sort_key(base) == _summary_sort_key(with_new)

    def test_sort_priority_unchanged(self):
        # arrival↑ -> collision↓ -> reward↑ -> length↓
        assert _summary_sort_key(
            {
                "arrival_rate": 1.0,
                "collision_rate": 0.9,
                "mean_episode_reward": 0.0,
                "mean_episode_length": 99.0,
            }
        ) > _summary_sort_key(
            {
                "arrival_rate": 0.5,
                "collision_rate": 0.0,
                "mean_episode_reward": 0.0,
                "mean_episode_length": 1.0,
            }
        )
        # New metrics must not make a worse-on-task summary "better".
        assert is_better_highway_summary(
            {
                "arrival_rate": 1.0,
                "collision_rate": 0.0,
                "mean_episode_reward": 1.0,
                "mean_episode_length": 1.0,
            },
            {
                "arrival_rate": 0.0,
                "collision_rate": 1.0,
                "mean_episode_reward": 100.0,
                "mean_episode_length": 1.0,
                "mean_throughput": 1000.0,
            },
        )
