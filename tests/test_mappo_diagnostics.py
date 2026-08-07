"""Tests for the runtime-only XuanCe MAPPO optimizer diagnostics patch."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from ca_commappo.algorithms.mappo_diagnostics import (
    DIAGNOSTIC_KEYS,
    _make_learner_update_wrapper,
    _make_train_epochs_wrapper,
    aggregate_update_infos,
    patch_mappo_diagnostics,
)


def test_aggregate_update_infos_preserves_keys_and_emits_mean_max():
    first = {
        "loss": 1.0,
        "shared/actor_loss": -2.0,
        "metadata": "first",
        "bad_scalar": float("nan"),
    }
    second = {
        "loss": 3.0,
        "shared/actor_loss": 2.0,
        "metadata": "last",
        "bad_scalar": float("inf"),
    }
    for index, key in enumerate(DIAGNOSTIC_KEYS, start=1):
        first[key] = float(index)
        second[key] = float(index + 2)

    result = aggregate_update_infos([first, second])

    # Original scalar keys remain logger-compatible, but now mean the entire
    # rollout instead of the final random minibatch.
    assert result["loss"] == pytest.approx(2.0)
    assert result["loss_mean"] == pytest.approx(2.0)
    assert result["loss_max"] == pytest.approx(3.0)
    assert result["shared/actor_loss"] == pytest.approx(0.0)
    assert result["metadata"] == "last"

    for index, key in enumerate(DIAGNOSTIC_KEYS, start=1):
        assert result[key] == pytest.approx(index + 1.0)
        assert result[f"{key}_mean"] == pytest.approx(index + 1.0)
        assert result[f"{key}_max"] == pytest.approx(index + 2.0)

    assert result["bad_scalar"] == 0.0
    assert result["diagnostics/update_count"] == 2
    assert result["diagnostics/nonfinite_scalar_count_total"] == 2
    assert all(
        math.isfinite(value)
        for value in result.values()
        if isinstance(value, float)
    )


def test_aggregate_update_infos_sums_per_minibatch_sanitization_counts():
    result = aggregate_update_infos(
        [
            {"loss": 1.0, "diagnostics/nonfinite_scalar_count": 2},
            {"loss": 2.0, "diagnostics/nonfinite_scalar_count": 1},
        ]
    )

    assert result["diagnostics/nonfinite_scalar_count_total"] == 3


class _FakeMemory:
    full = True

    def __init__(self) -> None:
        self.samples: list[tuple[int, ...]] = []
        self.clear_calls = 0

    def sample(self, indexes) -> dict[str, tuple[int, ...]]:
        sample_indexes = tuple(int(index) for index in indexes)
        self.samples.append(sample_indexes)
        return {"indexes": sample_indexes}

    def clear(self) -> None:
        self.clear_calls += 1


class _FakeLearner:
    def __init__(self) -> None:
        self.update_calls = 0

    def update(self, _sample) -> dict[str, object]:
        self.update_calls += 1
        value = float(self.update_calls)
        info: dict[str, object] = {
            "loss": value,
            "shared/actor_loss": -value,
            "tag": f"update-{self.update_calls}",
        }
        for key in DIAGNOSTIC_KEYS:
            info[key] = value
        return info


class _FakeTrainCallback:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, object]]] = []

    def on_train_epochs_end(self, current_step: int, **kwargs) -> None:
        # The XuanCe callback must see the same aggregated dictionary returned
        # to the agent logger, not the last minibatch.
        self.calls.append((current_step, kwargs))


def test_train_epochs_wrapper_aggregates_every_feedforward_minibatch():
    original_calls = []

    def original_train_epochs(_self, n_epochs=1):
        original_calls.append(n_epochs)
        return {"delegated": True}

    memory = _FakeMemory()
    learner = _FakeLearner()
    callback = _FakeTrainCallback()
    agents = SimpleNamespace(
        use_rnn=False,
        memory=memory,
        buffer_size=6,
        batch_size=2,
        learner=learner,
        callback=callback,
        current_step=192,
        current_episode=12,
        policy=object(),
    )

    wrapped = _make_train_epochs_wrapper(original_train_epochs)
    result = wrapped(agents, n_epochs=2)

    assert original_calls == []
    assert learner.update_calls == 6  # 2 epochs * 3 minibatches
    assert len(memory.samples) == 6
    assert memory.clear_calls == 1
    assert result["diagnostics/update_count"] == 6
    assert result["loss"] == pytest.approx(3.5)
    assert result["loss_mean"] == pytest.approx(3.5)
    assert result["loss_max"] == pytest.approx(6.0)
    assert result["tag"] == "update-6"
    for key in DIAGNOSTIC_KEYS:
        assert result[f"{key}_mean"] == pytest.approx(3.5)
        assert result[f"{key}_max"] == pytest.approx(6.0)

    assert len(callback.calls) == 1
    current_step, callback_kwargs = callback.calls[0]
    assert current_step == 192
    assert callback_kwargs["update_info"] is result
    assert callback_kwargs["n_epochs"] == 2
    assert callback_kwargs["buffer_size"] == 6


def test_train_epochs_wrapper_leaves_recurrent_path_unchanged():
    delegated = []

    def original_train_epochs(_self, n_epochs=1):
        delegated.append(n_epochs)
        return {"original": 7.0}

    agents = SimpleNamespace(use_rnn=True)
    wrapped = _make_train_epochs_wrapper(original_train_epochs)

    assert wrapped(agents, n_epochs=4) == {"original": 7.0}
    assert delegated == [4]


class _FakeLearnerCallback:
    def __init__(self) -> None:
        self.agent_calls = 0

    def on_update_agent_wise(self, _iterations, _agent_key, **_kwargs):
        self.agent_calls += 1
        return {"callback/value": 8.0}


def test_learner_wrapper_computes_active_mask_ppo_diagnostics_and_restores_callback():
    callback = _FakeLearnerCallback()
    learner = SimpleNamespace(
        callback=callback,
        clip_range=0.2,
        policy=SimpleNamespace(parameters_model=[]),
    )

    def original_update(self, _sample):
        callback_info = self.callback.on_update_agent_wise(
            1,
            "shared",
            mask_values=torch.tensor([1.0, 1.0, 0.0, 1.0]),
            ratio=torch.tensor([1.0, 1.5, 0.5, float("inf")]),
            # XuanCe passes entropy after applying the active mask.
            entropy=torch.tensor([0.2, 0.4, 0.0, float("nan")]),
        )
        return {
            "shared/actor_loss": -2.0,
            "shared/critic_loss": 4.0,
            "shared/entropy": 0.3,
            "gradient_norm": 3.0,
            "bad/value": float("nan"),
            **callback_info,
        }

    wrapped = _make_learner_update_wrapper(original_update)
    result = wrapped(learner, sample={})

    assert learner.callback is callback
    assert callback.agent_calls == 1
    expected_kl = ((1.0 - 1.0) - math.log(1.0) + (1.5 - 1.0) - math.log(1.5)) / 2
    assert result["diagnostics/approx_kl"] == pytest.approx(expected_kl)
    assert result["diagnostics/clip_fraction"] == pytest.approx(0.5)
    assert result["diagnostics/ratio_p05"] == pytest.approx(1.025)
    assert result["diagnostics/ratio_p95"] == pytest.approx(1.475)
    assert result["diagnostics/entropy"] == pytest.approx(0.3)
    assert result["diagnostics/actor_loss"] == pytest.approx(-2.0)
    assert result["diagnostics/critic_loss"] == pytest.approx(4.0)
    assert result["diagnostics/gradient_norm"] == pytest.approx(3.0)
    assert result["diagnostics/active_mask_fraction"] == pytest.approx(0.75)
    assert result["diagnostics/invalid_active_ratio_fraction"] == pytest.approx(1 / 3)
    assert result["bad/value"] == 0.0
    assert result["diagnostics/nonfinite_scalar_count"] == 1
    assert all(
        math.isfinite(value)
        for value in result.values()
        if isinstance(value, float)
    )


def test_learner_wrapper_computes_gradient_norm_when_xuance_does_not_report_it():
    callback = _FakeLearnerCallback()
    parameter = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    parameter.grad = torch.tensor([3.0, 4.0])
    learner = SimpleNamespace(
        callback=callback,
        clip_range=0.2,
        policy=SimpleNamespace(parameters_model=[parameter]),
    )

    def original_update(self, _sample):
        self.callback.on_update_agent_wise(
            1,
            "shared",
            mask_values=torch.tensor([1.0]),
            ratio=torch.tensor([1.0]),
            entropy=torch.tensor([0.5]),
        )
        return {
            "shared/actor_loss": 1.0,
            "shared/critic_loss": 2.0,
        }

    result = _make_learner_update_wrapper(original_update)(learner, sample={})

    assert "gradient_norm" not in result
    assert result["diagnostics/gradient_norm"] == pytest.approx(5.0)


def test_patch_mappo_diagnostics_targets_mappo_and_is_idempotent():
    from xuance.torch.agents.multi_agent_rl.mappo_agents import MAPPO_Agents
    from xuance.torch.learners.multi_agent_rl.mappo_learner import MAPPO_Learner

    patch_mappo_diagnostics()
    learner_update = MAPPO_Learner.update
    train_epochs = MAPPO_Agents.train_epochs

    assert getattr(learner_update, "_ca_commappo_mappo_diagnostics_patch", False)
    assert getattr(train_epochs, "_ca_commappo_mappo_diagnostics_patch", False)

    patch_mappo_diagnostics()
    assert MAPPO_Learner.update is learner_update
    assert MAPPO_Agents.train_epochs is train_epochs
