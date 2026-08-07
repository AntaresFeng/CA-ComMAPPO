"""Runtime-only optimizer diagnostics for XuanCe's feedforward MAPPO.

XuanCe 1.4.3 returns only the final minibatch's ``update_info`` from
``train_epochs``.  That makes apparently smooth training logs depend on one
random minibatch and hides PPO failure modes such as a growing KL divergence
or a high clipping fraction.  This module patches the relevant MAPPO methods
*in memory*; it never edits the installed XuanCe package.

Call :func:`patch_mappo_diagnostics` once before constructing/running MAPPO
agents.  Calling it repeatedly is safe.  Feedforward MAPPO updates then expose
per-minibatch diagnostics under ``diagnostics/...`` and ``train_epochs``
returns rollout-wide means and maxima over all ``n_epochs * n_minibatches``
optimizer updates.  Existing numeric keys remain available and now contain
their rollout-wide mean; ``<key>_mean`` and ``<key>_max`` make the aggregation
explicit.  Recurrent MAPPO is deliberately delegated to XuanCe unchanged.
"""

from __future__ import annotations

import functools
import math
import numbers
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

_PATCH_FLAG = "_ca_commappo_mappo_diagnostics_patch"
_ORIGINAL_METHOD = "_ca_commappo_original_method"

DIAGNOSTIC_KEYS = (
    "diagnostics/approx_kl",
    "diagnostics/clip_fraction",
    "diagnostics/ratio_p05",
    "diagnostics/ratio_p95",
    "diagnostics/entropy",
    "diagnostics/actor_loss",
    "diagnostics/critic_loss",
    "diagnostics/gradient_norm",
    "diagnostics/active_mask_fraction",
)


def _scalar_float(value: Any) -> float | None:
    """Return a scalar as ``float``, or ``None`` for non-scalar log values."""

    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return float(value.detach().cpu().item())

    # Covers NumPy scalars and single-element arrays without treating general
    # arrays/histograms from callbacks as scalar metrics.
    size = getattr(value, "size", None)
    if size == 1:
        try:
            return float(value.item())
        except (AttributeError, TypeError, ValueError):
            return None
    return None


def _finite_or_zero(value: Any) -> tuple[float, int]:
    """Convert a scalar to a finite float and report a non-finite replacement."""

    scalar = _scalar_float(value)
    if scalar is None:
        raise TypeError(f"Expected a scalar metric, got {type(value).__name__}.")
    if math.isfinite(scalar):
        return scalar, 0
    return 0.0, 1


def aggregate_update_infos(
    update_infos: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate every scalar in a rollout's optimizer update dictionaries.

    For backward compatibility, an original scalar key (for example
    ``"gradient_norm"`` or ``"shared/actor_loss"``) maps to its arithmetic
    mean across all updates where it was reported.  Explicit ``_mean`` and
    ``_max`` keys are emitted alongside it.  The most recent value is retained
    for non-scalar metadata.  Non-finite scalars are replaced with ``0.0`` and
    counted, so diagnostics never inject NaN/Inf into a logger while still
    making the anomaly visible.
    """

    if not update_infos:
        return {}

    scalar_values: dict[str, list[float]] = defaultdict(list)
    result: dict[str, Any] = {}
    nonfinite_count = 0
    reported_nonfinite_count = 0

    for update_info in update_infos:
        reported_count = _scalar_float(
            update_info.get("diagnostics/nonfinite_scalar_count")
        )
        if reported_count is not None and math.isfinite(reported_count):
            reported_nonfinite_count += max(0, int(reported_count))
        for key, value in update_info.items():
            scalar = _scalar_float(value)
            if scalar is None:
                result[key] = value
                continue
            finite, replaced = _finite_or_zero(scalar)
            scalar_values[key].append(finite)
            nonfinite_count += replaced

    for key, values in scalar_values.items():
        mean_value = math.fsum(values) / len(values)
        max_value = max(values)
        # Keep the XuanCe/callback key compatible, but make its new semantics
        # explicit through the two suffixed aliases.
        result[key] = mean_value
        result[f"{key}_mean"] = mean_value
        result[f"{key}_max"] = max_value

    result["diagnostics/update_count"] = len(update_infos)
    result["diagnostics/nonfinite_scalar_count_total"] = (
        reported_nonfinite_count + nonfinite_count
    )
    return result


class _DiagnosticCallbackProxy:
    """Delegate a learner callback while observing agent-wise PPO tensors."""

    def __init__(self, callback: Any, clip_range: float) -> None:
        self._callback = callback
        self._clip_range = abs(float(clip_range))
        self._ratios: list[torch.Tensor] = []
        self._entropy_sum = 0.0
        self._entropy_count = 0
        self._active_by_key: dict[str, int] = defaultdict(int)
        self._active_count = 0
        self._mask_count = 0
        self._invalid_active_ratio_count = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._callback, name)

    def on_update_agent_wise(
        self, iterations: int, agent_key: str, **kwargs: Any
    ) -> dict:
        self._capture(agent_key, kwargs)
        return self._callback.on_update_agent_wise(iterations, agent_key, **kwargs)

    def _capture(self, agent_key: str, kwargs: Mapping[str, Any]) -> None:
        mask = kwargs.get("mask_values")
        ratio = kwargs.get("ratio")
        if not isinstance(mask, torch.Tensor):
            return

        mask_flat = mask.detach().reshape(-1)
        self._mask_count += mask_flat.numel()
        active_mask = torch.isfinite(mask_flat) & (mask_flat > 0)
        active_count = int(active_mask.sum().item())
        self._active_count += active_count
        self._active_by_key[str(agent_key)] += active_count

        if isinstance(ratio, torch.Tensor):
            ratio_flat = ratio.detach().reshape(-1)
            common_size = min(mask_flat.numel(), ratio_flat.numel())
            if common_size:
                ratio_flat = ratio_flat[:common_size]
                ratio_active = active_mask[:common_size]
                valid = ratio_active & torch.isfinite(ratio_flat) & (ratio_flat > 0)
                valid_ratios = ratio_flat[valid].to(dtype=torch.float64, device="cpu")
                if valid_ratios.numel():
                    self._ratios.append(valid_ratios)
                self._invalid_active_ratio_count += int(
                    (ratio_active.sum() - valid.sum()).item()
                )

        entropy = kwargs.get("entropy")
        if isinstance(entropy, torch.Tensor):
            entropy_flat = entropy.detach().reshape(-1)
            common_size = min(mask_flat.numel(), entropy_flat.numel())
            if common_size:
                entropy_active = active_mask[:common_size]
                entropy_values = entropy_flat[:common_size]
                valid_entropy = entropy_active & torch.isfinite(entropy_values)
                if valid_entropy.any():
                    selected = entropy_values[valid_entropy].to(
                        dtype=torch.float64, device="cpu"
                    )
                    self._entropy_sum += float(selected.sum().item())
                    self._entropy_count += selected.numel()

    def _weighted_loss(self, info: Mapping[str, Any], suffix: str) -> float:
        weighted_sum = 0.0
        total_weight = 0
        for agent_key, active_count in self._active_by_key.items():
            scalar = _scalar_float(info.get(f"{agent_key}/{suffix}"))
            if scalar is None or not math.isfinite(scalar) or active_count <= 0:
                continue
            weighted_sum += scalar * active_count
            total_weight += active_count
        return weighted_sum / total_weight if total_weight else 0.0

    def diagnostics(self, info: Mapping[str, Any]) -> dict[str, float]:
        if self._ratios:
            ratios = torch.cat(self._ratios)
            log_ratio = torch.log(ratios)
            approx_kl = float(((ratios - 1.0) - log_ratio).mean().item())
            clipped = torch.abs(ratios - 1.0) > self._clip_range
            clip_fraction = float(clipped.to(torch.float64).mean().item())
            ratio_p05 = float(torch.quantile(ratios, 0.05).item())
            ratio_p95 = float(torch.quantile(ratios, 0.95).item())
        else:
            approx_kl = clip_fraction = 0.0
            ratio_p05 = ratio_p95 = 0.0

        active_fraction = (
            self._active_count / self._mask_count if self._mask_count else 0.0
        )
        entropy = (
            self._entropy_sum / self._entropy_count if self._entropy_count else 0.0
        )
        invalid_ratio_fraction = (
            self._invalid_active_ratio_count / self._active_count
            if self._active_count
            else 0.0
        )

        return {
            "diagnostics/approx_kl": approx_kl,
            "diagnostics/clip_fraction": clip_fraction,
            "diagnostics/ratio_p05": ratio_p05,
            "diagnostics/ratio_p95": ratio_p95,
            "diagnostics/entropy": entropy,
            "diagnostics/actor_loss": self._weighted_loss(info, "actor_loss"),
            "diagnostics/critic_loss": self._weighted_loss(info, "critic_loss"),
            "diagnostics/active_mask_fraction": active_fraction,
            "diagnostics/invalid_active_ratio_fraction": invalid_ratio_fraction,
        }


def _gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """Compute a finite total L2 gradient norm without mutating gradients."""

    squared_norm = 0.0
    found_gradient = False
    for parameter in parameters:
        gradient = getattr(parameter, "grad", None)
        if gradient is None:
            continue
        found_gradient = True
        gradient = gradient.detach()
        if not torch.isfinite(gradient).all():
            return 0.0
        norm = float(torch.linalg.vector_norm(gradient).item())
        squared_norm += norm * norm
    return math.sqrt(squared_norm) if found_gradient else 0.0


def _sanitize_scalar_info(info: dict[str, Any]) -> int:
    """Replace scalar NaN/Inf in-place and return the replacement count."""

    replaced = 0
    for key, value in tuple(info.items()):
        scalar = _scalar_float(value)
        if scalar is None:
            continue
        finite, was_replaced = _finite_or_zero(scalar)
        info[key] = finite
        replaced += was_replaced
    return replaced


def _make_learner_update_wrapper(
    original_update: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """Wrap ``MAPPO_Learner.update`` with active-mask-aware diagnostics."""

    @functools.wraps(original_update)
    def wrapper(self: Any, sample: Any) -> dict[str, Any]:
        original_callback = self.callback
        callback_proxy = _DiagnosticCallbackProxy(
            callback=original_callback,
            clip_range=getattr(self, "clip_range", 0.0),
        )
        self.callback = callback_proxy
        try:
            info = original_update(self, sample)
        finally:
            self.callback = original_callback

        if not isinstance(info, dict):
            raise TypeError(
                "MAPPO_Learner.update must return a dict for diagnostics aggregation."
            )

        info.update(callback_proxy.diagnostics(info))
        gradient_norm = _scalar_float(info.get("gradient_norm"))
        if gradient_norm is None or not math.isfinite(gradient_norm):
            parameters = getattr(getattr(self, "policy", None), "parameters_model", ())
            gradient_norm = _gradient_norm(parameters)
        info["diagnostics/gradient_norm"] = gradient_norm
        replaced = _sanitize_scalar_info(info)
        info["diagnostics/nonfinite_scalar_count"] = replaced
        return info

    setattr(wrapper, _PATCH_FLAG, True)
    setattr(wrapper, _ORIGINAL_METHOD, original_update)
    return wrapper


def _make_train_epochs_wrapper(
    original_train_epochs: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """Aggregate all feedforward optimizer updates in one rollout."""

    @functools.wraps(original_train_epochs)
    def wrapper(self: Any, n_epochs: int = 1) -> dict[str, Any]:
        # Ratio/mask shapes differ for recurrent batches.  Until recurrent
        # behavior has dedicated coverage, retain XuanCe's exact implementation.
        if getattr(self, "use_rnn", False):
            return original_train_epochs(self, n_epochs=n_epochs)

        info_train: dict[str, Any] = {}
        if self.memory.full:
            update_infos: list[Mapping[str, Any]] = []
            indexes = np.arange(self.buffer_size)
            for _ in range(n_epochs):
                np.random.shuffle(indexes)
                for start in range(0, self.buffer_size, self.batch_size):
                    end = start + self.batch_size
                    sample_idx = indexes[start:end]
                    sample = self.memory.sample(sample_idx)
                    update_info = self.learner.update(sample)
                    if not isinstance(update_info, Mapping):
                        raise TypeError(
                            "MAPPO learner update must return a mapping for "
                            "diagnostics aggregation."
                        )
                    update_infos.append(update_info)

            info_train = aggregate_update_infos(update_infos)
            self.callback.on_train_epochs_end(
                self.current_step,
                policy=self.policy,
                memory=self.memory,
                current_episode=self.current_episode,
                n_epochs=n_epochs,
                buffer_size=self.buffer_size,
                update_info=info_train,
            )
            self.memory.clear()
        return info_train

    setattr(wrapper, _PATCH_FLAG, True)
    setattr(wrapper, _ORIGINAL_METHOD, original_train_epochs)
    return wrapper


def patch_mappo_diagnostics() -> None:
    """Install the idempotent runtime patch for feedforward torch MAPPO."""

    from xuance.torch.agents.multi_agent_rl.mappo_agents import MAPPO_Agents
    from xuance.torch.learners.multi_agent_rl.mappo_learner import MAPPO_Learner

    if not getattr(MAPPO_Learner.update, _PATCH_FLAG, False):
        MAPPO_Learner.update = _make_learner_update_wrapper(MAPPO_Learner.update)
    if not getattr(MAPPO_Agents.train_epochs, _PATCH_FLAG, False):
        MAPPO_Agents.train_epochs = _make_train_epochs_wrapper(
            MAPPO_Agents.train_epochs
        )


__all__ = [
    "DIAGNOSTIC_KEYS",
    "aggregate_update_infos",
    "patch_mappo_diagnostics",
]
