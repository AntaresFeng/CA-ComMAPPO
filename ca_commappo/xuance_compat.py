"""Compatibility shims for the local XuanCe version used by CA-ComMAPPO."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np


def patch_xuance_marl_buffer_aliases() -> None:
    """Patch XuanCe 1.4.3 MARL buffer field-name drift in this process.

    `BaseBuffer` stores the newer names `num_envs`, `per_env_buffer_size`,
    `observation_space`, and `action_space`, but `MARL_OnPolicyBuffer` still
    reads the older names `n_envs`, `n_size`, `obs_space`, and `act_space`.
    Adding aliases after `BaseBuffer.__init__` lets MAPPO build its on-policy
    buffer without editing installed files under `.venv`.
    """
    from xuance.common.memory_tools_marl import BaseBuffer

    if getattr(BaseBuffer, "_ca_commappo_alias_patch", False):
        return

    original_init = BaseBuffer.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Keep the shim narrow: only expose the legacy names used downstream.
        self.n_envs = self.num_envs
        self.n_size = self.per_env_buffer_size
        self.obs_space = self.observation_space
        self.act_space = self.action_space

    BaseBuffer.__init__ = patched_init
    BaseBuffer._ca_commappo_alias_patch = True


def patch_xuance_mappo_masked_statistics() -> None:
    """Exclude inactive agents from XuanCe MAPPO training statistics.

    XuanCe 1.4.3 masks inactive agents in the actor and critic losses, but its
    feedforward on-policy buffer normalizes advantages before applying that
    mask, and ``MAPPO_Learner.update`` updates ``ValueNorm`` with every return.
    Large placeholder values from inactive agents can therefore change the
    scale seen by active samples despite contributing no loss.

    The patch deliberately leaves recurrent sampling and learning untouched.
    It is safe to call repeatedly in the same process.
    """
    from xuance.common.memory_tools_marl import MARL_OnPolicyBuffer
    from xuance.torch.learners.multi_agent_rl.mappo_learner import MAPPO_Learner

    _patch_feedforward_advantage_normalization(MARL_OnPolicyBuffer)
    _patch_feedforward_mappo_value_norm(MAPPO_Learner)


def _patch_feedforward_advantage_normalization(buffer_class: type) -> None:
    if getattr(buffer_class, "_ca_commappo_masked_advnorm_patch", False):
        return

    original_sample = buffer_class.sample

    @wraps(original_sample)
    def patched_sample(self, indexes=None):
        samples = original_sample(self, indexes)
        if not self.use_advantage_norm:
            return samples

        for agent_key in self.agent_keys:
            advantages = self.data["advantages"][agent_key]
            masks = self.data["agent_mask"][agent_key]
            env_choices, step_choices = divmod(indexes, self.n_size)
            advantage_batch = advantages[env_choices, step_choices]
            active = masks[env_choices, step_choices].astype(bool, copy=False)
            normalized = np.zeros_like(advantage_batch, dtype=np.float32)
            if np.any(active):
                active_advantages = advantage_batch[active]
                normalized[active] = (
                    active_advantages - np.mean(active_advantages)
                ) / (np.std(active_advantages) + 1e-8)
            samples["advantages"][agent_key] = normalized
        return samples

    buffer_class.sample = patched_sample
    buffer_class._ca_commappo_masked_advnorm_patch = True


def _patch_feedforward_mappo_value_norm(learner_class: type) -> None:
    if getattr(learner_class, "_ca_commappo_masked_valuenorm_patch", False):
        return

    original_update = learner_class.update

    @wraps(original_update)
    def patched_update(self, sample):
        if not self.use_value_norm:
            return original_update(self, sample)

        masks_by_model = _feedforward_masks_by_model(self, sample)
        normalizers = self.value_normalizer
        original_normalizers = dict(normalizers)
        masked_normalizers = {
            key: _MaskedValueNormalizer(normalizer, masks_by_model[key])
            for key, normalizer in original_normalizers.items()
        }
        normalizers.update(masked_normalizers)
        try:
            return original_update(self, sample)
        finally:
            normalizers.update(original_normalizers)

    learner_class.update = patched_update
    learner_class._ca_commappo_masked_valuenorm_patch = True


def _feedforward_masks_by_model(learner: Any, sample: dict) -> dict[str, np.ndarray]:
    agent_masks = sample["agent_mask"]
    if learner.use_parameter_sharing:
        stacked = np.stack(
            [np.asarray(agent_masks[key]) for key in learner.agent_keys], axis=1
        )
        return {learner.model_keys[0]: stacked.reshape(-1).astype(bool)}

    masks_by_model = {}
    for index, model_key in enumerate(learner.model_keys):
        agent_key = model_key if model_key in agent_masks else learner.agent_keys[index]
        masks_by_model[model_key] = (
            np.asarray(agent_masks[agent_key]).reshape(-1).astype(bool)
        )
    return masks_by_model


class _MaskedValueNormalizer:
    """Delegate to a ValueNorm while filtering its update batch by a mask."""

    def __init__(self, normalizer: Any, active_mask: np.ndarray):
        self._normalizer = normalizer
        self._active_mask = active_mask

    def update(self, input_vector):
        import torch

        values = torch.as_tensor(input_vector)
        active = torch.as_tensor(
            self._active_mask, dtype=torch.bool, device=values.device
        )
        if values.shape[0] != active.numel():
            raise ValueError(
                "MAPPO ValueNorm target/mask size mismatch: "
                f"{values.shape[0]} targets for {active.numel()} masks."
            )
        if torch.any(active):
            self._normalizer.update(values[active])

    def __getattr__(self, name: str):
        return getattr(self._normalizer, name)
