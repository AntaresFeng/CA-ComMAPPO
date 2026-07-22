"""Runtime patch so CommNet/IC3Net ``action()`` works without ``info``.

CommNet and IC3Net ``action()`` build ``alive_ally`` from a per-env ``info``
list (``for data in info``) and raise ``TypeError`` when ``info is None``. The
project evaluators ``evaluate_highway_policy`` and ``_agent_action_for_single_env``
call ``agents.action(...)`` *without* ``info`` (they predate CommNet support),
so ``--mode test`` / ``--mode benchmark`` crash for these agents.

This patch wraps ``CommNet_Agents.action`` and ``IC3Net_Agents.action`` so that
when ``info is None`` they synthesise an all-ones ``info`` (every agent alive).
That is exactly what CommNet's own training rollout ``run_episodes`` does
(``info = [{'agent_mask': {k: True ...}} for ... in range(num_envs)]``), so eval
behaviour stays faithful to training.

The patch is idempotent and never edits the installed ``xuance`` package -- it
monkey-patches the class method at runtime, mirroring ``xuance_compat.py``.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

_PATCH_FLAG = "_ca_commappo_info_patch"


def _make_info_default_wrapper(
    original_action: Callable[..., Any],
) -> Callable[..., Any]:
    """Wrap ``original_action`` so a missing ``info`` defaults to all-alive."""

    @functools.wraps(original_action)
    def wrapper(
        self,
        obs_dict,
        state=None,
        avail_actions_dict=None,
        rnn_hidden_actor=None,
        rnn_hidden_critic=None,
        test_mode=False,
        info=None,
        **kwargs,
    ):
        if info is None:
            n_env = len(obs_dict)
            info = [
                {"agent_mask": {k: 1 for k in self.agent_keys}} for _ in range(n_env)
            ]
        return original_action(
            self,
            obs_dict,
            state=state,
            avail_actions_dict=avail_actions_dict,
            rnn_hidden_actor=rnn_hidden_actor,
            rnn_hidden_critic=rnn_hidden_critic,
            test_mode=test_mode,
            info=info,
            **kwargs,
        )

    setattr(wrapper, _PATCH_FLAG, True)
    return wrapper


def patch_comm_agents_info() -> None:
    """Patch CommNet/IC3Net ``action()`` to tolerate ``info=None``.

    Safe to call multiple times.
    """
    from xuance.torch.agents import CommNet_Agents, IC3Net_Agents

    if not getattr(CommNet_Agents.action, _PATCH_FLAG, False):
        CommNet_Agents.action = _make_info_default_wrapper(CommNet_Agents.action)
    # IC3Net defines its own action(); patch it independently (TarMAC, which
    # inherits IC3Net's action, is covered transitively but out of scope).
    if not getattr(IC3Net_Agents.action, _PATCH_FLAG, False):
        IC3Net_Agents.action = _make_info_default_wrapper(IC3Net_Agents.action)
