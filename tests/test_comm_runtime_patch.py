"""Tests for the CommNet/IC3Net ``action()`` info-default runtime patch.

CommNet and IC3Net ``action()`` build ``alive_ally`` from ``info`` and raise
``TypeError`` when ``info is None``. The project evaluators
(``evaluate_highway_policy`` / ``_agent_action_for_single_env``) call
``agents.action(...)`` without ``info``, so test/benchmark crash without this
patch. The patch synthesises an all-ones ``info`` (every agent alive), which is
exactly what CommNet's own training rollout (``run_episodes``) uses.
"""

from __future__ import annotations


class _FakeComm:
    """Minimal stand-in for a CommNet/IC3Net agents object."""

    agent_keys = ["agent_0", "agent_1"]

    def __init__(self) -> None:
        self.last_info = "untouched"

    def action(
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
        self.last_info = info
        return {
            "rnn_hidden_actor": None,
            "rnn_hidden_critic": None,
            "actions": [],
            "log_pi": {},
            "values": {},
        }


def test_info_default_wrapper_injects_all_ones_when_info_is_none():
    from ca_commappo.algorithms.comm_runtime_patch import _make_info_default_wrapper

    _FakeComm.action = _make_info_default_wrapper(_FakeComm.action)
    fake = _FakeComm()

    # Two parallel envs, no info supplied -> must be synthesised.
    fake.action(obs_dict=[{}, {}], test_mode=True)

    expected = [{"agent_mask": {"agent_0": 1, "agent_1": 1}} for _ in range(2)]
    assert fake.last_info == expected


def test_info_default_wrapper_passes_through_when_info_is_supplied():
    from ca_commappo.algorithms.comm_runtime_patch import _make_info_default_wrapper

    class _Fake:
        agent_keys = ["agent_0", "agent_1"]

        def __init__(self) -> None:
            self.last_info = None

        def action(
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
            self.last_info = info
            return {
                "rnn_hidden_actor": None,
                "rnn_hidden_critic": None,
                "actions": [],
                "log_pi": {},
                "values": {},
            }

    _Fake.action = _make_info_default_wrapper(_Fake.action)
    fake = _Fake()
    supplied = [{"agent_mask": {"agent_0": 1, "agent_1": 0}}]

    fake.action(obs_dict=[{}], info=supplied, test_mode=True)

    assert fake.last_info is supplied


def test_patch_targets_both_classes_and_is_idempotent():
    from ca_commappo.algorithms.comm_runtime_patch import patch_comm_agents_info
    from xuance.torch.agents import CommNet_Agents, IC3Net_Agents

    patch_comm_agents_info()
    assert getattr(CommNet_Agents.action, "_ca_commappo_info_patch", False) is True
    assert getattr(IC3Net_Agents.action, "_ca_commappo_info_patch", False) is True

    # Second call must be a no-op (same wrapper object retained).
    before_commnet = CommNet_Agents.action
    before_ic3net = IC3Net_Agents.action
    patch_comm_agents_info()
    assert CommNet_Agents.action is before_commnet
    assert IC3Net_Agents.action is before_ic3net
