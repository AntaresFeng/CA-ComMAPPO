from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch
from xuance.torch.learners.multi_agent_rl.mappo_learner import MAPPO_Learner

from ca_commappo.algorithms.ca_commappo_agents import (
    CA_ComMAPPO_Agents,
    CA_ComMAPPO_Learner,
)


def _bare_agents() -> CA_ComMAPPO_Agents:
    agents = object.__new__(CA_ComMAPPO_Agents)
    agents.n_agents = 2
    agents.agent_keys = ["agent_0", "agent_1"]
    agents._rollout_agent_mask = None
    return agents


def test_rollout_mask_advances_after_partial_termination_and_resets():
    agents = _bare_agents()

    partial = agents._mask_from_infos(
        [
            {
                "agent_mask": {"agent_0": True, "agent_1": True},
                "next_agent_mask": {"agent_0": True, "agent_1": False},
            }
        ],
        1,
        use_next_mask=True,
    )
    reset = agents._mask_from_infos(
        [
            {
                "next_agent_mask": {"agent_0": False, "agent_1": False},
                "reset_obs": {"agent_0": 0, "agent_1": 0},
            }
        ],
        1,
    )

    assert np.array_equal(partial, [[1.0, 0.0]])
    assert np.array_equal(reset, [[1.0, 1.0]])


def test_explicit_info_mask_takes_precedence_over_rollout_fallback():
    agents = _bare_agents()
    agents._rollout_agent_mask = np.array([[0.0, 1.0]], dtype=np.float32)

    explicit = agents._mask_from_infos(
        [{"agent_mask": {"agent_0": True, "agent_1": False}}],
        1,
    )
    fallback = agents._mask_from_infos(None, 1)

    assert np.array_equal(explicit, [[1.0, 0.0]])
    assert np.array_equal(fallback, [[0.0, 1.0]])


def test_action_mask_prefers_current_transition_mask_over_next_mask():
    agents = _bare_agents()

    current = agents._mask_from_infos(
        [
            {
                "agent_mask": {"agent_0": True, "agent_1": False},
                "next_agent_mask": {"agent_0": False, "agent_1": True},
            }
        ],
        1,
    )

    assert np.array_equal(current, [[1.0, 0.0]])


class _MaskCapturingPolicy:
    def __init__(self):
        self.active_mask = None
        self.seen_masks = []
        comm = SimpleNamespace(
            lambda_value=torch.tensor(1.0),
            raw_lambda=SimpleNamespace(grad=torch.tensor(0.25)),
        )
        self.actor_representation = {"agent_0": SimpleNamespace(comm=comm)}

    @contextmanager
    def communication_mask(self, mask):
        previous = self.active_mask
        self.active_mask = mask
        try:
            yield
        finally:
            self.active_mask = previous


def test_learner_update_uses_minibatch_agent_mask(monkeypatch):
    policy = _MaskCapturingPolicy()
    learner = object.__new__(CA_ComMAPPO_Learner)
    learner.policy = policy
    learner.use_parameter_sharing = True
    learner.use_actions_mask = False
    learner.use_global_state = True
    learner.model_keys = ["agent_0"]
    learner.n_agents = 2
    learner.build_training_data = lambda **_kwargs: {
        "batch_size": 2,
        "agent_mask": {"agent_0": torch.tensor([1.0, 0.0, 0.0, 1.0])},
    }

    def fake_update(self, _sample):
        self.policy.seen_masks.append(self.policy.active_mask.clone())
        return {}

    monkeypatch.setattr(MAPPO_Learner, "update", fake_update)

    info = CA_ComMAPPO_Learner.update(learner, sample={})

    assert torch.equal(policy.seen_masks[0], torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert policy.active_mask is None
    assert info["diagnostics/conflict_lambda"] == 1.0
    assert info["diagnostics/conflict_lambda_raw_grad"] == 0.25
