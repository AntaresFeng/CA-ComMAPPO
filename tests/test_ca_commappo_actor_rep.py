"""Tests for CAComMAPPOActorRep (PI encoder + conflict-aware comm, forward shape).

These pin the actor representation contract for Phase 3:

* ``output_shapes`` is ``{'state': (H + comm_dim,)}``;
* ``forward`` on a param-sharing batch ``(n_env*n_agents, 105)`` returns a
  ``{'state': (batch, H + comm_dim)}`` dict with finite values;
* the per-agent scaled ``ttc``/``intent`` derived from the ego row are correct:
  an ego heading *toward* the conflict point C=(0,0) is flagged approaching
  (``intent=1``, ``0 <= ttc < 1``); an ego heading *away* is receding and
  receives the capped rationally scaled TTC.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ca_commappo.algorithms.ca_commappo_agents import CAComMAPPOActorRep
from ca_commappo.algorithms.conflict_aware_comm import TTC_MAX

VEHICLES_COUNT = 15
N_FEATURES = 7
VIEW_DIM = VEHICLES_COUNT * N_FEATURES  # 105


@pytest.fixture(autouse=True)
def _deterministic_torch():
    torch.manual_seed(0)
    np.random.seed(0)
    yield
    torch.manual_seed(0)


def _build_rep(n_agents: int = 2, hidden: int = 16) -> CAComMAPPOActorRep:
    return CAComMAPPOActorRep(
        input_shape=(VIEW_DIM,),
        hidden_sizes=[hidden, hidden],
        normalize=torch.nn.LayerNorm,
        initialize=torch.nn.init.orthogonal_,
        activation=torch.nn.ReLU,
        device="cpu",
        n_agents=n_agents,
        comm_dim=hidden,
        recede_penalty=10.0,
        ttc_max=TTC_MAX,
    )


def _ego_row(x, y, vx, vy, cos_h, sin_h) -> torch.Tensor:
    """One ego row [presence=1, x, y, vx, vy, cos_h, sin_h]."""
    return torch.tensor([1.0, x, y, vx, vy, cos_h, sin_h], dtype=torch.float32)


def _obs_for_agents(ego_rows: list[torch.Tensor]) -> torch.Tensor:
    """Build a (n_agents, 105) batch: ego at index 0, rest zeroed (absent)."""
    n = len(ego_rows)
    obs = torch.zeros(n, VEHICLES_COUNT, N_FEATURES)
    for i, ego in enumerate(ego_rows):
        obs[i, 0, :] = ego
    return obs.reshape(n, VIEW_DIM)


class TestOutputContract:
    def test_output_shapes(self):
        rep = _build_rep(n_agents=2, hidden=16)
        assert rep.output_shapes == {"state": (16 + 16,)}

    def test_forward_shape_and_finite(self):
        rep = _build_rep(n_agents=2, hidden=16)
        obs = _obs_for_agents(
            [_ego_row(1, 0, 0, 0, -1, 0), _ego_row(-1, 0, 0, 0, 1, 0)]
        )
        out = rep(obs.numpy())
        assert set(out.keys()) == {"state"}
        assert out["state"].shape == (2, 32)
        assert torch.isfinite(out["state"]).all()

    def test_inactive_agent_cannot_send_or_receive_messages(self):
        rep = _build_rep(n_agents=3, hidden=16)
        obs = _obs_for_agents(
            [
                _ego_row(1.0, 0.0, -1.0, 0.0, -1.0, 0.0),
                _ego_row(0.0, 1.0, 0.0, -1.0, 0.0, -1.0),
                _ego_row(-1.0, 0.0, 1.0, 0.0, 1.0, 0.0),
            ]
        )
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        baseline = rep(obs, agent_mask=mask)["state"].reshape(1, 3, 32)

        changed = obs.clone()
        changed[2] = torch.randn_like(changed[2]) * 100.0
        mutated = rep(changed, agent_mask=mask)["state"].reshape(1, 3, 32)

        assert torch.allclose(baseline[:, :2], mutated[:, :2])
        assert torch.equal(baseline[:, 2, 16:], torch.zeros(1, 16))

    def test_runtime_mask_context_is_restored_after_exit(self):
        rep = _build_rep(n_agents=2, hidden=16)
        mask = torch.tensor([[1.0, 0.0]])
        with rep.communication_mask(mask):
            assert rep._runtime_agent_mask is mask
        assert rep._runtime_agent_mask is None

    def test_trace_is_read_only_and_exports_masked_attention_contract(self):
        rep = _build_rep(n_agents=3, hidden=16)
        obs = _obs_for_agents(
            [
                _ego_row(1.0, 0.0, -1.0, 0.0, -1.0, 0.0),
                _ego_row(0.0, 1.0, 0.0, -1.0, 0.0, -1.0),
                _ego_row(-1.0, 0.0, 1.0, 0.0, 1.0, 0.0),
            ]
        )
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        baseline = rep(obs, agent_mask=mask)["state"]
        traces = []

        with rep.communication_trace(traces.append):
            traced = rep(obs, agent_mask=mask)["state"]

        assert torch.equal(baseline, traced)
        assert len(traces) == 1
        trace = traces[0]
        assert torch.equal(trace["active_mask"], mask)
        assert trace["scaled_ttc"].shape == (1, 3)
        assert trace["intent"].shape == (1, 3)
        assert trace["attention"].shape == (1, 3, 3)
        assert torch.allclose(trace["attention"][0, :2].sum(dim=-1), torch.ones(2))
        assert torch.equal(trace["attention"][0, 2], torch.zeros(3))
        assert trace["effective_lambda"].item() == pytest.approx(1.0)
        assert trace["use_conflict_prior"] is True
        assert rep._runtime_trace_sink is None

    def test_trace_context_restores_after_sink_exception(self):
        rep = _build_rep(n_agents=2, hidden=16)
        obs = _obs_for_agents(
            [_ego_row(1, 0, -1, 0, -1, 0), _ego_row(-1, 0, 1, 0, 1, 0)]
        )

        def fail(_trace):
            raise RuntimeError("trace sink failed")

        with (
            pytest.raises(RuntimeError, match="trace sink failed"),
            rep.communication_trace(fail),
        ):
            rep(obs)
        assert rep._runtime_trace_sink is None


class TestTTCIntent:
    def test_approaching_ego_gets_finite_ttc_and_intent_one(self):
        rep = _build_rep(n_agents=1, hidden=16)
        # Agent at (1, 0) moving toward origin: heading = (-1, 0).
        ego = _ego_row(x=1.0, y=0.0, vx=-1.0, vy=0.0, cos_h=-1.0, sin_h=0.0)
        obs = _obs_for_agents([ego])
        rep(obs.numpy())  # run forward (exercises _ttc_intent internally)
        # Re-derive ttc/intent directly to assert semantics.
        x = rep.encoder(torch.as_tensor(obs.numpy()))  # ensures encoder path ok
        assert x is not None
        ttc, intent = rep._ttc_intent(ego.unsqueeze(0))
        assert intent.item() == 1.0
        assert ttc.item() == pytest.approx(0.5)

    def test_receding_ego_gets_ttc_max_and_intent_zero(self):
        rep = _build_rep(n_agents=1, hidden=16)
        # Agent at (1, 0) heading away from origin: heading = (1, 0).
        ego = _ego_row(x=1.0, y=0.0, vx=1.0, vy=0.0, cos_h=1.0, sin_h=0.0)
        ttc, intent = rep._ttc_intent(ego.unsqueeze(0))
        assert intent.item() == 0.0
        assert ttc.item() == pytest.approx(TTC_MAX / (TTC_MAX + 1.0))

    def test_ttc_scaling_is_bounded_and_monotonic(self):
        rep = _build_rep(n_agents=1, hidden=16)
        near = _ego_row(0.5, 0.0, -1.0, 0.0, -1.0, 0.0)
        far = _ego_row(2.0, 0.0, -1.0, 0.0, -1.0, 0.0)
        ttc_near, _ = rep._ttc_intent(near.unsqueeze(0))
        ttc_far, _ = rep._ttc_intent(far.unsqueeze(0))
        assert 0.0 <= ttc_near.item() < ttc_far.item() < 1.0

    def test_mixed_pair_comm_runs_without_error(self):
        rep = _build_rep(n_agents=2, hidden=16)
        approaching = _ego_row(1.0, 0.0, -1.0, 0.0, -1.0, 0.0)
        receding = _ego_row(1.0, 0.0, 1.0, 0.0, 1.0, 0.0)
        obs = _obs_for_agents([approaching, receding])
        out = rep(obs.numpy())
        assert out["state"].shape == (2, 32)
