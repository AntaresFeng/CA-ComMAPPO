"""Tests for ConflictAwareComm (the CA-ComMAPPO conflict-aware attention module).

These tests pin down the Phase 3 comm module contract:

* output shape is ``(n_env, n_agents, comm_dim)``;
* attention weights sum to 1 over receivers ``j`` and exclude self (``j == i``);
* dead agents (``alive_ally == 0``) are masked out;
* the conflict prior ``b_ij = -lambda * |ttc_i - ttc_j|`` is monotonically
  decreasing in ``|ttc_i - ttc_j|`` (closer arrival times -> higher attention);
* a pair of agents both approaching the conflict point (small, similar ttc)
  gets higher mutual attention than a pair of receding agents (ttc = TTC_MAX).
"""

from __future__ import annotations

import math

import pytest
import torch

from ca_commappo.algorithms.conflict_aware_comm import TTC_MAX, ConflictAwareComm

MSG_DIM = 6  # e.g. H=4 hidden + ttc(1) + intent(1)
TTC_SLOT = 4
INTENT_SLOT = 5
COMM_DIM = 8
N_AGENTS = 3
TTC_SCALED_MAX = TTC_MAX / (TTC_MAX + 1.0)


@pytest.fixture(autouse=True)
def _deterministic_torch():
    torch.manual_seed(0)
    yield
    torch.manual_seed(0)


def _build_comm() -> ConflictAwareComm:
    return ConflictAwareComm(
        msg_dim=MSG_DIM,
        comm_dim=COMM_DIM,
        n_agents=N_AGENTS,
        ttc_slot=TTC_SLOT,
        intent_slot=INTENT_SLOT,
        recede_penalty=10.0,
        device="cpu",
    )


def _messages(n_env, n_agents, *, ttc_values, seed=1):
    """Build m_send with random z/head and a chosen ttc per agent.

    ttc_values contains the bounded values carried by the message. Intent is
    zero only for the capped receding value.
    """
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(n_env, n_agents, MSG_DIM, generator=g)
    ttc = torch.as_tensor(ttc_values, dtype=torch.float32)
    m[..., TTC_SLOT] = ttc
    m[..., INTENT_SLOT] = (ttc < TTC_SCALED_MAX).float()
    return m


def _set_effective_lambda(comm: ConflictAwareComm, value: float) -> None:
    raw = math.log(math.expm1(value))
    with torch.no_grad():
        comm.raw_lambda.fill_(raw)


class TestOutputContract:
    def test_forward_shape(self):
        comm = _build_comm()
        m = _messages(2, N_AGENTS, ttc_values=[[1.0, 2.0, 3.0], [1.0, 1.0, 5.0]])
        alive = torch.ones(2, N_AGENTS)
        c = comm(m, alive)
        assert c.shape == (2, N_AGENTS, COMM_DIM)
        assert torch.isfinite(c).all()

    def test_lambda_initial_value_is_preserved_by_positive_parameterization(self):
        comm = _build_comm()
        assert comm.lambda_value.item() == pytest.approx(1.0)

    def test_lambda_stays_positive_and_has_finite_gradient(self):
        comm = _build_comm()
        with torch.no_grad():
            comm.raw_lambda.fill_(-100.0)
        assert 0.0 <= comm.lambda_value.item() < 1e-20

        with torch.no_grad():
            comm.raw_lambda.fill_(0.0)
            comm.W_q.weight.zero_()
            comm.W_q.bias.zero_()
            comm.W_k.weight.zero_()
            comm.W_k.bias.zero_()
        m = _messages(1, N_AGENTS, ttc_values=[[0.0, 0.2, 0.8]])
        loss = comm.attention(m, torch.ones(1, N_AGENTS))[0, 0, 1]
        loss.backward()
        assert comm.raw_lambda.grad is not None
        assert torch.isfinite(comm.raw_lambda.grad)
        assert comm.raw_lambda.grad.abs().item() > 0.0


class TestAttentionNormalization:
    def test_weights_sum_to_one_over_receivers(self):
        comm = _build_comm()
        m = _messages(1, N_AGENTS, ttc_values=[[0.1, 0.2, 0.3]])
        alive = torch.ones(1, N_AGENTS)
        alpha = comm.attention(m, alive)
        assert alpha.shape == (1, N_AGENTS, N_AGENTS)
        # each receiver i sums to 1 over j
        assert torch.allclose(alpha.sum(dim=-1), torch.ones(1, N_AGENTS), atol=1e-5)

    def test_self_is_excluded(self):
        comm = _build_comm()
        m = _messages(1, N_AGENTS, ttc_values=[[0.1, 0.2, 0.3]])
        alive = torch.ones(1, N_AGENTS)
        alpha = comm.attention(m, alive)
        # diagonal (i == j) must be exactly zero
        diag = torch.diagonal(alpha, dim1=-2, dim2=-1)
        assert torch.allclose(diag, torch.zeros_like(diag))

    def test_dead_agent_is_masked_out(self):
        comm = _build_comm()
        m = _messages(1, N_AGENTS, ttc_values=[[0.1, 0.2, 0.3]])
        alive = torch.ones(1, N_AGENTS)
        alive[0, 1] = 0.0  # agent 1 dead -> nobody attends to it
        alpha = comm.attention(m, alive)
        assert torch.allclose(alpha[:, :, 1], torch.zeros(1, N_AGENTS), atol=1e-6)
        assert torch.allclose(alpha[:, 1, :], torch.zeros(1, N_AGENTS), atol=1e-6)
        assert torch.allclose(comm(m, alive)[:, 1, :], torch.zeros(1, COMM_DIM))


class TestConflictPriorMonotonicity:
    def test_closer_ttc_gets_higher_attention(self):
        """With q.k zeroed, alpha_ij = softmax(-lambda*|ttc_i - ttc_j|)."""
        comm = _build_comm()
        # Zero the q/k projections so scores are driven only by b_ij.
        with torch.no_grad():
            comm.W_q.weight.zero_()
            comm.W_q.bias.zero_()
            comm.W_k.weight.zero_()
            comm.W_k.bias.zero_()

        # Agent 0 sees two partners: agent 1 (ttc close to 0) and agent 2 (ttc far).
        m = _messages(1, N_AGENTS, ttc_values=[[0.0, 0.2, 0.9]])
        alive = torch.ones(1, N_AGENTS)
        alpha = comm.attention(m, alive)[0]  # (n_agents, n_agents)

        a_01 = alpha[0, 1].item()
        a_02 = alpha[0, 2].item()
        assert a_01 > a_02

    def test_lambda_scales_bias(self):
        comm_small = _build_comm()
        comm_large = _build_comm()
        with torch.no_grad():
            for c in (comm_small, comm_large):
                c.W_q.weight.zero_()
                c.W_q.bias.zero_()
                c.W_k.weight.zero_()
                c.W_k.bias.zero_()
        _set_effective_lambda(comm_small, 0.1)
        _set_effective_lambda(comm_large, 10.0)

        m = _messages(1, N_AGENTS, ttc_values=[[0.0, 0.2, 0.9]])
        alive = torch.ones(1, N_AGENTS)
        a_small = comm_small.attention(m, alive)[0, 0, 1].item()
        a_large = comm_large.attention(m, alive)[0, 0, 1].item()
        # larger lambda -> sharper preference for close-ttc partner 1 over partner 2
        assert a_large > a_small


class TestConflictScenario:
    def test_approaching_pair_attends_more_than_receding_pair(self):
        """Two agents both approaching C (small similar ttc) vs both receding
        (ttc = TTC_MAX). The approaching pair should have higher mutual alpha."""
        comm = _build_comm()
        with torch.no_grad():
            comm.W_q.weight.zero_()
            comm.W_q.bias.zero_()
            comm.W_k.weight.zero_()
            comm.W_k.bias.zero_()

        # Approaching: both small, similar ttc.
        m_approach = _messages(1, N_AGENTS, ttc_values=[[0.5, 0.55, TTC_SCALED_MAX]])
        # Receding: both TTC_MAX.
        m_recede = _messages(
            1,
            N_AGENTS,
            ttc_values=[[TTC_SCALED_MAX, TTC_SCALED_MAX, TTC_SCALED_MAX]],
        )
        alive = torch.ones(1, N_AGENTS)

        alpha_approach = comm.attention(m_approach, alive)[0, 0, 1].item()
        alpha_recede = comm.attention(m_recede, alive)[0, 0, 1].item()
        assert alpha_approach > alpha_recede


class TestAblationNoConflictPrior:
    """Phase 4 ablation (c): attention comm but b_ij = 0 (no conflict prior).

    With W_q/W_k zeroed the scores are b_ij only, so disabling the prior makes
    attention uniform over valid receivers, while enabling it (with differing
    ttc) makes it non-uniform.
    """

    def _build(self, use_prior: bool) -> ConflictAwareComm:
        return ConflictAwareComm(
            msg_dim=MSG_DIM,
            comm_dim=COMM_DIM,
            n_agents=N_AGENTS,
            ttc_slot=TTC_SLOT,
            intent_slot=INTENT_SLOT,
            recede_penalty=10.0,
            use_conflict_prior=use_prior,
            device="cpu",
        )

    def _zero_qk(self, comm):
        with torch.no_grad():
            comm.W_q.weight.zero_()
            comm.W_q.bias.zero_()
            comm.W_k.weight.zero_()
            comm.W_k.bias.zero_()

    def test_no_prior_gives_uniform_attention_with_zeroed_qk(self):
        comm = self._build(use_prior=False)
        self._zero_qk(comm)
        m = _messages(1, N_AGENTS, ttc_values=[[0.1, 0.2, 0.3]])
        alive = torch.ones(1, N_AGENTS)
        alpha = comm.attention(m, alive)[0]  # (A, A)
        # Each receiver i has n_agents-1 valid partners (self excluded); with
        # b_ij=0 the softmax is uniform -> each partner gets 1/(A-1).
        for i in range(N_AGENTS):
            partners = [alpha[i, j].item() for j in range(N_AGENTS) if j != i]
            assert all(abs(p - partners[0]) < 1e-6 for p in partners)
            assert abs(partners[0] - 1.0 / (N_AGENTS - 1)) < 1e-6

    def test_prior_makes_attention_nonuniform_with_zeroed_qk(self):
        comm = self._build(use_prior=True)
        self._zero_qk(comm)
        m = _messages(1, N_AGENTS, ttc_values=[[0.1, 0.2, 0.9]])
        alive = torch.ones(1, N_AGENTS)
        alpha = comm.attention(m, alive)[0]
        # Agent 0's two partners (1, 2) have |ttc diff| 1 and 9 -> non-uniform.
        a01, a02 = alpha[0, 1].item(), alpha[0, 2].item()
        assert abs(a01 - a02) > 1e-4

    def test_no_prior_output_is_independent_of_lambda(self):
        comm = self._build(use_prior=False)
        m = _messages(1, N_AGENTS, ttc_values=[[0.1, 0.2, 0.9]])
        alive = torch.ones(1, N_AGENTS)
        _set_effective_lambda(comm, 0.1)
        low_lambda = comm(m, alive)
        _set_effective_lambda(comm, 10.0)
        high_lambda = comm(m, alive)
        assert torch.equal(low_lambda, high_lambda)
