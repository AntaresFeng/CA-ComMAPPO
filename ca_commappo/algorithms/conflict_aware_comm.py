"""Conflict-aware communication module for CA-ComMAPPO.

Implements the conflict-aware attention aggregation from the paper plan
(``docs/thesis-planning.md`` §3, frozen in ``docs/ca_commappo_design.md``):

    m_i = f_m(z_i, intent_i, ttc_i)          # agent i's message (built by the
                                              # actor representation; carries ttc
                                              # and intent in its tail)
    c_i = sum_{j != i} alpha_ij * W_v m_j     # conflict-aware aggregation
    alpha_ij = softmax_j( q_i^T k_j + b_ij )  # soft conflict-prior attention

with the conflict prior

    b_ij = -lambda * |ttc_i - ttc_j| - mu * (R_i + R_j),   R_i = 1 - intent_i

The first term makes two agents that reach the conflict point at similar times
attend to each other; the second (recency penalty) ensures pairs of receding
agents (both ``ttc == TTC_MAX``) do not get spuriously high attention just
because their ttcs match.

This is a drop-in replacement for the *aggregation* step of XuanCe's ``CommNet``
(mean -> conflict-aware attention). It operates on a batch reshaped to
``(n_env, n_agents, msg_dim)`` and returns ``c`` of shape
``(n_env, n_agents, comm_dim)``.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn

from xuance.torch import Module

# Cap for the TTC proxy of agents that are not approaching the conflict point
# (receding / stationary). Matches docs/ca_commappo_design.md.
TTC_MAX = 50.0


class ConflictAwareComm(Module):
    """Conflict-aware attention message aggregation.

    Parameters
    ----------
    msg_dim:        dimension of each agent's message ``m_i`` (must carry ``ttc``
                    at ``ttc_slot`` and ``intent`` at ``intent_slot``).
    comm_dim:       output dimension of ``c_i`` (the attention value dim).
    n_agents:       number of controlled agents (for the self-mask).
    ttc_slot:       index of the ``ttc`` field in ``m_i``.
    intent_slot:    index of the ``intent`` field in ``m_i`` (used for the recede
                    penalty ``R_i = 1 - intent_i``).
    recede_penalty: fixed hyperparameter ``mu`` penalising receding pairs.
    lam_init:       initial value of the learnable scalar ``lambda``.
    """

    def __init__(
        self,
        msg_dim: int,
        comm_dim: int,
        n_agents: int,
        ttc_slot: int,
        intent_slot: int,
        recede_penalty: float = 10.0,
        lam_init: float = 1.0,
        device: Optional[Union[str, int, torch.device]] = None,
    ):
        super().__init__()
        self.msg_dim = msg_dim
        self.comm_dim = comm_dim
        self.n_agents = n_agents
        self.ttc_slot = ttc_slot
        self.intent_slot = intent_slot
        self.recede_penalty = float(recede_penalty)
        self.device = device

        self.W_q = nn.Linear(msg_dim, comm_dim, device=device)
        self.W_k = nn.Linear(msg_dim, comm_dim, device=device)
        self.W_v = nn.Linear(msg_dim, comm_dim, device=device)
        self.lam = nn.Parameter(torch.tensor(float(lam_init), device=device))
        self.scale = float(comm_dim) ** 0.5

    @staticmethod
    def _alive_2d(alive_ally, n_env: int, n_agents: int, device) -> torch.Tensor:
        a = torch.as_tensor(alive_ally, dtype=torch.float32, device=device)
        if a.dim() == 3:
            a = a.reshape(n_env, n_agents)
        return a

    def _scores(self, m_send: torch.Tensor, alive_ally):
        """Return (scores, v, valid) where scores[i,j] = q_i.k_j + b_ij, masked."""
        n_env, n_agents, _ = m_send.shape
        device = m_send.device

        ttc = m_send[..., self.ttc_slot]  # (E, A)
        intent = m_send[..., self.intent_slot]  # (E, A)
        recede = 1.0 - intent  # R_i

        q = self.W_q(m_send)  # (E, A, D)
        k = self.W_k(m_send)
        v = self.W_v(m_send)

        dot = torch.einsum("eid,ejd->eij", q, k) / self.scale  # (E, A, A)
        ttc_bias = -self.lam * torch.abs(ttc[:, :, None] - ttc[:, None, :])
        recede_bias = -self.recede_penalty * (recede[:, :, None] + recede[:, None, :])
        scores = dot + ttc_bias + recede_bias

        alive = self._alive_2d(alive_ally, n_env, n_agents, device)  # (E, A)
        alive_j = alive[:, None, :].expand(n_env, n_agents, n_agents).bool()
        eye = torch.eye(n_agents, device=device, dtype=torch.bool)
        self_mask = eye[None].expand(n_env, n_agents, n_agents)
        valid = (~self_mask) & alive_j  # j != i and alive
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~valid, neg_inf)
        return scores, v, valid

    def attention(self, m_send: torch.Tensor, alive_ally) -> torch.Tensor:
        """Return the attention weights alpha (E, A, A) for inspection/export."""
        scores, _, valid = self._scores(m_send, alive_ally)
        any_valid = valid.any(dim=-1, keepdim=True)  # (E, A, 1)
        safe = torch.where(any_valid, scores, torch.zeros_like(scores))
        alpha = torch.softmax(safe, dim=-1)
        return torch.where(any_valid, alpha, torch.zeros_like(alpha))

    def forward(self, m_send: torch.Tensor, alive_ally) -> torch.Tensor:
        """Aggregate messages -> c (E, A, comm_dim)."""
        scores, v, valid = self._scores(m_send, alive_ally)
        any_valid = valid.any(dim=-1, keepdim=True)
        safe = torch.where(any_valid, scores, torch.zeros_like(scores))
        alpha = torch.softmax(safe, dim=-1)
        alpha = torch.where(any_valid, alpha, torch.zeros_like(alpha))
        c = torch.einsum("eij,ejd->eid", alpha, v)  # (E, A, D)
        return torch.where(any_valid, c, torch.zeros_like(c))
