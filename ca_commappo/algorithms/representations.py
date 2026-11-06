"""Permutation-invariant social-attention representation for highway-env.

The ego (controlled) vehicle is always at observation index 0 (highway-env's
``KinematicObservation`` shuffles only ``obs[1:]``), so it is used as the
fixed query. The remaining vehicles are pooled with a presence-masked
attention, making the encoding invariant to their order.

The same class serves both the actor and the MAPPO global-state critic:

* actor input  : one agent's flattened view, shape ``(105,)``  -> 1 view;
* critic input : the concatenated global state, shape
  ``(n_agents * 105,)`` -> ``n_agents`` views. Per-view embeddings are
  mean-pooled. Pooling is order-invariant, but the configured input shape and
  policy agent identities still constrain checkpoint compatibility across
  vehicle counts; this encoder alone does not establish size generalization.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from xuance.torch import Module
from xuance.torch.utils import ModuleType, mlp_block, nn, torch

# highway-env intersection observation layout (see configs/mappo/*.yaml):
#   features = [presence, x, y, vx, vy, cos_h, sin_h], vehicles_count = 15.
_VEHICLES_COUNT = 15
_N_FEATURES = 7
_VIEW_DIM = _VEHICLES_COUNT * _N_FEATURES  # 105
_EGO_INDEX = 0
_PRESENCE_INDEX = 0


class PermutationInvariantAttention(Module):
    """Permutation-invariant attention encoder, signature-aligned with ``Basic_MLP``.

    The constructor accepts the same kwargs XuanCe's ``_build_representation``
    passes (including the RNN/CNN kwargs it always forwards), swallowing the
    irrelevant ones via ``**kwargs`` so registration is a drop-in.
    """

    def __init__(
        self,
        input_shape: Sequence[int],
        hidden_sizes: Sequence[int],
        normalize: ModuleType | None = None,
        initialize: Callable[..., torch.Tensor] | None = None,
        activation: ModuleType | None = None,
        device: str | int | torch.device | None = None,
        **kwargs,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_sizes = list(hidden_sizes)
        self.normalize = normalize
        self.initialize = initialize
        self.activation = activation
        self.device = device

        input_dim = (
            int(np.prod(input_shape))
            if hasattr(input_shape, "__iter__")
            else int(input_shape)
        )
        if input_dim % _VIEW_DIM != 0:
            raise ValueError(
                f"PermutationInvariantAttention expects an input dimension that is a "
                f"multiple of {_VIEW_DIM} (vehicles_count * n_features), got {input_dim}."
            )
        self.n_views = input_dim // _VIEW_DIM
        self.output_shapes = {"state": (self.hidden_sizes[-1],)}

        d_phi = self.hidden_sizes[0]

        # phi: pairwise encoder over [ego, neighbour] -> d_phi.
        phi_layers, _ = mlp_block(
            2 * _N_FEATURES, d_phi, normalize, activation, initialize, device
        )
        self.phi = nn.Sequential(*phi_layers)

        # Attention score head (learned query) and value projection.
        self.score = nn.Linear(d_phi, 1, device=device)
        self.value = nn.Linear(d_phi, d_phi, device=device)
        if initialize is not None:
            initialize(self.score.weight)
            nn.init.constant_(self.score.bias, 0.0)
            initialize(self.value.weight)
            nn.init.constant_(self.value.bias, 0.0)

        # Output head: [ego_features, pooled_context] -> hidden_sizes[-1].
        out_layers: list[nn.Module] = []
        dim = _N_FEATURES + d_phi
        for h in self.hidden_sizes:
            block, _ = mlp_block(dim, h, normalize, activation, initialize, device)
            out_layers.extend(block)
            dim = h
        self.output = nn.Sequential(*out_layers)

    def _reshape_views(self, x: torch.Tensor) -> torch.Tensor:
        """(B, input_dim) or (input_dim,) -> (B, n_views, V, F)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch = x.shape[0]
        return x.reshape(batch, self.n_views, _VEHICLES_COUNT, _N_FEATURES)

    def forward(self, observations: np.ndarray):
        x = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        views = self._reshape_views(x)  # (B, n_views, V, F)

        ego = views[:, :, _EGO_INDEX, :]  # (B, n_views, F)
        neighbours = views[:, :, _EGO_INDEX + 1 :, :]  # (B, n_views, V-1, F)
        presence = neighbours[..., _PRESENCE_INDEX]  # (B, n_views, V-1)

        ego_b = ego.unsqueeze(-2).expand_as(neighbours)  # (B, n_views, V-1, F)
        pair = torch.cat([ego_b, neighbours], dim=-1)  # (B, n_views, V-1, 2F)
        e = self.phi(pair)  # (B, n_views, V-1, d_phi)

        logits = self.score(e).squeeze(-1)  # (B, n_views, V-1)
        mask = presence > 0.0
        neg_inf = torch.finfo(logits.dtype).min
        logits = torch.where(mask, logits, torch.full_like(logits, neg_inf))

        any_present = mask.any(dim=-1, keepdim=True)  # (B, n_views, 1)
        # Rows with no present neighbour would NaN under softmax(all -inf); give
        # them a uniform distribution and zero the result afterwards.
        safe_logits = torch.where(any_present, logits, torch.zeros_like(logits))
        alpha = torch.softmax(safe_logits, dim=-1)  # (B, n_views, V-1)

        v = self.value(e)  # (B, n_views, V-1, d_phi)
        z = (alpha.unsqueeze(-1) * v).sum(dim=-2)  # (B, n_views, d_phi)
        z = torch.where(any_present, z, torch.zeros_like(z))

        cat = torch.cat([ego, z], dim=-1)  # (B, n_views, F + d_phi)
        per_view = self.output(cat)  # (B, n_views, H)
        state = per_view.mean(dim=1)  # (B, H) -- invariant over views
        return {"state": state}
