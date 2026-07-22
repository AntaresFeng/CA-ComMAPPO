"""Tests for the PermutationInvariantAttention representation.

These tests pin down the Phase 1 Attention-MAPPO encoder contract:

* the ego vehicle (observation index 0) is the fixed query;
* the surrounding vehicles (indices 1..V-1) are pooled with a masked attention
  so the encoding is invariant to their ordering (highway-env shuffles
  ``obs[1:]`` when ``order == "shuffled"``, and re-sorts them by distance every
  step otherwise);
* vehicles flagged ``presence == 0`` are masked out of the attention;
* the same module handles the actor's single-view input (105-dim) and the
  MAPPO global-state critic's multi-view input (n_agents * 105).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ca_commappo.algorithms.representations import PermutationInvariantAttention

VEHICLES_COUNT = 15
N_FEATURES = 7
VIEW_DIM = VEHICLES_COUNT * N_FEATURES  # 105


@pytest.fixture(autouse=True)
def _deterministic_torch():
    """Keep encoder construction deterministic so tests are reproducible."""
    torch.manual_seed(0)
    np.random.seed(0)
    yield
    # Nothing to close for a pure nn.Module, but keep the finally contract.
    torch.manual_seed(0)


def _build_encoder(
    input_dim: int, hidden_sizes=(128, 128)
) -> PermutationInvariantAttention:
    return PermutationInvariantAttention(
        input_shape=(input_dim,),
        hidden_sizes=list(hidden_sizes),
        normalize=torch.nn.LayerNorm,
        initialize=torch.nn.init.orthogonal_,
        activation=torch.nn.ReLU,
        device="cpu",
    )


def _make_scene(batch: int, n_views: int, *, seed: int = 1) -> torch.Tensor:
    """Build a synthetic observation with a present ego and random neighbours.

    Returns shape ``(batch, n_views * VIEW_DIM)``. The ego (vehicle index 0) of
    every view is always present; a couple of neighbours are marked absent
    (``presence == 0``) with zeroed features, mirroring highway-env padding.
    """
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((batch, n_views, VEHICLES_COUNT, N_FEATURES)).astype(
        np.float32
    )
    # Ego always present.
    obs[:, :, 0, 0] = 1.0
    # Some neighbours present, some absent (presence is feature index 0).
    obs[:, :, 1, 0] = 1.0
    obs[:, :, 3, 0] = 1.0
    obs[:, :, 2, 0] = 0.0
    obs[:, :, 2, 1:] = 0.0
    obs[:, :, 4, 0] = 0.0
    obs[:, :, 4, 1:] = 0.0
    return torch.from_numpy(obs.reshape(batch, n_views * VIEW_DIM))


class TestOutputContract:
    def test_output_shapes_matches_last_hidden_size(self):
        encoder = _build_encoder(VIEW_DIM, hidden_sizes=(128, 128))
        assert encoder.output_shapes == {"state": (128,)}

    def test_output_shapes_for_multi_view_critic_dim(self):
        # 2-agent global state critic: 2 * 105 = 210.
        encoder = _build_encoder(2 * VIEW_DIM, hidden_sizes=(128, 128))
        assert encoder.output_shapes == {"state": (128,)}

    def test_forward_returns_state_dict_with_tensor(self):
        encoder = _build_encoder(VIEW_DIM)
        out = encoder(_make_scene(batch=4, n_views=1).numpy())
        assert isinstance(out, dict)
        assert set(out.keys()) == {"state"}
        assert torch.is_tensor(out["state"])
        assert out["state"].shape == (4, 128)


class TestPermutationInvariance:
    def test_shuffling_neighbours_leaves_output_unchanged(self):
        encoder = _build_encoder(VIEW_DIM)
        scene = _make_scene(batch=3, n_views=1, seed=7)
        base = encoder(scene.numpy())["state"]

        # Permute the non-ego vehicles (indices 1..V-1) for each batch row.
        rng = np.random.default_rng(11)
        permuted = scene.clone()
        views = permuted.view(3, VEHICLES_COUNT, N_FEATURES)
        order = rng.permutation(np.arange(1, VEHICLES_COUNT))
        for b in range(3):
            # Keep ego at 0, reorder the rest, keep the same presence pattern.
            rest = views[b, 1:, :].clone()
            views[b, 1:, :] = rest[order - 1, :]
        shuffled = encoder(permuted.numpy())["state"]

        assert torch.allclose(base, shuffled, atol=1e-5, rtol=1e-4)

    def test_swapping_two_present_neighbours_leaves_output_unchanged(self):
        encoder = _build_encoder(VIEW_DIM)
        scene = _make_scene(batch=1, n_views=1, seed=5)
        base = encoder(scene.numpy())["state"]

        swapped = scene.clone()
        views = swapped.view(1, VEHICLES_COUNT, N_FEATURES)
        # Swap two present neighbours (indices 1 and 3 are present in _make_scene).
        views[:, [1, 3], :] = views[:, [3, 1], :].clone()
        out = encoder(swapped.numpy())["state"]

        assert torch.allclose(base, out, atol=1e-5, rtol=1e-4)

    def test_ego_position_is_distinguished(self):
        """Moving the ego into another slot must change the encoding."""
        encoder = _build_encoder(VIEW_DIM)
        scene = _make_scene(batch=1, n_views=1, seed=5)
        base = encoder(scene.numpy())["state"]

        moved = scene.clone()
        views = moved.view(1, VEHICLES_COUNT, N_FEATURES)
        # Swap ego (index 0) with a present neighbour (index 1).
        views[:, [0, 1], :] = views[:, [1, 0], :].clone()
        out = encoder(moved.numpy())["state"]

        assert not torch.allclose(base, out, atol=1e-5, rtol=1e-4)


class TestPresenceMask:
    def test_absent_neighbour_features_do_not_affect_output(self):
        encoder = _build_encoder(VIEW_DIM)
        scene = _make_scene(batch=1, n_views=1, seed=3)
        base = encoder(scene.numpy())["state"]

        # Mutate only the non-presence features of an absent neighbour (index 2).
        mutated = scene.clone()
        views = mutated.view(1, VEHICLES_COUNT, N_FEATURES)
        views[:, 2, 1:] = torch.tensor([9.0, -9.0, 4.0, -4.0, 1.0, -1.0])
        out = encoder(mutated.numpy())["state"]

        assert torch.allclose(base, out, atol=1e-5, rtol=1e-4)

    def test_all_neighbours_absent_produces_finite_output(self):
        encoder = _build_encoder(VIEW_DIM)
        scene = _make_scene(batch=2, n_views=1, seed=3)
        views = scene.view(2, VEHICLES_COUNT, N_FEATURES)
        # Mask every neighbour out (presence == 0, features == 0).
        views[:, 1:, :] = 0.0
        out = encoder(scene.numpy())["state"]

        assert torch.isfinite(out).all()
        assert out.shape == (2, 128)


class TestMultiViewCriticInput:
    def test_two_agent_critic_input_forward_shape(self):
        encoder = _build_encoder(2 * VIEW_DIM)
        out = encoder(_make_scene(batch=4, n_views=2).numpy())
        assert out["state"].shape == (4, 128)

    def test_swapping_agent_views_leaves_output_unchanged(self):
        """The mean-pool over views makes the critic invariant to agent order."""
        encoder = _build_encoder(2 * VIEW_DIM)
        scene = _make_scene(batch=3, n_views=2, seed=9)
        base = encoder(scene.numpy())["state"]

        swapped = scene.clone()
        views = swapped.view(3, 2, VEHICLES_COUNT, N_FEATURES)
        views[:, [0, 1], :, :] = views[:, [1, 0], :, :].clone()
        out = encoder(swapped.numpy())["state"]

        assert torch.allclose(base, out, atol=1e-5, rtol=1e-4)
