"""Runtime registration of custom XuanCe components.

This mirrors ``ca_commappo/xuance_compat.py``'s approach: custom classes live in
this repository and are injected into XuanCe's torch registries *at runtime*
before an agent is constructed, so the installed ``xuance`` package under
``.venv`` is never edited.

Call ``register_components()`` from ``train_highway_intersection.run()`` before
``get_agent_class(...)`` (see the prompt's §1.1 / §0.2). The call is idempotent.
"""

from __future__ import annotations

from typing import Dict, Type

from xuance.torch import Module
from xuance.torch.representations import REGISTRY_Representation

# Lazy imports of the classes to avoid pulling heavyweight torch graphs at
# module import time; they are only needed when registration actually runs.


def _register(registry: Dict[str, Type[Module]], key: str, cls: Type[Module]) -> None:
    """Register ``cls`` under ``key`` unless already registered (idempotent)."""
    if registry.get(key) is cls:
        return
    registry[key] = cls


def register_components() -> None:
    """Register all CA-ComMAPPO custom components into XuanCe's registries.

    Safe to call multiple times. Currently registers:

    * ``PermutationInvariantAttention`` -> ``REGISTRY_Representation``
      (Attention-MAPPO / CA-ComMAPPO shared encoder);
    * ``CA_ComMAPPO_Policy`` -> ``REGISTRY_Policy`` (thin MAAC_Policy subclass;
      comm lives in the actor representation);
    * ``CA_ComMAPPO_Learner`` -> ``REGISTRY_Learners`` (thin MAPPO_Learner subclass).
    """
    from xuance.torch.learners import REGISTRY_Learners
    from xuance.torch.policies import REGISTRY_Policy

    from ca_commappo.algorithms.ca_commappo_agents import (
        CA_ComMAPPO_Learner,
        CA_ComMAPPO_Policy,
    )
    from ca_commappo.algorithms.representations import PermutationInvariantAttention

    _register(
        REGISTRY_Representation,
        "PermutationInvariantAttention",
        PermutationInvariantAttention,
    )
    _register(REGISTRY_Policy, "CA_ComMAPPO_Policy", CA_ComMAPPO_Policy)
    _register(REGISTRY_Learners, "CA_ComMAPPO_Learner", CA_ComMAPPO_Learner)
