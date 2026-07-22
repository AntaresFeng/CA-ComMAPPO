"""Custom XuanCe components for CA-ComMAPPO and its baselines.

All classes here are *runtime-registered* into XuanCe's torch registries (see
``registration.py``) rather than edited into the installed ``xuance`` package.
This keeps ``.venv`` untouched, mirroring ``ca_commappo/xuance_compat.py``.
"""

from ca_commappo.algorithms.representations import PermutationInvariantAttention
from ca_commappo.algorithms.registration import register_components

__all__ = ["PermutationInvariantAttention", "register_components"]
