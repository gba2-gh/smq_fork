"""The within-recording permutation adjacency diagnostic (permuted_asot),
including the mandatory perm/inverse_perm restoration convention from
instructions §3 (v1.3 correction). This is a paired sanity check, not a
chance baseline: it destroys temporal order but not recording-level
action/subject composition.
"""

from __future__ import annotations

import numpy as np


def build_permutation(lengths: np.ndarray, window: int, rng: np.random.Generator) -> np.ndarray:
    """perm[j] = original window index occupying shuffled position j.

    Only complete equal-length windows (real length == window) are permuted
    among themselves; the terminal partial window (if any) stays in place.
    Because every permuted window has the identical real length `window`,
    this is exact-count- and exact-frame-weighted-count-preserving by
    construction: only which code sits at which position changes.
    """
    num_windows = len(lengths)
    perm = np.arange(num_windows, dtype=np.int64)
    complete_idx = np.flatnonzero(lengths == window)
    if len(complete_idx) > 1:
        perm[complete_idx] = rng.permutation(complete_idx)
    return perm


def apply_permutation(codes: np.ndarray, perm: np.ndarray) -> np.ndarray:
    return codes[perm]


def restore_states(states_shuffled: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """state_original[perm[j]] = states_shuffled[j], i.e. restore to the
    original timeline. Equivalent to state_shuffled[inverse_perm]."""
    inverse_perm = np.argsort(perm)
    return states_shuffled[inverse_perm]
