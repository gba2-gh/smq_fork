"""A0+filter: a fixed, one-pass, frame-weighted mode filter applied to A0's
raw states. No fitting; derived from A0's saved predictions (instructions
§4)."""

from __future__ import annotations

import numpy as np


def mode_filter(states: np.ndarray, lengths: np.ndarray, half_width: int) -> np.ndarray:
    """For each window t, vote among windows s in the same recording with
    |t-s|<=half_width, weighted by lengths[s]. Ties keep the original state
    at t if tied, else the smallest state id. Single pass, no iteration."""
    num_windows = len(states)
    if half_width <= 0:
        return states.copy()
    max_state = int(states.max()) + 1 if num_windows else 0
    output = np.empty_like(states)
    for t in range(num_windows):
        lo = max(0, t - half_width)
        hi = min(num_windows, t + half_width + 1)
        votes = np.zeros(max_state, dtype=np.float64)
        for s in range(lo, hi):
            votes[states[s]] += lengths[s]
        best = float(votes.max())
        tied = np.flatnonzero(votes == best)
        if states[t] in tied:
            output[t] = states[t]
        else:
            output[t] = int(tied.min())
    return output
