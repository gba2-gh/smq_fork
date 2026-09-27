"""A-cont: the continuous-feature transport competitor. Same transport and
alternating machinery as categorical.py, with D[t,a] = 1 - z[t].mu[a] on unit
vectors (v1.3: in [0,2], the pinned upstream unary convention, not divided
by two) and prototype updates instead of an emission table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from script.action_transport import transport
from script.action_transport.categorical import FitResult, RecordingInput, fit_alternating


def l2_normalize(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (unit vectors, zero_flags). Zero rows stay zero (flagged)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    zero = norms[:, 0] <= 1e-12
    unit = vectors / np.where(zero[:, None], 1.0, norms)
    unit[zero] = 0.0
    return unit.astype(np.float64), zero


@dataclass
class ContinuousRecordingInput(RecordingInput):
    z: np.ndarray = None  # float64 [L, dim], unit-normalized
    zero_flags: np.ndarray = None  # bool [L]


def build_mu0(grouping: np.ndarray, fitting_recordings: list[ContinuousRecordingInput],
              num_states: int) -> tuple[np.ndarray, np.ndarray]:
    """mu0[a] = normalize(sum over fitting windows with g[code[t]]==a of
    length[t]*z[t]). Returns (mu0, zero-mean flags per state)."""
    dim = fitting_recordings[0].z.shape[1]
    accum = np.zeros((num_states, dim), dtype=np.float64)
    for rec in fitting_recordings:
        weighted = rec.lengths.astype(np.float64)[:, None] * rec.z
        np.add.at(accum, grouping[rec.codes], weighted)
    return l2_normalize(accum)


def cosine_cost(mu: np.ndarray, z: np.ndarray, zero_flags: np.ndarray) -> np.ndarray:
    """D[t,a] = 1 - z[t].mu[a], in [0,2]. Zero input vectors cost 1 for every state."""
    d = 1.0 - z @ mu.T
    d[zero_flags] = 1.0
    return d


def continuous_cost_fn(mu: np.ndarray, rec: ContinuousRecordingInput) -> np.ndarray:
    return cosine_cost(mu, rec.z, rec.zero_flags)


def prototype_update(recordings, responsibilities, mu):
    """mu[a] = normalize(sum length[t]*R[t,a]*z[t]); a zero update keeps the
    previous unit prototype and is flagged as stale."""
    accum = np.zeros_like(mu)
    for rec in recordings:
        weighted = rec.lengths.astype(np.float64)[:, None] * responsibilities[rec.name]
        accum += weighted.T @ rec.z
    new_mu, zero = l2_normalize(accum)
    stale = [int(s) for s in np.flatnonzero(zero)]
    new_mu[zero] = mu[zero]
    return new_mu, stale


def fit_continuous(recordings: list[ContinuousRecordingInput], mu0: np.ndarray, num_states: int,
                   coeffs: transport.Coeffs, inner_steps: int, outer_cap: int,
                   outer_patience: int, outer_reltol: float, device: torch.device,
                   deadline: float | None = None) -> FitResult:
    return fit_alternating(recordings, mu0, num_states, continuous_cost_fn, prototype_update,
                           lambda _mu: 0.0, coeffs, inner_steps, outer_cap, outer_patience,
                           outer_reltol, device, deadline)
