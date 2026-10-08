"""Recording-residual co-occurrence grouping (docs/plans/RESIDUAL_COOCCURRENCE_TEST_PLAN.md).

The temporal affinity A counts states that occur within h windows of each other. Part of A only reflects that two states occur
in the same recordings. Its expectation under a within-recording random permutation of windows, E, is computed exactly
(no sampling). The residual W = max(0, A - E) keeps only adjacency that recording membership does not predict; the states are
then grouped by spectral clustering of W, exactly as the temporal arm groups A. Label-free; fitting plans only."""
from __future__ import annotations

import numpy as np

from script.unit_grouping import config as C


def expected_shuffled_affinity(plans: list, horizon: int = C.HORIZON) -> np.ndarray:
    """E[i, j] = expectation of A[i, j] when each recording's rows are randomly permuted, symmetrized, zero diagonal.

    For a recording with L rows, column sums S and Gram G = R^T R, the expected sum over ordered pairs of distinct rows at a
    given lag is (S S^T - G) / (L (L - 1)) per pair, and lag k has L - k pairs."""
    M = plans[0].shape[1]
    E = np.zeros((M, M))
    for R in plans:
        R = R.astype(np.float64)
        L = len(R)
        if L < 2:
            continue
        S = R.sum(axis=0)
        per_pair = (np.outer(S, S) - R.T @ R) / (L * (L - 1))
        n_pairs = sum(L - k for k in range(1, min(horizon, L - 1) + 1))
        E += n_pairs * per_pair
    E = (E + E.T) / 2.0
    np.fill_diagonal(E, 0.0)
    return E


def residual_affinity(A: np.ndarray, E: np.ndarray) -> np.ndarray:
    W = np.maximum(A - E, 0.0)
    np.fill_diagonal(W, 0.0)
    return W


def residual_share(A: np.ndarray, E: np.ndarray) -> float:
    """Label-free: the share of off-diagonal affinity that recording membership does not explain."""
    return float(np.maximum(A - E, 0.0).sum() / max(A.sum(), 1e-12))
