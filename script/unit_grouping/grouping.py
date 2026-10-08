"""Label-free groupings of M transport states into C actions. Pure numpy/sklearn; takes plans and prototypes only.

Inputs are fitting-recording soft plans (list of [L, M] arrays whose rows sum to one) and prototypes. Evaluation recordings and
labels never reach this module."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from script.unit_grouping import config as C


def affinity(plans: list, horizon: int = C.HORIZON) -> np.ndarray:
    """A[i, j] = sum over recordings, t, k=1..h of R[t, i] R[t+k, j] (within recordings only), symmetrized, zero diagonal."""
    M = plans[0].shape[1]
    A = np.zeros((M, M))
    for R in plans:
        R = R.astype(np.float64)
        for k in range(1, min(horizon, len(R) - 1) + 1):
            A += R[:-k].T @ R[k:]
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    return A


def shuffle_rows(plans: list, seed: int) -> list:
    """Within-recording row shuffle: each recording keeps its state occupancy, local adjacency is destroyed."""
    out = []
    for i, R in enumerate(plans):
        rng = np.random.default_rng([seed, 31, i])
        out.append(R[rng.permutation(len(R))])
    return out


def spectral_groups(A: np.ndarray, n_groups: int, seed: int) -> np.ndarray:
    """Normalized spectral clustering: top eigenvectors of D^-1/2 A D^-1/2, row-normalized, KMeans(k-means++)."""
    d = np.clip(A.sum(axis=1), 1e-12, None)
    N = A / np.sqrt(d)[:, None] / np.sqrt(d)[None, :]
    vals, vecs = np.linalg.eigh((N + N.T) / 2.0)
    emb = vecs[:, np.argsort(vals)[::-1][:n_groups]]
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    km = KMeans(n_clusters=n_groups, init="k-means++", n_init=C.SPECTRAL_N_INIT, random_state=seed).fit(emb)
    return km.labels_.astype(np.int64)


def appearance_groups(prototypes: np.ndarray, mass: np.ndarray, n_groups: int, seed: int) -> np.ndarray:
    """Weighted KMeans on the state prototypes (weights = state mass on the fitting recordings); no temporal information."""
    km = KMeans(n_clusters=n_groups, init="k-means++", n_init=C.SPECTRAL_N_INIT, random_state=seed)
    km.fit(prototypes, sample_weight=np.clip(mass, 1e-12, None))
    return km.labels_.astype(np.int64)


def size_matched_random(sizes: np.ndarray, n: int, seed: int) -> list:
    """n random partitions of len = sum(sizes) states with exactly the given group sizes."""
    rng = np.random.default_rng([seed, 47])
    labels = np.repeat(np.arange(len(sizes)), sizes)
    return [rng.permutation(labels) for _ in range(n)]


def relabel(states: dict, groups: np.ndarray) -> dict:
    """Window states (names -> [L] ints in 0..M-1) -> group labels. Window count and order are unchanged."""
    return {n: groups[s].astype(np.int32) for n, s in states.items()}


def state_mass(plans: list, lengths: list) -> np.ndarray:
    return sum((R * np.asarray(l, dtype=np.float64)[:, None]).sum(axis=0) for R, l in zip(plans, lengths))
