"""
Temporal partitions ("edges" arrays) for the segmentation-first programme.

A partition of a recording of T frames is an int array `edges` of length n+1 with
edges[0] = 0, edges[-1] = T and strictly increasing entries; segment i covers
frames [edges[i], edges[i+1]). Every frame belongs to exactly one segment.

Conditions
  fixed_edges     original one-second grid; the final partial window is kept as its
                  own (shorter) segment (documented "tail = own segment" convention)
  oracle_edges    maximal constant-label runs of the existing annotation
  random_edges    duration-matched randomisation: permute the given durations within
                  the recording and re-accumulate (exact cover of the recording)
  dp_edges        label-free exact dynamic programme (Experiment 2)
"""
import math

import numpy as np
import torch
from numba import njit


# ---------------------------------------------------------------- basic partitions
def fixed_edges(T, W):
    e = list(range(0, T, W)) + [T]
    return np.array(e, dtype=np.int64)


def oracle_edges(labels):
    labels = np.asarray(labels)
    cut = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    return np.concatenate([[0], cut, [len(labels)]]).astype(np.int64)


def durations(edges):
    return np.diff(edges)


def random_edges(durs, rng):
    """Permute `durs` (multiset preserved) and re-accumulate."""
    p = rng.permutation(len(durs))
    return np.concatenate([[0], np.cumsum(np.asarray(durs)[p])]).astype(np.int64)


def check_partition(edges, T):
    e = np.asarray(edges)
    assert e[0] == 0 and e[-1] == T, (e[0], e[-1], T)
    assert np.all(np.diff(e) > 0), "empty or reversed segment"
    assert int(np.diff(e).sum()) == T
    return True


def frame_segment_ids(edges, T):
    """seg id of every frame (each frame in exactly one segment)."""
    ids = np.repeat(np.arange(len(edges) - 1), np.diff(edges))
    assert len(ids) == T
    return ids


def null_informativeness(durs, edges_random_list, edges_oracle):
    """Diagnostics for whether a duration-permutation null is meaningful for one recording."""
    d = np.asarray(durs)
    n = len(d)
    vals, counts = np.unique(d, return_counts=True)
    n_perm = math.factorial(n) // int(np.prod([math.factorial(c) for c in counts])) if n <= 20 else 10 ** 9
    same = float(np.mean([np.array_equal(e, edges_oracle) for e in edges_random_list]))
    cv = float(d.std() / d.mean()) if n > 0 else 0.0
    return dict(n_segments=int(n), distinct_arrangements=int(min(n_perm, 10 ** 9)), duration_cv=cv,
                frac_randomisations_identical_to_oracle=same)


# ---------------------------------------------------------------- exact DP
@njit(cache=True)
def _dp(costb, Lmin, Lmax, lam):
    """costb[b, l-Lmin] = cost of the segment ending at b with length l (inf if invalid).
    returns (best[T+1], choice[T+1]) minimising sum(cost) + lam * n_segments exactly."""
    Tp1 = costb.shape[0]
    best = np.full(Tp1, np.inf)
    choice = np.zeros(Tp1, dtype=np.int64)
    best[0] = 0.0
    for b in range(1, Tp1):
        lmax = min(Lmax, b)
        bv = np.inf
        bl = 0
        for l in range(Lmin, lmax + 1):
            a = b - l
            if best[a] == np.inf:
                continue
            c = best[a] + costb[b, l - Lmin] + lam
            if c < bv:
                bv = c
                bl = l
        best[b] = bv
        choice[b] = bl
    return best, choice


def linear_fit_cost_end_indexed(Z, Lmin, Lmax, device="cuda", chunk_ends=64):
    """
    costb[b, l-Lmin] = mean-over-dimensions residual sum of squares of the least-squares
    linear fit (per dimension, against time) to Z[b-l:b], summed over the segment's frames.

    Z: (T, D) already standardised (per-dimension unit variance on fitting frames), so the
    cost is in units of "standardised variance x frames" and is NOT inflated by D:
    dividing by D makes lambda comparable across the 352-d (LARa) and 400-d (BABEL) latents.
    Computed in float64 from prefix sums (closed-form linear regression), exact.
    """
    T, D = Z.shape
    z = torch.as_tensor(Z, dtype=torch.float64, device=device)
    u = torch.arange(T, dtype=torch.float64, device=device)[:, None]
    S1 = torch.cat([torch.zeros(1, D, dtype=torch.float64, device=device), torch.cumsum(z, 0)], 0)
    Sxz = torch.cat([torch.zeros(1, D, dtype=torch.float64, device=device), torch.cumsum(u * z, 0)], 0)
    S2 = torch.cat([torch.zeros(1, dtype=torch.float64, device=device), torch.cumsum((z * z).sum(1), 0)], 0)
    Ls = torch.arange(Lmin, Lmax + 1, dtype=torch.float64, device=device)
    nL = len(Ls)
    Sxx = Ls * (Ls ** 2 - 1) / 12.0
    ubar = (Ls - 1) / 2.0
    out = torch.full((T + 1, nL), float("inf"), dtype=torch.float64, device=device)
    for b0 in range(Lmin, T + 1, chunk_ends):
        bs = torch.arange(b0, min(T, b0 + chunk_ends - 1) + 1, device=device)
        a = bs[:, None] - Ls.long()[None, :]                      # (B, L)
        valid = a >= 0
        ac = a.clamp(min=0)
        sz = S1[bs][:, None, :] - S1[ac]                          # (B, L, D)
        sq = S2[bs][:, None] - S2[ac]                             # (B, L)
        suz = (Sxz[bs][:, None, :] - Sxz[ac]) - ac.double()[..., None] * sz
        cov = suz - ubar[None, :, None] * sz
        rss = sq - (sz ** 2).sum(-1) / Ls[None, :] - (cov ** 2).sum(-1) / Sxx[None, :]
        rss = (rss.clamp(min=0) / D)
        rss = torch.where(valid, rss, torch.full_like(rss, float("inf")))
        out[bs] = rss
    return out.cpu().numpy()


def dp_segment(costb, Lmin, Lmax, lam):
    """Exact optimal partition from a precomputed end-indexed cost matrix."""
    T = costb.shape[0] - 1
    if T < Lmin:
        return np.array([0, T], dtype=np.int64), dict(fallback_single=True, total_cost=float("nan"))
    best, choice = _dp(costb, Lmin, Lmax, float(lam))
    assert np.isfinite(best[T])
    edges = [T]
    b = T
    while b > 0:
        b -= int(choice[b])
        edges.append(b)
    edges = np.array(edges[::-1], dtype=np.int64)
    return edges, dict(fallback_single=False, total_cost=float(best[T]))


def partition_cost(costb, edges, Lmin):
    """Sum of segment costs (without penalty) of an arbitrary partition (for checks)."""
    tot = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        tot += costb[b, (b - a) - Lmin]
    return tot


def brute_force_dp(costb, Lmin, Lmax, lam):
    """Exponential reference (small T only) used to validate the DP."""
    T = costb.shape[0] - 1
    best = (float("inf"), None)

    def rec(pos, edges, acc):
        nonlocal best
        if pos == T:
            if acc < best[0]:
                best = (acc, list(edges))
            return
        for l in range(Lmin, min(Lmax, T - pos) + 1):
            c = costb[pos + l, l - Lmin] + lam
            if np.isfinite(c):
                rec(pos + l, edges + [pos + l], acc + c)

    rec(0, [0], 0.0)
    return best
