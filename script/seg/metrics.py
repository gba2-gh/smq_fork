"""
Evaluation metrics for the segmentation-first programme.

Everything that is aggregated over recordings is returned as PER-RECORDING arrays
(sums and counts), so that paired resampling of participants/recordings can be done
downstream by weighting recordings, with no recomputation of the retrieval.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.model.eval_utils import edit_score, f_score  # noqa: E402  (published-convention F1/Edit)

OVERLAPS = [0.10, 0.25, 0.50]


# ------------------------------------------------------------------ retrieval
def precision_at_k(D, Qw, Gcnt, qgrp, ggrp, k, chunk=512):
    """
    Tie-safe class-conditional precision@k with group exclusion.

    D     (Q, G) torch distances between query items and gallery items
    Qw    (Q, C) number of query anchors of each label carried by the query item
    Gcnt  (G, C) number of gallery anchors of each label carried by each gallery item
    qgrp  (Q,)   group id of each query item; gallery items with the same group are excluded
    ggrp  (G,)
    k     precision@k is computed over the k nearest gallery ANCHORS (an item carrying
          several anchors contributes them all at its distance; the item straddling the
          cut-off is used fractionally, i.e. expected precision under random tie-breaking).

    Returns (all per query item, NOT multiplied by Qw)
      P      (Q, C)  precision@k for retrieving label c
      prev   (Q, C)  gallery prevalence of c under the same exclusion (random-ranking expectation)
      valid  (Q,)    True if the query has a non-empty gallery after exclusion
    Qw is only kept in the signature for the brute-force test; callers form S = Qw * P * valid.
    """
    Q, G = D.shape
    C = Gcnt.shape[1]
    dev = D.device
    gtot = Gcnt.sum(1)
    S = torch.zeros(Q, C, dtype=torch.float64, device=dev)
    ch = torch.zeros(Q, C, dtype=torch.float64, device=dev)
    valid_q = torch.zeros(Q, dtype=torch.bool, device=dev)
    kk = min(k, G)
    for s in range(0, Q, chunk):
        e = min(Q, s + chunk)
        d = D[s:e].clone()
        same = qgrp[s:e, None] == ggrp[None, :]
        d[same] = float("inf")
        valid_item = (~same) & (gtot[None, :] > 0)
        d[~valid_item] = float("inf")
        dv, order = torch.topk(d, kk, dim=1, largest=False)            # ascending
        cnt = Gcnt[order].double()                                       # (q, kk, C)
        cnt = cnt * torch.isfinite(dv)[..., None]
        tot = cnt.sum(-1)
        cum = torch.cumsum(tot, 1)
        before = cum - tot
        take = torch.minimum(torch.clamp(k - before, min=0), tot)
        frac = take / tot.clamp(min=1e-12)
        rel = (cnt * frac[..., None]).sum(1)                             # (q, C) relevant mass among top-k
        mass = take.sum(1)                                               # actual mass used (<= k)
        P = rel / mass.clamp(min=1e-12)[:, None]
        S[s:e] = P
        valid_q[s:e] = mass > 0
        # chance: prevalence in gallery excluding the group
        gc = Gcnt.double().sum(0)[None, :] - same.double() @ Gcnt.double()   # (q, C)
        prev = gc / gc.sum(1, keepdim=True).clamp(min=1e-12)
        ch[s:e] = prev
    return S, ch, valid_q


def aggregate_per_recording(S, Qw, rec_of_query, n_rec, ncls):
    """Sum per-item (Q, C) arrays into per-recording (R, C) arrays."""
    Sr = np.zeros((n_rec, ncls))
    nr = np.zeros((n_rec, ncls))
    np.add.at(Sr, rec_of_query, S.cpu().numpy() if torch.is_tensor(S) else S)
    np.add.at(nr, rec_of_query, Qw.cpu().numpy() if torch.is_tensor(Qw) else Qw)
    return Sr, nr


def class_balanced(Sr, nr, w=None, min_n=1):
    """Class-balanced mean of (sum S / sum n) over classes with at least min_n queries."""
    if w is None:
        w = np.ones(len(Sr))
    S = (Sr * w[:, None]).sum(0)
    n = (nr * w[:, None]).sum(0)
    ok = n >= min_n
    if not ok.any():
        return float("nan")
    return float(np.mean(S[ok] / n[ok]))


# ------------------------------------------------------------------ contingency metrics
def _entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def nmi_from_contingency(M):
    """Permutation-invariant NMI (arithmetic-mean normalisation) from a cluster x label table."""
    M = np.asarray(M, dtype=np.float64)
    n = M.sum()
    if n == 0:
        return float("nan")
    pc, pl = M.sum(1) / n, M.sum(0) / n
    hc, hl = _entropy(pc), _entropy(pl)
    P = M / n
    nz = P > 0
    mi = float((P[nz] * np.log(P[nz] / (pc[:, None] * pl[None, :])[nz])).sum())
    denom = 0.5 * (hc + hl)
    return mi / denom if denom > 0 else 0.0


def ari_from_contingency(M):
    M = np.asarray(M, dtype=np.float64)
    n = M.sum()
    c2 = lambda x: x * (x - 1) / 2.0
    sij = c2(M).sum()
    sa = c2(M.sum(1)).sum()
    sb = c2(M.sum(0)).sum()
    tot = c2(n)
    exp = sa * sb / tot if tot > 0 else 0.0
    mx = 0.5 * (sa + sb)
    return float((sij - exp) / (mx - exp)) if mx != exp else 0.0


def hungarian_map(M):
    """One-to-one cluster->label mapping maximising matched frames (benchmark convention).
    Uses the table it is given (evaluation labels): a benchmark-protocol readout, not deployable."""
    M = np.asarray(M, dtype=np.float64)
    K, C = M.shape
    n = max(K, C)
    P = np.zeros((n, n))
    P[:K, :C] = M
    r, c = linear_sum_assignment(-P)
    mp = {int(i): int(j) for i, j in zip(r, c) if i < K}
    # clusters matched to a padded (non-existent) label get label -1 (always wrong)
    return {i: (j if j < C else -1) for i, j in mp.items()}


def majority_map(M):
    """Cluster -> majority label from the table it is given (used on FITTING recordings only)."""
    M = np.asarray(M)
    out = {}
    for k in range(M.shape[0]):
        out[k] = int(np.argmax(M[k])) if M[k].sum() > 0 else -1
    return out


# ------------------------------------------------------------------ segmental metrics
def frame_scores(pred_labels, gt):
    """Published-convention F1@{10,25,50} components and Edit for one recording
    (src/model/eval_utils; the MS-TCN implementation used by SMQ)."""
    tp = np.zeros(3); fp = np.zeros(3); fn = np.zeros(3)
    for i, o in enumerate(OVERLAPS):
        tp[i], fp[i], fn[i] = f_score(pred_labels, gt, o)
    return dict(correct=int((pred_labels == gt).sum()), n=int(len(gt)),
                edit=float(edit_score(pred_labels, gt)), tp=tp, fp=fp, fn=fn)


def boundary_positions(seq):
    """Frame indices i (1..T-1) at which seq[i] != seq[i-1]."""
    seq = np.asarray(seq)
    return np.flatnonzero(seq[1:] != seq[:-1]) + 1


def match_boundaries(pred, gt, tol):
    """One-to-one boundary matching within +-tol frames. Greedy two-pointer on sorted positions
    is a maximum-cardinality matching for a common tolerance in 1-D. Handles empty sides:
    no predictions -> tp=fp=0, fn=|gt|; no ground truth -> tp=fn=0, fp=|pred|."""
    p = np.sort(np.asarray(pred)); g = np.sort(np.asarray(gt))
    i = j = tp = 0
    while i < len(p) and j < len(g):
        if abs(p[i] - g[j]) <= tol:
            tp += 1; i += 1; j += 1
        elif p[i] < g[j] - tol:
            i += 1
        else:
            j += 1
    return tp, len(p) - tp, len(g) - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if (p == p and r == r and p + r > 0) else (0.0 if (tp + fp + fn) else float("nan"))
    return p, r, f
