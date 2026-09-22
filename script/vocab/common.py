"""Shared helpers for the vocabulary programme (Experiments 1-3). Reuses script/seg (frozen features, folds,
anchors, partitions, retrieval) unchanged."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from script.seg import partition as P  # noqa: E402
from script.seg import pipeline as PL  # noqa: E402
from script.seg import represent as R  # noqa: E402

OUT = ROOT / "results" / "vocab"
OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "figures" / "vocab"
FIG.mkdir(parents=True, exist_ok=True)

DATASETS = ("lara", "babel1")
KMEANS_SEEDS = PL.KMEANS_SEEDS          # (0, 1, 2)
K_MULT = (1, 4, 16)


def mean_vectors(z_raw, edges, mu, sd):
    """Mean of the standardised frame latents over each segment (exact frame mean). (n_seg, D)"""
    z = (np.asarray(z_raw, dtype=np.float32) - mu) / sd
    C = np.concatenate([np.zeros((1, z.shape[1]), np.float64), np.cumsum(z.astype(np.float64), 0)], 0)
    L = np.diff(edges)[:, None]
    return ((C[edges[1:]] - C[edges[:-1]]) / L).astype(np.float32)


def seg_vectors(rep, z_raw, edges, mu, sd):
    if rep == "mean":
        return mean_vectors(z_raw, edges, mu, sd)
    return R.segment_vectors(z_raw, edges, mu, sd, PL.N_BINS)


def partitions(data, kind):
    if kind == "fixed":
        return [P.fixed_edges(int(t), data.W) for t in data.T]
    if kind == "oracle":
        return [P.oracle_edges(l) for l in data.labels]
    raise ValueError(kind)


def seg_majority_label(lab, edges, C):
    cnt = PL.seg_label_counts(lab, edges, C)
    return cnt.argmax(1), cnt.max(1) / cnt.sum(1)
