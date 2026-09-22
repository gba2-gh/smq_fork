"""Shared constants and small helpers for the segmentation-first programme."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Only the datasets named in scope (HuGaDB is out of scope for this programme).
DATASETS = {
    "lara":   dict(fps=50, patch=50, joints=22, feats=6),
    "babel1": dict(fps=30, patch=30, joints=25, feats=3),
    "babel2": dict(fps=30, patch=30, joints=25, feats=3),
    "babel3": dict(fps=30, patch=30, joints=25, feats=3),
}


def read_mapping(dataset):
    m = {}
    for line in (ROOT / "data" / dataset / "mapping" / "mapping.txt").read_text().splitlines():
        i, a = line.strip().split(" ", 1)
        m[int(i)] = a
    return m


def load_gt_labels(root, name, a2i):
    txt = (Path(root) / "groundTruth" / f"{name}.txt").read_text().splitlines()
    return np.array([a2i[a] for a in txt], dtype=np.int64)


def runs(labels):
    """Maximal constant-label runs as (start, end_exclusive, label)."""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return []
    cut = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut, [len(labels)]])
    return [(int(s), int(e), int(labels[s])) for s, e in zip(starts, ends)]


def lara_subject(name):
    """LARa file 'L01_S05_R12' -> '05'."""
    return name.split("_")[1][1:]
