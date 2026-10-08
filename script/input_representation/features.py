"""The two new per-window inputs and their registration in the existing Phase 5 preparation pipeline.

The pipeline (`script.week5.timescale_inspection.tokenizers`) is not edited: new specs and one extraction wrapper are
installed at run time, so the standardize -> PCA-64 -> K-means-512 -> split path and its artifact layout are unchanged.

  relational_bodyscale  T2 with every frame's pairwise distances divided by the recording's body-scale proxy
  centered_motion       centroid-centred joint positions (divided by the same scale) plus centroid velocity; window mean,
                        sd and mean |change|. Not heading-invariant: orientation stays visible (declared)

Body-scale proxy: per recording, the median over frames of the mean pairwise inter-joint distance. It uses only that
recording's own positions (label-free) and is pose-dependent only weakly. Neither input uses labels or subject IDs."""
from __future__ import annotations

import numpy as np

from script.week5.raw_kmeans_cross_dataset import datasets as D
from script.week5.timescale_inspection import tokenizers as TOK
from script.input_representation import config as C

NEW_SOURCES = ("relational_bodyscale", "centered_motion")


def body_scale(dist: np.ndarray) -> float:
    """dist (T, J) pairwise distances of one recording."""
    return float(np.median(dist.mean(axis=1)))


def frame_features(pos: np.ndarray, source: str) -> np.ndarray:
    """pos (T, V, 3) -> per-frame features (T, J)."""
    dist = TOK.frame_distances(pos)
    scale = body_scale(dist)
    if source == "relational_bodyscale":
        return (dist / scale).astype(np.float32)
    if source == "centered_motion":
        centroid = pos.mean(axis=1)                                   # (T, 3)
        rel = (pos - centroid[:, None, :]) / scale                    # (T, V, 3)
        vel = np.zeros_like(centroid)
        vel[1:] = np.diff(centroid, axis=0) / scale
        return np.concatenate([rel.reshape(len(pos), -1), vel], axis=1).astype(np.float32)
    raise ValueError(source)


def extract_new(spec: D.DatasetSpec, source: str, names: list[str]) -> dict[str, dict]:
    ch = TOK.position_channels(spec.name)
    out = {}
    for name in names:
        arr = np.load(D.features_dir(spec) / name)[ch]                 # (3, T, V, M)
        if arr.shape[3] != 1:
            raise ValueError("single-person recordings only")
        pos = arr[..., 0].transpose(1, 2, 0).astype(np.float32)        # (T, V, 3)
        T = pos.shape[0]
        P = int(np.ceil(T / spec.window))
        lengths = np.full(P, spec.window, dtype=np.int32)
        if P * spec.window - T:
            lengths[-1] = spec.window - (P * spec.window - T)
        out[name] = dict(flat=TOK.window_stats(frame_features(pos, source), spec.window), lengths=lengths, num_patches=P)
    return out


_installed = False


def install() -> None:
    """Register the new tokenizers and extend `extract` for their sources (idempotent, in-process only)."""
    global _installed
    if _installed:
        return
    for variant in C.NEW_VARIANTS:
        key, source = C.VARIANTS[variant]
        TOK.SPECS[key] = TOK.TokenizerSpec(key, key, source, 64, False, (C.DATASET,), selectable=False)
    original = TOK.extract

    def extract(spec, source, names, device=None):
        return extract_new(spec, source, names) if source in NEW_SOURCES else original(spec, source, names, device)

    TOK.extract = extract
    _installed = True


def prepare(variant: str, seed: int):
    """Build (or reuse) the variant's preparation for one seed; returns its directory."""
    install()
    key = C.VARIANTS[variant][0]
    return TOK.prepare_cell(C.DATASET, key, seed, C.prep_dir(variant, seed))
