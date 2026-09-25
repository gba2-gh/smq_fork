"""Provenance, caching, tokenizers, and folds for Stage A.

Reuses the verified Q1/Q2 primitives (script/exp2round/q1q2/core.py and
cache_and_gate.py) rather than reimplementing checkpoint verification,
windowing, or vocabulary fitting. See instructions_agent §2-3.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.action_transport import config
from script.exp2round.q1q2 import cache_and_gate
from script.exp2round.q1q2 import core

_GATE_CACHE: dict[str, dict] = {}


def verify_provenance(dataset: str) -> dict:
    """Run the established checkpoint/quantization/D7 gate once per process.

    Raises if the gate fails; per instructions §2, "Stop the affected dataset
    on failure, retaining the other" -- callers decide whether to continue
    with the other dataset.
    """
    if dataset not in _GATE_CACHE:
        result = cache_and_gate.verify(dataset)
        if not result["passed"]:
            raise RuntimeError(f"{dataset} provenance gate failed: {result}")
        _GATE_CACHE[dataset] = result
    return _GATE_CACHE[dataset]


@dataclass
class RecordingMeta:
    name: str
    subject: str
    frame_count: int
    starts: np.ndarray  # int32 [L], window start frame index
    lengths: np.ndarray  # int32 [L], real frame count per window (<=W)


@dataclass
class LoadedDataset:
    dataset: str
    core_dataset: "core.Dataset"
    metas: list[RecordingMeta]
    fold_of: np.ndarray  # int, aligned with core_dataset.names order


def load_dataset(dataset: str) -> LoadedDataset:
    """Verify provenance, load the same-checkpoint latent cache, and build the
    fixed window grid (independent of normalization: it only depends on frame
    counts, per instructions §2/§4)."""
    verify_provenance(dataset)
    ds = core.load_dataset(dataset, representation="latent")
    window = config.WINDOW[dataset]
    metas = []
    for name, subject, array in zip(ds.names, ds.subjects, ds.arrays):
        n = len(array)
        starts = np.arange(0, n, window, dtype=np.int32)
        lengths = np.minimum(window, n - starts).astype(np.int32)
        metas.append(RecordingMeta(name, subject, int(n), starts, lengths))
    lengths_frames = [len(gt) for gt in ds.gts]
    fold_of = np.asarray(
        core.make_folds(ds.subjects, lengths_frames, n_folds=4, seed=config.SAMPLING_SEED),
        dtype=np.int32,
    )
    return LoadedDataset(dataset, ds, metas, fold_of)


def prepare_arrays(loaded: LoadedDataset, normalize: bool) -> list[np.ndarray]:
    """Per-recording latent arrays, z-scored per recording if requested. This
    is an offline preprocessing condition per instructions §2, not a fitted
    parameter -- it needs no fitting population."""
    if normalize:
        return core.recording_zscore(loaded.core_dataset.arrays)
    return [np.asarray(a, dtype=np.float32) for a in loaded.core_dataset.arrays]


def fit_indices(loaded: LoadedDataset, protocol: str, fold: int | None) -> tuple[int, ...]:
    if protocol == "pooled":
        return tuple(range(len(loaded.metas)))
    if fold is None:
        raise ValueError("subject_disjoint requires a fold")
    return tuple(int(i) for i in np.flatnonzero(loaded.fold_of != fold))


def held_out_indices(loaded: LoadedDataset, protocol: str, fold: int | None) -> tuple[int, ...]:
    if protocol == "pooled":
        return tuple(range(len(loaded.metas)))
    return tuple(int(i) for i in np.flatnonzero(loaded.fold_of == fold))


@dataclass
class Tokenizer:
    dataset: str
    normalize: bool
    seed: int
    protocol: str
    fold: int | None
    fit_indices: tuple[int, ...]
    centers_fine: np.ndarray  # [500, d] KMeans centers in PCA space
    centers_kc: np.ndarray  # [C, d]
    codes_fine: list[np.ndarray]  # per recording, int32 [L]
    codes_kc: list[np.ndarray]
    pca_windows: list[np.ndarray]  # per recording, float32 [L, d] (post scaler+PCA)
    fit_sample_ids: np.ndarray
    terminal_windows: int
    n_fit_windows: int


def _cache_path(dataset: str, normalize: bool, protocol: str, fold: int | None, seed: int) -> Path:
    norm_part = "norm" if normalize else "raw"
    fold_part = "pooled" if fold is None else f"fold{fold}"
    name = f"{dataset}_{norm_part}_{protocol}_{fold_part}_seed{seed}_v{config.PROTOCOL_VERSION}.npz"
    return config.OUT_ROOT / "cache" / "tokenizers" / name


def build_or_load_tokenizer(loaded: LoadedDataset, arrays: list[np.ndarray], normalize: bool,
                            protocol: str, fold: int | None, seed: int) -> Tokenizer:
    path = _cache_path(loaded.dataset, normalize, protocol, fold, seed)
    fit_idx = fit_indices(loaded, protocol, fold)
    if path.exists():
        blob = np.load(path, allow_pickle=True)
        if tuple(int(i) for i in blob["fit_indices"]) == fit_idx:
            return Tokenizer(
                dataset=loaded.dataset, normalize=normalize, seed=seed, protocol=protocol,
                fold=fold, fit_indices=fit_idx,
                centers_fine=blob["centers_fine"], centers_kc=blob["centers_kc"],
                codes_fine=list(blob["codes_fine"]), codes_kc=list(blob["codes_kc"]),
                pca_windows=list(blob["pca_windows"]), fit_sample_ids=blob["fit_sample_ids"],
                terminal_windows=int(blob["terminal_windows"]), n_fit_windows=int(blob["n_fit_windows"]),
            )
    window = config.WINDOW[loaded.dataset]
    prepared = core.prepare_windows(arrays, window, fit_idx, max_sample=10_000, seed=config.SAMPLING_SEED)
    kmeans_fine = core.fit_vocabulary(prepared, config.K_FINE, seed)
    kmeans_kc = core.fit_vocabulary(prepared, config.num_actions(loaded.dataset), seed)
    codes_fine = core.assign_windows(kmeans_fine, prepared.transformed)
    codes_kc = core.assign_windows(kmeans_kc, prepared.transformed)
    tok = Tokenizer(
        dataset=loaded.dataset, normalize=normalize, seed=seed, protocol=protocol, fold=fold,
        fit_indices=fit_idx, centers_fine=kmeans_fine.cluster_centers_.astype(np.float32),
        centers_kc=kmeans_kc.cluster_centers_.astype(np.float32),
        codes_fine=codes_fine, codes_kc=codes_kc, pca_windows=prepared.transformed,
        fit_sample_ids=prepared.fit_ids, terminal_windows=prepared.terminal_windows,
        n_fit_windows=len(prepared.fit_features),
    )
    _save_tokenizer(path, tok)
    return tok


def _save_tokenizer(path: Path, tok: Tokenizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    codes_fine_obj = np.empty(len(tok.codes_fine), dtype=object)
    codes_fine_obj[:] = tok.codes_fine
    codes_kc_obj = np.empty(len(tok.codes_kc), dtype=object)
    codes_kc_obj[:] = tok.codes_kc
    pca_obj = np.empty(len(tok.pca_windows), dtype=object)
    pca_obj[:] = tok.pca_windows
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temp, fit_indices=np.asarray(tok.fit_indices, dtype=np.int32),
        centers_fine=tok.centers_fine, centers_kc=tok.centers_kc,
        codes_fine=codes_fine_obj, codes_kc=codes_kc_obj, pca_windows=pca_obj,
        fit_sample_ids=tok.fit_sample_ids, terminal_windows=tok.terminal_windows,
        n_fit_windows=tok.n_fit_windows,
    )
    import os
    os.replace(temp, path)
