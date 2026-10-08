"""MotionBERT window features for BABEL-2, registered in the existing Phase 5 preparation pipeline (not edited: a spec and an
`extract` wrapper are installed at run time), plus a per-recording feature cache so the three seeds share one extraction.

Per-frame 512-d MotionBERT representation (frozen) -> the same window statistics as T2 (mean, sd, mean |change| over each
8-frame window, real frames only) -> 1536-d, then the unchanged standardize -> PCA-64 -> K-means-512 path."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from script.pretrained_encoder import config as C
from script.pretrained_encoder import encoder as E
from script.week5.raw_kmeans_cross_dataset import datasets as D
from script.week5.timescale_inspection import tokenizers as TOK

_model = None
_checked = False


def view_rotation(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def checkpoint_ok() -> None:
    global _checked
    if not _checked:
        if hashlib.sha256(E.CHECKPOINT.read_bytes()).hexdigest() != E.CHECKPOINT_SHA256:
            raise RuntimeError("MotionBERT checkpoint hash mismatch")
        _checked = True


def frame_features(ntu: np.ndarray, device=None, deg: float = C.VIEW_ROTATION_DEG) -> np.ndarray:
    """ntu (T, 25, 3) in mm -> (T, 512) per-frame features."""
    global _model
    checkpoint_ok()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _model is None:
        _model = E.load_model(device)
    rotated = ntu @ view_rotation(deg).T
    return E.recording_features(_model, rotated, device)[0]


def cache_path(name: str) -> Path:
    tag = f"{E.CHECKPOINT_SHA256[:8]}_v{C.VIEW_ROTATION_DEG}"
    return C.RESULTS / "feature_cache" / tag / (Path(name).stem + ".npy")


def window_features(frame_feat: np.ndarray, window: int) -> np.ndarray:
    return TOK.window_stats(frame_feat, window)


def extract_mb(spec: D.DatasetSpec, names: list) -> dict:
    out = {}
    for name in names:
        p = cache_path(name)
        if p.exists():
            feat = np.load(p)
        else:
            arr = np.load(D.features_dir(spec) / name)
            if arr.shape[3] != 1:
                raise ValueError("single-person recordings only")
            feat = frame_features(arr[..., 0].transpose(1, 2, 0))
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, feat)
        T = feat.shape[0]
        P = int(np.ceil(T / spec.window))
        lengths = np.full(P, spec.window, dtype=np.int32)
        if P * spec.window - T:
            lengths[-1] = spec.window - (P * spec.window - T)
        out[name] = dict(flat=window_features(feat, spec.window), lengths=lengths, num_patches=P)
    return out


_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    TOK.SPECS[C.MB_TOKENIZER] = TOK.TokenizerSpec(C.MB_TOKENIZER, "MB", C.MB_SOURCE, 64, False, (C.DATASET,), selectable=False)
    original = TOK.extract

    def extract(spec, source, names, device=None):
        return extract_mb(spec, names) if source == C.MB_SOURCE else original(spec, source, names, device)

    TOK.extract = extract
    _installed = True


def prepare(seed: int):
    install()
    return TOK.prepare_cell(C.DATASET, C.MB_TOKENIZER, seed, C.prep_dir("mb", seed))
