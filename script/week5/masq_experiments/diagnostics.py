"""Identity diagnostics for one scored unit: do the native codes and the pre-quantization latents separate subjects or
recordings? No training, no GPU. Reads the unit's predictions.npz.

Code AMI (adjusted mutual information, chance-corrected because K is small) with subject (HuGaDB, LARa; subject parsed
from the recording name) and with recording; same-recording share of nearest neighbours of the patch-mean latents
(candidates within one patch of the query excluded). The per-class block uses labels (analysis only).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

from script.week5.masq_experiments import config as C
from script.week5.raw_kmeans_cross_dataset import datasets as D

MAX_FRAMES = 300_000  # declared cap: an evenly spaced subsample above this
MIN_CLASS_FRAMES = 1000


def _subsample(n: int) -> np.ndarray:
    return np.arange(n) if n <= MAX_FRAMES else np.linspace(0, n - 1, MAX_FRAMES).astype(np.int64)


def ami(a: np.ndarray, b: np.ndarray) -> float:
    return float(adjusted_mutual_info_score(a, b, average_method="arithmetic"))


def code_association(names, preds, gts, dataset: str) -> dict:
    spec = D.SPECS[dataset]
    codes = np.concatenate([np.asarray(p) for p in preds]).astype(np.int64)
    rec = np.concatenate([np.full(len(p), i) for i, p in enumerate(preds)])
    gt = np.concatenate([np.asarray(g) for g in gts]).astype(np.int64)
    idx = _subsample(len(codes))
    out = dict(frames=int(len(codes)), frames_used=int(len(idx)), recording_ami=ami(codes[idx], rec[idx]), subject_ami=None,
               per_class=None)
    if spec.has_subject:
        subjects = sorted({D.subject_of(spec, n) for n in names})
        sid_of = {s: i for i, s in enumerate(subjects)}
        sid = np.concatenate([np.full(len(p), sid_of[D.subject_of(spec, n)]) for n, p in zip(names, preds)])
        out["subject_ami"] = ami(codes[idx], sid[idx])
        per, total, weighted = {}, 0, 0.0
        for c in np.unique(gt):
            sel = np.flatnonzero(gt == c)
            if len(sel) < MIN_CLASS_FRAMES:
                continue
            sub = sel[_subsample(len(sel))]
            per[str(int(c))] = dict(frames=int(len(sel)), subject_ami=ami(codes[sub], sid[sub]))
            total += len(sel)
            weighted += len(sel) * per[str(int(c))]["subject_ami"]
        out["per_class"] = dict(uses_labels=True, classes=per, frame_weighted_mean=(weighted / total if total else None))
    return out


def latent_neighbours(names, patch_latents) -> float | None:
    from script.week5.timescale_inspection.units9 import nn_same_recording_share
    feats = {n: np.asarray(f, dtype=np.float32) for n, f in zip(names, patch_latents) if len(f)}
    return float(nn_same_recording_share(feats, exclude_radius=1)) if len(feats) > 1 else None


def diagnose_arrays(names, preds, gts, patch_latents, dataset: str) -> dict:
    res = code_association(names, preds, gts, dataset)
    res["latent_same_recording_nn_share"] = latent_neighbours(names, patch_latents)
    return res


def diagnose_unit(unit: C.Unit, predictions_npz: Path, identity: str | None = None) -> dict:
    with np.load(predictions_npz, allow_pickle=True) as z:
        names, preds, gts, lats = list(z["names"]), list(z["pred"]), list(z["gt"]), list(z["patch_latents"])
    res = diagnose_arrays(names, preds, gts, lats, unit.dataset)
    res.update(unit_id=unit.unit_id, identity=identity)
    return res
