"""The only module allowed to receive action labels (instructions §1, §7).
Frame broadcast, Hungarian mapping, the five-metric scorer, and descriptive
diagnostics. Reuses the established scorer (script.exp2round.q1q2.core and
script.repro.d_error_decomposition) rather than reimplementing MoF/Edit/F1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.exp2round.q1q2 import core
from script.repro.d_error_decomposition import rle
from script.repro.d_error_decomposition import score as score_frames


def broadcast_to_frames(states: np.ndarray, starts: np.ndarray, lengths: np.ndarray,
                        frame_count: int) -> np.ndarray:
    """Broadcast each window's state to its real frames. starts/lengths must
    cover [0, frame_count) exactly once."""
    if int(lengths.sum()) != frame_count or int(starts[0]) != 0:
        raise AssertionError("window spans do not cover the recording exactly")
    return np.repeat(np.asarray(states, dtype=np.int32), lengths)


def mapping_of(predictions: list[np.ndarray], mapped: list[np.ndarray]) -> dict[int, int]:
    """Recover the raw-state -> label mapping that core.map_predictions applied."""
    raw = np.concatenate(predictions)
    out = np.concatenate(mapped)
    pairs = np.unique(np.stack([raw, out], axis=1), axis=0)
    return {int(r): int(m) for r, m in pairs}


def score_pooled(gts: list[np.ndarray], predictions: list[np.ndarray]) -> tuple[dict, dict]:
    """One dataset-level Hungarian mapping across every recording."""
    mapped = core.map_predictions(gts, predictions, "hungarian")
    return score_frames(gts, mapped), {"all": mapping_of(predictions, mapped)}


def score_subject_disjoint(fold_of: np.ndarray, gts: list[np.ndarray],
                           predictions: list[np.ndarray]) -> tuple[dict, dict]:
    """One Hungarian mapping per held-out fold (extra oracle flexibility vs
    pooled -- disclosed in the report), then the scorer's own aggregation
    over all out-of-fold recordings in their original order."""
    mapped: list = [None] * len(gts)
    mappings = {}
    for fold in sorted({int(f) for f in fold_of}):
        idx = [i for i, f in enumerate(fold_of) if int(f) == fold]
        fold_preds = [predictions[i] for i in idx]
        fold_mapped = core.map_predictions([gts[i] for i in idx], fold_preds, "hungarian")
        mappings[f"fold{fold}"] = mapping_of(fold_preds, fold_mapped)
        for i, m in zip(idx, fold_mapped):
            mapped[i] = m
    return score_frames(gts, mapped), mappings


def _run_durations(frames: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([rle(f)[0] for f in frames]).astype(np.float64)


def segment_diagnostics(frame_preds: list[np.ndarray], frame_gts: list[np.ndarray],
                        subjects: list[str], fps: int) -> dict:
    """Descriptive diagnostics per prediction set: action-run counts and
    durations (predicted runs merge adjacent equal raw states), state
    occupancy/entropy, and frame-weighted subject-state mutual information.
    Uses labels only to count ground-truth runs for the ratio; subject IDs are
    metadata. Nothing here selects or fits anything."""
    pred_dur = _run_durations(frame_preds) / fps
    gt_dur = _run_durations(frame_gts) / fps
    states = np.concatenate(frame_preds)
    _, counts = np.unique(states, return_counts=True)
    occupancy = counts / counts.sum()
    subject_frames = np.concatenate([np.full(len(f), s) for f, s in zip(frame_preds, subjects)])
    return {
        "pred_runs": int(len(pred_dur)),
        "gt_runs": int(len(gt_dur)),
        "run_ratio": float(len(pred_dur) / max(1, len(gt_dur))),
        "pred_run_median_s": float(np.median(pred_dur)),
        "pred_run_p10_s": float(np.percentile(pred_dur, 10)),
        "pred_run_p90_s": float(np.percentile(pred_dur, 90)),
        "gt_run_median_s": float(np.median(gt_dur)),
        "occupied_states": int(len(counts)),
        "occupancy_entropy": float(-(occupancy * np.log(occupancy)).sum()),
        "max_state_fraction": float(occupancy.max()),
        "mi_state_subject": float(mutual_info_score(subject_frames, states)),
    }
