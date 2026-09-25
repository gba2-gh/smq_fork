"""The only module allowed to receive action labels (instructions §1, §7).
Frame broadcast, Hungarian mapping, the five-metric scorer, and label-free
diagnostics (subject-state mutual information). Reuses the established
scorer via script.exp2round.q1q2.core and script.repro.d_error_decomposition
rather than reimplementing MoF/Edit/F1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.exp2round.q1q2 import core
from script.repro.d_error_decomposition import score as score_frames


def broadcast_to_frames(states: np.ndarray, starts: np.ndarray, lengths: np.ndarray,
                        frame_count: int) -> np.ndarray:
    """Broadcast each window's state to its real frames. starts/lengths must
    give exact, non-overlapping coverage of [0, frame_count)."""
    frames = np.empty(frame_count, dtype=np.int32)
    covered = np.zeros(frame_count, dtype=bool)
    for state, start, length in zip(states, starts, lengths):
        if length <= 0:
            continue
        frames[start:start + length] = state
        covered[start:start + length] = True
    if not covered.all():
        missing = int((~covered).sum())
        raise AssertionError(f"{missing} frames not covered by any window")
    return frames


def score_pooled(gts: list[np.ndarray], predictions: list[np.ndarray]) -> dict:
    """One dataset-level Hungarian mapping across every recording."""
    return core.score_mapped(gts, predictions, method="hungarian")


def score_subject_disjoint(fold_of: np.ndarray, gts: list[np.ndarray],
                           predictions: list[np.ndarray]) -> dict:
    """One Hungarian mapping per held-out fold (extra oracle flexibility vs
    pooled -- disclosed in the report), then the scorer's own aggregation
    over all out-of-fold recordings in their original order."""
    mapped = [None] * len(gts)
    for fold in sorted(set(int(f) for f in fold_of)):
        idx = [i for i, f in enumerate(fold_of) if int(f) == fold]
        fold_gts = [gts[i] for i in idx]
        fold_preds = [predictions[i] for i in idx]
        fold_mapped = core.map_predictions(fold_gts, fold_preds, "hungarian")
        for i, m in zip(idx, fold_mapped):
            mapped[i] = m
    return score_frames(gts, mapped)


def subject_state_mi(states_per_recording: list[np.ndarray], subjects: list[str]) -> dict:
    """Frame-weighted subject-by-state mutual information (instructions §7).
    Subject IDs are metadata, used only for this descriptive diagnostic --
    never to fit, select, or score a model."""
    units = np.concatenate(states_per_recording)
    subject_labels = np.concatenate([
        np.full(len(states), subject) for states, subject in zip(states_per_recording, subjects)
    ])
    return {
        "mi_state_subject": float(mutual_info_score(units, subject_labels)),
        "n_windows": int(len(units)),
        "occupied_states": int(len(np.unique(units))),
    }
