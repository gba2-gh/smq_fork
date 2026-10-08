"""Phase 0.1-0.2: label-only ceilings.

0.1 Token grid: the ground truth downsampled to a token grid (majority label per
window), broadcast back to frames and scored against the original. This is the
best score reachable if every token were labelled perfectly.

0.2 Smoothing: a centred mode filter of half-width r tokens applied to the
token-grid ground truth. It shows how much true structure a smoothing
neighbourhood of that reach destroys even with perfect evidence. The OT temporal
term is not a mode filter, so this is an indicative bound, not an exact one.

Both use the repository scorer with dataset-level Hungarian matching (for oracle
labels the mapping is the identity; this is checked and recorded).
"""

from __future__ import annotations

import numpy as np

from script.action_transport import evaluate as at_evaluate


def token_labels(gt: np.ndarray, window: int) -> np.ndarray:
    """Majority label per window on the zero-anchored grid (last window may be partial).
    Ties go to the smallest label id, as `q1q2.core.majority`."""
    gt = np.asarray(gt)
    n = int(np.ceil(len(gt) / window))
    classes = np.unique(gt)
    idx = np.searchsorted(classes, gt)
    padded = np.full(n * window, -1, dtype=np.int64)
    padded[:len(gt)] = idx
    grid = padded.reshape(n, window)
    counts = np.stack([(grid == k).sum(axis=1) for k in range(len(classes))], axis=1)
    return classes[np.argmax(counts, axis=1)].astype(np.int32)


def to_frames(tokens: np.ndarray, window: int, frame_count: int) -> np.ndarray:
    return np.repeat(np.asarray(tokens, dtype=np.int32), window)[:frame_count]


def mode_filter(tokens: np.ndarray, radius: int) -> np.ndarray:
    """Centred mode over [t-r, t+r] (clipped at the ends). A tie that includes the
    current label keeps it; otherwise the smallest tied label wins."""
    tokens = np.asarray(tokens)
    if radius <= 0 or len(tokens) <= 1:
        return tokens.copy()
    classes, idx = np.unique(tokens, return_inverse=True)
    onehot = np.zeros((len(tokens) + 1, len(classes)), dtype=np.int64)
    onehot[1:][np.arange(len(tokens)), idx] = 1
    csum = np.cumsum(onehot, axis=0)
    t = np.arange(len(tokens))
    lo, hi = np.clip(t - radius, 0, None), np.clip(t + radius + 1, None, len(tokens))
    counts = csum[hi] - csum[lo]
    best = counts.max(axis=1)
    keep = counts[t, idx] == best
    out = np.where(keep, idx, np.argmax(counts, axis=1))
    return classes[out].astype(tokens.dtype)


def radius_for(rule: str, value: float, num_tokens: int) -> int:
    """`fixed`: value tokens. `relative`: floor(value * N). `relative_floor1`: max(1, floor(value * N))."""
    if rule == "fixed":
        return int(value)
    r = int(np.floor(value * num_tokens))
    return max(1, r) if rule == "relative_floor1" else r


def score_oracle(gts: list[np.ndarray], preds: list[np.ndarray]) -> dict:
    metrics, mappings = at_evaluate.score_pooled(gts, preds)
    mapping = mappings["all"]
    return dict(metrics, mapping_is_identity=all(k == v for k, v in mapping.items()))


def grid_ceiling(gts: dict[str, np.ndarray], window: int) -> dict:
    names = sorted(gts)
    preds = [to_frames(token_labels(gts[n], window), window, len(gts[n])) for n in names]
    return score_oracle([gts[n] for n in names], preds)


def smoothing_ceiling(gts: dict[str, np.ndarray], window: int, rule: str, value: float) -> dict:
    names = sorted(gts)
    preds, radii = [], []
    for n in names:
        tok = token_labels(gts[n], window)
        r = radius_for(rule, value, len(tok))
        radii.append(r)
        preds.append(to_frames(mode_filter(tok, r), window, len(gts[n])))
    out = score_oracle([gts[n] for n in names], preds)
    radii = np.asarray(radii)
    return dict(out, median_radius_tokens=float(np.median(radii)), share_radius_zero=float(np.mean(radii == 0)))
