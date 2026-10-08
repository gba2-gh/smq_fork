"""Phase 0.3: failure profile of saved `rkx_v2` predictions (label-aware diagnostics only).

Predictions are mapped with the same dataset-level Hungarian mapping the run used,
then described: segment lengths, per-class results, background (`none`) errors,
clip-length strata, and an error type for every ground-truth segment.
"""

from __future__ import annotations

import numpy as np

from script.exp2round.q1q2 import core
from script.repro.d_error_decomposition import rle
from script.repro.d_error_decomposition import score as score_frames

LENGTH_BINS = ((0.0, 1.0, "<1s"), (1.0, 2.0, "1-2s"), (2.0, float("inf"), ">=2s"))


def map_all(gts: list[np.ndarray], preds: list[np.ndarray]) -> list[np.ndarray]:
    return core.map_predictions(gts, preds, "hungarian")


def runs(a: np.ndarray) -> list[tuple[int, int, int]]:
    """(start, end, label) of every maximal run."""
    lengths, labels, starts = rle(a)
    return [(int(s), int(s + l), int(v)) for l, v, s in zip(lengths, labels, starts)]


def segment_length_stats(seqs: list[np.ndarray], fps: int) -> dict:
    lengths = np.concatenate([rle(s)[0] for s in seqs]) / fps
    return dict(segments_per_rec=float(len(lengths) / len(seqs)), median_s=float(np.median(lengths)),
                mean_s=float(lengths.mean()), share_under_1s=float(np.mean(lengths < 1.0)))


def classify_segments(gt: np.ndarray, pm: np.ndarray, fps: int) -> list[dict]:
    """One row per ground-truth segment: hit (a same-label predicted run with IoU >= 0.5),
    else by the majority mapped label over its frames: `boundary` (own label, wrong extent),
    `merged` (a neighbouring segment's label), `wrong_label` (anything else)."""
    gruns, pruns = runs(gt), runs(pm)
    out = []
    for i, (s, e, c) in enumerate(gruns):
        hit = False
        for ps, pe, pc in pruns:
            if pe <= s:
                continue
            if ps >= e:
                break
            if pc == c:
                inter = min(e, pe) - max(s, ps)
                union = max(e, pe) - min(s, ps)
                if inter / union >= 0.5:
                    hit = True
                    break
        if hit:
            kind = "hit"
        else:
            m = core.majority(pm[s:e])
            neighbours = {gruns[j][2] for j in (i - 1, i + 1) if 0 <= j < len(gruns)}
            kind = "boundary" if m == c else ("merged" if m in neighbours else "wrong_label")
        out.append(dict(label=c, seconds=(e - s) / fps, kind=kind))
    return out


def length_bin(seconds: float) -> str:
    return next(name for lo, hi, name in LENGTH_BINS if lo <= seconds < hi)


def profile(names: list[str], gts: dict[str, np.ndarray], raw_preds: dict[str, np.ndarray], fps: int,
            class_names: dict[int, str]) -> dict:
    """Everything for one (cell, role). Returns dict of row lists keyed by table."""
    g = [gts[n] for n in names]
    pm = map_all(g, [raw_preds[n] for n in names])
    overall = score_frames(g, pm)
    seg_rows = [r for gt, p in zip(g, pm) for r in classify_segments(gt, p, fps)]

    summary = dict(overall, **{f"gt_{k}": v for k, v in segment_length_stats(g, fps).items()},
                   **{f"pred_{k}": v for k, v in segment_length_stats(pm, fps).items()})
    err = np.concatenate([gt != p for gt, p in zip(g, pm)])
    gt_all, pm_all = np.concatenate(g), np.concatenate(pm)
    none_id = next((k for k, v in class_names.items() if v == "none"), None)
    if none_id is not None:
        wrong = err.sum()
        summary["share_errors_involving_none"] = float(((gt_all == none_id) | (pm_all == none_id))[err].sum() / max(wrong, 1))
        keep = [gt != none_id for gt in g]
        g2 = [gt[k] for gt, k in zip(g, keep) if k.any()]
        p2 = [p[k] for p, k in zip(pm, keep) if k.any()]
        excl = score_frames(g2, p2)
        summary.update({f"no_none_{k}": v for k, v in excl.items()})

    per_class = []
    for c in sorted(np.unique(gt_all)):
        mine = [r for r in seg_rows if r["label"] == c]
        tp = int(((gt_all == c) & (pm_all == c)).sum())
        per_class.append(dict(label=int(c), name=class_names.get(int(c), str(c)), frame_share=float(np.mean(gt_all == c)),
                              frame_recall=tp / max(int((gt_all == c).sum()), 1),
                              frame_precision=tp / max(int((pm_all == c).sum()), 1),
                              segments=len(mine), segment_hit_rate=float(np.mean([r["kind"] == "hit" for r in mine])) if mine else float("nan"),
                              median_segment_s=float(np.median([r["seconds"] for r in mine])) if mine else float("nan")))

    error_types = []
    for _, _, bname in LENGTH_BINS + ((None, None, "all"),):
        mine = seg_rows if bname == "all" else [r for r in seg_rows if length_bin(r["seconds"]) == bname]
        row = dict(length_bin=bname, segments=len(mine))
        for kind in ("hit", "boundary", "merged", "wrong_label"):
            row[kind] = float(np.mean([r["kind"] == kind for r in mine])) if mine else float("nan")
        error_types.append(row)

    lengths = np.asarray([len(x) for x in g])
    which = np.digitize(lengths, np.quantile(lengths, [1 / 3, 2 / 3]), right=True)
    tertiles = []
    for t in range(3):
        idx = list(np.flatnonzero(which == t))
        if not idx:
            continue
        sc = score_frames([g[i] for i in idx], [pm[i] for i in idx])
        tertiles.append(dict(tertile=t + 1, recordings=len(idx), median_length_s=float(np.median(lengths[idx]) / fps), **sc))
    return dict(summary=summary, per_class=per_class, error_types=error_types, tertiles=tertiles)
