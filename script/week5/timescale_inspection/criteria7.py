"""Phase 7, Stage 0: label-free span criteria, as declared in PHASE7_LABEL_FREE_SCALE.md (K1, K3, K4 and two constant
references), and the gate that scores them against the spans F1@50 preferred.

Nothing here reads a ground-truth label, except `acceptable_spans`, which is the gate itself (it defines which picks
count as right). Every criterion is a pure function of predictions or token features, so it can be tested on
synthetic data.
"""

from __future__ import annotations

import itertools
import statistics

import numpy as np
from scipy.optimize import linear_sum_assignment

BAR = 3.0
AUTOCORR_LEVEL = 0.5


# -- K1: cross-seed stability ------------------------------------------------------------------------------------
def hungarian_relabel(src: list[np.ndarray], ref: list[np.ndarray]) -> dict[int, int]:
    """Map `src`'s state labels onto `ref`'s by maximum frame co-occurrence, pooled over recordings.
    Labels of `src` with no partner keep a fresh label that matches nothing in `ref`."""
    a, b = np.concatenate(src).astype(np.int64), np.concatenate(ref).astype(np.int64)
    la, lb = np.unique(a), np.unique(b)
    ia = {v: i for i, v in enumerate(la)}
    ib = {v: i for i, v in enumerate(lb)}
    co = np.zeros((len(la), len(lb)), dtype=np.int64)
    np.add.at(co, (np.vectorize(ia.get)(a), np.vectorize(ib.get)(b)), 1)
    rows, cols = linear_sum_assignment(-co)
    mapping = {int(la[r]): int(lb[c]) for r, c in zip(rows, cols)}
    fresh = int(max(lb.max(), la.max())) + 1
    for v in la:
        if int(v) not in mapping:
            mapping[int(v)] = fresh
            fresh += 1
    return mapping


def segmental_f1(pred: list[np.ndarray], ref: list[np.ndarray], overlap: float = 0.5) -> float:
    """Pooled segmental F1 at `overlap`, using the repository's own `f_score` (the same code that scores every
    result here), with `ref` in the role of the reference."""
    from src.model.eval_utils import f_score
    tp = fp = fn = 0.0
    for p, r in zip(pred, ref):
        a, b, c = f_score(list(p), list(r), overlap, bg_class=[])
        tp, fp, fn = tp + a, fp + b, fn + c
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 100.0 * (2 * prec * rec / (prec + rec)) if prec + rec else 0.0


def pair_agreement(seg_a: list[np.ndarray], seg_b: list[np.ndarray]) -> float:
    """Symmetric agreement between two segmentations of the same recordings: map B onto A and score with A as the
    reference, map A onto B and score with B as the reference, then average."""
    if [len(x) for x in seg_a] != [len(x) for x in seg_b]:
        raise ValueError("segmentations must cover the same recordings with the same frame counts")
    m_ba = hungarian_relabel(seg_b, seg_a)
    m_ab = hungarian_relabel(seg_a, seg_b)
    f1 = segmental_f1([np.vectorize(m_ba.get)(x) for x in seg_b], seg_a)
    f2 = segmental_f1([np.vectorize(m_ab.get)(x) for x in seg_a], seg_b)
    return (f1 + f2) / 2


def k1_stability(preds_by_seed: dict[int, dict[str, np.ndarray]]) -> float:
    """Mean pairwise agreement over vocabulary seeds, on the recordings every seed has."""
    seeds = sorted(preds_by_seed)
    if len(seeds) < 2:
        return float("nan")
    names = sorted(set.intersection(*(set(preds_by_seed[s]) for s in seeds)))
    vals = [pair_agreement([preds_by_seed[a][n] for n in names], [preds_by_seed[b][n] for n in names])
            for a, b in itertools.combinations(seeds, 2)]
    return float(np.mean(vals))


# -- K3: self-consistency ----------------------------------------------------------------------------------------
def run_lengths(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if len(x) == 0:
        return np.zeros(0, dtype=np.int64)
    cuts = np.flatnonzero(np.diff(x) != 0) + 1
    return np.diff(np.concatenate([[0], cuts, [len(x)]]))


def median_pred_duration(preds_by_seed: dict[int, dict[str, np.ndarray]], fps: int) -> float:
    lens = np.concatenate([run_lengths(p) for by in preds_by_seed.values() for p in by.values()])
    return float(np.median(lens)) / fps


# -- K4: feature time scale --------------------------------------------------------------------------------------
def decorrelation_lag(x: np.ndarray, level: float = AUTOCORR_LEVEL) -> float | None:
    """First lag (in tokens) at which the mean over dimensions of the autocorrelation of `x` (P x d, already
    standardized) drops below `level`. None if it never does within half the recording."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    var = (x * x).mean(axis=0)
    keep = var > 1e-12
    if x.shape[0] < 4 or not keep.any():
        return None
    x, var = x[:, keep], var[keep]
    for lag in range(1, x.shape[0] // 2 + 1):
        r = float(np.mean((x[:-lag] * x[lag:]).mean(axis=0) / var))
        if r < level:
            return float(lag)
    return None


def feature_tau_seconds(features_by_name: dict[str, np.ndarray], window: int, fps: int) -> float:
    """Median over recordings of the decorrelation lag, in seconds. Features are standardized per dimension over
    the whole dataset first (the 'standardized continuous token features' of the plan)."""
    allx = np.concatenate(list(features_by_name.values())).astype(np.float64)
    mu, sd = allx.mean(axis=0), allx.std(axis=0)
    sd[sd < 1e-12] = 1.0
    lags = [decorrelation_lag((x - mu) / sd) for x in features_by_name.values()]
    lags = [l for l in lags if l is not None]
    if not lags:
        return float("nan")
    return float(np.median(lags)) * window / fps


# -- picks and the gate ------------------------------------------------------------------------------------------
def mask_seconds(span: int, window: int, fps: int) -> float:
    return span * window / fps


def pick_k1(scores: dict[int, float]) -> int:
    valid = {s: v for s, v in scores.items() if not np.isnan(v)}
    return max(valid, key=lambda s: (valid[s], -s))  # ties go to the shorter span (declared: deterministic)


def pick_largest_fitting(spans: list[int], limit_by_span: dict[int, float] | float, window: int, fps: int) -> int:
    """Largest span whose mask duration is <= its limit; the smallest span if none fits (K3 and K4)."""
    ok = [s for s in sorted(spans)
          if mask_seconds(s, window, fps) <= (limit_by_span[s] if isinstance(limit_by_span, dict) else limit_by_span)]
    return max(ok) if ok else min(spans)


def acceptable_spans(f1_by_span: dict[int, float], bar: float = BAR) -> set[int]:
    best = max(f1_by_span.values())
    return {s for s, v in f1_by_span.items() if v >= best - bar}


def gate(picks_by_dataset: dict[str, int], acceptable_by_dataset: dict[str, set[int]]) -> dict:
    per = {d: picks_by_dataset[d] in acceptable_by_dataset[d] for d in picks_by_dataset}
    return dict(per_dataset=per, passed=bool(per) and all(per.values()), n_right=sum(per.values()), n=len(per))


def mean(values) -> float:
    return statistics.mean(values)


# -- diagnostics that the selectors themselves cannot see ------------------------------------------------------------
def occupancy(preds_by_seed: dict[int, dict[str, np.ndarray]]) -> dict:
    """Collapse and granularity diagnostics. K1 gives a perfect score to identical collapsed segmentations (one state
    everywhere), so every K1 value is reported next to these."""
    per_rec_states = [len(np.unique(p)) for by in preds_by_seed.values() for p in by.values()]
    per_rec_runs = [len(run_lengths(p)) for by in preds_by_seed.values() for p in by.values()]
    dataset_states = [len(np.unique(np.concatenate(list(by.values())))) for by in preds_by_seed.values()]
    return dict(mean_states_per_recording=float(np.mean(per_rec_states)),
                share_single_state_recordings=float(np.mean([n == 1 for n in per_rec_states])),
                mean_runs_per_recording=float(np.mean(per_rec_runs)),
                states_used_dataset=float(np.mean(dataset_states)))


def feature_tau_coverage(features_by_name: dict[str, np.ndarray], window: int, fps: int) -> dict:
    """K4's estimate together with how many recordings it is defined on (undefined: the autocorrelation never drops
    below the level within half the recording). Coverage is reported, never hidden."""
    allx = np.concatenate(list(features_by_name.values())).astype(np.float64)
    mu, sd = allx.mean(axis=0), allx.std(axis=0)
    sd[sd < 1e-12] = 1.0
    lags = [decorrelation_lag((x - mu) / sd) for x in features_by_name.values()]
    defined = [l for l in lags if l is not None]
    return dict(tau_s=float(np.median(defined)) * window / fps if defined else float("nan"),
                n_defined=len(defined), n_total=len(lags))


# -- the declared gate, with incompleteness made explicit ------------------------------------------------------------
def gate_declared(picks: dict[str, int | None], acceptable: dict[str, set[int]], complete: dict[str, bool]) -> dict:
    """picks / acceptable for the complete datasets; `complete` lists every DECLARED dataset.
    fail       : the pick is wrong (or undefined) on any complete dataset -- established even if others are missing;
    incomplete : no failure, but some declared dataset is incomplete -- never reported as a pass;
    pass       : right on every declared dataset, all complete."""
    per = {}
    for d, ok in complete.items():
        if not ok:
            per[d] = None
        else:
            per[d] = picks.get(d) is not None and picks[d] in acceptable[d]
    status = "fail" if any(v is False for v in per.values()) else ("incomplete" if any(v is None for v in per.values()) else "pass")
    return dict(per_dataset=per, status=status, n_right=sum(1 for v in per.values() if v), n_complete=sum(complete.values()),
                n_declared=len(complete))
