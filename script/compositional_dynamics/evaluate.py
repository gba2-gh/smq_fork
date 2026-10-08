"""The only module that reads action labels: scoring and post-fit diagnostics. Nothing here feeds back into fitting."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

from script.action_transport import evaluate as at_evaluate
from script.compositional_dynamics import config as C
from script.compositional_dynamics import data as Dm
from script.repro.d_error_decomposition import rle


def load_labels(names: list) -> dict:
    from script.week5.raw_kmeans_cross_dataset import datasets as D
    return D.load_gts(D.SPECS[C.DATASET], names)


def frame_predictions(states: dict, pop: Dm.Population, starts: dict, frame_counts: dict) -> dict:
    return {n: at_evaluate.broadcast_to_frames(states[n], starts[n], pop.lengths[pop.slice_of(r)], frame_counts[n])
            for r, n in enumerate(pop.names)}


def score_states(states: dict, pop: Dm.Population, gts: dict, starts: dict, frame_counts: dict) -> dict:
    """One Hungarian mapping fitted globally over THIS population's recordings (never per recording)."""
    preds = frame_predictions(states, pop, starts, frame_counts)
    names = pop.names
    g, p = [gts[n] for n in names], [preds[n] for n in names]
    metrics, mappings = at_evaluate.score_pooled(g, p)
    mapping = mappings["all"]
    mapped = [np.array([mapping.get(int(v), -1) for v in seq], dtype=np.int32) for seq in p]
    gt_all, mp_all = np.concatenate(g), np.concatenate(mapped)
    classes = np.unique(gt_all)
    recall = {int(c): float(np.mean(mp_all[gt_all == c] == c)) for c in classes}
    support = {int(c): int(np.sum(gt_all == c)) for c in classes}
    raw_all = np.concatenate(p)
    states_u = np.unique(raw_all)
    confusion = {int(s): {int(c): int(np.sum((raw_all == s) & (gt_all == c))) for c in classes} for s in states_u}
    pred_runs = sum(len(rle(m)[0]) for m in p)
    gt_runs = sum(len(rle(x)[0]) for x in g)
    return dict(metrics={k: float(v) for k, v in metrics.items()}, mapping={str(k): int(v) for k, v in mapping.items()},
                per_class_recall=recall, support=support, confusion=confusion, pred_runs=int(pred_runs), gt_runs=int(gt_runs),
                states_used=int(len(states_u)))


def agreement(a: dict, b: dict, pop: Dm.Population) -> float:
    """Frame-weighted share of windows with the same state id under a common numbering (no remapping)."""
    num = den = 0.0
    for r, n in enumerate(pop.names):
        w = pop.lengths[pop.slice_of(r)]
        num += float(w[a[n] == b[n]].sum())
        den += float(w.sum())
    return num / den


def association(states: dict, pop: Dm.Population, gts: dict, starts: dict, frame_counts: dict) -> dict:
    """MI (nats) and arithmetic NMI of the frame-level assignment with recording, subject and action."""
    preds = frame_predictions(states, pop, starts, frame_counts)
    pred = np.concatenate([preds[n] for n in pop.names])
    rec = np.concatenate([np.full(len(preds[n]), i) for i, n in enumerate(pop.names)])
    subj_ids = {s: i for i, s in enumerate(sorted({Dm.subject_of(n) for n in pop.names}))}
    subj = np.concatenate([np.full(len(preds[n]), subj_ids[Dm.subject_of(n)]) for n in pop.names])
    act = np.concatenate([gts[n] for n in pop.names])
    out = {}
    for key, other in (("recording", rec), ("subject", subj), ("action", act)):
        out[key] = dict(mi=float(mutual_info_score(pred, other)),
                        nmi=float(normalized_mutual_info_score(pred, other, average_method="arithmetic")))
    return out


def assigned_nll(states: dict, pop: Dm.Population, occ: np.ndarray, cond: np.ndarray | None) -> dict:
    """Mean per-frame NLL of each window's code under its decoded state (every window once, length weighted)."""
    idx = np.concatenate([states[n] for n in pop.names])
    rows = np.arange(pop.n)
    w = pop.lengths / pop.lengths.sum()
    out = dict(occupancy_nll=float((w * occ[rows, idx]).sum()))
    if cond is not None:
        out["conditional_nll"] = float((w * cond[rows, idx]).sum())
    return out
