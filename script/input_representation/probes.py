"""Label-using diagnostics on a prepared input. They never feed any decoder or selection of a decoder setting.

- action probe (upper bound, subject-grouped): multinomial logistic regression from the window's continuous PCA-64
  feature to its majority action label, trained on 3 of the 4 subject folds and tested on the held-out fold;
  plus a code-table probe (majority label per K-means code) as the ceiling for the discrete units.
  Reported as length-weighted accuracy against the window's majority label, averaged over the 4 folds.
- subject probe (identity left in the feature, metadata only): the same classifier predicting the subject, trained on a
  random half of the recordings and tested on the other half (recording-disjoint, same subjects)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from script.compositional_dynamics import data as Dm
from script.input_representation import config as C
from script.quantized_input_comparison.artifacts import Loader


def window_majority(gt: np.ndarray, starts: np.ndarray, lengths: np.ndarray, n_classes: int) -> np.ndarray:
    onehot = np.zeros((len(gt) + 1, n_classes), dtype=np.int32)
    onehot[1:] = np.cumsum(np.eye(n_classes, dtype=np.int32)[gt], axis=0)
    ends = starts + lengths
    return (onehot[ends] - onehot[starts]).argmax(axis=1)


def load_windows(prep_dir, family: str, names=None):
    """Per-window arrays over all recordings: features, codes, lengths, majority label, recording index, subject."""
    from script.week5.raw_kmeans_cross_dataset import datasets as D
    loader = Loader(prep_dir)
    names = sorted(loader.recording_names()) if names is None else names
    spec = D.SPECS[C.DATASET]
    gts = D.load_gts(spec, names)
    n_classes = int(max(g.max() for g in gts.values())) + 1
    X, codes, lens, y, rec = [], [], [], [], []
    for i, n in enumerate(names):
        r = loader.recording(family, n)
        X.append(r["continuous"])
        codes.append(r["codes"])
        lens.append(r["lengths"])
        y.append(window_majority(gts[n], r["starts"].astype(np.int64), r["lengths"].astype(np.int64), n_classes))
        rec.append(np.full(len(r["codes"]), i))
    subj = np.array([Dm.subject_of(n) for n in names])
    return dict(names=names, X=np.concatenate(X), codes=np.concatenate(codes).astype(np.int64), lengths=np.concatenate(lens).astype(np.float64),
                y=np.concatenate(y), rec=np.concatenate(rec), subject=subj, n_classes=n_classes)


def _fit_predict(Xtr, ytr, Xte, cap: int, seed: int):
    rng = np.random.default_rng(seed)
    if len(Xtr) > cap:
        idx = rng.choice(len(Xtr), cap, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C.LR_C, max_iter=C.LR_MAX_ITER)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict(sc.transform(Xte))


def _weighted_acc(pred, y, w) -> float:
    return float(100.0 * (w * (pred == y)).sum() / w.sum())


def run_probes(w: dict, cap: int = C.PROBE_CAP) -> dict:
    names = w["names"]
    folds = Dm.subject_folds(names)
    subj_of_win = w["subject"][w["rec"]]
    act_lr, act_code, majority = [], [], []
    for k, held in enumerate(folds):
        test = np.isin(subj_of_win, held)
        tr = ~test
        pred = _fit_predict(w["X"][tr], w["y"][tr], w["X"][test], cap, C.PROBE_SEED + k)
        act_lr.append(_weighted_acc(pred, w["y"][test], w["lengths"][test]))
        table = np.zeros((512, w["n_classes"]))
        np.add.at(table, (w["codes"][tr], w["y"][tr]), w["lengths"][tr])
        glob = np.bincount(w["y"][tr], weights=w["lengths"][tr], minlength=w["n_classes"]).argmax()
        lookup = np.where(table.sum(1) > 0, table.argmax(1), glob)
        act_code.append(_weighted_acc(lookup[w["codes"][test]], w["y"][test], w["lengths"][test]))
        majority.append(_weighted_acc(np.full(test.sum(), glob), w["y"][test], w["lengths"][test]))
    # subject probe: recording-disjoint halves
    rng = np.random.default_rng(C.PROBE_SEED)
    order = rng.permutation(len(names))
    train_recs = set(order[: len(names) // 2].tolist())
    tr = np.isin(w["rec"], list(train_recs))
    subj_ids = {s: i for i, s in enumerate(sorted(set(w["subject"])))}
    ysub = np.array([subj_ids[s] for s in subj_of_win])
    pred = _fit_predict(w["X"][tr], ysub[tr], w["X"][~tr], cap, C.PROBE_SEED)
    sub_acc = _weighted_acc(pred, ysub[~tr], w["lengths"][~tr])
    chance = float(100.0 * np.bincount(ysub[~tr], weights=w["lengths"][~tr]).max() / w["lengths"][~tr].sum())
    return dict(action_probe_continuous=float(np.mean(act_lr)), action_probe_continuous_per_fold=act_lr,
                action_probe_code_table=float(np.mean(act_code)), majority_baseline=float(np.mean(majority)),
                subject_probe_accuracy=sub_acc, subject_probe_chance=chance, probe_cap=cap)
