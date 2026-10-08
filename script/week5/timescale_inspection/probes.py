"""Phase 0.4: action information per token, by representation, plus the
masked-prediction loss split by whether the masked span crosses a ground-truth
boundary. Label-aware diagnostics only; nothing here selects anything.

Representations of the same complete windows:
  pca      the patch's PCA-64 feature (the tokenizer's continuous input)
  centre   the K-means centre of the patch's code
  onehot   the code itself (linear probe on one-hot; the "nn" column is the
           per-code majority label learned on the train groups)
  context  the contextual embedding from the selected checkpoint (saved by rkx_v2)

Split: groups (subjects where the dataset has them, else recordings) are split
70/30 with a fixed seed, identical for every representation and vocabulary seed.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from script.exp2round.q1q2 import core
from script.quantized_input_comparison import contextual as C
from script.week5.raw_kmeans_cross_dataset import datasets as D
from script.week5.raw_kmeans_cross_dataset.prepare import FAMILY

TRAIN_CAP, TEST_CAP, SPLIT_SEED = 20_000, 10_000, 0


def group_of(spec: D.DatasetSpec, name: str) -> str:
    s = D.subject_of(spec, name) if spec.has_subject else None
    return s if s is not None else name


def split_groups(groups: list[str]) -> set[str]:
    """The 30% test groups, chosen by a seeded hash order (stable across processes)."""
    uniq = sorted(set(groups), key=lambda g: hashlib.sha256(f"{SPLIT_SEED}:{g}".encode()).hexdigest())
    return set(uniq[:max(1, round(0.3 * len(uniq)))])


def token_table(spec: D.DatasetSpec, loader, gts: dict[str, np.ndarray], emb: dict[str, np.ndarray] | None,
                family: str = FAMILY):
    """Complete windows only. Returns dict of arrays (pca, centre, codes, context, label, test)."""
    lookup = loader.lookup(family)
    names = sorted(gts)
    test_groups = split_groups([group_of(spec, n) for n in names])
    cols = {k: [] for k in ("pca", "codes", "context", "label", "test")}
    for n in names:
        rec = loader.recording(family, n)
        keep = rec["lengths"] == spec.window
        if not keep.any():
            continue
        if rec["continuous"].shape[1] != lookup.shape[1]:
            raise AssertionError("continuous feature is not the PCA space of the codebook")
        cols["pca"].append(rec["continuous"][keep])
        cols["codes"].append(rec["codes"][keep])
        if emb is not None:
            cols["context"].append(emb[n][keep])
        gt = gts[n]
        cols["label"].append(np.asarray([core.majority(gt[s:s + spec.window]) for s in rec["starts"][keep]]))
        cols["test"].append(np.full(int(keep.sum()), group_of(spec, n) in test_groups))
    out = {k: np.concatenate(v) for k, v in cols.items() if v}
    out["centre"] = lookup[out["codes"]]
    return out


def _subsample(idx: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(rng.choice(idx, cap, replace=False)) if len(idx) > cap else idx


def run_probes(table: dict) -> list[dict]:
    rng = np.random.default_rng(SPLIT_SEED)
    tr = _subsample(np.flatnonzero(~table["test"]), TRAIN_CAP, rng)
    te = _subsample(np.flatnonzero(table["test"]), TEST_CAP, rng)
    y = table["label"]
    num_codes = int(table["codes"].max()) + 1
    rows = []
    for rep in ("pca", "centre", "onehot", "context"):
        if rep == "context" and "context" not in table:
            continue
        if rep == "onehot":
            X = np.zeros((len(y), num_codes), dtype=np.float32)
            X[np.arange(len(y)), table["codes"]] = 1.0
        else:
            X = table[rep].astype(np.float32)
        Xs = StandardScaler().fit(X[tr]).transform(X) if rep != "onehot" else X
        lin = LogisticRegression(max_iter=500).fit(Xs[tr], y[tr]).predict(Xs[te])
        if rep == "onehot":
            fallback = core.majority(y[tr])
            per_code = {int(c): core.majority(y[tr][table["codes"][tr] == c]) for c in np.unique(table["codes"][tr])}
            nn = np.asarray([per_code.get(int(c), fallback) for c in table["codes"][te]])
        else:
            nn = KNeighborsClassifier(1, metric="cosine", algorithm="brute").fit(X[tr], y[tr]).predict(X[te])
        for probe, pred in (("linear", lin), ("nn", nn)):
            rows.append(dict(representation=rep, probe=probe, n_train=len(tr), n_test=len(te),
                             accuracy=float(accuracy_score(y[te], pred)),
                             balanced_accuracy=float(balanced_accuracy_score(y[te], pred)),
                             macro_f1=float(f1_score(y[te], pred, average="macro"))))
    return rows


@torch.no_grad()
def masked_loss_split(model, loader, val_names: list[str], gts: dict[str, np.ndarray], *, context: int, stride: int,
                      fraction: float, span: int, mask_base: int, device, family: str = FAMILY,
                      radius: int = 0) -> dict:
    """Validation masked-prediction loss with the run's own validation masks (guard 0), split by
    whether the contiguous masked run containing a target spans a ground-truth label change.
    Weighted by real frames per window, as the training validation loss."""
    chunks = C.make_vector_chunks(loader, family, val_names, chunk=context, stride=stride)
    masks = C.chunk_masks(chunks, mask_base, fraction, span, radius=radius)
    acc = {k: [0.0, 0.0, 0.0, 0] for k in ("crossing", "within")}  # loss*w, w, correct*w, targets
    for b in range(0, len(chunks), 64):
        part, pm = chunks[b:b + 64], masks[b:b + 64]
        inp, tgt, w, elig, pad = C._encode_id(part, pm, model.mask_id, device)
        _, logits = model.forward(inp, pad)
        ce = F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none").cpu().numpy()
        hit = (logits.argmax(-1) == tgt).cpu().numpy()
        wn = w.cpu().numpy()
        for i, (chunk, (selected, eligible)) in enumerate(zip(part, pm)):
            gt = gts[chunk.recording]
            pos = 0
            sel = np.asarray(selected[:chunk.valid_len], dtype=bool)
            while pos < len(sel):
                if not sel[pos]:
                    pos += 1
                    continue
                end = pos
                while end + 1 < len(sel) and sel[end + 1]:
                    end += 1
                f0 = int(chunk.starts[pos])
                f1 = int(chunk.starts[end]) + int(chunk.lengths[end])
                key = "crossing" if len(np.unique(gt[f0:f1])) > 1 else "within"
                for t in range(pos, end + 1):
                    if eligible[t]:
                        a = acc[key]
                        a[0] += ce[i, t] * wn[i, t]
                        a[1] += wn[i, t]
                        a[2] += hit[i, t] * wn[i, t]
                        a[3] += 1
                pos = end + 1
    total_w = acc["crossing"][1] + acc["within"][1]
    out = dict(val_loss_all=(acc["crossing"][0] + acc["within"][0]) / max(total_w, 1e-12),
               share_targets_crossing=acc["crossing"][3] / max(acc["crossing"][3] + acc["within"][3], 1))
    for k, (lw, w, cw, n) in acc.items():
        out[f"{k}_targets"] = n
        out[f"{k}_loss"] = lw / w if w else float("nan")
        out[f"{k}_top1"] = cw / w if w else float("nan")
    return out


def run_white_nn(table: dict) -> list[dict]:
    """Phase 5's third probe: 1-NN (cosine) on the PCA feature after per-coordinate standardization fit on the train
    groups -- the metric a whitened K-means would use. Same split, caps and labels as `run_probes`; only the `pca`
    representation, labelled `nn_white`."""
    rng = np.random.default_rng(SPLIT_SEED)
    tr = _subsample(np.flatnonzero(~table["test"]), TRAIN_CAP, rng)
    te = _subsample(np.flatnonzero(table["test"]), TEST_CAP, rng)
    y = table["label"]
    X = table["pca"].astype(np.float32)
    Xs = StandardScaler().fit(X[tr]).transform(X)
    pred = KNeighborsClassifier(1, metric="cosine", algorithm="brute").fit(Xs[tr], y[tr]).predict(Xs[te])
    return [dict(representation="pca", probe="nn_white", n_train=len(tr), n_test=len(te),
                 accuracy=float(accuracy_score(y[te], pred)), balanced_accuracy=float(balanced_accuracy_score(y[te], pred)),
                 macro_f1=float(f1_score(y[te], pred, average="macro")))]
