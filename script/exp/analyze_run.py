"""
Pre-registered readouts (results/PREREG_T1-T4.md) for one dump, CPU only.

  (a)  bijective Hungarian, as published (surplus codes score as wrong)
  (b)  many-to-one: each code -> its majority GT class   [uses GT; control-relative only]
  (c1) geometry merge: average-linkage agglomerative clustering of the codebook
       into C groups under the quantiser's own patch distance, then (a)
  (c2) temporal merge: spectral clustering of the symmetrised code-transition
       count graph into C groups, then (a)
  JSD (report convention) for a, c1, c2; patch-grid oracle for this patch size
  probes: code one-hot and (if present) pre-quantisation latent, 5-fold grouped
  codebook: perplexity, codes unused at evaluation

For downsampled (T4 Sweep R) dumps pass --rate r: JSD is computed on lengths
multiplied by r (native-frame resolution) and durations are also given in s.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model.eval_utils import create_correspondences, edit_score, f_score
from script.repro.e1_jsd import pair_jsd, activity_of, frame_weighted

warnings.filterwarnings("ignore")
OVERLAPS = [.1, .25, .5]
N_CLASSES = {"hugadb": 10, "lara": 8, "babel1": 5}
NATIVE_FPS = {"hugadb": 60, "lara": 50, "babel1": 30}


def rle_lengths(a):
    ch = np.flatnonzero(a[1:] != a[:-1]) + 1
    return np.diff(np.concatenate(([0], ch, [len(a)])))


def score(gts, preds):
    correct = total = 0
    edit = 0.0
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
    for g, p in zip(gts, preds):
        correct += int((g == p).sum()); total += len(g)
        edit += edit_score(p, g)
        for i, o in enumerate(OVERLAPS):
            a, b, c = f_score(p, g, o)
            tp[i] += a; fp[i] += b; fn[i] += c
    out = dict(MoF=100 * correct / total, Edit=edit / len(gts))
    for i, o in enumerate(OVERLAPS):
        pr = tp[i] / (tp[i] + fp[i]) if tp[i] + fp[i] else 0.0
        rc = tp[i] / (tp[i] + fn[i]) if tp[i] + fn[i] else 0.0
        out[f"F1@{int(o * 100)}"] = 100 * (2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return {k: round(v, 3) for k, v in out.items()}


def jsd(gts, preds, names, dataset, rate):
    acc = {}
    for g, p, n in zip(gts, preds, names):
        a = activity_of(dataset, n)
        acc.setdefault(a, {"jsd": [], "frames": 0})
        acc[a]["jsd"].append(pair_jsd(rle_lengths(g) * rate, rle_lengths(p) * rate))
        acc[a]["frames"] += len(g)
    return round(frame_weighted(acc), 3)


def hungarian_map(gts, preds):
    _, pr2gt = create_correspondences(np.concatenate(gts), np.concatenate(preds), mapping=True)
    lut = np.full(int(max(pr2gt)) + 1, -999, dtype=np.int64)
    for k, v in pr2gt.items():
        lut[int(k)] = int(v)
    return [lut[p] for p in preds]


def patch_distance_matrix(cb):
    """Quantiser distance between codebook entries: sum over frames of per-frame L2."""
    K = cb.shape[0]
    D = np.zeros((K, K))
    for i in range(K):
        D[i] = np.sqrt(((cb[i][None] - cb) ** 2).sum(-1)).sum(-1)
    return (D + D.T) / 2


def geometry_merge(cb, C):
    K = cb.shape[0]
    if K <= C:
        return np.arange(K)
    Z = linkage(squareform(patch_distance_matrix(cb), checks=False), method="average")
    return fcluster(Z, t=C, criterion="maxclust") - 1


def temporal_merge(patch_codes, K, C, seed=0):
    if K <= C:
        return np.arange(K)
    A = np.zeros((K, K))
    for c in patch_codes:
        c = np.asarray(c, dtype=np.int64)
        d = c[1:] != c[:-1]
        np.add.at(A, (c[:-1][d], c[1:][d]), 1)
    S = A + A.T + 1e-3
    np.fill_diagonal(S, 0)
    return SpectralClustering(n_clusters=C, affinity="precomputed", random_state=seed,
                              assign_labels="discretize").fit_predict(S)


def probe(X, y, groups):
    if len(np.unique(y)) < 2:
        return float("nan")
    accs = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, n_jobs=1).fit(sc.transform(X[tr]), y[tr])
        accs.append(float((clf.predict(sc.transform(X[te])) == y[te]).mean()))
    return round(100 * float(np.mean(accs)), 3)


def patch_oracle(g, W):
    out = np.empty_like(g)
    for s in range(0, len(g), W):
        v, c = np.unique(g[s:s + W], return_counts=True)
        out[s:s + W] = v[np.argmax(c)]
    return out


def analyze(d, rate=1, skip_latent_probe=False):
    dataset = str(d["dataset"])
    C = N_CLASSES[dataset]
    K, W = int(d["K"]), int(d["W"])
    names = [str(n) for n in d["names"]]
    gts = [np.asarray(g, dtype=np.int64) for g in d["gt"]]
    codes = [np.asarray(c, dtype=np.int64) for c in d["codes"]]
    frames = [np.repeat(c, W)[:len(g)] for c, g in zip(codes, gts)]

    res = dict(dataset=dataset, source=str(d["ckpt"]), K=K, W=W, C=C, rate=rate,
               patch_seconds=W * rate / NATIVE_FPS[dataset])

    # (a) published protocol
    mapped = hungarian_map(gts, frames)
    res["a_bijective"] = score(gts, mapped)
    res["a_bijective"]["JSD"] = jsd(gts, mapped, names, dataset, rate)

    # (b) many-to-one (GT-assisted)
    G, P = np.concatenate(gts), np.concatenate(frames)
    lut = np.zeros(K, dtype=np.int64)
    for k in range(K):
        m = P == k
        if m.any():
            v, c = np.unique(G[m], return_counts=True)
            lut[k] = v[np.argmax(c)]
    res["b_many_to_one"] = score(gts, [lut[f] for f in frames])

    # (c1) geometry merge, (c2) temporal merge -- both label-free
    for key, groups in (("c1_geometry_merge", geometry_merge(np.asarray(d["codebook"], float), C)),
                        ("c2_temporal_merge", temporal_merge(codes, K, C))):
        merged = [groups[f] for f in frames]
        mm = hungarian_map(gts, merged)
        res[key] = score(gts, mm)
        res[key]["JSD"] = jsd(gts, mm, names, dataset, rate)
        res[key]["n_groups"] = int(len(np.unique(groups)))

    # segment statistics, in seconds
    fps = NATIVE_FPS[dataset] / rate
    gl = np.concatenate([rle_lengths(g) for g in gts]) / fps
    pl = np.concatenate([rle_lengths(f) for f in frames]) / fps
    res["durations_s"] = dict(gt_mean=round(float(gl.mean()), 3), gt_median=round(float(np.median(gl)), 3),
                              pred_mean=round(float(pl.mean()), 3), pred_median=round(float(np.median(pl)), 3),
                              segments_pred_over_gt=round(len(pl) / len(gl), 3))

    oracle = [patch_oracle(g, W) for g in gts]
    res["patch_grid_oracle"] = score(gts, oracle)
    res["patch_grid_oracle"]["JSD"] = jsd(gts, oracle, names, dataset, rate)

    # probes (patch level, grouped by sequence)
    y = np.asarray(d["probe_label"], dtype=np.int64)
    grp = np.asarray(d["probe_group"])
    probe_codes = np.concatenate([codes[i] for i in np.unique(grp)])
    res["probe_code_onehot"] = probe(np.eye(K)[probe_codes], y, grp)
    if not skip_latent_probe and "probe_latent" in d.files:
        res["probe_latent"] = probe(np.asarray(d["probe_latent"], dtype=np.float32), y, grp)
    res["nmi_code_vs_gt_patch"] = round(float(normalized_mutual_info_score(y, probe_codes)), 4)

    cnt = np.bincount(np.concatenate(codes), minlength=K).astype(float)
    p = cnt / cnt.sum()
    res["codebook"] = dict(perplexity=round(float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())), 3),
                           unused_codes=int((cnt == 0).sum()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rate", type=int, default=1)
    args = ap.parse_args()
    d = np.load(args.dump, allow_pickle=True)
    res = analyze(d, rate=args.rate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("dataset", "K", "W", "a_bijective", "probe_code_onehot")}))


if __name__ == "__main__":
    main()
