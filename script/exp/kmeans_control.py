"""
T1 control: mini-batch k-means on flattened raw-feature patches, at every K
and seed the SMQ runs use, put through the identical readouts (analyze_run.py).
This separates "a larger codebook helps" from "a *learned* larger codebook helps".

Raw features are z-scored per channel over the dataset (IMU / mocap channels
have very different scales). Patches follow SMQ's layout: ceil(T/W) patches per
sequence, the last partial patch edge-padded. Each fit produces a dump-like
record (codes, centroids as the codebook, the same probe subset) so
analyze_run.analyze scores it exactly like an SMQ dump; the latent probe is
replaced by a probe on patch-mean raw features, computed once per dataset.

Outputs: results/exp/analysis/T1_kmeans/<ds>/K<K>_s<seed>.json
"""
import json
import os
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from script.exp.analyze_run import analyze, probe
from src.model.eval_utils import read_mapping_file

SEEDS = [1538574472, 111, 222]
C = {"hugadb": 10, "lara": 8, "babel1": 5}
W_OF = {"hugadb": 60, "lara": 50, "babel1": 30}
OUT = Path("results/exp/analysis/T1_kmeans")
_cache = {}


class Dump(dict):
    """dict with the .files attribute analyze() checks for optional keys."""
    @property
    def files(self):
        return list(self.keys())


def load_patches(ds, probe_seqs=250):
    if ds in _cache:
        return _cache[ds]
    root = Path("data") / ds
    W = W_OF[ds]
    mapping = read_mapping_file(root / "mapping" / "mapping.txt")
    actions = {v: k for k, v in mapping.items()}
    vids = sorted(os.listdir(root / "features"))
    seqs, gts = [], []
    for v in vids:
        a = np.load(root / "features" / v)                     # (C, T, V, M)
        seqs.append(a[..., 0].transpose(1, 0, 2).reshape(a.shape[1], -1).astype(np.float32))
        gts.append(np.array([actions[x] for x in
                             (root / "groundTruth" / v.replace(".npy", ".txt")).read_text().splitlines()]))
    allf = np.concatenate(seqs)
    mu, sd = allf.mean(0), allf.std(0) + 1e-6
    del allf

    flat, counts, probe_lab, probe_grp, probe_mean = [], [], [], [], []
    probe_idx = set(np.linspace(0, len(vids) - 1, min(probe_seqs, len(vids))).astype(int).tolist())
    for i, (x, g) in enumerate(zip(seqs, gts)):
        x = (x - mu) / sd
        T = len(x)
        P = int(np.ceil(T / W))
        xp = np.concatenate([x, np.repeat(x[-1:], P * W - T, axis=0)]).reshape(P, W, -1)
        flat.append(xp.reshape(P, -1))
        counts.append(P)
        if i in probe_idx:
            probe_mean.append(xp.mean(1))
            lab = np.empty(P, np.int16)
            for p in range(P):
                v, c = np.unique(g[p * W:(p + 1) * W], return_counts=True)
                lab[p] = v[np.argmax(c)]
            probe_lab.append(lab)
            probe_grp.append(np.full(P, i, np.int32))
    data = dict(vids=vids, gts=gts, X=np.concatenate(flat), counts=np.array(counts), W=W,
                F=seqs[0].shape[1], probe_lab=np.concatenate(probe_lab),
                probe_grp=np.concatenate(probe_grp), probe_mean=np.concatenate(probe_mean))
    _cache.clear()
    _cache[ds] = data
    return data


def pending():
    for ds in ["hugadb", "babel1", "lara"]:
        for mult in [1, 2, 4, 8, 16]:
            for seed in SEEDS:
                out = OUT / ds / f"K{C[ds] * mult}_s{seed}.json"
                if not out.exists():
                    yield ds, C[ds] * mult, seed, out


def run_next(log):
    item = next(pending(), None)
    if item is None:
        return False
    ds, K, seed, out = item
    d = load_patches(ds)
    raw_probe_file = OUT / ds / "raw_patch_mean_probe.json"
    if not raw_probe_file.exists():
        raw_probe_file.parent.mkdir(parents=True, exist_ok=True)
        raw_probe_file.write_text(json.dumps(
            {"probe_raw_patch_mean": probe(d["probe_mean"], d["probe_lab"].astype(np.int64), d["probe_grp"])}))

    km = MiniBatchKMeans(n_clusters=K, batch_size=4096, n_init=3, max_iter=100,
                         random_state=seed).fit(d["X"])
    labels = km.labels_.astype(np.int16)
    codes = np.split(labels, np.cumsum(d["counts"])[:-1])
    dump = Dump(dataset=ds, ckpt=f"kmeans_raw_K{K}_s{seed}", K=K, W=d["W"], names=np.array(d["vids"]),
                gt=d["gts"], codes=codes,
                codebook=km.cluster_centers_.reshape(K, d["W"], d["F"]),
                probe_label=d["probe_lab"], probe_group=d["probe_grp"])
    res = analyze(dump, rate=1, skip_latent_probe=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    log(f"kmeans control {ds} K={K} seed={seed} done")
    return True
