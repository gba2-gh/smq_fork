"""
Shared fit / assign / evaluate pipeline. Experiments 1 and 2 both go through
`run_fold`, so that the only thing that differs between conditions is the partition.

Nothing in `fit` sees labels: the scaler and k-means are fitted on segment vectors only.
Labels enter (a) the declared oracle partition, and (b) evaluation.

Setting (named accurately): TRANSDUCTIVE. The frozen released encoder was trained without
labels on ALL recordings of the dataset, including the evaluation recordings of every fold.
A fold split only means that the SCALER and the CODEBOOK (k-means) are fitted without the
evaluation recordings (and, for LARa, without the evaluation participants).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from script.seg import metrics as M  # noqa: E402
from script.seg import partition as P  # noqa: E402
from script.seg import represent as R  # noqa: E402
from script.seg.common import DATASETS, lara_subject, load_gt_labels, read_mapping  # noqa: E402

CACHE = ROOT / "results" / "seg" / "cache"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- prespecified constants (see PROTOCOL.md) ----
N_BINS = 8
PURITY_TAU = 0.8          # segment-level retrieval eligibility (primary); 0.0 = all segments (sensitivity)
KNN_K = 10                # precision@k (k gallery anchors); 50 as sensitivity
KMEANS_SEEDS = (0, 1, 2)
N_RANDOM = 10
RANDOM_SEED_BASE = 1000
BOUND_TOL_S = (0.5, 1.0)
ANCHOR_SEED = 20260921
FOLD_SEED = 0


class Data:
    def __init__(self, ds):
        cfg = DATASETS[ds]
        self.ds, self.fps, self.W = ds, cfg["fps"], cfg["patch"]
        meta = np.load(CACHE / ds / "_meta.npz", allow_pickle=True)
        self.names = [str(n) for n in meta["names"]]
        self.z = [np.load(CACHE / ds / f"{n}.npy") for n in self.names]
        mp = read_mapping(ds)
        ids = sorted(mp)
        self.class_names = [mp[i] for i in ids]
        self.C = len(ids)
        remap = {k: i for i, k in enumerate(ids)}
        a2i = {v: k for k, v in mp.items()}
        self.labels = [np.vectorize(remap.get)(load_gt_labels(ROOT / "data" / ds, n, a2i)) for n in self.names]
        self.T = np.array([len(z) for z in self.z])
        self.smq_codes = [np.asarray(c, dtype=np.int64) for c in meta["codes"]]
        self.codebook = meta["codebook"]
        self.n = len(self.names)
        # retrieval-exclusion group: participant for LARa; recording otherwise (no participant ids)
        if ds == "lara":
            self.group = np.array([int(lara_subject(n)) for n in self.names])
            self.group_kind = "participant"
        else:
            self.group = np.arange(self.n)
            self.group_kind = "recording"
        # background-like label kept as an ordinary class (it is one of the K=C benchmark classes)
        self.none_class = self.class_names.index("None" if ds == "lara" else "none")

    def make_folds(self):
        rng = np.random.default_rng(FOLD_SEED)
        if self.ds == "lara":
            parts = np.array(sorted(set(self.group)))
            perm = rng.permutation(parts)
            nf = 4
            f_of_p = {int(p): i % nf for i, p in enumerate(perm)}
            return np.array([f_of_p[int(g)] for g in self.group]), nf
        nf = 5
        perm = rng.permutation(self.n)
        fold = np.empty(self.n, dtype=int)
        fold[perm] = np.arange(self.n) % nf
        return fold, nf

    def make_anchors(self):
        """Timestamps chosen independently of segmentation and labels: one per second of
        recording, uniformly jittered inside its second (deterministic per recording)."""
        out = []
        for r in range(self.n):
            rng = np.random.default_rng([ANCHOR_SEED, r])
            ns = int(np.ceil(self.T[r] / self.fps))
            t = np.floor((np.arange(ns) + rng.random(ns)) * self.fps).astype(np.int64)
            out.append(t[t < self.T[r]])
        return out


def anchors_hash(anchors, data):
    import hashlib
    h = hashlib.sha256()
    for r, t in enumerate(anchors):
        h.update(np.asarray(t, dtype=np.int64).tobytes())
        h.update(data.labels[r][t].astype(np.int8).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ small helpers
def seg_label_counts(lab, edges, C):
    oh = np.zeros((len(lab) + 1, C))
    oh[1:] = np.cumsum(np.eye(C)[lab], 0)
    return oh[edges[1:]] - oh[edges[:-1]]


def cdist_np(A, B=None):
    A = torch.as_tensor(A, dtype=torch.float32, device=DEV)
    B = A if B is None else torch.as_tensor(B, dtype=torch.float32, device=DEV)
    return torch.cdist(A, B)


def smq_code_distances(codebook):
    """SMQ's own patch distance between codebook entries: sum over frames of per-frame Euclidean."""
    cb = torch.as_tensor(codebook, dtype=torch.float32, device=DEV)         # (K, W, D)
    K = cb.shape[0]
    d = torch.zeros(K, K, device=DEV)
    for i in range(K):
        d[i] = torch.sqrt(((cb[i][None] - cb) ** 2).sum(-1)).sum(-1)
    return d


def _retrieval_arrays(V, Dkk, code, rec_of_item, Qcnt, group_of_item, ev_index, C, K, k):
    """
    Continuous (if V given) and quantised (if Dkk given) retrieval for one item set.
    Items are segments; Qcnt[s, c] = number of anchors (or 1 for the majority label) carried.
    Returns dict mode -> dict(S, n, ch) with per-recording (n_ev, C) arrays.
    """
    n_ev = len(ev_index)
    out = {}
    Qt = torch.as_tensor(Qcnt, dtype=torch.float32, device=DEV)
    gr = torch.as_tensor(group_of_item, device=DEV)
    if V is not None and len(V) > 0:
        D = cdist_np(V)
        Pm, prev, valid = M.precision_at_k(D, Qt, Qt, gr, gr, k)
        v = valid[:, None].double()
        Sq = (Qt.double() * Pm * v).cpu().numpy()
        nq = (Qt.double() * v).cpu().numpy()
        cq = (Qt.double() * prev * v).cpu().numpy()
        out["cont"] = {}
        for name, arr in (("S", Sq), ("n", nq), ("ch", cq)):
            a = np.zeros((n_ev, C)); np.add.at(a, rec_of_item, arr); out["cont"][name] = a
    if Dkk is not None and len(code) > 0:
        N = np.zeros((int(group_of_item.max()) + 1, K, C))
        np.add.at(N, (group_of_item, code), Qcnt)
        Ntot = N.sum(0)
        S = np.zeros((n_ev, C)); nn = np.zeros((n_ev, C)); ch = np.zeros((n_ev, C))
        Dk = torch.as_tensor(Dkk, dtype=torch.float32, device=DEV)
        zeros = torch.zeros(K, dtype=torch.long, device=DEV)
        ones = torch.ones(K, dtype=torch.long, device=DEV)
        for g in np.unique(group_of_item):
            Gc = torch.as_tensor(Ntot - N[g], dtype=torch.float32, device=DEV)          # gallery excludes group g
            Pm, prev, valid = M.precision_at_k(Dk, Gc, Gc, zeros, ones, k)              # no further exclusion
            Pm, prev, v = Pm.cpu().numpy(), prev.cpu().numpy(), valid.cpu().numpy()[:, None]
            m = group_of_item == g
            idx = np.flatnonzero(m)
            np.add.at(S, rec_of_item[idx], Qcnt[idx] * Pm[code[idx]] * v[code[idx]])
            np.add.at(nn, rec_of_item[idx], Qcnt[idx] * v[code[idx]])
            np.add.at(ch, rec_of_item[idx], Qcnt[idx] * prev[code[idx]] * v[code[idx]])
        out["q"] = dict(S=S, n=nn, ch=ch)
    return out


def score_assignment(data, ev, edges, codes, K, Vev, Dkk, anchors, maps, do_f1, k=KNN_K):
    """
    Evaluate one clustering of segments on the evaluation recordings `ev` of a fold.

    codes[r]  cluster id per segment;   Vev[r] (n_seg, F) segment vectors or None (original SMQ)
    Dkk       (K, K) distances between prototypes for the quantised retrieval
    maps      dict(h=hungarian_map computed here, t=train-fitted majority map given)
    """
    C = data.C
    n_ev = len(ev)
    res = {}
    cont = np.zeros((n_ev, K, C))
    fcodes = []
    for i, r in enumerate(ev):
        fc = np.repeat(codes[r], np.diff(edges[r]))
        assert len(fc) == data.T[r]
        fcodes.append(fc)
        np.add.at(cont[i], (fc, data.labels[r]), 1)
    hmap = M.hungarian_map(cont.sum(0))
    tmap = maps["t"]
    res["cont"] = cont
    res["hmap"] = hmap
    res["tmap"] = tmap
    res["hcorrect"] = np.array([sum(cont[i][k_, hmap[k_]] for k_ in range(K) if hmap[k_] >= 0) for i in range(n_ev)])
    res["tcorrect"] = np.array([sum(cont[i][k_, tmap[k_]] for k_ in range(K) if tmap[k_] >= 0) for i in range(n_ev)])
    res["nframes"] = np.array([data.T[r] for r in ev])
    res["nseg"] = np.array([len(edges[r]) - 1 for r in ev])
    res["ncodechg"] = np.array([len(M.boundary_positions(fc)) for fc in fcodes])

    # boundaries (class-agnostic, one-to-one); code-change boundaries and raw segment cuts
    for name, getter in (("bcode", lambda i, r: M.boundary_positions(fcodes[i])),
                         ("bcut", lambda i, r: edges[r][1:-1])):
        arr = np.zeros((n_ev, len(BOUND_TOL_S), 3))
        for i, r in enumerate(ev):
            pb = getter(i, r)
            gb = M.boundary_positions(data.labels[r])
            for j, ts in enumerate(BOUND_TOL_S):
                arr[i, j] = M.match_boundaries(pb, gb, int(round(ts * data.fps)))
        res[name] = arr

    if do_f1:
        for tag, mp in (("h", hmap), ("t", tmap)):
            lut = np.array([mp[k_] for k_ in range(K)])
            f = np.zeros((n_ev, 3, 3)); e = np.zeros(n_ev)
            for i, r in enumerate(ev):
                s = M.frame_scores(lut[fcodes[i]], data.labels[r])
                f[i] = np.stack([s["tp"], s["fp"], s["fn"]], 1)
                e[i] = s["edit"]
            res[f"f1_{tag}"] = f
            res[f"edit_{tag}"] = e

    # ---- retrieval (item sets: common anchors; eligible segments at two purity levels)
    ev_pos = {r: i for i, r in enumerate(ev)}
    for setname in ("ca", "seg80", "seg00"):
        Vs, cs, rs, gs, Qs = [], [], [], [], []
        for r in ev:
            e = edges[r]
            cnt = seg_label_counts(data.labels[r], e, C)
            if setname == "ca":
                t = anchors[r]
                sid = np.searchsorted(e, t, side="right") - 1
                q = np.zeros((len(e) - 1, C))
                np.add.at(q, (sid, data.labels[r][t]), 1)
                keep = q.sum(1) > 0
            else:
                tau = PURITY_TAU if setname == "seg80" else 0.0
                pur = cnt.max(1) / cnt.sum(1)
                keep = pur >= tau
                q = np.zeros_like(cnt)
                q[np.arange(len(cnt)), cnt.argmax(1)] = 1.0
            idx = np.flatnonzero(keep)
            if len(idx) == 0:
                continue
            Qs.append(q[idx]); cs.append(codes[r][idx])
            rs.append(np.full(len(idx), ev_pos[r])); gs.append(np.full(len(idx), data.group[r]))
            if Vev is not None:
                Vs.append(Vev[r][idx])
        Qs = np.concatenate(Qs); cs = np.concatenate(cs); rs = np.concatenate(rs); gs = np.concatenate(gs)
        # remap groups to 0..G-1
        ug, gs = np.unique(gs, return_inverse=True)
        Vc = np.concatenate(Vs) if Vs else None
        ra = _retrieval_arrays(Vc, Dkk, cs, rs, Qs, gs, ev, C, K, k)
        for mode, d in ra.items():
            for nm, a in d.items():
                res[f"{setname}_{mode}_{nm}"] = a
    return res


# ------------------------------------------------------------------ fold driver
def fit_and_score_condition(data, cond_edges, fit, ev, mu, sd, anchors, seeds, do_f1, weighting="segment", k=KNN_K):
    """cond_edges: list (per recording) of partitions. Returns {seed: result dict for `ev`}."""
    K = data.C
    Vfit_list = [R.segment_vectors(data.z[r], cond_edges[r], mu, sd, N_BINS) for r in fit]
    Vfit = np.concatenate(Vfit_list)
    wfit = np.concatenate([np.diff(cond_edges[r]) for r in fit]).astype(np.float64) if weighting == "duration" else None
    Vev = {r: R.segment_vectors(data.z[r], cond_edges[r], mu, sd, N_BINS) for r in ev}
    out = {}
    for seed in seeds:
        km = KMeans(n_clusters=K, n_init=5, random_state=seed, max_iter=300).fit(Vfit, sample_weight=wfit)
        cen = km.cluster_centers_.astype(np.float32)

        def assign(V):
            return torch.cdist(torch.as_tensor(V, device=DEV), torch.as_tensor(cen, device=DEV)).argmin(1).cpu().numpy()

        # train-fitted majority map from FITTING recordings only (uses fitting labels: supervised readout)
        Mfit = np.zeros((K, data.C))
        for r, V in zip(fit, Vfit_list):
            fc = np.repeat(assign(V), np.diff(cond_edges[r]))
            np.add.at(Mfit, (fc, data.labels[r]), 1)
        tmap = M.majority_map(Mfit)
        codes = {r: assign(Vev[r]) for r in ev}
        Dkk = cdist_np(cen).cpu().numpy()
        out[seed] = score_assignment(data, ev, cond_edges, codes, K, Vev, Dkk, anchors, dict(t=tmap), do_f1, k)
        out[seed]["fit_nseg"] = len(Vfit)
    return out


def score_original_smq(data, fit, ev, anchors, do_f1, k=KNN_K):
    """Original SMQ codes (released checkpoint) on the SAME fold partitions and metrics.
    Its codebook was learned transductively on all recordings; prototypes distances use SMQ's
    own patch metric. No continuous retrieval (its patch embedding is a W x D window)."""
    K = data.C
    edges = [P.fixed_edges(int(t), data.W) for t in data.T]
    codes = {}
    for r in list(fit) + list(ev):
        starts = edges[r][:-1]
        codes[r] = data.smq_codes[r][starts]
        # SMQ assigns one code per patch: verify the frame-level codes are constant within a patch
        assert np.array_equal(np.repeat(codes[r], np.diff(edges[r])), data.smq_codes[r])
    Mfit = np.zeros((K, data.C))
    for r in fit:
        np.add.at(Mfit, (data.smq_codes[r], data.labels[r]), 1)
    tmap = M.majority_map(Mfit)
    Dkk = smq_code_distances(data.codebook).cpu().numpy()
    res = score_assignment(data, ev, edges, codes, K, None, Dkk, anchors, dict(t=tmap), do_f1, k)
    return res


def merge_fold_results(dst, src, ev, n_rec, fold):
    """Write a fold's per-recording arrays (rows = ev) into dataset-level arrays (rows = all recordings).
    Cluster identities are fold-specific: 'cont' stays per-recording and is pooled per fold later;
    'hmap'/'tmap' are stored per fold."""
    for key, val in src.items():
        if key in ("hmap", "tmap", "fit_nseg"):
            dst.setdefault(key, {})[fold] = val
            continue
        val = np.asarray(val)
        if key not in dst:
            dst[key] = np.zeros((n_rec,) + val.shape[1:], dtype=val.dtype)
        dst[key][np.asarray(ev)] = val
