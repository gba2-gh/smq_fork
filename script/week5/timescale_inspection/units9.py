"""Phase 9 units: iteration-2 targets (per-layer K-means of the trained I1 model's states), the I2 preparation family
with the loader's dimensional contract, and the label-free diagnostics (occupancy, cross-seed agreement, association
with recording and subject).

The I2 family (`I2Source`, a read-only drop-in for `quantized_input_comparison.artifacts.Loader`):
  codes       iteration-2 codes (K-means-512 of layer-2 states)
  continuous  T2's PCA-64 window features, unchanged (the model input)
  lookup      for each new code, the frame-weighted mean of T2's continuous features over its windows (64-d), the
              same quantity `initial_grouping` computes; the 128-d teacher centroids are stored separately
  mean, rms   T2's
  grouping    `initial_grouping` from the new codes and T2's continuous features
The trainer takes its input dimension from `lookup.shape[1]`, so the lookup MUST be 64-d (gate G9f).

No action label is read anywhere in this module. Subject (LARa) is recording metadata, not an action label.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from threadpoolctl import threadpool_limits

from script.action_transport import categorical
from script.fsq_reconstruction.evaluate import per_code_pca_centers
from script.quantized_input_comparison import fit as fitmod
from script.week5.raw_kmeans_cross_dataset import datasets as D

NUM_CODES = 512
LAYER = 2  # 1-based; preselected (HuBERT Base took iteration-2 targets from the 6th of 12 layers)
KMEANS_KW = dict(n_clusters=NUM_CODES, init="k-means++", n_init=5, max_iter=300, tol=1e-4, algorithm="lloyd")


# -- iteration-2 targets --------------------------------------------------------------------------------------------

def cluster_states(states_by_name: dict[str, np.ndarray], sample_ids: list[tuple[str, int]], seed: int,
                   threads: int) -> KMeans:
    """K-means-512 of the sampled windows' states (the same sampled windows T2's K-means used), with the vocabulary
    seed and T2's preparation thread count."""
    sample = np.stack([states_by_name[n][p] for n, p in sample_ids]).astype(np.float32)
    if len(sample) < NUM_CODES:
        raise RuntimeError(f"only {len(sample)} sampled windows, need >= {NUM_CODES} for K-means")
    with threadpool_limits(limits=threads):
        return KMeans(random_state=seed, **KMEANS_KW).fit(sample)


def assign_codes(km: KMeans, states_by_name: dict[str, np.ndarray], threads: int) -> dict[str, np.ndarray]:
    with threadpool_limits(limits=threads):
        return {n: km.predict(np.ascontiguousarray(s, dtype=np.float32)).astype(np.int64) for n, s in states_by_name.items()}


class I2Source:
    """Duck-typed read-only `Loader`: codes replaced by `codes_by_name`, lookup/grouping replaced, everything else
    (continuous features, starts/lengths, mean, RMS, split, frame counts) read from the base T2 preparation. The
    `family` argument is ignored: there is exactly one family here, T2's feature space with new targets."""

    def __init__(self, base, base_family: str, codes_by_name: dict[str, np.ndarray], lookup: np.ndarray,
                 grouping: np.ndarray, teacher_centroids: np.ndarray | None = None, meta: dict | None = None):
        self.base, self.base_family = base, base_family
        self.codes_by_name, self._lookup, self._grouping = codes_by_name, lookup, grouping
        self.teacher_centroids, self.meta = teacher_centroids, meta or {}
        self.run_dir = base.run_dir

    def recording_names(self):
        return self.base.recording_names()

    def split_membership(self, name):
        return self.base.split_membership(name)

    def frame_count(self, name):
        return self.base.frame_count(name)

    def recording(self, family, name):
        rec = dict(self.base.recording(self.base_family, name))
        rec["codes"] = self.codes_by_name[name]
        return rec

    def lookup(self, family):
        return self._lookup

    def grouping(self, family):
        return self._grouping

    def mean(self, family):
        return self.base.mean(self.base_family)

    def rms(self, family):
        return self.base.rms(self.base_family)


def build_family(base, base_family: str, codes_by_name: dict[str, np.ndarray], spec: D.DatasetSpec, seed: int,
                 teacher_centroids: np.ndarray | None = None, meta: dict | None = None) -> I2Source:
    """`initial_grouping` from `codes_by_name` and T2's continuous features; lookup = its per-code centres."""
    names = sorted(codes_by_name)
    cont = {n: base.recording(base_family, n)["continuous"] for n in names}
    lengths = {n: base.recording(base_family, n)["lengths"] for n in names}
    grouping, centers, counts = fitmod.initial_grouping(
        {n: np.asarray(codes_by_name[n], dtype=np.int32) for n in names}, lengths, cont, NUM_CODES, spec.num_states,
        spec.fps, spec.window, seed)
    meta = dict(meta or {}, occupied_codes=int((counts > 0).sum()))
    return I2Source(base, base_family, {n: np.asarray(c) for n, c in codes_by_name.items()},
                    np.asarray(centers, dtype=np.float32), np.asarray(grouping), teacher_centroids, meta)


def save_family(src: I2Source, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "codes.npz", **{n: c.astype(np.int32) for n, c in src.codes_by_name.items()})
    np.save(out_dir / "lookup.npy", src._lookup)
    np.save(out_dir / "grouping.npy", src._grouping)
    if src.teacher_centroids is not None:
        np.save(out_dir / "teacher_centroids.npy", src.teacher_centroids)
    (out_dir / "meta.json").write_text(json.dumps(src.meta, indent=2, default=str))


def load_family(base, base_family: str, out_dir: Path) -> I2Source:
    out_dir = Path(out_dir)
    with np.load(out_dir / "codes.npz") as z:
        codes = {n: z[n].astype(np.int64) for n in z.files}
    tc = out_dir / "teacher_centroids.npy"
    return I2Source(base, base_family, codes, np.load(out_dir / "lookup.npy"), np.load(out_dir / "grouping.npy"),
                    np.load(tc) if tc.exists() else None, json.loads((out_dir / "meta.json").read_text()))


def check_family_dims(src: I2Source, family: str, model_input_dim: int | None = None) -> dict:
    """Gate G9f: lookup, continuous input, mean and (if given) the model's projection width agree in dimension; the
    RMS is a scalar equal to the base's; codes lie in [0, NUM_CODES)."""
    name = src.recording_names()[0]
    cont = src.recording(family, name)["continuous"]
    dim = cont.shape[1]
    checks = dict(lookup_dim=src.lookup(family).shape == (NUM_CODES, dim), mean_dim=np.asarray(src.mean(family)).shape == (dim,),
                  rms_scalar=np.asarray(src.rms(family)).shape == (), rms_equal_base=float(src.rms(family)) == float(src.base.rms(src.base_family)),
                  codes_in_range=all(int(c.min()) >= 0 and int(c.max()) < NUM_CODES for c in src.codes_by_name.values()),
                  grouping_len=src.grouping(family).shape == (NUM_CODES,))
    if model_input_dim is not None:
        checks["model_input_dim"] = model_input_dim == dim
    return dict(status="passed" if all(bool(v) for v in checks.values()) else "failed",
                checks={k: bool(v) for k, v in checks.items()}, dim=int(dim))


# -- diagnostics (label-free) ---------------------------------------------------------------------------------------

def _concat(codes_by_name: dict[str, np.ndarray], cap: int = 300_000) -> tuple[np.ndarray, np.ndarray]:
    """(codes, recording index) over every window, in sorted-name order, subsampled evenly to at most `cap` windows."""
    names = sorted(codes_by_name)
    codes = np.concatenate([np.asarray(codes_by_name[n]) for n in names])
    rec = np.concatenate([np.full(len(codes_by_name[n]), i) for i, n in enumerate(names)])
    if len(codes) > cap:
        idx = np.linspace(0, len(codes) - 1, cap).astype(np.int64)
        codes, rec = codes[idx], rec[idx]
    return codes, rec


def nmi(a, b) -> float:
    return float(normalized_mutual_info_score(a, b))


def cross_seed_nmi(codes_by_seed: dict[int, dict[str, np.ndarray]]) -> float | None:
    """Mean normalized mutual information between the codes of every pair of seeds, over the same windows."""
    seeds = sorted(codes_by_seed)
    if len(seeds) < 2:
        return None
    flat = {s: np.concatenate([np.asarray(codes_by_seed[s][n]) for n in sorted(codes_by_seed[s])]) for s in seeds}
    n = len(next(iter(flat.values())))
    idx = np.linspace(0, n - 1, min(n, 300_000)).astype(np.int64)
    vals = [nmi(flat[a][idx], flat[b][idx]) for i, a in enumerate(seeds) for b in seeds[i + 1:]]
    return float(np.mean(vals))


def chance_nmi(codes_by_seed: dict[int, dict[str, np.ndarray]], seed: int = 0) -> float | None:
    """The same measure with the second seed's codes shuffled across windows: the finite-sample floor of cross-seed NMI.
    With 512 codes and few windows (cells >> windows) NMI between unrelated labelings is far from 0, so cross-seed
    agreement is only read against this floor and against the T2 reference at the same window count."""
    seeds = sorted(codes_by_seed)
    if len(seeds) < 2:
        return None
    flat = {s: np.concatenate([np.asarray(codes_by_seed[s][n]) for n in sorted(codes_by_seed[s])]) for s in seeds[:2]}
    n = len(flat[seeds[0]])
    idx = np.linspace(0, n - 1, min(n, 300_000)).astype(np.int64)
    shuffled = np.random.default_rng(seed).permutation(flat[seeds[1]])
    return nmi(flat[seeds[0]][idx], shuffled[idx])


def recording_nmi(codes_by_name: dict[str, np.ndarray]) -> float:
    codes, rec = _concat(codes_by_name)
    return nmi(codes, rec)


def subject_nmi(codes_by_name: dict[str, np.ndarray], spec: D.DatasetSpec) -> float | None:
    if not spec.has_subject:
        return None
    names = sorted(codes_by_name)
    subj = {s: i for i, s in enumerate(sorted({D.subject_of(spec, n) for n in names}))}
    codes = np.concatenate([np.asarray(codes_by_name[n]) for n in names])
    sid = np.concatenate([np.full(len(codes_by_name[n]), subj[D.subject_of(spec, n)]) for n in names])
    if len(codes) > 300_000:
        idx = np.linspace(0, len(codes) - 1, 300_000).astype(np.int64)
        codes, sid = codes[idx], sid[idx]
    return nmi(codes, sid)


def occupancy(codes_by_name: dict[str, np.ndarray]) -> dict:
    allc = np.concatenate([np.asarray(c) for c in codes_by_name.values()])
    counts = np.bincount(allc, minlength=NUM_CODES).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return dict(occupied=int((counts > 0).sum()), perplexity=float(np.exp(-(p * np.log(p)).sum())))


def nn_same_recording_share(feat_by_name: dict[str, np.ndarray], n_query: int = 3000, n_pool: int = 30000,
                            seed: int = 0, exclude_radius: int = 8) -> float:
    """Share of nearest neighbours (cosine) that lie in the same recording, ignoring candidates within
    `exclude_radius` windows of the query in its own recording (temporal adjacency is trivially similar). Windows with a
    zero vector are skipped. A high share means the representation separates recordings, not actions."""
    names = sorted(feat_by_name)
    feats = np.concatenate([feat_by_name[n] for n in names]).astype(np.float32)
    rec = np.concatenate([np.full(len(feat_by_name[n]), i) for i, n in enumerate(names)])
    pos = np.concatenate([np.arange(len(feat_by_name[n])) for n in names])
    norms = np.linalg.norm(feats, axis=1)
    ok = np.flatnonzero(norms > 1e-9)
    rng = np.random.default_rng(seed)
    pool = rng.choice(ok, size=min(n_pool, len(ok)), replace=False)
    query = rng.choice(pool, size=min(n_query, len(pool)), replace=False)
    P = feats[pool] / norms[pool][:, None]
    same = total = 0
    for s in range(0, len(query), 500):
        q = query[s:s + 500]
        sim = (feats[q] / norms[q][:, None]) @ P.T
        near = (rec[q][:, None] == rec[pool][None, :]) & (np.abs(pos[q][:, None] - pos[pool][None, :]) <= exclude_radius)
        sim[near] = -np.inf
        best = pool[np.argmax(sim, axis=1)]
        same += int((rec[best] == rec[q]).sum())
        total += len(q)
    return same / total
