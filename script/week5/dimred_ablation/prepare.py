"""Token preparation for one (dataset, variant, seed) cell of the dimensionality-reduction ablation.

`timescale_inspection.tokenizers._build` step for step (same window grid, fitting sample, split, K-means-512,
conditioning statistics, initial grouping, artifact layout), with the PCA stage replaced by a `reducers.Variant`.
With the `base` variant the path reproduces each anchor's tokens; the gates check that against the saved artifacts.

Anchors (the best configuration per dataset, outputs/all_runs/all_runs.xlsx, sheet "relevant results"):
  hugadb  fsq_pilot_002 VQ encoder (epoch 30) latents, 15-frame windows, PCA-64, `context` (guard 14, span 8)
          -- task2_v1 vq_kmeans / context_id, 49.55 F1@50 (seed 111 only)
  lara    T2 inter-joint distances, 12-frame windows, PCA-64, `context` (guard 0, span 2)
          -- ts5_v1 phase 6 S3, 24.02
  babel2  T2 inter-joint distances, 8-frame windows, PCA-64, `context` (guard 0, span 2)
          -- ts5_v1 phase 5 stage D, 30.68
  babel3  raw frames, 8-frame windows, whitened PCA-64, `pca` (no contextualizer: OT on the reduced window vector)
          -- ts5_v1 phase 5 T1 / pca, 26.04
"""

from __future__ import annotations

import dataclasses
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from script.fsq_reconstruction.experiment import write_json_atomic
from script.quantized_input_comparison import artifacts
from script.quantized_input_comparison import fit as fitmod
from script.quantized_input_comparison.families import window_starts
from script.week5.dimred_ablation import reducers as R
from script.week5.raw_kmeans_cross_dataset import datasets as D
from script.week5.timescale_inspection import tokenizers as TOK

SCHEMA = 1
NUM_CODES = 512


@dataclass(frozen=True)
class Anchor:
    dataset: str
    source: str  # vq_latent | relational | raw
    whiten: bool
    arm: str  # context | pca
    guard: int
    span: int
    reference: str  # where the anchor's number comes from
    reference_f1_50: float

    @property
    def window(self) -> int:
        return D.SPECS[self.dataset].window


ANCHORS = {
    "hugadb": Anchor("hugadb", "vq_latent", False, "context", 14, 8,
                     "task2_v1 vq_kmeans / context_id (seed 111 only)", 49.55),
    "lara": Anchor("lara", "relational", False, "context", 0, 2, "ts5_v1 phase 6 S3, T2 span 2", 24.02),
    "babel2": Anchor("babel2", "relational", False, "context", 0, 2, "ts5_v1 phase 5 stage D, T2 span 2", 30.68),
    "babel3": Anchor("babel3", "raw", True, "pca", 0, 0, "ts5_v1 phase 5 stage A, T1 / pca", 26.04),
}


@dataclass(frozen=True)
class PrepKey:
    dataset: str
    variant: str
    source: str
    window: int
    num_states: int
    seed: int
    method: str
    dim: int | None
    alpha: float
    var: float | None
    standardize: bool
    recenter: bool
    sfa_pre: int
    solver: str
    n_landmarks: int
    num_codes: int = NUM_CODES
    sample_cap: int = 10_000
    split_seed: int = 111
    recording_cap: int | None = None
    threads: int = TOK.PREP_THREADS


# sklearn's K-means runs one OpenMP thread per `threads` and each calls OpenBLAS concurrently; at 32 concurrent callers
# OpenBLAS overflows its precompiled per-thread buffers ("precompiled NUM_THREADS exceeded") and on full-dimension
# inputs the process died without a traceback (dr_v2, babel2 identity, three times). The PCA variants keep the anchors'
# 32 threads so they still reproduce the saved tokens exactly; every other method runs with SAFE_THREADS.
SAFE_THREADS = 8
ANCHOR_THREAD_METHODS = ("pca", "pca_var")


def family_of(variant: str) -> str:
    return f"dr_{variant}"


def make_key(dataset: str, variant: str, seed: int, recording_cap: int | None = None) -> PrepKey:
    a, v, spec = ANCHORS[dataset], R.BY_KEY[variant], D.SPECS[dataset]
    return PrepKey(dataset, variant, a.source, spec.window, spec.num_states, seed, v.method, v.dim,
                   v.alpha(a.whiten), v.var, v.standardize, v.recenter, v.sfa_pre, v.solver,
                   v.n_landmarks, recording_cap=recording_cap,
                   threads=TOK.PREP_THREADS if v.method in ANCHOR_THREAD_METHODS else SAFE_THREADS)


# -- extraction (cached: it depends on the dataset only, never on the variant or seed) -----------------------------

_CACHE: dict[tuple, dict[str, dict]] = {}


def extract_vq_latent(names: list[str], device: torch.device) -> dict[str, dict]:
    """The fsq_pilot_002 VQ encoder (epoch 30, frozen), 15-frame windows of concatenated per-frame latents (1,440-d):
    the `vq_encoder` population of `quantized_input_comparison` (qic_v3), which task2_v1's vq_kmeans family used."""
    from script.quantized_input_comparison import extract as QX
    source_cfg = QX.load_source_pilot_config()
    model, _ = QX.load_frozen_model("vq", source_cfg, device)
    out = QX.extract_population("vq_encoder", source_cfg, model, device, names)
    del model
    return {n: dict(flat=r["flat"], lengths=r["lengths"], num_patches=r["num_patches"]) for n, r in out.items()}


def extract(dataset: str, names: list[str], device: torch.device | None) -> dict[str, dict]:
    a = ANCHORS[dataset]
    key = (dataset, a.source, tuple(names))
    if key not in _CACHE:
        _CACHE.clear()  # one dataset resident at a time
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if a.source == "vq_latent":
            if D.SPECS[dataset].window != 15:
                raise ValueError("the VQ encoder population is defined on 15-frame windows")
            _CACHE[key] = extract_vq_latent(names, dev)
        else:
            _CACHE[key] = TOK.extract(D.SPECS[dataset], a.source, names, dev)
    return _CACHE[key]


def recentred(raw: dict[str, dict], window: int) -> dict[str, dict]:
    """Each recording's window vectors minus that recording's mean over its complete windows (a copy)."""
    out = {}
    for n, rec in raw.items():
        complete = rec["lengths"] == window
        mu = rec["flat"][complete].astype(np.float64).mean(axis=0) if complete.any() else 0.0
        out[n] = dict(rec, flat=(rec["flat"] - mu).astype(np.float32))
    return out


# -- population fit -------------------------------------------------------------------------------------------------

@dataclass
class Population:
    name: str
    scaler: StandardScaler
    reducer: R.Reducer
    kmeans: KMeans
    sample_ids: list[tuple[str, int]]
    n_fit_windows: int
    n_occupied: int
    info: dict


def project(pop: Population, flat: np.ndarray) -> np.ndarray:
    """With the plain-PCA reducer this is `fitmod.project_pca` exactly."""
    return np.asarray(pop.reducer.transform(pop.scaler.transform(flat))).astype(np.float32)


def _recording_means_std(raw: dict[str, dict], scaler: StandardScaler, window: int) -> dict[str, np.ndarray]:
    out = {}
    for n, rec in raw.items():
        complete = rec["lengths"] == window
        rows = rec["flat"][complete] if complete.any() else rec["flat"]
        out[n] = scaler.transform(rows).astype(np.float64).mean(axis=0)
    return out


def _adjacent_pairs(raw: dict[str, dict], sample_ids: list[tuple[str, int]], window: int):
    prev, nxt = [], []
    for n, p in sample_ids:
        lengths = raw[n]["lengths"]
        if p + 1 < len(lengths) and int(lengths[p + 1]) == window:
            prev.append((n, p))
            nxt.append((n, p + 1))
    return prev, nxt


def fit_population(name: str, raw: dict[str, dict], sample_ids: list[tuple[str, int]], key: PrepKey) -> Population:
    variant = R.BY_KEY[key.variant]
    sample = fitmod.gather(raw, sample_ids)
    if len(sample) < key.num_codes:
        raise RuntimeError(f"population {name!r}: only {len(sample)} sampled complete windows, "
                           f"need >= {key.num_codes} for K-means")
    scaler = StandardScaler(with_std=key.standardize).fit(sample)
    sample_std = scaler.transform(sample)
    extra = {}
    if variant.method == "wrpca":
        means = _recording_means_std(raw, scaler, key.window)
        extra["sample_rec_means"] = np.stack([means[n] for n, _ in sample_ids])
    if variant.method == "sfa":
        prev, nxt = _adjacent_pairs(raw, sample_ids, key.window)
        extra["pairs_prev"] = scaler.transform(fitmod.gather(raw, prev))
        extra["pairs_next"] = scaler.transform(fitmod.gather(raw, nxt))
    reducer, sample_red = R.fit_reducer(variant, key.alpha, sample_std, key.seed, **extra)
    kmeans = KMeans(n_clusters=key.num_codes, init="k-means++", n_init=5, max_iter=300, tol=1e-4, algorithm="lloyd",
                    random_state=key.seed)
    kmeans.fit(np.asarray(sample_red, dtype=np.float32) if reducer.sk_pca is None else sample_red)
    occupied = len(set(kmeans.labels_.tolist()))
    info = dict(reducer.info, input_dim=int(sample.shape[1]), out_dim=reducer.out_dim, alpha=key.alpha,
                retained_variance=R.retained_variance(reducer, np.asarray(sample_std, dtype=np.float64)),
                kmeans_inertia_per_window=float(kmeans.inertia_ / len(sample)))
    return Population(name, scaler, reducer, kmeans, sample_ids, len(sample), occupied, info)


def save_population(run_dir: Path, pop: Population) -> None:
    d = run_dir / "preprocessing"
    d.mkdir(parents=True, exist_ok=True)
    scale = pop.scaler.scale_ if pop.scaler.scale_ is not None else np.ones_like(pop.scaler.mean_)
    np.savez(d / f"{pop.name}_scaler.npz", mean_=pop.scaler.mean_, scale_=scale)
    red = pop.reducer
    if red.kernel is not None:
        k = red.kernel
        np.savez(d / f"{pop.name}_reducer.npz", landmarks=k["landmarks"], sigma=k["sigma"], col_mean=k["col_mean"],
                 all_mean=k["all_mean"], alphas=k["alphas"], divisor=k["divisor"])
        np.savez(d / f"{pop.name}_kmeans.npz", cluster_centers_=pop.kmeans.cluster_centers_,
                 n_iter_=pop.kmeans.n_iter_, inertia_=pop.kmeans.inertia_)
        return
    np.savez(d / f"{pop.name}_reducer.npz", basis=red.basis(),
             divisor=(np.sqrt(red.sk_pca.explained_variance_) if (red.sk_pca is not None and red.sk_pca.whiten)
                      else (red.divisor_ if red.divisor_ is not None else np.ones(red.out_dim))),
             mean_=red.sk_pca.mean_ if red.sk_pca is not None else red.mean_)
    np.savez(d / f"{pop.name}_kmeans.npz", cluster_centers_=pop.kmeans.cluster_centers_,
             n_iter_=pop.kmeans.n_iter_, inertia_=pop.kmeans.inertia_)


# -- cell preparation -----------------------------------------------------------------------------------------------

def manifest_ok(run_dir: Path, key: PrepKey, names: list[str], counts: dict[str, int]) -> bool:
    run_dir = Path(run_dir)
    if not (run_dir / "manifest.json").exists():
        return False
    try:
        m = artifacts.read_manifest(run_dir)
        return (m.get("schema") == SCHEMA and m.get("prep_key") == asdict(key) and
                sorted(m["recording_names"]) == sorted(names) and
                {k: int(v) for k, v in m["recording_frame_counts"].items()} == counts and
                artifacts.family_artifacts_complete(run_dir, family_of(key.variant), names, key.num_codes) and
                (run_dir / "split.json").exists())
    except (OSError, KeyError, ValueError):
        return False


def is_prepared(dataset: str, variant: str, seed: int, run_dir: Path, recording_cap: int | None = None) -> bool:
    spec = D.SPECS[dataset]
    names = D.recording_names(spec, recording_cap)
    return manifest_ok(run_dir, make_key(dataset, variant, seed, recording_cap), names, D.frame_counts(spec, names))


def prepare_cell(dataset: str, variant: str, seed: int, run_dir: Path, recording_cap: int | None = None,
                 device: torch.device | None = None) -> Path:
    key = make_key(dataset, variant, seed, recording_cap)
    spec = D.SPECS[dataset]
    names = D.recording_names(spec, recording_cap)
    counts = D.frame_counts(spec, names)
    run_dir = Path(run_dir)
    if manifest_ok(run_dir, key, names, counts):
        return run_dir
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    raw = extract(dataset, names, device)  # outside the thread limit: GPU / file reads only
    with threadpool_limits(limits=key.threads):
        _build(spec, key, run_dir, names, counts, raw)
    return run_dir


def _build(spec: D.DatasetSpec, key: PrepKey, run_dir: Path, names: list[str], counts: dict[str, int],
           raw: dict[str, dict]) -> None:
    family = family_of(key.variant)
    if key.recenter:
        raw = recentred(raw, spec.window)
    frame_counts = {n: int(sum(raw[n]["lengths"])) for n in names}
    if frame_counts != counts:
        raise RuntimeError("windowed frame counts disagree with the feature files' own shapes")
    lengths = fitmod.window_lengths(raw)
    all_ids = fitmod.complete_window_ids(lengths, spec.window)
    sample_ids = fitmod.sample_window_ids(all_ids, key.sample_cap, key.seed)
    artifacts.save_sample_ids(run_dir, sample_ids)
    train_names, val_names = fitmod.shared_validation_split(frame_counts, key.split_seed)
    artifacts.save_split(run_dir, train_names, val_names, frame_counts)
    train_set = set(train_names)
    train_complete = [(n, p) for n, p in all_ids if n in train_set]

    pop = fit_population(family, raw, sample_ids, key)
    save_population(run_dir, pop)
    feats = {n: project(pop, rec["flat"]) for n, rec in raw.items()}
    records = {}
    for name, rec in raw.items():
        records[name] = dict(codes=pop.kmeans.predict(feats[name]).astype(np.int64), continuous=feats[name],
                             starts=window_starts(rec["num_patches"], spec.window),
                             lengths=rec["lengths"].astype(np.int32))
    lookup = pop.kmeans.cluster_centers_.astype(np.float32)
    stats = fitmod.conditioning_stats(feats, sample_ids, train_complete)
    codes_by_name = {n: r["codes"] for n, r in records.items()}
    lengths_by_name = {n: r["lengths"] for n, r in records.items()}
    grouping, _, counts_w = fitmod.initial_grouping(codes_by_name, lengths_by_name, feats, key.num_codes,
                                                    spec.num_states, spec.fps, spec.window, key.seed)
    window_counts = np.zeros(key.num_codes, dtype=np.int64)
    for codes in codes_by_name.values():
        np.add.at(window_counts, codes, 1)
    artifacts.save_family(run_dir, family, records, lookup, stats["mean"], stats["rms"], grouping, counts_w,
                          window_counts)
    p = window_counts / max(window_counts.sum(), 1)
    nz = p[p > 0]
    write_json_atomic(run_dir / "reducer_info.json", dict(
        pop.info, kmeans_occupied=int(pop.n_occupied), code_perplexity=float(np.exp(-(nz * np.log(nz)).sum())),
        n_fit_windows=int(pop.n_fit_windows)))
    write_json_atomic(run_dir / "manifest.json", dict(
        schema=SCHEMA, prep_key=asdict(key), families=[family], recording_names=names,
        recording_frame_counts=frame_counts, num_windows={family: {n: int(len(r["codes"])) for n, r in records.items()}},
        feature_dims={family: int(lookup.shape[1])}, sample_ids_hash=artifacts.sample_ids_hash(sample_ids),
        n_train_recordings=len(train_names), n_val_recordings=len(val_names),
        kmeans_occupied=int(pop.n_occupied), n_fit_windows=int(pop.n_fit_windows)))
