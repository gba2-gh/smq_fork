"""Phase 4 preparation: raw patches of positions plus first differences (velocities),
for one (dataset, normalization, seed) cell.

Mirrors `raw_kmeans_cross_dataset.prepare` step for step (same sampling, PCA-64,
K-means-512, split, grouping, occupancy) so the only difference from that run's
`raw_kmeans` family is the extra velocity channels -- everything downstream
(contextualizer, OT decoder, scorer) is unchanged. Written under a distinct
family name (`raw_kmeans_velocity`) in the same qic-layout directory, so it never
touches `raw_kmeans_cross_dataset`'s own runs and is read by the same unchanged
`artifacts.Loader`.

Velocity: frame[t] - frame[t-1], scaled by fps (units/second); frame 0 has no
predecessor and gets a zero velocity (never a wrapped or reflected value). With
`include_velocity=False`, this reduces to `raw_kmeans_cross_dataset.prepare.extract_raw`
exactly -- tested directly against it.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits

from script.exp2round.q1q2 import core as q1q2core
from script.fsq_reconstruction.experiment import write_json_atomic
from script.quantized_input_comparison import artifacts
from script.quantized_input_comparison import fit as fitmod
from script.quantized_input_comparison.families import kmeans_lookup, window_starts
from script.week5.raw_kmeans_cross_dataset import datasets as D

SCHEMA = 1
PREP_THREADS = 32  # matched to raw_kmeans_cross_dataset.prepare, for the same thread-order reasons
FAMILY = "raw_kmeans_velocity"
NORMALIZATIONS = ("global", "per_recording")


@dataclass(frozen=True)
class PrepKey:
    dataset: str
    normalization: str
    seed: int
    window: int
    num_states: int
    num_codes: int = 512
    pca_dim: int = 64
    sample_cap: int = 10_000
    split_seed: int = 111
    recording_cap: int | None = None
    threads: int = PREP_THREADS
    include_velocity: bool = True


def extract_velocity_raw(spec: D.DatasetSpec, names: list[str], normalization: str,
                         include_velocity: bool = True) -> dict[str, dict]:
    """Raw patches on the zero-padded window grid. Channels are [positions] or
    [positions, velocities] (doubled), concatenated before any normalization.
    `per_recording` z-scores every column (position and velocity alike) by its
    own recording's mean/std over frames (floor: std < 1e-8 -> 1)."""
    if normalization not in NORMALIZATIONS:
        raise ValueError(normalization)
    window, fps, out = spec.window, spec.fps, {}
    for name in names:
        arr = np.load(D.features_dir(spec) / name)  # (C, T, V, M)
        T = arr.shape[1]
        frames_np = arr.transpose(1, 0, 2, 3).reshape(T, -1).astype(np.float32)
        if include_velocity:
            vel = np.zeros_like(frames_np)
            vel[1:] = (frames_np[1:] - frames_np[:-1]) * fps
            frames_np = np.concatenate([frames_np, vel], axis=1)
        if normalization == "per_recording":
            frames_np = q1q2core.recording_zscore([frames_np])[0]
        frames = torch.as_tensor(frames_np, dtype=torch.float32)
        num_patches = int(np.ceil(T / window))
        pad = num_patches * window - T
        if pad:
            frames = F.pad(frames, (0, 0, 0, pad))
        lengths = np.full(num_patches, window, dtype=np.int32)
        if pad:
            lengths[-1] = window - pad
        out[name] = dict(flat=frames.reshape(num_patches, -1).numpy(), lengths=lengths, num_patches=num_patches)
    return out


def build_family_records(raw: dict[str, dict], window: int, pop: fitmod.FittedPopulation) -> dict:
    """The `kmeans`-kind equivalent of `families.build_family_records`, without registering a new
    family in `quantized_input_comparison.config` (which every other run's identity hash depends on)."""
    out = {}
    for name, rec in raw.items():
        pca_feats, ids = fitmod.assign_kmeans(pop, rec["flat"])
        out[name] = dict(codes=ids, continuous=pca_feats, starts=window_starts(rec["num_patches"], window),
                         lengths=rec["lengths"].astype(np.int32))
    return out


def manifest_ok(run_dir: Path, key: PrepKey, names: list[str], counts: dict[str, int]) -> bool:
    path = run_dir / "manifest.json"
    if not path.exists():
        return False
    try:
        m = artifacts.read_manifest(run_dir)
        return (m.get("schema") == SCHEMA and m.get("prep_key") == asdict(key) and
                sorted(m["recording_names"]) == sorted(names) and
                {k: int(v) for k, v in m["recording_frame_counts"].items()} == counts and
                artifacts.family_artifacts_complete(run_dir, FAMILY, names, key.num_codes) and
                (run_dir / "split.json").exists())
    except (OSError, KeyError, ValueError):
        return False


def is_prepared(spec: D.DatasetSpec, normalization: str, seed: int, run_dir: Path,
                recording_cap: int | None = None, num_codes: int = 512) -> bool:
    key = PrepKey(spec.name, normalization, seed, spec.window, spec.num_states, num_codes=num_codes,
                  recording_cap=recording_cap)
    names = D.recording_names(spec, recording_cap)
    return manifest_ok(Path(run_dir), key, names, D.frame_counts(spec, names))


def prepare_cell(spec: D.DatasetSpec, normalization: str, seed: int, run_dir: Path,
                 recording_cap: int | None = None, num_codes: int = 512) -> Path:
    key = PrepKey(spec.name, normalization, seed, spec.window, spec.num_states, num_codes=num_codes,
                  recording_cap=recording_cap)
    names = D.recording_names(spec, recording_cap)
    counts = D.frame_counts(spec, names)
    run_dir = Path(run_dir)
    if manifest_ok(run_dir, key, names, counts):
        return run_dir
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    with threadpool_limits(limits=key.threads):
        _build(spec, normalization, seed, run_dir, key, names, counts, num_codes)
    return run_dir


def _build(spec: D.DatasetSpec, normalization: str, seed: int, run_dir: Path, key: PrepKey, names: list[str],
           counts: dict[str, int], num_codes: int) -> None:
    raw = extract_velocity_raw(spec, names, normalization, key.include_velocity)
    frame_counts = {n: int(sum(raw[n]["lengths"])) for n in names}
    if frame_counts != counts:
        raise RuntimeError("windowed frame counts disagree with the feature files' own shapes")
    lengths = fitmod.window_lengths(raw)
    all_ids = fitmod.complete_window_ids(lengths, spec.window)
    sample_ids = fitmod.sample_window_ids(all_ids, key.sample_cap, seed)
    artifacts.save_sample_ids(run_dir, sample_ids)
    train_names, val_names = fitmod.shared_validation_split(frame_counts, key.split_seed)
    artifacts.save_split(run_dir, train_names, val_names, frame_counts)
    train_set = set(train_names)
    train_complete = [(n, p) for n, p in all_ids if n in train_set]

    pop = fitmod.fit_population("raw", raw, sample_ids, key.pca_dim, num_codes, seed)
    artifacts.save_population(run_dir, pop)
    pca_family = {n: fitmod.project_pca(pop, rec["flat"]) for n, rec in raw.items()}
    records = build_family_records(raw, spec.window, pop)
    lookup = kmeans_lookup(pop)
    stats = fitmod.conditioning_stats({n: r["continuous"] for n, r in records.items()}, sample_ids, train_complete)
    codes_by_name = {n: r["codes"] for n, r in records.items()}
    lengths_by_name = {n: r["lengths"] for n, r in records.items()}
    grouping, _, counts_w = fitmod.initial_grouping(codes_by_name, lengths_by_name, pca_family, num_codes,
                                                    spec.num_states, spec.fps, spec.window, seed)
    window_counts = np.zeros(num_codes, dtype=np.int64)
    for codes in codes_by_name.values():
        np.add.at(window_counts, codes, 1)
    artifacts.save_family(run_dir, FAMILY, records, lookup, stats["mean"], stats["rms"], grouping, counts_w,
                          window_counts)
    write_json_atomic(run_dir / "manifest.json", dict(
        schema=SCHEMA, prep_key=asdict(key), families=[FAMILY], recording_names=names,
        recording_frame_counts=frame_counts, num_windows={FAMILY: {n: int(len(r["codes"])) for n, r in records.items()}},
        feature_dims={FAMILY: int(lookup.shape[1])}, sample_ids_hash=artifacts.sample_ids_hash(sample_ids),
        n_train_recordings=len(train_names), n_val_recordings=len(val_names),
        kmeans_occupied=int(pop.n_occupied), n_fit_windows=int(pop.n_fit_windows)))
