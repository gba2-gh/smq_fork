"""Per-(dataset, normalization, seed) raw K-means preparation.

Writes a directory in exactly the layout of `quantized_input_comparison`
artifacts (one family, `raw_kmeans`), so the calibration pipeline reads it
through the unchanged `artifacts.Loader`. The arithmetic mirrors
`quantized_input_comparison/prepare.py` step for step (raw patches on the same
grid, <=10,000 sampled complete windows, StandardScaler -> randomized PCA-64 ->
K-means-512, scalar RMS, pooled mean, grouping, occupancy), so HuGaDB / global /
seed 111 reproduces `qic_v3`'s raw family exactly (gate 1).

The seed drives the vocabulary (sample, PCA, K-means, grouping). The
train/validation split uses a fixed split seed (111) for every vocabulary seed.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits

from script.action_transport import config as at_config
from script.exp2round.q1q2 import core as q1q2core
from script.fsq_reconstruction.experiment import write_json_atomic
from script.quantized_input_comparison import artifacts
from script.quantized_input_comparison import fit as fitmod
from script.quantized_input_comparison import families
from script.week5.raw_kmeans_cross_dataset import datasets as D

SCHEMA = 2
# qic_v3 was built with every core of the machine (32 threads). Randomized PCA and K-means reduce in a
# thread-count-dependent order, so the preparation numerics (not the codes) shift in the sixth decimal at other
# counts. Preparation is therefore pinned to that count, independently of the launcher's 8-thread cap for training and
# OT, and the count is part of the preparation identity.
PREP_THREADS = 32
FAMILY = "raw_kmeans"
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


def extract_raw(spec: D.DatasetSpec, names: list[str], normalization: str) -> dict[str, dict]:
    """Raw patches on the zero-padded window grid, as `extract_raw_dataset`.
    `per_recording` z-scores every channel by its own recording's mean / std over
    frames (floor: std < 1e-8 -> 1), before patching."""
    if normalization not in NORMALIZATIONS:
        raise ValueError(normalization)
    window, out = spec.window, {}
    for name in names:
        arr = np.load(D.features_dir(spec) / name)  # (C, T, V, M)
        T = arr.shape[1]
        frames_np = arr.transpose(1, 0, 2, 3).reshape(T, -1)
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
    """True when `run_dir` holds verified artifacts for exactly this cell (no building, no deadline logic)."""
    key = PrepKey(spec.name, normalization, seed, spec.window, spec.num_states, num_codes=num_codes,
                  recording_cap=recording_cap)
    names = D.recording_names(spec, recording_cap)
    return manifest_ok(Path(run_dir), key, names, D.frame_counts(spec, names))


def prepare_cell(spec: D.DatasetSpec, normalization: str, seed: int, run_dir: Path,
                 recording_cap: int | None = None, num_codes: int = 512) -> Path:
    """Build (or verify and reuse) one cell's artifacts. Artifacts are derived and
    deterministic, so an incomplete directory (no manifest) is rebuilt from scratch."""
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

    raw = extract_raw(spec, names, normalization)
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
    records = families.build_family_records(FAMILY, raw, spec.window, pop)
    lookup = families.kmeans_lookup(pop)
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


def code_diagnostics(spec: D.DatasetSpec, loader, gts: dict[str, np.ndarray]) -> dict:
    """Occupancy of the vocabulary and, where subjects exist, how much the codes say about the
    subject versus the action. Label-aware diagnostics only; never used for selection."""
    from sklearn.metrics import mutual_info_score
    occ = loader.occupancy(FAMILY)
    counts, wc = occ["counts"], occ["window_counts"]
    p = wc[wc > 0] / wc.sum()
    out = dict(occupied_codes=int((wc > 0).sum()), perplexity=float(np.exp(-(p * np.log(p)).sum())),
               singleton_codes=int((wc == 1).sum()), mean_windows_per_occupied_code=float(wc.sum() / max((wc > 0).sum(), 1)),
               top10_share=float(np.sort(wc)[::-1][:10].sum() / wc.sum()))
    units, actions, subjects = [], [], []
    for name in loader.recording_names():
        rec = loader.recording(FAMILY, name)
        gt = gts[name]
        for code, start, length in zip(rec["codes"], rec["starts"], rec["lengths"]):
            if int(length) == spec.window:
                units.append(int(code))
                actions.append(q1q2core.majority(gt[int(start):int(start) + int(length)]))
                subjects.append(D.subject_of(spec, name))
    units, actions = np.asarray(units), np.asarray(actions)
    out["mi_code_action"] = float(mutual_info_score(units, actions))
    if spec.has_subject:
        subj = np.asarray(subjects)
        out["mi_code_subject"] = float(mutual_info_score(units, subj))
        out["cmi_code_subject_given_action"] = float(q1q2core.conditional_mi(units, subj, actions))
    return out
