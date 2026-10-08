"""Manifest + per-recording array files (never one giant in-memory object),
and the loader Task 2 is expected to consume.

Layout under results/quantized_input_comparison/<run_id>/:

    manifest.json                  config, seeds, checkpoint hashes, dims,
                                    sample-id hash, source identities
    split.json                     shared recording-level train/val split
    preprocessing/<population>_scaler.npz   StandardScaler mean_/scale_
    preprocessing/<population>_pca.npz      PCA components_/mean_/explained_variance_
    preprocessing/<population>_kmeans.npz   KMeans cluster_centers_
    preprocessing/sample_ids.json           shared <=10,000 fitting window ids
    families/<family>/lookup.npy            [num_codes, feature_dim] float32
    families/<family>/mean.npy              [feature_dim] float32 (fitting-window pooled mean)
    families/<family>/rms.npy               scalar float32 (single train-split coordinate RMS,
                                             shared by quantized and continuous inputs)
    families/<family>/grouping.npy          [num_codes] int32 (seed-111 code->action init)
    families/<family>/occupancy.npz         counts [num_codes] float64 (frame-weighted),
                                             window_counts [num_codes] int64 (window-count,
                                             unweighted), n_occupied int scalar
    families/<family>/recordings/<name>.npz codes int64[P], continuous float32[P,d],
                                             starts int32[P], lengths int32[P]

Cache reuse: `Loader`/`prepare.py` compare `manifest.json`'s schema version,
checkpoint hashes, config, recording names and per-recording frame counts
against the current run before treating an existing run_dir as reusable; any
mismatch is a hard error, never a silent overwrite. Reuse of any
preprocessing/family artifact is gated on a verified, matching manifest
already existing -- a `--resume` against a run_dir that has partial artifacts
but no manifest.json (e.g. a prior run crashed mid-way) is refused outright
rather than trusting the unverifiable partial state.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from script.fsq_reconstruction.experiment import write_json_atomic
from script.quantized_input_comparison import config as cfg
from script.quantized_input_comparison import fit as fitmod


def sample_ids_hash(ids: list[tuple[str, int]]) -> str:
    payload = json.dumps([[n, int(p)] for n, p in ids], sort_keys=False).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def save_population(run_dir: Path, pop: fitmod.FittedPopulation) -> None:
    d = run_dir / "preprocessing"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{pop.name}_scaler.npz", mean_=pop.scaler.mean_, scale_=pop.scaler.scale_, var_=pop.scaler.var_)
    np.savez(d / f"{pop.name}_pca.npz", components_=pop.pca.components_, mean_=pop.pca.mean_,
             explained_variance_ratio_=pop.pca.explained_variance_ratio_, n_components_=pop.pca.n_components_)
    np.savez(d / f"{pop.name}_kmeans.npz", cluster_centers_=pop.kmeans.cluster_centers_,
             n_iter_=pop.kmeans.n_iter_, inertia_=pop.kmeans.inertia_)


def save_sample_ids(run_dir: Path, ids: list[tuple[str, int]]) -> None:
    write_json_atomic(run_dir / "preprocessing" / "sample_ids.json",
                       dict(ids=[[n, int(p)] for n, p in ids], hash=sample_ids_hash(ids)))


def load_sample_ids(run_dir: Path) -> list[tuple[str, int]]:
    payload = json.loads((run_dir / "preprocessing" / "sample_ids.json").read_text())
    return [(n, int(p)) for n, p in payload["ids"]]


def save_split(run_dir: Path, train_names: list[str], val_names: list[str], frame_counts: dict[str, int]) -> None:
    write_json_atomic(run_dir / "split.json",
                       dict(train=train_names, val=val_names, frame_counts=frame_counts))


def load_split(run_dir: Path) -> dict:
    return json.loads((run_dir / "split.json").read_text())


def save_family(run_dir: Path, family: str, records: dict[str, dict], lookup: np.ndarray,
                 mean: np.ndarray, rms, grouping: np.ndarray, counts: np.ndarray,
                 window_counts: np.ndarray) -> None:
    fam_dir = run_dir / "families" / family
    rec_dir = fam_dir / "recordings"
    rec_dir.mkdir(parents=True, exist_ok=True)
    np.save(fam_dir / "lookup.npy", lookup.astype(np.float32))
    np.save(fam_dir / "mean.npy", mean.astype(np.float32))
    np.save(fam_dir / "rms.npy", np.asarray(rms, dtype=np.float32))  # scalar, not [feature_dim]
    np.save(fam_dir / "grouping.npy", grouping.astype(np.int32))
    np.savez(fam_dir / "occupancy.npz", counts=counts.astype(np.float64),
             window_counts=window_counts.astype(np.int64), n_occupied=int((counts > 0).sum()))
    for name, rec in records.items():
        temp = rec_dir / f"{name}.tmp.npz"
        final = rec_dir / f"{name}.npz"
        np.savez(temp, codes=rec["codes"].astype(np.int64), continuous=rec["continuous"].astype(np.float32),
                  starts=rec["starts"].astype(np.int32), lengths=rec["lengths"].astype(np.int32))
        os.replace(temp, final)


def write_manifest(run_dir: Path, prep_cfg: cfg.PrepConfig, checkpoint_hashes: dict, sample_hash: str,
                    dims: dict, names: list[str], num_windows: dict,
                    recording_frame_counts: dict[str, int] | None = None) -> None:
    manifest = dict(schema_version=cfg.ARTIFACT_SCHEMA_VERSION,
                     config=asdict(prep_cfg), checkpoint_sha256=checkpoint_hashes, sample_ids_hash=sample_hash,
                     feature_dims=dims, recording_names=names, num_windows=num_windows,
                     recording_frame_counts=recording_frame_counts or {},
                     receptive_radius=cfg.RECEPTIVE_RADIUS, receptive_field=cfg.RECEPTIVE_FIELD,
                     native_dim_expected=cfg.NATIVE_DIM, families=list(cfg.FAMILIES))
    write_json_atomic(run_dir / "manifest.json", manifest)


def read_manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text())


def manifest_matches(run_dir: Path, prep_cfg: cfg.PrepConfig, checkpoint_hashes: dict,
                      recording_names: list[str] | None = None,
                      recording_frame_counts: dict[str, int] | None = None) -> bool:
    """True only if every identity the manifest records for a prior run
    matches the current invocation: artifact schema version, full config
    (incl. seed/window/etc.), both source checkpoint hashes, and -- when
    given -- the exact set of recordings extracted and their per-recording
    frame counts (so a changed data directory under an unchanged
    config/checkpoint/recording-name-set is still caught, not just a
    changed recording *set*). Any mismatch means `--resume` must refuse to
    reuse the run, not silently proceed.

    A manifest written before `schema_version` existed (e.g. qic_v1, which
    predates the scalar-RMS fix) reads back with `schema_version` absent,
    i.e. `None`, which never equals `cfg.ARTIFACT_SCHEMA_VERSION` -- such a
    manifest always fails this check, by construction, rather than being
    silently accepted."""
    if not (run_dir / "manifest.json").exists():
        return False
    m = read_manifest(run_dir)
    if m.get("schema_version") != cfg.ARTIFACT_SCHEMA_VERSION:
        return False
    if m["config"] != asdict(prep_cfg) or m["checkpoint_sha256"] != checkpoint_hashes:
        return False
    if recording_names is not None and sorted(m.get("recording_names", [])) != sorted(recording_names):
        return False
    if recording_frame_counts is not None:
        recorded = m.get("recording_frame_counts") or {}
        if {k: int(v) for k, v in recorded.items()} != {k: int(v) for k, v in recording_frame_counts.items()}:
            return False
    return True


def family_artifacts_complete(run_dir: Path, family: str, names: list[str], num_codes: int) -> bool:
    """True only if `family`'s on-disk artifacts are structurally complete
    and valid for `names` under `num_codes`: every fixed array file exists,
    rms.npy is a scalar, lookup.npy's first axis is num_codes, and every
    recording in `names` has a per-recording .npz. Used by `--resume` to
    decide whether a family can be SKIPPED (never regenerated/overwritten)
    rather than treated as reusable just because `resume` was passed."""
    fam_dir = run_dir / "families" / family
    fixed = ["lookup.npy", "mean.npy", "rms.npy", "grouping.npy"]
    if not all((fam_dir / f).exists() for f in fixed) or not (fam_dir / "occupancy.npz").exists():
        return False
    try:
        lookup = np.load(fam_dir / "lookup.npy")
        rms = np.load(fam_dir / "rms.npy")
        occ = np.load(fam_dir / "occupancy.npz")
    except Exception:
        return False
    if lookup.ndim != 2 or lookup.shape[0] != num_codes:
        return False
    if np.asarray(rms).shape != ():
        return False
    if "counts" not in occ or "window_counts" not in occ or occ["counts"].shape != (num_codes,):
        return False
    rec_dir = fam_dir / "recordings"
    return all((rec_dir / f"{name}.npz").exists() for name in names)


class Loader:
    """Task 2's read interface: per family and recording, code ids,
    continuous features, window starts/lengths, plus the family's fixed
    lookup table, conditioning stats, initial grouping and the shared
    split. Reads lazily per recording (never loads the whole run into
    memory at once)."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.manifest = read_manifest(self.run_dir)
        self.split = load_split(self.run_dir)

    def families(self) -> list[str]:
        return self.manifest["families"]

    def recording_names(self) -> list[str]:
        return self.manifest["recording_names"]

    def lookup(self, family: str) -> np.ndarray:
        return np.load(self.run_dir / "families" / family / "lookup.npy")

    def mean(self, family: str) -> np.ndarray:
        return np.load(self.run_dir / "families" / family / "mean.npy")

    def rms(self, family: str) -> np.ndarray:
        """Scalar float32 (0-d array): the single train-split coordinate RMS
        shared by quantized and continuous inputs for this family. Not a
        [feature_dim] array -- use the same scalar to scale every
        coordinate of both the quantized lookup vectors and the continuous
        features, preserving their relative geometry."""
        return np.load(self.run_dir / "families" / family / "rms.npy")

    def grouping(self, family: str) -> np.ndarray:
        return np.load(self.run_dir / "families" / family / "grouping.npy")

    def occupancy(self, family: str) -> dict:
        z = np.load(self.run_dir / "families" / family / "occupancy.npz")
        return dict(counts=z["counts"], window_counts=z["window_counts"], n_occupied=int(z["n_occupied"]))

    def recording(self, family: str, name: str) -> dict:
        z = np.load(self.run_dir / "families" / family / "recordings" / f"{name}.npz")
        frame_count = self.split["frame_counts"][name]
        return dict(name=name, frame_count=frame_count, codes=z["codes"], continuous=z["continuous"],
                    starts=z["starts"], lengths=z["lengths"])

    def frame_count(self, name: str) -> int:
        return self.split["frame_counts"][name]

    def split_membership(self, name: str) -> str:
        if name in self.split["train"]:
            return "train"
        if name in self.split["val"]:
            return "val"
        return "unknown"
