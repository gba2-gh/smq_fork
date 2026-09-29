"""Encoder-training sweep: does reconstruction training improve the encoder's
features for segmentation, compared with the untrained encoder or with no
encoder at all?

Reuses a finished pilot run's saved checkpoints (no autoencoder training
here) and runs the pilot's identical downstream pipeline -- PCA-64 +
K-means-512 -> fresh masked contextualizer -> categorical and contextual
ASOT -- on:

    raw      raw input patches (no encoder)
    vq_eN    the VQ arm's pre-quantization encoder features at epoch N
    fsq_eN   the FSQ arm's pre-quantization encoder features at epoch N

Epoch 0 is the shared pre-training initialization, identical for both arms.
Unlike the pilot, the PCA/K-means seed follows the downstream seed, so each
replicate includes tokenizer variability. `information.csv` also logs
code<->action and code<->subject mutual information per representation and
seed. These are label-based diagnostics and are never used for fitting.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.model.smq import SMQModel

from script.action_transport.budget import DeadlineExceeded
from script.exp2round.q1q2 import core as q1q2
from script.fsq_reconstruction.experiment import (
    DATASET_DEFAULTS, OUT_ROOT, ROOT, PilotConfig, check_identity_against_released_meta, extract_dataset,
    fit_pca_kmeans, run_stream_downstream, transform_and_assign, write_csv, write_json_atomic,
)
from script.fsq_reconstruction.quantizers import FSQSMQModel

SWEEP_VERSION = "1.0"
DEFAULT_REPRESENTATIONS = ("raw", "vq_e0", "vq_e10", "vq_e20", "vq_e30")
DEFAULT_SEEDS = (111, 222, 1538574472, 333, 444)


def load_source_config(source_dir: Path) -> PilotConfig:
    raw = json.loads((source_dir / "manifest.json").read_text())["config"]
    return PilotConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})


def extract_raw_dataset(features_path: Path, window: int) -> dict[str, dict]:
    """Raw input patches on the same patch grid as `extract_recording`:
    per frame, the flattened (C, V, M) input; per patch, W frames
    concatenated, with the terminal patch zero-padded."""
    out = {}
    for name in sorted(os.listdir(features_path)):
        arr = np.load(features_path / name)  # (C,T,V,M)
        T = arr.shape[1]
        frames = torch.as_tensor(arr.transpose(1, 0, 2, 3).reshape(T, -1), dtype=torch.float32)
        num_patches = int(np.ceil(T / window))
        pad = num_patches * window - T
        if pad:
            frames = F.pad(frames, (0, 0, 0, pad))
        flat = frames.reshape(num_patches, -1).numpy()
        lengths = np.full(num_patches, window, dtype=np.int32)
        if pad:
            lengths[-1] = window - pad
        out[name] = dict(flat=flat, ids=np.zeros(num_patches, dtype=np.int64), lengths=lengths,
                         num_patches=num_patches)
    return out


def checkpoint_path(source_dir: Path, rep: str) -> Path:
    match = re.fullmatch(r"(vq|fsq)_e(\d+)", rep)
    if not match:
        raise SystemExit(f"unknown representation {rep!r}; expected raw, vq_eN or fsq_eN")
    return source_dir / "checkpoints" / match.group(1) / f"epoch-{match.group(2)}.model"


def load_encoder_model(source_cfg: PilotConfig, rep: str, path: Path, device: torch.device):
    kwargs = source_cfg.smq_kwargs()
    model = SMQModel(**kwargs) if rep.startswith("vq_") else FSQSMQModel(fsq_levels=source_cfg.levels, **kwargs)
    # Only the encoder output (pre-quantization latent) is read, so later edits to the FSQ adapter do not
    # change the extracted features even when they change the quantized path.
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)


def extract_representation(rep: str, source_cfg: PilotConfig, source_dir: Path, device: torch.device):
    d = DATASET_DEFAULTS[source_cfg.dataset]
    features_path = ROOT / "data" / source_cfg.dataset / "features"
    if rep == "raw":
        return extract_raw_dataset(features_path, d["patch_size"]), None
    path = checkpoint_path(source_dir, rep)
    model = load_encoder_model(source_cfg, rep, path, device)
    extracted = extract_dataset(model, features_path, d["patch_size"], d["num_features"], d["num_joints"],
                                d["num_person"], device)
    return extracted, hashlib.sha256(path.read_bytes()).hexdigest()


def information_row(rep: str, seed: int, ids_by_name: dict[str, np.ndarray], lengths_by_name: dict,
                    identity: dict, window: int, vocab) -> dict:
    units, actions, subjects = [], [], []
    for name in sorted(ids_by_name):
        gt = identity["gts_by_name"][name]
        subject = identity["meta_by_name"][name].subject
        for p, length in enumerate(lengths_by_name[name]):
            units.append(ids_by_name[name][p])
            actions.append(q1q2.majority(gt[p * window:p * window + int(length)]))
            subjects.append(subject)
    values = q1q2.information_values(np.asarray(units), np.asarray(actions), np.asarray(subjects))
    values.pop("occupancy")
    return dict(representation=rep, downstream_seed=seed, kmeans_fit_occupied=vocab.n_occupied,
                pca_explained_variance=float(vocab.pca.explained_variance_ratio_.sum()), **values)


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def run_sweep(cfg: PilotConfig, source_cfg: PilotConfig, source_dir: Path, run_dir: Path,
              representations: tuple[str, ...], device: torch.device, deadline: float | None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    window = DATASET_DEFAULTS[cfg.dataset]["patch_size"]
    expected_rows_per_job = 2 * len(cfg.final_rungs)

    seg_rows = read_csv_rows(run_dir / "segmentation.csv")
    counts: dict[str, int] = {}
    for r in seg_rows:
        jid = f"{r['stream']}_s{r['downstream_seed']}"
        counts[jid] = counts.get(jid, 0) + 1
    done = {jid for jid, n in counts.items() if n >= expected_rows_per_job}
    seg_rows = [r for r in seg_rows if f"{r['stream']}_s{r['downstream_seed']}" in done]
    info_rows = [r for r in read_csv_rows(run_dir / "information.csv")
                 if f"{r['representation']}_s{r['downstream_seed']}" in done]
    if done:
        print(f"[resume] skipping {len(done)} completed job(s): {sorted(done)}", flush=True)

    manifest = dict(sweep_version=SWEEP_VERSION, source_run=str(source_dir), source_config=asdict(source_cfg),
                    downstream_config=asdict(cfg), representations=list(representations),
                    started_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"), checkpoint_sha256={})
    not_run: list[dict] = []
    identity = None
    for rep in representations:
        pending = [s for s in cfg.downstream_seeds if f"{rep}_s{s}" not in done]
        if not pending:
            continue
        print(f"[extract] {rep} ...", flush=True)
        extracted, ckpt_hash = extract_representation(rep, source_cfg, source_dir, device)
        manifest["checkpoint_sha256"][rep] = ckpt_hash
        write_json_atomic(run_dir / "manifest.json", manifest)
        if identity is None:
            identity = check_identity_against_released_meta(cfg, extracted)
        lengths = {name: rec["lengths"] for name, rec in extracted.items()}
        for seed in pending:
            job_id = f"{rep}_s{seed}"
            if deadline is not None and time.perf_counter() > deadline:
                not_run.append(dict(job_id=job_id, reason="compute deadline reached"))
                continue
            print(f"[downstream] {job_id} ...", flush=True)
            vocab = fit_pca_kmeans(extracted, replace(cfg, preprocessing_seed=seed))
            ids, pca = {}, {}
            for name, rec in extracted.items():
                pca[name], ids[name] = transform_and_assign(vocab, rec["flat"])
            stream = dict(arm=rep, kind="kmeans", ids=ids, pca=pca, lengths=lengths)
            try:
                result = run_stream_downstream(rep, stream, cfg, seed, identity, deadline, run_dir / "artifacts",
                                               contextualizer_device=device)
            except DeadlineExceeded:
                not_run.append(dict(job_id=job_id, reason="interrupted at a cooperative deadline check"))
                continue
            seg_rows += result["rows"]
            info_rows.append(information_row(rep, seed, ids, lengths, identity, window, vocab))
            np.savez_compressed(run_dir / "artifacts" / f"{job_id}.npz", **result.get("artifacts", {}))
            write_csv(run_dir / "segmentation.csv", seg_rows)
            write_csv(run_dir / "information.csv", info_rows)
    write_csv(run_dir / "not_run.csv", not_run)
    print(f"Encoder sweep: {len(seg_rows)} segmentation rows, {len(not_run)} job(s) not run.", flush=True)


def smoke_overrides(cfg: PilotConfig) -> PilotConfig:
    return replace(cfg, downstream_seeds=(111,), levels=(2, 4, 4), kmeans_k=32, kmeans_sample_cap=2000,
                   contextualizer_max_steps=20, contextualizer_eval_every=5, contextualizer_patience_steps=20,
                   final_rungs=(5, 10), fit_outer_cap=2, fit_inner_steps=3)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Encoder-training sweep on a finished FSQ pilot's checkpoints")
    parser.add_argument("--source-run", dest="source_run", default="fsq_pilot_002")
    parser.add_argument("--run-id", dest="run_id", required=True)
    parser.add_argument("--representations", nargs="+", default=list(DEFAULT_REPRESENTATIONS))
    parser.add_argument("--downstream-seeds", dest="downstream_seeds", type=int, nargs="+",
                        default=list(DEFAULT_SEEDS))
    parser.add_argument("--hours", type=float, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    source_dir = OUT_ROOT / args.source_run
    source_cfg = load_source_config(source_dir)
    cfg = replace(source_cfg, downstream_seeds=tuple(args.downstream_seeds))
    representations = tuple(args.representations)
    if args.smoke:
        cfg = smoke_overrides(cfg)
        representations = ("raw", "vq_e0")
    run_dir = OUT_ROOT / "encoder_sweep" / args.run_id
    if (run_dir / "manifest.json").exists() and not args.resume:
        raise SystemExit(f"{run_dir}/manifest.json already exists; pass --resume or choose a new --run-id")
    for rep in representations:
        if rep != "raw" and not checkpoint_path(source_dir, rep).exists():
            raise SystemExit(f"missing checkpoint for {rep}: {checkpoint_path(source_dir, rep)}")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    deadline = time.perf_counter() + args.hours * 3600.0 if args.hours else None
    print(f"source={source_dir.name} run={run_dir} device={device} reps={representations} "
          f"seeds={cfg.downstream_seeds}", flush=True)
    run_sweep(cfg, source_cfg, source_dir, run_dir, representations, device, deadline)


if __name__ == "__main__":
    main()
