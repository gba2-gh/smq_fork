"""Stage B orchestrator (docs/plans/THREE_STAGE_MOTION_PLAN.md §4): trains
B-code, one model per (dataset, normalize, seed) pooled fitting population,
reusing Stage A's cached K=500 tokenizer. No action label enters training,
checkpoint selection, or the validation split.

CLI mirrors run.py: --validate-only / --preflight / --dry-run / full run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.action_transport import config, data
from script.action_transport.budget import DeadlineExceeded
from script.action_transport.stageb_configs import config_hash as compute_config_hash
from script.action_transport.stageb_configs import resolve_by_name
from script.action_transport.stageb_data import Chunk, build_validation_split, make_chunks
from script.action_transport.stageb_infer import extract_embeddings
from script.action_transport.stageb_model import ModelConfig
from script.action_transport.stageb_train import TrainConfig, save_checkpoint, train

# v1.7 (docs/plans/STAGE_C_AGENT_INSTRUCTIONS.md Part 1): adds --train-config
# so a run can freeze the optstudy-selected configuration (stage_b_003)
# instead of always using TrainConfig()'s v1.5 defaults. Effective
# TrainConfig is now part of the run identity via IDENTITY_FIELDS, so v1.5
# and "selected" runs can never silently share or resume each other's cells.
STAGEB_VERSION = "1.7"
OUT_ROOT = config.OUT_ROOT.parent / "action_transport_stage_b"

# Everything that determines what a checkpoint/embedding actually is. Changing
# any of these without a new run_id makes old and new cells non-comparable.
IDENTITY_FIELDS = ("stageb_version", "tokenizer_version", "train_config")


def source_sha256(*paths: Path) -> dict[str, str]:
    out = {}
    for path in paths:
        out[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def resolve_train_config(name: str, study_id: str = "optstudy_002") -> tuple[TrainConfig, dict]:
    """"v1.5" -> TrainConfig() (the original default). "selected" -> the
    optimization study's chosen grid configuration, verified against its
    recorded hash in <study_id>/sweep.csv (v1.7). Returns (cfg, provenance)."""
    if name == "v1.5":
        return TrainConfig(), {"train_config_name": "v1.5", "selected_from": None}
    if name != "selected":
        raise ValueError(f"unknown --train-config {name!r}")
    study_dir = OUT_ROOT / study_id
    selection_path = study_dir / "selection.json"
    sweep_path = study_dir / "sweep.csv"
    if not selection_path.exists():
        raise SystemExit(f"{selection_path} not found -- run stageb_optstudy --phase 1 first, or pass "
                         "--study-id to point at the study that produced it")
    selection = json.loads(selection_path.read_text())
    chosen = selection.get("chosen")
    if not chosen:
        raise SystemExit(f"{selection_path} has no chosen configuration")
    chosen_name = chosen["config"]
    cfg = resolve_by_name(chosen_name)
    effective_hash = compute_config_hash(cfg)
    with sweep_path.open(newline="", encoding="utf-8") as fh:
        import csv
        recorded = {r["config_hash"] for r in csv.DictReader(fh) if r["config"] == chosen_name}
    if not recorded:
        raise SystemExit(f"no sweep.csv rows for {chosen_name!r} in {sweep_path}")
    if recorded != {effective_hash}:
        raise SystemExit(
            f"the current grid's {chosen_name!r} hashes to {effective_hash}, but {sweep_path} recorded "
            f"{recorded} for it -- stageb_train.py/stageb_configs.py changed since selection; the "
            "'selected' configuration no longer means what the study chose. Investigate before training.")
    return cfg, {"train_config_name": "selected", "selected_from": {
        "study_id": study_id, "config_name": chosen_name, "config_hash": effective_hash,
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest()}}


def identity_snapshot(cfg: TrainConfig | None = None) -> dict:
    here = Path(__file__).resolve().parent
    cfg = cfg if cfg is not None else TrainConfig()
    return {
        "stageb_version": STAGEB_VERSION, "tokenizer_version": config.TOKENIZER_VERSION,
        "train_config": asdict(cfg), "config_hash": compute_config_hash(cfg),
        "source_sha256": source_sha256(here / "stageb_data.py", here / "stageb_model.py",
                                       here / "stageb_train.py", here / "stageb_infer.py"),
    }


def get_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def split_path(run_dir: Path, dataset: str, normalize: bool) -> Path:
    norm_part = "norm" if normalize else "raw"
    return run_dir / "splits" / f"{dataset}_{norm_part}_pooled.json"


def get_or_build_split(run_dir: Path, dataset: str, normalize: bool,
                       loaded: "data.LoadedDataset", split_source: Path | None = None) -> tuple[list[str], list[str]]:
    """v1.7: `split_source`, if given, is another Stage B run whose official
    split this run must match exactly (docs/plans/STAGE_C_AGENT_INSTRUCTIONS.md
    §1.3 -- stage_b_003 must train/select on stage_b_002's official split, not
    a fresh one). The split is still recomputed here (deterministic given the
    same seed and recording list), then checked against the source file
    rather than blindly copied, so a silent dataset/recording-list drift is
    caught instead of masked."""
    path = split_path(run_dir, dataset, normalize)
    if path.exists():
        blob = json.loads(path.read_text())
        return blob["train"], blob["val"]
    fit_names = [m.name for m in loaded.metas]
    frame_counts = {m.name: m.frame_count for m in loaded.metas}
    train_names, val_names = build_validation_split(fit_names, frame_counts, seed=config.SAMPLING_SEED)
    verified_against = None
    if split_source is not None:
        source_path = split_path(split_source, dataset, normalize)
        if not source_path.exists():
            raise SystemExit(f"--split-run's split file {source_path} does not exist")
        source_blob = json.loads(source_path.read_text())
        if source_blob["train"] != train_names or source_blob["val"] != val_names:
            raise SystemExit(
                f"recomputing the {dataset}/{'norm' if normalize else 'raw'} split does not reproduce "
                f"{source_path} exactly -- the dataset, recording list, or split code changed since that "
                "run; this run cannot reuse its official split as required")
        verified_against = str(source_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"train": train_names, "val": val_names, "seed": config.SAMPLING_SEED,
                                "verified_against": verified_against}, indent=2))
    return train_names, val_names


def cells_for(protocol: str) -> list[tuple[str, bool, int]]:
    """(dataset, normalize, seed), alternating datasets, per plan §3."""
    out = []
    for normalize in config.NORMALIZATIONS:
        for dataset in config.DATASETS:
            for seed in config.SEEDS:
                out.append((dataset, normalize, seed))
    return out


def cell_key(dataset: str, normalize: bool, seed: int) -> str:
    return f"{dataset}_{'norm' if normalize else 'raw'}_pooled_seed{seed}"


def codes_sha256(names: list[str], codes_by_name: dict[str, np.ndarray]) -> str:
    """v1.7: identifies the exact code sequences a checkpoint/embedding set
    was produced from, over all recordings in a fixed (sorted) order --
    Stage C's continuity gate compares this against a freshly rebuilt
    tokenizer cache before trusting stage_b_003's embeddings as paired with
    Stage A's grouping (docs/plans/STAGE_C_AGENT_INSTRUCTIONS.md §1.3)."""
    h = hashlib.sha256()
    for name in sorted(names):
        h.update(name.encode("utf-8"))
        h.update(codes_by_name[name].astype(np.int64).tobytes())
    return h.hexdigest()


def run_one(dataset: str, normalize: bool, seed: int, run_dir: Path, device: torch.device,
           cfg: TrainConfig = TrainConfig(), train_config_name: str = "v1.5",
           split_source: Path | None = None, deadline: float | None = None) -> dict:
    loaded = data.load_dataset(dataset)
    arrays = data.prepare_arrays(loaded, normalize)
    tok = data.build_or_load_tokenizer(loaded, arrays, normalize, "pooled", None, seed)
    codes_by_name = {loaded.metas[i].name: np.asarray(tok.codes_fine[i], dtype=np.int64)
                     for i in range(len(loaded.metas))}
    lengths_by_name = {m.name: m.lengths.astype(np.float32) for m in loaded.metas}

    train_names, val_names = get_or_build_split(run_dir, dataset, normalize, loaded, split_source)
    train_chunks = make_chunks({n: codes_by_name[n] for n in train_names}, lengths_by_name)
    val_chunks = make_chunks({n: codes_by_name[n] for n in val_names}, lengths_by_name)

    model, info = train(train_chunks, val_chunks, config.K_FINE, seed, device, cfg=cfg, deadline=deadline)
    ckpt_path = run_dir / "checkpoints" / f"{cell_key(dataset, normalize, seed)}.pt"
    all_names = [m.name for m in loaded.metas]
    codes_hash = codes_sha256(all_names, codes_by_name)
    extra = {"dataset": dataset, "normalize": normalize, "seed": seed, "protocol": "pooled",
             "tokenizer_version": config.TOKENIZER_VERSION, "stageb_version": STAGEB_VERSION,
             "train_config_name": train_config_name, "config_hash": compute_config_hash(cfg),
             "train_recordings": train_names, "val_recordings": val_names, "device": str(device),
             "codes_sha256": codes_hash, "cache_identity": loaded.core_dataset.cache_identity}
    save_checkpoint(ckpt_path, model, info, extra)

    embeddings_dir = run_dir / "embeddings" / cell_key(dataset, normalize, seed)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    val_set = set(val_names)
    adj_cos, adj_same, val_embs = [], [], []
    index = {"recording_order": [], "lengths": {}, "checkpoint_sha256": None,
             "cache_identity": loaded.core_dataset.cache_identity, "codes_sha256": codes_hash,
             "config_hash": compute_config_hash(cfg), "position_mode": cfg.position_mode,
             "extract_embeddings_sha256": source_sha256(Path(__file__).resolve().parent / "stageb_infer.py")["stageb_infer.py"]}
    for name in all_names:
        codes = codes_by_name[name]
        unit, zero = extract_embeddings(model, codes, device)
        np.savez_compressed(embeddings_dir / f"{name}.npz", embedding=unit, zero_flags=zero)
        index["recording_order"].append(name)
        index["lengths"][name] = int(len(codes))
        if name in val_set:
            adj_cos.append((unit[1:] * unit[:-1]).sum(1))
            adj_same.append(codes[1:] == codes[:-1])
            val_embs.append(unit)
    index["checkpoint_sha256"] = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    (embeddings_dir / "INDEX.json").write_text(json.dumps(index, indent=2))
    info["embedding_diagnostics"] = embedding_diagnostics(adj_cos, adj_same, val_embs)
    return info


def embedding_diagnostics(adj_cos, adj_same, val_embs) -> dict:
    """Label-free, on validation recordings: adjacent-window cosine of the
    embeddings vs the fraction of adjacent identical codes (temporal
    smoothing), and effective rank (collapse check)."""
    emb = np.concatenate(val_embs).astype(np.float64)
    s = np.linalg.svd(emb - emb.mean(0), compute_uv=False)
    p = s ** 2 / (s ** 2).sum()
    return {"adjacent_cosine": float(np.concatenate(adj_cos).mean()),
            "adjacent_same_code": float(np.concatenate(adj_same).mean()),
            "effective_rank": float(np.exp(-(p * np.log(p + 1e-12)).sum()))}


def cmd_run(args: argparse.Namespace) -> None:
    cfg, provenance = resolve_train_config(args.train_config, args.study_id)
    run_dir = OUT_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    identity = identity_snapshot(cfg)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        mismatched = [f for f in IDENTITY_FIELDS if manifest.get(f) != identity[f]]
        if mismatched:
            raise SystemExit(
                f"{run_dir} was created under a different Stage B identity ({mismatched} differ from the "
                "current code/config: old masks, architecture, or optimizer settings may not match what "
                "this process would produce). Start a new --run-id rather than resuming or mixing results.")
    else:
        manifest = {"run_id": args.run_id, "seeds": list(config.SEEDS), "launches": [],
                    "train_config_provenance": provenance, **identity}
    manifest["launches"].append({"started_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                 "device": args.device, "host": platform.node(),
                                 "torch": torch.__version__, "resume": bool(args.resume),
                                 "train_config": args.train_config, "split_run": args.split_run})
    manifest_path.write_text(json.dumps(manifest, indent=2))

    device = get_device(args.device)
    cells = cells_for("pooled")
    split_source = OUT_ROOT / args.split_run if args.split_run else None
    results_path = run_dir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    if results and not args.resume:
        raise SystemExit(f"{results_path} already has results; pass --resume or choose a new --run-id")
    todo = [c for c in cells if cell_key(*c) not in results]
    deadline = time.perf_counter() + args.hours * 3600.0 - 15 * 60 if args.hours else None
    print(f"{len(cells) - len(todo)} cells already done, {len(todo)} to train "
         f"(train_config={args.train_config})", flush=True)
    for dataset, normalize, seed in todo:
        if deadline is not None and time.perf_counter() > deadline:
            print(f"Deadline reached; stopping before {cell_key(dataset, normalize, seed)}", flush=True)
            break
        key = cell_key(dataset, normalize, seed)
        print(f"[stage B] training {key} on {device} ...", flush=True)
        try:
            info = run_one(dataset, normalize, seed, run_dir, device, cfg,
                           provenance["train_config_name"], split_source, deadline)
        except DeadlineExceeded:
            print(f"Deadline reached mid-training of {key}; not saved, stopping", flush=True)
            break
        results[key] = {k: info[k] for k in ("best_epoch", "best_val_loss", "epochs_run", "stopped_by",
                                             "model_masked_acc", "beats_neighbor_baseline", "diagnostics",
                                             "embedding_diagnostics", "runtime_seconds")}
        results[key]["best_step"] = info.get("best_step")
        results[key]["steps_run"] = info.get("steps_run")
        results[key]["config_hash"] = compute_config_hash(cfg)
        results_path.write_text(json.dumps(results, indent=2))
        print(f"    best_val_loss={info['best_val_loss']:.4f} at epoch {info['best_epoch']} "
             f"({info['epochs_run']} run), freq_baseline={info['diagnostics']['frequency_baseline_acc']:.3f}, "
             f"neighbor_baseline={info['diagnostics']['neighbor_baseline_acc']:.3f}, "
             f"model_acc={info['model_masked_acc']:.3f}, stopped_by={info['stopped_by']}, "
             f"{info['runtime_seconds']:.0f}s", flush=True)
    failing = [k for k, v in results.items() if not v.get("beats_neighbor_baseline")]
    print(f"Stage B run {args.run_id}: {len(results)}/{len(cells)} cells trained.")
    print("Diagnostic (model vs neighbor-copy baseline on validation masked accuracy; NOT a Stage C gate -- "
          "see plan §4 revision): " + ("all cells beat copy" if not failing else f"below copy: {failing}"))


def cmd_diagnostics(args: argparse.Namespace) -> None:
    """v1.7 (docs/plans/STAGE_C_AGENT_INSTRUCTIONS.md §1.4): subject-
    association diagnostics on already-saved embeddings, from the plan's §4
    (not computed by earlier Stage B versions). Label-free in the sense that
    fitting/checkpoint selection never sees subject IDs; here they are read
    only to report cosine-neighbor association, never to select anything."""
    run_dir = OUT_ROOT / args.run_id
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise SystemExit(f"{results_path} not found; run the cells first")
    out = {}
    rng = np.random.default_rng(config.SAMPLING_SEED)
    for dataset, normalize, seed in cells_for("pooled"):
        key = cell_key(dataset, normalize, seed)
        emb_dir = run_dir / "embeddings" / key
        if not emb_dir.exists():
            continue
        loaded = data.load_dataset(dataset)
        subject_of = {m.name: m.subject for m in loaded.metas}
        names, embs, rec_ids, subj_ids = [], [], [], []
        for m in loaded.metas:
            path = emb_dir / f"{m.name}.npz"
            if not path.exists():
                continue
            blob = np.load(path)
            unit = blob["embedding"]
            embs.append(unit)
            rec_ids += [m.name] * len(unit)
            subj_ids += [subject_of[m.name]] * len(unit)
        if not embs:
            continue
        emb = np.concatenate(embs).astype(np.float64)
        rec_ids = np.asarray(rec_ids)
        subj_ids = np.asarray(subj_ids)
        n_sample = min(10_000, len(emb))
        sample = rng.choice(len(emb), size=n_sample, replace=False)
        out[key] = nearest_neighbor_diagnostics(emb, rec_ids, subj_ids, sample)
        print(f"[diagnostics] {key}: n={n_sample} same_rec={out[key]['same_recording_rate']:.3f} "
             f"same_subj={out[key]['same_subject_rate']:.3f} "
             f"same_subj_excl_rec={out[key]['same_subject_rate_excl_same_recording']:.3f}", flush=True)
    (run_dir / "subject_association.json").write_text(json.dumps(out, indent=2))


def nearest_neighbor_diagnostics(emb: np.ndarray, rec_ids: np.ndarray, subj_ids: np.ndarray,
                                 sample: np.ndarray, k: int = 10) -> dict:
    """Cosine 10-NN same-recording/same-subject rates for `sample` points
    against the full population (self excluded), plus the same rates after
    excluding same-recording candidates, and the reference (sampling-
    composition) rates one would see from a uniform random neighbor."""
    norm = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-12)
    sims = norm[sample] @ norm.T  # [n_sample, N]
    same_rec, same_subj, same_subj_excl = [], [], []
    for row, i in enumerate(sample):
        s = sims[row].copy()
        s[i] = -np.inf  # exclude self
        order = np.argsort(-s)[:k]
        same_rec.append((rec_ids[order] == rec_ids[i]).mean())
        same_subj.append((subj_ids[order] == subj_ids[i]).mean())
        masked = s.copy()
        masked[rec_ids == rec_ids[i]] = -np.inf
        order_excl = np.argsort(-masked)[:k]
        same_subj_excl.append((subj_ids[order_excl] == subj_ids[i]).mean())
    n = len(emb)
    ref_same_rec = float(np.mean([(rec_ids == rec_ids[i]).sum() - 1 for i in sample])) / max(1, n - 1)
    ref_same_subj = float(np.mean([(subj_ids == subj_ids[i]).sum() - 1 for i in sample])) / max(1, n - 1)
    return {
        "n_sampled": int(len(sample)), "k": k,
        "same_recording_rate": float(np.mean(same_rec)),
        "same_subject_rate": float(np.mean(same_subj)),
        "same_subject_rate_excl_same_recording": float(np.mean(same_subj_excl)),
        "reference_same_recording_rate": ref_same_rec,
        "reference_same_subject_rate": ref_same_subj,
    }


def cmd_dry_run(args: argparse.Namespace) -> None:
    cfg, provenance = resolve_train_config(args.train_config, args.study_id)
    print(f"Effective TrainConfig ({provenance['train_config_name']}, hash {compute_config_hash(cfg)}): "
         f"{asdict(cfg)}")
    cells = cells_for("pooled")
    print(f"Stage B pooled: {len(cells)} models to train (dataset x normalize x seed)")
    for c in cells:
        print(f"  {cell_key(*c)}")
    print("Architecture: d_model=128, layers=4, heads=4, ff=512, dropout=0.1, max_context=128")
    print("Training: AdamW lr=3e-4 wd=0.01 batch=64 clip=1.0 max_epochs=500 patience=8 (v1.5)")


def cmd_preflight(args: argparse.Namespace) -> None:
    ok = True
    for dataset in config.DATASETS:
        try:
            data.verify_provenance(dataset)
            print(f"[PASS] {dataset} provenance gate")
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {dataset} provenance gate: {exc}")
            ok = False
    needed = [data._cache_path(d, n, "pooled", None, s) for d in config.DATASETS
              for n in config.NORMALIZATIONS for s in config.SEEDS]
    present = sum(p.exists() for p in needed)
    print(f"[{'PASS' if present == len(needed) else 'WARN'}] tokenizer caches {present}/{len(needed)} present "
         "(missing ones are rebuilt from Stage A's cache, which changes nothing about the codes)")
    device_ok = torch.cuda.is_available() or args.device != "cuda"
    print(f"[{'PASS' if device_ok else 'FAIL'}] device {args.device} (CUDA available: {torch.cuda.is_available()})")
    ok &= device_ok
    if args.train_config == "selected":
        try:
            _, provenance = resolve_train_config("selected", args.study_id)
            print(f"[PASS] --train-config selected resolves to "
                 f"{provenance['selected_from']['config_name']} (hash {provenance['selected_from']['config_hash']})")
        except SystemExit as exc:
            print(f"[FAIL] --train-config selected: {exc}")
            ok = False
    split_run_dir = OUT_ROOT / args.split_run if args.split_run else None
    if split_run_dir is not None:
        check_ok = split_run_dir.exists() and any(split_run_dir.glob("splits/*.json"))
        print(f"[{'PASS' if check_ok else 'FAIL'}] --split-run {args.split_run} has a splits/ directory")
        ok &= check_ok
    if not ok:
        raise SystemExit(1)


def cmd_validate_only(_args: argparse.Namespace) -> None:
    from script.action_transport import tests_stageb
    tests_stageb.run_all()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage B runner")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--diagnostics", action="store_true",
                      help="compute subject-association NN diagnostics on an existing run's saved embeddings")
    parser.add_argument("--hours", type=float, default=None)
    parser.add_argument("--run-id", dest="run_id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--train-config", dest="train_config", choices=["v1.5", "selected"], default="v1.5",
                        help="'v1.5': the original default TrainConfig(). 'selected': the optstudy-chosen "
                             "configuration (stage_b_003; docs/plans/STAGE_C_AGENT_INSTRUCTIONS.md Part 1)")
    parser.add_argument("--study-id", dest="study_id", default="optstudy_002",
                        help="the optstudy directory to resolve --train-config selected against")
    parser.add_argument("--split-run", dest="split_run", default=None,
                        help="reuse (and verify) this run's official train/val split instead of building a "
                             "fresh one -- required for stage_b_003 to match stage_b_002's official split")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.set_num_threads(min(config.CPU_THREAD_CAP, os.cpu_count() or 1))
    if args.validate_only:
        return cmd_validate_only(args)
    if args.preflight:
        return cmd_preflight(args)
    if args.dry_run:
        return cmd_dry_run(args)
    if args.diagnostics:
        if args.run_id is None:
            raise SystemExit("--diagnostics requires --run-id")
        return cmd_diagnostics(args)
    if args.hours is None or args.run_id is None:
        raise SystemExit("A full run requires --hours and --run-id")
    cmd_run(args)


if __name__ == "__main__":
    main()
