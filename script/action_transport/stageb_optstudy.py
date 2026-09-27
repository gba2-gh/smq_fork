"""Stage B optimization study (v1.6): why does B-code converge below the
copy-the-nearest-unmasked-code baseline (plan §4, stage_b_002)? Label-free
throughout: only code IDs and window lengths are used, never action labels.

Phases
------
0a overfit      Can the model memorize 16 chunks (train masks resample each step, as in
                normal training; the SAME 16 chunks are the eval set, with a fixed eval
                mask, so memorization is read off held-fixed positions)?
                If not, the setup cannot fit anything -> a bug or a broken
                optimizer configuration, not a data limit.
0b synthetic    A synthetic sequence of code runs (geometric lengths, like
                the real repetition rate), same masking. Copying is nearly
                optimal here, so a working model must at least match the copy
                baseline. Separates "cannot learn to copy" from "real data is
                hard".
1  sweep        Real data, 4 cells (hugadb/lara x raw/norm, seed 111), a
                fixed grid of optimization factors, in step mode. Trained on
                the official training recordings minus a tuning split;
                selected on the tuning split. The official validation split
                (used later for the final gate) is never touched here.

Selection rule (declared before the sweep runs; written to DESIGN.json):
choose ONE configuration for all cells, the one with the lowest mean tuning
cross-entropy (the training objective) across the 4 cells. Accuracy margins
over the copy baseline are reported, not optimized. If no configuration
beats the copy baseline on tuning accuracy in all 4 cells, the study says so
and recommends stopping Stage B rather than widening the search.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.action_transport import config, data
from script.action_transport.run_stage_b import OUT_ROOT, get_device, get_or_build_split, source_sha256
from script.action_transport.stageb_configs import BASELINE, GRID, STEP, config_hash, grid_configs
from script.action_transport.stageb_data import Chunk, build_validation_split, make_chunks
from script.action_transport.stageb_train import TrainConfig, evaluate_masked, fixed_masks, train

DESIGN_AMENDMENT = (
    "Amendment 1 (before any Phase 1 result existed): dropout factor removed (fixed at 0.1) and "
    "position_mode added. Reason: Phase 0 showed the model memorizes but fails to learn copying; Phase 0c's "
    "ALiBi rows were invalid (torch 2.1 eval fast path dropped the attention bias; fixed and tested); the "
    "corrected synthetic copy test showed ALiBi reaching near copy level (500 codes: 0.464 vs copy 0.482; "
    "sinusoidal 0.389). Dropout was not implicated by any diagnostic.")
CELLS = [(d, n) for n in config.NORMALIZATIONS for d in config.DATASETS]
SEED = 111


# --- phase 0 --------------------------------------------------------------------------

def phase0_overfit(chunks: list[Chunk], device, configs: dict[str, TrainConfig]) -> dict:
    """Memorization: train and evaluate on the SAME 16 chunks. Training masks
    still resample every step (ordinary training); only the evaluation mask
    is fixed (via `train`'s val_chunks=train_chunks, mask_base). A model that
    cannot reach near-zero eval loss here cannot fit anything at all."""
    tiny = chunks[:16]
    out = {}
    for name, cfg in configs.items():
        cfg = replace(cfg, batch_size=16, max_steps=2000, eval_every=100, patience_steps=10**9)
        # validate on the training chunks themselves: memorization, not generalization
        model, info = train(tiny, tiny, config.K_FINE, SEED, device, cfg, probe_chunks=tiny)
        out[name] = {"final_train_loss": info["history"][-1]["train_loss"],
                     "eval_loss": info["best_val_loss"], "eval_acc": info["model_masked_acc"],
                     "copy_acc": info["diagnostics"]["neighbor_baseline_acc"]}
        print(f"  [overfit] {name:28} eval_loss={out[name]['eval_loss']:.3f} "
              f"acc={out[name]['eval_acc']:.3f} (copy {out[name]['copy_acc']:.3f})", flush=True)
    return out


def synthetic_codes(n_recordings: int, length: int, vocab: int, mean_run: float, rng) -> dict[str, np.ndarray]:
    """Runs of identical codes with geometric lengths (mean `mean_run`);
    the next run's code is uniform over the vocabulary."""
    out = {}
    p = 1.0 / mean_run
    for r in range(n_recordings):
        codes = []
        while len(codes) < length:
            codes += [int(rng.integers(vocab))] * int(rng.geometric(p))
        out[f"syn{r:03d}"] = np.asarray(codes[:length], dtype=np.int64)
    return out


def phase0_synthetic(device, configs: dict[str, TrainConfig], mean_run: float) -> dict:
    rng = np.random.default_rng(SEED)
    codes = synthetic_codes(120, 400, config.K_FINE, mean_run, rng)
    lengths = {k: np.full(len(v), 15.0, dtype=np.float32) for k, v in codes.items()}
    names = sorted(codes)
    train_c = make_chunks({k: codes[k] for k in names[:100]}, lengths)
    val_c = make_chunks({k: codes[k] for k in names[100:]}, lengths)
    out = {}
    for name, cfg in configs.items():
        cfg = replace(cfg, max_steps=3000, eval_every=100, patience_steps=1000) if cfg.max_steps is None else cfg
        _, info = train(train_c, val_c, config.K_FINE, SEED, device, cfg)
        out[name] = {"val_loss": info["best_val_loss"], "acc": info["model_masked_acc"],
                     "copy_acc": info["diagnostics"]["neighbor_baseline_acc"], "by_depth": info["val_by_depth"],
                     "stopped_by": info["stopped_by"], "steps": info["steps_run"]}
        print(f"  [synthetic run={mean_run}] {name:28} acc={out[name]['acc']:.3f} "
              f"(copy {out[name]['copy_acc']:.3f}) steps={out[name]['steps']}", flush=True)
    return out


# --- phase 1 --------------------------------------------------------------------------

def cell_chunks(dataset: str, normalize: bool, split_dir: Path):
    """(study_train, study_tune, probe) chunks. The official validation split
    is excluded entirely; the tuning split is carved from the official
    training recordings with the same whole-recording rule and seed."""
    loaded = data.load_dataset(dataset)
    tok = data.build_or_load_tokenizer(loaded, data.prepare_arrays(loaded, normalize), normalize,
                                       "pooled", None, SEED)
    official_train, official_val = get_or_build_split(split_dir, dataset, normalize, loaded)
    counts = {m.name: m.frame_count for m in loaded.metas}
    study_train, study_tune = build_validation_split(official_train, counts, seed=config.SAMPLING_SEED + 1)
    assert not (set(study_tune) | set(study_train)) & set(official_val), "official validation must stay unseen"
    idx = {m.name: i for i, m in enumerate(loaded.metas)}
    codes = {m.name: np.asarray(tok.codes_fine[idx[m.name]], dtype=np.int64) for m in loaded.metas}
    lengths = {m.name: m.lengths.astype(np.float32) for m in loaded.metas}
    tr = make_chunks({n: codes[n] for n in study_train}, lengths)
    tu = make_chunks({n: codes[n] for n in study_tune}, lengths)
    probe_rng = np.random.default_rng(SEED)
    probe = [tr[i] for i in probe_rng.choice(len(tr), size=min(300, len(tr)), replace=False)]
    return tr, tu, probe, {"study_train": len(study_train), "study_tune": len(study_tune),
                           "official_val_excluded": len(official_val)}


def phase1_sweep(device, out_dir: Path, split_dir: Path, configs, cells) -> list[dict]:
    """v1.6 (B2 fix): each row records config_hash, checked against the
    CURRENT grid's hash for that config name before being treated as done
    -- resuming after a source or grid change re-trains rather than silently
    reusing a stale row. Learning curves are saved per fit under `curves/`."""
    rows_path = out_dir / "sweep.csv"
    current_hash = {name: config_hash(cfg) for name, cfg in configs}
    done: dict[tuple[str, str], str] = {}
    if rows_path.exists():
        with rows_path.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                done[(r["cell"], r["config"])] = r.get("config_hash", "")
    fields = ["cell", "config", "config_hash", "stopped_by", "steps_run", "best_step", "tune_loss", "tune_acc",
              "tune_copy_acc", "margin", "train_probe_loss", "train_probe_acc", "train_probe_copy_acc",
              "depth1_acc", "depth1_copy", "depth4plus_acc", "depth4plus_copy", "runtime_s", "lr", "norm_first",
              "schedule", "warmup_steps", "dropout", "tie_head", "position_mode"]
    curves_dir = out_dir / "curves"
    for dataset, normalize in cells:
        cell = f"{dataset}_{'norm' if normalize else 'raw'}"
        tr, tu, probe, sizes = cell_chunks(dataset, normalize, split_dir)
        print(f"[sweep] {cell}: {len(tr)} train / {len(tu)} tune chunks; {sizes}", flush=True)
        for name, cfg in configs:
            existing = done.get((cell, name))
            if existing is not None:
                if existing != current_hash[name]:
                    raise SystemExit(
                        f"{cell}/{name} was previously saved under a different effective configuration "
                        f"({existing} != {current_hash[name]}) -- e.g. a --smoke run, a since-changed grid, "
                        "or a source edit. Refusing to treat it as complete; start a new --study-id.")
                continue
            _, info = train(tr, tu, config.K_FINE, SEED, device, cfg, probe_chunks=probe)
            d = info["val_by_depth"]
            row = {"cell": cell, "config": name, "config_hash": current_hash[name],
                   "stopped_by": info["stopped_by"], "steps_run": info["steps_run"],
                   "best_step": info["best_step"], "tune_loss": info["best_val_loss"],
                   "tune_acc": info["model_masked_acc"], "tune_copy_acc": info["diagnostics"]["neighbor_baseline_acc"],
                   "margin": info["model_masked_acc"] - info["diagnostics"]["neighbor_baseline_acc"],
                   "train_probe_loss": info["train_probe"]["loss"], "train_probe_acc": info["train_probe"]["model_acc"],
                   "train_probe_copy_acc": info["train_probe"]["copy_acc"],
                   "depth1_acc": d.get("1", {}).get("model_acc"), "depth1_copy": d.get("1", {}).get("copy_acc"),
                   "depth4plus_acc": d.get("4+", {}).get("model_acc"), "depth4plus_copy": d.get("4+", {}).get("copy_acc"),
                   "runtime_s": info["runtime_seconds"], "lr": cfg.lr, "norm_first": cfg.norm_first,
                   "schedule": cfg.schedule, "warmup_steps": cfg.warmup_steps, "dropout": cfg.dropout,
                   "tie_head": cfg.tie_head, "position_mode": cfg.position_mode}
            new = not rows_path.exists()
            with rows_path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                if new:
                    writer.writeheader()
                writer.writerow(row)
            curves_dir.mkdir(parents=True, exist_ok=True)
            (curves_dir / f"{cell}__{name}.json").write_text(json.dumps(info["history"], indent=2))
            print(f"  {cell:12} {name:30} loss={row['tune_loss']:.3f} acc={row['tune_acc']:.3f} "
                  f"copy={row['tune_copy_acc']:.3f} margin={row['margin']:+.3f} "
                  f"train_acc={row['train_probe_acc']:.3f} {row['stopped_by']}@{row['steps_run']} "
                  f"{row['runtime_s']:.0f}s", flush=True)
    with rows_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def select(rows: list[dict], n_cells: int) -> dict:
    """The declared rule: lowest mean tuning CE across all cells, one config
    for every cell. Reports whether it (and any config) beats copy everywhere."""
    by_cfg: dict[str, list[dict]] = {}
    for r in rows:
        by_cfg.setdefault(r["config"], []).append(r)
    complete = {k: v for k, v in by_cfg.items() if len(v) == n_cells}
    ranked = sorted(complete, key=lambda k: np.mean([float(r["tune_loss"]) for r in complete[k]]))
    summary = []
    for k in ranked:
        rs = complete[k]
        summary.append({"config": k, "mean_tune_loss": float(np.mean([float(r["tune_loss"]) for r in rs])),
                        "mean_margin": float(np.mean([float(r["margin"]) for r in rs])),
                        "min_margin": float(min(float(r["margin"]) for r in rs)),
                        "beats_copy_all_cells": all(float(r["margin"]) > 0 for r in rs)})
    chosen = summary[0] if summary else None
    return {"rule": "lowest mean tuning cross-entropy across all cells; one config for all cells",
            "chosen": chosen, "any_config_beats_copy_all_cells": any(s["beats_copy_all_cells"] for s in summary),
            "ranking": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--phase", choices=["0", "0c", "1", "all"], default="all")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--split-run", default="stage_b_002",
                        help="Stage B run whose official train/val split to respect")
    parser.add_argument("--smoke", action="store_true", help="1 config x 1 cell x 300 steps (plumbing check)")
    args = parser.parse_args()
    # v1.6 (B2 fix): a smoke run can never share a directory with a full
    # run -- the previous code let `--smoke` and a full run reuse the same
    # --study-id, and phase1_sweep's (cell, config) resume key could not
    # tell a 300-step smoke row from a full 5000-step one. Force separate
    # namespaces instead of just documenting the risk.
    if args.smoke and not args.study_id.endswith("_smoke"):
        args.study_id += "_smoke"
    elif not args.smoke and args.study_id.endswith("_smoke"):
        raise SystemExit("a --study-id ending in _smoke is reserved for --smoke runs")
    device = get_device(args.device)
    out_dir = OUT_ROOT / args.study_id
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir = OUT_ROOT / args.split_run
    configs = grid_configs()
    cells = CELLS
    if args.smoke:
        configs = [(n, replace(c, max_steps=300, patience_steps=300)) for n, c in configs[:1]]
        cells = CELLS[:1]
    here = Path(__file__).resolve().parent
    source_hashes = source_sha256(here / "stageb_train.py", here / "stageb_model.py", here / "stageb_data.py")
    design = {"grid": {k: [str(v) for v in vs] for k, vs in GRID.items()}, "step_mode": STEP,
              "baseline_config": asdict(BASELINE), "cells": [f"{d}_{'norm' if n else 'raw'}" for d, n in CELLS],
              "seed": SEED, "selection_rule": select([], 0)["rule"], "smoke": bool(args.smoke),
              "source_sha256": source_hashes,
              "stop_rule": "if no config beats copy on tuning accuracy in all cells, recommend stopping Stage B",
              "official_validation_split": f"{args.split_run}/splits (never used by this study)",
              "labels_used": "none (code IDs and window lengths only)", "written": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    design["amendments"] = [DESIGN_AMENDMENT]
    design_path = out_dir / "DESIGN.json"
    if design_path.exists():
        existing = json.loads(design_path.read_text())
        sweep_exists = (out_dir / "sweep.csv").exists()
        if existing.get("amendments") != design["amendments"] and sweep_exists:
            raise SystemExit("DESIGN.json would change after Phase 1 results exist; start a new --study-id")
        if existing.get("source_sha256") != source_hashes and sweep_exists:
            raise SystemExit(
                f"{here}'s stageb_train/model/data.py changed since this study's sweep.csv rows were written "
                "(source_sha256 differs). Existing rows may not reflect the current code; start a new "
                "--study-id rather than silently mixing pre- and post-change results.")
        if existing.get("amendments") == design["amendments"]:
            design = existing  # keep the first-written record (including its original timestamp) as-is
    design_path.write_text(json.dumps(design, indent=2))
    print(f"device {device}; study dir {out_dir}{' (SMOKE)' if args.smoke else ''}", flush=True)

    if args.phase in ("0", "all"):
        pre_warm = replace(BASELINE, norm_first=True, schedule="cosine", warmup_steps=500, lr=1e-3,
                           max_steps=3000, eval_every=100, patience_steps=1000)
        p0_cfgs = {"baseline_v1.5": BASELINE, "baseline_v1.5_tied": replace(BASELINE, tie_head=True),
                   "preLN_cosine_wu500_lr1e-3": pre_warm,
                   "preLN_cosine_wu500_lr1e-3_tied": replace(pre_warm, tie_head=True)}
        tr, _, _, _ = cell_chunks("hugadb", False, split_dir)
        phase0 = {"overfit": phase0_overfit(tr, device, p0_cfgs)}
        for mean_run in (2.0, 4.0):
            phase0[f"synthetic_meanrun{mean_run:g}"] = phase0_synthetic(device, p0_cfgs, mean_run)
        (out_dir / "phase0.json").write_text(json.dumps(phase0, indent=2))
    if args.phase in ("0c",):
        # Plateau vs inductive bias: long budget, no early stopping, on synthetic copy data.
        long = dict(max_steps=15000, eval_every=500, patience_steps=10**9)
        base = replace(BASELINE, norm_first=True, schedule="cosine", warmup_steps=500, lr=1e-3, tie_head=True, **long)
        p0c_cfgs = {"baseline_v1.5_long": replace(BASELINE, **long),
                    "preLN_cosine_lr1e-3_tied_long": base,
                    "preLN_cosine_lr1e-3_tied_alibi_long": replace(base, position_mode="alibi")}
        phase0c = {"note": "rerun after the ALiBi eval fast-path fix; the earlier file is kept as "
                           "phase0c_INVALID_alibi_fastpath_bug.json"}
        for mean_run in (2.0, 4.0):
            phase0c[f"synthetic_meanrun{mean_run:g}"] = phase0_synthetic(device, p0c_cfgs, mean_run)
        (out_dir / "phase0c.json").write_text(json.dumps(phase0c, indent=2))
    if args.phase in ("1", "all"):
        rows = phase1_sweep(device, out_dir, split_dir, configs, cells)
        result = select(rows, len(cells))
        (out_dir / "selection.json").write_text(json.dumps(result, indent=2))
        chosen = result["chosen"]
        print(f"\nSelected (declared rule): {chosen}")
        print(f"Any config beats copy in all cells: {result['any_config_beats_copy_all_cells']}")


if __name__ == "__main__":
    main()
