"""Stage A orchestrator: validate-only, dry-run, timing-probe, and full
pooled/subject_disjoint runs with resume and an explicit runtime budget
(instructions §8). The user launches full runs; this module implements the
runner, not a launch decision.

NOTE ON THREAD CAPS: numerical-library thread env vars (OPENBLAS/MKL/OMP)
must be set before numpy/torch are imported, i.e. before this process
starts. run_stage_a.ps1 sets them prior to invoking python. `main()` also
calls torch.set_num_threads defensively, which is necessary but not
sufficient for numpy/BLAS.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.action_transport import categorical, config, continuous, data, evaluate, permute, smoothing, transport
from script.action_transport.budget import DeadlineExceeded, check_deadline

REPORT_RESERVE_SECONDS = 30 * 60


def set_thread_caps() -> int:
    cap = min(config.CPU_THREAD_CAP, os.cpu_count() or 1)
    torch.set_num_threads(cap)
    return cap


def get_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("-Device cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cpu")


# --- dataset/tokenizer caching within a run ---------------------------------

_LOADED: dict[str, "data.LoadedDataset"] = {}
_ARRAYS: dict[tuple[str, bool], list] = {}


def get_loaded(dataset: str) -> "data.LoadedDataset":
    if dataset not in _LOADED:
        _LOADED[dataset] = data.load_dataset(dataset)
    return _LOADED[dataset]


def get_arrays(dataset: str, normalize: bool) -> list:
    key = (dataset, normalize)
    if key not in _ARRAYS:
        _ARRAYS[key] = data.prepare_arrays(get_loaded(dataset), normalize)
    return _ARRAYS[key]


def get_tokenizer(cell: config.CellId) -> "data.Tokenizer":
    loaded = get_loaded(cell.dataset)
    arrays = get_arrays(cell.dataset, cell.normalize)
    return data.build_or_load_tokenizer(loaded, arrays, cell.normalize, cell.protocol, cell.fold, cell.seed)


# --- input assembly ---------------------------------------------------------

def categorical_inputs(loaded: "data.LoadedDataset", codes: list[np.ndarray],
                       indices: tuple[int, ...]) -> list[categorical.RecordingInput]:
    fps = config.FPS[loaded.dataset]
    window = config.WINDOW[loaded.dataset]
    return [
        categorical.RecordingInput(loaded.metas[i].name, np.asarray(codes[i], dtype=np.int32),
                                   loaded.metas[i].lengths, fps, window)
        for i in indices
    ]


def continuous_inputs(loaded: "data.LoadedDataset", tok: "data.Tokenizer",
                      indices: tuple[int, ...]) -> list[continuous.ContinuousRecordingInput]:
    fps = config.FPS[loaded.dataset]
    window = config.WINDOW[loaded.dataset]
    out = []
    for i in indices:
        z, zero = continuous.l2_normalize(np.asarray(tok.pca_windows[i], dtype=np.float64))
        out.append(continuous.ContinuousRecordingInput(
            loaded.metas[i].name, np.asarray(tok.codes_fine[i], dtype=np.int32),
            loaded.metas[i].lengths, fps, window, z, zero))
    return out


def coeffs_for(setting: config.CoeffSetting, arm: str) -> transport.Coeffs:
    beta = 0.0 if arm == "no_temporal" else setting.beta
    return transport.Coeffs(setting.a, beta, setting.lam, setting.eps)


# --- one fitted arm ----------------------------------------------------------

class CellRow(dict):
    """One row of cells.csv."""


def _empty_rows(cell: config.CellId, reason: str) -> list[CellRow]:
    rows = []
    for pid in config.prediction_set_ids(cell):
        rows.append(CellRow(prediction_set_id=pid, cell_key=cell.key(), dataset=cell.dataset,
                            normalize=cell.normalize, seed=cell.seed, protocol=cell.protocol,
                            fold=cell.fold, arm=cell.arm, status="unavailable", reason=reason,
                            MoF=None, Edit=None, **{"F1@10": None, "F1@25": None, "F1@50": None}))
    return rows


def run_categorical_arm(cell: config.CellId, loaded, tok, device, deadline, run_dir: Path):
    dataset = cell.dataset
    C = config.num_actions(dataset)
    is_kc = cell.arm == "categorical_asot_kc"
    K = config.vocab_size(dataset, cell.arm)
    codes_source = tok.codes_kc if is_kc else tok.codes_fine
    centers = tok.centers_kc if is_kc else tok.centers_fine

    fit_idx = data.fit_indices(loaded, cell.protocol, cell.fold)
    held_idx = data.held_out_indices(loaded, cell.protocol, cell.fold)
    all_idx = tuple(sorted(set(fit_idx) | set(held_idx)))

    fit_recs = categorical_inputs(loaded, codes_source, fit_idx)
    try:
        counts = categorical.frame_weighted_counts(fit_recs, K)
        grouping = categorical.build_grouping(centers, counts, C, cell.seed, is_kc_identity=is_kc)
    except categorical.UnavailableError as exc:
        return _empty_rows(cell, str(exc)), None

    theta0 = categorical.build_theta0(grouping, counts, C, K, config.eta(dataset))
    fit_coeffs = coeffs_for(config.FIT_SETTING, cell.arm)
    fit_result = categorical.fit_categorical(
        fit_recs, theta0, C, K, config.eta(dataset), fit_coeffs, config.FIT_INNER_STEPS,
        config.FIT_OUTER_CAP, config.FIT_OUTER_PATIENCE, config.FIT_OUTER_RELTOL, device, deadline)
    if fit_result.outer_status == "invalid":
        return _empty_rows(cell, "fitting produced a nonfinite/backtracking-failed objective"), None

    all_recs = categorical_inputs(loaded, codes_source, all_idx)
    rows: list[CellRow] = []
    filter_predictions: dict[str, dict[str, np.ndarray]] = {}
    predictions_by_setting: dict[str, dict[str, np.ndarray]] = {}
    for setting in config.SETTINGS:
        coeffs = coeffs_for(setting, cell.arm)
        final = categorical.infer_final(
            fit_result.theta, all_recs, C, fit_result.t_state, coeffs, device,
            config.FINAL_MAX_STEPS, config.FINAL_GRAD_TOL, config.FINAL_OBJ_RELTOL,
            config.FINAL_PATIENCE, config.BACKTRACK_MAX, deadline)
        predictions_by_setting[setting.name] = final.predictions
        rows.append(_score_and_row(cell, f"{cell.key()}__{setting.name}", loaded, all_idx,
                                   final.predictions, final.status, final.residuals, fit_result))
        if cell.arm == "no_temporal":
            filtered = {}
            for rec in all_recs:
                if rec.name not in final.predictions:
                    continue
                half_width = transport.kernel_half_width(rec.fps, rec.window, len(rec.codes))
                filtered[rec.name] = smoothing.mode_filter(final.predictions[rec.name], rec.lengths, half_width)
            filter_predictions[setting.name] = filtered
            rows.append(_score_and_row(cell, f"{cell.key()}_filter__{setting.name}", loaded, all_idx,
                                       filtered, {n: "ok" for n in filtered}, {n: None for n in filtered},
                                       fit_result, is_filter=True))

    artifacts = dict(theta=fit_result.theta, grouping=grouping, K=K, C=C)
    return rows, artifacts


def run_continuous_arm(cell: config.CellId, loaded, tok, device, deadline, run_dir: Path):
    dataset = cell.dataset
    C = config.num_actions(dataset)
    fit_idx = data.fit_indices(loaded, cell.protocol, cell.fold)
    held_idx = data.held_out_indices(loaded, cell.protocol, cell.fold)
    all_idx = tuple(sorted(set(fit_idx) | set(held_idx)))

    fit_recs = continuous_inputs(loaded, tok, fit_idx)
    counts = categorical.frame_weighted_counts(
        categorical_inputs(loaded, tok.codes_fine, fit_idx), config.K_FINE)
    try:
        grouping = categorical.build_grouping(tok.centers_fine, counts, C, cell.seed, is_kc_identity=False)
    except categorical.UnavailableError as exc:
        return _empty_rows(cell, str(exc)), None
    mu0, unavailable = continuous.build_mu0(grouping, fit_recs, C)
    if unavailable.any():
        return _empty_rows(cell, f"{int(unavailable.sum())} state(s) had a zero initial mean"), None

    fit_coeffs = coeffs_for(config.FIT_SETTING, cell.arm)
    fit_result = continuous.fit_continuous(
        fit_recs, mu0, C, fit_coeffs, config.FIT_INNER_STEPS, config.FIT_OUTER_CAP,
        config.FIT_OUTER_PATIENCE, config.FIT_OUTER_RELTOL, device, deadline)
    if fit_result.outer_status == "invalid":
        return _empty_rows(cell, "fitting produced a nonfinite/backtracking-failed objective"), None

    all_recs = continuous_inputs(loaded, tok, all_idx)
    rows: list[CellRow] = []
    for setting in config.SETTINGS:
        coeffs = coeffs_for(setting, cell.arm)
        final = continuous.infer_final(
            fit_result.mu, all_recs, C, coeffs, device, config.FINAL_MAX_STEPS, config.FINAL_GRAD_TOL,
            config.FINAL_OBJ_RELTOL, config.FINAL_PATIENCE, config.BACKTRACK_MAX,
            fit_result.t_state, deadline)
        rows.append(_score_and_row(cell, f"{cell.key()}__{setting.name}", loaded, all_idx,
                                   final.predictions, final.status, final.residuals, fit_result))
    artifacts = dict(mu=fit_result.mu, grouping=grouping, stale=fit_result.stale_prototypes)
    return rows, artifacts


def run_permutation_arm(cell: config.CellId, loaded, tok, device, deadline, run_dir: Path):
    """A1 refit on within-recording permuted codes, sharing A1's theta0/
    grouping, then restored to the original timeline before scoring
    (instructions §3, v1.3)."""
    dataset = cell.dataset
    C = config.num_actions(dataset)
    K = config.K_FINE
    all_idx = tuple(range(len(loaded.metas)))  # pooled only
    counts = categorical.frame_weighted_counts(
        categorical_inputs(loaded, tok.codes_fine, all_idx), K)
    try:
        grouping = categorical.build_grouping(tok.centers_fine, counts, C, cell.seed, is_kc_identity=False)
    except categorical.UnavailableError as exc:
        return _empty_rows(cell, str(exc)), None
    theta0 = categorical.build_theta0(grouping, counts, C, K, config.eta(dataset))

    rng = np.random.default_rng(cell.seed + config.PERMUTATION_SEED_OFFSET)
    window = config.WINDOW[dataset]
    perms: dict[str, np.ndarray] = {}
    permuted_recs = []
    for i in sorted(all_idx, key=lambda idx: loaded.metas[idx].name):
        meta = loaded.metas[i]
        perm = permute.build_permutation(meta.lengths, window, rng)
        perms[meta.name] = perm
        codes_perm = permute.apply_permutation(np.asarray(tok.codes_fine[i], dtype=np.int32), perm)
        permuted_recs.append(categorical.RecordingInput(meta.name, codes_perm, meta.lengths,
                                                         config.FPS[dataset], window))

    fit_coeffs = coeffs_for(config.FIT_SETTING, "categorical_asot")
    fit_result = categorical.fit_categorical(
        permuted_recs, theta0, C, K, config.eta(dataset), fit_coeffs, config.FIT_INNER_STEPS,
        config.FIT_OUTER_CAP, config.FIT_OUTER_PATIENCE, config.FIT_OUTER_RELTOL, device, deadline)
    if fit_result.outer_status == "invalid":
        return _empty_rows(cell, "permuted fit produced a nonfinite/backtracking-failed objective"), None

    rows: list[CellRow] = []
    for setting in config.SETTINGS:
        coeffs = coeffs_for(setting, "categorical_asot")
        final = categorical.infer_final(
            fit_result.theta, permuted_recs, C, fit_result.t_state, coeffs, device,
            config.FINAL_MAX_STEPS, config.FINAL_GRAD_TOL, config.FINAL_OBJ_RELTOL,
            config.FINAL_PATIENCE, config.BACKTRACK_MAX, deadline)
        restored = {name: permute.restore_states(states, perms[name])
                   for name, states in final.predictions.items()}
        rows.append(_score_and_row(cell, f"{cell.key()}__{setting.name}", loaded, all_idx,
                                   restored, final.status, final.residuals, fit_result))
    return rows, dict(theta=fit_result.theta, grouping=grouping, perms=perms)


def _score_and_row(cell: config.CellId, prediction_set_id: str, loaded, indices, predictions,
                   status_by_name, residuals_by_name, fit_result, is_filter: bool = False) -> CellRow:
    frame_preds, frame_gts, statuses, present_idx = [], [], [], []
    for i in indices:
        meta = loaded.metas[i]
        if meta.name not in predictions:
            continue
        frames = evaluate.broadcast_to_frames(predictions[meta.name], meta.starts, meta.lengths,
                                              meta.frame_count)
        frame_preds.append(frames)
        frame_gts.append(loaded.core_dataset.gts[i])
        statuses.append(status_by_name.get(meta.name, "ok"))
        present_idx.append(i)
    worst = "invalid"
    if statuses:
        order = {"converged": 0, "capped": 1, "ok": 0, "interrupted": 3, "invalid": 2}
        worst = max(statuses, key=lambda s: order.get(s, 2))
    if not frame_preds:
        row = CellRow(prediction_set_id=prediction_set_id, cell_key=cell.key(), dataset=cell.dataset,
                      normalize=cell.normalize, seed=cell.seed, protocol=cell.protocol, fold=cell.fold,
                      arm=cell.arm, status="invalid", reason="no recordings produced predictions",
                      MoF=None, Edit=None)
        row["F1@10"] = row["F1@25"] = row["F1@50"] = None
        return row
    if cell.protocol == "pooled":
        metrics = evaluate.score_pooled(frame_gts, frame_preds)
    else:
        fold_of = loaded.fold_of[list(present_idx)]
        metrics = evaluate.score_subject_disjoint(fold_of, frame_gts, frame_preds)
    fit_outer_status = getattr(fit_result, "outer_status", None) if not is_filter else None
    row = CellRow(prediction_set_id=prediction_set_id, cell_key=cell.key(), dataset=cell.dataset,
                  normalize=cell.normalize, seed=cell.seed, protocol=cell.protocol, fold=cell.fold,
                  arm=cell.arm, status=worst, reason="", fit_outer_status=fit_outer_status,
                  n_recordings=len(frame_preds), **metrics)
    return row


ARM_RUNNERS = {
    "no_temporal": run_categorical_arm,
    "categorical_asot": run_categorical_arm,
    "categorical_asot_kc": run_categorical_arm,
    "continuous_asot": run_continuous_arm,
    "permuted_asot": run_permutation_arm,
}


def run_one_cell(cell: config.CellId, device, deadline, run_dir: Path):
    loaded = get_loaded(cell.dataset)
    tok = get_tokenizer(cell)
    runner = ARM_RUNNERS[cell.arm]
    started = time.perf_counter()
    rows, artifacts = runner(cell, loaded, tok, device, deadline, run_dir)
    elapsed = time.perf_counter() - started
    for row in rows:
        row["runtime_seconds"] = elapsed
        row["device"] = str(device)
    if artifacts is not None:
        _save_artifacts(run_dir, cell, artifacts)
    _save_predictions(run_dir, cell, rows)
    return rows


def _save_artifacts(run_dir: Path, cell: config.CellId, artifacts: dict) -> None:
    path = run_dir / "artifacts" / f"{cell.key()}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.npz")
    clean = {k: v for k, v in artifacts.items() if not isinstance(v, dict)}
    np.savez_compressed(temp, **clean)
    os.replace(temp, path)


def _save_predictions(run_dir: Path, cell: config.CellId, rows: list[CellRow]) -> None:
    path = run_dir / "predictions" / f"{cell.key()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{k: (v if not isinstance(v, (np.floating, np.integer)) else v.item())
               for k, v in row.items()} for row in rows]
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temp, path)


# --- run manifest / cells.csv / resume --------------------------------------

CELLS_CSV_FIELDS = ["prediction_set_id", "cell_key", "dataset", "normalize", "seed", "protocol",
                    "fold", "arm", "status", "reason", "fit_outer_status", "n_recordings",
                    "runtime_seconds", "device", "MoF", "Edit", "F1@10", "F1@25", "F1@50"]


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def _load_completed(run_dir: Path) -> set[str]:
    path = run_dir / "cells.csv"
    if not path.exists():
        return set()
    import csv
    with path.open("r", newline="", encoding="utf-8") as fh:
        return {row["prediction_set_id"] for row in csv.DictReader(fh)}


def _append_cells_csv(run_dir: Path, rows: list[CellRow]) -> None:
    import csv
    path = run_dir / "cells.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CELLS_CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_not_run(run_dir: Path, entries: list[dict]) -> None:
    import csv
    path = run_dir / "not_run.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["cell_key", "dataset", "arm", "protocol", "reason"])
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry)


def write_manifest(run_dir: Path, args: argparse.Namespace, cells: list[config.CellId],
                   after_pooled_run: str | None) -> None:
    manifest = {
        "run_id": args.run_id, "protocol_version": config.PROTOCOL_VERSION,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": args.protocol, "device_requested": args.device,
        "hours_budget": args.hours, "after_pooled_run": after_pooled_run,
        "python": sys.version, "platform": platform.platform(),
        "packages": {"numpy": np.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__},
        "cpu_thread_cap": min(config.CPU_THREAD_CAP, os.cpu_count() or 1),
        "seeds": list(config.SEEDS), "n_cells": len(cells),
        "asot_pinned_commit": config.ASOT_PINNED_COMMIT,
    }
    _atomic_write(run_dir / "manifest.json", json.dumps(manifest, indent=2))


def write_planned_cells(run_dir: Path, cells: list[config.CellId]) -> None:
    import csv
    path = run_dir / "planned_cells.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cell_key", "dataset", "normalize", "seed", "protocol", "fold", "arm",
                         "prediction_set_ids"])
        for cell in cells:
            writer.writerow([cell.key(), cell.dataset, cell.normalize, cell.seed, cell.protocol,
                            cell.fold, cell.arm, ";".join(config.prediction_set_ids(cell))])


# --- CLI ---------------------------------------------------------------------

def cells_for_protocol(protocol: str) -> list[config.CellId]:
    if protocol == "pooled":
        return config.enumerate_pooled_cells() + config.enumerate_permutation_cells()
    if protocol == "subject_disjoint":
        return config.enumerate_subject_disjoint_cells()
    raise ValueError(protocol)


def cmd_dry_run(args: argparse.Namespace) -> None:
    cells = cells_for_protocol(args.protocol)
    n_pred_sets = sum(len(config.prediction_set_ids(c)) for c in cells)
    print(f"Protocol: {args.protocol}")
    print(f"Planned fits: {len(cells)}")
    print(f"Planned scored prediction sets: {n_pred_sets}")
    by_arm: dict[str, int] = {}
    for c in cells:
        by_arm[c.arm] = by_arm.get(c.arm, 0) + 1
    for arm, count in sorted(by_arm.items()):
        print(f"  {arm}: {count} fits")
    cache_dir = config.OUT_ROOT / "cache" / "tokenizers"
    n_cached = len(list(cache_dir.glob("*.npz"))) if cache_dir.exists() else 0
    print(f"Cached tokenizers on disk: {n_cached}")
    timings_path = config.OUT_ROOT / "cache" / "timings.json"
    if timings_path.exists():
        timings = json.loads(timings_path.read_text())
        per_fit = timings.get("median_seconds_per_fit")
        if per_fit:
            projected_hours = len(cells) * per_fit / 3600.0
            print(f"Timing-probe estimate: {per_fit:.1f}s/fit -> ~{projected_hours:.2f}h for this protocol "
                 f"(excludes reporting reserve of {REPORT_RESERVE_SECONDS/60:.0f} min)")
    else:
        print("No -TimingProbe results on disk yet; run with -TimingProbe first for a cost estimate.")


def cmd_timing_probe(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    results = []
    for dataset in config.DATASETS:
        loaded = get_loaded(dataset)
        small_idx = tuple(range(min(3, len(loaded.metas))))
        arrays = get_arrays(dataset, False)
        window = config.WINDOW[dataset]
        # Tiny tokenizer fit on the same small subset (label-blind timing only,
        # not a cached/valid tokenizer for real cells).
        from script.exp2round.q1q2 import core
        prepared = core.prepare_windows(arrays, window, small_idx, max_sample=2000, seed=config.SAMPLING_SEED)
        kmeans_fine = core.fit_vocabulary(prepared, config.K_FINE, config.SAMPLING_SEED)
        codes_fine = core.assign_windows(kmeans_fine, prepared.transformed)
        recs = categorical_inputs(loaded, codes_fine, small_idx)
        C = config.num_actions(dataset)
        counts = categorical.frame_weighted_counts(recs, config.K_FINE)
        try:
            grouping = categorical.build_grouping(kmeans_fine.cluster_centers_, counts, C,
                                                  config.SAMPLING_SEED, False)
        except categorical.UnavailableError:
            continue
        theta0 = categorical.build_theta0(grouping, counts, C, config.K_FINE, config.eta(dataset))
        coeffs = coeffs_for(config.FIT_SETTING, "categorical_asot")
        started = time.perf_counter()
        categorical.fit_categorical(recs, theta0, C, config.K_FINE, config.eta(dataset), coeffs,
                                    config.FIT_INNER_STEPS, outer_cap=3, outer_patience=config.FIT_OUTER_PATIENCE,
                                    outer_reltol=config.FIT_OUTER_RELTOL, device=device)
        elapsed = time.perf_counter() - started
        results.append({"dataset": dataset, "n_recordings": len(small_idx), "outer_iters": 3,
                        "seconds": elapsed, "seconds_per_recording_per_outer": elapsed / (len(small_idx) * 3)})
        print(f"[timing-probe] {dataset}: {elapsed:.2f}s for {len(small_idx)} recordings x 3 outer iters")
    if not results:
        print("Timing probe produced no results (all datasets unavailable).")
        return
    per_rec_outer = np.mean([r["seconds_per_recording_per_outer"] for r in results])
    # A rough per-fit estimate: ~median outer iterations we might expect before
    # capping (config.FIT_OUTER_CAP) times per-recording-outer cost times a
    # typical fitting-population recording count, plus a fixed final-inference
    # allowance. This is a projection, not a measurement of a real cell.
    estimate = {"per_recording_per_outer_seconds": float(per_rec_outer),
               "median_seconds_per_fit": float(per_rec_outer * 30 * config.FIT_OUTER_CAP),
               "probe_results": results, "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path = config.OUT_ROOT / "cache" / "timings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(estimate, indent=2))
    print(f"Saved timing estimate to {path}")
    print("This is a projection from a tiny label-blind probe, not a real-cell measurement.")


def cmd_validate_only(args: argparse.Namespace) -> None:
    from script.action_transport import tests as at_tests
    at_tests.run_all()


def cmd_run(args: argparse.Namespace) -> None:
    if args.protocol == "subject_disjoint" and not args.after_pooled_run:
        raise SystemExit("-Protocol subject_disjoint requires -AfterPooledRun <pooled run_id>")
    after_pooled_run = None
    if args.protocol == "subject_disjoint":
        pooled_dir = config.OUT_ROOT / args.after_pooled_run
        if not (pooled_dir / "REPORT.md").exists() and not (pooled_dir / "cells.csv").exists():
            raise SystemExit(f"named pooled run {args.after_pooled_run!r} has no report/cells.csv at {pooled_dir}")
        after_pooled_run = args.after_pooled_run

    set_thread_caps()
    device = get_device(args.device)
    run_dir = config.OUT_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cells = cells_for_protocol(args.protocol)
    write_planned_cells(run_dir, cells)
    write_manifest(run_dir, args, cells, after_pooled_run)

    completed = _load_completed(run_dir) if args.resume else set()
    if not args.resume and (run_dir / "cells.csv").exists():
        raise SystemExit(f"{run_dir}/cells.csv already exists; pass -Resume or choose a new -RunId")

    deadline_total = time.perf_counter() + args.hours * 3600.0
    reporting_deadline = deadline_total - REPORT_RESERVE_SECONDS
    not_run: list[dict] = []
    per_cell_seconds: list[float] = []

    for cell in cells:
        prediction_ids = config.prediction_set_ids(cell)
        if all(pid in completed for pid in prediction_ids):
            continue
        remaining = reporting_deadline - time.perf_counter()
        if remaining <= 0:
            not_run.append({"cell_key": cell.key(), "dataset": cell.dataset, "arm": cell.arm,
                           "protocol": cell.protocol, "reason": "compute deadline reached before this cell started"})
            continue
        if per_cell_seconds:
            projected = np.median(per_cell_seconds)
            if projected > remaining:
                not_run.append({"cell_key": cell.key(), "dataset": cell.dataset, "arm": cell.arm,
                               "protocol": cell.protocol,
                               "reason": f"projected {projected:.0f}s exceeds {remaining:.0f}s remaining"})
                continue
        started = time.perf_counter()
        try:
            rows = run_one_cell(cell, device, reporting_deadline, run_dir)
            _append_cells_csv(run_dir, rows)
            completed.update(config.prediction_set_ids(cell))
        except DeadlineExceeded:
            not_run.append({"cell_key": cell.key(), "dataset": cell.dataset, "arm": cell.arm,
                           "protocol": cell.protocol, "reason": "interrupted at cooperative deadline check"})
        per_cell_seconds.append(time.perf_counter() - started)

    _write_not_run(run_dir, not_run)
    print(f"Run {args.run_id} finished (or reached its deadline). "
         f"{len(completed)} prediction sets written, {len(not_run)} cells not run.")
    print(f"See {run_dir}/cells.csv and {run_dir}/not_run.csv.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage A runner")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timing-probe", action="store_true")
    parser.add_argument("--protocol", choices=["pooled", "subject_disjoint"], default="pooled")
    parser.add_argument("--after-pooled-run", dest="after_pooled_run", default=None)
    parser.add_argument("--hours", type=float, default=None)
    parser.add_argument("--run-id", dest="run_id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_thread_caps()
    if args.validate_only:
        cmd_validate_only(args)
        return
    if args.dry_run:
        cmd_dry_run(args)
        return
    if args.timing_probe:
        cmd_timing_probe(args)
        return
    if args.hours is None or args.run_id is None:
        raise SystemExit("A full run requires --hours and --run-id")
    if args.hours * 3600.0 <= REPORT_RESERVE_SECONDS:
        raise SystemExit(f"--hours must leave at least {REPORT_RESERVE_SECONDS/60:.0f} minutes for reporting")
    cmd_run(args)


if __name__ == "__main__":
    main()
