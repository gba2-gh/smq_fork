"""Stage A orchestrator: validate-only, preflight, dry-run, timing-probe, and
full pooled/subject_disjoint runs with resume and an explicit runtime budget
(instructions §8). The user launches full runs.

Thread caps: OPENBLAS/MKL/OMP env vars must be set before numpy/torch load,
i.e. before this process starts. run_stage_a.ps1 does that; main() also
calls torch.set_num_threads, which is necessary but not sufficient.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sklearn
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.action_transport import categorical, config, continuous, data, evaluate, permute, smoothing, transport
from script.action_transport.budget import DeadlineExceeded

REPORT_RESERVE_SECONDS = 30 * 60

CELLS_CSV_FIELDS = [
    "prediction_set_id", "cell_key", "dataset", "normalize", "seed", "protocol", "fold", "arm",
    "setting", "scope", "status", "reason",
    "fit_outer_status", "fit_outer_iterations", "fit_last_rel_change",
    "final_n_converged", "final_n_capped", "final_n_invalid", "final_residual_max",
    "final_residual_median_w", "final_steps_mean",
    "n_recordings", "runtime_seconds", "device",
    "MoF", "Edit", "F1@10", "F1@25", "F1@50",
    "pred_runs", "gt_runs", "run_ratio", "pred_run_median_s", "pred_run_p10_s", "pred_run_p90_s",
    "gt_run_median_s", "occupied_states", "occupancy_entropy", "max_state_fraction",
    "mi_state_subject",
]


def set_thread_caps() -> int:
    cap = min(config.CPU_THREAD_CAP, os.cpu_count() or 1)
    torch.set_num_threads(cap)
    return cap


def get_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("-Device cuda requested but CUDA is not available (no silent CPU fallback)")
        return torch.device("cuda")
    return torch.device("cpu")


# --- dataset/tokenizer caching within a process -----------------------------------

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
    return data.build_or_load_tokenizer(loaded, get_arrays(cell.dataset, cell.normalize),
                                        cell.normalize, cell.protocol, cell.fold, cell.seed)


# --- input assembly -----------------------------------------------------------------

def categorical_inputs(loaded, codes, indices) -> list[categorical.RecordingInput]:
    fps, window = config.FPS[loaded.dataset], config.WINDOW[loaded.dataset]
    return [categorical.RecordingInput(loaded.metas[i].name, np.asarray(codes[i], dtype=np.int32),
                                       loaded.metas[i].lengths, fps, window) for i in indices]


def continuous_inputs(loaded, tok, indices) -> list[continuous.ContinuousRecordingInput]:
    fps, window = config.FPS[loaded.dataset], config.WINDOW[loaded.dataset]
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


def final_solve(params, recs, num_states, cost_fn, warm, setting, arm, device, deadline):
    return categorical.infer_final(
        params, recs, num_states, cost_fn, warm, coeffs_for(setting, arm), device,
        config.FINAL_MAX_STEPS, config.FINAL_GRAD_TOL, config.FINAL_OBJ_RELTOL,
        config.FINAL_PATIENCE, config.BACKTRACK_MAX, deadline)


# --- prediction sets ------------------------------------------------------------------

@dataclass
class PredictionSet:
    prediction_set_id: str
    setting: str
    states: dict[str, np.ndarray]  # raw window states, ORIGINAL timeline
    final: "categorical.FinalResult | None"  # None for derived arms
    status_override: str | None = None
    reason: str = ""


@dataclass
class ArmOutput:
    prediction_sets: list[PredictionSet] = field(default_factory=list)
    fit: "categorical.FitResult | None" = None
    artifacts: dict = field(default_factory=dict)
    unavailable_reason: str | None = None


def _scoring_indices(loaded, cell: config.CellId) -> tuple[int, ...]:
    """Pooled: every recording. Subject-disjoint fold cell: held-out only."""
    return data.held_out_indices(loaded, cell.protocol, cell.fold)


def run_categorical_arm(cell, loaded, tok, device, deadline) -> ArmOutput:
    dataset = cell.dataset
    C = config.num_actions(dataset)
    is_kc = cell.arm == "categorical_asot_kc"
    K = config.vocab_size(dataset, cell.arm)
    codes = tok.codes_kc if is_kc else tok.codes_fine
    centers = tok.centers_kc if is_kc else tok.centers_fine
    fit_recs = categorical_inputs(loaded, codes, data.fit_indices(loaded, cell.protocol, cell.fold))
    counts = categorical.frame_weighted_counts(fit_recs, K)
    try:
        grouping = categorical.build_grouping(centers, counts, C, cell.seed, is_kc_identity=is_kc)
    except categorical.UnavailableError as exc:
        return ArmOutput(unavailable_reason=str(exc))
    theta0 = categorical.build_theta0(grouping, counts, C, K, config.eta(dataset))
    fit = categorical.fit_categorical(
        fit_recs, theta0, C, K, config.eta(dataset), coeffs_for(config.FIT_SETTING, cell.arm),
        config.FIT_INNER_STEPS, config.FIT_OUTER_CAP, config.FIT_OUTER_PATIENCE,
        config.FIT_OUTER_RELTOL, device, deadline)
    out = ArmOutput(fit=fit, artifacts=dict(theta=fit.params, theta0=theta0, grouping=grouping,
                                             code_counts=counts, objective_trace=np.asarray(fit.objective_trace)))
    if fit.outer_status == "invalid":
        out.unavailable_reason = "fitting hit a nonfinite value or backtracking failure"
        return out
    score_recs = categorical_inputs(loaded, codes, _scoring_indices(loaded, cell))
    for setting in config.SETTINGS:
        final = final_solve(fit.params, score_recs, C, categorical.categorical_cost_fn,
                            fit.log_t_state, setting, cell.arm, device, deadline)
        out.prediction_sets.append(PredictionSet(f"{cell.key()}__{setting.name}", setting.name,
                                                 final.states, final))
        if cell.arm == "no_temporal":
            filtered = {
                rec.name: smoothing.mode_filter(
                    final.states[rec.name], rec.lengths,
                    transport.kernel_half_width(rec.fps, rec.window, len(rec.codes)))
                for rec in score_recs if rec.name in final.states
            }
            out.prediction_sets.append(PredictionSet(
                f"{cell.key()}_filter__{setting.name}", setting.name, filtered, final,
                reason="derived from A0 raw states (no fit); status inherited from A0"))
    return out


def run_continuous_arm(cell, loaded, tok, device, deadline) -> ArmOutput:
    dataset = cell.dataset
    C = config.num_actions(dataset)
    fit_idx = data.fit_indices(loaded, cell.protocol, cell.fold)
    fit_recs = continuous_inputs(loaded, tok, fit_idx)
    counts = categorical.frame_weighted_counts(categorical_inputs(loaded, tok.codes_fine, fit_idx),
                                               config.K_FINE)
    try:
        grouping = categorical.build_grouping(tok.centers_fine, counts, C, cell.seed, is_kc_identity=False)
    except categorical.UnavailableError as exc:
        return ArmOutput(unavailable_reason=str(exc))
    mu0, zero_mean = continuous.build_mu0(grouping, fit_recs, C)
    if zero_mean.any():
        return ArmOutput(unavailable_reason=f"{int(zero_mean.sum())} state(s) had a zero initial mean")
    fit = continuous.fit_continuous(
        fit_recs, mu0, C, coeffs_for(config.FIT_SETTING, cell.arm), config.FIT_INNER_STEPS,
        config.FIT_OUTER_CAP, config.FIT_OUTER_PATIENCE, config.FIT_OUTER_RELTOL, device, deadline)
    out = ArmOutput(fit=fit, artifacts=dict(mu=fit.params, mu0=mu0, grouping=grouping,
                                             stale=np.asarray(fit.stale_prototypes, dtype=np.int32),
                                             objective_trace=np.asarray(fit.objective_trace)))
    if fit.outer_status == "invalid":
        out.unavailable_reason = "fitting hit a nonfinite value or backtracking failure"
        return out
    score_recs = continuous_inputs(loaded, tok, _scoring_indices(loaded, cell))
    for setting in config.SETTINGS:
        final = final_solve(fit.params, score_recs, C, continuous.continuous_cost_fn,
                            fit.log_t_state, setting, cell.arm, device, deadline)
        out.prediction_sets.append(PredictionSet(f"{cell.key()}__{setting.name}", setting.name,
                                                 final.states, final))
    return out


def run_permutation_arm(cell, loaded, tok, device, deadline) -> ArmOutput:
    """A1 refit on within-recording permuted codes, sharing A1's theta0 and
    grouping; states restored to the original timeline before scoring."""
    dataset = cell.dataset
    C, K = config.num_actions(dataset), config.K_FINE
    all_idx = tuple(range(len(loaded.metas)))
    counts = categorical.frame_weighted_counts(categorical_inputs(loaded, tok.codes_fine, all_idx), K)
    try:
        grouping = categorical.build_grouping(tok.centers_fine, counts, C, cell.seed, is_kc_identity=False)
    except categorical.UnavailableError as exc:
        return ArmOutput(unavailable_reason=str(exc))
    theta0 = categorical.build_theta0(grouping, counts, C, K, config.eta(dataset))
    rng = np.random.default_rng(cell.seed + config.PERMUTATION_SEED_OFFSET)
    window = config.WINDOW[dataset]
    perms: dict[str, np.ndarray] = {}
    recs = []
    for i in sorted(all_idx, key=lambda idx: loaded.metas[idx].name):
        meta = loaded.metas[i]
        perm = permute.build_permutation(meta.lengths, window, rng)
        perms[meta.name] = perm
        recs.append(categorical.RecordingInput(
            meta.name, permute.apply_permutation(np.asarray(tok.codes_fine[i], dtype=np.int32), perm),
            meta.lengths, config.FPS[dataset], window))
    fit = categorical.fit_categorical(
        recs, theta0, C, K, config.eta(dataset), coeffs_for(config.FIT_SETTING, "categorical_asot"),
        config.FIT_INNER_STEPS, config.FIT_OUTER_CAP, config.FIT_OUTER_PATIENCE,
        config.FIT_OUTER_RELTOL, device, deadline)
    names = sorted(perms)
    perm_obj = np.empty(len(names), dtype=object)
    perm_obj[:] = [perms[n] for n in names]
    out = ArmOutput(fit=fit, artifacts=dict(theta=fit.params, theta0=theta0, grouping=grouping,
                                             perm_names=np.asarray(names), perm=perm_obj,
                                             objective_trace=np.asarray(fit.objective_trace)))
    if fit.outer_status == "invalid":
        out.unavailable_reason = "permuted fit hit a nonfinite value or backtracking failure"
        return out
    for setting in config.SETTINGS:
        final = final_solve(fit.params, recs, C, categorical.categorical_cost_fn, fit.log_t_state,
                            setting, "categorical_asot", device, deadline)
        restored = {n: permute.restore_states(s, perms[n]) for n, s in final.states.items()}
        out.prediction_sets.append(PredictionSet(
            f"{cell.key()}__{setting.name}", setting.name, restored, final,
            reason="optimized in shuffled order; states restored to the original timeline before scoring"))
    return out


ARM_RUNNERS = {
    "no_temporal": run_categorical_arm,
    "categorical_asot": run_categorical_arm,
    "categorical_asot_kc": run_categorical_arm,
    "continuous_asot": run_continuous_arm,
    "permuted_asot": run_permutation_arm,
}


# --- scoring rows -------------------------------------------------------------------

def _base_row(cell: config.CellId, pid: str, setting: str, scope: str) -> dict:
    return dict(prediction_set_id=pid, cell_key=cell.key(), dataset=cell.dataset,
                normalize=cell.normalize, seed=cell.seed, protocol=cell.protocol, fold=cell.fold,
                arm=display_arm(cell.arm, pid), setting=setting, scope=scope)


def display_arm(arm: str, pid: str) -> str:
    return "no_temporal_filter" if "_filter__" in pid else arm


def _final_stats(final: "categorical.FinalResult | None", lengths_by_name: dict) -> dict:
    if final is None:
        return {}
    statuses = list(final.status.values())
    residuals = [(final.residuals[n], lengths_by_name[n]) for n in final.status
                 if final.residuals.get(n) is not None]
    stats = {
        "final_n_converged": statuses.count("converged"),
        "final_n_capped": statuses.count("capped"),
        "final_n_invalid": statuses.count("invalid"),
        "final_steps_mean": statistics.mean(final.steps.values()) if final.steps else None,
    }
    if residuals:
        values = np.asarray([r for r, _ in residuals])
        weights = np.asarray([w for _, w in residuals], dtype=np.float64)
        order = np.argsort(values)
        cumulative = np.cumsum(weights[order]) / weights.sum()
        stats["final_residual_max"] = float(values.max())
        stats["final_residual_median_w"] = float(values[order][np.searchsorted(cumulative, 0.5)])
    return stats


def _set_status(final: "categorical.FinalResult | None") -> str:
    if final is None:
        return "ok"
    statuses = set(final.status.values())
    if "invalid" in statuses:
        return "invalid"
    return "capped" if "capped" in statuses else "converged"


def score_frames_and_row(row: dict, loaded, indices, states: dict[str, np.ndarray], scope: str) -> dict:
    frame_preds, frame_gts, subjects, fold_of = [], [], [], []
    for i in indices:
        meta = loaded.metas[i]
        if meta.name not in states:
            continue
        frame_preds.append(evaluate.broadcast_to_frames(states[meta.name], meta.starts, meta.lengths,
                                                        meta.frame_count))
        frame_gts.append(loaded.core_dataset.gts[i])
        subjects.append(meta.subject)
        fold_of.append(int(loaded.fold_of[i]))
    row["n_recordings"] = len(frame_preds)
    if not frame_preds:
        row.update(status="invalid", reason=(row.get("reason") or "") + "; no recordings produced predictions")
        return row
    if scope == "oof":
        metrics, mapping = evaluate.score_subject_disjoint(np.asarray(fold_of), frame_gts, frame_preds)
    else:
        metrics, mapping = evaluate.score_pooled(frame_gts, frame_preds)
    row.update(metrics)
    row.update(evaluate.segment_diagnostics(frame_preds, frame_gts, subjects, config.FPS[loaded.dataset]))
    row["_mapping"] = mapping
    return row


def _save_states(run_dir: Path, pid: str, states: dict[str, np.ndarray], mapping: dict | None,
                 extra: dict) -> None:
    path = run_dir / "predictions" / f"{pid}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(states)
    obj = np.empty(len(names), dtype=object)
    obj[:] = [np.asarray(states[n], dtype=np.int16) for n in names]
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, names=np.asarray(names), states=obj, order="original_timeline",
                        mapping=json.dumps(mapping or {}), meta=json.dumps(extra, default=str))
    os.replace(temp, path)


def _save_artifacts(run_dir: Path, cell: config.CellId, out: ArmOutput) -> None:
    path = run_dir / "artifacts" / f"{cell.key()}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.npz")
    payload = dict(out.artifacts)
    if out.fit is not None:
        payload["outer_rel_changes"] = np.asarray(out.fit.last_rel_changes)
    np.savez_compressed(temp, **payload)
    os.replace(temp, path)


def run_one_cell(cell: config.CellId, device, deadline, run_dir: Path) -> list[dict]:
    loaded = get_loaded(cell.dataset)
    tok = get_tokenizer(cell)
    started = time.perf_counter()
    out = ARM_RUNNERS[cell.arm](cell, loaded, tok, device, deadline)
    elapsed = time.perf_counter() - started
    if out.artifacts:
        _save_artifacts(run_dir, cell, out)
    scope = "pooled" if cell.protocol == "pooled" else "fold"
    fit_info = {}
    if out.fit is not None:
        fit_info = dict(fit_outer_status=out.fit.outer_status, fit_outer_iterations=out.fit.outer_iterations,
                        fit_last_rel_change=out.fit.last_rel_changes[-1] if out.fit.last_rel_changes else None)
    rows = []
    if out.unavailable_reason is not None and not out.prediction_sets:
        for pid in config.prediction_set_ids(cell):
            row = _base_row(cell, pid, config.setting_of(pid), scope)
            row.update(fit_info, status="unavailable" if out.fit is None else "invalid",
                       reason=out.unavailable_reason, runtime_seconds=elapsed, device=str(device))
            rows.append(row)
        return rows
    indices = _scoring_indices(loaded, cell)
    lengths_by_name = {loaded.metas[i].name: int(loaded.metas[i].frame_count) for i in indices}
    for ps in out.prediction_sets:
        row = _base_row(cell, ps.prediction_set_id, ps.setting, scope)
        row.update(fit_info, status=ps.status_override or _set_status(ps.final), reason=ps.reason,
                   runtime_seconds=elapsed, device=str(device))
        row.update(_final_stats(ps.final, lengths_by_name))
        row = score_frames_and_row(row, loaded, indices, ps.states, scope)
        _save_states(run_dir, ps.prediction_set_id, ps.states, row.pop("_mapping", None),
                     {"cell": cell.key(), "setting": ps.setting, "scope": scope})
        rows.append(row)
    return rows


# --- SMQ baseline and out-of-fold aggregation ---------------------------------------------

def smq_baseline_rows(protocol: str, run_dir: Path, completed: set[str]) -> list[dict]:
    """Score the released checkpoint's own per-frame VQ codes with the same
    evaluator (dataset-level Hungarian for pooled; per-fold Hungarian then
    out-of-fold aggregation for subject_disjoint). No fitting."""
    rows = []
    for dataset, pid in config.smq_baseline_ids(protocol):
        if pid in completed:
            continue
        loaded = get_loaded(dataset)
        started = time.perf_counter()
        frame_codes = [np.asarray(c, dtype=np.int32) for c in loaded.core_dataset.codes]
        gts = loaded.core_dataset.gts
        scope = "pooled" if protocol == "pooled" else "oof"
        if protocol == "pooled":
            metrics, mapping = evaluate.score_pooled(gts, frame_codes)
        else:
            metrics, mapping = evaluate.score_subject_disjoint(loaded.fold_of, gts, frame_codes)
        row = dict(prediction_set_id=pid, cell_key=pid, dataset=dataset, normalize="n/a", seed="n/a",
                   protocol=protocol, fold=None, arm=config.SMQ_BASELINE_ARM, setting="n/a", scope=scope,
                   status="ok", reason="released-checkpoint VQ codes, same evaluator; no fit",
                   n_recordings=len(gts), device="cpu")
        row.update(metrics)
        row.update(evaluate.segment_diagnostics(frame_codes, gts, [m.subject for m in loaded.metas],
                                                config.FPS[dataset]))
        row["runtime_seconds"] = time.perf_counter() - started
        (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
        (run_dir / "predictions" / f"{pid}_mapping.json").write_text(json.dumps(mapping, indent=2))
        rows.append(row)
        print(f"[smq baseline] {dataset}: MoF {metrics['MoF']} F1@50 {metrics['F1@50']}", flush=True)
    return rows


def oof_rows(run_dir: Path, completed: set[str]) -> list[dict]:
    """Once all four fold cells of a (dataset, normalize, seed, arm) group
    exist, aggregate their held-out raw states: one Hungarian mapping per
    held-out fold, then the scorer's reduction over all out-of-fold
    recordings (plan §2)."""
    rows = []
    groups: dict[tuple, list[config.CellId]] = {}
    for cell in config.enumerate_subject_disjoint_cells():
        groups.setdefault((cell.dataset, cell.normalize, cell.seed, cell.arm), []).append(cell)
    for (dataset, normalize, seed, arm), cells in groups.items():
        for fold_pid_suffix in _suffixes(arm):
            fold_pids = [f"{c.key()}{fold_pid_suffix}" for c in cells]
            oof_pid = fold_pids[0].replace("_fold0_", "_oof_")
            if oof_pid in completed or not all(p in completed for p in fold_pids):
                continue
            paths = [run_dir / "predictions" / f"{p}.npz" for p in fold_pids]
            if not all(p.exists() for p in paths):
                continue
            states: dict[str, np.ndarray] = {}
            for path in paths:
                blob = np.load(path, allow_pickle=True)
                states.update({str(n): s.astype(np.int32) for n, s in zip(blob["names"], blob["states"])})
            loaded = get_loaded(dataset)
            template = config.CellId(dataset, normalize, seed, "subject_disjoint", 0, arm)
            setting = config.setting_of(oof_pid)
            row = _base_row(template, oof_pid, setting, "oof")
            row.update(cell_key=template.key().replace("_fold0_", "_oof_"), fold=None,
                       status="ok", reason="aggregated from 4 held-out folds (per-fold Hungarian)")
            row = score_frames_and_row(row, loaded, tuple(range(len(loaded.metas))), states, "oof")
            row.pop("_mapping", None)
            rows.append(row)
    return rows


def _suffixes(arm: str) -> list[str]:
    suffixes = [f"__{s.name}" for s in config.SETTINGS]
    if arm == "no_temporal":
        suffixes += [f"_filter__{s.name}" for s in config.SETTINGS]
    return suffixes


# --- run bookkeeping --------------------------------------------------------------------

def _load_completed(run_dir: Path) -> set[str]:
    path = run_dir / "cells.csv"
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != CELLS_CSV_FIELDS:
            raise SystemExit(f"{path} was written by a different runner version (columns differ); "
                             "start a new -RunId instead of resuming it")
        return {row["prediction_set_id"] for row in reader}


def _append_cells_csv(run_dir: Path, rows: list[dict]) -> None:
    path = run_dir / "cells.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CELLS_CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        fh.flush()
        os.fsync(fh.fileno())


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    temp = path.with_suffix(".tmp.csv")
    with temp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def write_manifest(run_dir: Path, args, n_cells: int, after_pooled_run: str | None) -> None:
    path = run_dir / "manifest.json"
    launch = {
        "started_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "hours_budget": args.hours,
        "resume": bool(args.resume), "device": args.device, "host": platform.node(),
        "python": sys.version, "platform": platform.platform(),
        "packages": {"numpy": np.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__},
        "cpu_thread_cap": min(config.CPU_THREAD_CAP, os.cpu_count() or 1),
        "protocol_version": config.PROTOCOL_VERSION,
    }
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest.get("protocol_version") != config.PROTOCOL_VERSION:
            raise SystemExit(f"{run_dir} was created under protocol {manifest.get('protocol_version')}; "
                             f"this runner is {config.PROTOCOL_VERSION}. Start a new -RunId.")
        manifest.setdefault("launches", []).append(launch)
    else:
        manifest = {
            "run_id": args.run_id, "protocol_version": config.PROTOCOL_VERSION,
            "tokenizer_version": config.TOKENIZER_VERSION, "protocol": args.protocol,
            "after_pooled_run": after_pooled_run, "n_planned_fits": n_cells,
            "seeds": list(config.SEEDS), "asot_pinned_commit": config.ASOT_PINNED_COMMIT,
            "settings": {s.name: vars(s) for s in config.SETTINGS},
            "launches": [launch],
        }
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(manifest, indent=2))
    os.replace(temp, path)


def write_planned_cells(run_dir: Path, cells: list[config.CellId], protocol: str) -> None:
    rows = [dict(cell_key=c.key(), dataset=c.dataset, normalize=c.normalize, seed=c.seed,
                 protocol=c.protocol, fold=c.fold, arm=c.arm,
                 prediction_set_ids=";".join(config.prediction_set_ids(c))) for c in cells]
    rows += [dict(cell_key=pid, dataset=d, normalize="n/a", seed="n/a", protocol=protocol, fold=None,
                  arm=config.SMQ_BASELINE_ARM, prediction_set_ids=pid)
             for d, pid in config.smq_baseline_ids(protocol)]
    _write_csv(run_dir / "planned_cells.csv",
               ["cell_key", "dataset", "normalize", "seed", "protocol", "fold", "arm", "prediction_set_ids"],
               rows)


def cells_for_protocol(protocol: str) -> list[config.CellId]:
    if protocol == "pooled":
        return config.enumerate_pooled_cells() + config.enumerate_permutation_cells()
    if protocol == "subject_disjoint":
        return config.enumerate_subject_disjoint_cells()
    raise ValueError(protocol)


def _projection(history: dict[tuple, list[float]], cell: config.CellId) -> float | None:
    """Median of observed runtimes for the same (dataset, arm); else the
    same dataset; else anything observed."""
    for key in ((cell.dataset, cell.arm), (cell.dataset, None), (None, None)):
        values = history.get(key)
        if values:
            return float(np.median(values))
    return None


# --- commands ---------------------------------------------------------------------------

def cmd_run(args) -> None:
    after_pooled_run = None
    if args.protocol == "subject_disjoint":
        if not args.after_pooled_run:
            raise SystemExit("-Protocol subject_disjoint requires -AfterPooledRun <pooled run_id>")
        pooled_dir = config.OUT_ROOT / args.after_pooled_run
        if not (pooled_dir / "REPORT.md").exists():
            raise SystemExit(f"named pooled run {args.after_pooled_run!r} has no REPORT.md at {pooled_dir}; "
                             "generate and read it first")
        after_pooled_run = args.after_pooled_run

    device = get_device(args.device)
    run_dir = config.OUT_ROOT / args.run_id
    if (run_dir / "cells.csv").exists() and not args.resume:
        raise SystemExit(f"{run_dir}/cells.csv already exists; pass -Resume or choose a new -RunId")
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = _load_completed(run_dir) if args.resume else set()

    cells = cells_for_protocol(args.protocol)
    write_manifest(run_dir, args, len(cells), after_pooled_run)
    write_planned_cells(run_dir, cells, args.protocol)

    deadline = time.perf_counter() + args.hours * 3600.0 - REPORT_RESERVE_SECONDS
    not_run: list[dict] = []
    history: dict[tuple, list[float]] = {}

    baseline = smq_baseline_rows(args.protocol, run_dir, completed)
    _append_cells_csv(run_dir, baseline)
    completed.update(r["prediction_set_id"] for r in baseline)

    todo = [c for c in cells if not all(p in completed for p in config.prediction_set_ids(c))]
    print(f"{len(cells) - len(todo)} fits already complete, {len(todo)} to run", flush=True)
    for index, cell in enumerate(todo, 1):
        remaining = deadline - time.perf_counter()
        projected = _projection(history, cell)
        if remaining <= 0 or (projected is not None and projected > remaining):
            reason = ("compute deadline reached" if remaining <= 0
                      else f"projected {projected:.0f}s exceeds {remaining:.0f}s remaining")
            not_run.append(dict(cell_key=cell.key(), dataset=cell.dataset, arm=cell.arm,
                                protocol=cell.protocol, reason=reason))
            continue
        print(f"[{index}/{len(todo)}] {cell.key()} ...", flush=True)
        started = time.perf_counter()
        try:
            rows = run_one_cell(cell, device, deadline, run_dir)
        except DeadlineExceeded:
            not_run.append(dict(cell_key=cell.key(), dataset=cell.dataset, arm=cell.arm,
                                protocol=cell.protocol, reason="interrupted at a cooperative deadline check"))
            continue
        _append_cells_csv(run_dir, rows)
        completed.update(r["prediction_set_id"] for r in rows)
        seconds = time.perf_counter() - started
        for key in ((cell.dataset, cell.arm), (cell.dataset, None), (None, None)):
            history.setdefault(key, []).append(seconds)
        summary = ", ".join(f"{r['prediction_set_id'].rsplit('__', 1)[-1]}:{r['status']}" for r in rows)
        print(f"    done in {seconds:.0f}s ({summary})", flush=True)
        if args.protocol == "subject_disjoint":
            extra = oof_rows(run_dir, completed)
            _append_cells_csv(run_dir, extra)
            completed.update(r["prediction_set_id"] for r in extra)

    _write_csv(run_dir / "not_run.csv", ["cell_key", "dataset", "arm", "protocol", "reason"], not_run)
    print(f"Run {args.run_id}: {len(completed)} prediction sets written, {len(not_run)} fits not run.")
    print(f"Next: python -m script.action_transport.report --run-id {args.run_id}")


def cmd_dry_run(args) -> None:
    cells = cells_for_protocol(args.protocol)
    n_sets = sum(len(config.prediction_set_ids(c)) for c in cells)
    print(f"Protocol: {args.protocol} (runner protocol version {config.PROTOCOL_VERSION})")
    print(f"Planned fits: {len(cells)}; scored prediction sets: {n_sets} "
          f"+ {len(config.smq_baseline_ids(args.protocol))} SMQ baseline jobs")
    by_arm: dict[str, int] = {}
    for c in cells:
        by_arm[c.arm] = by_arm.get(c.arm, 0) + 1
    for arm, count in sorted(by_arm.items()):
        print(f"  {arm}: {count} fits")
    needed = {data._cache_path(c.dataset, c.normalize, c.protocol, c.fold, c.seed) for c in cells}
    present = sum(p.exists() for p in needed)
    print(f"Tokenizer caches: {present}/{len(needed)} present (missing ones are built on first use)")
    timings = config.OUT_ROOT / "cache" / "timings.json"
    if timings.exists():
        per_fit = json.loads(timings.read_text()).get("median_seconds_per_fit")
        print(f"Timing-probe projection: {per_fit:.0f}s/fit -> ~{len(cells) * per_fit / 3600:.1f}h "
              f"+ {REPORT_RESERVE_SECONDS // 60} min reporting reserve (rough; see README for "
              "observed v1.3 timings)")
    else:
        print("No timing probe on disk; run -TimingProbe for a projection.")


def cmd_timing_probe(args) -> None:
    """Label-blind: fits A1 on a few recordings for 3 outer iterations plus
    one final solve; never scores."""
    from script.exp2round.q1q2 import core
    device = get_device(args.device)
    results = []
    for dataset in config.DATASETS:
        loaded = get_loaded(dataset)
        idx = tuple(range(min(4, len(loaded.metas))))
        prepared = core.prepare_windows(get_arrays(dataset, False), config.WINDOW[dataset], idx,
                                        max_sample=2000, seed=config.SAMPLING_SEED)
        kmeans = core.fit_vocabulary(prepared, config.K_FINE, config.SAMPLING_SEED)
        recs = categorical_inputs(loaded, core.assign_windows(kmeans, prepared.transformed), idx)
        C = config.num_actions(dataset)
        counts = categorical.frame_weighted_counts(recs, config.K_FINE)
        grouping = categorical.build_grouping(kmeans.cluster_centers_, counts, C, config.SAMPLING_SEED, False)
        theta0 = categorical.build_theta0(grouping, counts, C, config.K_FINE, config.eta(dataset))
        started = time.perf_counter()
        fit = categorical.fit_categorical(recs, theta0, C, config.K_FINE, config.eta(dataset),
                                          coeffs_for(config.FIT_SETTING, "categorical_asot"),
                                          config.FIT_INNER_STEPS, 3, config.FIT_OUTER_PATIENCE,
                                          config.FIT_OUTER_RELTOL, device)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        final = final_solve(fit.params, recs, C, categorical.categorical_cost_fn, fit.log_t_state,
                            config.SETTING_T, "categorical_asot", device, None)
        final_seconds = time.perf_counter() - started
        windows = sum(len(r.codes) for r in recs)
        total_windows = sum(len(m.lengths) for m in loaded.metas)
        per_outer = fit_seconds / 3 / windows * total_windows
        per_final = final_seconds / windows * total_windows
        results.append(dict(dataset=dataset, probe_windows=windows, dataset_windows=total_windows,
                            projected_outer_iteration_s=per_outer, projected_final_solve_s=per_final,
                            final_statuses=sorted(set(final.status.values()))))
        print(f"[timing-probe] {dataset}: ~{per_outer:.1f}s per outer iteration, ~{per_final:.1f}s per "
              f"final solve over the full dataset; probe final statuses {sorted(set(final.status.values()))}")
    # Rough per-fit estimate: 50 outer iterations (the cap) + two final solves.
    per_fit = float(np.mean([r["projected_outer_iteration_s"] * config.FIT_OUTER_CAP
                             + 2 * r["projected_final_solve_s"] for r in results]))
    path = config.OUT_ROOT / "cache" / "timings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"median_seconds_per_fit": per_fit, "probe": results,
                                "host": platform.node(), "device": args.device,
                                "measured_local": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2))
    print(f"Saved {path} (projection, not a real-cell measurement)")


def cmd_preflight(args) -> None:
    """Check that this machine has everything a run needs, without fitting."""
    failures, warnings = [], []

    def check(ok: bool, message: str, fatal: bool = True) -> None:
        tag = "PASS" if ok else ("FAIL" if fatal else "WARN")
        print(f"[{tag}] {message}")
        if not ok:
            (failures if fatal else warnings).append(message)

    from script.exp2round.q1q2 import core
    print(f"Python {sys.version.split()[0]}; numpy {np.__version__}; sklearn {sklearn.__version__}; "
          f"torch {torch.__version__}; host {platform.node()}")
    expected = {"numpy": "1.26.4", "sklearn": "1.2.2"}
    actual = {"numpy": np.__version__, "sklearn": sklearn.__version__}
    for pkg, version in expected.items():
        check(actual[pkg] == version, f"{pkg} {actual[pkg]} (reference machine: {version}; a different "
              "version can change KMeans/tokenizers if caches are rebuilt)", fatal=False)
    for dataset in config.DATASETS:
        ckpt = core.CHECKPOINTS[dataset]
        ok = ckpt.exists() and core.sha256_file(ckpt) == config.CHECKPOINT_SHA256[dataset]
        check(ok, f"{dataset} checkpoint {ckpt.relative_to(config.ROOT)} present with the declared SHA-256")
        preds = config.ROOT / "results" / "preds" / f"{dataset}_pretrained.npz"
        check(preds.exists(), f"{dataset} prediction dump {preds.relative_to(config.ROOT)}")
        caches = [core.OUT / "cache" / dataset, config.ROOT / "results" / "seg" / "cache" / dataset]
        cache = next((c for c in caches if (c / "_meta.npz").exists()), None)
        check(cache is not None, f"{dataset} latent cache (_meta.npz) in one of "
              f"{[str(c.relative_to(config.ROOT)) for c in caches]}")
        if cache is not None:
            meta = np.load(cache / "_meta.npz", allow_pickle=True)
            names = [str(n) for n in meta["names"]]
            missing = [n for n in names if not (cache / f"{n}.npy").exists()]
            check(not missing, f"{dataset} latent cache has all {len(names)} recordings"
                  + (f" (missing {len(missing)}, e.g. {missing[:3]})" if missing else ""))
            check(ckpt.exists() and str(meta["ckpt_sha256"]) == core.sha256_file(ckpt),
                  f"{dataset} latent cache was produced by this checkpoint")
    tokenizer_dir = config.OUT_ROOT / "cache" / "tokenizers"
    pooled_needed = {data._cache_path(c.dataset, c.normalize, c.protocol, c.fold, c.seed)
                     for c in cells_for_protocol("pooled")}
    present = sum(p.exists() for p in pooled_needed)
    check(present == len(pooled_needed), f"pooled tokenizer caches {present}/{len(pooled_needed)} in "
          f"{tokenizer_dir.relative_to(config.ROOT)} (missing ones are rebuilt; copying them keeps "
          "codes identical to the reference machine)", fatal=False)
    third_party = config.THIRD_PARTY_ASOT / "src" / "asot.py"
    check(third_party.exists(), "third_party/action_seg_ot/src/asot.py (parity test reference)", fatal=False)
    check(torch.cuda.is_available() or args.device == "cpu",
          f"device {args.device} available (CUDA available: {torch.cuda.is_available()})")
    free_gb = shutil.disk_usage(config.ROOT).free / 1e9
    check(free_gb > 5, f"{free_gb:.0f} GB free on the repo drive (a pooled run writes well under 1 GB)",
          fatal=False)
    print(f"\n{len(failures)} failure(s), {len(warnings)} warning(s)")
    if failures:
        raise SystemExit(1)


def cmd_validate_only(_args) -> None:
    from script.action_transport import tests as at_tests
    at_tests.run_all()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage A runner")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--timing-probe", action="store_true")
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
        return cmd_validate_only(args)
    if args.preflight:
        return cmd_preflight(args)
    if args.dry_run:
        return cmd_dry_run(args)
    if args.timing_probe:
        return cmd_timing_probe(args)
    if args.hours is None or args.run_id is None:
        raise SystemExit("A full run requires --hours and --run-id")
    if args.hours * 3600.0 <= REPORT_RESERVE_SECONDS:
        raise SystemExit(f"--hours must leave at least {REPORT_RESERVE_SECONDS // 60} minutes for reporting")
    cmd_run(args)


if __name__ == "__main__":
    main()
