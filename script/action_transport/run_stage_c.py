"""Stage C orchestrator (docs/plans/THREE_STAGE_MOTION_PLAN.md §5;
docs/plans/STAGE_C_AGENT_INSTRUCTIONS.md Part 2): contextual temporal
transport on the frozen `stage_b_003` embeddings, reusing Stage A's
predictions as comparison arms. Pooled protocol only. The user launches full
runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sklearn
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.action_transport import categorical, config, contextual, continuous, data, evaluate, smoothing, transport
from script.action_transport import run as stagea_run
from script.action_transport.budget import DeadlineExceeded

REPORT_RESERVE_SECONDS = 30 * 60
OUT_ROOT = config.OUT_ROOT.parent / "action_transport_stage_c"

FITTED_ARMS = ("context_only", "contextual_asot")
ARM_LABELS = {"context_only": "context-only", "context_only_filter": "context-only + filter",
             "contextual_asot": "contextual ASOT", "contextual_order_perm": "contextual-order diagnostic"}

CELLS_CSV_FIELDS = list(stagea_run.CELLS_CSV_FIELDS)  # identical columns; reused verbatim


# --- cell enumeration -----------------------------------------------------------------

def enumerate_pooled_cells() -> list[config.CellId]:
    cells = []
    for normalize in config.NORMALIZATIONS:
        for dataset in config.DATASETS:
            for seed in config.SEEDS:
                for arm in FITTED_ARMS:
                    cells.append(config.CellId(dataset, normalize, seed, "pooled", None, arm))
    return cells


def enumerate_order_diagnostic_cells() -> list[config.CellId]:
    return [config.CellId(dataset, normalize, config.SAMPLING_SEED, "pooled", None, "contextual_order_perm")
            for dataset in config.DATASETS for normalize in config.NORMALIZATIONS]


def all_cells() -> list[config.CellId]:
    return enumerate_pooled_cells() + enumerate_order_diagnostic_cells()


def prediction_set_ids(cell: config.CellId) -> list[str]:
    ids = [f"{cell.key()}__{setting.name}" for setting in config.SETTINGS]
    if cell.arm == "context_only":
        ids += [f"{cell.key()}_filter__{setting.name}" for setting in config.SETTINGS]
    return ids


def display_arm(arm: str, pid: str) -> str:
    return "context_only_filter" if "_filter__" in pid else arm


# --- dataset/tokenizer caching within a process (same convention as run.py) ------------

get_loaded = stagea_run.get_loaded
get_arrays = stagea_run.get_arrays
get_tokenizer = stagea_run.get_tokenizer


# --- fitting -------------------------------------------------------------------------

def final_solve(params, recs, num_states, cost_fn, warm, setting, arm, device, deadline):
    return categorical.infer_final(
        params, recs, num_states, cost_fn, warm, contextual.coeffs_for(setting, arm), device,
        config.FINAL_MAX_STEPS, config.FINAL_GRAD_TOL, config.FINAL_OBJ_RELTOL,
        config.FINAL_PATIENCE, config.BACKTRACK_MAX, deadline)


def run_context_arm(cell: config.CellId, loaded, tok, g: np.ndarray, embeddings, device,
                    deadline) -> "stagea_run.ArmOutput":
    dataset = cell.dataset
    C = config.num_actions(dataset)
    all_idx = tuple(range(len(loaded.metas)))
    fit_recs = contextual.context_inputs(loaded, tok, embeddings, all_idx)
    mu0, zero_mean = continuous.build_mu0(g, fit_recs, C)
    if zero_mean.any():
        return stagea_run.ArmOutput(unavailable_reason=f"{int(zero_mean.sum())} state(s) had a zero initial mean")
    fit = continuous.fit_continuous(
        fit_recs, mu0, C, contextual.coeffs_for(config.FIT_SETTING, cell.arm), config.FIT_INNER_STEPS,
        config.FIT_OUTER_CAP, config.FIT_OUTER_PATIENCE, config.FIT_OUTER_RELTOL, device, deadline)
    out = stagea_run.ArmOutput(fit=fit, artifacts=dict(mu=fit.params, mu0=mu0, grouping=g,
                                                       stale=np.asarray(fit.stale_prototypes, dtype=np.int32),
                                                       objective_trace=np.asarray(fit.objective_trace)))
    if fit.outer_status == "invalid":
        out.unavailable_reason = "fitting hit a nonfinite value or backtracking failure"
        return out
    for setting in config.SETTINGS:
        final = final_solve(fit.params, fit_recs, C, continuous.continuous_cost_fn, fit.log_t_state,
                            setting, cell.arm, device, deadline)
        out.prediction_sets.append(stagea_run.PredictionSet(f"{cell.key()}__{setting.name}", setting.name,
                                                             final.states, final))
        if cell.arm == "context_only":
            filtered = {
                rec.name: smoothing.mode_filter(
                    final.states[rec.name], rec.lengths,
                    transport.kernel_half_width(rec.fps, rec.window, len(rec.codes)))
                for rec in fit_recs if rec.name in final.states
            }
            out.prediction_sets.append(stagea_run.PredictionSet(
                f"{cell.key()}_filter__{setting.name}", setting.name, filtered, final,
                reason="derived from context_only raw states (no fit); status inherited from context_only"))
    return out


def run_order_diagnostic(cell: config.CellId, loaded, tok, stage_b_cell, g, device,
                         deadline) -> "stagea_run.ArmOutput":
    dataset, normalize, seed = cell.dataset, cell.normalize, cell.seed
    C = config.num_actions(dataset)
    restored = contextual.permuted_context_embeddings(stage_b_cell, loaded, tok, dataset, seed, device)
    perms = {name: perm for name, (_, _, perm) in restored.items()}
    fps_of, window_of = config.FPS[dataset], config.WINDOW[dataset]
    fit_recs = []
    for i, meta in enumerate(loaded.metas):
        z_restored, zero_restored, _ = restored[meta.name]
        z, zero_renorm = continuous.l2_normalize(z_restored.astype(np.float64))
        fit_recs.append(continuous.ContinuousRecordingInput(
            meta.name, np.asarray(tok.codes_fine[i], dtype=np.int32), meta.lengths, fps_of, window_of,
            z, zero_restored | zero_renorm))
    mu0, zero_mean = continuous.build_mu0(g, fit_recs, C)
    if zero_mean.any():
        return stagea_run.ArmOutput(unavailable_reason=f"{int(zero_mean.sum())} state(s) had a zero initial mean")
    fit = continuous.fit_continuous(
        fit_recs, mu0, C, contextual.coeffs_for(config.FIT_SETTING, "contextual_asot"), config.FIT_INNER_STEPS,
        config.FIT_OUTER_CAP, config.FIT_OUTER_PATIENCE, config.FIT_OUTER_RELTOL, device, deadline)
    perm_names = sorted(perms)
    perm_obj = np.empty(len(perm_names), dtype=object)
    perm_obj[:] = [perms[n] for n in perm_names]
    out = stagea_run.ArmOutput(fit=fit, artifacts=dict(mu=fit.params, mu0=mu0, grouping=g,
                                                       perm_names=np.asarray(perm_names), perm=perm_obj,
                                                       stale=np.asarray(fit.stale_prototypes, dtype=np.int32),
                                                       objective_trace=np.asarray(fit.objective_trace)))
    if fit.outer_status == "invalid":
        out.unavailable_reason = "order-diagnostic refit hit a nonfinite value or backtracking failure"
        return out
    for setting in config.SETTINGS:
        final = final_solve(fit.params, fit_recs, C, continuous.continuous_cost_fn, fit.log_t_state,
                            setting, "contextual_asot", device, deadline)
        out.prediction_sets.append(stagea_run.PredictionSet(
            f"{cell.key()}__{setting.name}", setting.name, final.states, final,
            reason="restored to original order before refitting/transport; predictions are already original-order"))
    return out


ARM_RUNNERS = {
    "context_only": run_context_arm,
    "contextual_asot": run_context_arm,
}


def run_one_cell(cell: config.CellId, stage_b_run: str, stage_a_run: str, device, deadline,
                 run_dir: Path) -> list[dict]:
    loaded = get_loaded(cell.dataset)
    tok = get_tokenizer(cell)
    started = time.perf_counter()
    try:
        stage_b_cell = contextual.load_stage_b_cell(stage_b_run, cell.dataset, cell.normalize, cell.seed)
        contextual.verify_tokenizer_continuity(loaded, tok, cell.dataset, cell.seed, stage_b_cell)
        C = config.num_actions(cell.dataset)
        g = contextual.verify_stage_a_grouping(loaded, tok, cell.dataset, cell.normalize, cell.seed,
                                               stage_a_run, C)
    except (contextual.IdentityError, contextual.UnavailableError) as exc:
        elapsed = time.perf_counter() - started
        rows = []
        for pid in prediction_set_ids(cell):
            row = stagea_run._base_row(cell, pid, config.setting_of(pid), "pooled")
            row.update(status="invalid_input", reason=str(exc), runtime_seconds=elapsed, device=str(device),
                      arm=display_arm(cell.arm, pid))
            rows.append(row)
        return rows
    if cell.arm == "contextual_order_perm":
        embeddings = None
        out = run_order_diagnostic(cell, loaded, tok, stage_b_cell, g, device, deadline)
    else:
        embeddings = contextual.load_embeddings(stage_b_cell)
        out = ARM_RUNNERS[cell.arm](cell, loaded, tok, g, embeddings, device, deadline)
    elapsed = time.perf_counter() - started
    if out.artifacts:
        path = run_dir / "artifacts" / f"{cell.key()}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp.npz")
        payload = dict(out.artifacts)
        if out.fit is not None:
            payload["outer_rel_changes"] = np.asarray(out.fit.last_rel_changes)
        np.savez_compressed(temp, **payload)
        os.replace(temp, path)
    fit_info = {}
    if out.fit is not None:
        fit_info = dict(fit_outer_status=out.fit.outer_status, fit_outer_iterations=out.fit.outer_iterations,
                        fit_last_rel_change=out.fit.last_rel_changes[-1] if out.fit.last_rel_changes else None)
    rows = []
    if out.unavailable_reason is not None and not out.prediction_sets:
        for pid in prediction_set_ids(cell):
            row = stagea_run._base_row(cell, pid, config.setting_of(pid), "pooled")
            row.update(fit_info, status="unavailable" if out.fit is None else "invalid",
                      reason=out.unavailable_reason, runtime_seconds=elapsed, device=str(device),
                      arm=display_arm(cell.arm, pid))
            rows.append(row)
        return rows
    indices = tuple(range(len(loaded.metas)))
    lengths_by_name = {loaded.metas[i].name: int(loaded.metas[i].frame_count) for i in indices}
    for ps in out.prediction_sets:
        row = stagea_run._base_row(cell, ps.prediction_set_id, ps.setting, "pooled")
        row["arm"] = display_arm(cell.arm, ps.prediction_set_id)
        row.update(fit_info, status=ps.status_override or stagea_run._set_status(ps.final), reason=ps.reason,
                  runtime_seconds=elapsed, device=str(device))
        row.update(stagea_run._final_stats(ps.final, lengths_by_name))
        row = stagea_run.score_frames_and_row(row, loaded, indices, ps.states, "pooled")
        stagea_run._save_states(run_dir, ps.prediction_set_id, ps.states, row.pop("_mapping", None),
                               {"cell": cell.key(), "setting": ps.setting, "scope": "pooled",
                                "stage_b_run": stage_b_run, "stage_a_run": stage_a_run})
        rows.append(row)
    return rows


# --- bookkeeping (mirrors run.py) ------------------------------------------------------

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


def write_manifest(run_dir: Path, args, n_cells: int) -> None:
    path = run_dir / "manifest.json"
    stage_b_manifest = contextual.verify_stage_b_manifest(contextual.STAGEB_OUT_ROOT / args.stage_b_run)
    launch = {"started_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "hours_budget": args.hours,
             "resume": bool(args.resume), "device": args.device, "host": platform.node(),
             "python": sys.version, "platform": platform.platform(),
             "packages": {"numpy": np.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__},
             "cpu_thread_cap": min(config.CPU_THREAD_CAP, os.cpu_count() or 1),
             "stagec_version": contextual.STAGEC_VERSION}
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest.get("stagec_version") != contextual.STAGEC_VERSION:
            raise SystemExit(f"{run_dir} was created under Stage C {manifest.get('stagec_version')}; "
                             f"this runner is {contextual.STAGEC_VERSION}. Start a new -RunId.")
        if manifest.get("stage_b_run") != args.stage_b_run or manifest.get("stage_a_run") != args.stage_a_run:
            raise SystemExit(f"{run_dir} was created against stage_b_run={manifest.get('stage_b_run')!r} "
                             f"stage_a_run={manifest.get('stage_a_run')!r}; this launch names "
                             f"{args.stage_b_run!r}/{args.stage_a_run!r}. Start a new -RunId.")
        manifest.setdefault("launches", []).append(launch)
    else:
        manifest = {"run_id": args.run_id, "stagec_version": contextual.STAGEC_VERSION,
                   "tokenizer_version": config.TOKENIZER_VERSION, "protocol": "pooled",
                   "n_planned_fits": n_cells, "seeds": list(config.SEEDS),
                   "stage_b_run": args.stage_b_run, "stage_a_run": args.stage_a_run,
                   "stage_b_config_hash": stage_b_manifest.get("config_hash"),
                   "settings": {s.name: vars(s) for s in config.SETTINGS}, "launches": [launch]}
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(manifest, indent=2))
    os.replace(temp, path)


def write_planned_cells(run_dir: Path, cells: list[config.CellId]) -> None:
    rows = [dict(cell_key=c.key(), dataset=c.dataset, normalize=c.normalize, seed=c.seed,
                arm=c.arm, prediction_set_ids=";".join(prediction_set_ids(c))) for c in cells]
    _write_csv(run_dir / "planned_cells.csv",
              ["cell_key", "dataset", "normalize", "seed", "arm", "prediction_set_ids"], rows)


def _projection(history: dict[tuple, list[float]], cell: config.CellId) -> float | None:
    for key in ((cell.dataset, cell.arm), (cell.dataset, None), (None, None)):
        values = history.get(key)
        if values:
            return float(np.median(values))
    return None


def verify_stage_a_reuse(stage_a_run: str) -> None:
    """§2.6: before using Stage A's cells.csv rows as comparison arms,
    re-score one saved Stage A prediction set per (dataset, normalize) from
    its own .npz and require the metrics to match cells.csv to 1e-9 --
    proves the evaluator and population are unchanged."""
    run_dir = config.OUT_ROOT / stage_a_run
    with (run_dir / "cells.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    checked = set()
    for row in rows:
        key = (row["dataset"], row["normalize"])
        if key in checked or row["arm"] != "continuous_asot" or row["setting"] != "T" or row["status"] == "invalid_input":
            continue
        pred_path = run_dir / "predictions" / f"{row['prediction_set_id']}.npz"
        if not pred_path.exists():
            continue
        blob = np.load(pred_path, allow_pickle=True)
        loaded = get_loaded(row["dataset"])
        name_to_idx = {m.name: i for i, m in enumerate(loaded.metas)}
        states = {str(n): s.astype(np.int32) for n, s in zip(blob["names"], blob["states"])}
        indices = tuple(name_to_idx[n] for n in states)
        recomputed = stagea_run.score_frames_and_row({}, loaded, indices, states, "pooled")
        for metric in ("MoF", "Edit", "F1@10", "F1@25", "F1@50"):
            if abs(float(recomputed[metric]) - float(row[metric])) > 1e-9:
                raise SystemExit(f"re-scoring {row['prediction_set_id']} gives {metric}={recomputed[metric]}, "
                                 f"but {stage_a_run}/cells.csv recorded {row[metric]} -- the evaluator or "
                                 "population has changed since Stage A ran")
        checked.add(key)
    if len(checked) < 4:
        print(f"[WARN] only verified evaluator reuse for {len(checked)}/4 (dataset,normalize) cells "
             f"(missing predictions in {stage_a_run})")


# --- commands ---------------------------------------------------------------------------

def get_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("-Device cuda requested but CUDA is not available (no silent fallback)")
        return torch.device("cuda")
    return torch.device("cpu")


def cmd_run(args) -> None:
    device = get_device(args.device)
    run_dir = OUT_ROOT / args.run_id
    if (run_dir / "cells.csv").exists() and not args.resume:
        raise SystemExit(f"{run_dir}/cells.csv already exists; pass -Resume or choose a new -RunId")
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = _load_completed(run_dir) if args.resume else set()

    verify_stage_a_reuse(args.stage_a_run)
    cells = all_cells()
    write_manifest(run_dir, args, len(cells))
    write_planned_cells(run_dir, cells)

    deadline = time.perf_counter() + args.hours * 3600.0 - REPORT_RESERVE_SECONDS
    not_run: list[dict] = []
    history: dict[tuple, list[float]] = {}

    todo = [c for c in cells if not all(p in completed for p in prediction_set_ids(c))]
    # order of execution (STAGE_C_AGENT_INSTRUCTIONS.md §2.4): contextual_asot T-priority fits first,
    # then context_only, then the diagnostics.
    order = {"contextual_asot": 0, "context_only": 1, "contextual_order_perm": 2}
    todo.sort(key=lambda c: order[c.arm])
    print(f"{len(cells) - len(todo)} fits already complete, {len(todo)} to run", flush=True)
    for index, cell in enumerate(todo, 1):
        remaining = deadline - time.perf_counter()
        projected = _projection(history, cell)
        if remaining <= 0 or (projected is not None and projected > remaining):
            reason = ("compute deadline reached" if remaining <= 0
                     else f"projected {projected:.0f}s exceeds {remaining:.0f}s remaining")
            not_run.append(dict(cell_key=cell.key(), dataset=cell.dataset, arm=cell.arm, reason=reason))
            continue
        print(f"[{index}/{len(todo)}] {cell.key()} ...", flush=True)
        started = time.perf_counter()
        try:
            rows = run_one_cell(cell, args.stage_b_run, args.stage_a_run, device, deadline, run_dir)
        except DeadlineExceeded:
            not_run.append(dict(cell_key=cell.key(), dataset=cell.dataset, arm=cell.arm,
                                reason="interrupted at a cooperative deadline check"))
            continue
        _append_cells_csv(run_dir, rows)
        completed.update(r["prediction_set_id"] for r in rows)
        seconds = time.perf_counter() - started
        for key in ((cell.dataset, cell.arm), (cell.dataset, None), (None, None)):
            history.setdefault(key, []).append(seconds)
        summary = ", ".join(f"{r['prediction_set_id'].rsplit('__', 1)[-1]}:{r['status']}" for r in rows)
        print(f"    done in {seconds:.0f}s ({summary})", flush=True)

    _write_csv(run_dir / "not_run.csv", ["cell_key", "dataset", "arm", "reason"], not_run)
    print(f"Stage C run {args.run_id}: {len(completed)} prediction sets written, {len(not_run)} fits not run.")
    print(f"Next: python -m script.action_transport.report_stage_c --run-id {args.run_id}")


def cmd_dry_run(args) -> None:
    cells = all_cells()
    n_sets = sum(len(prediction_set_ids(c)) for c in cells)
    print(f"Stage C protocol pooled (runner version {contextual.STAGEC_VERSION})")
    print(f"Planned fits: {len(cells)}; scored prediction sets: {n_sets}")
    by_arm: dict[str, int] = {}
    for c in cells:
        by_arm[c.arm] = by_arm.get(c.arm, 0) + 1
    for arm, count in sorted(by_arm.items()):
        print(f"  {arm}: {count} fits")
    print(f"stage_b_run={args.stage_b_run} stage_a_run={args.stage_a_run}")
    # Cost estimate from Stage A's observed A0/A-cont medians on this machine (README.md).
    medians = {"hugadb": 158.0 + 355.0, "lara": 242.0 + 574.0}  # context_only ~ A0; contextual_asot ~ A-cont
    per_dataset_seed_norm = sum(medians.values())
    est_fit_seconds = 3 * 2 * per_dataset_seed_norm  # 3 seeds x 2 normalizations x (context_only+contextual_asot)
    print(f"Rough projection from Stage A per-arm medians: ~{est_fit_seconds / 3600:.1f}h for the 24 pooled "
         f"fits, plus the 4 order-diagnostic fits (comparable cost to contextual_asot) "
         f"+ {REPORT_RESERVE_SECONDS // 60} min reporting reserve")


def cmd_preflight(args) -> None:
    failures, warnings = [], []

    def check(ok: bool, message: str, fatal: bool = True) -> None:
        tag = "PASS" if ok else ("FAIL" if fatal else "WARN")
        print(f"[{tag}] {message}")
        if not ok:
            (failures if fatal else warnings).append(message)

    try:
        contextual.verify_stage_b_manifest(contextual.STAGEB_OUT_ROOT / args.stage_b_run)
        check(True, f"stage_b_run {args.stage_b_run} has the frozen selected config_hash")
    except contextual.IdentityError as exc:
        check(False, str(exc))
    stage_a_dir = config.OUT_ROOT / args.stage_a_run
    check((stage_a_dir / "cells.csv").exists(), f"stage_a_run {args.stage_a_run} has cells.csv")
    check((stage_a_dir / "REPORT.md").exists(), f"stage_a_run {args.stage_a_run} has REPORT.md", fatal=False)
    for dataset in config.DATASETS:
        try:
            data.verify_provenance(dataset)
            check(True, f"{dataset} provenance gate")
        except Exception as exc:  # noqa: BLE001
            check(False, f"{dataset} provenance gate: {exc}")
    n_ok, n_total = 0, 0
    for dataset in config.DATASETS:
        for normalize in config.NORMALIZATIONS:
            for seed in config.SEEDS:
                n_total += 1
                try:
                    contextual.load_stage_b_cell(args.stage_b_run, dataset, normalize, seed)
                    n_ok += 1
                except contextual.IdentityError:
                    pass
    check(n_ok == n_total, f"stage_b_run {args.stage_b_run} embeddings/checkpoints present and "
         f"hash-consistent for {n_ok}/{n_total} cells", fatal=(n_ok == 0))
    device_ok = torch.cuda.is_available() or args.device != "cuda"
    check(device_ok, f"device {args.device} (CUDA available: {torch.cuda.is_available()})")
    print(f"\n{len(failures)} failure(s), {len(warnings)} warning(s)")
    if failures:
        raise SystemExit(1)


def cmd_validate_only(_args) -> None:
    from script.action_transport import tests_stagec
    tests_stagec.run_all()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage C runner")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage-b-run", dest="stage_b_run", default=None)
    parser.add_argument("--stage-a-run", dest="stage_a_run", default=None)
    parser.add_argument("--hours", type=float, default=None)
    parser.add_argument("--run-id", dest="run_id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stagea_run.set_thread_caps()
    if args.validate_only:
        return cmd_validate_only(args)
    if args.preflight:
        if not args.stage_b_run or not args.stage_a_run:
            raise SystemExit("--preflight requires --stage-b-run and --stage-a-run")
        return cmd_preflight(args)
    if args.dry_run:
        if not args.stage_b_run or not args.stage_a_run:
            raise SystemExit("--dry-run requires --stage-b-run and --stage-a-run")
        return cmd_dry_run(args)
    if args.hours is None or args.run_id is None or not args.stage_b_run or not args.stage_a_run:
        raise SystemExit("A full run requires --hours, --run-id, --stage-b-run and --stage-a-run")
    if args.hours * 3600.0 <= REPORT_RESERVE_SECONDS:
        raise SystemExit(f"--hours must leave at least {REPORT_RESERVE_SECONDS // 60} minutes for reporting")
    cmd_run(args)


if __name__ == "__main__":
    main()
