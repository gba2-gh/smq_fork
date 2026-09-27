"""Overnight frozen-representation study (docs/plans/FROZEN_STUDY_AGENT_INSTRUCTIONS.md
v1.0): discovery vs. oracle vs. pooling on frozen SMQ/stage_b_003
representations. Nothing here trains anything. This module only saves
artifacts: no report, no verdict, no metric printouts (status/timing only).

One command runs everything: identity checks -> P0 -> the P1..P5 queue.
The user launches the full run; this process only executes -ValidateOnly
and -Smoke on its own.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait as futures_wait
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.action_transport import categorical, config, contextual, continuous, transport
from script.action_transport import run
from script.action_transport.budget import DeadlineExceeded, check_deadline
from script.repro.d_error_decomposition import score as score_identity

FROZEN_STUDY_VERSION = "1.0"

DEVICE = torch.device("cpu")

RUNGS = [500, 1000, 2000, 4000]
H_LIST = (2, 8, 32, 64)
SEED_PRIMARY = config.SAMPLING_SEED  # 111
SEEDS_EXTRA = tuple(s for s in config.SEEDS if s != SEED_PRIMARY)  # (222, 1538574472)

STAGE_A_RUN = "stage_a_pooled_002"
STAGE_B_RUN = "stage_b_003"
STAGE_C_RUN = "stage_c_pooled_001"
STAGE_C_OUT_ROOT = config.OUT_ROOT.parent / "action_transport_stage_c"

FIT_OUTER_CAP_DEFAULT = 50
FIT_OUTER_CAP_EXTENDED = 150

PROBE_SUBSAMPLE = 50_000
PROBE_SEED = SEED_PRIMARY
PROBE_REPS = ("pca", "context") + tuple(f"pool_h{h}" for h in H_LIST)

OUT_ROOT = config.ROOT / "results" / "frozen_study"

# smoke-mode overrides (mutated only by apply_smoke_overrides(), never otherwise)
MAX_RECORDINGS_PER_DATASET: int | None = None

JOBS_CSV_FIELDS = ["job_id", "priority", "dataset", "normalize", "seed", "representation",
                    "h", "prototype_source", "outer_cap"]
STATUS_CSV_FIELDS = ["job_id", "status", "start", "end", "seconds", "error"]
ROWS_CSV_FIELDS = [
    "job_id", "priority", "dataset", "normalize", "seed", "representation", "h", "prototype_source",
    "rung_steps", "objective_total", "residual_max", "residual_median_w",
    "n_converged", "n_capped", "n_invalid", "frac_changed",
    "MoF", "Edit", "F1@10", "F1@25", "F1@50",
    "id_MoF", "id_Edit", "id_F1@10", "id_F1@25", "id_F1@50",
    "pred_runs", "gt_runs", "run_ratio", "pred_run_median_s", "pred_run_p10_s", "pred_run_p90_s",
    "gt_run_median_s", "occupied_states", "occupancy_entropy", "max_state_fraction", "mi_state_subject",
]
FITS_CSV_FIELDS = ["job_id", "outer_iterations", "outer_status", "last_rel_change", "stale_prototypes", "runtime"]
PROBE_CSV_FIELDS = ["job_id", "dataset", "normalize", "representation", "readout", "fold",
                     "accuracy_w", "macro_f1_w", "n_train", "n_test"]
BASELINE_CSV_FIELDS = ["prediction_set_id", "metric", "saved", "recomputed", "abs_diff"]
NOT_RUN_CSV_FIELDS = ["job_id", "reason"]


# --------------------------------------------------------------------------------------
# Window labels, pooling, oracle prototypes (labels enter ONLY here and in the probe)
# --------------------------------------------------------------------------------------

def window_labels(gt: np.ndarray, starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Majority frame label per window; np.bincount(...).argmax() ties go to
    the lowest label id. GT label ids are 1-indexed in this codebase; this
    function is label-id-agnostic (it never indexes mapping.txt)."""
    return np.array([np.bincount(gt[s:s + l]).argmax() for s, l in zip(starts, lengths)], dtype=np.int64)


def pool_windows(pca_windows: np.ndarray, lengths: np.ndarray, h: int) -> np.ndarray:
    """pooled[t] = sum_{|s-t|<=h} n_s*x[s] / sum n_s, edge-truncated (no wrap,
    no cross-recording averaging). h=0 returns pca_windows unchanged."""
    lengths_f = lengths.astype(np.float64)
    weighted = pca_windows.astype(np.float64) * lengths_f[:, None]
    zeros_row = np.zeros((1, pca_windows.shape[1]), dtype=np.float64)
    csum_w = np.concatenate([zeros_row, np.cumsum(weighted, axis=0)], axis=0)
    csum_n = np.concatenate([[0.0], np.cumsum(lengths_f)])
    num_windows = len(lengths_f)
    idx = np.arange(num_windows)
    lo = np.clip(idx - h, 0, num_windows - 1)
    hi = np.clip(idx + h, 0, num_windows - 1)
    num = csum_w[hi + 1] - csum_w[lo]
    den = csum_n[hi + 1] - csum_n[lo]
    return num / den[:, None]


def build_oracle_prototypes(recs: list, wl_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mu[n_present, dim], classes[n_present]); classes[row] is the
    true (1-indexed) GT label id that row corresponds to. Only present
    classes are used."""
    z = np.concatenate([r.z for r in recs])
    y = np.concatenate(wl_list)
    weights = np.concatenate([r.lengths for r in recs]).astype(np.float64)
    classes = np.unique(y)
    mu = np.zeros((len(classes), z.shape[1]), dtype=np.float64)
    for row, c in enumerate(classes):
        mask = y == c
        v = (weights[mask, None] * z[mask]).sum(0)
        mu[row] = v / (np.linalg.norm(v) + 1e-12)
    return mu, classes


def frame_weighted_metrics(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray,
                           classes: np.ndarray) -> tuple[float, float]:
    """Frame-weighted accuracy and frame-weighted macro-F1 (per-class
    weighted precision/recall/F1, unweighted mean across classes)."""
    acc = float((weights * (y_true == y_pred)).sum() / weights.sum())
    f1s = []
    for c in classes:
        tp = float(weights[(y_true == c) & (y_pred == c)].sum())
        fp = float(weights[(y_true != c) & (y_pred == c)].sum())
        fn = float(weights[(y_true == c) & (y_pred != c)].sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return acc, float(np.mean(f1s)) if f1s else 0.0


# --------------------------------------------------------------------------------------
# Representations
# --------------------------------------------------------------------------------------

def build_representation(rep: str, loaded, tok, embeddings, indices) -> list:
    if rep == "pca":
        return run.continuous_inputs(loaded, tok, indices)
    if rep == "context":
        return contextual.context_inputs(loaded, tok, embeddings, indices)
    if rep.startswith("pool_h"):
        h = int(rep[len("pool_h"):])
        fps, window = config.FPS[loaded.dataset], config.WINDOW[loaded.dataset]
        out = []
        for i in indices:
            meta = loaded.metas[i]
            pooled = pool_windows(np.asarray(tok.pca_windows[i], dtype=np.float64), meta.lengths, h)
            z, zero = continuous.l2_normalize(pooled)
            out.append(continuous.ContinuousRecordingInput(
                meta.name, np.asarray(tok.codes_fine[i], dtype=np.int32), meta.lengths, fps, window, z, zero))
        return out
    raise ValueError(f"unknown representation {rep!r}")


def scoped_indices(loaded) -> tuple[int, ...]:
    n = len(loaded.metas)
    if MAX_RECORDINGS_PER_DATASET is not None:
        n = min(n, MAX_RECORDINGS_PER_DATASET)
    return tuple(range(n))


# --------------------------------------------------------------------------------------
# Prototype sources
# --------------------------------------------------------------------------------------

def stage_a_artifact_path(dataset: str, normalize: bool, seed: int) -> Path:
    cell = config.CellId(dataset, normalize, seed, "pooled", None, "continuous_asot")
    return config.OUT_ROOT / STAGE_A_RUN / "artifacts" / f"{cell.key()}.npz"


def stage_c_artifact_path(dataset: str, normalize: bool, seed: int) -> Path:
    cell = config.CellId(dataset, normalize, seed, "pooled", None, "contextual_asot")
    return STAGE_C_OUT_ROOT / STAGE_C_RUN / "artifacts" / f"{cell.key()}.npz"


def disc_saved_mu(rep: str, dataset: str, normalize: bool, seed: int) -> np.ndarray:
    if rep == "pca":
        path = stage_a_artifact_path(dataset, normalize, seed)
    elif rep == "context":
        path = stage_c_artifact_path(dataset, normalize, seed)
    else:
        raise ValueError(f"no disc_saved prototypes are declared for representation {rep!r}")
    if not path.exists():
        raise contextual.IdentityError(f"{path} does not exist")
    return np.asarray(np.load(path)["mu"], dtype=np.float64)


def run_disc_fit(recs, num_states: int, grouping: np.ndarray, outer_cap: int,
                 deadline: float | None) -> "categorical.FitResult":
    mu0, zero_mean = continuous.build_mu0(grouping, recs, num_states)
    if zero_mean.any():
        raise categorical.UnavailableError(f"{int(zero_mean.sum())} state(s) had a zero initial mean")
    coeffs = transport.Coeffs(config.FIT_SETTING.a, config.FIT_SETTING.beta, config.FIT_SETTING.lam,
                              config.FIT_SETTING.eps)
    return continuous.fit_continuous(recs, mu0, num_states, coeffs, config.FIT_INNER_STEPS, outer_cap,
                                     config.FIT_OUTER_PATIENCE, config.FIT_OUTER_RELTOL, DEVICE, deadline)


# --------------------------------------------------------------------------------------
# The ladder: cumulative-rung cold-start solving under setting T only
# --------------------------------------------------------------------------------------

def ladder_solve(mu: np.ndarray, recs: list, metas: dict, gts: dict, fps: int, rungs: list[int],
                 deadline: float | None, oracle_classes: np.ndarray | None = None):
    """Returns (rung_rows, log_t_final, status_final, states_by_rung,
    hit_deadline). Each recording is frozen (never re-solved) once it
    reports 'converged' or 'invalid'. The patience streak resets at every
    rung boundary because each solve_final call starts fresh."""
    num_states = mu.shape[0]
    coeffs = transport.Coeffs(config.SETTING_T.a, config.SETTING_T.beta, config.SETTING_T.lam,
                              config.SETTING_T.eps)
    log_t: dict[str, torch.Tensor] = {}
    steps_done: dict[str, int] = {}
    status: dict[str, str] = {}
    residual: dict[str, float | None] = {}
    prev_states: dict[str, np.ndarray] | None = None
    rung_rows: list[dict] = []
    states_by_rung: dict[int, dict[str, np.ndarray]] = {}
    hit_deadline = False
    for rung in rungs:
        try:
            for rec in recs:
                check_deadline(deadline)
                name = rec.name
                if status.get(name) in ("converged", "invalid"):
                    continue
                remaining = rung - steps_done.get(name, 0)
                if remaining <= 0:
                    continue
                p, q, weights = categorical.recording_tensors(rec, num_states, DEVICE)
                cost = torch.as_tensor(continuous.continuous_cost_fn(mu, rec), dtype=torch.float64, device=DEVICE)
                init = log_t.get(name)
                if init is None:
                    init = transport.cold_start_log(p, q)
                result = transport.solve_final(init, cost, weights, p, q, coeffs, remaining,
                                               config.FINAL_GRAD_TOL, config.FINAL_OBJ_RELTOL,
                                               config.FINAL_PATIENCE, config.BACKTRACK_MAX)
                log_t[name] = result.log_t
                steps_done[name] = steps_done.get(name, 0) + result.accepted_steps
                status[name] = result.status
                if result.residual is not None:
                    residual[name] = result.residual
        except DeadlineExceeded:
            hit_deadline = True
            break
        states = {name: t.argmax(dim=1).cpu().numpy().astype(np.int32) for name, t in log_t.items()}
        states_by_rung[rung] = states
        row = _rung_row(recs, states, prev_states, mu, coeffs, status, residual, log_t, rung,
                        metas, gts, fps, oracle_classes)
        rung_rows.append(row)
        prev_states = states
    return rung_rows, log_t, status, states_by_rung, hit_deadline


def _rung_row(recs, states, prev_states, mu, coeffs, status, residual, log_t, rung,
             metas, gts, fps, oracle_classes) -> dict:
    frame_preds, frame_gts, subjects, id_frame_preds = [], [], [], []
    for rec in recs:
        s = states.get(rec.name)
        if s is None:
            continue
        meta = metas[rec.name]
        frame_preds.append(run.evaluate.broadcast_to_frames(s, meta.starts, meta.lengths, meta.frame_count))
        frame_gts.append(gts[rec.name])
        subjects.append(meta.subject)
        if oracle_classes is not None:
            id_frame_preds.append(run.evaluate.broadcast_to_frames(oracle_classes[s], meta.starts, meta.lengths,
                                                                    meta.frame_count))
    row: dict = {"rung_steps": rung}
    if frame_preds:
        metrics, _ = run.evaluate.score_pooled(frame_gts, frame_preds)
        row.update(metrics)
        row.update(run.evaluate.segment_diagnostics(frame_preds, frame_gts, subjects, fps))
        if oracle_classes is not None:
            id_metrics = score_identity(frame_gts, id_frame_preds)
            row.update({f"id_{k}": v for k, v in id_metrics.items()})
    row["n_converged"] = list(status.values()).count("converged")
    row["n_capped"] = list(status.values()).count("capped")
    row["n_invalid"] = list(status.values()).count("invalid")
    row["objective_total"] = _objective_total(recs, log_t, mu, coeffs)
    row["residual_max"], row["residual_median_w"] = _residual_stats(recs, residual)
    row["frac_changed"] = _frac_changed(recs, states, prev_states)
    return row


def _objective_total(recs, log_t, mu, coeffs) -> float:
    total = 0.0
    for rec in recs:
        if rec.name not in log_t:
            continue
        p, q, weights = categorical.recording_tensors(rec, mu.shape[0], DEVICE)
        cost = torch.as_tensor(continuous.continuous_cost_fn(mu, rec), dtype=torch.float64, device=DEVICE)
        obj = transport.objective_log(log_t[rec.name], cost, weights, coeffs.a, coeffs.beta, coeffs.lam,
                                      coeffs.eps, q)
        total += float(obj) * float(rec.lengths.sum())
    return total


def _residual_stats(recs, residual) -> tuple[float | None, float | None]:
    vals, weights = [], []
    for rec in recs:
        v = residual.get(rec.name)
        if v is not None:
            vals.append(v)
            weights.append(float(rec.lengths.sum()))
    if not vals:
        return None, None
    vals_a, weights_a = np.asarray(vals), np.asarray(weights)
    order = np.argsort(vals_a)
    cumulative = np.cumsum(weights_a[order]) / weights_a.sum()
    median = float(vals_a[order][np.searchsorted(cumulative, 0.5)])
    return float(vals_a.max()), median


def _frac_changed(recs, states, prev_states) -> float | None:
    if prev_states is None:
        return None
    changed = total = 0.0
    for rec in recs:
        now, prev = states.get(rec.name), prev_states.get(rec.name)
        if now is None or prev is None:
            continue
        w = rec.lengths.astype(np.float64)
        total += w.sum()
        changed += w[now != prev].sum()
    return float(changed / total) if total else None


# --------------------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Job:
    job_id: str
    priority: str
    kind: str  # "check_baselines" | "pilot" | "ladder" | "fit_ladder" | "probe"
    dataset: str | None = None
    normalize: bool | None = None
    seed: int | None = None
    representation: str | None = None
    h: int | None = None
    prototype_source: str | None = None
    outer_cap: int | None = None


def _norm_tag(normalize: bool) -> str:
    return "norm" if normalize else "raw"


def enumerate_jobs() -> list[Job]:
    jobs = [Job("check_baselines", "P0", "check_baselines"), Job("pilot", "P0", "pilot")]
    for norm in config.NORMALIZATIONS:
        for ds in config.DATASETS:
            for rep in ("pca", "context"):
                for proto in ("disc_saved", "oracle"):
                    jid = f"L_{ds}_{_norm_tag(norm)}_s{SEED_PRIMARY}_{rep}_{proto}"
                    jobs.append(Job(jid, "P1", "ladder", ds, norm, SEED_PRIMARY, rep, None, proto))
    for norm in config.NORMALIZATIONS:
        for ds in config.DATASETS:
            for h in H_LIST:
                jid = f"F_{ds}_{_norm_tag(norm)}_s{SEED_PRIMARY}_pool_h{h}"
                jobs.append(Job(jid, "P2", "fit_ladder", ds, norm, SEED_PRIMARY, f"pool_h{h}", h,
                               "disc_fit", FIT_OUTER_CAP_DEFAULT))
    for norm in config.NORMALIZATIONS:
        for ds in config.DATASETS:
            for h in H_LIST:
                jid = f"L_{ds}_{_norm_tag(norm)}_s{SEED_PRIMARY}_pool_h{h}_oracle"
                jobs.append(Job(jid, "P2", "ladder", ds, norm, SEED_PRIMARY, f"pool_h{h}", h, "oracle"))
    for ds in config.DATASETS:
        for rep in ("pca", "context"):
            jid = f"X_{ds}_raw_s{SEED_PRIMARY}_{rep}"
            jobs.append(Job(jid, "P3", "fit_ladder", ds, False, SEED_PRIMARY, rep, None, "disc_fit",
                           FIT_OUTER_CAP_EXTENDED))
    for norm in config.NORMALIZATIONS:
        for ds in config.DATASETS:
            jid = f"probe_{ds}_{_norm_tag(norm)}"
            jobs.append(Job(jid, "P4", "probe", ds, norm, PROBE_SEED))
    for seed in SEEDS_EXTRA:
        for norm in config.NORMALIZATIONS:
            for ds in config.DATASETS:
                for rep in ("pca", "context"):
                    jid = f"L_{ds}_{_norm_tag(norm)}_s{seed}_{rep}_disc_saved"
                    jobs.append(Job(jid, "P5", "ladder", ds, norm, seed, rep, None, "disc_saved"))
                jid = f"L_{ds}_{_norm_tag(norm)}_s{seed}_context_oracle"
                jobs.append(Job(jid, "P5", "ladder", ds, norm, seed, "context", None, "oracle"))
    for seed in SEEDS_EXTRA:
        for norm in config.NORMALIZATIONS:
            for ds in config.DATASETS:
                for h in (32, 64):
                    jid = f"F_{ds}_{_norm_tag(norm)}_s{seed}_pool_h{h}"
                    jobs.append(Job(jid, "P5", "fit_ladder", ds, norm, seed, f"pool_h{h}", h,
                                   "disc_fit", FIT_OUTER_CAP_DEFAULT))
    return jobs


def enumerate_smoke_jobs() -> list[Job]:
    """Every job KIND once, on the smoke-restricted population."""
    ds, norm, h = config.DATASETS[0], False, H_LIST[0]
    return [
        Job("check_baselines", "P0", "check_baselines"),
        Job("pilot", "P0", "pilot"),
        Job(f"L_{ds}_raw_s{SEED_PRIMARY}_pca_disc_saved", "P1", "ladder", ds, norm, SEED_PRIMARY, "pca", None,
            "disc_saved"),
        Job(f"L_{ds}_raw_s{SEED_PRIMARY}_pca_oracle", "P1", "ladder", ds, norm, SEED_PRIMARY, "pca", None,
            "oracle"),
        Job(f"F_{ds}_raw_s{SEED_PRIMARY}_pool_h{h}", "P2", "fit_ladder", ds, norm, SEED_PRIMARY, f"pool_h{h}",
            h, "disc_fit", 2),
        Job(f"X_{ds}_raw_s{SEED_PRIMARY}_pca", "P3", "fit_ladder", ds, False, SEED_PRIMARY, "pca", None,
            "disc_fit", 2),
        Job(f"probe_{ds}_raw", "P4", "probe", ds, norm, PROBE_SEED),
    ]


def job_to_row(job: Job) -> dict:
    return dict(job_id=job.job_id, priority=job.priority, dataset=job.dataset, normalize=job.normalize,
               seed=job.seed, representation=job.representation, h=job.h,
               prototype_source=job.prototype_source, outer_cap=job.outer_cap)


# --------------------------------------------------------------------------------------
# Job results
# --------------------------------------------------------------------------------------

@dataclass
class JobResult:
    job_id: str
    status: str
    seconds: float
    error: str | None = None
    rows: list[dict] = field(default_factory=list)
    fit_rows: list[dict] = field(default_factory=list)
    probe_rows: list[dict] = field(default_factory=list)
    baseline_rows: list[dict] = field(default_factory=list)
    pilot_data: dict | None = None
    artifact: dict | None = None


# --------------------------------------------------------------------------------------
# P0: check_baselines, pilot (run synchronously in the parent process)
# --------------------------------------------------------------------------------------

def _prediction_ids_to_check() -> list[tuple[Path, str]]:
    ids = []
    for ds in config.DATASETS:
        for norm in config.NORMALIZATIONS:
            for seed in config.SEEDS:
                cell = config.CellId(ds, norm, seed, "pooled", None, "continuous_asot")
                ids.append((config.OUT_ROOT / STAGE_A_RUN, f"{cell.key()}__T"))
                for arm in ("contextual_asot", "context_only"):
                    cell_c = config.CellId(ds, norm, seed, "pooled", None, arm)
                    ids.append((STAGE_C_OUT_ROOT / STAGE_C_RUN, f"{cell_c.key()}__T"))
    return ids


def run_check_baselines(job: Job) -> JobResult:
    started = time.perf_counter()
    rows = []
    for run_dir, pid in _prediction_ids_to_check():
        cells_path = run_dir / "cells.csv"
        pred_path = run_dir / "predictions" / f"{pid}.npz"
        with cells_path.open(newline="", encoding="utf-8") as fh:
            saved_row = next((r for r in csv.DictReader(fh) if r["prediction_set_id"] == pid), None)
        if saved_row is None or saved_row.get("status") == "invalid_input" or not pred_path.exists():
            rows.append(dict(prediction_set_id=pid, metric="n/a", saved=None, recomputed=None, abs_diff=None))
            continue
        blob = np.load(pred_path, allow_pickle=True)
        loaded = run.get_loaded(saved_row["dataset"])
        name_to_idx = {m.name: i for i, m in enumerate(loaded.metas)}
        states = {str(n): s.astype(np.int32) for n, s in zip(blob["names"], blob["states"])}
        indices = tuple(name_to_idx[n] for n in states)
        recomputed_row = run.score_frames_and_row({}, loaded, indices, states, "pooled")
        for metric in ("MoF", "Edit", "F1@10", "F1@25", "F1@50"):
            diff = abs(float(recomputed_row[metric]) - float(saved_row[metric]))
            rows.append(dict(prediction_set_id=pid, metric=metric, saved=saved_row[metric],
                            recomputed=recomputed_row[metric], abs_diff=diff))
            if diff > 1e-9:
                raise SystemExit(f"check_baselines: {pid} {metric} differs by {diff} (saved "
                                 f"{saved_row[metric]}, recomputed {recomputed_row[metric]}) -- aborting the "
                                 "whole run: the evaluator or population has changed")
    return JobResult(job.job_id, "complete", time.perf_counter() - started, baseline_rows=rows)


def run_pilot(job: Job) -> JobResult:
    started = time.perf_counter()
    per_dataset = {}
    for ds in config.DATASETS:
        loaded = run.get_loaded(ds)
        cell = config.CellId(ds, False, SEED_PRIMARY, "pooled", None, "continuous_asot")
        tok = run.get_tokenizer(cell)
        idx = tuple(range(min(20, len(loaded.metas))))
        recs = run.continuous_inputs(loaded, tok, idx)
        mu = disc_saved_mu("pca", ds, False, SEED_PRIMARY)
        coeffs = transport.Coeffs(config.SETTING_T.a, config.SETTING_T.beta, config.SETTING_T.lam,
                                  config.SETTING_T.eps)
        t0 = time.perf_counter()
        out = categorical.infer_final(mu, recs, mu.shape[0], continuous.continuous_cost_fn, None, coeffs,
                                      DEVICE, 500, config.FINAL_GRAD_TOL, config.FINAL_OBJ_RELTOL,
                                      config.FINAL_PATIENCE, config.BACKTRACK_MAX, None)
        elapsed = time.perf_counter() - t0
        total_steps = max(1, sum(out.steps.values()))
        per_dataset[ds] = dict(
            pilot_recordings=len(recs), pilot_windows=int(sum(len(r.codes) for r in recs)),
            pilot_seconds=elapsed, pilot_total_steps=total_steps,
            seconds_per_recording_step=elapsed / total_steps,
            n_recordings_full=len(loaded.metas), statuses=sorted(set(out.status.values())))
        print(f"[pilot] {ds}: {elapsed:.1f}s / {total_steps} recording-steps "
             f"({per_dataset[ds]['seconds_per_recording_step'] * 1000:.2f} ms/recording-step)", flush=True)
    return JobResult(job.job_id, "complete", time.perf_counter() - started,
                     pilot_data={"dataset_stats": per_dataset})


# --------------------------------------------------------------------------------------
# P1-P5: ladder / fit_ladder / probe (executed in worker processes)
# --------------------------------------------------------------------------------------

def _load_cell_context(dataset: str, normalize: bool, seed: int, need_embeddings: bool):
    loaded = run.get_loaded(dataset)
    cell = config.CellId(dataset, normalize, seed, "pooled", None, "continuous_asot")
    tok = run.get_tokenizer(cell)
    embeddings = None
    if need_embeddings:
        stage_b_cell = contextual.load_stage_b_cell(STAGE_B_RUN, dataset, normalize, seed)
        contextual.verify_tokenizer_continuity(loaded, tok, dataset, seed, stage_b_cell)
        embeddings = contextual.load_embeddings(stage_b_cell)
    return loaded, tok, embeddings


def execute_job(job: Job, deadline: float | None) -> JobResult:
    started = time.perf_counter()
    try:
        if job.kind == "probe":
            return run_probe(job, started, deadline)
        need_embeddings = (job.representation == "context")
        loaded, tok, embeddings = _load_cell_context(job.dataset, job.normalize, job.seed, need_embeddings)
        indices = scoped_indices(loaded)
        metas = {loaded.metas[i].name: loaded.metas[i] for i in indices}
        gts = {loaded.metas[i].name: loaded.core_dataset.gts[i] for i in indices}
        fps = config.FPS[job.dataset]
        recs = build_representation(job.representation, loaded, tok, embeddings, indices)

        fit_row, mu0, fit_trace, classes, oracle_classes = None, None, None, None, None
        if job.kind == "fit_ladder":
            num_states = config.num_actions(job.dataset)
            grouping = contextual.verify_stage_a_grouping(loaded, tok, job.dataset, job.normalize, job.seed,
                                                          STAGE_A_RUN, num_states)
            fit = run_disc_fit(recs, num_states, grouping, job.outer_cap, deadline)
            if fit.outer_status == "invalid":
                return JobResult(job.job_id, "error", time.perf_counter() - started,
                                error="fit hit a nonfinite value or backtracking failure")
            mu = fit.params
            mu0, zero_mean = continuous.build_mu0(grouping, recs, num_states)
            fit_trace = fit.objective_trace
            fit_row = dict(job_id=job.job_id, outer_iterations=fit.outer_iterations,
                          outer_status=fit.outer_status,
                          last_rel_change=fit.last_rel_changes[-1] if fit.last_rel_changes else None,
                          stale_prototypes=len(fit.stale_prototypes), runtime=time.perf_counter() - started)
        elif job.prototype_source == "disc_saved":
            mu = disc_saved_mu(job.representation, job.dataset, job.normalize, job.seed)
        elif job.prototype_source == "oracle":
            wl = [window_labels(gts[r.name], metas[r.name].starts, metas[r.name].lengths) for r in recs]
            mu, classes = build_oracle_prototypes(recs, wl)
            oracle_classes = classes
        else:
            raise ValueError(f"unhandled prototype_source {job.prototype_source!r}")

        rung_rows, log_t, status, states_by_rung, hit_deadline = ladder_solve(
            mu, recs, metas, gts, fps, RUNGS, deadline, oracle_classes)
        for r in rung_rows:
            r.update(job_id=job.job_id, priority=job.priority, dataset=job.dataset, normalize=job.normalize,
                    seed=job.seed, representation=job.representation, h=job.h,
                    prototype_source=job.prototype_source)

        if not rung_rows:
            job_status = "interrupted"
        elif hit_deadline:
            job_status = "partial"
        else:
            job_status = "complete"

        names_sorted = sorted(log_t)
        artifact = dict(mu=mu, names=np.asarray(names_sorted))
        if mu0 is not None:
            artifact["mu0"] = mu0
        if fit_trace is not None:
            artifact["fit_objective_trace"] = np.asarray(fit_trace)
        if classes is not None:
            artifact["classes"] = classes
        for rung, states in states_by_rung.items():
            obj = np.empty(len(names_sorted), dtype=object)
            obj[:] = [np.asarray(states[n], dtype=np.int16) for n in names_sorted]
            artifact[f"states_rung{rung}"] = obj

        return JobResult(job.job_id, job_status, time.perf_counter() - started, rows=rung_rows,
                        fit_rows=[fit_row] if fit_row else [], artifact=artifact)
    except (contextual.IdentityError, contextual.UnavailableError, categorical.UnavailableError) as exc:
        return JobResult(job.job_id, "unavailable", time.perf_counter() - started, error=str(exc))
    except DeadlineExceeded:
        return JobResult(job.job_id, "interrupted", time.perf_counter() - started,
                        error="deadline exceeded before any rung completed")
    except Exception as exc:  # noqa: BLE001 -- a worker-process job must never crash the pool
        return JobResult(job.job_id, "error", time.perf_counter() - started, error=repr(exc))


def run_probe(job: Job, started: float, deadline: float | None) -> JobResult:
    loaded, tok, embeddings = _load_cell_context(job.dataset, job.normalize, PROBE_SEED, True)
    indices = scoped_indices(loaded)
    gts = {loaded.metas[i].name: loaded.core_dataset.gts[i] for i in indices}
    wl = {loaded.metas[i].name: window_labels(gts[loaded.metas[i].name], loaded.metas[i].starts,
                                              loaded.metas[i].lengths) for i in indices}
    rows = []
    rng = np.random.default_rng(PROBE_SEED)
    for rep in PROBE_REPS:
        check_deadline(deadline)
        recs = build_representation(rep, loaded, tok, embeddings, indices)
        z_by_name = {r.name: r.z for r in recs}
        w_by_name = {r.name: r.lengths.astype(np.float64) for r in recs}
        for fold in range(4):
            check_deadline(deadline)
            train_names = [loaded.metas[i].name for i in indices if loaded.fold_of[i] != fold]
            test_names = [loaded.metas[i].name for i in indices if loaded.fold_of[i] == fold]
            if not train_names or not test_names:
                continue
            z_tr = np.concatenate([z_by_name[n] for n in train_names])
            y_tr = np.concatenate([wl[n] for n in train_names])
            w_tr = np.concatenate([w_by_name[n] for n in train_names])
            z_te = np.concatenate([z_by_name[n] for n in test_names])
            y_te = np.concatenate([wl[n] for n in test_names])
            w_te = np.concatenate([w_by_name[n] for n in test_names])
            if len(y_tr) > PROBE_SUBSAMPLE:
                sel = rng.choice(len(y_tr), size=PROBE_SUBSAMPLE, replace=False)
                z_tr, y_tr, w_tr = z_tr[sel], y_tr[sel], w_tr[sel]
            classes = np.unique(y_tr)

            scaler = StandardScaler()
            z_tr_s = scaler.fit_transform(z_tr)
            z_te_s = scaler.transform(z_te)
            clf = LogisticRegression(C=1.0, max_iter=2000)
            clf.fit(z_tr_s, y_tr, sample_weight=w_tr)
            acc_lr, f1_lr = frame_weighted_metrics(y_te, clf.predict(z_te_s), w_te, classes)
            rows.append(dict(job_id=job.job_id, dataset=job.dataset, normalize=job.normalize,
                            representation=rep, readout="logreg", fold=fold, accuracy_w=acc_lr,
                            macro_f1_w=f1_lr, n_train=len(y_tr), n_test=len(y_te)))

            centroids = np.zeros((len(classes), z_tr.shape[1]))
            for row, c in enumerate(classes):
                mask = y_tr == c
                v = (w_tr[mask, None] * z_tr[mask]).sum(0)
                centroids[row] = v / (np.linalg.norm(v) + 1e-12)
            pred_nc = classes[(z_te @ centroids.T).argmax(1)]
            acc_nc, f1_nc = frame_weighted_metrics(y_te, pred_nc, w_te, classes)
            rows.append(dict(job_id=job.job_id, dataset=job.dataset, normalize=job.normalize,
                            representation=rep, readout="nearest_centroid", fold=fold, accuracy_w=acc_nc,
                            macro_f1_w=f1_nc, n_train=len(y_tr), n_test=len(y_te)))
    return JobResult(job.job_id, "complete", time.perf_counter() - started, probe_rows=rows)


# --------------------------------------------------------------------------------------
# Startup identity checks (fail-fast; abort on any failure)
# --------------------------------------------------------------------------------------

def startup_checks(stage_b_run: str, stage_a_run: str) -> None:
    try:
        contextual.verify_stage_b_manifest(contextual.STAGEB_OUT_ROOT / stage_b_run)
        pca_by_cell: dict[tuple, list] = {}
        for ds in config.DATASETS:
            for norm in config.NORMALIZATIONS:
                for seed in config.SEEDS:
                    loaded = run.get_loaded(ds)
                    cell = config.CellId(ds, norm, seed, "pooled", None, "continuous_asot")
                    tok = run.get_tokenizer(cell)
                    stage_b_cell = contextual.load_stage_b_cell(stage_b_run, ds, norm, seed)
                    contextual.verify_tokenizer_continuity(loaded, tok, ds, seed, stage_b_cell)
                    contextual.verify_stage_a_grouping(loaded, tok, ds, norm, seed, stage_a_run,
                                                       config.num_actions(ds))
                    key = (ds, norm)
                    if key not in pca_by_cell:
                        pca_by_cell[key] = tok.pca_windows
                    else:
                        for a, b in zip(pca_by_cell[key], tok.pca_windows):
                            if not np.array_equal(a, b):
                                raise contextual.IdentityError(
                                    f"{ds}/{'norm' if norm else 'raw'}: pca_windows differ across seed "
                                    "caches -- identity check failed")
    except (contextual.IdentityError, contextual.UnavailableError, categorical.UnavailableError) as exc:
        raise SystemExit(f"startup identity check failed: {exc}") from exc
    print("[startup] all 12 (dataset, normalize, seed) cells passed identity checks", flush=True)


# --------------------------------------------------------------------------------------
# File I/O (single writer: the parent process)
# --------------------------------------------------------------------------------------

def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(temp, path)


def _write_csv_atomic(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.csv")
    with temp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def _append_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        fh.flush()
        os.fsync(fh.fileno())


def _save_artifact(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **payload)
    os.replace(temp, path)


def _log_line(run_dir: Path, line: str) -> None:
    path = run_dir / "logs" / "run.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {line}\n")


def _handle_result(run_dir: Path, result: JobResult, status: dict[str, dict]) -> None:
    _append_csv(run_dir / "rows.csv", ROWS_CSV_FIELDS, result.rows)
    _append_csv(run_dir / "fits.csv", FITS_CSV_FIELDS, result.fit_rows)
    _append_csv(run_dir / "probe.csv", PROBE_CSV_FIELDS, result.probe_rows)
    if result.baseline_rows:
        _write_csv_atomic(run_dir / "baseline_check.csv", BASELINE_CSV_FIELDS, result.baseline_rows)
    if result.pilot_data is not None:
        _write_json_atomic(run_dir / "pilot.json", result.pilot_data)
    if result.artifact is not None:
        _save_artifact(run_dir / "artifacts" / f"{result.job_id}.npz", result.artifact)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    status[result.job_id] = dict(job_id=result.job_id, status=result.status, start=now, end=now,
                                 seconds=round(result.seconds, 2), error=result.error or "")
    _write_csv_atomic(run_dir / "status.csv", STATUS_CSV_FIELDS, list(status.values()))
    _log_line(run_dir, f"{result.job_id}: {result.status} ({result.seconds:.1f}s)"
              + (f" -- {result.error}" if result.error else ""))
    print(f"[{result.status}] {result.job_id} ({result.seconds:.1f}s)", flush=True)


def _load_status(run_dir: Path) -> dict[str, dict]:
    path = run_dir / "status.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["job_id"]: row for row in csv.DictReader(fh)}


# --------------------------------------------------------------------------------------
# Manifest / declared analysis rules (never evaluated in code)
# --------------------------------------------------------------------------------------

DECLARED_ANALYSIS_RULES = {
    "Q1": "pooling matches context if the best-h pool result (disc_fit, final rung) is within 3 F1@50 "
          "points of context (disc_saved, final rung) on HuGaDB, in both normalizations.",
    "Q2": "the discovery gap is large if oracle minus disc_* is at least 10 F1@50 points at the final "
          "rung, for at least one representation, on both datasets.",
    "Q3": "LARa stays poor with labels if context oracle is below pca oracle at the final rung, in both "
          "normalizations.",
    "note": "These rules are declared for later analysis and are NOT evaluated by this runner.",
}

_IDENTITY_KEYS = ["frozen_study_version", "rungs", "h_list", "seeds_extra", "seed_primary", "stage_a_run",
                  "stage_b_run", "stage_c_run", "stage_b_config_hash", "probe_subsample", "probe_seed",
                  "fit_outer_cap_default", "fit_outer_cap_extended"]


def build_manifest(args) -> dict:
    try:
        stage_b_manifest = contextual.verify_stage_b_manifest(contextual.STAGEB_OUT_ROOT / STAGE_B_RUN)
    except contextual.IdentityError as exc:
        raise SystemExit(f"cannot build manifest: {exc}") from exc
    return dict(
        run_id=args.run_id, frozen_study_version=FROZEN_STUDY_VERSION, rungs=RUNGS, h_list=list(H_LIST),
        seeds_extra=list(SEEDS_EXTRA), seed_primary=SEED_PRIMARY, stage_a_run=STAGE_A_RUN,
        stage_b_run=STAGE_B_RUN, stage_c_run=STAGE_C_RUN,
        stage_b_config_hash=stage_b_manifest.get("config_hash"), probe_subsample=PROBE_SUBSAMPLE,
        probe_seed=PROBE_SEED, fit_outer_cap_default=FIT_OUTER_CAP_DEFAULT,
        fit_outer_cap_extended=FIT_OUTER_CAP_EXTENDED, workers=args.workers,
        threads_per_worker=args.threads_per_worker, hours=args.hours, host=platform.node(),
        python=sys.version, platform=platform.platform(),
        packages=dict(numpy=np.__version__, sklearn=sklearn.__version__, torch=torch.__version__),
        declared_analysis_rules=DECLARED_ANALYSIS_RULES,
        launches=[dict(started_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"), hours=args.hours,
                      resume=bool(args.resume))],
    )


def _check_manifest_identity(existing: dict, fresh: dict) -> None:
    for key in _IDENTITY_KEYS:
        if existing.get(key) != fresh.get(key):
            raise SystemExit(f"--resume refused: manifest field {key!r} differs "
                             f"({existing.get(key)!r} != {fresh.get(key)!r}); start a new --run-id")


# --------------------------------------------------------------------------------------
# Scheduler
# --------------------------------------------------------------------------------------

def cost_projection(job: Job, pilot: dict) -> float | None:
    if job.kind == "probe":
        return 300.0
    if job.kind not in ("ladder", "fit_ladder"):
        return None
    stats = pilot.get(job.dataset)
    if not stats:
        return None
    spd = stats["seconds_per_recording_step"]
    n_rec = stats["n_recordings_full"]
    cost = RUNGS[-1] * n_rec * spd
    if job.kind == "fit_ladder":
        cost += job.outer_cap * config.FIT_INNER_STEPS * n_rec * spd
    return cost


def _worker_init(threads: int) -> None:
    torch.set_num_threads(threads)


def cmd_run(args) -> None:
    run_dir = OUT_ROOT / args.run_id
    manifest = build_manifest(args)
    if args.resume:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"--resume requires an existing manifest at {manifest_path}")
        _check_manifest_identity(json.loads(manifest_path.read_text()), manifest)
        status = _load_status(run_dir)
    else:
        if (run_dir / "status.csv").exists():
            raise SystemExit(f"{run_dir}/status.csv already exists; pass --resume or choose a new --run-id")
        run_dir.mkdir(parents=True, exist_ok=True)
        status = {}
    _write_json_atomic(run_dir / "manifest.json", manifest)

    startup_checks(STAGE_B_RUN, STAGE_A_RUN)

    jobs = enumerate_smoke_jobs() if args.smoke else enumerate_jobs()
    _write_csv_atomic(run_dir / "jobs.csv", JOBS_CSV_FIELDS, [job_to_row(j) for j in jobs])
    jobs_by_id = {j.job_id: j for j in jobs}

    deadline = time.perf_counter() + args.hours * 3600.0 - 1800.0

    if status.get("check_baselines", {}).get("status") != "complete":
        _handle_result(run_dir, run_check_baselines(jobs_by_id["check_baselines"]), status)
    if status.get("pilot", {}).get("status") != "complete":
        _handle_result(run_dir, run_pilot(jobs_by_id["pilot"]), status)
    pilot_path = run_dir / "pilot.json"
    pilot_data = json.loads(pilot_path.read_text())["dataset_stats"] if pilot_path.exists() else {}

    todo = [j for j in jobs if j.job_id not in ("check_baselines", "pilot")
           and status.get(j.job_id, {}).get("status") != "complete"]
    not_run: list[dict] = []
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=_worker_init,
                            initargs=(args.threads_per_worker,)) as pool:
        futures = {}
        pending = list(todo)
        while pending or futures:
            while pending and len(futures) < args.workers:
                job = pending.pop(0)
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    not_run.append(dict(job_id=job.job_id, reason="compute deadline reached"))
                    continue
                projected = cost_projection(job, pilot_data)
                if projected is not None and projected > remaining:
                    not_run.append(dict(job_id=job.job_id,
                                       reason=f"projected {projected:.0f}s exceeds {remaining:.0f}s remaining"))
                    continue
                futures[pool.submit(execute_job, job, deadline)] = job
            if not futures:
                break
            done, _pending_futures = futures_wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                futures.pop(fut)
                _handle_result(run_dir, fut.result(), status)

    _write_csv_atomic(run_dir / "not_run.csv", NOT_RUN_CSV_FIELDS, not_run)
    print(f"frozen_study {args.run_id}: {len(status)} jobs recorded, {len(not_run)} not run.", flush=True)


def cmd_validate_only(_args) -> None:
    from script.action_transport import tests_frozen_study
    tests_frozen_study.run_all()


def apply_smoke_overrides() -> None:
    global RUNGS, H_LIST, MAX_RECORDINGS_PER_DATASET
    RUNGS = [10, 20]
    H_LIST = (2,)
    MAX_RECORDINGS_PER_DATASET = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen-representation study runner")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--hours", type=float, default=None)
    parser.add_argument("--run-id", dest="run_id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--threads-per-worker", dest="threads_per_worker", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.set_num_threads(min(args.threads_per_worker, os.cpu_count() or 1))
    if args.validate_only:
        return cmd_validate_only(args)
    if args.smoke:
        apply_smoke_overrides()
        args.run_id = args.run_id or "_smoke"
        if not args.run_id.endswith("_smoke"):
            raise SystemExit("-Smoke requires a --run-id ending in _smoke")
        args.hours = args.hours or 1.0
        args.workers = min(args.workers, 2)
        return cmd_run(args)
    if args.run_id and args.run_id.endswith("_smoke"):
        raise SystemExit("--run-id must not end in _smoke unless --smoke is given")
    if args.hours is None or args.run_id is None:
        raise SystemExit("A full run requires --hours and --run-id")
    if args.hours <= 0.5:
        raise SystemExit("--hours must be greater than 0.5")
    cmd_run(args)


if __name__ == "__main__":
    main()
