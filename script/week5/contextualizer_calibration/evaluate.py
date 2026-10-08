"""Checkpoint decoding for the calibration pilot: contextual embeddings
(explicit context/stride), pooled continuous-cosine OT fit (CPU), rung ladder
scoring, evaluation identities (for exact de-duplication) and the Task 2
reproduction comparison.

The fit/score path reuses `quantized_input_comparison.ot_arms` and
`action_transport.continuous` unchanged: 10 states, a=0.7, beta=0.3,
lambda=0.05, epsilon=0.07, 25 inner steps, outer cap 50 (150 for the solver
sensitivity), final rungs continuing transport with the fitted prototypes.
Labels are read only inside scoring (`labels.gts_for`), never while fitting
or choosing a checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import torch

from script.action_transport import continuous
from script.action_transport.budget import check_deadline
from script.action_transport.stageb_data import chunk_starts
from script.action_transport.stageb_model import CodeContextModel
from script.fsq_reconstruction.evaluate import FinalSolveInvalid
from script.fsq_reconstruction.experiment import write_json_atomic
from script.quantized_input_comparison import contextual as C
from script.quantized_input_comparison import ot_arms as ot
from script.week5.contextualizer_calibration import train as T
from script.week5.contextualizer_calibration.plan import JobSpec, Scale, stable_hash

METRICS = ("MoF", "Edit", "F1_10", "F1_25", "F1_50")
CSV_METRICS = ("MoF", "Edit", "F1@10", "F1@25", "F1@50")


class EvalInvalid(RuntimeError):
    """The OT fit hit a nonfinite value or a backtracking failure."""


def array_sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# --- embeddings ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(arm: str, model: CodeContextModel, loader, family: str, name: str, lookup, rms: float,
                       device: torch.device, chunk: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Every window retained, no masking; chunks of `chunk` tokens at `stride`
    with a final anchored chunk (`chunk_starts`), padded and key-masked,
    hidden states averaged where chunks overlap, then L2-normalized. For
    (128, 64) this is `contextual.extract_vector_embeddings` exactly."""
    model.eval()
    rec = loader.recording(family, name)
    codes, continuous_feat = rec["codes"], rec["continuous"]
    length = len(codes)
    accum = np.zeros((length, model.cfg.d_model), dtype=np.float64)
    counts = np.zeros(length, dtype=np.float64)
    for start in chunk_starts(length, chunk, stride):
        end = min(start + chunk, length)
        valid = end - start
        pad = torch.zeros(1, chunk, dtype=torch.bool)
        pad[0, valid:] = True
        if arm in ("context_id", "context_geometry_init"):
            ids = np.zeros(chunk, dtype=np.int64)
            ids[:valid] = codes[start:end]
            hidden, _ = model.forward(torch.as_tensor(ids[None], dtype=torch.int64, device=device), pad.to(device))
        else:
            dim = lookup.shape[1]
            vecs = np.zeros((chunk, dim), dtype=np.float32)
            vecs[:valid] = lookup[codes[start:end]] / rms
            hidden, _ = model.forward_vectors(torch.as_tensor(vecs[None], dtype=torch.float32, device=device),
                                              torch.zeros(1, chunk, dtype=torch.bool, device=device), pad.to(device))
        accum[start:end] += hidden[0, :valid].cpu().numpy().astype(np.float64)
        counts[start:end] += 1.0
    mean = accum / counts[:, None]
    norms = np.linalg.norm(mean, axis=1, keepdims=True)
    zero = norms[:, 0] <= 1e-12
    unit = mean / np.where(zero[:, None], 1.0, norms)
    unit[zero] = 0.0
    return unit.astype(np.float32), zero


# --- identities ---------------------------------------------------------------------------

def solver_settings(outer_cap: int, rungs: tuple) -> dict:
    return dict(num_states=ot.NUM_STATES, coeffs=asdict(ot.COEFFS), inner_steps=ot.FIT_INNER_STEPS,
                outer_cap=outer_cap, outer_patience=ot.FIT_OUTER_PATIENCE, outer_reltol=ot.FIT_OUTER_RELTOL,
                rungs=list(rungs), eta=ot.ETA)


def eval_identity(*, state_sha: str, job: JobSpec, scale: Scale, outer_cap: int, qic_hash: str, loader,
                  names: list[str]) -> str:
    """Two evaluations share results only when tensors, architecture,
    input/preprocessing, inference settings, grouping and solver all agree.
    Equal losses, steps or file names are never enough."""
    family = job.family
    cfg = T.train_config(job, scale)
    return stable_hash(dict(
        state=state_sha, arch=T.model_settings(cfg), arm=job.arm, family=family,
        input=dict(qic=qic_hash, lookup=array_sha(loader.lookup(family)), rms=float(loader.rms(family))),
        inference=dict(context=job.context, stride=job.stride, anchoring="final-anchored-v1"),
        grouping=array_sha(loader.grouping(family)), solver=solver_settings(outer_cap, scale.rungs),
        recordings=stable_hash(names)))


# --- decoding -----------------------------------------------------------------------------

def build_model_from_state(job: JobSpec, scale: Scale, loader, state: dict, device: torch.device) -> CodeContextModel:
    lookup = loader.lookup(job.family)
    cfg = T.train_config(job, scale)
    mode, dim = ("vector", lookup.shape[1]) if job.arm == "context_quantized" else ("id", None)
    model = CodeContextModel(C.model_config(lookup.shape[0], mode, dim, cfg))
    model.load_state_dict(state)
    return model.to(device).eval()


def decode_state(job: JobSpec, scale: Scale, loader, state: dict, outer_cap: int, device: torch.device,
                 deadline: float | None, out_dir: Path, meta: dict) -> dict:
    """Embeddings -> pooled OT fit -> scored rung ladder for one checkpoint
    state; writes embeddings / prototypes / traces / predictions / ladder."""
    started = time.perf_counter()
    names = T.bounded_split(loader, scale.recording_cap)[0]
    family = job.family
    lookup, rms = loader.lookup(family), float(loader.rms(family))
    model = build_model_from_state(job, scale, loader, state, device)
    emb, zero, codes, lens, starts = {}, {}, {}, {}, {}
    extract_started = time.perf_counter()
    for n in names:
        check_deadline(deadline)
        emb[n], zero[n] = extract_embeddings(job.arm, model, loader, family, n, lookup, rms, device,
                                             job.context, job.stride)
        rec = loader.recording(family, n)
        codes[n], lens[n], starts[n] = rec["codes"], rec["lengths"], rec["starts"]
    extract_seconds = time.perf_counter() - extract_started
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "embeddings.npz", **{f"{n}__emb": emb[n] for n in names},
                        **{f"{n}__zero": zero[n] for n in names})
    recs = ot.continuous_recordings_from_embeddings(codes, lens, emb, zero)
    mu0 = ot.mu0_from_grouping(loader.grouping(family), recs)
    fit = continuous.fit_continuous(recs, mu0, ot.NUM_STATES, ot.COEFFS, ot.FIT_INNER_STEPS, outer_cap,
                                    ot.FIT_OUTER_PATIENCE, ot.FIT_OUTER_RELTOL, ot.DEVICE, deadline)
    np.savez(out_dir / "ot.npz", mu0=mu0, mu=fit.params, objective_trace=np.asarray(fit.objective_trace),
             outer_status=fit.outer_status, outer_iterations=fit.outer_iterations)
    if fit.outer_status == "invalid":
        raise EvalInvalid(f"OT fit invalid (outer_status={fit.outer_status}, iterations={fit.outer_iterations})")
    frame_counts = {n: loader.frame_count(n) for n in names}
    predictions: dict = {}
    ladder = ot.score_continuous(fit.params, recs, starts, frame_counts, names, deadline, rungs=scale.rungs,
                                 predictions_out=predictions)
    window_states = predictions.pop("__window_states__", {})
    np.savez_compressed(out_dir / "predictions.npz", **{f"{n}__frame_pred": predictions[n] for n in predictions},
                        **{f"{n}__window_states": window_states[n] for n in window_states})
    info = dict(status="done", outer_cap=outer_cap, ot_status=fit.outer_status, ot_outer_iterations=fit.outer_iterations,
                n_scored_recordings=len(names), scored_recording_names=names, device=str(device),
                extract_seconds=extract_seconds, seconds=time.perf_counter() - started, **meta)
    write_json_atomic(out_dir / "info.json", info)
    write_json_atomic(out_dir / "ladder.json", dict(rows=[asdict(r) for r in ladder]))
    return info


def eval_complete(out_dir: Path, identity: str, scale: Scale, loader=None, num_states: int | None = None) -> bool:
    """Reusable only when artifacts exist, the identity matches, metrics are
    finite, every rung is present and predictions cover exactly the scored
    recordings' frames and windows."""
    required = [out_dir / n for n in ("info.json", "ladder.json", "ot.npz", "predictions.npz", "embeddings.npz")]
    if any(not p.exists() or p.stat().st_size == 0 for p in required):
        return False
    try:
        info = json.loads((out_dir / "info.json").read_text())
        rows = json.loads((out_dir / "ladder.json").read_text())["rows"]
        if info.get("status") != "done" or info.get("identity") != identity or info.get("ot_status") == "invalid":
            return False
        if {r["rung_steps"] for r in rows} != set(scale.rungs) or len(rows) != len(scale.rungs):
            return False
        if any(not math.isfinite(float(r[m])) for r in rows for m in METRICS):
            return False
        if any(int(r.get("status_counts", {}).get("invalid", 0)) > 0 for r in rows):
            return False
        names = info["scored_recording_names"]
        if not names or len(names) != info["n_scored_recordings"] or len(set(names)) != len(names):
            return False
        if loader is not None and set(names) != set(T.bounded_split(loader, scale.recording_cap)[0]):
            return False
        num_states = ot.NUM_STATES if num_states is None else num_states
        with np.load(out_dir / "ot.npz") as proto:
            if not {"mu0", "mu", "objective_trace", "outer_status", "outer_iterations"} <= set(proto.files):
                return False
            mu0, mu, trace = proto["mu0"], proto["mu"], proto["objective_trace"]
            if mu.ndim != 2 or mu.shape[0] != num_states or mu0.shape != mu.shape:
                return False
            if not (np.isfinite(mu).all() and np.isfinite(mu0).all() and np.isfinite(trace).all()):
                return False
            if str(proto["outer_status"]) == "invalid":
                return False
            width = mu.shape[1]
        fam = info.get("family")
        with np.load(out_dir / "embeddings.npz") as emb_file:
            if set(emb_file.files) != {f"{n}__emb" for n in names} | {f"{n}__zero" for n in names}:
                return False
            for n in names:
                emb, zero = emb_file[f"{n}__emb"], emb_file[f"{n}__zero"]
                if emb.ndim != 2 or emb.shape[1] != width or zero.shape != (emb.shape[0],) or zero.dtype != bool:
                    return False
                if not np.isfinite(emb).all():
                    return False
                if loader is not None and emb.shape[0] != loader.manifest["num_windows"][fam][n]:
                    return False
        with np.load(out_dir / "predictions.npz") as pred:
            if set(pred.files) != {f"{n}__frame_pred" for n in names} | {f"{n}__window_states" for n in names}:
                return False
            for n in names:
                frame, states = pred[f"{n}__frame_pred"], pred[f"{n}__window_states"]
                if frame.ndim != 1 or states.ndim != 1 or not len(frame) or not len(states):
                    return False
                if not (np.issubdtype(frame.dtype, np.integer) and np.issubdtype(states.dtype, np.integer)):
                    return False
                if states.min() < 0 or states.max() >= num_states or frame.min() < 0 or frame.max() >= num_states:
                    return False
                if loader is not None and (len(frame) != loader.frame_count(n) or
                                           len(states) != loader.manifest["num_windows"][fam][n]):
                    return False
    except (KeyError, TypeError, ValueError, OSError, EOFError, BadZipFile, json.JSONDecodeError):
        return False
    return True


def rescore_predictions(out_dir: Path) -> dict:
    """Recompute pooled metrics from the saved final-rung frame predictions
    (labels read here, for scoring only)."""
    from script.action_transport import evaluate as at_evaluate
    from script.quantized_input_comparison import labels as qic_labels
    info = json.loads((out_dir / "info.json").read_text())
    names = sorted(info["scored_recording_names"])
    gts = qic_labels.gts_for(names)
    with np.load(out_dir / "predictions.npz") as pred:
        metrics, _ = at_evaluate.score_pooled([gts[n] for n in names], [pred[f"{n}__frame_pred"] for n in names])
    return metrics


def ladder_rows(rows: list[dict]) -> list[dict]:
    return [dict(rung_steps=r["rung_steps"], MoF=r["MoF"], Edit=r["Edit"], **{"F1@10": r["F1_10"], "F1@25": r["F1_25"],
               "F1@50": r["F1_50"]}, status_counts=r["status_counts"], seconds=r["seconds"]) for r in rows]


# --- reproduction against Task 2 ----------------------------------------------------------

def compare_to_task2(job_info: dict, best_state_sha: str, eval_dir: Path, task2_cell: Path) -> dict:
    """Compare a reference job's validation-selected decode with the saved Task
    2 cell: all five metrics at every rung (exact), selected step, validation
    trajectory, checkpoint tensors and embeddings. The gate is exact
    agreement of metrics and selected step; trajectory/tensor/embedding
    differences are recorded for investigation, never explained away."""
    t2_info = json.loads((task2_cell / "info.json").read_text())
    t2_rows = {r["rung_steps"]: r for r in json.loads((task2_cell / "ladder.json").read_text())["rows"]}
    mine = {r["rung_steps"]: r for r in json.loads((eval_dir / "ladder.json").read_text())["rows"]}
    diffs = {f"{m}@{rung}": float(mine[rung][m]) - float(t2_rows[rung][m]) for rung in sorted(t2_rows) if rung in mine
             for m in METRICS}
    traj = [abs(a["val_loss"] - b["val_loss"]) for a, b in zip(job_info["history"], t2_info["history"])]
    t2_state = torch.load(task2_cell / "checkpoint_best.pt", map_location="cpu")["state_dict"]
    with np.load(eval_dir / "embeddings.npz") as mine_emb, np.load(task2_cell / "embeddings.npz") as t2_emb:
        keys_equal = set(mine_emb.files) == set(t2_emb.files)
        emb_max = max((float(np.abs(mine_emb[k].astype(np.float64) - t2_emb[k].astype(np.float64)).max())
                       for k in t2_emb.files if k in mine_emb.files and mine_emb[k].shape == t2_emb[k].shape),
                      default=float("nan"))
    metrics_exact = bool(diffs) and all(v == 0.0 for v in diffs.values()) and set(t2_rows) == set(mine)
    step_equal = job_info["selected_step"] == t2_info["best_step"]
    return dict(metrics_exact=metrics_exact, selected_step_equal=step_equal, selected_step=job_info["selected_step"],
                task2_selected_step=t2_info["best_step"], metric_diffs=diffs,
                history_length_equal=len(job_info["history"]) == len(t2_info["history"]),
                validation_max_abs_diff=max(traj) if traj else None,
                checkpoint_tensors_equal=best_state_sha == T.tensor_hash(t2_state),
                embeddings_keys_equal=keys_equal, embeddings_max_abs_diff=emb_max,
                f1_50_rung_4000=float(mine[max(mine)]["F1_50"]) if mine else None,
                task2_f1_50_rung_4000=float(t2_rows[max(t2_rows)]["F1_50"]),
                passed=metrics_exact and step_equal)
