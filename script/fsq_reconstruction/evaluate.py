"""Shared tokenizer/contextualizer/OT integration and reporting for the FSQ
reconstruction pilot (docs/plans/FSQ_RECONSTRUCTION_PILOT_PLAN.md Revision 3
§7-§8). Reuses the generic transport/contextualizer engines from
script/action_transport; never touches contextual.py's file-bound Stage
B/C loaders (this pilot trains brand-new per-stream contextualizers, not
stage_b_003).

K=512 note (§7: "log(512) cost normalization ... Pass an explicit
experiment configuration with K=512 through every helper"):
`categorical.py`'s `emission_cost`/`prior_penalty`/`fit_categorical` hardcode
`LOG_500 = math.log(500)` for the historical K=500 pipeline, which must not
be edited (§7: "Do not modify global historical K=500 settings"). This
module therefore reimplements only those two small cost/prior functions
parameterized by `log_k`, and reuses the generic, already-K-agnostic engine
underneath them (`categorical.fit_alternating`, `categorical.infer_final`,
`categorical.emission_update`, `categorical.recording_tensors`,
`categorical.build_grouping`, `categorical.build_theta0`) unmodified.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.action_transport import categorical, config as at_config, continuous, evaluate as at_evaluate, transport
from script.action_transport.budget import check_deadline
from script.action_transport.stageb_data import build_validation_split, make_chunks
from script.action_transport.stageb_infer import extract_embeddings
from script.action_transport.stageb_model import CodeContextModel, ModelConfig
from script.action_transport.stageb_train import TrainConfig, train as train_stageb
from script.repro.d_error_decomposition import score as score_identity

DEVICE = torch.device("cpu")  # OT solves are CPU per the historical action_transport convention

CONTEXTUALIZER_CONFIG_NAME = "postLN_cosine_wu500_lr0.0003_alibi"  # reused by name, not by loading stage_b_003


def contextualizer_config(**overrides) -> TrainConfig:
    """Resolves the selected architecture (postLN_cosine_wu500_lr0.0003_alibi)
    via stageb_configs.resolve_by_name, with the pilot's step budget (§7:
    'context length 128, stride 64, mask span 8, mask fraction 0.30 and at
    most 5,000 optimizer updates') applied on top."""
    from script.action_transport.stageb_configs import resolve_by_name
    cfg = resolve_by_name(CONTEXTUALIZER_CONFIG_NAME)
    fields = dict(mask_fraction=0.30, mask_span=8, max_steps=5000, eval_every=100, patience_steps=1000)
    fields.update(overrides)
    from dataclasses import replace
    return replace(cfg, **fields)


# --- categorical OT at an explicit K (log(K) cost, not the historical log(500)) --------

def emission_cost_k(theta: np.ndarray, codes: np.ndarray, log_k: float) -> np.ndarray:
    return (-np.log(theta[:, codes]) / log_k).T


def prior_penalty_k(theta: np.ndarray, a: float, eta: float, num_codes: int, log_k: float) -> float:
    return float(-a * eta / (num_codes * log_k) * np.log(theta).sum())


def fit_categorical_k(recordings: list[categorical.RecordingInput], theta0: np.ndarray, num_states: int,
                      num_codes: int, eta: float, coeffs: transport.Coeffs, inner_steps: int, outer_cap: int,
                      outer_patience: int, outer_reltol: float, device: torch.device, log_k: float,
                      deadline: float | None = None) -> categorical.FitResult:
    def cost_fn(theta, rec):
        return emission_cost_k(theta, rec.codes, log_k)

    def update_fn(recs, responsibilities, _theta):
        new_theta, _ = categorical.emission_update(recs, responsibilities, num_states, num_codes, eta)
        return new_theta, []

    def prior_fn(theta):
        return prior_penalty_k(theta, coeffs.a, eta, num_codes, log_k)

    return categorical.fit_alternating(recordings, theta0, num_states, cost_fn, update_fn, prior_fn, coeffs,
                                       inner_steps, outer_cap, outer_patience, outer_reltol, device, deadline)


def per_code_pca_centers(recs: list[categorical.RecordingInput], pca_by_name: dict[str, np.ndarray],
                         num_codes: int) -> tuple[np.ndarray, np.ndarray]:
    """§7: 'calculate each occupied unit's frame-weighted mean in that arm's
    encoder-window PCA space'. Works identically for native codes (an
    empirical center, not a learned parameter) and K-means codes (close to,
    but not necessarily identical to, the fitted K-means centroid, since
    K-means was fit on a <=10,000-patch sample)."""
    dim = next(iter(pca_by_name.values())).shape[1]
    accum = np.zeros((num_codes, dim), dtype=np.float64)
    counts = np.zeros(num_codes, dtype=np.float64)
    for rec in recs:
        feats = pca_by_name[rec.name]
        w = rec.lengths.astype(np.float64)
        np.add.at(accum, rec.codes, w[:, None] * feats)
        np.add.at(counts, rec.codes, w)
    centers = np.zeros_like(accum)
    occupied = counts > 0
    centers[occupied] = accum[occupied] / counts[occupied][:, None]
    return centers, counts


@dataclass
class RungResult:
    rung_steps: int
    MoF: float
    Edit: float
    F1_10: float
    F1_25: float
    F1_50: float
    status_counts: dict
    seconds: float


def categorical_final_ladder(theta: np.ndarray, recs: list[categorical.RecordingInput], num_states: int,
                             log_k: float, coeffs: transport.Coeffs, rungs: tuple[int, ...],
                             gts_by_name: dict[str, np.ndarray], starts_by_name: dict[str, np.ndarray],
                             frame_counts_by_name: dict[str, int], deadline: float | None = None) -> list[RungResult]:
    """Cumulative-rung cold-start final solve (frozen_study.py's ladder
    convention), evaluated under dataset-level Hungarian mapping."""

    def cost_fn(th, rec):
        return emission_cost_k(th, rec.codes, log_k)

    log_t: dict[str, torch.Tensor] = {}
    steps_done: dict[str, int] = {}
    status: dict[str, str] = {}
    out = []
    for rung in rungs:
        started = time.perf_counter()
        for rec in recs:
            check_deadline(deadline)
            if status.get(rec.name) in ("converged", "invalid"):
                continue
            remaining = rung - steps_done.get(rec.name, 0)
            if remaining <= 0:
                continue
            p, q, weights = categorical.recording_tensors(rec, num_states, DEVICE)
            cost = torch.as_tensor(cost_fn(theta, rec), dtype=torch.float64, device=DEVICE)
            init = log_t.get(rec.name) if rec.name in log_t else transport.cold_start_log(p, q)
            result = transport.solve_final(init, cost, weights, p, q, coeffs, remaining,
                                           at_config.FINAL_GRAD_TOL, at_config.FINAL_OBJ_RELTOL,
                                           at_config.FINAL_PATIENCE, at_config.BACKTRACK_MAX)
            log_t[rec.name] = result.log_t
            steps_done[rec.name] = steps_done.get(rec.name, 0) + result.accepted_steps
            status[rec.name] = result.status
        states = {name: t.argmax(dim=1).cpu().numpy().astype(np.int32) for name, t in log_t.items()}
        frame_preds, frame_gts = [], []
        for rec in recs:
            s = states.get(rec.name)
            if s is None:
                continue
            frame_preds.append(at_evaluate.broadcast_to_frames(s, starts_by_name[rec.name], rec.lengths,
                                                                frame_counts_by_name[rec.name]))
            frame_gts.append(gts_by_name[rec.name])
        metrics, _ = at_evaluate.score_pooled(frame_gts, frame_preds) if frame_preds else ({
            "MoF": float("nan"), "Edit": float("nan"), "F1@10": float("nan"), "F1@25": float("nan"),
            "F1@50": float("nan")}, {})
        counts = {}
        for s in status.values():
            counts[s] = counts.get(s, 0) + 1
        out.append(RungResult(rung, metrics["MoF"], metrics["Edit"], metrics["F1@10"], metrics["F1@25"],
                              metrics["F1@50"], counts, time.perf_counter() - started))
    return out


# --- contextualizer (fresh per stream/seed; §7) ---------------------------------------

@dataclass
class ContextualizerResult:
    model: CodeContextModel
    info: dict


def train_contextualizer(codes_by_name: dict[str, np.ndarray], lengths_by_name: dict[str, np.ndarray],
                         frame_counts: dict[str, int], num_codes: int, seed: int, split_seed: int,
                         device: torch.device = DEVICE, deadline: float | None = None,
                         max_steps: int = 5000, eval_every: int = 100,
                         patience_steps: int = 1000) -> ContextualizerResult:
    fit_names = sorted(codes_by_name)
    train_names, val_names = build_validation_split(fit_names, frame_counts, seed=split_seed)
    train_chunks = make_chunks({n: codes_by_name[n] for n in train_names}, lengths_by_name)
    val_chunks = make_chunks({n: codes_by_name[n] for n in val_names}, lengths_by_name)
    cfg = contextualizer_config(max_steps=max_steps, eval_every=eval_every, patience_steps=patience_steps)
    model, info = train_stageb(train_chunks, val_chunks, num_codes, seed, device, cfg,
                               mask_base=at_config.SAMPLING_SEED, deadline=deadline)
    return ContextualizerResult(model, info)


def contextual_embeddings(model: CodeContextModel, codes_by_name: dict[str, np.ndarray],
                          device: torch.device = DEVICE) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for name, codes in codes_by_name.items():
        unit, zero = extract_embeddings(model, codes.astype(np.int64), device)
        out[name] = (unit, zero)
    return out


def contextual_final_ladder(mu: np.ndarray, recs: list[continuous.ContinuousRecordingInput],
                            coeffs: transport.Coeffs, rungs: tuple[int, ...],
                            gts_by_name: dict[str, np.ndarray], starts_by_name: dict[str, np.ndarray],
                            frame_counts_by_name: dict[str, int], deadline: float | None = None) -> list[RungResult]:
    num_states = mu.shape[0]
    log_t: dict[str, torch.Tensor] = {}
    steps_done: dict[str, int] = {}
    status: dict[str, str] = {}
    out = []
    for rung in rungs:
        started = time.perf_counter()
        for rec in recs:
            check_deadline(deadline)
            if status.get(rec.name) in ("converged", "invalid"):
                continue
            remaining = rung - steps_done.get(rec.name, 0)
            if remaining <= 0:
                continue
            p, q, weights = categorical.recording_tensors(rec, num_states, DEVICE)
            cost = torch.as_tensor(continuous.continuous_cost_fn(mu, rec), dtype=torch.float64, device=DEVICE)
            init = log_t.get(rec.name) if rec.name in log_t else transport.cold_start_log(p, q)
            result = transport.solve_final(init, cost, weights, p, q, coeffs, remaining,
                                           at_config.FINAL_GRAD_TOL, at_config.FINAL_OBJ_RELTOL,
                                           at_config.FINAL_PATIENCE, at_config.BACKTRACK_MAX)
            log_t[rec.name] = result.log_t
            steps_done[rec.name] = steps_done.get(rec.name, 0) + result.accepted_steps
            status[rec.name] = result.status
        states = {name: t.argmax(dim=1).cpu().numpy().astype(np.int32) for name, t in log_t.items()}
        frame_preds, frame_gts = [], []
        for rec in recs:
            s = states.get(rec.name)
            if s is None:
                continue
            frame_preds.append(at_evaluate.broadcast_to_frames(s, starts_by_name[rec.name], rec.lengths,
                                                                frame_counts_by_name[rec.name]))
            frame_gts.append(gts_by_name[rec.name])
        metrics, _ = at_evaluate.score_pooled(frame_gts, frame_preds) if frame_preds else ({
            "MoF": float("nan"), "Edit": float("nan"), "F1@10": float("nan"), "F1@25": float("nan"),
            "F1@50": float("nan")}, {})
        counts = {}
        for s in status.values():
            counts[s] = counts.get(s, 0) + 1
        out.append(RungResult(rung, metrics["MoF"], metrics["Edit"], metrics["F1@10"], metrics["F1@25"],
                              metrics["F1@50"], counts, time.perf_counter() - started))
    return out


# --- report -----------------------------------------------------------------------

def rung_rows_to_dicts(stream: str, seed: int, method: str, rows: list[RungResult]) -> list[dict]:
    out = []
    for r in rows:
        out.append(dict(stream=stream, downstream_seed=seed, method=method, rung_steps=r.rung_steps,
                       MoF=r.MoF, Edit=r.Edit, **{"F1@10": r.F1_10, "F1@25": r.F1_25, "F1@50": r.F1_50},
                       status_counts=r.status_counts, seconds=r.seconds))
    return out
