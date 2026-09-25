"""A-cont: the continuous-feature transport competitor. Same transport
machinery as categorical.py, but D[t,a]=1-z[t].mu[a] on unit vectors (v1.3:
in [0,2], matching the pinned upstream unary convention -- not divided by
two), with prototype updates instead of an emission table
(docs/plans/THREE_STAGE_MOTION_PLAN.md §3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from script.action_transport import transport
from script.action_transport.budget import check_deadline
from script.action_transport.categorical import RecordingInput, UnavailableError, _cold_start


def l2_normalize(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (unit vectors, zero_flags). Zero rows stay zero (flagged)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    zero = (norms[:, 0] <= 1e-12)
    safe_norms = np.where(norms <= 1e-12, 1.0, norms)
    unit = vectors / safe_norms
    unit[zero] = 0.0
    return unit.astype(np.float64), zero


def build_mu0(grouping: np.ndarray, fitting_recordings: list["ContinuousRecordingInput"],
             num_states: int) -> tuple[np.ndarray, np.ndarray]:
    """mu0[a] = normalize(sum over fitting windows with g[code[t]]==a of
    length[t]*z[t]). Returns (mu0, unavailable_flags per state)."""
    dim = fitting_recordings[0].z.shape[1] if fitting_recordings else 0
    accum = np.zeros((num_states, dim), dtype=np.float64)
    for rec in fitting_recordings:
        weights = rec.lengths.astype(np.float64)[:, None]
        groups = grouping[rec.codes]
        for state in range(num_states):
            mask = groups == state
            if mask.any():
                accum[state] += (weights[mask] * rec.z[mask]).sum(axis=0)
    mu0, unavailable = l2_normalize(accum)
    return mu0, unavailable


def cosine_cost(mu: np.ndarray, z: np.ndarray, zero_flags: np.ndarray) -> np.ndarray:
    """D[t,a] = 1 - z[t].mu[a], in [0,2]. Zero input vectors cost 1 for
    every state (v1.3)."""
    d = 1.0 - z @ mu.T  # [L, num_states]
    d[zero_flags] = 1.0
    return d


@dataclass
class ContinuousRecordingInput(RecordingInput):
    z: np.ndarray = None  # float64 [L, dim], unit-normalized
    zero_flags: np.ndarray = None  # bool [L]


@dataclass
class ContinuousFitResult:
    mu: np.ndarray
    t_state: dict[str, torch.Tensor]
    outer_iterations: int
    outer_status: str
    objective_trace: list = field(default_factory=list)
    stale_prototypes: list = field(default_factory=list)  # states whose update was zero this run


def fit_continuous(recordings: list[ContinuousRecordingInput], mu0: np.ndarray, num_states: int,
                   coeffs: transport.Coeffs, inner_steps: int, outer_cap: int,
                   outer_patience: int, outer_reltol: float, device: torch.device,
                   deadline: float | None = None) -> ContinuousFitResult:
    mu = mu0.copy()
    q = torch.full((num_states,), 1.0 / num_states, dtype=torch.float64, device=device)
    t_state: dict[str, torch.Tensor] = {}
    p_cache: dict[str, torch.Tensor] = {}
    weight_cache: dict[str, torch.Tensor | None] = {}
    prev_total = None
    good_outer = 0
    objective_trace = []
    status = "capped"
    outer_used = 0
    stale: list[int] = []
    for outer in range(outer_cap):
        outer_used = outer + 1
        total_obj = 0.0
        any_invalid = False
        accum = np.zeros_like(mu)
        for rec in recordings:
            check_deadline(deadline)
            total_frames = float(rec.lengths.sum())
            if rec.name not in p_cache:
                p_cache[rec.name] = torch.as_tensor(rec.lengths / total_frames, dtype=torch.float64, device=device)
                num_windows = len(rec.codes)
                half_width = transport.kernel_half_width(rec.fps, rec.window, num_windows)
                weight_cache[rec.name] = transport.build_kernel(num_windows, half_width, device)
            p = p_cache[rec.name]
            weights = weight_cache[rec.name]
            cost = torch.as_tensor(cosine_cost(mu, rec.z, rec.zero_flags), dtype=torch.float64, device=device)
            if rec.name not in t_state:
                t_state[rec.name] = _cold_start(p, q)
            result = transport.solve_inner(t_state[rec.name], cost, weights, p, q, coeffs, inner_steps)
            if result.status == "invalid":
                any_invalid = True
                continue
            t_state[rec.name] = result.t
            total_obj += (result.trace[-1] if result.trace else 0.0) * total_frames
            r_matrix = (result.t / p.unsqueeze(1)).cpu().numpy()
            weighted = rec.lengths.astype(np.float64)[:, None] * r_matrix  # [L, num_states]
            accum += weighted.T @ rec.z
        if any_invalid:
            return ContinuousFitResult(mu, t_state, outer_used, "invalid", objective_trace, stale)
        new_mu, zero_update = l2_normalize(accum)
        stale = [int(s) for s in np.flatnonzero(zero_update)]
        for state in stale:
            new_mu[state] = mu[state]
        mu = new_mu
        objective_trace.append(total_obj)
        rel = (abs(total_obj - prev_total) / max(1.0, abs(prev_total))
               if prev_total is not None else float("inf"))
        prev_total = total_obj
        good_outer = good_outer + 1 if rel <= outer_reltol else 0
        if good_outer >= outer_patience:
            status = "converged"
            break
    return ContinuousFitResult(mu, t_state, outer_used, status, objective_trace, stale)


@dataclass
class ContinuousFinalResult:
    predictions: dict[str, np.ndarray]
    status: dict[str, str]
    residuals: dict[str, float | None]


def infer_final(mu: np.ndarray, recordings: list[ContinuousRecordingInput], num_states: int,
                coeffs: transport.Coeffs, device: torch.device, max_steps: int, grad_tol: float,
                rel_tol: float, patience: int, backtrack_max: int,
                warm_start: dict[str, torch.Tensor] | None = None,
                deadline: float | None = None) -> ContinuousFinalResult:
    q = torch.full((num_states,), 1.0 / num_states, dtype=torch.float64, device=device)
    predictions, status, residuals = {}, {}, {}
    for rec in recordings:
        check_deadline(deadline)
        total_frames = float(rec.lengths.sum())
        p = torch.as_tensor(rec.lengths / total_frames, dtype=torch.float64, device=device)
        num_windows = len(rec.codes)
        half_width = transport.kernel_half_width(rec.fps, rec.window, num_windows)
        weights = transport.build_kernel(num_windows, half_width, device)
        cost = torch.as_tensor(cosine_cost(mu, rec.z, rec.zero_flags), dtype=torch.float64, device=device)
        t_init = warm_start[rec.name] if warm_start and rec.name in warm_start else _cold_start(p, q)
        result = transport.solve_final(t_init, cost, weights, p, q, coeffs, max_steps,
                                       grad_tol, rel_tol, patience, backtrack_max)
        status[rec.name] = result.status
        residuals[rec.name] = result.residual
        if result.status != "invalid":
            r_matrix = (result.t / p.unsqueeze(1)).cpu().numpy()
            predictions[rec.name] = r_matrix.argmax(axis=1).astype(np.int32)
    return ContinuousFinalResult(predictions, status, residuals)
