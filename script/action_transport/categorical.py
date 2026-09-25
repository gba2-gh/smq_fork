"""Categorical arms A0, A1, A1-KC: vocabulary-level initialization, the
fixed-denominator emission cost, the pooled emission update, and the
alternating fit (docs/plans/THREE_STAGE_MOTION_PLAN.md §3, instructions §4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.cluster import KMeans

from script.action_transport import transport
from script.action_transport.budget import check_deadline

LOG_500 = math.log(500)


class UnavailableError(RuntimeError):
    """A configuration is unavailable per a declared, label-free rule (e.g.
    fewer than C distinct occupied prototypes). Never worked around."""


@dataclass
class RecordingInput:
    name: str
    codes: np.ndarray  # int32 [L]
    lengths: np.ndarray  # int32 [L], real frame counts
    fps: int
    window: int


def build_grouping(centers: np.ndarray, counts: np.ndarray, num_states: int,
                   seed: int, is_kc_identity: bool) -> np.ndarray:
    """g(u): vocabulary-level initialization grouping (instructions §4).

    At K=C with distinct occupied prototypes, g[u]=u directly (no redundant
    KMeans fit). Otherwise, weighted KMeans on occupied prototype centers,
    with unoccupied prototypes taking the nearest fitted group center.
    """
    num_codes = centers.shape[0]
    occupied = np.flatnonzero(counts > 0)
    if len(occupied) < num_states:
        raise UnavailableError(
            f"only {len(occupied)} occupied prototypes, need >= {num_states}")
    if is_kc_identity:
        if num_codes != num_states:
            raise UnavailableError("K=C identity grouping requires K==C")
        return np.arange(num_codes, dtype=np.int32)
    model = KMeans(n_clusters=num_states, init="k-means++", n_init=5, max_iter=300,
                   tol=1e-4, algorithm="lloyd", random_state=seed)
    model.fit(centers[occupied], sample_weight=counts[occupied])
    grouping = np.empty(num_codes, dtype=np.int32)
    grouping[occupied] = model.labels_.astype(np.int32)
    unoccupied = np.flatnonzero(counts == 0)
    if len(unoccupied):
        deltas = centers[unoccupied][:, None, :] - model.cluster_centers_[None, :, :]
        distances = (deltas ** 2).sum(axis=2)
        grouping[unoccupied] = distances.argmin(axis=1).astype(np.int32)
    return grouping


def frame_weighted_counts(recordings: list[RecordingInput], num_codes: int) -> np.ndarray:
    counts = np.zeros(num_codes, dtype=np.float64)
    for rec in recordings:
        counts += np.bincount(rec.codes, weights=rec.lengths.astype(np.float64), minlength=num_codes)
    return counts


def build_theta0(grouping: np.ndarray, counts: np.ndarray, num_states: int,
                 num_codes: int, eta: float) -> np.ndarray:
    theta = np.zeros((num_states, num_codes), dtype=np.float64)
    for state in range(num_states):
        mask = grouping == state
        theta[state, mask] = counts[mask]
    theta = theta + eta / num_codes
    theta = theta / theta.sum(axis=1, keepdims=True)
    return theta


def emission_cost(theta: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """D[t,a] = -log(theta[a, code[t]]) / log(500), fixed denominator for
    every vocabulary size (v1.3 correction: never switch to log(C))."""
    return (-np.log(theta[:, codes]) / LOG_500).T  # [L, num_states]


def emission_update(recordings: list[RecordingInput], responsibilities: dict[str, np.ndarray],
                    num_states: int, num_codes: int, eta: float) -> np.ndarray:
    count = np.zeros((num_states, num_codes), dtype=np.float64)
    for rec in recordings:
        r_matrix = responsibilities[rec.name]  # [L, num_states]
        weighted = rec.lengths.astype(np.float64)[:, None] * r_matrix
        for state in range(num_states):
            count[state] += np.bincount(rec.codes, weights=weighted[:, state], minlength=num_codes)
    theta = (count + eta / num_codes)
    theta = theta / theta.sum(axis=1, keepdims=True)
    return theta, count


def prior_penalty(theta: np.ndarray, a: float, eta: float, num_codes: int) -> float:
    """-a*eta/(K*log(500)) * sum(log(theta)) from the complete fitting
    objective (docs plan §3)."""
    return float(-a * eta / (num_codes * LOG_500) * np.log(theta).sum())


def _cold_start(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return p.unsqueeze(1) * q.unsqueeze(0)


@dataclass
class FitResult:
    theta: np.ndarray
    responsibilities: dict[str, np.ndarray]  # per recording, R = T/p[:,None], final-outer (pre re-solve)
    t_state: dict[str, torch.Tensor]  # per recording, warm-startable T after fitting
    outer_iterations: int
    outer_status: str  # "converged" | "capped" | "invalid"
    objective_trace: list = field(default_factory=list)


def fit_categorical(recordings: list[RecordingInput], theta0: np.ndarray, num_states: int,
                    num_codes: int, eta: float, coeffs: transport.Coeffs, inner_steps: int,
                    outer_cap: int, outer_patience: int, outer_reltol: float,
                    device: torch.device, deadline: float | None = None) -> FitResult:
    """Alternating inexact minimization (instructions §5): each outer
    iteration solves every fitting recording with `inner_steps` warm-started
    accepted mirror steps, then applies one pooled emission update."""
    theta = theta0.copy()
    q = torch.full((num_states,), 1.0 / num_states, dtype=torch.float64, device=device)
    t_state: dict[str, torch.Tensor] = {}
    p_cache: dict[str, torch.Tensor] = {}
    weight_cache: dict[str, torch.Tensor | None] = {}
    prev_total = None
    good_outer = 0
    objective_trace = []
    status = "capped"
    outer_used = 0
    for outer in range(outer_cap):
        outer_used = outer + 1
        total_obj = 0.0
        any_invalid = False
        responsibilities: dict[str, np.ndarray] = {}
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
            cost = torch.as_tensor(emission_cost(theta, rec.codes), dtype=torch.float64, device=device)
            if rec.name not in t_state:
                t_state[rec.name] = _cold_start(p, q)
            result = transport.solve_inner(t_state[rec.name], cost, weights, p, q, coeffs, inner_steps)
            if result.status == "invalid":
                any_invalid = True
                continue
            t_state[rec.name] = result.t
            total_obj += (result.trace[-1] if result.trace else 0.0) * total_frames
            responsibilities[rec.name] = (result.t / p.unsqueeze(1)).cpu().numpy()
        if any_invalid:
            return FitResult(theta, responsibilities, t_state, outer_used, "invalid", objective_trace)
        theta, _ = emission_update(recordings, responsibilities, num_states, num_codes, eta)
        total_obj += prior_penalty(theta, coeffs.a, eta, num_codes)
        objective_trace.append(total_obj)
        rel = (abs(total_obj - prev_total) / max(1.0, abs(prev_total))
               if prev_total is not None else float("inf"))
        prev_total = total_obj
        good_outer = good_outer + 1 if rel <= outer_reltol else 0
        if good_outer >= outer_patience:
            status = "converged"
            break
    return FitResult(theta, responsibilities, t_state, outer_used, status, objective_trace)


@dataclass
class FinalResult:
    predictions: dict[str, np.ndarray]  # per recording, raw argmax state id per window
    status: dict[str, str]
    residuals: dict[str, float | None]

    def worst_status(self) -> str:
        order = {"converged": 0, "capped": 1, "invalid": 2}
        statuses = list(self.status.values()) or ["converged"]
        return max(statuses, key=lambda s: order.get(s, 2))


def infer_final(theta: np.ndarray, recordings: list[RecordingInput], num_states: int,
                warm_start: dict[str, torch.Tensor] | None, coeffs: transport.Coeffs,
                device: torch.device, max_steps: int, grad_tol: float, rel_tol: float,
                patience: int, backtrack_max: int, deadline: float | None = None) -> FinalResult:
    """Final re-solve at frozen theta, one setting. Used both for the
    fitting population's final re-solve and for frozen held-out inference."""
    q = torch.full((num_states,), 1.0 / num_states, dtype=torch.float64, device=device)
    predictions, status, residuals = {}, {}, {}
    for rec in recordings:
        check_deadline(deadline)
        total_frames = float(rec.lengths.sum())
        p = torch.as_tensor(rec.lengths / total_frames, dtype=torch.float64, device=device)
        num_windows = len(rec.codes)
        half_width = transport.kernel_half_width(rec.fps, rec.window, num_windows)
        weights = transport.build_kernel(num_windows, half_width, device)
        cost = torch.as_tensor(emission_cost(theta, rec.codes), dtype=torch.float64, device=device)
        t_init = warm_start[rec.name] if warm_start and rec.name in warm_start else _cold_start(p, q)
        result = transport.solve_final(t_init, cost, weights, p, q, coeffs, max_steps,
                                       grad_tol, rel_tol, patience, backtrack_max)
        status[rec.name] = result.status
        residuals[rec.name] = result.residual
        if result.status != "invalid":
            r_matrix = (result.t / p.unsqueeze(1)).cpu().numpy()
            predictions[rec.name] = r_matrix.argmax(axis=1).astype(np.int32)
    return FinalResult(predictions, status, residuals)
