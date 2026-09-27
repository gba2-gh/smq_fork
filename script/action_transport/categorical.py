"""Categorical arms A0, A1, A1-KC: vocabulary-level initialization, the
fixed-denominator emission cost, the pooled emission update, and the
alternating fit (docs/plans/THREE_STAGE_MOTION_PLAN.md §3, instructions §4).
Solver states are log T (see transport.py, v1.3.1).
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
    KMeans fit). Otherwise weighted KMeans on occupied prototype centers,
    with unoccupied prototypes taking the nearest fitted group center.
    """
    num_codes = centers.shape[0]
    occupied = np.flatnonzero(counts > 0)
    if len(occupied) < num_states:
        raise UnavailableError(f"only {len(occupied)} occupied prototypes, need >= {num_states}")
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
        grouping[unoccupied] = (deltas ** 2).sum(axis=2).argmin(axis=1).astype(np.int32)
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
    return theta / theta.sum(axis=1, keepdims=True)


def emission_cost(theta: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """D[t,a] = -log(theta[a, code[t]]) / log(500), fixed denominator for
    every vocabulary size (v1.3: never switch to log(C))."""
    return (-np.log(theta[:, codes]) / LOG_500).T  # [L, num_states]


def emission_update(recordings: list[RecordingInput], responsibilities: dict[str, np.ndarray],
                    num_states: int, num_codes: int, eta: float):
    count = np.zeros((num_states, num_codes), dtype=np.float64)
    for rec in recordings:
        weighted = rec.lengths.astype(np.float64)[:, None] * responsibilities[rec.name]
        for state in range(num_states):
            count[state] += np.bincount(rec.codes, weights=weighted[:, state], minlength=num_codes)
    theta = count + eta / num_codes
    return theta / theta.sum(axis=1, keepdims=True), count


def prior_penalty(theta: np.ndarray, a: float, eta: float, num_codes: int) -> float:
    """-a*eta/(K*log(500)) * sum(log(theta)) from the complete fitting objective."""
    return float(-a * eta / (num_codes * LOG_500) * np.log(theta).sum())


def recording_tensors(rec: RecordingInput, num_states: int, device: torch.device):
    """(p, q, kernel weights) for one recording."""
    p = torch.as_tensor(rec.lengths / float(rec.lengths.sum()), dtype=torch.float64, device=device)
    q = torch.full((num_states,), 1.0 / num_states, dtype=torch.float64, device=device)
    num_windows = len(rec.codes)
    half_width = transport.kernel_half_width(rec.fps, rec.window, num_windows)
    return p, q, transport.build_kernel(num_windows, half_width, device)


def responsibilities_from_log(log_t: torch.Tensor, p: torch.Tensor) -> np.ndarray:
    """R = T / p[:,None], computed in the log domain."""
    return torch.exp(log_t - torch.log(p).unsqueeze(1)).cpu().numpy()


@dataclass
class FitResult:
    params: np.ndarray  # theta [C,K] (categorical) or mu [C,d] (continuous)
    log_t_state: dict[str, torch.Tensor]  # per recording, warm-startable log T after fitting
    outer_iterations: int
    outer_status: str  # "converged" | "capped" | "invalid"
    objective_trace: list = field(default_factory=list)
    last_rel_changes: list = field(default_factory=list)
    stale_prototypes: list = field(default_factory=list)  # continuous only


def fit_alternating(recordings: list[RecordingInput], params0: np.ndarray, num_states: int,
                    cost_fn, update_fn, prior_fn, coeffs: transport.Coeffs, inner_steps: int,
                    outer_cap: int, outer_patience: int, outer_reltol: float,
                    device: torch.device, deadline: float | None = None) -> FitResult:
    """Inexact alternating minimization (instructions §5): each outer
    iteration solves every fitting recording with `inner_steps` warm-started
    accepted mirror steps, then applies one exact pooled parameter update.

    cost_fn(params, rec) -> [L,C] cost; update_fn(recordings, R_by_name,
    params) -> (new_params, stale_list); prior_fn(params) -> float."""
    params = params0.copy()
    cache: dict[str, tuple] = {}
    log_t_state: dict[str, torch.Tensor] = {}
    prev_total = None
    good_outer = 0
    trace, rels, stale = [], [], []
    status = "capped"
    outer_used = 0
    for outer in range(outer_cap):
        outer_used = outer + 1
        responsibilities: dict[str, np.ndarray] = {}
        for rec in recordings:
            check_deadline(deadline)
            if rec.name not in cache:
                cache[rec.name] = recording_tensors(rec, num_states, device)
            p, q, weights = cache[rec.name]
            cost = torch.as_tensor(cost_fn(params, rec), dtype=torch.float64, device=device)
            if rec.name not in log_t_state:
                log_t_state[rec.name] = transport.cold_start_log(p, q)
            result = transport.solve_inner(log_t_state[rec.name], cost, weights, p, q, coeffs, inner_steps)
            if result.status == "invalid":
                return FitResult(params, log_t_state, outer_used, "invalid", trace, rels, stale)
            log_t_state[rec.name] = result.log_t
            responsibilities[rec.name] = responsibilities_from_log(result.log_t, p)
        params, stale = update_fn(recordings, responsibilities, params)
        # v1.6.1 fix: the outer stopping trace must be the declared objective
        # of the *returned* full iterate, sum_r N_r F(T_new; params_new) +
        # prior(params_new). Summing each recording's pre-update inner-solve
        # objective (evaluated at the OLD params) and only adding the prior
        # at the new params mixes two different parameter states and can
        # only ever be too low, so it can trigger the outer-convergence
        # check on iterates that have not actually converged. This
        # recomputes the unary/structural/marginal/entropy terms at the
        # already-solved couplings (log_t_state) under the just-updated
        # params -- no re-solving, one extra objective evaluation per
        # recording per outer iteration.
        total_obj = 0.0
        for rec in recordings:
            p, q, weights = cache[rec.name]
            cost = torch.as_tensor(cost_fn(params, rec), dtype=torch.float64, device=device)
            total_obj += float(transport.objective_log(log_t_state[rec.name], cost, weights, coeffs.a,
                                                        coeffs.beta, coeffs.lam, coeffs.eps, q)) * float(rec.lengths.sum())
        total_obj += prior_fn(params)
        trace.append(total_obj)
        rel = (abs(total_obj - prev_total) / max(1.0, abs(prev_total))
               if prev_total is not None else float("inf"))
        rels.append(rel)
        prev_total = total_obj
        good_outer = good_outer + 1 if rel <= outer_reltol else 0
        if good_outer >= outer_patience:
            status = "converged"
            break
    return FitResult(params, log_t_state, outer_used, status, trace, rels, stale)


def fit_categorical(recordings: list[RecordingInput], theta0: np.ndarray, num_states: int,
                    num_codes: int, eta: float, coeffs: transport.Coeffs, inner_steps: int,
                    outer_cap: int, outer_patience: int, outer_reltol: float,
                    device: torch.device, deadline: float | None = None) -> FitResult:
    def cost_fn(theta, rec):
        return emission_cost(theta, rec.codes)

    def update_fn(recs, responsibilities, _theta):
        new_theta, _ = emission_update(recs, responsibilities, num_states, num_codes, eta)
        return new_theta, []

    def prior_fn(theta):
        return prior_penalty(theta, coeffs.a, eta, num_codes)

    return fit_alternating(recordings, theta0, num_states, cost_fn, update_fn, prior_fn, coeffs,
                           inner_steps, outer_cap, outer_patience, outer_reltol, device, deadline)


@dataclass
class FinalResult:
    states: dict[str, np.ndarray]  # per recording, raw argmax state id per window
    status: dict[str, str]
    residuals: dict[str, float | None]
    steps: dict[str, int]
    rel_changes: dict[str, float | None]


def infer_final(params: np.ndarray, recordings: list[RecordingInput], num_states: int, cost_fn,
                warm_start: dict[str, torch.Tensor] | None, coeffs: transport.Coeffs,
                device: torch.device, max_steps: int, grad_tol: float, rel_tol: float,
                patience: int, backtrack_max: int, deadline: float | None = None) -> FinalResult:
    """Final re-solve at frozen parameters under one setting. Used both for
    the fitting population and for frozen held-out inference."""
    out = FinalResult({}, {}, {}, {}, {})
    for rec in recordings:
        check_deadline(deadline)
        p, q, weights = recording_tensors(rec, num_states, device)
        cost = torch.as_tensor(cost_fn(params, rec), dtype=torch.float64, device=device)
        log_t0 = (warm_start[rec.name] if warm_start and rec.name in warm_start
                  else transport.cold_start_log(p, q))
        result = transport.solve_final(log_t0, cost, weights, p, q, coeffs, max_steps,
                                       grad_tol, rel_tol, patience, backtrack_max)
        out.status[rec.name] = result.status
        out.residuals[rec.name] = result.residual
        out.steps[rec.name] = result.accepted_steps
        out.rel_changes[rec.name] = result.last_rel_change
        if result.status != "invalid":
            # argmax of R = argmax of log T per row (p[t] is a row constant)
            out.states[rec.name] = result.log_t.argmax(dim=1).cpu().numpy().astype(np.int32)
    return out


def categorical_cost_fn(theta: np.ndarray, rec: RecordingInput) -> np.ndarray:
    return emission_cost(theta, rec.codes)
