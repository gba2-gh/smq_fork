"""Torch float64 log-domain mirror-descent solver for the declared transport
objective (docs/plans/THREE_STAGE_MOTION_PLAN.md §3):

    F(T;theta) = a*sum(D*T)
               + beta/2*sum_{t,s,a,b} V[t,s]*T[t,a]*T[s,b]*1[a!=b]
               - epsilon*H(T)
               + lambda*genKL(m,q),   m = T.sum(axis=0)

with T's rows fixed at the real frame masses p (frames exactly balanced;
actions KL-relaxed toward uniform q).

v1.3.1 implementation fix (no change to the declared objective,
coefficients, or stopping rules): the solver state is log T, and the entropy
term uses the exact log T. v1.3 computed log(T + 1e-12). Entries below 1e-12
are legitimate at the optimum (a/eps = 10 and categorical costs up to ~2.5
give ratios near exp(-25)), and for them the floored log reported a false
gradient. The row-centered residual therefore never fell below its
tolerance, so every final solve was labelled `capped` after 500 steps even
when the objective had stopped changing to machine precision. The floored
gradient also pushed those negligible-mass entries toward 0 instead of their
true values; argmax assignments were unaffected, but the fixed point was not
the declared one. Instructions §5 already asked for "stable log-domain
mirror descent", which this now is.

Kernel construction (v1.3): the temporal half-width b is an integer computed
directly, never recovered as int(L*(b/L)); the amplitude is L/b using each
recording's true L. This module works per recording (no padded batching).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

MARGINAL_FLOOR = 1e-300  # only guards a column whose mass underflows to exactly 0


def build_kernel(num_windows: int, half_width: int, device: torch.device,
                 dtype: torch.dtype = torch.float64) -> torch.Tensor | None:
    """V restricted to a symmetric band: V[t,s] = L/b for 0<|t-s|<=b, else 0.
    Returns None when there is no temporal structure (L<=1 or b==0)."""
    if num_windows <= 1 or half_width <= 0:
        return None
    width = 2 * half_width + 1
    weights = torch.full((width,), float(num_windows) / float(half_width), dtype=dtype, device=device)
    weights[half_width] = 0.0
    return weights.view(1, 1, width)


def conv_apply(weights: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
    """sum_s weights[t-s] * x[s,c], independently per column c (the pinned
    reference's `mult_Cv` reshape trick; O(L*C), no dense L-by-L matrix)."""
    if weights is None:
        return torch.zeros_like(x)
    num_windows, num_states = x.shape
    pad = weights.shape[-1] // 2
    reshaped = x.transpose(0, 1).reshape(num_states, 1, num_windows)
    out = F.conv1d(reshaped, weights, padding=pad)
    return out.reshape(num_states, num_windows).transpose(0, 1)


def structural_grad(t: torch.Tensor, weights: torch.Tensor | None, beta: float) -> torch.Tensor:
    if weights is None or beta == 0.0:
        return torch.zeros_like(t)
    complement = t.sum(dim=1, keepdim=True) - t
    return beta * conv_apply(weights, complement)


@dataclass
class Coeffs:
    a: float
    beta: float
    lam: float
    eps: float


# --- log-domain core -----------------------------------------------------------

def gradient_log(log_t: torch.Tensor, cost: torch.Tensor, weights: torch.Tensor | None,
                 a: float, beta: float, lam: float, eps: float, q: torch.Tensor) -> torch.Tensor:
    """dF/dT evaluated at T = exp(log_t), with the exact entropy term."""
    t = torch.exp(log_t)
    marginal = t.sum(dim=0).clamp_min(MARGINAL_FLOOR)
    grad = a * cost + structural_grad(t, weights, beta) + eps * log_t
    return grad + lam * (torch.log(marginal / q) + 1.0).unsqueeze(0)


def objective_log(log_t: torch.Tensor, cost: torch.Tensor, weights: torch.Tensor | None,
                  a: float, beta: float, lam: float, eps: float, q: torch.Tensor) -> torch.Tensor:
    """The declared objective, computed independently of `gradient_log` so
    finite differences can check one against the other."""
    t = torch.exp(log_t)
    unary = a * (cost * t).sum()
    if weights is not None and beta != 0.0:
        complement = t.sum(dim=1, keepdim=True) - t
        structural = (beta / 2.0) * (t * conv_apply(weights, complement)).sum()
    else:
        structural = t.new_zeros(())
    mass = t.sum(dim=0).clamp_min(MARGINAL_FLOOR)
    marginal = lam * (mass * torch.log(mass / q) - mass + q).sum()
    neg_eps_h = eps * (t * (log_t - 1.0)).sum()  # = -epsilon * H(T), exact
    return unary + structural + marginal + neg_eps_h


# Probability-domain wrappers (tests and diagnostics; require T > 0).

def gradient(t, cost, weights, a, beta, lam, eps, q):
    return gradient_log(torch.log(t), cost, weights, a, beta, lam, eps, q)


def objective(t, cost, weights, a, beta, lam, eps, q):
    return objective_log(torch.log(t), cost, weights, a, beta, lam, eps, q)


def cold_start_log(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """log(p[:,None] * q[None,:]), the declared first-solve initialization."""
    return torch.log(p).unsqueeze(1) + torch.log(q).unsqueeze(0)


def row_project_log(log_t: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Rescale each row so exp(row) sums to p[t], in the log domain."""
    return log_t - torch.logsumexp(log_t, dim=1, keepdim=True) + log_p.unsqueeze(1)


def row_centered_residual(grad: torch.Tensor) -> float:
    return float((grad - grad.mean(dim=1, keepdim=True)).abs().max())


def mirror_step(log_t: torch.Tensor, cost: torch.Tensor, weights: torch.Tensor | None,
                coeffs: Coeffs, log_p: torch.Tensor, q: torch.Tensor,
                backtrack_max: int = 30, backtrack_tol: float = 1e-12):
    """One backtracked entropic mirror-descent step. Returns
    (log_t_new, f_new, grad_at_old, step_used, backtracks_used, accepted)."""
    args = (cost, weights, coeffs.a, coeffs.beta, coeffs.lam, coeffs.eps, q)
    grad = gradient_log(log_t, *args)
    f_old = objective_log(log_t, *args)
    threshold = float(f_old) + backtrack_tol * max(1.0, abs(float(f_old)))
    step = 1.0
    for attempt in range(backtrack_max + 1):
        candidate = row_project_log(log_t - step * grad, log_p)
        if torch.isfinite(candidate).all():
            f_new = objective_log(candidate, *args)
            if torch.isfinite(f_new) and float(f_new) <= threshold:
                return candidate, f_new, grad, step, attempt, True
        step *= 0.5
    return log_t, f_old, grad, step, backtrack_max, False


@dataclass
class SolveResult:
    log_t: torch.Tensor
    status: str  # "ok" (inner) | "converged" | "capped" | "invalid"
    trace: list = field(default_factory=list)
    residual: float | None = None
    accepted_steps: int = 0
    last_rel_change: float | None = None

    @property
    def t(self) -> torch.Tensor:
        return torch.exp(self.log_t)


def solve_inner(log_t_init: torch.Tensor, cost: torch.Tensor, weights: torch.Tensor | None,
                p: torch.Tensor, q: torch.Tensor, coeffs: Coeffs, max_accepted: int,
                backtrack_max: int = 30, backtrack_tol: float = 1e-12) -> SolveResult:
    """Fitting-mode inner solve: `max_accepted` warm-started accepted steps,
    or fewer if backtracking fails (invalid). Accepted steps never increase
    the declared objective beyond the backtracking tolerance."""
    log_p = torch.log(p)
    log_t = log_t_init
    trace = []
    for _ in range(max_accepted):
        log_new, f_new, _, _, _, accepted = mirror_step(log_t, cost, weights, coeffs, log_p, q,
                                                        backtrack_max, backtrack_tol)
        if not accepted:
            return SolveResult(log_t, "invalid", trace, accepted_steps=len(trace))
        trace.append(float(f_new))
        log_t = log_new
    return SolveResult(log_t, "ok", trace, accepted_steps=len(trace))


def solve_final(log_t_init: torch.Tensor, cost: torch.Tensor, weights: torch.Tensor | None,
                p: torch.Tensor, q: torch.Tensor, coeffs: Coeffs, max_steps: int = 500,
                grad_tol: float = 1e-5, rel_tol: float = 1e-7, patience: int = 5,
                backtrack_max: int = 30, backtrack_tol: float = 1e-12) -> SolveResult:
    """Final-inference solve: stop when the row-centered gradient max norm is
    <= grad_tol and the relative objective change is <= rel_tol for
    `patience` consecutive accepted steps; cap at max_steps."""
    log_p = torch.log(p)
    log_t = log_t_init
    prev_obj = None
    good_streak = 0
    trace = []
    last_residual = last_rel = None
    for step_index in range(max_steps):
        log_new, f_new, grad, _, _, accepted = mirror_step(log_t, cost, weights, coeffs, log_p, q,
                                                           backtrack_max, backtrack_tol)
        if not accepted:
            return SolveResult(log_t, "invalid", trace, last_residual, step_index, last_rel)
        f_value = float(f_new)
        trace.append(f_value)
        # The residual is measured at the pre-step iterate (where `grad` was
        # evaluated); the objective change is between the two iterates.
        last_residual = row_centered_residual(grad)
        last_rel = (abs(f_value - prev_obj) / max(1.0, abs(prev_obj))
                    if prev_obj is not None else float("inf"))
        log_t, prev_obj = log_new, f_value
        good_streak = good_streak + 1 if (last_residual <= grad_tol and last_rel <= rel_tol) else 0
        if good_streak >= patience:
            return SolveResult(log_t, "converged", trace, last_residual, step_index + 1, last_rel)
    return SolveResult(log_t, "capped", trace, last_residual, max_steps, last_rel)


def kernel_half_width(fps: int, window: int, num_windows: int) -> int:
    if num_windows <= 1:
        return 0
    return min(num_windows - 1, max(1, fps // window))
