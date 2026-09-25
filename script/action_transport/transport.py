"""Torch float64 mirror-descent solver for the declared transport objective
(docs/plans/THREE_STAGE_MOTION_PLAN.md §3):

    F(T;theta) = a*sum(D*T)
               + beta/2*sum_{t,s,a,b} V[t,s]*T[t,a]*T[s,b]*1[a!=b]
               - epsilon*H(T)
               + lambda*genKL(m,q),   m = T.sum(axis=0)

with T's rows fixed at the real frame masses p (frames are exactly balanced;
actions are KL-relaxed toward uniform q). The gradient below matches the
objective exactly (instructions §5's corrected, independently checked
objective/gradient pair -- the pinned upstream code halves the linear term in
its logged objective scalar but not in its gradient; production here does
not import or edit that upstream file, it reimplements the same solver
family with the declared objective).

Kernel construction: per instructions §3/§4 (v1.3 correction), the temporal
support half-width b is an integer computed directly from
`min(L-1, max(1, floor(fps/W)))`, never recovered by rounding a
radius-fraction back through int(L*(b/L)). The kernel amplitude is L/b using
each recording's *true* L; padded batch length must never enter it. This
module operates per recording (no padded batching) -- see
instructions_agent §5, "Implement per-recording correctness first ...
batching optional after equivalence tests."
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

TINY = 1e-12


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
    """sum_s weights[t-s] * x[s,c], applied independently per column c.

    Mirrors the pinned reference's `mult_Cv` reshape trick (each column
    becomes its own batch element of a length-1-channel 1D signal), which is
    what makes this equivalent to the reference's fast O(L*C) structural term
    without a dense L-by-L matrix."""
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
    row_sum = t.sum(dim=1, keepdim=True)
    complement = row_sum - t
    return beta * conv_apply(weights, complement)


def gradient(t: torch.Tensor, cost: torch.Tensor, weights: torch.Tensor | None,
            a: float, beta: float, lam: float, eps: float, q: torch.Tensor) -> torch.Tensor:
    grad = a * cost + structural_grad(t, weights, beta)
    grad = grad + eps * torch.log(t + TINY)
    marginal = t.sum(dim=0)
    grad = grad + lam * (torch.log(marginal / q + TINY) + 1.0).unsqueeze(0)
    return grad


def objective(t: torch.Tensor, cost: torch.Tensor, weights: torch.Tensor | None,
             a: float, beta: float, lam: float, eps: float, q: torch.Tensor) -> torch.Tensor:
    """Independently computed from `gradient` (not derived from it), per the
    requirement to check objective/gradient consistency by finite
    differences rather than trusting one against the other."""
    unary = a * (cost * t).sum()
    if weights is not None and beta != 0.0:
        row_sum = t.sum(dim=1, keepdim=True)
        complement = row_sum - t
        conv = conv_apply(weights, complement)
        structural = (beta / 2.0) * (t * conv).sum()
    else:
        structural = t.new_zeros(())
    marginal_mass = t.sum(dim=0)
    marginal = lam * (marginal_mass * torch.log(marginal_mass / q + TINY) - marginal_mass + q).sum()
    neg_eps_h = eps * (t * (torch.log(t + TINY) - 1.0)).sum()  # = -epsilon * H(T)
    return unary + structural + marginal + neg_eps_h


def row_normalize(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    row_sum = t.sum(dim=1, keepdim=True).clamp_min(1e-300)
    return t / row_sum * p.unsqueeze(1)


@dataclass
class Coeffs:
    a: float
    beta: float
    lam: float
    eps: float


def mirror_step(t: torch.Tensor, cost: torch.Tensor, weights: torch.Tensor | None,
                coeffs: Coeffs, p: torch.Tensor, q: torch.Tensor,
                backtrack_max: int = 30, backtrack_tol: float = 1e-12):
    """One backtracked mirror-descent step. Returns
    (t_new, f_new, grad_at_t_old, step_used, backtracks_used, accepted)."""
    grad = gradient(t, cost, weights, coeffs.a, coeffs.beta, coeffs.lam, coeffs.eps, q)
    f_old = objective(t, cost, weights, coeffs.a, coeffs.beta, coeffs.lam, coeffs.eps, q)
    step = 1.0
    for attempt in range(backtrack_max + 1):
        candidate = t * torch.exp(-step * grad)
        candidate = row_normalize(candidate, p)
        candidate = torch.nan_to_num(candidate, nan=0.0, posinf=0.0, neginf=0.0)
        f_new = objective(candidate, cost, weights, coeffs.a, coeffs.beta, coeffs.lam, coeffs.eps, q)
        threshold = f_old + backtrack_tol * max(1.0, abs(float(f_old)))
        if torch.isfinite(f_new) and float(f_new) <= threshold:
            return candidate, f_new, grad, step, attempt, True
        step *= 0.5
    return t, f_old, grad, step, backtrack_max, False


def row_centered_residual(grad: torch.Tensor) -> float:
    centered = grad - grad.mean(dim=1, keepdim=True)
    return float(centered.abs().max())


@dataclass
class SolveResult:
    t: torch.Tensor
    status: str  # "converged" | "capped" | "invalid"
    trace: list = field(default_factory=list)
    residual: float | None = None
    accepted_steps: int = 0
    last_rel_change: float | None = None


def solve_inner(t_init: torch.Tensor, cost: torch.Tensor, weights: torch.Tensor | None,
                p: torch.Tensor, q: torch.Tensor, coeffs: Coeffs, max_accepted: int,
                backtrack_max: int = 30, backtrack_tol: float = 1e-12) -> SolveResult:
    """Fitting-mode inner solve: exactly `max_accepted` warm-started accepted
    steps, or fewer if backtracking fails (invalid). Each accepted step is
    nonincreasing in the declared objective by construction."""
    t = t_init
    trace = []
    for _ in range(max_accepted):
        t_new, f_new, _, _, _, accepted = mirror_step(t, cost, weights, coeffs, p, q,
                                                       backtrack_max, backtrack_tol)
        if not accepted:
            return SolveResult(t, "invalid", trace, accepted_steps=len(trace))
        trace.append(float(f_new))
        t = t_new
    return SolveResult(t, "ok", trace, accepted_steps=len(trace))


def solve_final(t_init: torch.Tensor, cost: torch.Tensor, weights: torch.Tensor | None,
                p: torch.Tensor, q: torch.Tensor, coeffs: Coeffs, max_steps: int = 500,
                grad_tol: float = 1e-5, rel_tol: float = 1e-7, patience: int = 5,
                backtrack_max: int = 30, backtrack_tol: float = 1e-12) -> SolveResult:
    """Final-inference solve: run to the declared stationarity criterion
    (row-centered gradient max norm <=grad_tol AND relative objective change
    <=rel_tol, both held for `patience` consecutive accepted steps), capped
    at max_steps."""
    t = t_init
    prev_obj = None
    good_streak = 0
    trace = []
    last_residual = None
    last_rel = None
    for step_index in range(max_steps):
        t_new, f_new, grad, _, _, accepted = mirror_step(t, cost, weights, coeffs, p, q,
                                                          backtrack_max, backtrack_tol)
        if not accepted:
            return SolveResult(t, "invalid", trace, last_residual, step_index, last_rel)
        f_new_value = float(f_new)
        trace.append(f_new_value)
        residual = row_centered_residual(grad)
        rel = (abs(f_new_value - prev_obj) / max(1.0, abs(prev_obj))
               if prev_obj is not None else float("inf"))
        last_residual, last_rel = residual, rel
        t = t_new
        prev_obj = f_new_value
        good_streak = good_streak + 1 if (residual <= grad_tol and rel <= rel_tol) else 0
        if good_streak >= patience:
            return SolveResult(t, "converged", trace, residual, step_index + 1, rel)
    return SolveResult(t, "capped", trace, last_residual, max_steps, last_rel)


def kernel_half_width(fps: int, window: int, num_windows: int) -> int:
    if num_windows <= 1:
        return 0
    return min(num_windows - 1, max(1, fps // window))
