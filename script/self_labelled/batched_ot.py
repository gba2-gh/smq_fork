"""Batched temporal OT for pseudo-labelling a batch of recordings (docs/plans/SELF_LABELLED_EMBEDDING_PLAN.md, Section 4.1).

The objective is the existing one (script/action_transport/transport.py: unary, structural, relaxed column marginal, entropy)
applied to the concatenation of a batch's recordings:
  - row marginal   p_i = real frames of window i / total frames in the BATCH;
  - structural term block-diagonal (the temporal kernel never crosses a recording boundary), with each recording's kernel
    amplitude divided by that recording's frame share pi_r, so that with the column marginal switched off the problem decouples into
    exactly the per-recording problems;
  - column marginal relaxed toward q = 1/C over the WHOLE batch (equipartition across recordings).
`per_recording=True` instead applies the column marginal to each recording separately (the standard ASOT setting), used by L_noeq.
A batch of one recording reproduces transport.solve_inner exactly."""
from __future__ import annotations

import torch

from script.action_transport import transport
from script.action_transport.transport import Coeffs, MARGINAL_FLOOR


class Batch:
    """Everything the solver needs for one batch: block-diagonal kernel, row marginals, recording ids."""

    def __init__(self, lengths: list, frames: list, fps: int, window: int, num_states: int, device, dtype=torch.float64):
        self.device, self.dtype, self.C = device, dtype, num_states
        self.sizes = [len(f) for f in frames]
        self.n = sum(self.sizes)
        fr = torch.as_tensor([x for f in frames for x in f], dtype=dtype, device=device)
        total = fr.sum()
        self.p = fr / total
        self.rec = torch.repeat_interleave(torch.arange(len(self.sizes), device=device), torch.as_tensor(self.sizes, device=device))
        self.q = torch.full((num_states,), 1.0 / num_states, dtype=dtype, device=device)
        V = torch.zeros(self.n, self.n, dtype=dtype, device=device)
        off = 0
        for L, f in zip(self.sizes, frames):
            share = float(sum(f)) / float(total)
            w = transport.build_kernel(L, transport.kernel_half_width(fps, window, L), device, dtype)
            if w is not None:
                half = w.shape[-1] // 2
                block = torch.zeros(L, L, dtype=dtype, device=device)
                for k in range(-half, half + 1):
                    if k != 0:
                        idx = torch.arange(max(0, -k), min(L, L - k), device=device)
                        block[idx, idx + k] = w[0, 0, half + k]
                V[off:off + L, off:off + L] = block / share
            off += L
        self.V = V
        self.onehot = torch.nn.functional.one_hot(self.rec, len(self.sizes)).to(dtype)        # [N, R]

    def marginal(self, t: torch.Tensor, per_recording: bool):
        if per_recording:
            return self.onehot.T @ t                                                          # [R, C]
        return t.sum(dim=0, keepdim=True)                                                     # [1, C]


def _terms(log_t, cost, b: Batch, coeffs: Coeffs, per_recording: bool):
    t = torch.exp(log_t)
    complement = t.sum(dim=1, keepdim=True) - t
    vc = b.V @ complement
    mass = b.marginal(t, per_recording).clamp_min(MARGINAL_FLOOR)
    q = b.q.unsqueeze(0)
    if per_recording:
        # each recording's relaxed marginal is toward q scaled by its own mass share, as the per-recording problem has total mass pi_r
        share = (b.onehot.T @ b.p).unsqueeze(1)
        qq = share * q
    else:
        qq = q
    return t, vc, mass, qq


def objective(log_t, cost, b: Batch, coeffs: Coeffs, per_recording: bool = False):
    t, vc, mass, qq = _terms(log_t, cost, b, coeffs, per_recording)
    unary = coeffs.a * (cost * t).sum()
    structural = (coeffs.beta / 2.0) * (t * vc).sum() if coeffs.beta != 0.0 else t.new_zeros(())
    marginal = coeffs.lam * (mass * torch.log(mass / qq) - mass + qq).sum()
    neg_eps_h = coeffs.eps * (t * (log_t - 1.0)).sum()
    return unary + structural + marginal + neg_eps_h


def gradient(log_t, cost, b: Batch, coeffs: Coeffs, per_recording: bool = False):
    t, vc, mass, qq = _terms(log_t, cost, b, coeffs, per_recording)
    grad = coeffs.a * cost + coeffs.beta * vc + coeffs.eps * log_t
    colterm = torch.log(mass / qq) + 1.0                                                      # [R or 1, C]
    return grad + coeffs.lam * (colterm[b.rec] if per_recording else colterm)


def solve_inner(log_t0, cost, b: Batch, coeffs: Coeffs, steps: int, per_recording: bool = False,
                backtrack_max: int = 30, backtrack_tol: float = 1e-12):
    """`steps` accepted backtracked entropic mirror steps (same acceptance rule as transport.mirror_step). Returns
    (log_t, accepted_steps, ok)."""
    log_p = torch.log(b.p)
    log_t = transport.cold_start_log(b.p, b.q) if log_t0 is None else log_t0
    done = 0
    for _ in range(steps):
        grad = gradient(log_t, cost, b, coeffs, per_recording)
        f_old = float(objective(log_t, cost, b, coeffs, per_recording))
        threshold = f_old + backtrack_tol * max(1.0, abs(f_old))
        step, accepted = 1.0, False
        for _a in range(backtrack_max + 1):
            cand = transport.row_project_log(log_t - step * grad, log_p)
            if torch.isfinite(cand).all():
                f_new = float(objective(cand, cost, b, coeffs, per_recording))
                if f_new == f_new and f_new <= threshold:
                    log_t, accepted = cand, True
                    break
            step *= 0.5
        if not accepted:
            return log_t, done, False
        done += 1
    return log_t, done, True


def pseudo_labels(log_t: torch.Tensor, b: Batch) -> torch.Tensor:
    """Q = T / p: soft labels whose rows sum to one."""
    return torch.exp(log_t - torch.log(b.p).unsqueeze(1))
