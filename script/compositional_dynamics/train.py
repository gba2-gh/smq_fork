"""Alternating pseudo-target fitting on the FITTING recordings only.

Nothing in this module receives evaluation recordings or labels: the functions take a fitting `Population` only."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from script.action_transport import transport
from script.action_transport import categorical, continuous
from script.action_transport.budget import check_deadline
from script.compositional_dynamics import config as C
from script.compositional_dynamics import data as Dm
from script.compositional_dynamics.model import HistoryModel, build_model

CPU = torch.device("cpu")


class InvalidSolve(RuntimeError):
    pass


def coeffs() -> transport.Coeffs:
    return transport.Coeffs(**C.COEFFS)


# --- occupancy -------------------------------------------------------------------------------------------------

def theta_from(pop: Dm.Population, R: np.ndarray, num_codes: int = C.NUM_CODES) -> np.ndarray:
    """theta[a,k] ∝ pseudocount + sum_t (length[t]/window) R[t,a] 1[c[t]=k]; normalized over k (float64)."""
    w = pop.lengths.astype(np.float64) / C.WINDOW
    counts = np.stack([np.bincount(pop.codes, weights=w * R[:, a].astype(np.float64), minlength=num_codes)
                       for a in range(R.shape[1])])
    th = C.PSEUDOCOUNT + counts
    return th / th.sum(axis=1, keepdims=True)


# --- raw costs -------------------------------------------------------------------------------------------------

@torch.no_grad()
def components(model: HistoryModel | None, theta: np.ndarray, pop: Dm.Population, device, chunk: int = 4096,
               hist: np.ndarray | None = None, hmask: np.ndarray | None = None, want_repr: bool = False):
    """(occ_nll [N,C], cond_nll [N,C] or None, h [N,hidden] or None), float32. hist/hmask override the population's
    histories (order-sensitivity decode); the target code is always the window's own code."""
    occ = (-np.log(theta[:, pop.codes]).T).astype(np.float32)
    if model is None:
        return occ, None, None
    hist = pop.hist if hist is None else hist
    hmask = pop.hmask if hmask is None else hmask
    model.eval()
    cond = np.empty_like(occ)
    reps = np.empty((pop.n, model.hidden), dtype=np.float32) if want_repr else None
    for a in range(0, pop.n, chunk):
        sl = slice(a, a + chunk)
        h_t = torch.as_tensor(hist[sl], device=device)
        m_t = torch.as_tensor(hmask[sl], device=device)
        h = model.history_repr(h_t, m_t)
        lp = model.logp_from_repr(h)
        tgt = torch.as_tensor(pop.codes[sl], device=device)
        cond[sl] = (-torch.gather(lp, 2, tgt[:, None, None].expand(-1, lp.shape[1], 1)).squeeze(2)).cpu().numpy()
        if want_repr:
            reps[sl] = h.cpu().numpy()
    return occ, cond, reps


def raw_cost(occ: np.ndarray, cond: np.ndarray | None) -> np.ndarray:
    """D_raw = -0.5 log theta - 0.5 log p(c|history); occupancy arm: -log theta."""
    return occ if cond is None else C.ALPHA * occ + C.ALPHA * cond


def calibration_ids(pop: Dm.Population, cap: int) -> np.ndarray:
    complete = np.flatnonzero(pop.lengths == C.WINDOW)
    rng = np.random.default_rng(C.CALIB_SEED)
    return np.sort(rng.choice(complete, min(cap, len(complete)), replace=False))


def calibrate_scale(d_raw: np.ndarray, ids: np.ndarray) -> float:
    rng_ = d_raw[ids].max(axis=1) - d_raw[ids].min(axis=1)
    pos = rng_[rng_ > 0]
    s = float(np.median(pos)) if len(pos) else float("nan")
    if not np.isfinite(s) or s < C.MIN_SCALE:
        raise InvalidSolve(f"cost-scale calibration undefined (s={s})")
    return s


def scaled_cost(d_raw: np.ndarray, s: float) -> np.ndarray:
    return ((d_raw - d_raw.min(axis=1, keepdims=True)) / s).astype(np.float64)


def cost_stats(d_raw: np.ndarray, s: float) -> dict:
    rr = d_raw.max(axis=1) - d_raw.min(axis=1)
    q = [5, 25, 50, 75, 95]
    return dict(raw_range_q=np.percentile(rr, q).tolist(), scaled_range_q=np.percentile(rr / s, q).tolist(),
                near_equal_share=float(np.mean(rr < 1e-3 * s)), scaled_range_gt5_share=float(np.mean(rr / s > 5)))


# --- plans -----------------------------------------------------------------------------------------------------

class Plans:
    """Warm-startable transport plans for a population (CPU float64)."""

    def __init__(self, pop: Dm.Population):
        self.pop, self.recs = pop, Dm.recording_inputs(pop)
        self.tensors = {r.name: categorical.recording_tensors(r, C.NUM_STATES, CPU) for r in self.recs}
        self.log_t: dict = {}

    def set_from(self, log_t: dict) -> None:
        self.log_t = dict(log_t)

    def responsibilities(self) -> np.ndarray:
        out = np.empty((self.pop.n, C.NUM_STATES), dtype=np.float32)
        for r, rec in enumerate(self.recs):
            out[self.pop.slice_of(r)] = categorical.responsibilities_from_log(self.log_t[rec.name], self.tensors[rec.name][0])
        return out

    def refresh(self, cost: np.ndarray, inner_steps: int, deadline: float | None = None) -> dict:
        """inner_steps accepted warm-started steps per recording; an invalid solve stops the cell."""
        cf, steps, objective = coeffs(), 0, 0.0
        for r, rec in enumerate(self.recs):
            check_deadline(deadline)
            p, q, w = self.tensors[rec.name]
            if rec.name not in self.log_t:
                self.log_t[rec.name] = transport.cold_start_log(p, q)
            res = transport.solve_inner(self.log_t[rec.name], torch.as_tensor(cost[self.pop.slice_of(r)], dtype=torch.float64),
                                        w, p, q, cf, inner_steps)
            if res.status == "invalid":
                raise InvalidSolve(f"inner solve invalid for {rec.name}")
            self.log_t[rec.name] = res.log_t
            steps += res.accepted_steps
            objective += res.trace[-1] * float(rec.lengths.sum()) if res.trace else 0.0
        return dict(accepted_steps=steps, objective=objective)


def plan_stats(R: np.ndarray, lengths: np.ndarray) -> dict:
    w = lengths / lengths.sum()
    mass = (R * w[:, None]).sum(axis=0)
    ent = float(-(w * (R * np.log(np.clip(R, 1e-12, None))).sum(axis=1)).sum())
    return dict(state_mass=mass.tolist(), posterior_entropy=ent)


# --- matched B0 -----------------------------------------------------------------------------------------------

def b0_inputs(data: Dm.CellData, pop: Dm.Population, fit_continuous_by_name: dict | None = None):
    """ContinuousRecordingInput list for the T2 'centre' arm: unit-normalized (lookup[code] - mean)."""
    recs = []
    for r, rec in enumerate(Dm.recording_inputs(pop)):
        raw = data.lookup_raw[rec.codes].astype(np.float64) - data.mean[None, :]
        z, zero = continuous.l2_normalize(raw)
        recs.append(continuous.ContinuousRecordingInput(rec.name, rec.codes, rec.lengths, rec.fps, rec.window, z, zero))
    return recs


def b0_grouping(data: Dm.CellData, names: list) -> np.ndarray:
    """The historical initializer (frame-weighted per-code PCA centres -> weighted KMeans(C)) restricted to `names`."""
    from script.quantized_input_comparison import fit as fitmod
    codes, lens, pca = {}, {}, {}
    for n in names:
        r = data.loader.recording(C.FAMILY, n)
        codes[n], lens[n], pca[n] = r["codes"], r["lengths"], r["continuous"]
    grouping, _, _ = fitmod.initial_grouping(codes, lens, pca, C.NUM_CODES, C.NUM_STATES, C.FPS, C.WINDOW, data.seed)
    return grouping


@dataclass
class B0Fit:
    mu: np.ndarray
    grouping: np.ndarray
    plans: Plans
    outer_iterations: int
    outer_status: str
    seconds: float


def fit_b0(data: Dm.CellData, scale: C.Scale, deadline: float | None = None) -> B0Fit:
    t0 = time.perf_counter()
    grouping = b0_grouping(data, data.fit_names)
    recs = b0_inputs(data, data.fit)
    mu0, _ = continuous.build_mu0(grouping, recs, C.NUM_STATES)
    fit = continuous.fit_continuous(recs, mu0, C.NUM_STATES, coeffs(), C.B0_INNER_STEPS, scale.b0_outer_cap,
                                    C.B0_OUTER_PATIENCE, C.B0_OUTER_RELTOL, CPU, deadline)
    if fit.outer_status == "invalid":
        raise InvalidSolve(f"B0 fit invalid (iterations={fit.outer_iterations})")
    plans = Plans(data.fit)
    plans.set_from(fit.log_t_state)
    return B0Fit(fit.params, grouping, plans, fit.outer_iterations, fit.outer_status, time.perf_counter() - t0)


def b0_costs(data: Dm.CellData, pop: Dm.Population, mu: np.ndarray) -> np.ndarray:
    out = np.empty((pop.n, C.NUM_STATES))
    for r, rec in enumerate(b0_inputs(data, pop)):
        out[pop.slice_of(r)] = continuous.continuous_cost_fn(mu, rec)
    return out


# --- neural / occupancy fitting ---------------------------------------------------------------------------------

@dataclass
class Fitted:
    arm: str
    theta: np.ndarray
    model: HistoryModel | None
    scale_s: float
    calib_ids: np.ndarray
    plans: Plans
    refreshes: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    losses: list = field(default_factory=list)
    seconds: float = 0.0
    updates: int = 0


def sample_targets(pop: Dm.Population, n_updates: int, rng: np.random.Generator) -> np.ndarray:
    """(n_updates, batch) window ids drawn with replacement proportional to real frame length (fitting windows only)."""
    p = pop.lengths / pop.lengths.sum()
    return rng.choice(pop.n, size=(n_updates, C.BATCH), p=p)


def weighted_loss(R_batch: torch.Tensor, nll: torch.Tensor) -> torch.Tensor:
    """0.5 x mean over the batch of sum_a R[b,a] * nll[b,a]; R is a detached pseudo-target."""
    return C.ALPHA * (R_batch.detach() * nll).sum(dim=1).mean()


def train_block(model, opt, pop: Dm.Population, R: np.ndarray, draws: np.ndarray, device, record: list) -> None:
    """Detached-responsibility-weighted conditional NLL, with its 0.5 factor, averaged over the drawn batch."""
    model.train()
    R_t = torch.as_tensor(R, dtype=torch.float32)
    for ids in draws:
        ids_t = torch.as_tensor(ids)
        hist = torch.as_tensor(pop.hist[ids], device=device)
        hmask = torch.as_tensor(pop.hmask[ids], device=device)
        tgt = torch.as_tensor(pop.codes[ids], device=device)
        nll = model.target_nll(hist, hmask, tgt)
        loss = weighted_loss(R_t[ids_t].to(device), nll)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), C.CLIP)
        opt.step()
        record.append((float(loss), float(gn)))


def fit_arm(data: Dm.CellData, arm: str, scale: C.Scale, R0: np.ndarray, log_t0: dict, device,
            deadline: float | None = None) -> Fitted:
    """arm in {m1_ordered, m1_bag, occupancy}. R0 / log_t0: the matched B0's fitting plans (same for every arm)."""
    t0 = time.perf_counter()
    pop = data.fit
    neural = arm in C.NEURAL_ARMS
    theta = theta_from(pop, R0)
    plans = Plans(pop)
    plans.set_from(log_t0)
    model = opt = None
    rng = np.random.default_rng([data.seed, 7])
    losses: list = []
    if neural:
        model = build_model(data.lookup_fixed, C.NUM_STATES, C.history_length(), data.seed, arm).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
        train_block(model, opt, pop, R0, sample_targets(pop, scale.block_updates, rng), device, losses)
    train_s = time.perf_counter() - t0
    ids = calibration_ids(pop, scale.calib_cap)
    tc = time.perf_counter()
    occ, cond, _ = components(model, theta, pop, device)
    comp_s = time.perf_counter() - tc
    s = calibrate_scale(raw_cost(occ, cond), ids)
    out = Fitted(arm, theta, model, s, ids, plans)
    refresh_s = steps_total = 0.0
    for block in range(1, scale.n_blocks):
        check_deadline(deadline)
        tc = time.perf_counter()
        occ, cond, _ = components(model, theta, pop, device)
        comp_s += time.perf_counter() - tc
        d_raw = raw_cost(occ, cond)
        tr = time.perf_counter()
        info = plans.refresh(scaled_cost(d_raw, s), scale.inner_steps, deadline)
        refresh_s += time.perf_counter() - tr
        steps_total += info["accepted_steps"]
        R = plans.responsibilities()
        theta = theta_from(pop, R)
        out.refreshes.append(dict(block=block, updates_before=block * scale.block_updates, **info,
                                  **cost_stats(d_raw, s), **plan_stats(R, pop.lengths.astype(np.float64)),
                                  mean_train_loss=float(np.mean([l for l, _ in losses[-scale.block_updates:]])) if neural else None))
        if neural:
            tt = time.perf_counter()
            train_block(model, opt, pop, R, sample_targets(pop, scale.block_updates, rng), device, losses)
            train_s += time.perf_counter() - tt
    out.timings = dict(train_s=train_s, updates=len(losses), comp_s=comp_s, comp_calls=scale.n_blocks,
                       comp_windows_per_call=pop.n, refresh_s=refresh_s, refresh_steps=steps_total)
    out.theta, out.losses, out.updates = theta, losses, len(losses)
    out.seconds = time.perf_counter() - t0
    return out
