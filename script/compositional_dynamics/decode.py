"""Final decoding of a frozen model: cold start per recording, continuation ladder (same semantics as
fsq_reconstruction.evaluate.contextual_final_ladder), on precomputed per-window cost matrices."""
from __future__ import annotations

import numpy as np
import torch

from script.action_transport import categorical, transport
from script.action_transport import config as at_config
from script.action_transport.budget import check_deadline
from script.compositional_dynamics import config as C
from script.compositional_dynamics import data as Dm
from script.compositional_dynamics.train import CPU, InvalidSolve, coeffs


def decode_ladder(cost: np.ndarray, pop: Dm.Population, rungs: tuple, deadline: float | None = None,
                  multiplier: float = 1.0) -> dict:
    """cost [N, C] per-window costs of the population. Returns per-rung window states plus the final solver record.

    Each rung continues the previous rung's iterate; recordings that converged or became invalid stop, and a rung's
    budget is the remaining steps to that endpoint. 'capped' is never reported as converged."""
    recs = Dm.recording_inputs(pop)
    cf = coeffs()
    log_t, steps, status, resid, final_obj = {}, {}, {}, {}, {}
    states_by_rung, status_counts = {}, {}
    for rung in rungs:
        for r, rec in enumerate(recs):
            check_deadline(deadline)
            if status.get(rec.name) in ("converged", "invalid"):
                continue
            remaining = rung - steps.get(rec.name, 0)
            if remaining <= 0:
                continue
            p, q, w = categorical.recording_tensors(rec, C.NUM_STATES, CPU)
            c = torch.as_tensor(cost[pop.slice_of(r)] * multiplier, dtype=torch.float64)
            init = log_t.get(rec.name, transport.cold_start_log(p, q))
            res = transport.solve_final(init, c, w, p, q, cf, remaining, at_config.FINAL_GRAD_TOL, at_config.FINAL_OBJ_RELTOL,
                                        at_config.FINAL_PATIENCE, at_config.BACKTRACK_MAX)
            log_t[rec.name] = res.log_t
            steps[rec.name] = steps.get(rec.name, 0) + res.accepted_steps
            status[rec.name], resid[rec.name] = res.status, res.residual
            final_obj[rec.name] = res.trace[-1] if res.trace else None
        invalid = [n for n, s in status.items() if s == "invalid"]
        if invalid:
            raise InvalidSolve(f"final solve invalid at rung {rung} for {len(invalid)} recording(s), e.g. {invalid[:3]}")
        states_by_rung[rung] = {n: t.argmax(dim=1).numpy().astype(np.int32) for n, t in log_t.items()}
        counts: dict = {}
        for s in status.values():
            counts[s] = counts.get(s, 0) + 1
        status_counts[rung] = counts
    # soft plans R = T / p of the final iterate (rows sum to one), under the cost passed in
    resp = {}
    for r, rec in enumerate(recs):
        p = torch.as_tensor(rec.lengths / float(rec.lengths.sum()), dtype=torch.float64)
        resp[rec.name] = torch.exp(log_t[rec.name] - torch.log(p).unsqueeze(1)).numpy().astype(np.float32)
    return dict(states=states_by_rung, status_counts=status_counts, status=status, steps=steps, residual=resid,
                objective=final_obj, rungs=list(rungs), responsibilities=resp)
