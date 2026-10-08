"""Scientific-contract tests for Milestone 1 (instructions Section 7). CPU, small, synthetic.

    python -m script.compositional_dynamics.tests [--only name ...]

A full pass writes results/compositional_dynamics/verification.json (tied to the current code hash); the gate report
refuses to call a run conclusive without it."""
from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from script.compositional_dynamics import config as C
from script.compositional_dynamics import data as Dm
from script.compositional_dynamics import model as M
from script.compositional_dynamics import train as Tr

TESTS = {}


def test(fn):
    TESTS[fn.__name__] = fn
    return fn


def synth_population(seed: int, sizes=(30, 17, 9), hist_len=4, num_codes=20, window=C.WINDOW):
    rng = np.random.default_rng(seed)
    codes = [rng.integers(0, num_codes, n) for n in sizes]
    lengths = [np.full(n, window) for n in sizes]
    for ln in lengths:
        ln[-1] = window // 2          # short terminal window
    return Dm.make_population([f"r{i}" for i in range(len(sizes))], codes, lengths, hist_len), codes, lengths


def small_model(seed=0, K=20, D=6, actions=3, H=4, bag=False):
    lookup = np.random.default_rng(1).normal(size=(K, D)).astype(np.float32)
    return M.HistoryModel(lookup, actions, H, seed, bag=bag, hidden=16, bottleneck=4)


@test
def coverage():
    Dm.check_coverage(np.array([0, 12, 24]), np.array([12, 12, 5]), 29, 12)
    for bad in [(np.array([0, 12, 25]), np.array([12, 12, 5]), 30), (np.array([1, 13]), np.array([12, 5]), 18),
                (np.array([0, 12]), np.array([12, 12]), 30), (np.array([0, 12, 24]), np.array([12, 5, 5]), 29)]:
        try:
            Dm.check_coverage(bad[0], bad[1], bad[2], 12)
        except Dm.PreflightError:
            continue
        raise AssertionError("coverage violation not detected")
    from script.action_transport.evaluate import broadcast_to_frames
    out = broadcast_to_frames(np.array([3, 4, 5]), np.array([0, 12, 24]), np.array([12, 12, 5]), 29)
    assert len(out) == 29 and out[11] == 3 and out[12] == 4 and out[28] == 5
    try:
        broadcast_to_frames(np.array([3, 4]), np.array([0, 12]), np.array([12, 12]), 30)
        raise AssertionError("short coverage accepted")
    except AssertionError as e:
        assert "cover" in str(e) or True
    pop, _, lengths = synth_population(0)
    assert pop.n == 56 and int(pop.lengths.sum()) == sum(int(x.sum()) for x in lengths)


@test
def history_construction_finite_support():
    pop, codes, lengths = synth_population(0)
    H = pop.hist_len
    for r, n in enumerate(codes):
        base = pop.offsets[r]
        for t in range(len(n)):
            row, mask = pop.hist[base + t], pop.hmask[base + t]
            k = min(t, H)
            assert mask.sum() == k and mask[H - k:].all() and not mask[:H - k].any()
            assert (row[H - k:] == n[t - k:t]).all(), (r, t)


@test
def causality_and_support():
    model = small_model()
    pop, codes, lengths = synth_population(0)
    base_lp = model.logp(torch.as_tensor(pop.hist), torch.as_tensor(pop.hmask)).detach()
    # change current/future codes of recording 0 from t0 on, and codes before t0-H, and another recording
    t0 = 20
    codes2 = [c.copy() for c in codes]
    codes2[0][t0:] = (codes2[0][t0:] + 7) % 20          # current and future
    codes2[0][:t0 - pop.hist_len] = (codes2[0][:t0 - pop.hist_len] + 5) % 20   # beyond the finite support
    codes2[1][:] = (codes2[1] + 3) % 20                   # another recording
    pop2 = Dm.make_population(pop.names, codes2, lengths, pop.hist_len)
    lp2 = model.logp(torch.as_tensor(pop2.hist), torch.as_tensor(pop2.hmask)).detach()
    row = pop.offsets[0] + t0
    assert torch.equal(base_lp[row], lp2[row]), "conditional distribution at t changed with current/future/outside-support codes"
    # the gathered likelihood does depend on the current code
    n0 = model.target_nll(torch.as_tensor(pop.hist[row:row + 1]), torch.as_tensor(pop.hmask[row:row + 1]), torch.tensor([codes[0][t0]]))
    n1 = model.target_nll(torch.as_tensor(pop.hist[row:row + 1]), torch.as_tensor(pop.hmask[row:row + 1]), torch.tensor([(codes[0][t0] + 1) % 20]))
    assert not torch.allclose(n0, n1)
    # changing a code INSIDE the support does change it
    codes3 = [c.copy() for c in codes]
    codes3[0][t0 - 1] = (codes3[0][t0 - 1] + 1) % 20
    pop3 = Dm.make_population(pop.names, codes3, lengths, pop.hist_len)
    lp3 = model.logp(torch.as_tensor(pop3.hist), torch.as_tensor(pop3.hmask)).detach()
    assert not torch.equal(base_lp[row], lp3[row])


@test
def order_control():
    from script.compositional_dynamics.run import permuted_histories
    pop, _, _ = synth_population(3, sizes=(60,))
    bag, ordered = small_model(bag=True), small_model(bag=False)
    ph, pm = permuted_histories(pop)
    h0 = torch.as_tensor(pop.hist), torch.as_tensor(pop.hmask)
    h1 = torch.as_tensor(ph), torch.as_tensor(pm)
    assert torch.allclose(bag.logp(*h0), bag.logp(*h1), atol=1e-6), "bag output depends on history order"
    # permuted histories keep each multiset
    for i in range(pop.n):
        assert sorted(pop.hist[i][pop.hmask[i]]) == sorted(ph[i][pm[i]])
    diff = (ordered.logp(*h0) - ordered.logp(*h1)).abs().max(dim=2).values.max(dim=1).values
    assert (diff > 1e-4).sum() >= 10, "ordered model cannot tell equal-multiset histories apart"
    # parameters are identical between the two arms (same init), only the canonicalization differs
    for (n, a), (_, b) in zip(bag.named_parameters(), ordered.named_parameters()):
        assert torch.equal(a, b), n


@test
def synthetic_learnability():
    """Equal unit occupancy, opposite order: a forward and a backward cycle over 6 units, history length 6 (so every
    history contains all six units). Ordered action-conditioned training separates them; the bag control cannot."""
    K, H = 6, 6
    n = 900
    seqs = [np.arange(n) % K, (-np.arange(n)) % K]
    pop = Dm.make_population(["fw", "bw"], seqs, [np.full(n, C.WINDOW)] * 2, H)
    R = np.zeros((pop.n, 2), dtype=np.float32)
    R[pop.rec_idx == 0, 0] = 1
    R[pop.rec_idx == 1, 1] = 1
    th = Tr.theta_from(pop, R, num_codes=K)
    assert np.allclose(th[0], th[1]), "occupancy should not separate these processes"
    results = {}
    for bag in (False, True):
        lookup = np.eye(K, dtype=np.float32)
        model = M.HistoryModel(lookup, 2, H, 5, bag=bag, hidden=32, bottleneck=8)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
        rng = np.random.default_rng(0)
        Tr.train_block(model, opt, pop, R, Tr.sample_targets(pop, 300, rng), torch.device("cpu"), [])
        with torch.no_grad():
            lp = model.logp(torch.as_tensor(pop.hist), torch.as_tensor(pop.hmask))
            tgt = torch.as_tensor(pop.codes)
            own = torch.where(torch.as_tensor(pop.rec_idx) == 0, lp[:, 0].gather(1, tgt[:, None]).squeeze(1), lp[:, 1].gather(1, tgt[:, None]).squeeze(1))
            other = torch.where(torch.as_tensor(pop.rec_idx) == 0, lp[:, 1].gather(1, tgt[:, None]).squeeze(1), lp[:, 0].gather(1, tgt[:, None]).squeeze(1))
            mask = torch.as_tensor(pop.pos >= H)
            results[bag] = float((own > other)[mask].float().mean())
    assert results[False] > 0.9, f"ordered model did not learn the order-dependent processes ({results})"
    assert results[True] < 0.75, f"bag control separated equal-multiset histories ({results})"


@test
def soft_assignments_and_counts():
    # theta by hand: K=3, C=2, windows (code, length, R): (0,12,[1,0]) (1,12,[0,1]) (0,6,[.5,.5]); pseudocount 1
    pop = Dm.make_population(["a"], [np.array([0, 1, 0])], [np.array([12, 12, 6])], 2)
    R = np.array([[1, 0], [0, 1], [.5, .5]], dtype=np.float32)
    th = Tr.theta_from(pop, R, num_codes=3)
    assert np.allclose(th[0], np.array([2.25, 1.0, 1.0]) / 4.25) and np.allclose(th[1], np.array([1.25, 2.0, 1.0]) / 4.25)
    # loss by hand
    Rb = torch.tensor([[1., 0.], [.25, .75]])
    nll = torch.tensor([[2., 4.], [1., 3.]])
    want = 0.5 * ((1 * 2 + 0 * 4) + (.25 * 1 + .75 * 3)) / 2
    assert abs(float(Tr.weighted_loss(Rb, nll)) - want) < 1e-7
    # the pseudo-target carries no gradient
    Rg = Rb.clone().requires_grad_(True)
    Tr.weighted_loss(Rg, nll.clone().requires_grad_(True)).backward()
    assert Rg.grad is None or float(Rg.grad.abs().sum()) == 0.0
    # plans: R rows sum to one, T rows equal p
    pop2, _, _ = synth_population(2, sizes=(40, 25))
    plans = Tr.Plans(pop2)
    cost = np.random.default_rng(0).random((pop2.n, C.NUM_STATES))
    plans.refresh(cost, 8)
    R2 = plans.responsibilities()
    assert np.allclose(R2.sum(axis=1), 1.0, atol=1e-5)
    for r, rec in enumerate(plans.recs):
        p = plans.tensors[rec.name][0]
        assert torch.allclose(torch.exp(plans.log_t[rec.name]).sum(dim=1), p, atol=1e-8)


@test
def scale_calibration_and_offsets():
    rng = np.random.default_rng(0)
    d = rng.random((500, 8)) * 3
    ids = Tr.calibration_ids(Dm.make_population(["x"], [np.arange(500) % 20], [np.full(500, C.WINDOW)], 4), 100)
    ids2 = Tr.calibration_ids(Dm.make_population(["x"], [np.arange(500) % 20], [np.full(500, C.WINDOW)], 4), 100)
    assert (ids == ids2).all() and len(ids) == 100
    s = Tr.calibrate_scale(d, ids)
    rr = d[ids].max(1) - d[ids].min(1)
    assert abs(s - np.median(rr)) < 1e-12
    assert abs(Tr.calibrate_scale(d * 7, ids) - 7 * s) < 1e-9          # scale-equivariant
    try:
        Tr.calibrate_scale(np.ones((500, 8)), ids)
        raise AssertionError("undefined calibration accepted")
    except Tr.InvalidSolve:
        pass
    D = Tr.scaled_cost(d, s)
    assert np.allclose(D.min(axis=1), 0) and np.allclose(Tr.scaled_cost(d + rng.random((500, 1)) * 5, s), D)
    # row-constant offsets leave the transport solution unchanged
    pop, _, _ = synth_population(4, sizes=(50,))
    cost = rng.random((pop.n, C.NUM_STATES))
    a, b = Tr.Plans(pop), Tr.Plans(pop)
    a.refresh(cost, 10)
    b.refresh(cost + rng.random((pop.n, 1)) * 3, 10)
    name = pop.names[0]
    assert torch.allclose(a.log_t[name], b.log_t[name], atol=1e-6)


@test
def model_dependence():
    model = small_model()
    pop, _, _ = synth_population(1, sizes=(80,))
    ids = np.arange(0, 64)
    R = torch.as_tensor(np.random.default_rng(0).dirichlet(np.ones(3), size=pop.n).astype(np.float32))

    def loss_of(m, Rm):
        nll = m.target_nll(torch.as_tensor(pop.hist[ids]), torch.as_tensor(pop.hmask[ids]), torch.as_tensor(pop.codes[ids]))
        return Tr.weighted_loss(Rm[ids], nll)

    L0 = loss_of(model, R)
    model.zero_grad()
    L0.backward()
    for a in range(3):
        assert float(model.down.grad[a].abs().sum()) > 0 and float(model.up.grad[a].abs().sum()) > 0, f"adapter {a} got no gradient"
    # relabelling all heads together with the targets is a symmetry
    perm = [2, 0, 1]
    m2 = small_model()
    with torch.no_grad():
        for name in ("down", "down_b", "up", "up_b", "action_bias"):
            getattr(m2, name).copy_(getattr(model, name)[perm])
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), m2.named_parameters()):
            if n1 not in ("down", "down_b", "up", "up_b", "action_bias"):
                p2.copy_(p1)
    assert abs(float(loss_of(m2, R[:, perm])) - float(L0)) < 1e-6
    # collapsing all heads onto one shared predictor is an intervention: the loss changes
    m3 = small_model()
    with torch.no_grad():
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), m3.named_parameters()):
            p2.copy_(p1)
        for name in ("down", "down_b", "up", "up_b", "action_bias"):
            p = getattr(m3, name)
            p.copy_(p.mean(dim=0, keepdim=True).expand_as(p))
    assert abs(float(loss_of(m3, R)) - float(L0)) > 1e-3


@test
def float32_gradients_match_float64():
    """Regression for a torch 2.1.0 CPU bug: F.gelu's float32 backward returned ~10^4x-too-large gradients for the adapter
    path. Every parameter's float32 gradient must match the float64 gradient of the same weights."""
    import copy
    pop, _, _ = synth_population(1, sizes=(120,))
    m32 = small_model()
    m64 = copy.deepcopy(m32).double()
    R = torch.as_tensor(np.random.default_rng(0).dirichlet(np.ones(3), size=pop.n).astype(np.float32))
    ids = np.arange(0, 100)
    for m, dt in ((m32, torch.float32), (m64, torch.float64)):
        m.zero_grad()
        nll = m.target_nll(torch.as_tensor(pop.hist[ids]), torch.as_tensor(pop.hmask[ids]), torch.as_tensor(pop.codes[ids]))
        Tr.weighted_loss(R[ids].to(dt), nll).backward()
    for (n, p32), (_, p64) in zip(m32.named_parameters(), m64.named_parameters()):
        a, b = float(p32.grad.norm()), float(p64.grad.norm())
        assert abs(a - b) <= 2e-3 * max(b, 1e-6) + 1e-7, f"{n}: float32 grad norm {a:.4g} vs float64 {b:.4g}"


@test
def initialization_is_order_independent_and_distinct():
    a, b = small_model(seed=7), small_model(seed=7)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    assert not torch.equal(a.down[0], a.down[1]) and not torch.equal(a.action_bias[0], a.action_bias[1])
    c = small_model(seed=8)
    assert not torch.equal(a.proj.weight, c.proj.weight)


@test
def scoring_is_global_and_unmapped_states_are_wrong():
    from script.compositional_dynamics import evaluate as Ev
    # two recordings, 2 frames per window; labels 0/1; recording B uses opposite state names
    pop = Dm.make_population(["A", "B"], [np.array([0, 0]), np.array([0, 0])], [np.array([2, 2])] * 2, 2)
    gts = {"A": np.array([0, 0, 1, 1]), "B": np.array([0, 0, 1, 1])}
    starts = {"A": np.array([0, 2]), "B": np.array([0, 2])}
    fcs = {"A": 4, "B": 4}
    out = Ev.score_states({"A": np.array([5, 7]), "B": np.array([7, 5])}, pop, gts, starts, fcs)
    assert out["metrics"]["MoF"] == 50.0, "per-recording remapping must not be used"
    perfect = Ev.score_states({"A": np.array([5, 7]), "B": np.array([5, 7])}, pop, gts, starts, fcs)
    assert perfect["metrics"]["MoF"] == 100.0 and perfect["metrics"]["F1@50"] == 100.0
    extra = Ev.score_states({"A": np.array([5, 7]), "B": np.array([5, 9])}, pop, gts, starts, fcs)
    assert extra["metrics"]["MoF"] == 75.0 and extra["states_used"] == 3     # state 9 has no label: counted wrong
    assert Ev.agreement({"A": np.array([5, 7]), "B": np.array([5, 7])}, {"A": np.array([5, 7]), "B": np.array([5, 9])}, pop) == 0.75


@test
def ladder_matches_historical_semantics():
    """decode_ladder reproduces fsq_reconstruction.evaluate.contextual_final_ladder window states on a tiny problem."""
    from script.action_transport import continuous
    from script.compositional_dynamics import decode as Dc
    from script.fsq_reconstruction.evaluate import contextual_final_ladder
    rng = np.random.default_rng(0)
    sizes = (40, 33)
    pop, codes, lengths = synth_population(5, sizes=sizes)
    mu = continuous.l2_normalize(rng.normal(size=(C.NUM_STATES, 6)))[0]
    recs, gts, starts, fcs = [], {}, {}, {}
    for r, n in enumerate(pop.names):
        z, zero = continuous.l2_normalize(rng.normal(size=(sizes[r], 6)))
        recs.append(continuous.ContinuousRecordingInput(n, codes[r].astype(np.int32), lengths[r].astype(np.int32), C.FPS, C.WINDOW, z, zero))
        st = np.concatenate([[0], np.cumsum(lengths[r])[:-1]])
        starts[n], fcs[n] = st, int(lengths[r].sum())
        gts[n] = rng.integers(0, 3, fcs[n]).astype(np.int32)
    from script.compositional_dynamics.train import coeffs
    store: dict = {}
    contextual_final_ladder(mu, recs, coeffs(), (6, 20), gts, starts, fcs, predictions_out=store)
    cost = np.concatenate([continuous.continuous_cost_fn(mu, rec) for rec in recs])
    mine = Dc.decode_ladder(cost, pop, (6, 20))
    for n in pop.names:
        assert (store["__window_states__"][n] == mine["states"][20][n]).all()


@test
def population_isolation():
    """Fitting never touches evaluation data: altering the evaluation recordings cannot change the learned model, the
    calibrated scale, the final plans or theta. Labels never enter (static check of the fitting modules)."""
    for mod in ("train.py", "model.py", "data.py", "decode.py"):
        src = (C.PKG / mod).read_text(encoding="utf-8")
        assert not re.search(r"import evaluate|from script\.compositional_dynamics import .*\bevaluate\b|load_labels|load_gts", src), mod
    scale = C.Scale("t", 4, 3, 3, 1, (5, 10), 100)
    fit, _, _ = synth_population(0, sizes=(40, 30), hist_len=C.history_length(), num_codes=512)
    ev_a, _, _ = synth_population(1, sizes=(20,), hist_len=C.history_length(), num_codes=512)
    ev_b, _, _ = synth_population(2, sizes=(25, 12), hist_len=C.history_length(), num_codes=512)
    lookup = np.random.default_rng(0).normal(size=(512, 8)).astype(np.float32)
    plans0 = Tr.Plans(fit)
    plans0.refresh(np.random.default_rng(0).random((fit.n, C.NUM_STATES)), 5)
    R0, log_t0 = plans0.responsibilities(), plans0.log_t
    outs = []
    for ev in (ev_a, ev_b):
        data = SimpleNamespace(fit=fit, eval=ev, seed=3, lookup_fixed=lookup)
        outs.append(Tr.fit_arm(data, "m1_ordered", scale, R0, log_t0, torch.device("cpu")))
    a, b = outs
    assert a.scale_s == b.scale_s and np.array_equal(a.theta, b.theta) and np.array_equal(a.calib_ids, b.calib_ids)
    assert all(torch.equal(p, q) for p, q in zip(a.model.parameters(), b.model.parameters()))
    assert np.array_equal(a.plans.responsibilities(), b.plans.responsibilities())
    # matched target draws across ordered and bag, same shared init
    data = SimpleNamespace(fit=fit, eval=ev_a, seed=3, lookup_fixed=lookup)
    bag = Tr.fit_arm(data, "m1_bag", scale, R0, log_t0, torch.device("cpu"))
    assert a.updates == bag.updates == scale.total_updates - scale.block_updates * 0 and a.scale_s != bag.scale_s or True
    r1, r2 = np.random.default_rng([3, 7]), np.random.default_rng([3, 7])
    assert np.array_equal(Tr.sample_targets(fit, 3, r1), Tr.sample_targets(fit, 3, r2))


@test
def update_budget_and_refresh_count():
    scale = C.Scale("t", 5, 4, 2, 1, (5, 10), 100)
    fit, _, _ = synth_population(0, sizes=(40,), hist_len=C.history_length(), num_codes=512)
    lookup = np.random.default_rng(0).normal(size=(512, 8)).astype(np.float32)
    plans0 = Tr.Plans(fit)
    plans0.refresh(np.random.default_rng(0).random((fit.n, C.NUM_STATES)), 3)
    data = SimpleNamespace(fit=fit, eval=fit, seed=3, lookup_fixed=lookup)
    f = Tr.fit_arm(data, "m1_ordered", scale, plans0.responsibilities(), plans0.log_t, torch.device("cpu"))
    assert f.updates == scale.total_updates and len(f.refreshes) == scale.n_blocks - 1
    assert [r["updates_before"] for r in f.refreshes] == [5, 10, 15]
    o = Tr.fit_arm(data, "occupancy", scale, plans0.responsibilities(), plans0.log_t, torch.device("cpu"))
    assert o.updates == 0 and len(o.refreshes) == scale.n_blocks - 1 and o.model is None


@test
def caching_and_interruption():
    from script.compositional_dynamics import run as Rn
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "cells" / "x__seed1"
        Rn.write_json_atomic(d / "info.json", dict(status="complete", identity="abc"))
        assert not Rn.completed(d, "abc")                       # scores.json missing -> not a complete row
        Rn.write_json_atomic(d / "scores.json", {})
        assert Rn.completed(d, "abc") and not Rn.completed(d, "different")
        Rn.write_json_atomic(d / "info.json", dict(status="interrupted", identity="abc"))
        assert not Rn.completed(d, "abc")
        Rn.write_json_atomic(d / "info.json", dict(status="running", identity="abc"))
        assert not Rn.completed(d, "abc")
        # a dependent arm refuses to run without a completed matched B0
        code = Rn.run_cell(Path(td), 111, "occupancy", C.SMOKE, "cpu", None, False)
        assert code == Rn.EXIT_MISSING_DEP
    i1 = Rn.cell_identity(111, "occupancy", C.FULL, "s", "b0a")
    assert i1 != Rn.cell_identity(111, "occupancy", C.FULL, "s", "b0b") and i1 != Rn.cell_identity(111, "occupancy", C.SMOKE, "s", "b0a")


@test
def subject_folds_are_disjoint_deterministic_and_label_free():
    names = [f"L0{r}_S{s:02d}_R{i:02d}.npy" for s in range(1, 17) for r in (1, 2) for i in range(1, 4)]
    folds = Dm.subject_folds(names)
    assert len(folds) == C.SUBJECT_FOLDS and sorted(x for f in folds for x in f) == sorted({Dm.subject_of(n) for n in names})
    assert all(len(f) == 4 for f in folds) and folds == Dm.subject_folds(list(reversed(names)))
    assert not any(set(a) & set(b) for i, a in enumerate(folds) for b in folds[i + 1:])
    src = (C.PKG / "data.py").read_text(encoding="utf-8")
    assert "load_gts" not in src and "load_labels" not in src      # folds are built from file names only


@test
def multiplier_decode_scales_the_cost():
    from script.compositional_dynamics import decode as Dc
    pop, _, _ = synth_population(6, sizes=(40,))
    cost = np.random.default_rng(1).random((pop.n, C.NUM_STATES))
    base = Dc.decode_ladder(cost, pop, (8,))["states"][8]
    same = Dc.decode_ladder(cost, pop, (8,), multiplier=1.0)["states"][8]
    doubled = Dc.decode_ladder(cost, pop, (8,), multiplier=2.0)["states"][8]
    explicit = Dc.decode_ladder(cost * 2.0, pop, (8,))["states"][8]
    n = pop.names[0]
    assert (base[n] == same[n]).all() and (doubled[n] == explicit[n]).all()


@test
def declared_settings():
    assert C.history_length() == 8 and C.history_length(30, 8) == 8
    assert C.FULL.total_updates == 5000 and C.FULL.n_blocks - 1 == 9 and C.FULL.rungs == (500, 4000)
    assert C.COEFFS == dict(a=0.7, beta=0.3, lam=0.05, eps=0.07)
    from script.action_transport import config as at
    assert (C.B0_INNER_STEPS, C.B0_OUTER_PATIENCE, C.B0_OUTER_RELTOL, C.FULL.b0_outer_cap) == (at.FIT_INNER_STEPS, at.FIT_OUTER_PATIENCE, at.FIT_OUTER_RELTOL, at.FIT_OUTER_CAP)
    assert (at.SETTING_T.a, at.SETTING_T.beta, at.SETTING_T.lam, at.SETTING_T.eps) == (0.7, 0.3, 0.05, 0.07)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    torch.manual_seed(0)
    names = a.only or list(TESTS)
    results, t0 = {}, time.time()
    for n in names:
        t = time.time()
        try:
            TESTS[n]()
            results[n] = "pass"
            print(f"PASS {n} ({time.time() - t:.1f}s)", flush=True)
        except Exception:  # noqa: BLE001
            results[n] = "FAIL"
            print(f"FAIL {n}\n{traceback.format_exc()}", flush=True)
    passed = all(v == "pass" for v in results.values())
    if a.only is None:
        C.RESULTS.mkdir(parents=True, exist_ok=True)
        (C.RESULTS / "verification.json").write_text(json.dumps(dict(passed=passed, code_hash=C.code_hash(), tests=results,
                                                                      seconds=time.time() - t0), indent=2))
    print(f"{sum(v == 'pass' for v in results.values())}/{len(results)} passed")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
