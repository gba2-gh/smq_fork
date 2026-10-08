"""Focused verification for the contextualizer calibration pilot. Synthetic
and fast: no full-data anchor training and no OT fit is run here.

    python -m script.week5.contextualizer_calibration.run_calibration --preflight

`verify_run_dir` checks a finished smoke run (saved-prediction rescoring,
explicit incomplete units); the runner calls it after `--smoke`.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from script.action_transport.stageb_data import chunk_starts
from script.action_transport.stageb_model import CodeContextModel, ModelConfig
from script.quantized_input_comparison import contextual as C
from script.quantized_input_comparison import masking as qic_masking
from script.week5.contextualizer_calibration import evaluate as E
from script.week5.contextualizer_calibration import plan as P
from script.week5.contextualizer_calibration import search_space as S
from script.week5.contextualizer_calibration import train as T

CPU = torch.device("cpu")


# --- synthetic data --------------------------------------------------------------------------

class FakeLoader:
    """Minimal stand-in for `artifacts.Loader`: one family, a few recordings of 15-frame windows."""

    def __init__(self, n_train=6, n_val=2, num_codes=16, dim=5, seed=0, lengths=None, family="fam"):
        rng = np.random.default_rng(seed)
        self.family, self.num_codes, self.dim = family, num_codes, dim
        self._lookup = rng.normal(size=(num_codes, dim)).astype(np.float32)
        self.recs, self.split = {}, {}
        lengths = lengths or [int(x) for x in rng.integers(150, 260, size=n_train + n_val)]
        for i, n in enumerate(lengths):
            name = f"r{i:02d}"
            starts = (np.arange(n) * 15).astype(np.int32)
            self.recs[name] = dict(codes=rng.integers(0, num_codes, size=n).astype(np.int64),
                                   continuous=rng.normal(size=(n, dim)).astype(np.float32), starts=starts,
                                   lengths=np.full(n, 15, dtype=np.int32), frames=int(n * 15))
            self.split[name] = "train" if i < len(lengths) - n_val else "val"
        self.manifest = dict(num_windows={family: {n: len(r["codes"]) for n, r in self.recs.items()}})

    def recording_names(self):
        return list(self.recs)

    def recording(self, family, name):
        return {k: v for k, v in self.recs[name].items() if k != "frames"}

    def frame_count(self, name):
        return self.recs[name]["frames"]

    def split_membership(self, name):
        return self.split[name]

    def lookup(self, family):
        return self._lookup

    def rms(self, family):
        return np.asarray(1.3, dtype=np.float32)

    def grouping(self, family):
        return (np.arange(self.num_codes) % 10).astype(np.int32)


def small_cfg(max_steps=30, eval_every=10, batch=8, **kw):
    return C.base_train_config(max_steps=max_steps, eval_every=eval_every, patience_steps=10**9, batch_size=batch,
                               warmup_steps=5, **kw)


def chunks_for(loader, ctx=128, stride=64):
    names = loader.recording_names()
    train = [n for n in names if loader.split_membership(n) == "train"]
    val = [n for n in names if loader.split_membership(n) == "val"]
    return (C.make_vector_chunks(loader, loader.family, train, chunk=ctx, stride=stride),
            C.make_vector_chunks(loader, loader.family, val, chunk=ctx, stride=stride))


def core(loader, arm, cfg, radius=14, ckpt_dir=None, seed=111, ctx=128, stride=64, **kw):
    train, val = chunks_for(loader, ctx, stride)
    lookup, rms = loader.lookup(loader.family), float(loader.rms(loader.family))
    init = C.build_matched_init(lookup.shape[0], lookup.shape[1], cfg, seed)
    return T.train_core(arm=arm, train_chunks=train, val_chunks=val, num_codes=lookup.shape[0],
                        input_dim=lookup.shape[1], seed=seed, cfg=cfg, init=init, device=CPU,
                        lookup=lookup if arm == "context_quantized" else None, rms=rms, guard_radius=radius,
                        ckpt_dir=ckpt_dir, **kw)


def states_equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


# --- 1. plan counts ------------------------------------------------------------------------------

def test_plan_counts_and_ids():
    jobs, evals = P.enumerate_training_jobs(), None
    evals = P.enumerate_evaluations(jobs)
    assert (len(jobs), len(evals)) == (14, 58), (len(jobs), len(evals))
    jobs_c = P.enumerate_training_jobs(True)
    evals_c = P.enumerate_evaluations(jobs_c)
    assert (len(jobs_c), len(evals_c)) == (15, 60)
    assert not any(j.group == "target_match_combined" for j in jobs)
    assert len({j.job_id for j in jobs_c}) == 15 and len({e.eval_id for e in evals_c}) == 60
    by_id = {j.job_id: j for j in jobs}
    h3 = [j for j in jobs if j.group == "horizon3k"]
    assert {(j.pipeline, j.guard_radius, j.seed, j.horizon) for j in h3} == {("native_vq", 14, 111, 3000), ("raw", 0, 111, 3000)}
    s2 = [j for j in jobs if j.group == "seed222"]
    assert {(j.pipeline, j.guard_radius, j.seed, j.horizon) for j in s2} == {("native_vq", 14, 222, 5000), ("raw", 0, 222, 5000)}
    main = [j for j in jobs if j.group == "main"]
    assert len(main) == 8 and {j.guard_radius for j in main if j.pipeline == "raw"} == {0}
    guard = [j for j in jobs if j.group == "guard_ref"]
    assert len(guard) == 1 and guard[0].guard_radius == 14 and guard[0].pipeline == "raw"
    for j in jobs:
        assert j.context * j.batch_chunks == P.NOMINAL_TOKENS_PER_UPDATE
        assert j.job_id == P.make_job_id(j.pipeline, j.context, j.mask_span, j.guard_radius, j.horizon, j.seed, j.target_count)
    tm = [j for j in jobs if j.group == "target_match"][0]
    assert tm.ref_job_id in by_id and by_id[tm.ref_job_id].mask_span == 8 and by_id[tm.ref_job_id].context == tm.context
    assert jobs[0].job_id.startswith("native_vq__ctx128_span8_g14") and jobs[1].group == "guard_ref"
    assert sum(e.kind == "outer_sensitivity" for e in evals) == 4 and all(e.outer_cap == 150 for e in evals if e.kind == "outer_sensitivity")
    assert not any("probe" in j.job_id for j in jobs)
    return True


def test_effective_settings_identity():
    jobs = P.enumerate_training_jobs()
    ids = {j.job_id: P.job_identity(j, P.FULL) for j in jobs}
    assert len(set(ids.values())) == len(jobs)
    assert ids == {j.job_id: P.job_identity(j, P.FULL) for j in jobs}
    assert P.job_identity(jobs[0], P.FULL) != P.job_identity(jobs[0], P.SMOKE)
    cfg = T.train_config(jobs[0], P.FULL)
    assert (cfg.max_steps, cfg.warmup_steps, cfg.schedule, cfg.batch_size, cfg.mask_span) == (5000, 500, "cosine", 64, 8)
    assert (cfg.lr, cfg.weight_decay, cfg.grad_clip, cfg.position_mode, cfg.norm_first, cfg.tie_head) == (3e-4, 0.01, 1.0, "alibi", False, False)
    return True


# --- 2. parity with Task 2 -----------------------------------------------------------------------------

def test_model_config_parity_with_task2():
    cfg = T.train_config(P.enumerate_training_jobs()[0], P.FULL)
    mc = C.model_config(512, "vector", 1440, cfg)
    assert (mc.d_model, mc.n_layers, mc.n_heads, mc.d_ff, mc.dropout, mc.norm_first, mc.tie_head, mc.position_mode) == \
        (128, 4, 4, 512, 0.1, False, False, "alibi")
    assert T.model_settings(cfg) == dict(d_model=128, n_layers=4, n_heads=4, d_ff=512, dropout=0.1, norm_first=False,
                                         tie_head=False, position_mode="alibi")
    return True


def test_training_parity_with_task2_trajectory():
    loader = FakeLoader()
    for arm in ("context_id", "context_quantized"):
        cfg = small_cfg()
        train, val = chunks_for(loader)
        lookup, rms = loader.lookup("fam"), float(loader.rms("fam"))
        init = C.build_matched_init(16, 5, cfg, 111)
        ref = C.train_vector_contextualizer(arm, train, val, 16, 5, 111, cfg, init, CPU, lookup=lookup, rms=rms)
        mine = core(loader, arm, cfg)
        assert mine.history == ref.info["history"], arm
        assert mine.best_step == ref.info["best_step"] and mine.best_val == ref.info["best_val_loss"]
        assert states_equal(mine.best_state, ref.model.state_dict()) and states_equal(mine.last_state, ref.info["last_state"])
        assert mine.stats["skipped_train_chunks"] == ref.info["skipped_train_chunks"]
    return True


def test_checkpoint_logging_leaves_training_unchanged():
    loader = FakeLoader()
    cfg = small_cfg()
    plain = core(loader, "context_quantized", cfg)
    with tempfile.TemporaryDirectory() as tmp:
        logged = core(loader, "context_quantized", cfg, ckpt_dir=Path(tmp), fixed_steps={"ckpt_1k": 10, "ckpt_3k": 20})
        assert logged.history == plain.history and states_equal(logged.last_state, plain.last_state)
        files = sorted(p.name for p in Path(tmp).glob("step*.pt"))
        assert files and all(row["file"] in files for row in logged.ledger)
        assert {10, 20, 30} <= {row["step"] for row in logged.ledger}  # fixed + final checkpoints exist
        for row in logged.ledger:
            state = T.load_state(Path(tmp) / row["file"])["state_dict"]
            assert T.tensor_hash(state) == row["tensor_sha256"]
    return True


def test_default_mask_path_matches_task2_mask_construction():
    loader = FakeLoader()
    train, _ = chunks_for(loader)
    c = train[0]
    a = C.chunk_masks([c], 111, 0.3, 8, radius=14)[0]
    b = qic_masking.build_eligible_mask(c.starts, c.lengths, c.valid_len, c.frame_count,
                                        np.random.default_rng(__import__("script.action_transport.stageb_data", fromlist=["x"]).stable_seed(c.recording, c.start, 111)),
                                        radius=14, fraction=0.3, span=8)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    return True


# --- 3. horizons, update counting, accumulation, empty targets ---------------------------------------------

def test_schedule_horizons_are_real():
    loader = FakeLoader()
    short = core(loader, "context_id", small_cfg(max_steps=20))
    long = core(loader, "context_id", small_cfg(max_steps=30))
    floor = 0.1 * 3e-4
    assert abs(short.history[-1]["lr"] - floor) < 1e-12 and abs(long.history[-1]["lr"] - floor) < 1e-12
    at20_long = next(h["lr"] for h in long.history if h["step"] == 20)
    assert at20_long > floor * 1.5, "a 30-update schedule must not have decayed at update 20 as a 20-update one has"
    assert short.stats["accepted_updates"] == 20 and long.stats["accepted_updates"] == 30
    full = {j.horizon: P.FULL.horizon(j.horizon) for j in P.enumerate_training_jobs()}
    assert full == {5000: 5000, 3000: 3000}
    return True


def test_weighted_accumulation_matches_monolithic_batch():
    """With dropout off, the accumulated update sees the same weighted loss and gradient as the
    monolithic batch. Compared at the first update (identical model state): later Adam steps
    amplify float-order noise on near-zero-gradient parameters, which says nothing about the loss."""
    loader = FakeLoader(n_train=8)
    cfg = small_cfg(max_steps=1, eval_every=1, batch=8, dropout=0.0)
    a = core(loader, "context_quantized", cfg)
    b = core(loader, "context_quantized", cfg, micro_chunks=3)
    ha, hb = a.history[0], b.history[0]
    assert abs(ha["train_loss"] - hb["train_loss"]) < 1e-5 * max(1.0, ha["train_loss"])
    assert abs(ha["grad_norm"] - hb["grad_norm"]) < 1e-4 * max(1.0, ha["grad_norm"])
    assert abs(ha["val_loss"] - hb["val_loss"]) < 1e-3
    return True


def test_empty_target_batches_do_not_advance_and_failure_is_bounded():
    loader = FakeLoader()
    cfg = small_cfg(max_steps=12, eval_every=4)
    real = T._build
    calls = {"n": 0}

    def flaky(chunk, rng, radius, fraction, span):
        calls["n"] += 1
        sel, elig = real(chunk, rng, radius, fraction, span)
        return (sel, np.zeros_like(elig)) if calls["n"] <= 2 * 8 else (sel, elig)  # first two batches empty

    T._build = flaky
    try:
        res = core(loader, "context_id", cfg)
    finally:
        T._build = real
    assert res.stats["accepted_updates"] == 12 and res.stats["skipped_batches"] == 2
    assert res.stats["skipped_train_chunks"] == 16 and res.stats["attempted_batches"] == 14
    assert [h["step"] for h in res.history] == [4, 8, 12]
    try:
        core(loader, "context_id", small_cfg(max_steps=4, eval_every=2), radius=100_000)
    except RuntimeError as exc:
        assert "zero eligible" in str(exc)
    else:
        raise AssertionError("persistently empty targets must fail loudly")
    return True


# --- 4. guard radius, spans, matched counts --------------------------------------------------------------------

def test_guard_radius_zero_vs_fourteen_and_span_eligibility():
    loader = FakeLoader(lengths=[200, 200, 200])
    train, _ = chunks_for(loader)
    c = train[0]
    sel = np.zeros(128, dtype=bool)
    sel[10:40] = True
    e0 = qic_masking.eligible_loss_positions(c.starts, c.lengths, c.valid_len, c.frame_count, sel, radius=0)
    e14 = qic_masking.eligible_loss_positions(c.starts, c.lengths, c.valid_len, c.frame_count, sel, radius=14)
    assert np.array_equal(e0, sel) and e14.sum() == 26  # raw patches never overlap; guarded drops 2 per side
    for run, expected in ((1, 0), (4, 0), (5, 1), (8, 4), (16, 12)):
        s = np.zeros(128, dtype=bool)
        s[40:40 + run] = True
        e = qic_masking.eligible_loss_positions(c.starts, c.lengths, c.valid_len, c.frame_count, s, radius=14)
        assert e.sum() == expected, (run, e.sum())
        assert S.guard_min_tokens(14, 15) == 5 and (run >= 5) == (expected > 0)
        assert qic_masking.eligible_loss_positions(c.starts, c.lengths, c.valid_len, c.frame_count, s, radius=0).sum() == run
    seen = set()
    real = qic_masking.build_eligible_mask

    def spy(*a, **k):
        seen.add(k.get("radius"))
        return real(*a, **k)

    qic_masking.build_eligible_mask = spy
    try:
        core(FakeLoader(), "context_id", small_cfg(max_steps=4, eval_every=2), radius=0)
    finally:
        qic_masking.build_eligible_mask = real
    assert seen == {0}, f"radius must reach both training and validation mask construction, saw {seen}"
    return True


def test_matched_target_counts_retries_and_isolated_rng():
    loader = FakeLoader()
    train, _ = chunks_for(loader)
    batch = train[:8]
    seeds = [int(x) for x in np.random.default_rng(3).integers(0, 2**31 - 1, size=8)]
    kw = dict(radius=14, fraction=0.3, span_long=16, span_ref=8, base_seed=111, epoch=0, batch_index=0)
    torch_state, np_state = torch.get_rng_state(), np.random.get_state()
    out = T.matched_batch_masks(batch, seeds, **kw)
    assert torch.equal(torch_state, torch.get_rng_state()) and all(np.array_equal(x, y) for x, y in zip(np_state[1:2], np.random.get_state()[1:2]))
    again = T.matched_batch_masks(batch, seeds, **kw)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(out["masks"], again["masks"]))  # deterministic
    assert out["n_after"] == out["n_ref"] == sum(int(e.sum()) for _, e in out["masks"]) and out["n_before"] >= out["n_ref"]
    for (sel, elig), c in zip(out["masks"], batch):
        assert (elig <= sel).all()  # supervised positions are still masked inputs
    ref_total = sum(int(real[1].sum()) for real in (T._build(c, np.random.default_rng(s), 14, 0.3, 8) for c, s in zip(batch, seeds)))
    assert out["n_ref"] == ref_total
    # retry path: the first long-mask draws fall short, a later retry stream supplies enough
    real_build, state = T._build, {"long_calls": 0}

    def short_then_ok(chunk, rng, radius, fraction, span):
        sel, elig = real_build(chunk, rng, radius, fraction, span)
        if span == 16:
            state["long_calls"] += 1
            if state["long_calls"] <= 8:  # the initial draw falls short; the first retry stream suffices
                elig = np.zeros_like(elig)
        return sel, elig

    T._build = short_then_ok
    try:
        retried = T.matched_batch_masks(batch, seeds, **kw)
        assert retried["retries"] == 1 and retried["n_after"] == retried["n_ref"]
        state["long_calls"] = -10**9  # never supply enough -> explicit unavailable with batch identity
        T._build = lambda c, r, rad, f, span: (lambda sel_elig: (sel_elig[0], np.zeros_like(sel_elig[1])) if span == 16 else sel_elig)(real_build(c, r, rad, f, span))
        try:
            T.matched_batch_masks(batch, seeds, max_attempts=3, **kw)
        except T.TargetMatchUnavailable as exc:
            assert "epoch=0" in str(exc)
        else:
            raise AssertionError("expected TargetMatchUnavailable")
    finally:
        T._build = real_build
    paired = core(loader, "context_quantized", small_cfg(max_steps=12, eval_every=4, mask_span=16), reference_span=8)
    reference = core(loader, "context_quantized", small_cfg(max_steps=12, eval_every=4))
    assert np.array_equal(paired.arrays["n_ref"], reference.arrays["eligible"]), "paired reference counts must equal the span-8 job's"
    assert np.array_equal(paired.arrays["eligible"], paired.arrays["n_ref"])
    return True


# --- 5. stride / anchoring / coverage -----------------------------------------------------------------------

def test_chunk_coverage_and_embedding_parity():
    for length in (1, 5, 63, 64, 65, 127, 128, 129, 200, 513):
        for ctx, stride in ((128, 64), (64, 32), (256, 128)):
            starts = chunk_starts(length, ctx, stride)
            covered = np.zeros(length, dtype=bool)
            for s in starts:
                covered[s:min(s + ctx, length)] = True
            assert covered.all() and (length <= ctx or starts[-1] == length - ctx), (length, ctx, stride)
    loader = FakeLoader(lengths=[1, 70, 130, 200, 300], n_train=3, n_val=2)
    cfg = small_cfg()
    lookup, rms = loader.lookup("fam"), float(loader.rms("fam"))
    init = C.build_matched_init(16, 5, cfg, 111)
    for arm in ("context_id", "context_quantized"):
        model = C.build_arm_model(arm, 16, 5, cfg, 111, init, device=CPU)
        for name in loader.recording_names():
            ref, ref_zero = C.extract_vector_embeddings(arm, model, loader, "fam", name, lookup, rms, CPU)
            mine, zero = E.extract_embeddings(arm, model, loader, "fam", name, lookup, rms, CPU, 128, 64)
            assert np.array_equal(ref, mine) and np.array_equal(ref_zero, zero), (arm, name)
            for ctx, stride in ((64, 32), (256, 128)):
                emb, z = E.extract_embeddings(arm, model, loader, "fam", name, lookup, rms, CPU, ctx, stride)
                assert emb.shape == (len(loader.recs[name]["codes"]), 128) and not z.any()
    return True


# --- 6. replay -----------------------------------------------------------------------------------------------

def _hist(values: dict, last=5000, every=100, default=9.0):
    steps = range(every, last + 1, every)
    return [(s, values.get(s, default)) for s in steps]


def test_replay_patience_rule():
    r = T.replay_patience(_hist({100: 5.0, 500: 3.0}), 1000)
    assert (r["stop_step"], r["selected_step"]) == (1500, 500)  # boundary: step - best >= 1000
    r = T.replay_patience(_hist({100: 5.0, 500: 3.0, 1400: 2.9}), 1000)
    assert (r["stop_step"], r["selected_step"]) == (2400, 1400)  # later improvement before the stop resets patience
    r = T.replay_patience(_hist({100: 5.0, 500: 3.0, 600: 3.0}), 1000)
    assert r["selected_step"] == 500  # earliest tie kept
    r = T.replay_patience(_hist({100: 5.0, 4400: 2.0, 500: 3.0}), 1000)
    assert r["stop_step"] == 1500 and r["selected_step"] == 500  # best at 4400 comes after the stop
    r = T.replay_patience(_hist({100: 5.0, 900: 4.0, 1800: 3.0, 2700: 2.9, 3600: 2.8, 4400: 2.0}), 1000)
    assert r["stop_step"] is None and r["selected_step"] == 4400  # no stop: best through the endpoint
    pairs = _hist({100: 5.0, 500: 3.0, 3000: 2.0})
    assert T.replay_patience(pairs, 1000)["selected_step"] == 500 and T.full_history_best(pairs) == (3000, 2.0)
    loader = FakeLoader()
    with tempfile.TemporaryDirectory() as tmp:
        res = core(loader, "context_id", small_cfg(max_steps=30, eval_every=2), ckpt_dir=Path(tmp),
                   fixed_steps={"ckpt_1k": 8, "ckpt_3k": 16, "ckpt_5k": 30})
        scale = replace(P.SMOKE, fixed_steps={"ckpt_1k": 8, "ckpt_3k": 16, "ckpt_5k": 30}, patience=6)
        roles = T.checkpoint_roles(res, scale, 30)["roles"]
        for role, step in roles.items():
            assert (Path(tmp) / f"step{step:05d}.pt").exists(), (role, step)
        assert roles["final"] == 30 and roles["selected"] == res.best_step
    return True


# --- 7. positions -----------------------------------------------------------------------------------------------

def test_position_none_and_old_modes_unchanged():
    base = dict(num_codes=20, d_model=16, n_layers=2, n_heads=2, d_ff=32, dropout=0.0)
    codes = torch.randint(0, 21, (2, 10), generator=torch.Generator().manual_seed(1))
    pad = torch.zeros(2, 10, dtype=torch.bool)
    perm = torch.randperm(10, generator=torch.Generator().manual_seed(2))
    outs = {}
    for mode in ("none", "sinusoidal", "alibi"):
        torch.manual_seed(0)
        model = CodeContextModel(ModelConfig(position_mode=mode, **base)).eval()
        h, _ = model(codes, pad)
        h_perm, _ = model(codes[:, perm], pad)
        outs[mode] = torch.allclose(h[:, perm], h_perm, atol=1e-5)
        assert hasattr(model, "alibi_slopes") == (mode == "alibi")
    assert outs == {"none": True, "sinusoidal": False, "alibi": False}, outs
    torch.manual_seed(0)
    model = CodeContextModel(ModelConfig(position_mode="none", **base)).eval()
    pad2 = pad.clone()
    pad2[:, 8:] = True
    other = codes.clone()
    other[:, 8:] = 3
    h1, _ = model(codes, pad2)
    h2, _ = model(other, pad2)
    assert torch.allclose(h1[:, :8], h2[:, :8], atol=1e-5), "padded keys must not influence valid positions"
    # Old modes: the extended model must equal a from-scratch statement of the pre-`none` forward pass
    # (same process, same thread count, so no numerical-environment dependence).
    from script.action_transport.stageb_model import sinusoidal_positions

    def pre_none_forward(model, input_codes, padding_mask):
        cfg = model.cfg
        batch, length = input_codes.shape
        if cfg.position_mode == "sinusoidal":
            positions = sinusoidal_positions(length, cfg.d_model, input_codes.device)
            x = model.dropout(model.embedding(input_codes) + positions.unsqueeze(0))
            hidden = model.encoder(x, src_key_padding_mask=padding_mask)
        else:
            x = model.dropout(model.embedding(input_codes))
            idx = torch.arange(length, device=input_codes.device)
            bias = -model.alibi_slopes[:, None, None] * (idx[:, None] - idx[None, :]).abs().float()
            bias = bias.unsqueeze(0).expand(batch, -1, -1, -1).clone()
            bias = bias.masked_fill(padding_mask[:, None, None, :], float("-inf"))
            hidden = model._encode_with_bias(x, bias.reshape(batch * cfg.n_heads, length, length))
        return hidden, model.head(hidden)

    for mode in ("sinusoidal", "alibi"):
        for train_mode in (False, True):
            cfg = ModelConfig(num_codes=20, d_model=16, n_layers=2, n_heads=2, d_ff=32, dropout=0.3, position_mode=mode)
            torch.manual_seed(0)
            model = CodeContextModel(cfg)
            model.train(train_mode)
            ids = torch.randint(0, 21, (3, 10), generator=torch.Generator().manual_seed(5))
            pd = torch.zeros(3, 10, dtype=torch.bool)
            pd[:, 8:] = True
            torch.manual_seed(123)
            h1, l1 = model(ids, pd)
            torch.manual_seed(123)
            h2, l2 = pre_none_forward(model, ids, pd)
            assert torch.equal(h1, h2) and torch.equal(l1, l2), (mode, train_mode)
    try:
        ModelConfig(position_mode="bogus", **base)
        CodeContextModel(ModelConfig(position_mode="bogus", **base))
    except ValueError:
        pass
    else:
        raise AssertionError("unknown position modes must still be rejected")
    return True


# --- 8. identities, dedup, resume, invalid solves -----------------------------------------------------------------------

def test_eval_identity_and_dedup():
    loader = FakeLoader()
    job = P.enumerate_training_jobs()[0]
    names = loader.recording_names()
    kw = dict(job=job, scale=P.FULL, outer_cap=50, qic_hash="q", loader=loader, names=names)
    a = E.eval_identity(state_sha="s1", **kw)
    assert a == E.eval_identity(state_sha="s1", **kw)
    assert a != E.eval_identity(state_sha="s2", **kw)
    assert a != E.eval_identity(state_sha="s1", **{**kw, "outer_cap": 150})
    assert a != E.eval_identity(state_sha="s1", **{**kw, "job": P.enumerate_training_jobs()[2]})  # other context/stride
    assert a != E.eval_identity(state_sha="s1", **{**kw, "scale": P.SMOKE})  # different rungs
    assert a != E.eval_identity(state_sha="s1", **{**kw, "qic_hash": "other"})
    # equal losses / steps / names are never part of the identity
    assert T.tensor_hash({"w": torch.zeros(3)}) != T.tensor_hash({"w": torch.ones(3)})
    return True


def _bare_run(tmp: Path):
    from script.week5.contextualizer_calibration import run_calibration as R
    run = object.__new__(R.Run)
    run.args = type("A", (), dict(run_id="t", include_combined_target_match=False, continue_on_repro_mismatch=False))()
    run.scale, run.device, run.deadline, run.run_dir = P.SMOKE, CPU, None, tmp
    run.loader, run.jobs = FakeLoader(), P.enumerate_training_jobs()
    run.evals = P.enumerate_evaluations(run.jobs, P.SMOKE.outer_cap, P.SMOKE.outer_cap_sensitivity)
    run.job_by_id = {j.job_id: j for j in run.jobs}
    run.job_identity = {j.job_id: "id" for j in run.jobs}
    run.job_infos, run.timings, run.not_run, run.stop = {}, {}, {}, False
    run.verified_evals, run.repro_status, run.env = set(), "pending", {"device": "cpu"}
    run.checkpoint_check = {"vq": dict(match=True), "fsq": dict(match=True)}
    run.impl_hash, run.qic_hash, run.names = R.source_hash(), "q", run.loader.recording_names()
    return run


def test_interrupted_fit_restarts_and_deadline_is_explicit():
    from script.week5.contextualizer_calibration import run_calibration as R
    with tempfile.TemporaryDirectory() as tmp:
        run = _bare_run(Path(tmp))
        job = run.jobs[0]
        jd = run.job_dir(job)
        (jd / "ckpt").mkdir(parents=True)
        (jd / "ckpt" / "step00010.pt").write_bytes(b"partial")
        (jd / "train_started.json").write_text("{}")
        real = T.run_training_job
        T.run_training_job = lambda *a, **k: dict(job_id=job.job_id, status="done", peak_memory_bytes=None,
                                                  roles={}, identity="id")
        try:
            run.ensure_job(job)
        finally:
            T.run_training_job = real
        assert not (jd / "ckpt").exists() and (jd / "training_info.json").exists()
        line = json.loads((Path(tmp) / "restarts.jsonl").read_text().splitlines()[0])
        assert line["job_id"] == job.job_id and "restarted from initialization" in line["reason"]
        late = _bare_run(Path(tmp) / "late")
        late.deadline = __import__("time").perf_counter() - 1.0
        assert late.ensure_job(late.jobs[1]) is False
        assert late.not_run[late.jobs[1].job_id]["status"] == "deadline"
    return True


def test_invalid_solver_never_reaches_scoring_and_is_not_marked_done():
    with tempfile.TemporaryDirectory() as tmp:
        run = _bare_run(Path(tmp))
        job = run.jobs[0]
        model = CodeContextModel(C.model_config(16, "vector", 5, T.train_config(job, P.SMOKE)))
        state = T.cpu_state(model)
        jd = run.job_dir(job)
        T.save_state(jd / "ckpt" / "step00004.pt", state, dict(step=4))
        (jd / "checkpoints.json").write_text(json.dumps(dict(ledger=[dict(step=4, val_loss=1.0, file="step00004.pt",
                                                                          tensor_sha256=T.tensor_hash(state))])))
        run.job_infos[job.job_id] = dict(roles={"selected": 4})
        real = E.decode_state

        def boom(*a, **k):
            raise E.EvalInvalid("nonfinite")

        E.decode_state = boom
        try:
            ev = next(e for e in run.evals if e.job_id == job.job_id and e.ckpt_role == "selected")
            assert run.ensure_eval(ev) is False
        finally:
            E.decode_state = real
        rec = json.loads(run.record_path(ev).read_text())
        assert rec["status"] == "invalid" and run.not_run[ev.eval_id]["status"] == "invalid"
        assert not (Path(tmp) / "evals" / rec["identity"] / "ladder.json").exists()
    return True


def test_reproduction_gate_against_saved_task2_artifacts():
    """Exercise the gate with Task 2's own saved cells standing in for the new
    decodes: identical artifacts pass; any metric or selected-step change fails
    and stops dependent cells. Read-only on Task 2's results."""
    import shutil
    from script.week5.contextualizer_calibration import run_calibration as R
    try:
        task2 = R.resolve_source("task2", "task2_v1")
    except SystemExit:
        return True  # source run not present on this machine
    with tempfile.TemporaryDirectory() as tmp:
        for scenario in ("same", "metric_changed", "step_changed"):
            run = _bare_run(Path(tmp) / scenario)
            run.scale, run.task2_dir, run.env = P.FULL, task2, {}
            run.jobs = P.enumerate_training_jobs()
            run.evals = P.enumerate_evaluations(run.jobs)
            run.job_by_id = {j.job_id: j for j in run.jobs}
            for pipeline, (cell, _) in R.REPRODUCTION_TARGETS.items():
                src = task2 / "cells" / cell
                job = run.jobs[0] if pipeline == "native_vq" else run.jobs[1]
                info = json.loads((src / "info.json").read_text())
                state = torch.load(src / "checkpoint_best.pt", map_location="cpu")["state_dict"]
                run.job_infos[job.job_id] = dict(history=info["history"], selected_step=info["best_step"] + (
                    100 if scenario == "step_changed" and pipeline == "raw" else 0),
                    role_tensor_sha256={"selected": T.tensor_hash(state)})
                eval_dir = run.run_dir / "evals" / f"id_{pipeline}"
                eval_dir.mkdir(parents=True)
                ladder = json.loads((src / "ladder.json").read_text())
                if scenario == "metric_changed" and pipeline == "native_vq":
                    ladder["rows"][-1]["F1_50"] += 0.01
                (eval_dir / "ladder.json").write_text(json.dumps(ladder))
                shutil.copy(src / "embeddings.npz", eval_dir / "embeddings.npz")
                ev = next(e for e in run.evals if e.job_id == job.job_id and e.ckpt_role == "selected")
                rec = run.record_path(ev)
                rec.parent.mkdir(parents=True, exist_ok=True)
                rec.write_text(json.dumps(dict(identity=f"id_{pipeline}", status="done")))
            # artifact verification of records is covered by its own tests; here the comparison is under test
            run.verified_record = lambda ev: json.loads(run.record_path(ev).read_text())
            run.reproduction_gate()
            result = json.loads((run.run_dir / "reproduction.json").read_text())
            if scenario == "same":
                assert result["status"] == "passed" and not run.stop, result
                assert all(r["metrics_exact"] and r["checkpoint_tensors_equal"] and r["embeddings_max_abs_diff"] == 0.0
                           and r["declared_target_met"] for r in result["rows"])
                assert sorted(r["declared_target_f1_50"] for r in result["rows"]) == [36.08, 38.81]
            else:
                assert result["status"] == "failed" and run.stop, scenario
                assert (run.run_dir / "reproduction_mismatch_environment.json").exists()
    return True


def test_probe_code_path_on_tiny_synthetic_config():
    """The width-256 / 8-layer probe itself is NOT run here; this exercises its code path with a
    tiny synthetic configuration so a bug does not surface only on the user's launch."""
    from script.week5.contextualizer_calibration import run_calibration as R
    args = type("A", (), dict(probe_micro_chunks=None, run_id="t", qic_run_id="x"))()
    loader = FakeLoader(n_train=6, n_val=2)
    with tempfile.TemporaryDirectory() as tmp:
        for micro in (None, 2):
            args.probe_micro_chunks = micro
            res = R.run_probe(args, CPU, loader, dict(width=32, layers=2, heads=4, d_ff=64, context=64, stride=32,
                                                       batch=4, updates=12), Path(tmp))
            assert res["status"] == "ok" and res["accepted_updates"] == 12 and res["peak_memory_bytes"] is None
            assert res["first10_seconds_per_update"] > 0 and res["later_seconds_per_update"] > 0
            assert res["extraction_windows"] > 0 and (Path(tmp) / "probe_result.json").exists()
    return True


def make_valid_eval_dir(d: Path, loader, identity: str, scale, family="fam", states=10, width=4) -> list[str]:
    names = T.bounded_split(loader, scale.recording_cap)[0]
    d.mkdir(parents=True, exist_ok=True)
    emb, pred = {}, {}
    for n in names:
        windows = len(loader.recs[n]["codes"])
        emb[f"{n}__emb"] = np.full((windows, width), 0.5, dtype=np.float32)
        emb[f"{n}__zero"] = np.zeros(windows, dtype=bool)
        pred[f"{n}__frame_pred"] = np.zeros(loader.frame_count(n), dtype=np.int32)
        pred[f"{n}__window_states"] = np.zeros(windows, dtype=np.int32)
    np.savez_compressed(d / "embeddings.npz", **emb)
    np.savez_compressed(d / "predictions.npz", **pred)
    np.savez(d / "ot.npz", mu0=np.ones((states, width)), mu=np.ones((states, width)),
             objective_trace=np.array([1.0, 0.9]), outer_status="capped", outer_iterations=2)
    rows = [dict(rung_steps=r, MoF=1.0, Edit=1.0, F1_10=1.0, F1_25=1.0, F1_50=1.0, status_counts={"capped": 1},
                 seconds=0.1) for r in scale.rungs]
    (d / "ladder.json").write_text(json.dumps(dict(rows=rows)))
    (d / "info.json").write_text(json.dumps(dict(status="done", identity=identity, ot_status="capped", family=family,
                                                 n_scored_recordings=len(names), scored_recording_names=names)))
    return names


def test_eval_complete_validates_array_contents():
    scale = replace(P.SMOKE, recording_cap=None)
    loader = FakeLoader()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        names = make_valid_eval_dir(d, loader, "x", scale)
        assert E.eval_complete(d, "x", scale, loader)
        assert not E.eval_complete(d, "y", scale, loader)

        def corrupt(label, mutate):
            make_valid_eval_dir(d, loader, "x", scale)
            mutate()
            assert not E.eval_complete(d, "x", scale, loader), label

        def rewrite(fname, edit):
            with np.load(d / fname) as z:
                data = {k: z[k] for k in z.files}
            edit(data)
            np.savez_compressed(d / fname, **data)

        corrupt("garbage embeddings", lambda: (d / "embeddings.npz").write_bytes(b"not a zip" * 50))
        corrupt("garbage prototypes", lambda: (d / "ot.npz").write_bytes(b"not a zip" * 50))
        corrupt("nan embedding", lambda: rewrite("embeddings.npz", lambda z: z.update({f"{names[0]}__emb": np.full_like(z[f"{names[0]}__emb"], np.nan)})))
        corrupt("embedding length", lambda: rewrite("embeddings.npz", lambda z: z.update({f"{names[0]}__emb": z[f"{names[0]}__emb"][:-1], f"{names[0]}__zero": z[f"{names[0]}__zero"][:-1]})))
        corrupt("missing embedding key", lambda: rewrite("embeddings.npz", lambda z: z.pop(f"{names[0]}__zero")))
        corrupt("nan prototypes", lambda: rewrite("ot.npz", lambda z: z.update(mu=np.full((10, 4), np.nan))))
        corrupt("prototype shape", lambda: rewrite("ot.npz", lambda z: z.update(mu=np.ones((9, 4)), mu0=np.ones((9, 4)))))
        corrupt("nan predictions", lambda: rewrite("predictions.npz", lambda z: z.update({f"{names[0]}__frame_pred": np.full(len(z[f"{names[0]}__frame_pred"]), np.nan)})))
        corrupt("state out of range", lambda: rewrite("predictions.npz", lambda z: z.update({f"{names[0]}__window_states": np.full(len(z[f"{names[0]}__window_states"]), 10, dtype=np.int32)})))
        corrupt("negative frame", lambda: rewrite("predictions.npz", lambda z: z.update({f"{names[0]}__frame_pred": np.full(len(z[f"{names[0]}__frame_pred"]), -1, dtype=np.int32)})))
    return True


def test_stale_completion_records_are_removed_and_not_aggregated():
    scale = replace(P.SMOKE, recording_cap=None)
    with tempfile.TemporaryDirectory() as tmp:
        run = _bare_run(Path(tmp))
        run.scale = scale
        ev = run.evals[0]
        identity = "stale"
        make_valid_eval_dir(run.run_dir / "evals" / identity, run.loader, identity, scale)
        run.record_path(ev).parent.mkdir(parents=True, exist_ok=True)
        run.record_path(ev).write_text(json.dumps(dict(eval_id=ev.eval_id, job_id=ev.job_id, ckpt_role=ev.ckpt_role,
                                                       step=4, identity=identity, status="done", state_sha="s",
                                                       aliased_of=None)))
        run.write_aggregates()
        assert not run.record_path(ev).exists(), "completion record without a verified job must be removed"
        assert (run.run_dir / "segmentation.csv").read_text() == "" and (run.run_dir / "training.csv").read_text() == ""
    return True


def test_failed_training_is_not_retried_and_dependents_are_unavailable():
    with tempfile.TemporaryDirectory() as tmp:
        run = _bare_run(Path(tmp))
        job = run.jobs[2]
        calls = {"n": 0}
        real = T.run_training_job

        def boom(*a, **k):
            calls["n"] += 1
            raise RuntimeError("late failure")

        T.run_training_job = boom
        try:
            evals = [e for e in run.evals if e.job_id == job.job_id]
            assert len(evals) >= 5
            assert not any(run.ensure_eval(e) for e in evals)
        finally:
            T.run_training_job = real
        assert calls["n"] == 1
        assert run.not_run[job.job_id]["status"] == "failed"
        assert all(run.not_run[e.eval_id]["status"] == "unavailable" and "late failure" in run.not_run[e.eval_id]["reason"]
                   for e in evals)
    return True


def test_gate_survives_invalid_reference_decode_and_finalizes():
    with tempfile.TemporaryDirectory() as tmp:
        run = _bare_run(Path(tmp))
        run.scale = P.FULL
        run.evals = P.enumerate_evaluations(run.jobs)
        run.task2_dir = Path(tmp)
        for job in run.jobs[:2]:
            ev = next(e for e in run.evals if e.job_id == job.job_id and e.ckpt_role == "selected")
            run.job_infos[job.job_id] = dict(history=[], selected_step=1, role_tensor_sha256={"selected": "x"})
            run.record_path(ev).parent.mkdir(parents=True, exist_ok=True)
            run.record_path(ev).write_text(json.dumps(dict(status="invalid", identity="gone", reason="nonfinite")))
        run.reproduction_gate()
        result = json.loads((run.run_dir / "reproduction.json").read_text())
        assert result["status"] == "failed" and run.stop and all(not r["passed"] for r in result["rows"])
        run.job_infos.clear()
        run.write_aggregates()  # final bookkeeping still works
        assert (run.run_dir / "reproduction.csv").exists() and (run.run_dir / "not_run.csv").exists()
    return True


def test_environment_and_checkpoints_gate_resume_and_start():
    from script.week5.contextualizer_calibration import run_calibration as R
    with tempfile.TemporaryDirectory() as tmp:
        run = _bare_run(Path(tmp))
        run.qic_hash = run.task2_hash = "h"
        run.train_names, run.val_names = ["a"], ["b"]
        run.args.include_combined_target_match = False
        first = run.resume_identity()
        run.env = {"device": "cuda", "gpu": "x"}
        assert run.resume_identity() != first, "switching device/environment must change the resume identity"
        run.env = {"device": "cpu"}
        assert run.resume_identity() == first
        run.require_sources()
        run.checkpoint_check = {"vq": dict(match=False), "fsq": dict(match=True)}
        try:
            run.require_sources()
        except SystemExit as exc:
            assert "vq" in str(exc)
        else:
            raise AssertionError("a checkpoint hash mismatch must block execution")
    return True


def test_eval_complete_rejects_mismatches():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        assert not E.eval_complete(d, "x", P.SMOKE)
        for n in ("info.json", "ladder.json", "ot.npz", "predictions.npz", "embeddings.npz"):
            (d / n).write_text("{}")
        assert not E.eval_complete(d, "x", P.SMOKE)  # present but not valid
    return True


# --- 9. search declaration ---------------------------------------------------------------------------------------

def test_search_declaration():
    assert len(S.all_definitions()) == S.NOMINAL_COUNT == 10_368
    assert S.seconds_to_tokens(2, 60, 15) == 8 and S.seconds_to_tokens(0.25, 60, 15) == 1
    assert S.seconds_to_tokens(4, 60, 15) == 16 and S.seconds_to_tokens(32, 60, 15) == 128
    assert S.seconds_to_tokens(0.3, 60, 15) == 2  # ceiling, not rounding
    assert S.guard_min_tokens(14, 15) == 5 and S.guard_min_tokens(0, 15) == 1
    d1 = S.declare({"native_vq": 14, "raw": 0}, 60, 15)
    d2 = S.declare({"native_vq": 14, "raw": 0}, 60, 15)
    assert json.dumps(d1, sort_keys=True, default=str) == json.dumps(d2, sort_keys=True, default=str)
    for name, spans_ok in (("native_vq", lambda t: t >= 5), ("raw", lambda t: t >= 1)):
        entry = d1["pipelines"][name]
        cands = entry["candidates"]
        assert entry["complete"] and len(cands) == 12 and cands[0]["source"] == "frozen_reference"
        assert cands[0]["context_tokens"] == 128 and cands[0]["span_tokens"] == 8 and cands[0]["position"] == "alibi"
        assert all(spans_ok(c["span_tokens"]) for c in cands)
        assert len({S.effective_key(c) for c in cands}) == 12
    assert any(c["span_tokens"] in (1, 4) for c in d1["pipelines"]["raw"]["candidates"]) or \
        any(r["reason"] == "span_below_guard_minimum" for r in d1["rejections"])
    reasons = {r["reason"] for r in d1["rejections"]}
    assert "span_below_guard_minimum" in reasons and all(r["pipeline"] in ("native_vq", "raw") for r in d1["rejections"])
    tree = ast.parse((Path(S.__file__)).read_text())
    imported = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
        {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any("labels" in m for m in imported), "search declaration must be label-free"
    loader = FakeLoader(n_train=6, n_val=3)
    audited = S.declare({"native_vq": 14}, 60, 15, loader, {"native_vq": "fam"}, loader.recording_names()[:6],
                        loader.recording_names()[6:])
    assert audited["pipelines"]["native_vq"]["candidates"][1]["audit"]["train"]["eligible"] > 0
    return True


def test_label_free_training_modules():
    for module in (T, S):
        tree = ast.parse(Path(module.__file__).read_text())
        names = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
            {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not any(m.endswith("labels") or m.endswith("evaluate") and "action_transport" in m for m in names), module.__name__
    return True


ALL_TESTS = [
    test_plan_counts_and_ids,
    test_effective_settings_identity,
    test_model_config_parity_with_task2,
    test_training_parity_with_task2_trajectory,
    test_checkpoint_logging_leaves_training_unchanged,
    test_default_mask_path_matches_task2_mask_construction,
    test_schedule_horizons_are_real,
    test_weighted_accumulation_matches_monolithic_batch,
    test_empty_target_batches_do_not_advance_and_failure_is_bounded,
    test_guard_radius_zero_vs_fourteen_and_span_eligibility,
    test_matched_target_counts_retries_and_isolated_rng,
    test_chunk_coverage_and_embedding_parity,
    test_replay_patience_rule,
    test_position_none_and_old_modes_unchanged,
    test_eval_identity_and_dedup,
    test_interrupted_fit_restarts_and_deadline_is_explicit,
    test_invalid_solver_never_reaches_scoring_and_is_not_marked_done,
    test_eval_complete_rejects_mismatches,
    test_eval_complete_validates_array_contents,
    test_stale_completion_records_are_removed_and_not_aggregated,
    test_failed_training_is_not_retried_and_dependents_are_unavailable,
    test_gate_survives_invalid_reference_decode_and_finalizes,
    test_environment_and_checkpoints_gate_resume_and_start,
    test_reproduction_gate_against_saved_task2_artifacts,
    test_probe_code_path_on_tiny_synthetic_config,
    test_search_declaration,
    test_label_free_training_modules,
]


def run_all() -> bool:
    ok = True
    for fn in ALL_TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
        except Exception as exc:  # noqa: BLE001
            ok = False
            import traceback
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}", flush=True)
    print(f"{'ALL PASSED' if ok else 'SOME FAILED'} ({len(ALL_TESTS)} tests)")
    return ok


# --- smoke-run verification -----------------------------------------------------------------------------------------

def verify_run_dir(run_dir: Path, jobs, evals) -> bool:
    """A finished (smoke) run: plans saved with the right counts, every unit is
    either done or explicit in not_run.csv, no duplicate rows, and every
    realized evaluation's saved predictions rescore to its saved final-rung
    metrics."""
    from script.week5.contextualizer_calibration import report
    run_dir, problems = Path(run_dir), []
    if len(report.read(run_dir / "planned_training.csv")) != len(jobs):
        problems.append("planned_training.csv row count")
    if len(report.read(run_dir / "planned_evaluations.csv")) != len(evals):
        problems.append("planned_evaluations.csv row count")
    not_run = {r["unit"] for r in report.read(run_dir / "not_run.csv")}
    training = {r["job_id"] for r in report.read(run_dir / "training.csv")}
    for j in jobs:
        if j.job_id not in training and j.job_id not in not_run:
            problems.append(f"job {j.job_id} neither done nor explained")
    seg = report.read(run_dir / "segmentation.csv")
    keys = [(r["eval_id"], r["rung_steps"]) for r in seg]
    if len(keys) != len(set(keys)):
        problems.append("duplicate segmentation rows")
    done_evals = {r["eval_id"] for r in seg}
    for e in evals:
        if e.eval_id not in done_evals and e.eval_id not in not_run:
            problems.append(f"evaluation {e.eval_id} neither done nor explained")
    checked = 0
    for identity in sorted({r["identity"] for r in seg}):
        eval_dir = run_dir / "evals" / identity
        ladder = json.loads((eval_dir / "ladder.json").read_text())["rows"]
        last = max(ladder, key=lambda r: r["rung_steps"])
        metrics = E.rescore_predictions(eval_dir)
        for k_csv, k_json in (("MoF", "MoF"), ("Edit", "Edit"), ("F1@10", "F1_10"), ("F1@25", "F1_25"), ("F1@50", "F1_50")):
            if abs(float(metrics[k_csv]) - float(last[k_json])) > 1e-9:
                problems.append(f"{identity} {k_csv} rescoring {metrics[k_csv]} != saved {last[k_json]}")
        checked += 1
    result = dict(ok=not problems, problems=problems, rescored_evaluations=checked, jobs_done=len(training),
                  evals_done=len(done_evals), unfinished=sorted(not_run))
    (run_dir / "smoke_verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return not problems


if __name__ == "__main__":
    raise SystemExit(0 if run_all() else 1)
