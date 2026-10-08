"""Phase 9 preflight gates (PHASE9_LEARNED_UNITS.md, Revision 3). Every gate is a function of the runner `r` (which
carries the loaders, scale, device and cell class), so this module never imports `run_phase9` (run with `-m` that would
load a second copy of it).

  G9a  the copied trainer's default path retrains BABEL-2 seed 111 (T2, span 2, `context_id`) and reproduces the saved
       Phase 5 job (selected step, validation history, checkpoint tensors) and, once decoded, its saved ladder exactly
  G9b  masked inputs never reach the model; a visible window's continuous vector changes I1's outputs and not I1q's
  G9c  the extractor equals the Phase 5 extractor (code input), the independent continuous / quantized extractors, and
       the per-layer states are well formed and end at the final layer
  G9d  the cross-seed agreement measure is permutation invariant and ~0 for independent codes
  G9e  the I2 family builder with T2's own codes reproduces T2's grouping and per-code centres
  G9g  training chunks, batch order and masks are identical across arms; matched initialization of backbone and head
  (G9f, the dimensional contract, is checked per I2 family when it is built.)

Labels are never read.
"""

from __future__ import annotations

import copy
import json
import shutil
from dataclasses import replace

import numpy as np
import torch

from script.action_transport import categorical
from script.fsq_reconstruction.evaluate import per_code_pca_centers
from script.fsq_reconstruction.experiment import write_json_atomic
from script.quantized_input_comparison import contextual as C
from script.quantized_input_comparison.artifacts import Loader
from script.week5.contextualizer_calibration import evaluate as E
from script.week5.contextualizer_calibration import plan as P
from script.week5.contextualizer_calibration import train as T
from script.week5.timescale_inspection import run_phase5 as R5
from script.week5.timescale_inspection import run_phase7 as R7
from script.week5.timescale_inspection import tokenizers as TOK
from script.week5.timescale_inspection import trainer9 as T9
from script.week5.timescale_inspection import units9 as U9


def _status(checks: dict, **extra) -> dict:
    return dict(status="passed" if all(bool(v) for v in checks.values()) else "failed",
                checks={k: bool(v) for k, v in checks.items()}, **extra)


def _setup(r):
    cell = r.Cell9("babel2", "i1", 111)
    loader = r.base_loader(cell)
    if loader is None:
        raise RuntimeError("the BABEL-2 seed-111 T2 preparation is not available")
    return cell, loader, cell.spec.family


def g9a(r) -> dict:
    ref_job = r.phase5_dir / "jobs" / "t2_relational" / "babel2__t2_relational__seed111__span2"
    ref_unit = r.phase5_dir / "units" / "babel2__t2_relational__seed111__span2__context" / "info.json"
    ref_prep = r.phase5_dir / "prep" / "t2_relational" / "babel2__t2_relational__seed111"
    for p in (ref_job / "training_info.json", ref_unit, ref_prep):
        if not p.exists():
            return dict(status="failed", reason=f"reference not found: {p}")
    ref, saved = json.loads((ref_job / "training_info.json").read_text()), json.loads(ref_unit.read_text())
    loader = Loader(ref_prep)
    job = R5.make_job(R5.mk_cell("babel2", "t2_relational", 111, span=2))
    scratch = r.out / "gates" / "g9a_job"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    info = T9.run_training_job9(job, P.FULL, loader, scratch, r.device, r.deadline, "g9a", r.impl)
    checks = dict(selected_step=info["roles"]["selected"] == ref["roles"]["selected"],
                  history=[h["val_loss"] for h in info["history"]] == [h["val_loss"] for h in ref["history"]],
                  tensors=info["role_tensor_sha256"]["selected"] == ref["role_tensor_sha256"]["selected"])
    state, _ = r._state_of(scratch, info)
    names = sorted(loader.recording_names())
    z, zero = r.embeddings(job, loader, state, names)
    spec = TOK.dataset_spec("babel2")
    res = r.run_ot(spec, names, loader, z, zero, loader, loader.grouping(TOK.SPECS["t2_relational"].family))
    same, maxdiff = R7.ladders_equal(res["ladder"], saved["ladder"])
    checks["ladder"] = same
    shutil.rmtree(scratch / "ckpt", ignore_errors=True)
    return _status(checks, selected_step=info["roles"]["selected"], reference_step=ref["roles"]["selected"],
                   ladder_max_abs_diff=maxdiff, seconds=info["seconds"])


def _chunks(loader, fam, names, n):
    return C.make_vector_chunks(loader, fam, names[:3], chunk=128, stride=64)[:n]


def g9b(r) -> dict:
    cell, loader, fam = _setup(r)
    dev = r.device
    lookup, rms = loader.lookup(fam), float(loader.rms(fam))
    _, _, val_names = T.bounded_split(loader, 12)
    chunks = _chunks(loader, fam, val_names, 4)
    masks = C.chunk_masks(chunks, 111, 0.30, 2, radius=0)
    job = r.make_job(cell)
    cfg = T.train_config(job, P.SMOKE)
    init = C.build_matched_init(lookup.shape[0], lookup.shape[1], cfg, seed=111)
    checks, detail = {}, {}
    bumped = []
    for c in chunks:
        c2 = copy.copy(c)
        c2.continuous = c.continuous + 3.0
        bumped.append(c2)
    for arm in ("context_continuous", "context_quantized"):
        model = C.build_arm_model(arm, lookup.shape[0], lookup.shape[1], cfg, 111, init, device=dev).eval()
        fn = lambda c, arm=arm: T9.feature_fn(arm, lookup, c)  # noqa: E731
        vec, tgt, w, elig, mbool, pad = C._encode_vector(chunks, masks, fn, rms, dev)
        with torch.no_grad():
            h0, l0 = model.forward_vectors(vec, mbool, pad)
            v_masked = vec.clone()
            v_masked[mbool] += 5.0
            h1, l1 = model.forward_vectors(v_masked, mbool, pad)
            vec2 = C._encode_vector(bumped, masks, fn, rms, dev)[0]
            h2, _ = model.forward_vectors(vec2, mbool, pad)
        keep = ~pad
        checks[f"{arm}_masked_input_ignored"] = torch.equal(h0, h1) and torch.equal(l0, l1)
        changed = bool((h0[keep & ~mbool] != h2[keep & ~mbool]).any())
        detail[f"{arm}_visible_change_max_abs"] = float((h0 - h2)[keep].abs().max())
        checks[f"{arm}_visible_vector_effect"] = changed if arm == "context_continuous" else (not changed)
    # training level: the loss history responds to a visible continuous change in I1 and not in I1q
    names, tr, va = T.bounded_split(loader, 12)
    train_chunks, val_chunks = C.make_vector_chunks(loader, fam, tr[:6], 128, 64)[:24], C.make_vector_chunks(loader, fam, va[:3], 128, 64)[:8]
    bump = lambda cs: [_with(c, continuous=c.continuous + 3.0) for c in cs]  # noqa: E731
    for arm in ("context_continuous", "context_quantized"):
        hist = []
        for tc, vc in ((train_chunks, val_chunks), (bump(train_chunks), bump(val_chunks))):
            res = T9.train_core(arm=arm, train_chunks=tc, val_chunks=vc, num_codes=lookup.shape[0], input_dim=lookup.shape[1],
                                seed=111, cfg=cfg, init=init, device=dev, lookup=lookup if arm == "context_quantized" else None,
                                rms=rms, guard_radius=0, fixed_steps=P.SMOKE.fixed_steps, ckpt_dir=None, deadline=None)
            hist.append([h["val_loss"] for h in res.history])
        gap = max(abs(a - b) for a, b in zip(*hist))
        detail[f"{arm}_training_history_gap"] = gap
        checks[f"{arm}_training_effect"] = gap > 1e-3 if arm == "context_continuous" else gap <= 1e-4
    return _status(checks, **detail)


def _with(chunk, **kw):
    c2 = copy.copy(chunk)
    for k, v in kw.items():
        setattr(c2, k, v)
    return c2


def g9c(r) -> dict:
    cell, loader, fam = _setup(r)
    dev = r.device
    lookup, rms = loader.lookup(fam), float(loader.rms(fam))
    job = r.make_job(cell)
    cfg = T.train_config(job, P.SMOKE)
    init = C.build_matched_init(lookup.shape[0], lookup.shape[1], cfg, seed=111)
    names = sorted(loader.recording_names())[:3]
    checks, detail = {}, {}

    def diffs(model, arm, old):
        worst = 0.0
        for n in names:
            u0, z0 = old(n)
            u1, z1 = T9.extract_embeddings9(arm, model, loader, fam, n, lookup, rms, dev, 128, 64)
            worst = max(worst, float(np.abs(u0 - u1).max()), float(np.abs(z0.astype(float) - z1.astype(float)).max()))
        return worst

    m_id = C.build_arm_model("context_id", lookup.shape[0], lookup.shape[1], cfg, 111, init, device=dev).eval()
    d = diffs(m_id, "context_id", lambda n: E.extract_embeddings("context_id", m_id, loader, fam, n, lookup, rms, dev, 128, 64))
    checks["code_input_equals_phase5_extractor"], detail["code_input_max_abs"] = d == 0.0, d
    m_q = C.build_arm_model("context_quantized", lookup.shape[0], lookup.shape[1], cfg, 111, init, device=dev).eval()
    d = diffs(m_q, "context_quantized", lambda n: E.extract_embeddings("context_quantized", m_q, loader, fam, n, lookup, rms, dev, 128, 64))
    checks["centroid_input_equals_phase5_extractor"], detail["centroid_input_max_abs"] = d == 0.0, d
    m_c = C.build_arm_model("context_continuous", lookup.shape[0], lookup.shape[1], cfg, 111, init, device=dev).eval()
    d = diffs(m_c, "context_continuous", lambda n: C.extract_vector_embeddings("context_continuous", m_c, loader, fam, n, lookup, rms, dev))
    checks["continuous_input_equals_independent_extractor"], detail["continuous_input_max_abs"] = d == 0.0, d
    # (iii) per-layer states: shapes, finiteness, and the last layer is the final hidden state
    ok_shape = ok_finite = ok_last = True
    for n in names:
        unit, zero, states = T9.extract_embeddings9("context_continuous", m_c, loader, fam, n, lookup, rms, dev, 128, 64, layers=True)
        L = len(loader.recording(fam, n)["codes"])
        ok_shape &= len(states) == 4 and all(s.shape == (L, 128) for s in states)
        ok_finite &= all(np.isfinite(s).all() for s in states)
        last = states[-1].astype(np.float64)
        norms = np.linalg.norm(last, axis=1, keepdims=True)
        redo = (last / np.where(norms <= 1e-12, 1.0, norms)).astype(np.float32)
        ok_last &= float(np.abs(redo - unit).max()) <= 1e-6
    checks.update(layers_shapes=ok_shape, layers_finite=ok_finite, layer4_normalizes_to_extractor_output=ok_last)
    return _status(checks, **detail)


def g9d(r) -> dict:
    rng = np.random.default_rng(0)
    # enough windows that the finite-sample bias of NMI (about cells / 2N) is negligible: 20 x 20 cells, 40,000 windows
    a = {f"r{i}": rng.integers(0, 20, 8000) for i in range(5)}
    perm = rng.permutation(20)
    relabelled = {n: perm[c] for n, c in a.items()}
    indep = {n: rng.integers(0, 20, len(c)) for n, c in a.items()}
    same = U9.cross_seed_nmi({1: a, 2: relabelled})
    rand = U9.cross_seed_nmi({1: a, 2: indep})
    chance = U9.chance_nmi({1: a, 2: relabelled})  # identical codes, one side shuffled across windows: ~0
    checks = dict(permutation_invariant=abs(same - 1.0) < 1e-9, independent_near_zero=abs(rand) < 0.02, shuffled_near_zero=abs(chance) < 0.02)
    return _status(checks, relabelled=same, independent=rand, shuffled=chance)


def g9e(r) -> dict:
    cell, base, fam = _setup(r)
    spec = TOK.dataset_spec("babel2")
    names = sorted(base.recording_names())
    codes = {n: base.recording(fam, n)["codes"] for n in names}
    src = U9.build_family(base, fam, codes, spec, 111)
    recs = [categorical.RecordingInput(n, np.asarray(codes[n], dtype=np.int32), base.recording(fam, n)["lengths"].astype(np.int32),
                                       spec.fps, spec.window) for n in names]
    centers, counts = per_code_pca_centers(recs, {n: base.recording(fam, n)["continuous"] for n in names}, U9.NUM_CODES)
    checks = dict(grouping_equals_t2=bool(np.array_equal(src.grouping(fam), base.grouping(fam))),
                  lookup_equals_per_code_centres=bool(np.array_equal(src.lookup(fam), np.asarray(centers, dtype=np.float32))),
                  dims=U9.check_family_dims(src, fam)["status"] == "passed")
    return _status(checks)


def batch_plan(chunks, seed, cfg, n=50):
    """The epoch-0 batch order and masks exactly as `train_core` draws them (same RNG stream)."""
    rng = np.random.default_rng(seed * 1_000_003 + 0)
    order = rng.permutation(len(chunks))
    out = []
    for b in range(min(n, int(np.ceil(len(chunks) / cfg.batch_size)))):
        idx = order[b * cfg.batch_size:(b + 1) * cfg.batch_size]
        batch = [chunks[i] for i in idx]
        seeds = [int(rng.integers(0, 2**31 - 1)) for _ in batch]
        masks = [T._build(c, np.random.default_rng(s), 0, cfg.mask_fraction, cfg.mask_span) for c, s in zip(batch, seeds)]
        out.append((idx.tolist(), [m[0] for m in masks]))
    return out


def g9g(r) -> dict:
    cell, loader, fam = _setup(r)
    job = r.make_job(cell)
    cfg = T.train_config(job, r.scale)
    _, train_names, _ = T.bounded_split(loader, r.scale.recording_cap)
    lookup = loader.lookup(fam)
    chunks = C.make_vector_chunks(loader, fam, train_names, 128, 64)
    rng = np.random.default_rng(1)
    names = sorted(loader.recording_names())
    other = U9.I2Source(loader, fam, {n: rng.integers(0, U9.NUM_CODES, len(loader.recording(fam, n)["codes"])) for n in names},
                        lookup, loader.grouping(fam))
    chunks2 = C.make_vector_chunks(other, fam, train_names, 128, 64)
    a, b = batch_plan(chunks, 111, cfg), batch_plan(chunks2, 111, cfg)
    same_plan = len(a) == len(b) and all(x[0] == y[0] and all(np.array_equal(p, q) for p, q in zip(x[1], y[1])) for x, y in zip(a, b))
    rms = float(loader.rms(fam))
    init = C.build_matched_init(lookup.shape[0], lookup.shape[1], cfg, seed=111)
    states = {}
    for arm in ("context_geometry_init", "context_quantized", "context_continuous"):
        m = C.build_arm_model(arm, lookup.shape[0], lookup.shape[1], cfg, 111, init,
                              lookup_over_rms=(lookup / rms) if arm == "context_geometry_init" else None)
        states[arm] = {k: v for k, v in m.state_dict().items() if not k.startswith(("embedding", "input_proj", "mask_token"))}
    ref = states["context_continuous"]
    same_init = all(set(s) == set(ref) and all(torch.equal(s[k], ref[k]) for k in ref) for s in states.values())
    same_proj = torch.equal(C.build_arm_model("context_quantized", lookup.shape[0], lookup.shape[1], cfg, 111, init).input_proj.weight,
                            C.build_arm_model("context_continuous", lookup.shape[0], lookup.shape[1], cfg, 111, init).input_proj.weight)
    return _status(dict(batch_plan_identical=same_plan, backbone_and_head_init_identical=same_init, projection_init_identical=same_proj),
                   batches_compared=len(a))


def run_gates(r) -> None:
    r.inherit_phase5_gate("g3")
    if r.smoke:
        r.gates["g9a"] = dict(status="not_applicable_smoke", implementation_hash=r.impl)
    else:
        r._gate("g9a", lambda: g9a(r))
    for name, fn in (("g9b", g9b), ("g9c", g9c), ("g9d", g9d), ("g9e", g9e), ("g9g", g9g)):
        r._gate(name, lambda fn=fn: fn(r))
    write_json_atomic(r.gates_path, r.gates)
