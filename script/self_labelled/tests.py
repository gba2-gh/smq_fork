"""Tests for the self-labelled embedding test. python -m script.self_labelled.tests (a pass writes verification.json)."""
from __future__ import annotations

import json
import sys
import time
import traceback

import numpy as np
import torch

from script.action_transport import transport
from script.action_transport.transport import Coeffs
from script.self_labelled import augment as AU
from script.self_labelled import batched_ot as OT
from script.self_labelled import config as C
from script.self_labelled import model as M
from script.self_labelled import run as R
from script.self_labelled import train as TR

TESTS = {}
CPU = torch.device("cpu")
COEFFS = Coeffs(a=0.7, beta=0.3, lam=0.05, eps=0.07)


def test(fn):
    TESTS[fn.__name__] = fn
    return fn


def skeleton(T=40, seed=0):
    return np.random.default_rng(seed).normal(size=(T, 25, 3)) * 300


def bone(p, a, b):
    return np.linalg.norm(p[:, a] - p[:, b], axis=-1)


@test
def augmentation_geometry():
    R_ = AU.rotation_z(37.0)
    assert abs(np.linalg.det(R_) - 1) < 1e-12 and np.allclose(R_ @ R_.T, np.eye(3)) and np.allclose(R_ @ np.array([0, 0, 1.0]), [0, 0, 1.0])
    p = skeleton()
    m = AU.mirror(p)
    assert m.shape == p.shape
    for a, b in AU.LR_PAIRS:                                     # a mirrored body: left bones become the old right bones
        assert np.allclose(m[:, a], p[:, b] * np.array([-1, 1, 1]))
    assert np.allclose(bone(m, 12, 13), bone(p, 16, 17)) and np.allclose(bone(m, 16, 17), bone(p, 12, 13))
    assert np.allclose(bone(m, 4, 5), bone(p, 8, 9))
    assert np.array_equal(AU.SWAP[AU.SWAP], np.arange(25))
    s = AU.apply(p, 90.0, False, 1.2)
    assert np.allclose(bone(s, 0, 20), 1.2 * bone(p, 0, 20)) and np.allclose(bone(s, 12, 13), 1.2 * bone(p, 12, 13))
    assert AU.apply(p, 0.0, True, 0.9).shape[0] == p.shape[0]       # time is preserved
    d1, d2 = AU.draw_params("a.npy", 3), AU.draw_params("a.npy", 3)
    assert d1 == d2 and AU.draw_params("a.npy", 4) != d1 and AU.draw_params("b.npy", 3) != d1
    assert 0 <= d1["angle"] < 360 and C.SCALE_RANGE[0] <= d1["scale"] <= C.SCALE_RANGE[1]
    draws = [AU.draw_params("x.npy", v) for v in range(2000)]
    assert 0.4 < np.mean([d["do_mirror"] for d in draws]) < 0.6


@test
def augmented_features_align_with_canonical_windows():
    from script.pretrained_encoder import features as F
    ntu = skeleton(70, 1)
    canon = F.window_features(F.frame_features(ntu, CPU), C.WINDOW)
    p = AU.draw_params("t.npy", 0)
    var = F.window_features(F.frame_features(AU.apply(ntu, **p), CPU, deg=p["angle"]), C.WINDOW)
    assert canon.shape == var.shape == (9, 3 * 512) and np.isfinite(var).all()
    assert not np.allclose(canon, var)


def make_batch(sizes, seed=0, frames_last_short=True):
    rng = np.random.default_rng(seed)
    frames = [[C.WINDOW] * (n - 1) + [int(rng.integers(1, C.WINDOW + 1))] for n in sizes]
    return OT.Batch(sizes, frames, C.FPS, C.WINDOW, 4, CPU), frames


@test
def batched_ot_reproduces_the_existing_solver_for_one_recording():
    b, frames = make_batch([23])
    rng = np.random.default_rng(1)
    cost = torch.as_tensor(rng.random((23, 4)), dtype=torch.float64)
    log_t, n, ok = OT.solve_inner(None, cost, b, COEFFS, 12)
    p = torch.as_tensor(np.array(frames[0]) / float(sum(frames[0])), dtype=torch.float64)
    q = torch.full((4,), 0.25, dtype=torch.float64)
    w = transport.build_kernel(23, transport.kernel_half_width(C.FPS, C.WINDOW, 23), CPU)
    ref = transport.solve_inner(transport.cold_start_log(p, q), cost, w, p, q, COEFFS, 12)
    assert ok and n == 12 and ref.status == "ok" and torch.allclose(log_t, ref.log_t, atol=1e-9)
    assert torch.allclose(OT.objective(log_t, cost, b, COEFFS), transport.objective_log(log_t, cost, w, 0.7, 0.3, 0.05, 0.07, q), atol=1e-10)
    t = torch.exp(log_t)
    assert torch.allclose(t.sum(1), b.p, atol=1e-9)


@test
def batched_ot_blocks_do_not_leak_and_decouple():
    b, frames = make_batch([14, 9, 17], seed=2)
    rng = np.random.default_rng(3)
    cost = torch.as_tensor(rng.random((b.n, 4)), dtype=torch.float64)
    log_t, _, _ = OT.solve_inner(None, cost, b, COEFFS, 6)
    # a perturbation of one recording changes the structural gradient only inside that recording's rows
    t = torch.exp(log_t)
    g0 = OT.gradient(log_t, cost, b, Coeffs(0.7, 0.3, 0.0, 0.07))
    t2 = t.clone()
    t2[:14] *= 1.7
    g1 = OT.gradient(torch.log(t2), cost, b, Coeffs(0.7, 0.3, 0.0, 0.07))
    center = lambda g: g - g.mean(dim=1, keepdim=True)
    # (the entropy term changes the perturbed rows only; compare after removing it)
    ent = lambda lt: 0.07 * lt
    d = center((g1 - ent(torch.log(t2))) - (g0 - ent(log_t)))
    assert float(d[14:].abs().max()) < 1e-12 and float(d[:14].abs().max()) > 1e-6
    # without the column marginal, or with it applied per recording, the batch problem equals the per-recording problems
    for per_rec, lam in ((False, 0.0), (True, 0.05)):
        cf = Coeffs(0.7, 0.3, lam, 0.07)
        lt, _, ok = OT.solve_inner(None, cost, b, cf, 400, per_recording=per_rec)
        off = 0
        for i, n in enumerate(b.sizes):
            single = OT.Batch([n], [frames[i]], C.FPS, C.WINDOW, 4, CPU)
            ls, _, _ = OT.solve_inner(None, cost[off:off + n], single, cf, 400)
            share = float(sum(frames[i])) / float(sum(sum(f) for f in frames))
            a_ = torch.exp(lt[off:off + n]) / share
            b_ = torch.exp(ls)
            assert torch.allclose(a_, b_, atol=2e-4), (per_rec, i, float((a_ - b_).abs().max()))
            off += n


@test
def batch_marginal_is_equipartition_across_recordings():
    """Each recording is best explained by one state (state i for recording i). Per-recording equipartition forces every recording to
    spread over all states; batch equipartition lets each recording stay in its own state while the batch stays balanced."""
    n = 12
    b, frames = make_batch([n, n, n, n], seed=4)
    cost = torch.ones(b.n, 4, dtype=torch.float64) * 1.0
    for r in range(4):
        cost[r * n:(r + 1) * n, r] = 0.0
    cf = Coeffs(0.7, 0.0, 2.0, 0.07)                                   # strong marginal, no temporal term: isolates the marginal
    lt_batch, _, _ = OT.solve_inner(None, cost, b, cf, 200)
    lt_rec, _, _ = OT.solve_inner(None, cost, b, cf, 200, per_recording=True)
    own = lambda lt: float(np.mean([torch.exp(lt[r * n:(r + 1) * n, r]).sum() / torch.exp(lt[r * n:(r + 1) * n]).sum() for r in range(4)]))
    assert own(lt_batch) > 0.9 and own(lt_rec) < 0.6, (own(lt_batch), own(lt_rec))
    assert (torch.exp(lt_batch).sum(0) / torch.exp(lt_batch).sum()).min() > 0.2                 # batch stays balanced


class _FixedRng:
    pass


def synth_data(n_rec=8, seed=0, P=(20, 25)):
    rng = np.random.default_rng(seed)
    names = [f"r{i}.npy" for i in range(n_rec)]
    canon, bank, lengths = {}, {}, {}
    for n in names:
        L = int(rng.integers(*P))
        canon[n] = rng.normal(size=(L, C.IN_DIM)).astype(np.float32)
        bank[n] = rng.normal(size=(C.N_VARIANTS, L, C.IN_DIM)).astype(np.float16)
        lengths[n] = np.concatenate([np.full(L - 1, C.WINDOW), [3]]).astype(np.int64)
    return names, canon, bank, lengths


@test
def no_evaluation_leakage_and_determinism():
    names, canon, bank, lengths = synth_data()
    fit = names[:6]
    outs = []
    for tweak in (False, True):
        c2, b2 = dict(canon), dict(bank)
        if tweak:                                                   # change the held-out recordings only
            for n in names[6:]:
                c2[n] = c2[n] + 5.0
                b2[n] = b2[n] + 5.0
        enc, protos, (mean, std), log = TR.train(fit, c2, b2, lengths, "L_full", 7, CPU, updates=20, ot_steps=4)
        outs.append((enc, protos, mean, std, log))
    (e1, p1, m1, s1, l1), (e2, p2, m2, s2, l2) = outs
    assert all(torch.equal(a, b) for a, b in zip(e1.state_dict().values(), e2.state_dict().values()))
    assert torch.equal(p1, p2) and np.array_equal(m1, m2) and np.array_equal(s1, s2) and l1["loss"] == l2["loss"]
    import inspect
    assert "eval" not in inspect.signature(TR.train).parameters
    src = (C.PKG / "train.py").read_text(encoding="utf-8")
    assert "load_labels" not in src and "load_gts" not in src and "groundTruth" not in src


@test
def swapped_loss_by_hand_and_no_gradient_through_pseudolabels():
    protos = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    za = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    zb = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
    qa = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    qb = torch.tensor([[0.5, 0.5], [1.0, 0.0]], requires_grad=True)
    w = torch.tensor([1.0, 3.0])
    loss = TR.swapped_loss(za, zb, qa, qb, protos, w, tau=0.5)
    ls = lambda x: torch.log_softmax(torch.as_tensor(x) / 0.5, dim=0)
    la = [-(1 * ls([0.0, 1.0])[0]), -(1 * ls([1.0, 0.0])[1])]               # row 0 of zb is [0,1] -> logits [0,1]; row 1 is [1,0]
    lb = [-(0.5 * ls([1.0, 0.0])[0] + 0.5 * ls([1.0, 0.0])[1]), -(1 * ls([0.0, 1.0])[0])]
    want = ((la[0] + lb[0]) * 1 + (la[1] + lb[1]) * 3) / 4
    assert abs(float(loss) - float(want)) < 1e-6
    loss.backward()
    assert qa.grad is None and qb.grad is None and za.grad is not None and zb.grad is not None


@test
def float32_gradients_match_float64():
    import copy
    enc32 = M.Encoder()
    enc64 = copy.deepcopy(enc32).double()
    x = torch.randn(3, 11, C.IN_DIM)
    mask = torch.ones(3, 11, 1)
    mask[1, 8:] = 0
    for enc, dt in ((enc32, torch.float32), (enc64, torch.float64)):
        enc.zero_grad()
        z = enc(x.to(dt), mask.to(dt))
        (z * torch.linspace(0.5, 1.5, C.OUT_DIM, dtype=dt)).sum().backward()
    for (n, a), (_, b) in zip(enc32.named_parameters(), enc64.named_parameters()):
        ga, gb = float(a.grad.norm()), float(b.grad.norm())
        assert abs(ga - gb) <= 2e-3 * max(gb, 1e-6) + 1e-7, f"{n}: float32 {ga:.4g} vs float64 {gb:.4g}"


@test
def encoder_masking_and_receptive_field():
    enc = M.Encoder().eval()
    x = torch.randn(1, 30, C.IN_DIM)
    full = enc(x, torch.ones(1, 30, 1))
    x2 = x.clone()
    x2[0, 20:] += 7.0                                            # far-away windows
    far = enc(x2, torch.ones(1, 30, 1))
    r = M.receptive_field_windows()
    assert r == 4 and torch.allclose(full[0, :20 - r], far[0, :20 - r], atol=1e-5) and not torch.allclose(full[0, 20 - 1], far[0, 20 - 1], atol=1e-4)
    padded = torch.cat([x, torch.randn(1, 5, C.IN_DIM)], dim=1)
    mask = torch.cat([torch.ones(1, 30, 1), torch.zeros(1, 5, 1)], dim=1)
    assert torch.allclose(enc(padded, mask)[0, :30], full[0], atol=1e-5)               # padding never reaches valid windows
    assert float(enc(padded, mask)[0, 30:].abs().max()) == 0.0
    assert np.allclose(torch.linalg.norm(full[0], dim=-1).detach().numpy(), 1.0, atol=1e-5)


@test
def training_marginal_is_declared_and_keeps_states_in_use():
    """The configured decoding weight (0.05) let pseudo-labels fall to about 2 effective states and made the equipartition lever inert
    (label-free check, 400 updates). Training uses LAMBDA_TRAIN; here a short synthetic run must keep more states in use under it."""
    assert TR.TRAIN_COEFFS.lam == C.LAMBDA_TRAIN == 1.0 and TR.COEFFS.lam == 0.05
    assert (TR.TRAIN_COEFFS.a, TR.TRAIN_COEFFS.beta, TR.TRAIN_COEFFS.eps) == (TR.COEFFS.a, TR.COEFFS.beta, TR.COEFFS.eps)
    src = (C.PKG / "train.py").read_text(encoding="utf-8")
    assert "OT.solve_inner(None, cost, batch, TRAIN_COEFFS" in src and "OT.solve_inner(None, cost, batch, COEFFS" not in src
    names, canon, bank, lengths = synth_data(10, 5, P=(20, 40))
    _, _, _, log = TR.train(names, canon, bank, lengths, "L_full", 3, CPU, updates=40, ot_steps=25)
    assert np.mean(log["usage_entropy"][-20:]) > np.log(3.0), np.mean(log["usage_entropy"][-20:])


def _log(max_share, entropy, hard=4.5, rank=10.0, pcos=0.3, n=60):
    return dict(max_state_share=[max_share] * n, usage_entropy=[entropy] * n, hard_effective_states=[hard] * n,
                embedding_effective_rank=[rank] * n, max_prototype_cos=[pcos] * n)


@test
def collapse_detector():
    assert not TR.is_collapsed(_log(0.3, 1.5))["any"]
    assert TR.is_collapsed(_log(0.95, 0.2))["soft"]                                   # one state dominates
    assert TR.is_collapsed(_log(0.58, 0.70))["soft"]                                  # about 2 effective states (the observed failure)
    assert TR.is_collapsed(_log(0.2, 1.609, hard=1.0))["hard"]                         # balanced soft mass, one hard state
    assert TR.is_collapsed(_log(0.2, 1.609, rank=1.0))["embedding"]                    # embeddings on a line
    assert TR.is_collapsed(_log(0.2, 1.609, pcos=0.99))["prototype"]                   # indistinguishable prototypes
    early = dict(max_state_share=[0.95] * 10 + [0.3] * 50, usage_entropy=[0.2] * 10 + [1.5] * 50, hard_effective_states=[4.5] * 60,
                 embedding_effective_rank=[10.0] * 60, max_prototype_cos=[0.3] * 60)
    assert not TR.is_collapsed(early)["any"]                                           # a transient early peak is not a collapse


@test
def representation_collapse_is_detected_although_soft_usage_looks_healthy():
    """Identical embeddings and identical prototypes give uniform soft assignments (maximum usage entropy). The soft monitor alone would
    call that healthy; the hard-assignment, embedding-rank and prototype checks must not."""
    z = torch.nn.functional.normalize(torch.ones(50, C.OUT_DIM), dim=-1)
    protos = torch.nn.functional.normalize(torch.ones(C.NUM_STATES, C.OUT_DIM), dim=-1)
    b, frames = make_batch([25, 25])
    b = OT.Batch([25, 25], frames, C.FPS, C.WINDOW, C.NUM_STATES, CPU)
    cost = 1.0 - z.double() @ protos.double().T
    lt, _, ok = OT.solve_inner(None, cost, b, TR.TRAIN_COEFFS, 25)
    q = OT.pseudo_labels(lt, b)
    share = (q * b.p.unsqueeze(1)).sum(0)
    ent = float(-(share * torch.log(share)).sum())
    assert ent > np.log(C.NUM_STATES) - 1e-3                                           # soft usage looks perfectly balanced
    assert TR.effective_rank(z) < 1.0 + 1e-6
    gram = protos @ protos.T - 2.0 * torch.eye(C.NUM_STATES)
    log = _log(float(share.max()), ent, hard=1.0, rank=TR.effective_rank(z), pcos=float(gram.max()))
    flags = TR.is_collapsed(log)
    assert not flags["soft"] and flags["embedding"] and flags["prototype"] and flags["any"]


@test
def collapse_detector_on_real_log_fields():
    names, canon, bank, lengths = synth_data(8, 1)
    _, _, _, log = TR.train(names[:6], canon, bank, lengths, "L_full", 3, CPU, updates=5, ot_steps=3)
    assert log["collapsed"] == TR.is_collapsed(log)["any"] and len(log["loss"]) == 5
    for key in ("usage_entropy", "min_state_share", "hard_effective_states", "hard_min_share", "embedding_effective_rank", "max_prototype_cos", "ot_accepted"):
        assert len(log[key]) == 5, key
    assert all(acc == [3, 3] for acc in log["ot_accepted"])


@test
def failed_pseudo_label_solve_invalidates_training_without_an_update():
    names, canon, bank, lengths = synth_data(8, 2)
    torch.manual_seed(0)
    orig = OT.solve_inner
    calls = {"n": 0}

    def failing(log_t0, cost, b, coeffs, steps, per_recording=False, **kw):
        calls["n"] += 1
        lt, n, ok = orig(log_t0, cost, b, coeffs, steps, per_recording)
        return lt, 0, False                                                           # zero accepted steps
    OT.solve_inner = failing
    try:
        try:
            TR.train(names[:6], canon, bank, lengths, "L_full", 3, CPU, updates=5, ot_steps=4)
            raise AssertionError("training continued after a failed pseudo-label solve")
        except TR.OTFailure as exc:
            assert "0 of 4" in str(exc) and calls["n"] == 1
    finally:
        OT.solve_inner = orig
    import inspect
    src = inspect.getsource(TR.train)
    assert src.index("raise OTFailure") < src.index("loss.backward()")


@test
def layout_and_folds():
    from script.self_labelled import evaluate as EV
    names = EV.all_names()
    assert len(names) == 255
    fit, ev = EV.fold_split(names, "fold0")
    assert not set(fit) & set(ev) and len(fit) + len(ev) == 255 and len(ev) == 51
    lay = EV.layout(names[:5])
    for n in names[:5]:
        l = lay[n]
        assert l["starts"][0] == 0 and int(l["lengths"].sum()) == l["frame_count"] and (l["lengths"][:-1] == C.WINDOW).all()
    src = (C.PKG / "evaluate.py").read_text(encoding="utf-8")
    assert "def fold_split" in src


def _table(d_mof, d_f1, vs_f1=(5.0, 5.0), base=(50.0, 25.0), collapsed_full=0):
    t = {}
    for i, s_ in enumerate(C.SEEDS):
        t[("F0", s_)] = (base[0], base[1], 45.0, 0.3, 0.5, 0.2, 0)
        t[("F1", s_)] = (base[0], base[1], 45.0, 0.3, 0.5, 0.2, 0)
        t[("L_full", s_)] = (base[0] + d_mof[i], base[1] + d_f1[i], 47.0, 0.2, 0.3, 0.1, collapsed_full if i == 0 else 0)
        t[("L_noeq", s_)] = (base[0] + d_mof[i] - vs_f1[0], base[1] + d_f1[i] - vs_f1[1], 47.0, 0.2, 0.3, 0.1, 0)
        t[("L_noaug", s_)] = (base[0] + d_mof[i] - vs_f1[0], base[1] + d_f1[i] - vs_f1[1], 47.0, 0.2, 0.3, 0.1, 0)
    return t


DECL = (list(C.SEEDS), list(C.FOLDS), list(C.ARMS))


@test
def decision_rules():
    assert R.dev_rule([2.5, 2.1, 3, 2.2, 2.4], [0.0, 0.5, -0.5, 0.2, 0.3])["holds"]
    assert not R.dev_rule([2.5, 2.1, 3, 2.2, 2.4], [-2.0, -2.0, -2.5, 0.2, 0.3])["holds"]               # F1 mean (-1.2) below the floor
    assert not R.dev_rule([3, 3, 3, -0.1, -0.2], [0, 0, 0, 0, 0])["holds"]                              # only 3 of 5 seeds positive
    assert not R.dev_rule([1.0] * 5, [1.0] * 5)["holds"]                                               # no gain reaching +2
    t = _table([4.0] * 5, [3.0] * 5)
    d = R.decide(t, *DECL, True, False)
    assert d["verdict"] == "STAGE2_ELIGIBLE" and d["learning_claim"] and d["equipartition_claim"] and d["augmentation_claim"]
    t2 = _table([4.0] * 5, [3.0] * 5)
    for s_ in C.SEEDS:
        t2[("F1", s_)] = (54.0, 28.0, 45.0, 0.3, 0.5, 0.2, 0)
    assert R.decide(t2, *DECL, True, False)["verdict"] == "ADOPT_BUT_NOT_LEARNING"
    assert R.decide(_table([0.0] * 5, [0.0] * 5), *DECL, True, False)["verdict"] == "STOP"
    assert R.decide(_table([4.0] * 5, [3.0] * 5, collapsed_full=1), *DECL, True, False)["verdict"] == "BLOCKED_BY_COLLAPSE"
    assert R.decide(t, *DECL, False, False)["verdict"] == "INCONCLUSIVE"
    broken = dict(t)
    broken[("L_noeq", 111)] = None
    assert R.decide(broken, *DECL, True, False)["verdict"] == "INCONCLUSIVE"
    assert R.decide(t, list(C.SEEDS[:3]), list(C.FOLDS), list(C.ARMS), True, False)["verdict"] == "INCONCLUSIVE"
    assert R.decide(t, list(C.SEEDS), ["fold0"], list(C.ARMS), True, False)["verdict"] == "INCONCLUSIVE"        # one fold only
    assert R.decide(t, list(C.SEEDS), list(C.FOLDS), ["F0", "F1", "L_full"], True, False)["verdict"] == "INCONCLUSIVE"
    assert R.decide(t, *DECL, True, True)["verdict"].startswith("not computed")


@test
def stale_or_incomplete_cells_are_not_counted():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        good = dict(eval=dict(metrics={"MoF": 50.0, "F1@50": 25.0}, balanced_accuracy=45.0, association={"recording": {"nmi": 0.2}}),
                    nn_share_eval={"shared_context": {"share_all": 0.4}, "support_disjoint": {"share_eligible": 0.1}})
        for f in C.FOLDS:
            d = R.cell_dir(run_dir, "F1", 111, f)
            R.write_json(d / "scores.json", good)
            R.write_json(d / "info.json", dict(status="complete", identity=R.identity("F1", 111, f, False)))
        assert R.seed_metrics(run_dir, "F1", 111, C.FOLDS)[:2] == (50.0, 25.0)
        d = R.cell_dir(run_dir, "F1", 111, "fold3")
        R.write_json(d / "info.json", dict(status="complete", identity="stale0000000000"))           # an old implementation
        assert R.seed_metrics(run_dir, "F1", 111, C.FOLDS) is None
        R.write_json(d / "info.json", dict(status="invalid", identity=R.identity("F1", 111, "fold3", False)))
        assert R.seed_metrics(run_dir, "F1", 111, C.FOLDS) is None
        R.write_json(d / "info.json", dict(status="complete", identity=R.identity("F1", 111, "fold3", True)))  # a smoke cell
        assert R.seed_metrics(run_dir, "F1", 111, C.FOLDS) is None


@test
def f1_preprocessing_is_fitted_on_fitting_recordings_only():
    from script.self_labelled import augment as AU2, evaluate as EV
    names = EV.all_names()[:24]
    fit, ev = names[:18], names[18:]
    real = AU2.canonical_windows
    canon = real(names)
    try:
        AU2.canonical_windows = lambda ns: {n: canon[n] for n in ns}
        a = EV.f1_inputs(111, fit, ev)
        shifted = {n: (canon[n] + (50.0 if n in ev else 0.0)) for n in names}
        AU2.canonical_windows = lambda ns: {n: shifted[n] for n in ns}
        b = EV.f1_inputs(111, fit, ev)
    finally:
        AU2.canonical_windows = real
    assert all(np.allclose(a[n], b[n]) for n in fit)                                  # held-out windows did not change the fitted maps
    assert not all(np.allclose(a[n], b[n]) for n in ev)
    assert a[fit[0]].shape[1] == C.F1_PCA_DIM


@test
def neighbour_share_radii_and_eligibility():
    from script.self_labelled import evaluate as EV
    assert EV.support_radius_windows() == 42
    rng = np.random.default_rng(0)
    z = {f"r{i}": rng.normal(size=(30, 8)) + 5 * rng.normal(size=(1, 8)) for i in range(10)}     # strong per-recording offset
    out = EV.nn_share(z, n_query=200)
    assert out["shared_context"]["radius"] == C.NN_SHARED_CONTEXT_RADIUS and out["shared_context"]["share_all"] > 0.8
    sd = out["support_disjoint"]
    assert sd["radius"] == 42 and sd["queries_with_eligible"] == 0 and sd["share_eligible"] is None   # 30-window recordings: nothing eligible
    long = {f"r{i}": rng.normal(size=(120, 8)) + 5 * rng.normal(size=(1, 8)) for i in range(6)}
    sd2 = EV.nn_share(long, n_query=300)["support_disjoint"]
    assert sd2["queries_with_eligible"] > 0 and sd2["share_eligible"] > 0.8


@test
def decode_artifacts_are_saved():
    import tempfile
    from pathlib import Path
    from script.self_labelled import evaluate as EV
    sc = dict(metrics={"MoF": 1.0}, artifacts=dict(mu=np.eye(5), mu0=np.eye(5), cost_eval=np.zeros((4, 5), np.float32), outer_trace=[1.0, 0.5],
                                                   states={"a": np.array([0, 1], np.int16)}, first_rung_states={"a": np.array([0, 0], np.int16)},
                                                   frame_predictions_raw={"a": np.array([0, 0, 1], np.int16)},
                                                   frame_predictions_mapped={"a": np.array([2, 2, 3], np.int16)}))
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "decode.npz"
        rest = EV.save_decode(p, sc)
        with np.load(p) as z:                                          # close the archive (Windows cannot delete an open file)
            files = set(z.files)
        assert "artifacts" not in rest and files >= {"mu", "mu0", "cost_eval", "outer_trace", "states/a", "first_rung_states/a",
                                                     "frame_predictions_raw/a", "frame_predictions_mapped/a"}
    src = (C.PKG / "run.py").read_text(encoding="utf-8")
    assert "decode_primary.npz" in src and "decode_own_prototypes.npz" in src and "training_log.json" in src


@test
def bank_identity_tracks_settings_and_code():
    a = AU.bank_identity()
    old = C.N_VARIANTS
    try:
        C.N_VARIANTS = 9
        assert AU.bank_identity() != a
    finally:
        C.N_VARIANTS = old
    assert AU.bank_identity() == a and AU.bank_dir().name == a


@test
def caching_identity_and_declared_settings():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "c"
        ident = R.identity("L_full", 111, "fold0", False)
        R.write_json(d / "info.json", dict(status="complete", identity=ident))
        assert not R.completed(d, ident)                          # scores.json missing
        R.write_json(d / "scores.json", {})
        assert R.completed(d, ident) and not R.completed(d, R.identity("L_full", 222, "fold0", False))
        R.write_json(d / "info.json", dict(status="interrupted", identity=ident))
        assert not R.completed(d, ident)
    assert R.identity("L_full", 111, "fold0", True) != R.identity("L_full", 111, "fold0", False)
    assert (C.UPDATES, C.BATCH_RECS, C.TAU, C.FREEZE_PROTOS, C.NUM_STATES, C.OT_INNER_STEPS, C.N_VARIANTS) == (3000, 16, 0.1, 300, 5, 25, 8)
    from script.action_transport import config as at
    assert (at.SETTING_T.a, at.SETTING_T.beta, at.SETTING_T.lam, at.SETTING_T.eps) == (COEFFS.a, COEFFS.beta, COEFFS.lam, COEFFS.eps)


def main() -> int:
    results, t0 = {}, time.time()
    for n, fn in TESTS.items():
        try:
            fn()
            results[n] = "pass"
            print(f"PASS {n}", flush=True)
        except Exception:  # noqa: BLE001
            results[n] = "FAIL"
            print(f"FAIL {n}\n{traceback.format_exc()}", flush=True)
    ok = all(v == "pass" for v in results.values())
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "verification.json").write_text(json.dumps(dict(passed=ok, code_hash=C.code_hash(), tests=results, seconds=time.time() - t0), indent=2))
    print(f"{sum(v == 'pass' for v in results.values())}/{len(results)} passed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
