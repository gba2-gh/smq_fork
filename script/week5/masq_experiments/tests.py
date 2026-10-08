"""Unit tests and gates G1-G3 for the MASQ replication (CPU, float32: SMQ's quantizer builds float32 one-hots, so it is not float64-clean). Run: python -m script.week5.masq_experiments.tests [--only N ...]

G1 = test 1 (off-switch equivalence), G2 = test 2 (JLSD mechanics), G3 = test 3 (velocity loss).
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from script.week5.masq_experiments import config as C
from script.week5.masq_experiments.losses import compute_losses, velocity_loss
from script.week5.masq_experiments.model_masq import MASQModel, build_masq
from src.model.smq import SMQModel

SHAPES = {"hugadb": dict(C=6, V=6), "babel3": dict(C=3, V=25)}
PATCH, T_TEST, K_TEST = 4, 10, 5


def _smq_like(dataset: str, dtype=torch.float32) -> tuple[SMQModel, MASQModel]:
    d, s = C.DATASETS[dataset], C.SHARED
    kw = dict(in_channels=d["num_features"], filters=s["num_f_maps"], num_layers=s["num_layers"], latent_dim=s["latent_dim"],
              num_actions=K_TEST, num_joints=d["num_joints"], num_person=1, patch_size=PATCH, kmeans=False,
              kmeans_metric=s["kmeans_metric"], sampling_quantile=s["sampling_quantile"],
              replacement_strategy=s["replacement_strategy"], decay=s["decay"], dead_code_threshold=s["dead_code_threshold"],
              tc_weight=s["tc_weight"], tc_beta=s["tc_beta"])
    torch.manual_seed(0)
    smq = SMQModel(**kw)
    torch.manual_seed(0)
    masq = MASQModel(**kw, jlsd_p=0.0)
    masq.load_state_dict(smq.state_dict())
    return smq.to(dtype), masq.to(dtype)


def _batch(dataset: str, N: int = 2, T: int = T_TEST, dtype=torch.float32, pad: int = 3, seed: int = 5):
    g = torch.Generator().manual_seed(seed)
    d = C.DATASETS[dataset]
    x = torch.randn(N, d["num_features"], T, d["num_joints"], 1, generator=g, dtype=dtype)
    mask = torch.ones_like(x)
    if N > 1 and pad:
        mask[1, :, T - pad:] = 0
    return x * mask, mask


def _total(model, x, mask, vel_weight=0.0):
    out = model(x, mask)
    return out, compute_losses(model, x, mask, out, C.SHARED["mse_loss_weight"], C.SHARED["commit_weight"], vel_weight)


# -- test 1 (G1) ----------------------------------------------------------------------------------------------------------

def test_1_off_switch():
    for dataset in SHAPES:
        for mode in ("train", "eval"):
            smq, masq = _smq_like(dataset)
            getattr(smq, mode)(), getattr(masq, mode)()
            x, mask = _batch(dataset)
            torch.manual_seed(1)
            out_a, L_a = _total(smq, x, mask)
            torch.manual_seed(1)
            out_b, L_b = _total(masq, x, mask)
            assert torch.equal(out_a, out_b), f"{dataset}/{mode}: outputs differ"
            assert torch.equal(smq.indices, masq.indices), f"{dataset}/{mode}: indices differ"
            assert torch.equal(smq.commit_loss, masq.commit_loss), f"{dataset}/{mode}: commit loss differs"
            assert torch.equal(L_a["total"], L_b["total"]), f"{dataset}/{mode}: total loss differs"
            L_a["total"].backward()
            L_b["total"].backward()
            for (n, pa), (_, pb) in zip(smq.named_parameters(), masq.named_parameters()):
                ga = torch.zeros_like(pa) if pa.grad is None else pa.grad
                gb = torch.zeros_like(pb) if pb.grad is None else pb.grad
                assert torch.equal(ga, gb), f"{dataset}/{mode}: gradient of {n} differs (max {float((ga - gb).abs().max()):.3e})"
            assert masq.joint_keep is not None and bool((masq.joint_keep == 1).all())


# -- test 2 (G2) ----------------------------------------------------------------------------------------------------------

def _spy(model):
    captured = {}
    original = model.vq.forward

    def forward(x, mask):
        captured["x"] = x.detach().clone()
        return original(x, mask)

    model.vq.forward = forward
    return captured


def test_2_jlsd_mechanics():
    dataset, p = "hugadb", 0.5
    d = C.DATASETS[dataset]
    Z, V = C.SHARED["latent_dim"], d["num_joints"]
    torch.manual_seed(0)
    model = build_masq(dataset, K_TEST, jlsd_p=p, patch_size=PATCH)
    captured = _spy(model)
    x, mask = _batch(dataset, N=6, T=8, pad=3)

    def expected_latent():  # (N, T, V*Z) of the undropped encoder output
        N = x.shape[0]
        return model.latent.view(N, V, Z, -1).permute(0, 3, 1, 2).reshape(N, -1, V * Z)

    # train mode: dropped rows are zero at every (t, channel), kept rows are scaled by 1/(1-p); padding stays zero
    model.train()
    torch.manual_seed(3)
    model(x, mask)
    keep, got, ref = model.joint_keep, captured["x"], expected_latent()
    assert keep.shape == (6, 1, V)
    for n in range(6):
        for v in range(V):
            sl = got[n, :, v * Z:(v + 1) * Z]
            if keep[n, 0, v] == 0:
                assert bool((sl == 0).all()), f"dropped joint (n={n}, v={v}) not zero"
            else:
                assert torch.allclose(sl, ref[n, :, v * Z:(v + 1) * Z] / (1 - p), atol=1e-5), f"kept joint (n={n}, v={v}) wrong"
    assert 0 < int((keep == 0).sum()) < keep.numel(), "this seed must drop some and keep some rows"
    assert bool((got[1, -3:] == 0).all()), "padded frames must stay zero"

    # the drop rate over many draws is within 4 binomial SDs of p
    model2 = build_masq(dataset, K_TEST, jlsd_p=p, patch_size=PATCH)
    model2.train()
    xb, mb = _batch(dataset, N=100, T=8, pad=0, dtype=torch.float32)
    torch.manual_seed(11)
    dropped = total = 0
    for _ in range(20):
        model2(xb, mb)
        dropped += int((model2.joint_keep == 0).sum())
        total += model2.joint_keep.numel()
    sd = (total * p * (1 - p)) ** 0.5
    assert abs(dropped - total * p) <= 4 * sd, f"drop rate {dropped / total:.4f} not within 4 SD of {p}"

    # eval mode: no dropping
    model.eval()
    model(x, mask)
    assert bool((model.joint_keep == 1).all())
    assert torch.equal(captured["x"], expected_latent())


# -- test 3 (G3) ----------------------------------------------------------------------------------------------------------

def _velocity_by_hand(x, xh, mask, keep):
    N, Ch, T, V, M = x.shape
    s, count = 0.0, 0
    for n in range(N):
        for t in range(1, T):
            for v in range(V):
                for m in range(M):
                    if mask[n, 0, t, v, m] == 1 and mask[n, 0, t - 1, v, m] == 1 and keep[n, m, v] == 1:
                        for c in range(Ch):
                            s += float(((xh[n, c, t, v, m] - xh[n, c, t - 1, v, m]) - (x[n, c, t, v, m] - x[n, c, t - 1, v, m])) ** 2)
                        count += 1
    return s / max(count, 1)


def test_3_velocity_loss():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(2, 2, 4, 3, 1, generator=g, dtype=torch.float64)
    xh = torch.randn(2, 2, 4, 3, 1, generator=g, dtype=torch.float64)
    mask = torch.ones_like(x)
    mask[1, :, 3:] = 0  # sample 1: last frame padded
    keep = torch.ones(2, 1, 3, dtype=torch.float64)
    keep[0, 0, 1] = 0  # sample 0: joint 1 dropped
    got = float(velocity_loss(x, xh, mask, keep))
    assert abs(got - _velocity_by_hand(x, xh, mask, keep)) < 1e-12, "velocity loss differs from the hand-computed value"
    assert float(velocity_loss(x, x.clone(), mask, keep)) == 0.0, "loss must be zero when x_hat equals x"
    # unchanged when only dropped joints or padded frames of x_hat are perturbed
    xh2 = xh.clone()
    xh2[0, :, :, 1] += 5.0  # dropped joint of sample 0
    xh2[1, :, 3:] += 7.0  # padded frame of sample 1
    assert abs(float(velocity_loss(x, xh2, mask, keep)) - got) < 1e-12, "dropped joints / padded frames must not matter"
    # no gradient to the dropped joint
    xh3 = xh.clone().requires_grad_(True)
    velocity_loss(x, xh3, mask, keep).backward()
    assert float(xh3.grad[0, :, :, 1].abs().max()) == 0.0
    # normalization: sum over channels, mean over valid (pair, joint) points
    one = torch.zeros(1, 3, 3, 2, 1, dtype=torch.float64)
    one_hat = one.clone()
    one_hat[0, :, 1, 0, 0] = 2.0  # joint 0: step of 2 in each of 3 channels at t=1 only
    val = float(velocity_loss(one, one_hat, torch.ones_like(one), torch.ones(1, 1, 2, dtype=torch.float64)))
    # pairs: t=1 -> |2|^2*3 = 12 for joint 0; t=2 -> |-2|^2*3 = 12 for joint 0; joint 1 zero; 4 valid points -> 24/4 = 6
    assert abs(val - 6.0) < 1e-12, val


# -- test 4: seeding ------------------------------------------------------------------------------------------------------

def _param_hash(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def test_4_seeding():
    from script.week5.masq_experiments.train_masq import prepare
    a, b, c = (prepare(C.Unit("S0", "babel3", s)) for s in (111, 111, 222))
    assert a[1].list_of_examples == b[1].list_of_examples, "same seed must give the same batch order"
    assert _param_hash(a[0]) == _param_hash(b[0]), "same seed must give the same initial weights"
    assert a[1].list_of_examples != c[1].list_of_examples, "different seeds must give different batch orders"
    assert _param_hash(a[0]) != _param_hash(c[0]), "different seeds must give different initial weights"
    assert a[2] == 5, f"K = C for babel3 should be 5, got {a[2]}"


# -- test 5: identity and caching -----------------------------------------------------------------------------------------

def test_5_identity_caching():
    u = C.Unit("M-pub", "babel3", 111)
    ids = {C.unit_identity(u), C.unit_identity(C.Unit("S0", "babel3", 111)), C.unit_identity(C.Unit("M-pub", "babel3", 222)),
           C.unit_identity(C.Unit("M-pub", "babel2", 111)), C.unit_identity(C.Unit("M-fix", "babel3", 111)),
           C.unit_identity(u, smoke=True)}
    assert len(ids) == 6, "units, arms, seeds and smoke must have distinct identities"
    assert C.unit_identity(u) == C.unit_identity(u), "identity must be stable"
    assert len({C.unit_identity(u), C.score_identity(u), C.diagnostics_identity(u)}) == 3
    assert not any(f.endswith("config.py") for f in C.TRAIN_FILES + C.SCORE_FILES), "config constants must not be hashed"
    import json
    from script.week5.masq_experiments.run import step_done
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "scores.json"
        assert not step_done(p, "abc")
        p.write_text(json.dumps(dict(identity="abc")))
        assert step_done(p, "abc") and not step_done(p, "other"), "a stale identity must not count as done"
    s1 = C.stage_units("1")
    assert len(s1) == 30 and len({x.unit_id for x in s1}) == 30
    assert {x.arm for x in s1} == {"S0", "M-pub"} and s1[0].dataset == "babel3"
    s2 = C.stage_units("2")
    assert len(s2) == 27 and sum(x.arm == "M-fix" for x in s2) == 12 and not any(x.arm == "M-fix" and x.dataset == "hugadb" for x in s2)
    assert C.Unit("M-fix", "lara", 111).jlsd_p == 0.25 and C.Unit("M-pub", "lara", 111).jlsd_p == 0.20
    assert C.Unit("M-JL", "babel1", 111).vel_weight == 0.0 and C.Unit("M-pub", "babel1", 111).vel_weight == 0.001


# -- test 6: reading and verdict ------------------------------------------------------------------------------------------

def test_6_reading_verdict():
    from script.week5.masq_experiments import report as R
    assert R.reading([4.0, 3.5, 3.1]) == "passes the bar"
    assert R.reading([4.0, 5.0, -0.1]).startswith("within")
    assert R.reading([-4.0, -3.5, -3.1]) == "reverse passes the bar"
    assert R.reading([2.9, 2.9, 2.9]).startswith("within")
    assert R.reading([5.0, 5.0]) == "incomplete (2 seeds)" and R.reading([5.0]) == "incomplete (1 seed)" and R.reading([]) == "no cells"
    assert R.reading([6.0, 3.0, 3.0]) == "passes the bar"
    pub = C.MASQ_PUBLISHED["lara"]
    assert R.reproduced(dict(MoF=pub["MoF"] - 2.9, **{"F1@50": pub["F1@50"] + 2.9}), pub)
    assert not R.reproduced(dict(MoF=pub["MoF"] - 3.1, **{"F1@50": pub["F1@50"]}), pub)
    p = lambda *ds: {d: ("pass" if d in ds else "within") for d in C.DATASETS}  # noqa: E731
    ok = {d: True for d in C.DATASETS}
    no = {d: False for d in C.DATASETS}
    assert R.verdict(p("lara", "babel1", "babel2"), ok)[0] == "replicated"
    assert R.verdict(p("lara", "babel1", "babel2"), no)[0] == "partly replicated"  # R2 fails everywhere
    assert R.verdict(p("babel1", "babel2", "babel3"), ok)[0] == "partly replicated"  # three passes but not LARa
    assert R.verdict(p("lara"), ok)[0] == "partly replicated"
    assert R.verdict(p(), ok)[0] == "not replicated"
    s0 = {"hugadb": {"MoF": 41.6, "F1@50": 21.5}, "lara": {"MoF": 36.26 + 3.5, "F1@50": 15.75}}
    r = R.g5(s0)
    assert r["hugadb"]["passed"] and not r["lara"]["passed"] and not r["lara"]["checks"]["MoF"]


# -- test 7: no labels in training code -----------------------------------------------------------------------------------

def test_7_no_labels():
    here = Path(__file__).parent
    banned = re.compile(r"groundTruth|\bmapping\b|read_mapping_file|create_correspondences|gt_path|\bgt\b\s*=")
    for name in ("model_masq.py", "losses.py", "train_masq.py"):
        text = (here / name).read_text()
        # docstrings and comments may talk about labels; code may not
        code = re.sub(r'""".*?"""', "", text, flags=re.S)
        code = "\n".join(line.split("#")[0] for line in code.splitlines())
        hits = [(i + 1, line.strip()) for i, line in enumerate(code.splitlines()) if banned.search(line)]
        if name == "train_masq.py":  # the dataset paths dict names the label folder for get_num_actions only
            hits = [h for h in hits if "get_num_actions" not in h[1] and 'gt=root / "groundTruth"' not in h[1].replace("'", '"')
                    and 'mapping=root / "mapping"' not in h[1]]
        assert not hits, f"{name} references labels: {hits}"
    text = (here / "train_masq.py").read_text()
    assert text.count("get_num_actions(") == 1  # the single call (the import line has no parenthesis)
    assert "paths[\"gt\"]" in text and text.count("paths[\"gt\"]") == 1
    assert "paths[\"mapping\"]" not in text


# -- test 8: diagnostics --------------------------------------------------------------------------------------------------

def test_8_diagnostics():
    from script.week5.masq_experiments import diagnostics as G
    rng = np.random.default_rng(0)
    names = [f"HuGaDB_v2_various_{s:02d}_{r:02d}.npy" for s in range(1, 5) for r in range(3)]
    frames = 400
    subj = [int(n.split("_")[-2]) for n in names]
    gts = [np.where(np.arange(frames) < frames // 2, 0, 1).astype(np.int16) for _ in names]
    lats = [rng.normal(size=(20, 8)).astype(np.float32) for _ in names]
    same = [np.full(frames, s, dtype=np.int16) for s in subj]  # codes identical to the subject
    indep = [rng.integers(0, 5, frames).astype(np.int16) for _ in names]  # codes independent of the subject
    a = G.diagnose_arrays(names, same, gts, lats, "hugadb")
    b = G.diagnose_arrays(names, indep, gts, lats, "hugadb")
    assert a["subject_ami"] > 0.9, a["subject_ami"]
    assert abs(b["subject_ami"]) < 0.05, b["subject_ami"]
    assert a["per_class"]["uses_labels"] and set(a["per_class"]["classes"]) == {"0", "1"}
    assert 0.0 <= a["latent_same_recording_nn_share"] <= 1.0
    c = G.diagnose_arrays(names, same, gts, lats, "babel3")  # no subject for BABEL
    assert c["subject_ami"] is None and c["per_class"] is None


# -- tests 9 and 10 (G6, G7): the variant terms --------------------------------------------------------------------------

def test_9_variant_calibration():
    from script.week5.masq_experiments import losses_variants as LV
    from script.week5.masq_experiments.train_variants import calibrate
    x, mask = _batch("hugadb", N=3, T=12, pad=3, seed=7)
    x = x * 50
    a = LV.calibrate_weight("coord_scalefree", 0.0016, x, mask, 0.001)
    b = LV.calibrate_weight("coord_scalefree", 0.0016, x * 100, mask, 0.001)
    assert abs(a["weight"] - b["weight"]) / a["weight"] < 1e-4, "the calibrated weight must not depend on the data scale"
    assert abs(a["weight"] * a["term_init"] - 0.0016 * a["rec_init"]) <= 1e-5 * a["rec_init"], "weight x term(init) must equal rho x reconstruction(init)"
    assert a["rec_init"] > 0 and a["term_init"] > 0
    # calibrating a model changes neither the RNG stream, nor any weight or buffer (the codebook stays uninitialized), nor train mode
    unit = C.Unit("V2a", "hugadb", 111)
    st = C.settings(unit)
    torch.manual_seed(0)
    model = build_masq("hugadb", 10, jlsd_p=unit.jlsd_p)
    model.train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    torch.manual_seed(5)
    ref = torch.rand(4)
    torch.manual_seed(5)
    out = calibrate(model, unit, st, torch.device("cpu"))
    assert torch.equal(torch.rand(4), ref), "calibration must run under a forked RNG"
    assert float(model.vq.initted) == 0.0 and all(torch.equal(before[k], v) for k, v in model.state_dict().items()), "calibration must not touch the quantizer"
    assert model.training
    assert abs(out["weight"] * out["term_init"] - st["rho"] * out["rec_init"]) <= 1e-4 * out["rec_init"]
    assert len(out["files"]) == C.CALIB_FILES and out["files"] == sorted(out["files"])
    # the settings of baseline arms are unchanged by the variant additions (their masq_v1 identities stay valid)
    assert "vel_mode" not in C.settings(C.Unit("S0", "hugadb", 111)) and "vel_mode" in st
    assert C.Unit("V1", "hugadb", 111).jlsd_p == C.Unit("M-JL", "hugadb", 111).jlsd_p == 0.25


def test_10_latent_smoothness():
    from script.week5.masq_experiments import losses_variants as LV
    N, M, V, Z, T = 2, 1, 3, 4, 6
    g = torch.Generator().manual_seed(1)
    lat = torch.randn(N * M * V, Z, T, generator=g)
    mask = torch.ones(N, 2, T, V, M)
    mask[1, :, 4:] = 0
    keep = torch.ones(N, M, V)
    keep[0, 0, 1] = 0
    num = den = 0.0
    for n in range(N):
        for m in range(M):
            for v in range(V):
                r = n * M * V + m * V + v
                for t in range(1, T):
                    if mask[n, 0, t, v, m] == 1 and mask[n, 0, t - 1, v, m] == 1 and keep[n, m, v] == 1:
                        num += float(((lat[r, :, t] - lat[r, :, t - 1]) ** 2).sum())
                        den += float((lat[r, :, t] ** 2).sum())
    got = float(LV.latent_smoothness(lat, mask, keep))
    assert abs(got - num / den) < 1e-5, (got, num / den)
    assert abs(float(LV.latent_smoothness(lat * 7.0, mask, keep)) - got) < 1e-5, "must be scale invariant"
    assert float(LV.latent_smoothness(lat[:, :, :1].expand(-1, -1, T).contiguous(), mask, keep)) == 0.0, "constant trajectories have zero smoothness loss"
    lat3 = lat.clone()
    lat3[3:6, :, 4:] += 9.0  # sample 1's padded frames
    assert abs(float(LV.latent_smoothness(lat3, mask, keep)) - got) < 1e-5, "padded frames must not matter"
    # row mapping: only row (n=1, v=2) varies in time
    lat2 = torch.ones(N * M * V, Z, T)
    lat2[1 * M * V + 0 * V + 2] = torch.randn(Z, T, generator=g)
    full = torch.ones(N, 2, T, V, M)
    k1 = torch.ones(N, M, V)
    k2, k3 = k1.clone(), k1.clone()
    k2[1, 0, 2] = 0  # drops the varying row
    k3[0, 0, 2] = 0  # drops a constant row
    assert float(LV.latent_smoothness(lat2, full, k2)) == 0.0, "the varying row must be ignored when dropped"
    assert float(LV.latent_smoothness(lat2, full, k1)) > 0 and float(LV.latent_smoothness(lat2, full, k3)) > 0
    # gradient only reaches visible rows
    lg = lat.clone().requires_grad_(True)
    LV.latent_smoothness(lg, mask, keep).backward()
    assert float(lg.grad[0 * M * V + 0 * V + 1].abs().max()) == 0.0, "no gradient to the dropped joint (n=0, v=1)"


TESTS = {1: test_1_off_switch, 2: test_2_jlsd_mechanics, 3: test_3_velocity_loss, 4: test_4_seeding, 5: test_5_identity_caching,
         6: test_6_reading_verdict, 7: test_7_no_labels, 8: test_8_diagnostics,
         9: test_9_variant_calibration, 10: test_10_latent_smoothness}
GATES = {"G1": 1, "G2": 2, "G3": 3, "G6": 9, "G7": 10}


def run_tests(which=None) -> dict:
    results = {}
    for n in which or TESTS:
        t0 = time.perf_counter()
        try:
            TESTS[n]()
            results[n] = dict(status="passed", seconds=round(time.perf_counter() - t0, 2))
            print(f"PASS test {n}: {TESTS[n].__name__} ({results[n]['seconds']}s)", flush=True)
        except Exception as exc:  # noqa: BLE001
            results[n] = dict(status="failed", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc())
            print(f"FAIL test {n}: {TESTS[n].__name__}: {type(exc).__name__}: {exc}", flush=True)
    return results


def run_gate_tests() -> dict:
    res = run_tests(list(GATES.values()))
    return {g: dict(status=res[n]["status"], **{k: v for k, v in res[n].items() if k != "status"}) for g, n in GATES.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*")
    args = ap.parse_args(argv)
    res = run_tests(args.only)
    failed = [n for n, r in res.items() if r["status"] != "passed"]
    print(f"{len(res) - len(failed)}/{len(res)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
