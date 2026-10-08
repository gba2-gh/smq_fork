"""Tests for the input-representation test. python -m script.input_representation.tests"""
from __future__ import annotations

import json
import sys
import time
import traceback

import numpy as np

from script.input_representation import config as C
from script.input_representation import features as F
from script.input_representation import probes as P
from script.input_representation import run as R

TESTS = {}


def test(fn):
    TESTS[fn.__name__] = fn
    return fn


def rand_pose(T=40, V=22, seed=0):
    return np.random.default_rng(seed).normal(size=(T, V, 3)).astype(np.float32) * 100


def stats(pos, source):
    from script.week5.timescale_inspection import tokenizers as TOK
    return TOK.window_stats(F.frame_features(pos, source), 12)


@test
def bodyscale_removes_size_but_t2_keeps_it():
    from script.week5.timescale_inspection import tokenizers as TOK
    pos = rand_pose()
    t2_small = TOK.window_stats(TOK.frame_distances(pos), 12)
    t2_big = TOK.window_stats(TOK.frame_distances(pos * 1.3), 12)
    assert not np.allclose(t2_small, t2_big, rtol=1e-3)                                  # T2 encodes body size
    a, b = stats(pos, "relational_bodyscale"), stats(pos * 1.3, "relational_bodyscale")
    assert np.allclose(a, b, rtol=1e-4, atol=1e-5)                                       # the variant does not
    assert np.allclose(a, stats(pos + np.array([5.0, -3.0, 9.0], dtype=np.float32), "relational_bodyscale"), rtol=1e-3, atol=1e-4)


@test
def centered_motion_invariances_and_dimensions():
    pos = rand_pose(seed=1)
    base = stats(pos, "centered_motion")
    assert base.shape == (4, 3 * (22 * 3 + 3))
    assert np.allclose(base, stats(pos + np.array([7.0, 1.0, -4.0], dtype=np.float32), "centered_motion"), rtol=1e-3, atol=1e-3)   # translation
    assert np.allclose(base, stats(pos * 0.7, "centered_motion"), rtol=1e-3, atol=1e-3)                                            # body size
    R90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    assert not np.allclose(base, stats(pos @ R90.T, "centered_motion"), rtol=1e-2)      # heading stays visible (declared)


@test
def window_features_use_only_their_own_frames():
    pos = rand_pose(T=36, seed=2)
    a = stats(pos, "centered_motion")
    pos2 = pos.copy()
    pos2[24:] += 50.0                                   # change only the last window's frames
    b = stats(pos2, "centered_motion")
    assert np.allclose(a[:2], b[:2], atol=1e-3)         # earlier windows unaffected apart from the (global) scale proxy
    # note: the body-scale proxy is per recording, so scale-sensitive entries can move; translation leaves it unchanged


@test
def window_majority_by_hand():
    gt = np.array([0, 0, 1, 1, 2, 2, 2, 2, 2])
    got = P.window_majority(gt, np.array([0, 3, 7]), np.array([3, 4, 2]), 3)      # windows [0,0,1] [1,2,2,2] [2,2]
    assert got.tolist() == [0, 2, 2]
    assert P.window_majority(np.array([1, 1, 1, 0]), np.array([0]), np.array([4]), 2).tolist() == [1]


@test
def probes_separate_signal_from_noise():
    names = [f"L0{r}_S{s:02d}_R{i:02d}.npy" for s in range(1, 17) for r in (1,) for i in range(1, 4)]
    rng = np.random.default_rng(0)
    n_per = 60
    X, y, rec, codes, lens = [], [], [], [], []
    for i, _ in enumerate(names):
        label = rng.integers(0, 3, n_per)
        feat = rng.normal(size=(n_per, 6)) + label[:, None] * np.array([3, 0, 0, 0, 0, 0])   # action signal only
        X.append(feat); y.append(label); rec.append(np.full(n_per, i)); codes.append(label * 100 + rng.integers(0, 3, n_per)); lens.append(np.full(n_per, 12.0))
    w = dict(names=names, X=np.concatenate(X), y=np.concatenate(y), rec=np.concatenate(rec), codes=np.concatenate(codes),
             lengths=np.concatenate(lens), subject=np.array([R_ for R_ in (n.split("_")[1] for n in names)]), n_classes=3)
    out = P.run_probes(w, cap=5000)
    assert out["action_probe_continuous"] > 85 and out["action_probe_code_table"] > 95
    assert out["subject_probe_accuracy"] < out["subject_probe_chance"] + 15           # no subject signal was planted


def rows_for(mof_new, f1_new, probe, mof_t2=(40.0, 40.0, 40.0), f1_t2=(20.0, 20.0, 20.0)):
    rows = {}
    for i, s in enumerate(C.SEEDS):
        rows[("t2", s)] = dict(MoF=mof_t2[i], F1_50=f1_t2[i], action_probe_continuous=probe)
        for v in C.NEW_VARIANTS:
            rows[(v, s)] = dict(MoF=mof_new[i], F1_50=f1_new[i], action_probe_continuous=probe)
    return rows


@test
def decision_rules():
    d = R.decide(rows_for((44, 44.5, 45), (24, 24, 24.2), 60), False)
    assert d["verdict"] == "ADOPT_INPUT"
    d = R.decide(rows_for((44, 44.5, 39.9), (24, 24, 24.2), 60), False)          # one seed not positive on MoF
    assert d["verdict"] != "ADOPT_INPUT"
    d = R.decide(rows_for((41, 41, 41), (21, 21, 21), 60), False)                # +1: within the bar; probe 60 vs MoF 40 -> gap 20
    assert d["verdict"] == "ORGANISATION_PROBLEM"
    d = R.decide(rows_for((41, 41, 41), (21, 21, 21), 48), False)                # gap 8
    assert d["verdict"] == "REPRESENTATION_LIMIT"
    bad = rows_for((41, 41, 41), (21, 21, 21), 48)
    del bad[("t2_centered", 222)]
    assert R.decide(bad, False)["verdict"] == "INCONCLUSIVE"
    assert R.decide(rows_for((50, 50, 50), (30, 30, 30), 60), True)["verdict"].startswith("not computed")


@test
def installation_and_isolation():
    from script.week5.timescale_inspection import tokenizers as TOK
    F.install()
    F.install()
    for v in C.NEW_VARIANTS:
        assert C.VARIANTS[v][0] in TOK.SPECS and TOK.SPECS[C.VARIANTS[v][0]].family == C.family(v)
    assert "t2_relational" in TOK.SPECS                                          # the existing specs are untouched
    for mod in ("features.py", "config.py"):
        src = (C.PKG / mod).read_text(encoding="utf-8")
        assert "load_gts" not in src and "groundTruth" not in src, mod            # labels only enter through probes.py
    from script.compositional_dynamics import config as CD
    old_family, old_dir = CD.FAMILY, CD.prep_dir
    try:
        R.patch_compositional("t2_centered")
        assert CD.FAMILY == "tok5_t2c_centered" and CD.prep_dir(111) == C.prep_dir("t2_centered", 111)
    finally:
        CD.FAMILY, CD.prep_dir = old_family, old_dir
    assert C.prep_dir("t2", 111).name.startswith("lara__t2_relational__seed111")


@test
def declared_settings():
    assert C.BAR == 3.0 and C.GAP_THRESHOLD == 15.0 and C.SEEDS == (111, 222, 1538574472)
    assert set(C.NEW_VARIANTS) <= set(C.VARIANTS)


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
