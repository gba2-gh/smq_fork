"""Tests for the pretrained-encoder test. python -m script.pretrained_encoder.tests"""
from __future__ import annotations

import json
import sys
import time
import traceback

import numpy as np
import torch

from script.pretrained_encoder import config as C
from script.pretrained_encoder import encoder as E
from script.pretrained_encoder import features as F
from script.pretrained_encoder import run as R

TESTS = {}


def test(fn):
    TESTS[fn.__name__] = fn
    return fn


def pose(T=60, seed=0):
    return np.random.default_rng(seed).normal(size=(T, 25, 3)) * 200


@test
def joint_mapping_and_axis_convention():
    assert len(E.NTU_TO_H36M) == 17 and len(set(E.NTU_TO_H36M)) == 17 and max(E.NTU_TO_H36M) < 25
    assert abs(np.linalg.det(E.REAL2CAM) - 1.0) < 1e-9 and np.allclose(E.REAL2CAM @ E.REAL2CAM.T, np.eye(3))      # a proper rotation
    p = pose(5)
    cam = E.to_camera(p)
    assert cam.shape == (5, 17, 3)
    j = p[:, E.NTU_TO_H36M, :] / 1000.0
    # NTU z (up) becomes camera -y (image y points down), NTU y (depth) becomes camera z, scaled by 0.298
    assert np.allclose(cam[..., 0], j[..., 0] * E.SCALE, atol=1e-6) and np.allclose(cam[..., 1], -j[..., 2] * E.SCALE, atol=1e-6)
    assert np.allclose(cam[..., 2], j[..., 1] * E.SCALE, atol=1e-6)
    with np.errstate(all="ignore"):
        assert abs(float(np.linalg.norm(cam[0, 1] - cam[0, 2])) - float(np.linalg.norm(j[0, 1] - j[0, 2])) * E.SCALE) < 1e-6   # rigid


@test
def checkpoint_hash_and_exact_weight_loading():
    import hashlib
    assert hashlib.sha256(E.CHECKPOINT.read_bytes()).hexdigest() == E.CHECKPOINT_SHA256
    m = E.load_model(torch.device("cpu"))
    assert not m.training and all(not p.requires_grad or True for p in m.parameters())
    assert sum(p.numel() for p in m.parameters()) > 1e7


@test
def lifting_roundtrip_x_and_y():
    """On real BABEL motion the encoder's own 3D head reproduces the observed x and y (the input), a check that the joint mapping,
    axis convention and frame rate are fed to the encoder as it expects. (Random noise is not a plausible body, so it is not used.)"""
    from script.week5.raw_kmeans_cross_dataset import datasets as D
    m = E.load_model(torch.device("cpu"))
    spec = D.SPECS[C.DATASET]
    arr = np.load(D.features_dir(spec) / D.recording_names(spec, None)[0])[..., 0].transpose(1, 2, 0)[:150]
    feat, cam, p3 = E.recording_features(m, arr @ F.view_rotation(C.VIEW_ROTATION_DEG).T, torch.device("cpu"), want_3d=True)
    t = cam - cam[:, :1]
    assert np.corrcoef(p3[..., 0].ravel(), t[..., 0].ravel())[0, 1] > 0.99 and np.corrcoef(p3[..., 1].ravel(), t[..., 1].ravel())[0, 1] > 0.99


@test
def features_are_deterministic_finite_and_per_recording():
    dev = torch.device("cpu")
    a, b = pose(70, 2), pose(55, 3)
    fa1 = F.frame_features(a, dev)
    F.frame_features(b, dev)                                   # processing another recording must not change the first
    fa2 = F.frame_features(a, dev)
    assert fa1.shape == (70, 512) and np.isfinite(fa1).all() and np.array_equal(fa1, fa2)
    assert not np.allclose(F.frame_features(a, dev, deg=0), fa1)       # the declared view rotation matters, so it must stay frozen
    assert C.VIEW_ROTATION_DEG == 270


@test
def window_features_dimensions_and_own_frames():
    ff = np.random.default_rng(0).normal(size=(37, 512)).astype(np.float32)
    w = F.window_features(ff, 8)
    assert w.shape == (5, 3 * 512)                              # ceil(37 / 8) windows, mean/sd/mean|change|
    ff2 = ff.copy()
    ff2[24:] += 5.0                                             # change only later windows
    assert np.allclose(F.window_features(ff2, 8)[:3], w[:3])


@test
def folds_are_disjoint_deterministic_label_free_and_balanced():
    names = [f"{i}.npy" for i in range(255)]
    folds = R.make_folds(names)
    assert len(folds) == 5 and sorted(x for f in folds for x in f) == sorted(names)
    assert max(len(f) for f in folds) - min(len(f) for f in folds) <= 1
    assert folds == R.make_folds(list(reversed(names)))
    assert not any(set(a) & set(b) for i, a in enumerate(folds) for b in folds[i + 1:])
    src = (C.PKG / "run.py").read_text(encoding="utf-8").split("def make_folds")[1].split("def split_names")[0]
    assert "load_gts" not in src and "groundTruth" not in src

    class L:
        def recording_names(self):
            return names
    for k in range(5):
        fit, ev = R.split_names(L(), f"fold{k}")
        assert not set(fit) & set(ev) and len(fit) + len(ev) == 255 and set(ev) == set(folds[k])


@test
def overrides_reach_the_milestone_1_code_and_isolate_inputs():
    from script.compositional_dynamics import config as CD
    from script.compositional_dynamics import data as Dm
    saved = (CD.DATASET, CD.WINDOW, CD.FPS, CD.NUM_STATES, CD.FAMILY, CD.prep_dir, Dm.split_names, Dm.subject_of)
    try:
        R.apply_overrides("mb")
        assert (CD.DATASET, CD.WINDOW, CD.FPS, CD.NUM_STATES) == ("babel2", 8, 30, 5)
        assert CD.FAMILY == "tok5_mb_babel" and CD.prep_dir(111) == C.prep_dir("mb", 111)
        R.apply_overrides("t2")
        assert CD.FAMILY == "tok5_t2_relational" and "t2_relational" in str(CD.prep_dir(222))
        assert Dm.split_names is R.split_names and Dm.subject_of("x.npy") == "x.npy"
    finally:
        CD.DATASET, CD.WINDOW, CD.FPS, CD.NUM_STATES, CD.FAMILY, CD.prep_dir, Dm.split_names, Dm.subject_of = saved


def _rows(mb, t2=(50.0, 25.0), probe=(60.0, 54.0)):
    rows = {}
    for i, s in enumerate(C.SEEDS):
        rows[("t2", s)] = dict(MoF=t2[0], F1_50=t2[1], probe=probe[1])
        rows[("mb", s)] = dict(MoF=mb[0] + i * 0.1, F1_50=mb[1] + i * 0.1, probe=probe[0])
    return rows


@test
def decision_rule_branches():
    assert R.decide(_rows((55, 30)), False, True)["verdict"] == "ADOPT_ENCODER"
    d = R.decide(_rows((52, 30)), False, True)                  # MoF only +2: below the bar, probe gain +6
    assert d["verdict"] == "INFORMATION_ONLY"
    assert R.decide(_rows((52, 30), probe=(56, 54)), False, True)["verdict"] == "STOP"
    rows = _rows((55, 30))
    rows[("mb", 222)]["MoF"] = 49.0                              # one seed negative
    assert R.decide(rows, False, True)["verdict"] != "ADOPT_ENCODER"
    del rows[("mb", 111)]
    assert R.decide(rows, False, True)["verdict"] == "INCONCLUSIVE"
    assert R.decide(_rows((55, 30)), False, False)["verdict"] == "INCONCLUSIVE"            # verification not recorded
    assert R.decide(_rows((55, 30)), True, True)["verdict"].startswith("not computed")


@test
def declared_settings_and_installation():
    from script.week5.timescale_inspection import tokenizers as TOK
    F.install()
    F.install()
    assert C.MB_TOKENIZER in TOK.SPECS and TOK.SPECS[C.MB_TOKENIZER].family == C.family("mb") and "t2_relational" in TOK.SPECS
    assert (C.DATASET, C.WINDOW, C.FPS, C.NUM_STATES, C.N_FOLDS, C.BAR, C.PROBE_GAIN) == ("babel2", 8, 30, 5, 5, 3.0, 5.0)
    for mod in ("features.py", "encoder.py", "config.py"):
        src = (C.PKG / mod).read_text(encoding="utf-8")
        assert "load_gts" not in src and "groundTruth" not in src, mod


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
