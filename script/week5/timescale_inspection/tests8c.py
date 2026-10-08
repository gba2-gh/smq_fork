"""Tests for phase 8c. Run: python -m script.week5.timescale_inspection.tests8c"""

from __future__ import annotations

import contextlib
import json
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from script.week5.raw_kmeans_cross_dataset import datasets as D  # noqa: E402
from script.week5.timescale_inspection import report8a as R8  # noqa: E402
from script.week5.timescale_inspection import report8c as R8C  # noqa: E402
from script.week5.timescale_inspection import run_phase5 as R5  # noqa: E402
from script.week5.timescale_inspection import run_phase8c as R8CR  # noqa: E402
from script.week5.timescale_inspection import tokenizers as TOK  # noqa: E402
from script.week5.timescale_inspection import tokenizers8 as T8  # noqa: E402
from script.week5.timescale_inspection import tokenizers8c as T8C  # noqa: E402

TS = ROOT / "results" / "week5" / "timescale_inspection"
SRC = TS / "ts5_v1"


@contextlib.contextmanager
def scratch_runner(**kw):
    run_id = "_tests8c_scratch"
    out = TS / run_id
    shutil.rmtree(out, ignore_errors=True)
    args = Namespace(run_id=run_id, source_run="ts5_v1", hours=kw.get("hours"), smoke=kw.get("smoke", False), cpu=False,
                     dry_run=False)
    try:
        yield R8CR.Phase8cRunner(args)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def _skeleton(psi, lift=0.0):
    """Frames x 25 x 3 skeleton: hips and shoulders define a horizontal body axis at angle psi; one off-axis joint so
    the coordinates are not symmetric under reflection."""
    T = len(psi)
    pos = np.zeros((T, 25, 3))
    c, s = np.cos(psi), np.sin(psi)
    for l, r, half in ((12, 16, 100.0), (4, 8, 180.0)):
        pos[:, l, 0], pos[:, l, 1] = -half * c, -half * s
        pos[:, r, 0], pos[:, r, 1] = half * c, half * s
    pos[:, 3, :] = np.stack([-s * 80.0, c * 80.0, np.full(T, 500.0 + lift)], axis=1)  # head: in front of and above the body
    return pos.astype(np.float32)


def test_window_heading_uses_the_middle_frame_the_nearest_valid_frame_and_flags_invalid_windows():
    psi = np.arange(16, dtype=float)
    valid = np.ones(16, bool)
    h, rot = T8C.window_headings(psi, valid, 8)
    assert np.array_equal(h, [4.0, 12.0]) and rot.all()
    valid[4] = False  # middle frame of window 0 invalid: the nearest valid frame is 3 or 5 (3 first on ties)
    h, rot = T8C.window_headings(psi, valid, 8)
    assert h[0] in (3.0, 5.0) and rot.all()
    valid[8:16] = False
    h, rot = T8C.window_headings(psi, valid, 8)
    assert h[1] == 0.0 and rot.tolist() == [True, False]
    return True


def test_canonicalization_removes_heading_but_keeps_gravity_chirality_and_within_window_turning():
    fps, W = 30.0, 8
    base = _skeleton(np.full(32, 0.4))
    c0, st0 = T8C.canonical_positions(base, "babel2", W)
    assert st0["windows_unrotated"] == 0
    for ang in (0.9, -2.2):  # any constant yaw gives identical canonical coordinates
        c1, _ = T8C.canonical_positions(T8.rotate_z(base, ang), "babel2", W)
        assert np.abs(c0 - c1).max() < 1e-3
    # canonical body axis points along +axis 0 at every window's middle frame
    sk = T8.skeleton("babel2")
    psi, valid = T8.body_heading(c0, sk)
    assert np.abs(psi[[4, 12, 20, 28]]).max() < 1e-5
    # gravity kept: lowering the head changes the canonical coordinates; chirality kept: a mirror image differs
    c_low, _ = T8C.canonical_positions(_skeleton(np.full(32, 0.4), lift=-50.0), "babel2", W)
    assert np.abs(c0 - c_low).max() > 40.0
    mir = base.copy()
    mir[..., 0] *= -1
    c_m, _ = T8C.canonical_positions(mir, "babel2", W)
    assert np.abs(c0 - c_m).max() > 10.0
    # within-window turning kept: a turn at 1 rad/s leaves non-zero spread of the heading inside the canonical window
    turn = _skeleton(0.4 + 1.0 * np.arange(32) / fps)
    ct, _ = T8C.canonical_positions(turn, "babel2", W)
    pt, _ = T8.body_heading(ct, sk)
    assert np.abs(pt[0:8]).max() > 0.05 and abs(pt[4]) < 1e-5
    return True


def test_rotation_disabled_leaves_positions_untouched_and_t6_layout_is_t0s():
    spec = TOK.dataset_spec("babel2")
    name = D.recording_names(spec, 1)[0]
    pos = T8.load_positions(spec, name)
    c, st = T8C.canonical_positions(pos, "babel2", spec.window, rotate=False)
    assert np.array_equal(c, pos)
    arr = np.load(D.features_dir(spec) / name)
    from script.week5.raw_kmeans_cross_dataset import prepare as PREP
    t0 = PREP.extract_raw(spec, [name], "global")[name]
    mine = T8C.canon_flat(c, spec.window)
    assert np.array_equal(mine["flat"], t0["flat"]) and np.array_equal(mine["lengths"], t0["lengths"])
    return True


def test_feature_widths_t6_t6s_and_unweighted_t5():
    for ds, V in (("babel2", 25), ("lara", 22)):
        spec = TOK.dataset_spec(ds)
        name = D.recording_names(spec, 1)[0]
        pos = T8.load_positions(spec, name)
        cpos, _ = T8C.canonical_positions(pos, ds, spec.window)
        assert T8C.canon_flat(cpos, spec.window)["flat"].shape[1] == 3 * V * spec.window
        assert T8C.canon_stats_features(cpos, ds, spec.window).shape[1] == 9 * (V - 1)
    assert 9 * 24 == 216 and 9 * 21 == 189
    out, st = T8C.extract8c(TOK.dataset_spec("babel2"), "relational+vertical+heading_std", D.recording_names(TOK.dataset_spec("babel2"), 1))
    assert next(iter(out.values()))["flat"].shape[1] == 978
    return True


def test_real_recordings_pass_the_reduction_and_invariance_checks():
    for ds in ("babel2", "lara"):
        r = T8C.reduction_check(ds)
        assert r["status"] == "passed", (ds, r)
        v = T8C.invariance_check(ds)
        assert v["status"] == "passed", (ds, v)
        assert v["fixed_rotation_max_abs_diff"] < 1e-2 and v["turn_max_abs_diff"] > 10 and v["mirror_max_abs_diff"] > 10
    return True


def test_unweighted_fit_never_reweights_and_reports_pre_pca_shares():
    spec = TOK.dataset_spec("babel2")
    names = D.recording_names(spec, 30)
    raw, st = T8C.extract8c(spec, "relational+vertical+heading_std", names)
    ids = [(n, 0) for n in names][:30]
    width = 978
    fit, w = T8.fit_population8("t", raw, ids * 20, 8, 16, 0, {"all": slice(0, width)})
    assert w == {}
    sl = T8C.share_slices("relational+vertical+heading_std", "babel2", width)
    assert (sl["dist"].stop, sl["vertical"].stop, sl["heading"].stop) == (900, 975, 978)
    return True


def test_keys_register_without_touching_earlier_tokenizers_and_no_labels_are_read():
    before = {k: TOK.SPECS[k] for k in ("t0_raw", "t2_relational", "t4_smq", "t5_both", "t5_gravity", "t5_heading")}
    T8C.register_8c_tokenizers()
    assert all(TOK.SPECS[k] is v for k, v in before.items())
    for k in T8C.CANON_KEYS:
        s = TOK.SPECS[k]
        assert s.span == 2 and s.guard == 0 and not s.selectable and s.pca_dim == 64 and not s.whiten
        assert {"babel2", "babel3", "lara"} <= set(s.datasets) and "hugadb" not in s.datasets
    assert TOK.SPECS["t6_norot"].datasets == ("babel2",) and TOK.SPECS["t5_dist_only_std"].datasets == ("babel2",)
    assert not any(k in TOK.SELECTABLE for k in T8C.CANON_KEYS)
    src = Path(T8C.__file__).read_text(encoding="utf-8")
    assert "groundTruth" not in src and "load_gts" not in src and "mapping" not in src
    return True


def test_stage_contents_order_and_identities():
    A, B = R8CR.stage_items("A"), R8CR.stage_items("B")
    assert len(A) == 20 and len(B) == 6
    assert [it.cell.seed for it in A] == sorted([it.cell.seed for it in A], key=R8CR.SEEDS.index)  # ordered by seed
    assert [it.cell.seed for it in B] == [111, 111, 222, 222, 1538574472, 1538574472]
    weighted = [it.cell for it in A if it.cell.tok == "t5_both"]
    assert [(c.dataset, c.seed) for c in weighted] == [("babel2", 1538574472), ("babel3", 1538574472)]
    assert {it.cell.tok for it in A} == set(R8CR.BABEL_TOKENS) | {"t5_both"} and {it.cell.tok for it in B} == set(R8CR.LARA_TOKENS)
    assert all(it.arms == ("context",) and it.cell.span == 2 and it.cell.guard == 0 for it in A + B)
    with scratch_runner() as r:
        ids = [r.identity_of(c) for c in r.base_cells]
        assert len(set(ids)) == len(ids) == 26
    return True


def test_blocking_rules_follow_the_gates():
    with scratch_runner(hours=1e-9) as r:
        ok = dict(status="passed")
        r.gates = {g: ok for g in ("g8c1", "g8c2", "g8c3", "g8c4", "g8b_babel2", "g8b_babel3", "g8b_lara", "g3")}
        mk = R5.mk_cell
        assert all(r.blocked(mk(ds, t, 111)) is None for ds in ("babel2", "lara") for t in R8CR.LARA_TOKENS)
        r.gates["g8c1"] = dict(status="failed")
        assert r.blocked(mk("babel2", "t6_heading_canonical", 111)) and r.blocked(mk("babel2", "t6s_heading_canonical_stats", 111))
        assert r.blocked(mk("babel2", "t5_both_std", 111)) is None
        r.gates["g8c1"], r.gates["g8c2"] = ok, dict(status="failed")
        assert r.blocked(mk("babel2", "t6s_heading_canonical_stats", 111)) and r.blocked(mk("babel2", "t6_heading_canonical", 111)) is None
        r.gates["g8c4"] = dict(status="failed")
        assert r.blocked(mk("babel3", "t5_both_std", 111)) and r.blocked(mk("babel3", "t5_both", 111)) is None
        r.gates["g8c3"] = dict(status="failed")
        assert r.blocked(mk("babel3", "t6_heading_canonical", 111))
        r.gates["g3"] = dict(status="failed")
        assert r.blocked(mk("lara", "t6_heading_canonical", 111))
        counts: dict = {}
        r.gates["g8c3"] = r.gates["g3"] = ok
        r.run_item(R8CR.stage_items("A")[0], counts, "A")
        assert counts == {"deadline": 1} and not any((r.out / "prep").rglob("manifest.json"))
    return True


def test_references_exist_for_every_comparison():
    have = R8C.cells(TS / "_nonexistent", SRC)
    for ds in ("babel2", "babel3"):
        assert sorted(have["t0"][ds]) == sorted(R8C.SEEDS), ds
        assert sorted(have["t2"][ds]) == sorted(R8C.SEEDS)
        assert {111, 222} <= set(have["t5w"][ds]), ds
    assert sorted(have["t2"]["lara"]) == sorted(R8C.SEEDS)
    return True


def _unit(root: Path, name: str, seed_dir: str, f1: float, **extra):
    d = root / "units" / seed_dir / "info.json"
    d.parent.mkdir(parents=True, exist_ok=True)
    m = dict(MoF=40.0, Edit=30.0, F1_10=35.0, F1_25=32.0, F1_50=f1)
    d.write_text(json.dumps(dict(status="done", ladder=[dict(rung_steps=4000, **m)],
                                 segment_stats=dict(mean_pred_segments=3.0, mean_gt_segments=4.0), **extra)))


def test_report_readings_pass_reverse_within_and_incomplete():
    tmp = TS / "_tests8c_report"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        out, src = tmp / "out", tmp / "src"
        (out / "prep").mkdir(parents=True)
        for ds in ("babel2", "babel3", "lara"):
            for s in R8C.SEEDS:
                phase, cid = R8.t2_cell_id(ds, s)
                _unit(src / phase, "", f"{cid}__context", 30.0)
        for ds in ("babel2", "babel3"):
            for s in R8C.SEEDS:
                _unit(src / "phase6", "", f"{ds}__t0_raw__seed{s}__span2__context", 20.0)
                _unit(src / "phase8a", "", f"{ds}__t5_both__seed{s}__context", 25.0)
        for s in R8C.SEEDS:  # Q1: T6 beats T0 on babel2 by the bar on every seed
            _unit(out, "", f"babel2__t6_heading_canonical__seed{s}__context", 24.0)
        for s in R8C.SEEDS:  # Q3 babel2: std worse than weighted by 5 on every seed -> reverse pass
            _unit(out, "", f"babel2__t5_both_std__seed{s}__context", 20.0)
        for s in (111, 222):  # incomplete
            _unit(out, "", f"babel3__t6_heading_canonical__seed{s}__context", 29.0)
        for s, f in zip(R8C.SEEDS, (36.0, 36.0, 20.0)):  # Q2 babel2: mean +1.33, mixed -> within the bar
            _unit(out, "", f"babel2__t6s_heading_canonical_stats__seed{s}__context", f)
        (out / "gates.json").write_text("{}")
        (out / "manifest.json").write_text(json.dumps(dict(run_id="x", status="finished")))
        R8C.write_report(out, src)
        text = (out / "REPORT.md").read_text(encoding="utf-8")

        def line(cid, ds):
            sec = text.split(f"### {cid}")[1].split("###")[0]
            return next(l for l in sec.split("\n") if f"**{ds}**" in l)
        assert "A better (passes the bar)" in line("Q1", "babel2") and "+4.00" in line("Q1", "babel2")
        assert "B better (reverse passes the bar)" in line("Q3", "babel2")
        assert "within the bar: comparable performance, not equivalence" in line("Q2", "babel2")
        assert "incomplete (2 seeds)" in line("D1", "babel3")
        assert "equivalent" not in text.replace("not equivalence", "").lower()
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_g8c1_t6_without_rotation_and_g8c4_unweighted_distance_path_reproduce_earlier_preparations():
    with scratch_runner() as r:
        a = r.gate_g8c1()
        assert a["status"] == "passed" and all(a["checks"].values()), a
        b = r.gate_g8c4()
        assert b["status"] == "passed" and all(b["checks"].values()), b
    return True


ALL_TESTS = [
    test_window_heading_uses_the_middle_frame_the_nearest_valid_frame_and_flags_invalid_windows,
    test_canonicalization_removes_heading_but_keeps_gravity_chirality_and_within_window_turning,
    test_rotation_disabled_leaves_positions_untouched_and_t6_layout_is_t0s,
    test_feature_widths_t6_t6s_and_unweighted_t5,
    test_real_recordings_pass_the_reduction_and_invariance_checks,
    test_unweighted_fit_never_reweights_and_reports_pre_pca_shares,
    test_keys_register_without_touching_earlier_tokenizers_and_no_labels_are_read,
    test_stage_contents_order_and_identities,
    test_blocking_rules_follow_the_gates,
    test_references_exist_for_every_comparison,
    test_report_readings_pass_reverse_within_and_incomplete,
    test_g8c1_t6_without_rotation_and_g8c4_unweighted_distance_path_reproduce_earlier_preparations,
]

if __name__ == "__main__":
    failed = 0
    for t in ALL_TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            import traceback
            failed += 1
            print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"{len(ALL_TESTS) - failed}/{len(ALL_TESTS)} passed")
    sys.exit(1 if failed else 0)
