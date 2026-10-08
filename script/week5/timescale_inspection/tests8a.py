"""Tests for phase 8a. Run: python -m script.week5.timescale_inspection.tests8a"""

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
from script.week5.timescale_inspection import run_phase5 as R5  # noqa: E402
from script.week5.timescale_inspection import run_phase8a as R8A  # noqa: E402
from script.week5.timescale_inspection import tokenizers as TOK  # noqa: E402
from script.week5.timescale_inspection import tokenizers8 as T8  # noqa: E402

TS = ROOT / "results" / "week5" / "timescale_inspection"
SRC = TS / "ts5_v1"


@contextlib.contextmanager
def scratch_runner(**kw):
    run_id = "_tests8a_scratch"
    out = TS / run_id
    shutil.rmtree(out, ignore_errors=True)
    args = Namespace(run_id=run_id, source_run="ts5_v1", hours=kw.get("hours"), smoke=kw.get("smoke", False), cpu=False,
                     dry_run=False)
    try:
        yield R8A.Phase8aRunner(args)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_block_layout_and_declared_feature_counts():
    for ds, dist, vert in (("babel2", 900, 75), ("lara", 693, 63)):
        for blocks, want in ((("dist",), dist), (("dist", "vertical"), dist + vert), (("dist", "heading"), dist + 3),
                             (("dist", "vertical", "heading"), dist + vert + 3)):
            sl = T8.block_slices(blocks, ds)
            assert sum(s.stop - s.start for s in sl.values()) == want, (ds, blocks)
            assert sl["dist"] == slice(0, dist)
    assert T8.BLOCKS_BY_SOURCE["relational"] == ("dist",)
    return True


def test_extracted_features_match_the_layout_on_real_recordings():
    for ds in ("babel2", "lara"):
        spec = TOK.dataset_spec(ds)
        name = D.recording_names(spec, 1)[0]
        pos = T8.load_positions(spec, name)
        for source, blocks in T8.BLOCKS_BY_SOURCE.items():
            flat, _ = T8.window_features(pos, blocks, ds, spec.window, spec.fps)
            assert flat.shape[1] == sum(s.stop - s.start for s in T8.block_slices(blocks, ds).values()), (ds, source)
            assert np.isfinite(flat).all()
        # the distance block is byte-identical to T2's extraction
        t2 = TOK.extract_relational(spec, [name])[name]["flat"]
        mine, _ = T8.window_features(pos, ("dist", "vertical", "heading"), ds, spec.window, spec.fps)
        assert np.array_equal(mine[:, :t2.shape[1]], t2)
    return True


def test_lara_height_block_drops_the_duplicate_collar():
    spec = TOK.dataset_spec("lara")
    name = D.recording_names(spec, 1)[0]
    pos = T8.load_positions(spec, name)
    assert np.abs(pos[:, 2] - pos[:, 12]).max() < 1e-3  # the two collars are the same point
    v = T8.vertical_frames(pos, T8.skeleton("lara"))
    assert v.shape[1] == 20 + 1  # 21 non-root segments minus the duplicate, plus trunk tilt
    keep = [j for j in range(22) if j != 21 and j != 12]
    assert np.array_equal(v[:, :20], pos[:, keep, 2])
    b = T8.vertical_frames(T8.load_positions(TOK.dataset_spec("babel2"), D.recording_names(TOK.dataset_spec("babel2"), 1)[0]),
                           T8.skeleton("babel2"))
    assert b.shape[1] == 24 + 1
    return True


def _skeleton_axis(psi):
    """Frames x 25 x 3 skeleton whose hips and shoulders define a horizontal body axis at angle psi (per frame)."""
    T = len(psi)
    pos = np.zeros((T, 25, 3))
    c, s = np.cos(psi), np.sin(psi)
    for l, r, half in ((12, 16, 100.0), (4, 8, 180.0)):
        pos[:, l, 0], pos[:, l, 1] = -half * c, -half * s
        pos[:, r, 0], pos[:, r, 1] = half * c, half * s
    return pos


def test_heading_rate_recovers_a_constant_turn_including_across_the_wrap():
    fps, W = 30.0, 8
    for omega in (1.0, -2.5):
        psi = 3.0 + omega * np.arange(64) / fps  # crosses +pi for omega > 0
        pos = _skeleton_axis(psi)
        ang, valid = T8.body_heading(pos, T8.skeleton("babel2"))
        assert valid.all()
        st = T8.heading_stats(ang, valid, W, fps)
        assert np.allclose(st[:, 0], omega, atol=1e-4), st[:, 0]
        assert np.allclose(st[:, 1], abs(omega), atol=1e-4) and np.allclose(st[:, 2], 0, atol=1e-4)
    return True


def test_heading_is_undefined_when_the_body_axis_is_vertical_and_windows_never_borrow_frames():
    fps, W = 30.0, 8
    psi = np.zeros(32)
    pos = _skeleton_axis(psi)
    lying = pos.copy()
    lying[8:16] = pos[8:16][:, :, [2, 1, 0]]  # swap x and the vertical axis: the body axis now points along axis 2
    ang, valid = T8.body_heading(lying, T8.skeleton("babel2"))
    assert not valid[8:16].any() and valid[:8].all() and valid[16:].all()
    st = T8.heading_stats(ang, valid, W, fps)
    assert np.all(st[1] == 0)  # window 1 has no valid consecutive pair
    # a heading jump between windows must not leak into either window's rate
    psi2 = np.concatenate([np.zeros(8), np.full(8, 1.0)])
    a2, v2 = T8.body_heading(_skeleton_axis(psi2), T8.skeleton("babel2"))
    st2 = T8.heading_stats(a2, v2, W, fps)
    assert np.allclose(st2, 0, atol=1e-6)
    return True


def test_real_recording_is_yaw_invariant_and_a_turn_adds_exactly_omega():
    for ds in ("babel2", "lara"):
        r = T8.invariance_check(ds)
        assert r["status"] == "passed", (ds, r)
        assert r["turn_max_abs_err"] < 1e-4 and r["windows_tested"] > 5
    return True


def test_block_weighting_equalizes_first_singular_values_and_leaves_the_distance_block_alone():
    rng = np.random.default_rng(0)
    n = 400
    a = rng.normal(size=(n, 30)) @ rng.normal(size=(30, 30)) * 5.0
    b = rng.normal(size=(n, 6)) @ rng.normal(size=(6, 6)) * 0.3
    flat = np.concatenate([a, b], axis=1).astype(np.float32)
    ids = [("r", i) for i in range(n)]
    pop = {"r": dict(flat=flat)}
    slices = {"dist": slice(0, 30), "vertical": slice(30, 36)}
    fit, w = T8.fit_population8("t", pop, ids, 8, 16, 0, slices)
    std = fit.scaler.transform(flat)
    s_a = np.linalg.svd(std[:, :30], compute_uv=False)[0]
    s_b = np.linalg.svd(std[:, 30:], compute_uv=False)[0]
    assert abs(s_a - s_b) / s_a < 1e-4, (s_a, s_b)
    plain = TOK.fit_population("t", pop, ids, 8, 16, 0)
    assert np.allclose(fit.scaler.scale_[:30], plain.scaler.scale_[:30]) and w["dist"]["weight"] == 1.0
    one, _ = T8.fit_population8("t", pop, ids, 8, 16, 0, {"dist": slice(0, 36)})
    assert np.array_equal(one.kmeans.cluster_centers_, plain.kmeans.cluster_centers_)  # one block == T2's path
    return True


def test_t5_keys_register_without_touching_earlier_tokenizers():
    before = {k: TOK.SPECS[k] for k in ("t0_raw", "t1_whiten", "t2_relational", "t3_lara_pos256", "t3b_lara_pos64", "t4_smq")}
    T8.register_t5_tokenizers()
    assert all(TOK.SPECS[k] is v for k, v in before.items())
    for k in ("t5_both", "t5_heading", "t5_gravity"):
        s = TOK.SPECS[k]
        assert s.span == 2 and s.guard == 0 and not s.selectable and s.pca_dim == 64 and not s.whiten
        assert s.family == f"tok5_{k}" and "babel2" in s.datasets and "lara" in s.datasets and "hugadb" not in s.datasets
    assert TOK.SELECTABLE == tuple(k for k, s in TOK.SPECS.items() if s.selectable)
    assert "t5_both" not in TOK.SELECTABLE
    return True


def test_extraction_reads_no_labels():
    src = Path(T8.__file__).read_text(encoding="utf-8")
    assert "groundTruth" not in src and "load_gts" not in src and "mapping" not in src
    return True


def test_stage_contents_and_cell_identities():
    A, B = R8A.stage_items("A"), R8A.stage_items("B")
    assert len(A) == 12 and len(B) == 2
    assert [it.cell.tok for it in A[:4]] == ["t5_both"] * 4  # T5gh first, on both datasets and both seeds
    assert {it.cell.seed for it in A + B} == {111, 222}
    assert all(it.arms == ("context",) and it.cell.span == 2 and it.cell.guard == 0 for it in A + B)
    assert {it.cell.dataset for it in B} == {"lara"} and {it.cell.tok for it in B} == {"t5_both"}
    assert R8A.stage_items("C", decisions={}) == []
    dec = {"babel2|t5_both": dict(run_third_seed=True), "babel3|t5_both": dict(run_third_seed=False),
           "lara|t5_both": dict(run_third_seed=True)}
    C = R8A.stage_items("C", decisions=dec)
    assert [(it.cell.dataset, it.cell.tok, it.cell.seed) for it in C] == [("babel2", "t5_both", 1538574472),
                                                                         ("lara", "t5_both", 1538574472)]
    assert len(R8A.stage_items("C")) == 7  # the identity / upper-bound set
    with scratch_runner() as r:
        ids = [r.identity_of(c) for c in r.base_cells]
        assert len(set(ids)) == len(ids) == 12 + 2 + 7
        a, b = (r.identity_of(R5.mk_cell("babel2", t, 111)) for t in ("t5_both", "t5_gravity"))
        assert a != b
        assert all(c.cell_id.count("__span") == 0 for c in r.base_cells)  # span 2 is the T5 default
    return True


def test_references_exist_for_every_screening_and_third_seed():
    for ds in R8.DATASETS:
        for s in R8.SEED_ORDER:
            ref = R8.t2_reference(SRC, ds, s)
            assert ref is not None and 0 < ref["F1_50"] < 100, (ds, s)
    assert R8.t2_cell_id("lara", 111)[0] == "phase6" and R8.t2_cell_id("babel3", 111)[0] == "phase5"
    return True


def _fake_unit(out: Path, ds: str, tok: str, seed: int, metrics: dict):
    d = out / "units" / f"{ds}__{tok}__seed{seed}__context"
    d.mkdir(parents=True, exist_ok=True)
    ladder = [dict(rung_steps=500, **{k: 0.0 for k in R8.METRICS}), dict(rung_steps=4000, **metrics)]
    (d / "info.json").write_text(json.dumps(dict(
        status="done", dataset=ds, tokenizer=tok, seed=seed, arm="context", span=2, window=D.SPECS[ds].window,
        ladder=ladder, segment_stats=dict(mean_pred_segments=3.0, mean_gt_segments=4.0))))


def _fake_t2(src: Path, ds: str, seed: int, f1: float):
    phase, cid = R8.t2_cell_id(ds, seed)
    d = src / phase / "units" / f"{cid}__context"
    d.mkdir(parents=True, exist_ok=True)
    m = dict(MoF=40.0, Edit=30.0, F1_10=35.0, F1_25=32.0, F1_50=f1)
    (d / "info.json").write_text(json.dumps(dict(status="done", ladder=[dict(rung_steps=4000, **m)],
                                                 segment_stats=dict(mean_pred_segments=3.0, mean_gt_segments=4.0))))


def test_third_seed_rule_boundary_and_incomplete_screening():
    tmp = TS / "_tests8a_rule"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        out, src = tmp / "out", tmp / "src"
        base = dict(MoF=40.0, Edit=30.0, F1_10=35.0, F1_25=32.0)
        for ds in ("babel2", "babel3"):
            for s in (111, 222):
                _fake_t2(src, ds, s, 30.0)
        for tok, gain in (("t5_both", 1.0), ("t5_heading", 0.99), ("t5_gravity", 5.0)):  # exactly the threshold passes
            for s in (111, 222):
                _fake_unit(out, "babel2", tok, s, dict(base, F1_50=30.0 + gain))
        for s in (111, 222):  # a big gain on one seed and a loss on the other averages to +1.0
            _fake_unit(out, "babel3", "t5_both", s, dict(base, F1_50=30.0 + (11.0 if s == 111 else -9.0)))
        _fake_unit(out, "babel3", "t5_gravity", 111, dict(base, F1_50=40.0))  # second screening seed missing
        dec = R8.third_seed_decisions(out, src)
        assert dec["babel2|t5_both"]["run_third_seed"] is True
        assert dec["babel2|t5_heading"]["run_third_seed"] is False and dec["babel2|t5_heading"]["reason"] == "below threshold"
        assert dec["babel2|t5_gravity"]["run_third_seed"] is True
        assert dec["babel3|t5_both"]["run_third_seed"] is True  # the rule is on the mean, as declared
        assert dec["babel3|t5_gravity"]["run_third_seed"] is False and dec["babel3|t5_gravity"]["reason"] == "screening seeds incomplete"
        assert (dec["babel2|t5_both"]["gain_f1_50"], dec["babel2|t5_both"]["threshold"]) == (1.0, 1.0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return True


def test_report_labels_two_seed_cells_as_screening_and_applies_the_bar_to_three():
    tmp = TS / "_tests8a_report"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        out, src = tmp / "out", tmp / "src"
        (out / "prep").mkdir(parents=True)
        base = dict(MoF=40.0, Edit=30.0, F1_10=35.0, F1_25=32.0)
        for s in R8.SEED_ORDER:
            _fake_t2(src, "babel2", s, 30.0)
            _fake_t2(src, "babel3", s, 30.0)
        for s in (111, 222):
            _fake_unit(out, "babel2", "t5_both", s, dict(base, F1_50=36.0))
            _fake_unit(out, "babel3", "t5_both", s, dict(base, F1_50=36.0))
        for s in R8.SEED_ORDER:
            _fake_unit(out, "babel3", "t5_gravity", s, dict(base, F1_50=34.0))
        _fake_unit(out, "babel3", "t5_heading", 111, dict(base, F1_50=40.0))
        _fake_unit(out, "babel3", "t5_heading", 222, dict(base, F1_50=28.0))
        _fake_unit(out, "babel3", "t5_heading", 1538574472, dict(base, F1_50=40.0))
        (out / "gates.json").write_text("{}")
        (out / "manifest.json").write_text(json.dumps(dict(run_id="x", status="finished")))
        R8.write_report(out, src)
        text = (out / "REPORT.md").read_text(encoding="utf-8")
        row = {}
        for l in text.split("\n"):
            if l.startswith("| babel"):
                row.setdefault(l.split("|")[1].strip() + "/" + l.split("|")[2].strip(), l)  # first table = the main one
        assert "screening only (2 seeds)" in row["babel2/T5gh"]
        assert "**passes the bar**" in row["babel3/T5g"]
        assert "does not pass the bar" in row["babel3/T5h"]  # mean +4.0 but one seed negative
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_dry_run_projects_under_the_budget_and_blocks_work_past_the_deadline():
    with scratch_runner(hours=1e-9) as r:
        counts: dict = {}
        r.gates = {"g8a": dict(status="passed"), "g8c": dict(status="passed"), "g8b_babel2": dict(status="passed"),
                   "g3": dict(status="passed")}
        r.run_item(R8A.stage_items("A")[0], counts, "A")
        assert counts == {"deadline": 1}
        assert not any((r.out / "prep").rglob("manifest.json"))
        r.gates["g8b_babel2"] = dict(status="failed")
        assert r.blocked(R5.mk_cell("babel2", "t5_both", 111)) is not None
        assert r.blocked(R5.mk_cell("babel3", "t5_both", 111)) is None
        r.gates["g3"] = dict(status="failed")
        assert r.blocked(R5.mk_cell("lara", "t5_both", 111)) is not None
    return True


def test_g8a_t5_path_with_only_the_distance_block_reproduces_t2_on_real_data():
    with scratch_runner() as r:
        res = r.gate_g8a()
        assert res["status"] == "passed", res
        assert all(res["checks"].values()) and res["lookup_max_abs_diff"] <= 1e-4
        # nothing was written upstream
        assert not (SRC / "phase5" / "prep" / "t2_relational" / "babel2__t2_relational__seed111" / "t5_blocks.json").exists()
    return True


def test_geometry_gate_passes_on_real_data_and_fails_on_a_wrong_joint_map():
    for ds in ("babel2", "babel3", "lara"):
        assert T8.geometry_check(ds, 20)["status"] == "passed", ds
    wrong = T8.Skeleton(root=1, hips=(4, 8), shoulders=(12, 16), trunk=(0, 2), drop=(), head=3, feet=(15, 19),
                        order=(3, (12, 16), (4, 8), (13, 17), (15, 19)))  # hips/shoulders swapped
    old = T8.SKELETONS["babel"]
    T8.SKELETONS["babel"] = wrong
    try:
        assert T8.geometry_check("babel2", 20)["status"] == "failed"
    finally:
        T8.SKELETONS["babel"] = old
    return True


ALL_TESTS = [
    test_block_layout_and_declared_feature_counts,
    test_extracted_features_match_the_layout_on_real_recordings,
    test_lara_height_block_drops_the_duplicate_collar,
    test_heading_rate_recovers_a_constant_turn_including_across_the_wrap,
    test_heading_is_undefined_when_the_body_axis_is_vertical_and_windows_never_borrow_frames,
    test_real_recording_is_yaw_invariant_and_a_turn_adds_exactly_omega,
    test_block_weighting_equalizes_first_singular_values_and_leaves_the_distance_block_alone,
    test_t5_keys_register_without_touching_earlier_tokenizers,
    test_extraction_reads_no_labels,
    test_stage_contents_and_cell_identities,
    test_references_exist_for_every_screening_and_third_seed,
    test_third_seed_rule_boundary_and_incomplete_screening,
    test_report_labels_two_seed_cells_as_screening_and_applies_the_bar_to_three,
    test_dry_run_projects_under_the_budget_and_blocks_work_past_the_deadline,
    test_geometry_gate_passes_on_real_data_and_fails_on_a_wrong_joint_map,
    test_g8a_t5_path_with_only_the_distance_block_reproduces_t2_on_real_data,
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
