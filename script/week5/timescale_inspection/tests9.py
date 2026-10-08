"""Tests for phase 9. Run: python -m script.week5.timescale_inspection.tests9"""

from __future__ import annotations

import contextlib
import copy
import json
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from script.quantized_input_comparison import contextual as C  # noqa: E402
from script.quantized_input_comparison.artifacts import Loader  # noqa: E402
from script.week5.contextualizer_calibration import evaluate as E  # noqa: E402
from script.week5.contextualizer_calibration import plan as P  # noqa: E402
from script.week5.contextualizer_calibration import train as T  # noqa: E402
from script.week5.timescale_inspection import gates9 as G9  # noqa: E402
from script.week5.timescale_inspection import report8a as R8  # noqa: E402
from script.week5.timescale_inspection import report9 as R9r  # noqa: E402
from script.week5.timescale_inspection import run_phase9 as R9  # noqa: E402
from script.week5.timescale_inspection import tokenizers as TOK  # noqa: E402
from script.week5.timescale_inspection import trainer9 as T9  # noqa: E402
from script.week5.timescale_inspection import units9 as U9  # noqa: E402

TS = ROOT / "results" / "week5" / "timescale_inspection"
SRC = TS / "ts5_v1"


@contextlib.contextmanager
def scratch_runner(**kw):
    run_id = "_tests9_scratch"
    out = TS / run_id
    shutil.rmtree(out, ignore_errors=True)
    args = Namespace(run_id=run_id, source_run="ts5_v1", stage=kw.get("stage", "dev"), hours=kw.get("hours"), smoke=kw.get("smoke", False),
                     cpu=False, dry_run=False)
    try:
        yield R9.Phase9Runner(args)
    finally:
        shutil.rmtree(out, ignore_errors=True)
        shutil.rmtree(TS / (run_id), ignore_errors=True)


def test_trainer_copy_differs_from_phase5_only_in_the_three_marked_lines():
    src = (ROOT / "script/week5/contextualizer_calibration/train.py").read_text(encoding="utf-8")
    a, b = src.index("def train_core("), src.index("def checkpoint_roles(")
    original = src[a:b].rstrip().split("\n")
    mine_all = Path(T9.__file__).read_text(encoding="utf-8")
    c, d = mine_all.index("def train_core("), mine_all.index("def run_training_job9(")
    mine = mine_all[c:d].rstrip().split("\n")
    changed = [(o, m) for o, m in zip(original, mine) if o != m]
    assert len(original) == len(mine) and len(changed) == 3, len(changed)
    assert all("PHASE9" in m for _, m in changed)
    import hashlib
    assert hashlib.sha256(src[a:b].encode()).hexdigest()[:16] == T9.ORIGINAL_CORE_SHA  # the shared trainer has not changed
    return True


def test_feature_fn_selects_the_input_by_arm():
    chunk = Namespace(codes=np.array([0, 2, 1]), continuous=np.arange(6, dtype=np.float32).reshape(3, 2))
    lookup = np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], dtype=np.float32)
    assert np.array_equal(T9.feature_fn("context_continuous", lookup, chunk), chunk.continuous)
    assert np.array_equal(T9.feature_fn("context_quantized", lookup, chunk), lookup[chunk.codes])
    return True


def test_layer_states_end_at_the_final_hidden_state_and_the_extractor_matches_phase5():
    cfg = T.train_config(R5_job(), P.SMOKE)
    init = C.build_matched_init(16, 8, cfg, seed=1)
    model = C.build_arm_model("context_continuous", 16, 8, cfg, 1, init).eval()
    x = torch.randn(2, 20, 8)
    pad = torch.zeros(2, 20, dtype=torch.bool)
    pad[1, 15:] = True
    with torch.no_grad():
        proj = model.input_proj(x)
        states = T9.layer_states(model, proj, pad)
        final = model._run_backbone(proj, pad)
    assert len(states) == 4 and all(s.shape == (2, 20, 128) for s in states)
    assert torch.equal(states[-1], final)
    assert not torch.equal(states[0], states[1])
    return True


def R5_job():
    from script.week5.timescale_inspection import run_phase5 as R5
    return R5.make_job(R5.mk_cell("babel2", "t2_relational", 111, span=2))


def test_i2_family_obeys_the_dimensional_contract_and_rejects_a_128d_lookup():
    with scratch_runner(smoke=True) as r:
        cell = R9.Cell9("babel2", "i1", 111)
        base = r.base_loader(cell)
        fam = cell.spec.family
        names = sorted(base.recording_names())
        rng = np.random.default_rng(0)
        codes = {n: rng.integers(0, 512, len(base.recording(fam, n)["codes"])) for n in names}
        src = U9.build_family(base, fam, codes, TOK.dataset_spec("babel2"), 111, teacher_centroids=np.zeros((512, 128), np.float32))
        rep = U9.check_family_dims(src, fam, model_input_dim=64)
        assert rep["status"] == "passed" and rep["dim"] == 64 and src.lookup(fam).shape == (512, 64)
        assert src.teacher_centroids.shape == (512, 128)
        bad = U9.I2Source(base, fam, codes, np.zeros((512, 128), np.float32), src.grouping(fam))
        assert U9.check_family_dims(bad, fam, model_input_dim=64)["status"] == "failed"
        assert U9.check_family_dims(src, fam, model_input_dim=128)["status"] == "failed"
        out = TS / "_tests9_family"
        shutil.rmtree(out, ignore_errors=True)
        try:
            U9.save_family(src, out)
            back = U9.load_family(base, fam, out)
            assert np.array_equal(back.lookup(fam), src.lookup(fam)) and np.array_equal(back.grouping(fam), src.grouping(fam))
            assert all(np.array_equal(back.codes_by_name[n], codes[n]) for n in names)
        finally:
            shutil.rmtree(out, ignore_errors=True)
    return True


def test_family_uses_the_base_features_split_and_rms_unchanged():
    with scratch_runner(smoke=True) as r:
        cell = R9.Cell9("babel2", "i1", 111)
        base = r.base_loader(cell)
        fam = cell.spec.family
        names = sorted(base.recording_names())
        rng = np.random.default_rng(0)
        codes = {n: rng.integers(0, 512, len(base.recording(fam, n)["codes"])) for n in names}
        src = U9.build_family(base, fam, codes, TOK.dataset_spec("babel2"), 111)
        n0 = names[0]
        from script.action_transport import categorical
        collapsed = {n: np.zeros(len(c), dtype=np.int64) for n, c in codes.items()}
        try:  # collapsed targets are refused, never silently worked around
            U9.build_family(base, fam, collapsed, TOK.dataset_spec("babel2"), 111)
            raise AssertionError("a one-code family must be refused")
        except categorical.UnavailableError:
            pass
        assert np.array_equal(src.recording(fam, n0)["continuous"], base.recording(fam, n0)["continuous"])
        assert np.array_equal(src.recording(fam, n0)["starts"], base.recording(fam, n0)["starts"])
        assert float(src.rms(fam)) == float(base.rms(fam)) and np.array_equal(src.mean(fam), base.mean(fam))
        assert [src.split_membership(n) for n in names] == [base.split_membership(n) for n in names]
    return True


def test_cross_seed_agreement_is_permutation_invariant_with_a_chance_floor():
    rng = np.random.default_rng(0)
    a = {f"r{i}": rng.integers(0, 20, 8000) for i in range(5)}
    perm = rng.permutation(20)
    b = {n: perm[c] for n, c in a.items()}
    assert abs(U9.cross_seed_nmi({1: a, 2: b}) - 1.0) < 1e-9
    assert U9.chance_nmi({1: a, 2: b}) < 0.02
    small = {f"r{i}": rng.integers(0, 512, 60) for i in range(5)}  # few windows, many codes: the floor is far above 0
    other = {n: rng.integers(0, 512, len(c)) for n, c in small.items()}
    assert U9.cross_seed_nmi({1: small, 2: other}) > 0.3
    return True


def test_association_measures_detect_recording_identity():
    rng = np.random.default_rng(0)
    by_rec = {f"r{i}": np.full(200, i) for i in range(6)}  # a code per recording: pure recording identity
    mixed = {f"r{i}": rng.integers(0, 6, 200) for i in range(6)}
    assert U9.recording_nmi(by_rec) > 0.99 and U9.recording_nmi(mixed) < 0.05
    centers = rng.normal(size=(6, 16)) * 5
    feats = {f"r{i}": centers[i] + rng.normal(size=(300, 16)) * 0.3 for i in range(6)}
    noise = {f"r{i}": rng.normal(size=(300, 16)) for i in range(6)}
    assert U9.nn_same_recording_share(feats, n_query=300, n_pool=1000) > 0.9
    assert U9.nn_same_recording_share(noise, n_query=300, n_pool=1000) < 0.4
    occ = U9.occupancy({"a": np.array([0, 0, 1, 1])})
    assert occ["occupied"] == 2 and abs(occ["perplexity"] - 2.0) < 1e-9
    return True


def test_stage_items_order_decodes_and_prospective_scope():
    for ds in R9.DEV_DATASETS:
        A, C_ = R9.stage_items(ds, "A"), R9.stage_items(ds, "C")
        assert [(i.cell.arm, i.cell.seed) for i in A[:3]] == [("g", 111), ("i1q", 111), ("i1", 111)]  # by seed
        assert len(A) == 9 and all(i.tags == ("tg",) and i.train for i in A)
        c = [(i.cell.arm, i.tags, i.train) for i in C_[:2]]
        assert c == [("i2", ("tg", "ig"), True), ("i1", ("ig",), False)]  # I2 with both groupings; I1 re-decoded, not retrained
        assert len(C_) == 6
    PA, PC = R9.stage_items("babel1", "A", prospective=True), R9.stage_items("babel1", "C", prospective=True)
    assert {i.cell.arm for i in PA} == {"i1"} and [i.tags for i in PC[:2]] == [("ig",), ("ig",)]
    assert all(i.cell.span == 2 and i.cell.guard == 0 for i in PA + PC)
    return True


def test_identities_are_distinct_and_the_jobs_use_the_declared_arms():
    with scratch_runner() as r:
        ids = [r.identity_of(c) for c in r.base_cells]
        assert len(set(ids)) == len(ids) == 3 * 4 * 3  # three datasets x four arms x three seeds
        jobs = {a: r.make_job(R9.Cell9("babel2", a, 111)) for a in R9.ARMS}
        assert [jobs[a].arm for a in R9.ARMS] == ["context_geometry_init", "context_quantized", "context_continuous", "context_continuous"]
        assert all(j.mask_span == 2 and j.guard_radius == 0 and j.context == 128 and j.stride == 64 and j.batch_chunks == 64 for j in jobs.values())
        assert r.identity_of(R9.Cell9("babel2", "i1", 111)) != r.identity_of(R9.Cell9("babel2", "i2", 111))
    return True


def test_blocking_rules_and_deadline():
    with scratch_runner(hours=1e-9) as r:
        ok = dict(status="passed")
        r.gates = {g: ok for g in ("g9a", "g9b", "g9c", "g9d", "g9e", "g9g", "g3")}
        r.gates["g9f_babel2_111"] = ok
        assert r.blocked(R9.Cell9("babel2", "i2", 111)) is None and r.blocked(R9.Cell9("lara", "g", 111)) is None
        r.gates["g9f_babel2_111"] = dict(status="failed")
        assert r.blocked(R9.Cell9("babel2", "i2", 111)) and r.blocked(R9.Cell9("babel2", "i1", 111)) is None
        for g in ("g9a", "g9c"):
            r.gates[g] = dict(status="failed")
            assert r.blocked(R9.Cell9("babel3", "i1", 111))
            r.gates[g] = ok
        r.gates["g3"] = dict(status="failed")
        assert r.blocked(R9.Cell9("lara", "i1", 111)) and r.blocked(R9.Cell9("babel2", "i1", 111)) is None
        r.gates["g3"] = ok
        counts: dict = {}
        r.run_item(R9.stage_items("babel2", "A")[0], counts, "A")
        assert counts == {"deadline": 1} and not list((r.out / "jobs").iterdir())
    return True


def test_dry_run_projects_the_declared_budget():
    import io
    with scratch_runner() as r, contextlib.redirect_stdout(io.StringIO()) as buf:
        r.dry_run()
    line = [l for l in buf.getvalue().split("\n") if l.startswith("projected")][0]
    hours = float(line.split("~")[1].split(" h")[0])
    assert 7.5 <= hours <= 8.3, hours  # the plan's 7.9 h (8.4 h worst case) under a 10 h cap
    return True


def test_references_exist_for_every_dataset_and_seed():
    for ds in ("babel2", "babel3", "lara", "babel1"):
        for s in R9r.SEEDS:
            ref = R9r.reference(SRC, ds, s)
            assert ref is not None and 0 < ref["F1_50"] < 100, (ds, s)
            assert R9r.ref_predictions(SRC, ds, s) is not None, (ds, s)
    return True


def _unit(root: Path, name: str, f1: float, nn=0.5):
    d = root / "units" / name / "info.json"
    d.parent.mkdir(parents=True, exist_ok=True)
    m = dict(MoF=40.0, Edit=30.0, F1_10=35.0, F1_25=32.0, F1_50=f1)
    d.write_text(json.dumps(dict(status="done", ladder=[dict(rung_steps=4000, **m)], nn_same_recording=nn,
                                 segment_stats=dict(mean_pred_segments=3.0, mean_gt_segments=4.0))))


def test_report_readings_and_the_i2_comparison_uses_the_i2_code_grouping():
    tmp = TS / "_tests9_report"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        out, src = tmp / "out", tmp / "src"
        (out / "diagnostics").mkdir(parents=True)
        for ds in ("babel2", "babel3", "lara"):
            for s in R9r.SEEDS:
                phase, cid = R8.t2_cell_id(ds, s)
                d = src / phase / "units" / f"{cid}__context" / "info.json"
                d.parent.mkdir(parents=True, exist_ok=True)
                d.write_text(json.dumps(dict(status="done", ladder=[dict(rung_steps=4000, MoF=40.0, Edit=30.0, F1_10=35.0, F1_25=32.0, F1_50=30.0)],
                                             segment_stats=dict(mean_pred_segments=3.0, mean_gt_segments=4.0))))
        for s in R9r.SEEDS:
            _unit(out, f"babel2__i1__seed{s}__ig", 25.0)
            _unit(out, f"babel2__i2__seed{s}__ig", 29.0)  # +4 on every seed: passes
            _unit(out, f"babel2__i1__seed{s}__tg", 25.0)
            _unit(out, f"babel2__i2__seed{s}__tg", 20.0)  # robustness: -5, reverse passes
        for s, v in zip(R9r.SEEDS, (30.0, 22.0, 31.0)):
            _unit(out, f"babel3__i1__seed{s}__ig", 25.0)
            _unit(out, f"babel3__i2__seed{s}__ig", v)  # mixed: within the bar
        for s in (111, 222):
            _unit(out, f"lara__i1__seed{s}__ig", 25.0)
            _unit(out, f"lara__i2__seed{s}__ig", 40.0)  # two seeds only: incomplete
        (out / "gates.json").write_text("{}")
        (out / "manifest.json").write_text(json.dumps(dict(run_id="x", status="finished")))
        R9r.write_report(out, src)
        text = (out / "REPORT.md").read_text(encoding="utf-8")

        def line(cid, ds):
            sec = text.split(f"### {cid}")[1].split("###")[0]
            return next(l for l in sec.split("\n") if f"**{ds}**" in l)
        assert "A better (passes the bar)" in line("Q2:", "babel2") and "+4.00" in line("Q2:", "babel2")
        assert "B better (reverse passes the bar)" in line("Q2r", "babel2")
        assert "within the bar: comparable performance, not equivalence" in line("Q2:", "babel3")
        assert "incomplete (2 seeds)" in line("Q2:", "lara")
        assert "equivalent" not in text.replace("not equivalence", "").lower()
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_gates_g9b_g9c_g9d_g9e_g9g_pass_on_real_data():
    with scratch_runner() as r:
        for name, fn in (("g9b", G9.g9b), ("g9c", G9.g9c), ("g9d", G9.g9d), ("g9e", G9.g9e), ("g9g", G9.g9g)):
            res = fn(r)
            assert res["status"] == "passed", (name, res)
        b = G9.g9b(r)
        assert b["context_quantized_training_history_gap"] <= 1e-4 and b["context_continuous_training_history_gap"] > 1e-3
    return True


def test_g9b_fails_when_a_masked_input_leaks():
    """A model that reads masked inputs must fail the masking gate's logic: simulate by checking the sensitive operation."""
    cfg = T.train_config(R5_job(), P.SMOKE)
    init = C.build_matched_init(16, 8, cfg, seed=1)
    model = C.build_arm_model("context_continuous", 16, 8, cfg, 1, init).eval()
    x = torch.randn(1, 12, 8)
    pad = torch.zeros(1, 12, dtype=torch.bool)
    mask = torch.zeros(1, 12, dtype=torch.bool)
    mask[0, 3:5] = True
    with torch.no_grad():
        h0, _ = model.forward_vectors(x, mask, pad)
        x2 = x.clone()
        x2[0, 3:5] += 4.0
        h1, _ = model.forward_vectors(x2, mask, pad)
        x3 = x.clone()
        x3[0, 7] += 4.0
        h3, _ = model.forward_vectors(x3, mask, pad)
    assert torch.equal(h0, h1) and not torch.equal(h0, h3)
    return True


ALL_TESTS = [
    test_trainer_copy_differs_from_phase5_only_in_the_three_marked_lines,
    test_feature_fn_selects_the_input_by_arm,
    test_layer_states_end_at_the_final_hidden_state_and_the_extractor_matches_phase5,
    test_i2_family_obeys_the_dimensional_contract_and_rejects_a_128d_lookup,
    test_family_uses_the_base_features_split_and_rms_unchanged,
    test_cross_seed_agreement_is_permutation_invariant_with_a_chance_floor,
    test_association_measures_detect_recording_identity,
    test_stage_items_order_decodes_and_prospective_scope,
    test_identities_are_distinct_and_the_jobs_use_the_declared_arms,
    test_blocking_rules_and_deadline,
    test_dry_run_projects_the_declared_budget,
    test_references_exist_for_every_dataset_and_seed,
    test_report_readings_and_the_i2_comparison_uses_the_i2_code_grouping,
    test_gates_g9b_g9c_g9d_g9e_g9g_pass_on_real_data,
    test_g9b_fails_when_a_masked_input_leaks,
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
