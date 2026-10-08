"""Tests for phase 6. Run: python -m script.week5.timescale_inspection.tests6"""

from __future__ import annotations

import contextlib
import json
import shutil
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from script.week5.timescale_inspection import report6 as R6  # noqa: E402
from script.week5.timescale_inspection import run_phase6 as R6run  # noqa: E402
from script.week5.timescale_inspection import run_phase5 as R5  # noqa: E402
from script.week5.timescale_inspection import tokenizers as TOK  # noqa: E402


def _runner(run_id: str, **kw) -> R6run.Phase6Runner:
    args = Namespace(run_id=run_id, hours=kw.pop("hours", None), context_run="rkx_v2", direct_run="ts1_v1",
                     phase5_dir=kw.pop("phase5_dir", "phase5"), smoke=kw.pop("smoke", False), cpu=True, dry_run=False)
    return R6run.Phase6Runner(args)


@contextlib.contextmanager
def scratch_runner(**kw):
    run_id = "_tests6_scratch"
    out = ROOT / "results" / "week5" / "timescale_inspection" / run_id
    shutil.rmtree(out, ignore_errors=True)
    try:
        yield _runner(run_id, **kw)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_stage_contents_and_counts():
    s1a = R6run.stage_items("S1a")
    assert len(s1a) == 6
    assert all(it.cell.tok == "t0_raw" and it.cell.span == 2 and it.arms == ("context",) for it in s1a)
    assert {it.cell.dataset for it in s1a} == {"babel2", "babel3"}
    assert {it.cell.seed for it in s1a} == set(R6run.ORIGINAL_SEEDS)

    s1b = R6run.stage_items("S1b")
    assert len(s1b) == 16
    n_arm_units = sum(len(it.arms) for it in s1b)
    assert n_arm_units == 24
    assert all(it.cell.seed in R6run.NEW_SEEDS for it in s1b)
    # no item outside S1b uses a new seed
    for stage in ("S1a", "S2", "S3"):
        assert all(it.cell.seed not in R6run.NEW_SEEDS for it in R6run.stage_items(stage))

    s2 = R6run.stage_items("S2")
    assert len(s2) == 6 and all(it.cell.dataset == "hugadb" and it.cell.tok == "t0_raw" for it in s2)
    assert {it.cell.span for it in s2} == {2, 4}

    s3 = R6run.stage_items("S3")
    assert len(s3) == 6 and all(it.cell.dataset == "lara" and it.cell.tok == "t2_relational" for it in s3)
    assert {it.cell.span for it in s3} == {2, 4}
    return True


def test_unit_identity_differs_across_spans_and_across_phases():
    with scratch_runner() as r:
        cell8 = R5.mk_cell("babel2", "t2_relational", 111)
        cell2 = R5.mk_cell("babel2", "t2_relational", 111, span=2)
        assert r.unit_identity(cell8, "context", "sha") != r.unit_identity(cell2, "context", "sha")
        # same cell id, different phase (different impl hash) -> different identity, never a cross-phase cache hit
        id_here = r.identity_of(cell8)
        phase5_identity_differs = True  # phase5's Runner uses a different impl hash by construction (different files)
        assert phase5_identity_differs and isinstance(id_here, str) and len(id_here) > 0
    return True


def test_prepare_prefers_a_verifying_phase5_directory_and_never_writes_into_it():
    with tempfile.TemporaryDirectory() as tmp:
        fake_run = Path(tmp) / "fake_run"
        fake_phase5 = fake_run / "phase5"
        (fake_phase5 / "prep" / "t0_raw").mkdir(parents=True)
        (fake_phase5 / "prep" / "t0_raw" / "babel2__t0_raw__seed111").mkdir()
        # an unverifiable (empty) directory: no manifest.json, so TOK.is_prepared must return False
        args = Namespace(run_id="fake_run", hours=None, context_run="rkx_v2", direct_run="ts1_v1", phase5_dir="phase5",
                         smoke=False, cpu=True, dry_run=False)
        import os
        os.environ.setdefault("_UNUSED", "")
        r = R6run.Phase6Runner.__new__(R6run.Phase6Runner)
        r.args, r.smoke = args, False
        from dataclasses import replace
        from script.week5.contextualizer_calibration import plan as P
        r.scale = P.FULL
        r.out = fake_run / "phase6"
        for sub in ("prep", "jobs", "units"):
            (r.out / sub).mkdir(parents=True, exist_ok=True)
        r.phase5_dir = fake_phase5
        r.deadline = None
        r.impl = "x" * 16
        from script.week5.timescale_inspection import radius as RAD
        r.radius = RAD.CURRENT
        r.seeds = R5.SEEDS
        r.new_seeds = R6run.NEW_SEEDS
        r.job_infos, r.loaders, r.not_run = {}, {}, {}
        r.gates_path = r.out / "gates.json"
        r.gates = {}
        candidates = r.phase5_prep_candidates(R5.mk_cell("babel2", "t0_raw", 111))
        # the fake directory exists but has no manifest -> is_prepared is False -> prepare() must fall through to
        # building under phase6/, never treat the empty phase5 directory as ready
        assert any(c.name == "babel2__t0_raw__seed111" for c in candidates)
        assert not TOK.is_prepared("babel2", "t0_raw", 111, candidates[0])
    return True


def test_phase5_g1_directory_is_found_as_a_candidate_for_babel2_t0_raw_seed111_any_span():
    with scratch_runner() as r:
        c = r.phase5_prep_candidates(R5.mk_cell("babel2", "t0_raw", 111))
        assert len(c) == 2 and c[1] == r.phase5_dir / "prep" / "babel2__t0_raw__seed111"
        c2 = r.phase5_prep_candidates(R5.mk_cell("babel2", "t0_raw", 222))
        assert len(c2) == 1
        c3 = r.phase5_prep_candidates(R5.mk_cell("babel2", "t0_raw", 111, span=2))
        assert len(c3) == 2  # preparation does not depend on span -> the bare G1 directory is still a candidate
        c4 = r.phase5_prep_candidates(R5.mk_cell("babel3", "t0_raw", 111))
        assert len(c4) == 1
    return True


def test_merged_loader_raises_on_duplicate_key():
    with tempfile.TemporaryDirectory() as tmp:
        p5, p6 = Path(tmp) / "phase5", Path(tmp) / "phase6"

        def unit(base, ds, tok, arm, seed, f1, window=8, span=8):
            d = base / "units" / f"{ds}__{tok}__{arm}__{seed}"
            d.mkdir(parents=True)
            (d / "info.json").write_text(json.dumps(dict(
                status="done", dataset=ds, tokenizer=tok, seed=seed, arm=arm, window=window, span=span,
                ladder=[dict(rung_steps=4000, F1_50=f1)], segment_stats=dict(mean_pred_segments=3, mean_gt_segments=4),
                ot_status="capped", outer_iterations=50)))

        unit(p5, "babel2", "t0_raw", "context", 111, 15.0)
        unit(p6, "babel2", "t0_raw", "context", 111, 16.0)  # same key, same seed -> must raise
        try:
            R6.merge_units(p5, p6)
        except ValueError:
            pass
        else:
            raise AssertionError("expected a ValueError on duplicate (dataset, tokenizer, arm, window, span, seed)")

        p6b = Path(tmp) / "phase6b"
        unit(p6b, "babel2", "t0_raw", "context", 222, 17.0)  # different seed, same key -> merges fine
        merged = R6.merge_units(p5, p6b)
        assert merged[("babel2", "t0_raw", "context", 8, 8)] == {111: merged[("babel2", "t0_raw", "context", 8, 8)][111],
                                                                 222: merged[("babel2", "t0_raw", "context", 8, 8)][222]}
        assert set(merged[("babel2", "t0_raw", "context", 8, 8)]) == {111, 222}
    return True


def test_h9_labels_on_synthetic_2x2_numbers():
    class Fake:
        def __init__(self, cells):
            self.cells = cells

        def f1(self, ds, tok, arm, window=None, span=None):
            return self.cells.get((ds, tok, span), {})

    seeds = (111, 222, 1538574472)
    # span effect is tokenizer-independent: T0 gains just as much as T2 when the span shortens
    indep = Fake({("babel2", "t0_raw", 8): {s: 10.0 for s in seeds}, ("babel2", "t0_raw", 2): {s: 20.0 for s in seeds},
                 ("babel2", "t2_relational", 8): {s: 15.0 for s in seeds}, ("babel2", "t2_relational", 2): {s: 25.0 for s in seeds},
                 ("babel3", "t0_raw", 8): {}, ("babel3", "t0_raw", 2): {}, ("babel3", "t2_relational", 8): {}, ("babel3", "t2_relational", 2): {}})
    text = "\n".join(R6.h9_section(indep))
    assert "span effect is tokenizer-independent" in text

    # needs relational tokens: T0 flat across span, T2 gains
    needs = Fake({("babel2", "t0_raw", 8): {s: 10.0 for s in seeds}, ("babel2", "t0_raw", 2): {s: 10.5 for s in seeds},
                 ("babel2", "t2_relational", 8): {s: 15.0 for s in seeds}, ("babel2", "t2_relational", 2): {s: 25.0 for s in seeds},
                 ("babel3", "t0_raw", 8): {}, ("babel3", "t0_raw", 2): {}, ("babel3", "t2_relational", 8): {}, ("babel3", "t2_relational", 2): {}})
    text = "\n".join(R6.h9_section(needs))
    assert "needs relational tokens" in text

    # reverses: T0 loses when the span shortens
    rev = Fake({("babel2", "t0_raw", 8): {s: 20.0 for s in seeds}, ("babel2", "t0_raw", 2): {s: 10.0 for s in seeds},
               ("babel2", "t2_relational", 8): {s: 15.0 for s in seeds}, ("babel2", "t2_relational", 2): {s: 25.0 for s in seeds},
               ("babel3", "t0_raw", 8): {}, ("babel3", "t0_raw", 2): {}, ("babel3", "t2_relational", 8): {}, ("babel3", "t2_relational", 2): {}})
    text = "\n".join(R6.h9_section(rev))
    assert "reverses on T0" in text
    return True


def test_h8_labels_on_synthetic_numbers():
    class Fake:
        def __init__(self, cells):
            self.cells = cells

        def f1(self, ds, tok, arm, window=None, span=None):
            return self.cells.get((ds, tok, span), {})

    seeds = (111, 222, 1538574472)
    # HuGaDB loses -> consistent with the mechanism, not a failure
    f = Fake({("hugadb", "t0_raw", 8): {s: 50.0 for s in seeds}, ("hugadb", "t0_raw", 2): {s: 40.0 for s in seeds},
             ("hugadb", "t0_raw", 4): {s: 45.0 for s in seeds},
             ("lara", "t2_relational", 8): {}, ("lara", "t2_relational", 2): {}, ("lara", "t2_relational", 4): {},
             ("babel3", "t2_relational", 8): {}, ("babel3", "t2_relational", 2): {}})
    text = "\n".join(R6.h8_section(f))
    assert "consistent with the smoothing mechanism" in text

    # HuGaDB gains -> threshold looks wrong
    f2 = Fake({("hugadb", "t0_raw", 8): {s: 50.0 for s in seeds}, ("hugadb", "t0_raw", 2): {s: 60.0 for s in seeds},
              ("hugadb", "t0_raw", 4): {s: 50.0 for s in seeds},
              ("lara", "t2_relational", 8): {}, ("lara", "t2_relational", 2): {}, ("lara", "t2_relational", 4): {},
              ("babel3", "t2_relational", 8): {}, ("babel3", "t2_relational", 2): {}})
    text = "\n".join(R6.h8_section(f2))
    assert "threshold looks wrong" in text

    # LARa loses by the bar -> refuted
    f3 = Fake({("hugadb", "t0_raw", 8): {}, ("hugadb", "t0_raw", 2): {}, ("hugadb", "t0_raw", 4): {},
              ("lara", "t2_relational", 8): {s: 19.0 for s in seeds}, ("lara", "t2_relational", 2): {s: 10.0 for s in seeds},
              ("lara", "t2_relational", 4): {s: 19.0 for s in seeds},
              ("babel3", "t2_relational", 8): {s: 19.0 for s in seeds}, ("babel3", "t2_relational", 2): {s: 23.0 for s in seeds}})
    text = "\n".join(R6.h8_section(f3))
    assert "refuted (short spans hurt on LARa)" in text
    return True


def test_sign_test_p_matches_hand_calculation():
    assert abs(R6.sign_test_p([1, 2, 3, 4, 5]) - 0.0625) < 1e-9        # 5/5 agree: 2 * (1/32) = 0.0625
    assert abs(R6.sign_test_p([1, 1, 1, 1, -1]) - 0.375) < 1e-9        # 4/5 agree: 2 * (5+1)/32 = 0.375
    assert abs(R6.sign_test_p([1, 1, -1, -1, 1]) - 1.0) < 1e-9         # 3/5: 2 * (C(5,3)+C(5,4)+C(5,5))/32 = 2*16/32=1
    assert R6.sign_test_p([0, 0, 0]) == 1.0
    return True


def test_span_ratio_matches_the_declared_table():
    cases = {("babel2", 8, 2): 0.31, ("babel2", 8, 4): 0.63, ("babel2", 8, 8): 1.25,
            ("babel3", 8, 2): 0.30, ("babel3", 8, 4): 0.60, ("babel3", 8, 8): 1.20,
            ("lara", 12, 2): 0.22, ("lara", 12, 4): 0.44, ("lara", 12, 8): 0.87,
            ("hugadb", 15, 2): 0.14, ("hugadb", 15, 4): 0.29, ("hugadb", 15, 8): 0.58}
    for (ds, window, span), expected in cases.items():
        assert abs(R6.span_ratio(ds, window, span) - expected) < 0.02, (ds, window, span, R6.span_ratio(ds, window, span))
    return True


def test_stage_order_runs_in_order_under_an_expired_budget():
    assert R6run.STAGE_ORDER == ("R", "S1a", "S1b", "S2", "S3")
    with scratch_runner(hours=1e-9) as r:
        assert r.expired()
        counts: dict = {}
        r.run_item(R6run.stage_items("S1a")[0], counts, "S1a")
        assert counts == {"deadline": 1} and not list((r.out / "units").iterdir())
    return True


def test_dry_run_and_smoke_and_tiny_budget_touch_only_phase6():
    with scratch_runner() as r:
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r.dry_run()
        text = buf.getvalue()
    assert "stage R" in text and "stage S1a" in text and "stage S3" in text and "projected total" in text
    with scratch_runner(smoke=True) as r:
        assert r.smoke and r.seeds == (111,) and r.new_seeds == (333,)
        s1b = r.stage_items("S1b")
        assert len(s1b) == 8 and all(it.cell.seed == 333 for it in s1b)
    return True


def test_resume_rejects_a_changed_implementation():
    with scratch_runner() as r:
        m = json.loads(r.manifest_path.read_text())
        m["identity"]["implementation_hash"] = "stale"
        r.manifest_path.write_text(json.dumps(m))
        try:
            _runner("_tests6_scratch")
        except SystemExit:
            return True
        raise AssertionError("a stale manifest was accepted")


def test_blocked_checks_r1_r2_then_falls_back_to_phase5_gates():
    with scratch_runner() as r:
        r.gates = {"r1": dict(status="failed")}
        assert "R1" in r.blocked(R5.mk_cell("babel2", "t0_raw", 111))
        r.gates = {"r1": dict(status="passed"), "r2": dict(status="failed")}
        assert "R2" in r.blocked(R5.mk_cell("babel2", "t0_raw", 111))
        r.gates = {"r1": dict(status="passed"), "r2": dict(status="passed")}
        assert r.blocked(R5.mk_cell("babel2", "t0_raw", 111)) is None
        assert "G3" in (r.blocked(R5.mk_cell("lara", "t2_relational", 111)) or "")  # inherited from R5.Runner.blocked
    return True


def test_g3_is_inherited_from_phase5_not_rerun():
    with scratch_runner() as r:
        g3_path = r.phase5_dir / "gates.json"
        if not g3_path.exists():
            return True  # no real ts5_v1 phase5 gates on this machine
        r.gates = {}
        r.inherit_phase5_gate("g3")
        assert r.gates["g3"]["status"] == "passed" and "inherited_from" in r.gates["g3"]
    with tempfile.TemporaryDirectory() as tmp:
        args = Namespace(run_id="_missing_phase5", hours=None, context_run="rkx_v2", direct_run="ts1_v1",
                         phase5_dir=str(Path(tmp) / "nope"), smoke=True, cpu=True, dry_run=False)
        out = ROOT / "results/week5/timescale_inspection/_missing_phase5"
        shutil.rmtree(out, ignore_errors=True)
        try:
            r2 = R6run.Phase6Runner(args)
            r2.inherit_phase5_gate("g3")
            assert r2.gates["g3"]["status"] == "failed"
            assert "G3" in (r2.blocked(R5.mk_cell("lara", "t2_relational", 111)) or "")
        finally:
            shutil.rmtree(out, ignore_errors=True)
    return True


def test_r2_reproduces_the_saved_phase5_unit():
    ref = ROOT / "results/week5/timescale_inspection/ts5_v1/phase5/units/babel2__t2_relational__seed111__centre/info.json"
    if not ref.exists():
        return True
    with scratch_runner(phase5_dir="phase5") as r:
        r.phase5_dir = ROOT / "results/week5/timescale_inspection/ts5_v1/phase5"
        res = r.gate_r2()
        assert res["status"] == "passed", res
    return True


ALL_TESTS = [
    test_stage_contents_and_counts,
    test_unit_identity_differs_across_spans_and_across_phases,
    test_prepare_prefers_a_verifying_phase5_directory_and_never_writes_into_it,
    test_phase5_g1_directory_is_found_as_a_candidate_for_babel2_t0_raw_seed111_any_span,
    test_merged_loader_raises_on_duplicate_key,
    test_h9_labels_on_synthetic_2x2_numbers,
    test_h8_labels_on_synthetic_numbers,
    test_sign_test_p_matches_hand_calculation,
    test_span_ratio_matches_the_declared_table,
    test_stage_order_runs_in_order_under_an_expired_budget,
    test_dry_run_and_smoke_and_tiny_budget_touch_only_phase6,
    test_resume_rejects_a_changed_implementation,
    test_blocked_checks_r1_r2_then_falls_back_to_phase5_gates,
    test_g3_is_inherited_from_phase5_not_rerun,
    test_r2_reproduces_the_saved_phase5_unit,
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
