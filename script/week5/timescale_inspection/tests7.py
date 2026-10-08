"""Tests for phase 7. Run: python -m script.week5.timescale_inspection.tests7"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from script.week5.timescale_inspection import criteria7 as K  # noqa: E402
from script.week5.timescale_inspection import run_phase5 as R5  # noqa: E402
from script.week5.timescale_inspection import run_phase7 as R7  # noqa: E402
from script.week5.timescale_inspection import tokenizers as TOK  # noqa: E402

TS5 = ROOT / "results" / "week5" / "timescale_inspection" / "ts5_v1"


@contextlib.contextmanager
def scratch_runner(**kw):
    run_id = "_tests7_scratch"
    out = ROOT / "results" / "week5" / "timescale_inspection" / run_id
    shutil.rmtree(out, ignore_errors=True)
    args = Namespace(run_id=run_id, hours=kw.get("hours"), context_run="rkx_v2", direct_run="ts1_v1",
                     phase5_dir=str(TS5 / "phase5"), smoke=kw.get("smoke", False), cpu=False, dry_run=False)
    try:
        yield R7.Phase7Runner(args)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def _segs(labels_and_lengths):
    return np.concatenate([np.full(n, l) for l, n in labels_and_lengths])


def test_agreement_is_label_permutation_invariant_and_drops_for_different_boundaries():
    a = [_segs([(0, 30), (1, 20), (2, 50)]), _segs([(1, 40), (0, 60)])]
    perm = {0: 7, 1: 3, 2: 5}
    b = [np.vectorize(perm.get)(x) for x in a]
    assert abs(K.pair_agreement(a, b) - 100.0) < 1e-9
    c = [_segs([(0, 10), (1, 60), (2, 30)]), _segs([(1, 90), (0, 10)])]
    assert K.pair_agreement(a, c) < 60.0
    try:
        K.pair_agreement(a, [x[:-1] for x in a])
    except ValueError:
        return True
    raise AssertionError("mismatched frame counts were accepted")


def test_k1_prefers_the_span_whose_seeds_agree():
    base = [_segs([(0, 30), (1, 30), (2, 40)]) for _ in range(4)]
    stable = {s: {f"r{i}": x for i, x in enumerate(base)} for s in (1, 2, 3)}
    rng = np.random.default_rng(0)
    def jitter():
        a = int(rng.integers(5, 55))  # boundary positions move per seed; the recording length (100) does not
        return _segs([(0, a), (1, 60 - a), (2, 40)])
    noisy = {s: {f"r{i}": jitter() for i in range(4)} for s in (1, 2, 3)}
    assert K.k1_stability(stable) > K.k1_stability(noisy)
    assert K.pick_k1({2: K.k1_stability(noisy), 8: K.k1_stability(stable)}) == 8
    assert K.pick_k1({2: 50.0, 8: 50.0}) == 2  # declared tie-break: the shorter span
    return True


def test_run_lengths_and_median_duration():
    assert list(K.run_lengths(np.array([0, 0, 1, 1, 1, 2]))) == [2, 3, 1]
    preds = {1: {"a": _segs([(0, 30), (1, 30)])}, 2: {"a": _segs([(0, 60)])}}
    assert K.median_pred_duration(preds, fps=30) == 1.0  # run lengths 30, 30, 60 frames -> median 30 -> 1 s
    return True


def test_decorrelation_lag_is_longer_for_slower_features():
    rng = np.random.default_rng(1)
    def ar(phi, n=400, d=4):
        x = np.zeros((n, d))
        for t in range(1, n):
            x[t] = phi * x[t - 1] + rng.normal(size=d)
        return x
    fast, slow = K.decorrelation_lag(ar(0.3)), K.decorrelation_lag(ar(0.95))
    assert fast is not None and slow is not None and slow > fast
    assert K.feature_tau_seconds({"a": ar(0.95), "b": ar(0.95)}, window=8, fps=30) > 0
    return True


def test_pick_rules_and_gate():
    assert K.pick_largest_fitting([2, 4, 8], 1.2, window=8, fps=30) == 4  # 2.13 s > 1.2 >= 1.07 s
    assert K.pick_largest_fitting([2, 4, 8], 0.1, window=8, fps=30) == 2  # nothing fits -> smallest
    assert K.pick_largest_fitting([2, 4, 8], {2: 9, 4: 0.5, 8: 9}, window=8, fps=30) == 8  # per-span limits (K3)
    assert K.acceptable_spans({2: 29.4, 4: 27.0, 8: 21.4}) == {2, 4}
    g = K.gate({"babel2": 2, "hugadb": 2}, {"babel2": {2, 4}, "hugadb": {8}})
    assert g["passed"] is False and g["n_right"] == 1
    assert K.gate({"babel2": 4, "hugadb": 8}, {"babel2": {2, 4}, "hugadb": {8}})["passed"] is True
    return True


def test_stage_contents_and_s0_sources_exist():
    h1, h2, h3 = (R7.heldout_items(s) for s in ("H1", "H2", "H3"))
    assert len(h1) == 6 and sum(len(i.arms) for i in h1) == 9
    assert len(h2) == 3 and all(i.cell.tok == "t0_raw_b1" and i.cell.span == 2 for i in h2)
    assert len(h3) == 6 and all(i.cell.seed in (333, 444) for i in h3)
    assert all(i.cell.dataset == "babel1" for i in h1 + h2 + h3)
    s0 = R7.s0_items()
    assert len(s0) == 33 and all(c.seed in R5.SEEDS for _, c in s0)
    if TS5.exists():
        missing = [f"{ph}/{c.cell_id}" for ph, c in s0 if not (TS5 / ph / "units" / f"{c.cell_id}__context" / "info.json").exists()]
        assert not missing, missing
    return True


def test_heldout_registration_leaves_existing_tokenizers_and_identities_unchanged():
    assert TOK.SPECS["t0_raw"].datasets == ("babel2", "babel3", "hugadb", "lara")
    assert TOK.SPECS["t2_relational"].datasets == ("babel2", "babel3", "lara")
    assert TOK.SPECS["t2_relational_b1"].datasets == ("babel1",) and TOK.SPECS["t2_relational_b1"].source == "relational"
    info = TS5 / "phase5" / "jobs" / "t2_relational" / "babel2__t2_relational__seed111__span2" / "training_info.json"
    if info.exists():
        src = R7.SourceView(TS5 / "phase5", TS5 / "phase5", device_cpu=True)
        cell = R5.mk_cell("babel2", "t2_relational", 111, span=2)
        assert src.identity_of(cell) == json.loads(info.read_text())["identity"]
    return True


def _tree_mtimes(d: Path) -> dict:
    return {str(p): p.stat().st_mtime for p in d.rglob("*")}


def test_s0_reproduces_a_saved_unit_exactly_and_never_writes_upstream():
    if not (TS5 / "phase5" / "units" / "babel2__t2_relational__seed111__span2__context").exists():
        return True
    before = _tree_mtimes(TS5 / "phase5")
    with scratch_runner() as r:
        counts = {}
        phase, cell = "phase5", R5.mk_cell("babel2", "t2_relational", 111, span=2)
        r.run_s0_item(phase, cell, counts, "test")
        info = json.loads((r.s0_dir(phase, cell) / "info.json").read_text())
        assert counts == {"s0_reproduced": 1}, counts
        assert info["reproduced"] and info["max_abs_diff"] == 0.0
        with np.load(r.s0_dir(phase, cell) / "predictions.npz") as z:
            assert len(z.files) > 100 and all(z[k].ndim == 1 for k in z.files[:5])
        counts2 = {}
        r.run_s0_item(phase, cell, counts2, "test")  # resume: reused, not recomputed
        assert counts2 == {"s0_cached": 1}
    assert _tree_mtimes(TS5 / "phase5") == before  # the source run is untouched
    return True


def test_g7_generalized_t0_reproduces_rkx_v2_on_babel1():
    if not (ROOT / "results/week5/raw_kmeans_cross_dataset/rkx_v2/prep/babel1__global__seed111").exists():
        return True
    with scratch_runner() as r:
        res = r.gate_g7()
        assert res["status"] == "passed", res
    return True


def test_expired_budget_starts_nothing_and_dry_run_projects():
    with scratch_runner(hours=1e-9) as r:
        counts = {}
        r.run_item(R7.heldout_items("H1")[0], counts, "H1")
        assert counts == {"deadline": 1} and not list((r.out / "units").iterdir())
        r.run_s0_item(*R7.s0_items()[0], counts, "S0")
        assert counts.get("deadline") == 2 and not list((r.out / "s0").iterdir())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r.dry_run()
        assert "stage S0: 33 re-decodes" in buf.getvalue() and "projected total" in buf.getvalue()
    return True


def test_resume_rejects_a_changed_implementation():
    with scratch_runner() as r:
        m = json.loads(r.manifest_path.read_text())
        m["identity"]["implementation_hash"] = "stale"
        r.manifest_path.write_text(json.dumps(m))
        args = Namespace(run_id="_tests7_scratch", hours=None, context_run="rkx_v2", direct_run="ts1_v1",
                         phase5_dir=str(TS5 / "phase5"), smoke=False, cpu=False, dry_run=False)
        try:
            R7.Phase7Runner(args)
        except SystemExit:
            return True
        raise AssertionError("stale manifest accepted")


def test_declared_gate_never_passes_on_incomplete_data():
    acc = {"babel2": {2, 4}, "hugadb": {8}}
    assert K.gate_declared({"babel2": 2}, acc, {"babel2": True, "hugadb": False})["status"] == "incomplete"
    assert K.gate_declared({"babel2": 8}, acc, {"babel2": True, "hugadb": False})["status"] == "fail"  # wrong where data exist
    assert K.gate_declared({"babel2": 2, "hugadb": 8}, acc, {"babel2": True, "hugadb": True})["status"] == "pass"
    assert K.gate_declared({"babel2": 2, "hugadb": None}, acc, {"babel2": True, "hugadb": True})["status"] == "fail"  # undefined
    return True


def test_occupancy_exposes_collapse_that_k1_rewards():
    one = {s: {f"r{i}": np.zeros(100, int) for i in range(3)} for s in (1, 2, 3)}
    assert K.k1_stability(one) == 100.0  # the known weakness
    occ = K.occupancy(one)
    assert occ["share_single_state_recordings"] == 1.0 and occ["mean_runs_per_recording"] == 1.0
    return True


def test_completeness_requires_declared_spans_common_seeds_and_identical_coverage():
    from script.week5.timescale_inspection import report7 as R
    rec = lambda n=100: {"preds": {"a": np.zeros(n, int), "b": np.zeros(50, int)}, "metrics": {}}
    full = {("babel1", sp): {s: rec() for s in (1, 2, 3)} for sp in (2, 8)}
    assert R.completeness(full, "babel1") == ([1, 2, 3], None)
    missing = {k: v for k, v in full.items() if k[1] != 8}
    assert "missing" in R.completeness(missing, "babel1")[1]
    few = {k: {s: v[s] for s in (1, 2)} for k, v in full.items()}
    assert "common" in R.completeness(few, "babel1")[1]
    bad = {k: dict(v) for k, v in full.items()}
    bad[("babel1", 8)][2] = rec(99)
    assert "coverage" in R.completeness(bad, "babel1")[1]
    assert R.DECLARED_SPANS["babel1"] == (2, 8) and all(8 in v for v in R.DECLARED_SPANS.values())
    return True


def test_threeseed_babel1_comparison_runs_before_the_redecodes():
    order = R7.STAGE_ORDER
    assert order.index("H1") < order.index("S0") and order.index("H2") < order.index("S0") and order[-1] == "K"
    return True


def test_analysis_hash_covers_the_selector_code():
    from script.week5.timescale_inspection import report7 as R
    h = R.analysis_hash()
    assert len(h) == 16 and h != R7.source_hash()
    return True


ALL_TESTS = [
    test_agreement_is_label_permutation_invariant_and_drops_for_different_boundaries,
    test_k1_prefers_the_span_whose_seeds_agree,
    test_run_lengths_and_median_duration,
    test_decorrelation_lag_is_longer_for_slower_features,
    test_pick_rules_and_gate,
    test_declared_gate_never_passes_on_incomplete_data,
    test_occupancy_exposes_collapse_that_k1_rewards,
    test_completeness_requires_declared_spans_common_seeds_and_identical_coverage,
    test_threeseed_babel1_comparison_runs_before_the_redecodes,
    test_analysis_hash_covers_the_selector_code,
    test_stage_contents_and_s0_sources_exist,
    test_heldout_registration_leaves_existing_tokenizers_and_identities_unchanged,
    test_expired_budget_starts_nothing_and_dry_run_projects,
    test_resume_rejects_a_changed_implementation,
    test_g7_generalized_t0_reproduces_rkx_v2_on_babel1,
    test_s0_reproduces_a_saved_unit_exactly_and_never_writes_upstream,
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
