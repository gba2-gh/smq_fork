"""Tests for phase 1. Run: python -m script.week5.timescale_inspection.tests1"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from script.action_transport import continuous, transport  # noqa: E402
from script.week5.timescale_inspection import radius as RAD  # noqa: E402
from script.week5.timescale_inspection import run_phase1 as R1  # noqa: E402


def test_current_matches_original_kernel_function():
    for fps, window in ((60, 15), (50, 12), (30, 8)):
        for n in (0, 1, 2, 3, 5, 27, 500):
            assert RAD.CURRENT.half_width(fps, window, n) == transport.kernel_half_width(fps, window, n)
    return True


def test_settings_resolution():
    f = lambda s, fps, w, n: s.half_width(fps, w, n)
    assert f(RAD.BY_NAME["rel_4"], 30, 8, 16) == 1  # floor of 0.64 -> floored at 1
    assert f(RAD.BY_NAME["rel_4"], 30, 8, 500) == 20
    assert f(RAD.BY_NAME["rel_2"], 30, 8, 27) == 1
    assert f(RAD.BY_NAME["sec_0.25"], 30, 8, 100) == 1 and f(RAD.BY_NAME["sec_0.25"], 60, 15, 100) == 1
    assert f(RAD.BY_NAME["sec_0.5"], 30, 8, 100) == 2 and f(RAD.BY_NAME["sec_1.0"], 30, 8, 100) == 4
    assert f(RAD.BY_NAME["sec_1.0"], 60, 15, 100) == 4 and f(RAD.BY_NAME["sec_1.0"], 50, 12, 100) == 4
    assert f(RAD.BY_NAME["fixed_0"], 30, 8, 100) == 0 and f(RAD.BY_NAME["fixed_4"], 30, 8, 3) == 2  # clamp to L-1
    assert all(s.half_width(30, 8, 1) == 0 for s in RAD.SCREEN)  # a one-token recording has no temporal term
    return True


def test_signature_detects_identical_solves():
    lens = [20, 27, 40, 55]
    sig = lambda s, fps=30, w=8: RAD.signature(s, fps, w, lens)
    assert sig(RAD.BY_NAME["fixed_3"]) == sig(RAD.CURRENT)  # BABEL: fps // window = 3
    assert sig(RAD.BY_NAME["fixed_4"], 60, 15) == sig(RAD.CURRENT, 60, 15)  # HuGaDB: 4
    assert sig(RAD.BY_NAME["sec_0.25"]) == sig(RAD.BY_NAME["fixed_1"])
    assert sig(RAD.BY_NAME["fixed_2"]) != sig(RAD.CURRENT)
    return True


def test_override_scoped_recorded_and_restored():
    original = transport.kernel_half_width
    with RAD.kernel_override(RAD.BY_NAME["fixed_2"]) as used:
        assert transport.kernel_half_width(30, 8, 50) == 2 and used == {50: 2}
        from script.action_transport import categorical
        rec = categorical.RecordingInput("r", np.zeros(30, np.int32), np.full(30, 8, np.int32), 30, 8)
        _, _, kernel = categorical.recording_tensors(rec, 3, torch.device("cpu"))
        assert kernel.shape[-1] == 5 and used[30] == 2  # the solver's own kernel uses the override
    assert transport.kernel_half_width is original
    try:
        with RAD.kernel_override(RAD.BY_NAME["fixed_2"]):
            raise RuntimeError("x")
    except RuntimeError:
        pass
    assert transport.kernel_half_width is original
    return True


def _synthetic(num_recs=4, length=48, states=3, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(states, dim))
    recs, gts, starts, counts = [], {}, {}, {}
    for i in range(num_recs):
        labels = np.repeat(rng.integers(0, states, length // 8), 8)
        z = centres[labels] + 0.8 * rng.normal(size=(length, dim))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        name = f"r{i}"
        recs.append(continuous.ContinuousRecordingInput(name, labels.astype(np.int32), np.full(length, 4, np.int32),
                                                        30, 4, z, np.zeros(length, bool)))
        gts[name] = np.repeat(labels, 4).astype(np.int32)
        starts[name] = np.arange(length) * 4
        counts[name] = length * 4
    mu0, _ = continuous.build_mu0(np.arange(states), recs, states)
    return recs, mu0, gts, starts, counts, states


def _metrics(res):
    return [{k: v for k, v in row.items() if k != "seconds"} for row in res["ladder"]]


def test_fit_and_score_radius_changes_result_and_current_is_stable():
    recs, mu0, gts, starts, counts, states = _synthetic()
    run = lambda s: R1.fit_and_score(recs, mu0, states, R1.coeffs_for(0.3), s, 8, (50, 100), gts, starts, counts)
    cur, cur2 = run(RAD.CURRENT), run(RAD.CURRENT)
    assert _metrics(cur) == _metrics(cur2), "a decode is deterministic"
    same = run(RAD.Setting("fixed_7", "fixed", 7))  # fps // window = 30 // 4 = 7 on this grid
    assert _metrics(same) == _metrics(cur) and same["half_widths_used"] == cur["half_widths_used"]
    off = run(RAD.BY_NAME["fixed_0"])
    assert _metrics(off) != _metrics(cur) and set(off["half_widths_used"].values()) == {0}
    assert cur["half_widths_used"] == {"48": 7}
    return True


def test_beta_changes_the_solve():
    recs, mu0, gts, starts, counts, states = _synthetic()
    run = lambda b: R1.fit_and_score(recs, mu0, states, R1.coeffs_for(b), RAD.CURRENT, 8, (50,), gts, starts, counts)
    assert _metrics(run(0.1)) != _metrics(run(0.5))
    assert R1.coeffs_for(0.3) == R1.ot.COEFFS
    return True


def test_selection_uses_only_the_development_set_and_resumes():
    from script.week5.timescale_inspection import report1
    with tempfile.TemporaryDirectory() as tmp:
        run = R1.Runner.__new__(R1.Runner)
        run.out = Path(tmp)
        vals = {"sec_0.25": 10, "sec_0.5": 14, "sec_1.0": 12, "rel_2": 11, "rel_4": 11, "rel_8": 9}
        held_out = {"sec_0.25": 99, "sec_0.5": 1, "sec_1.0": 1, "rel_2": 1, "rel_4": 50, "rel_8": 1}
        seen = []

        def last_f1(unit):
            seen.append(unit.dataset)
            return float(vals[unit.setting.name]) + unit.seed % 3 * 0.0
        run.last_f1 = last_f1
        out = run.select()
        assert out["chosen"] == {"sec": "sec_0.5", "rel": "rel_2"}, out  # rel tie 11/11 -> earlier (smaller radius)
        assert set(seen) == {"babel2"} and held_out  # BABEL-3 / others never consulted
        seen.clear()
        assert run.select() == out and not seen  # a stored selection is not recomputed
        assert report1.SETTING_ORDER[0] == "current"
    return True


def test_direct_inputs_equal_the_repository_direct_arms_on_real_prep():
    from script.quantized_input_comparison import ot_arms
    from script.quantized_input_comparison.artifacts import Loader
    prep = ROOT / "results" / "week5" / "raw_kmeans_cross_dataset" / "rkx_v2" / "prep" / "babel2__global__seed111"
    if not (prep / "manifest.json").exists():
        return True  # source run not present on this machine
    run = R1.Runner.__new__(R1.Runner)
    run.source = type("S", (), {"dir": prep.parents[1]})()
    loader = Loader(prep)
    names = sorted(loader.recording_names())
    for inp, arm in (("pca", "direct_continuous"), ("centre", "direct_quantized")):
        unit = R1.Unit("babel2", "global", 111, RAD.CURRENT, 0.3, "inputs", inp)
        _, spec, _, _, emb, zero, codes, lens, starts = run.load_inputs(unit)
        ref = {r.name: r for r in ot_arms.continuous_recordings_from_raw(loader, "raw_kmeans", arm, names)}
        for n in names:
            assert np.array_equal(emb[n], ref[n].z) and np.array_equal(zero[n], ref[n].zero_flags), (inp, n)
            assert np.array_equal(codes[n], ref[n].codes) and np.array_equal(lens[n], ref[n].lengths)
        assert float(np.abs(np.linalg.norm(emb[names[0]], axis=1) - 1).max()) < 1e-9
    return True


def test_unit_ids_keep_context_ids_and_separate_inputs():
    c = R1.Unit("babel2", "global", 111, RAD.CURRENT, 0.3, "screen")
    p = R1.Unit("babel2", "global", 111, RAD.CURRENT, 0.3, "inputs", "pca")
    q = R1.Unit("babel2", "global", 111, RAD.CURRENT, 0.3, "inputs", "centre")
    assert c.uid == "babel2__global__seed111__current__beta0.3"
    assert len({c.uid, p.uid, q.uid}) == 3 and p.uid.endswith("__in-pca")
    return True


def test_report_reads_inputs_and_gives_the_reading():
    import json
    from script.week5.timescale_inspection import report1
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "units").mkdir()
        (out / "manifest.json").write_text(json.dumps(dict(
            run_id="t", status="finished", identity=dict(source_run="rkx_v2", outer_cap=50))))
        base = {"context": 40.0, "centre": 20.0, "pca": 30.0}
        for inp, v in base.items():
            for seed in R1.SEEDS:
                uid = f"hugadb__global__seed{seed}__current__beta0.3" + ("" if inp == "context" else f"__in-{inp}")
                d = out / "units" / uid
                d.mkdir()
                ladder = [dict(rung_steps=4000, MoF=1.0, Edit=2.0, F1_10=3.0, F1_25=4.0, F1_50=v + seed % 3)]
                (d / "info.json").write_text(json.dumps(dict(
                    status="done", unit=uid, dataset="hugadb", normalization="global", seed=seed, input=inp,
                    setting="current", beta=0.3, ot_status="capped", ladder=ladder,
                    segment_stats=dict(mean_pred_segments=5, mean_gt_segments=10, mean_pred_states_used=3))))
        text = report1.write_report(out).read_text(encoding="utf-8")
        assert "## No-contextualizer inputs" in text and "the contextualizer adds" in text
        assert "quantization costs" in text and "context − centre" in text
    return True


ALL_TESTS = [
    test_current_matches_original_kernel_function,
    test_settings_resolution,
    test_signature_detects_identical_solves,
    test_override_scoped_recorded_and_restored,
    test_fit_and_score_radius_changes_result_and_current_is_stable,
    test_beta_changes_the_solve,
    test_selection_uses_only_the_development_set_and_resumes,
    test_direct_inputs_equal_the_repository_direct_arms_on_real_prep,
    test_unit_ids_keep_context_ids_and_separate_inputs,
    test_report_reads_inputs_and_gives_the_reading,
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
