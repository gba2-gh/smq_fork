"""Focused verification for the cross-dataset raw K-means run. Fast: synthetic
data plus read-only checks against the real label files and saved SMQ scores.

    python -m script.week5.raw_kmeans_cross_dataset.run_cross_dataset --preflight

Set SMQ_SLOW_TESTS=1 to also reproduce `qic_v3`'s HuGaDB raw family from scratch (about two minutes).
`verify_run_dir` checks a finished smoke run (saved-prediction rescoring, explicit unfinished units).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from script.fsq_reconstruction.encoder_sweep import extract_raw_dataset
from script.quantized_input_comparison.artifacts import Loader
from script.week5.contextualizer_calibration import evaluate as E
from script.week5.contextualizer_calibration import plan as P
from script.week5.contextualizer_calibration import train as T
from script.week5.raw_kmeans_cross_dataset import datasets as D
from script.week5.raw_kmeans_cross_dataset import decode as DEC
from script.week5.raw_kmeans_cross_dataset import prepare as PREP
from script.week5.raw_kmeans_cross_dataset import report as REP

SYNTH = D.DatasetSpec("synth", 30, 8, 3, 4, False)


class SyntheticData:
    """A tiny on-disk dataset (features + labels) patched into the datasets module."""

    def __init__(self, tmp: Path, n=8, seed=0):
        rng = np.random.default_rng(seed)
        self.feat = tmp / "features"
        self.feat.mkdir(parents=True)
        self.gts, self.names = {}, []
        for i in range(n):
            T_ = int(rng.integers(300, 460))
            name = f"rec{i:02d}.npy"
            base = rng.normal(size=(2, T_, 3, 1)).astype(np.float32)
            labels = np.repeat(rng.integers(0, 3, size=T_ // 60 + 1), 60)[:T_]
            for c in range(3):  # make the classes separable in the signal
                base[:, labels == c] += 1.5 * c
            base[1, :, 0, 0] = 5.0  # a constant channel (exercises the std floor)
            np.save(self.feat / name, base)
            self.gts[name] = labels.astype(np.int32)
            self.names.append(name)
        self._features_dir, self._load_gts = D.features_dir, D.load_gts
        D.features_dir = lambda spec: self.feat
        D.load_gts = lambda spec, names: {n: self.gts[n] for n in names}

    def close(self):
        D.features_dir, D.load_gts = self._features_dir, self._load_gts


def small_scale():
    return replace(P.SMOKE, recording_cap=None)


# --- plan -----------------------------------------------------------------------------------------

def test_cells_and_frozen_configuration():
    from script.week5.raw_kmeans_cross_dataset import run_cross_dataset as X
    cells = X.make_cells()
    assert len(cells) == 18 and len({c.cell_id for c in cells}) == 18
    by = {}
    for c in cells:
        by.setdefault((c.dataset, c.normalization), []).append(c.seed)
    assert {k: tuple(v) for k, v in by.items()} == {
        ("hugadb", "global"): X.SEEDS, ("babel1", "global"): X.SEEDS, ("babel2", "global"): X.SEEDS,
        ("babel3", "global"): X.SEEDS, ("lara", "global"): X.SEEDS, ("lara", "per_recording"): X.SEEDS}
    assert [c.dataset for c in cells][-6:] == ["lara"] * 6, "LARa (slowest) runs last"
    ref = P.primary_job("reference", "raw")
    for c in cells:
        job = X.make_job(c)
        assert replace(job, job_id=ref.job_id, seed=ref.seed) == replace(ref, group="main"), "frozen configuration"
        assert (job.guard_radius, job.context, job.mask_span, job.horizon, job.seed) == (0, 128, 8, 5000, c.seed)
    s = D.SPECS
    assert (s["hugadb"].window, s["hugadb"].num_states, s["lara"].window, s["lara"].num_states) == (15, 10, 12, 8)
    assert all(s[b].window == 8 and s[b].fps == 30 and s[b].num_states == 5 for b in ("babel1", "babel2", "babel3"))
    from script.action_transport.transport import kernel_half_width
    assert [kernel_half_width(s[d].fps, s[d].window, 1000) for d in ("hugadb", "lara", "babel1")] == [4, 4, 3]
    assert abs(s["babel1"].token_seconds - 8 / 30) < 1e-12
    return True


# --- labels and scoring gate (real data, read-only) -------------------------------------------------------

def test_label_loading_and_scoring_gate_on_real_data():
    from script.week5.raw_kmeans_cross_dataset import run_cross_dataset as X
    for ds, spec in D.SPECS.items():
        g = X.gate_label_loading_and_scoring(spec)
        assert g["passed"] and g["labels_equal_saved_gt"], (ds, g)
        names = D.recording_names(spec)[:20]
        assert len({int(x) for v in D.load_gts(spec, names).values() for x in np.unique(v)}) <= spec.num_states, ds
    return True


# --- preparation --------------------------------------------------------------------------------------------

def test_global_extraction_equals_existing_raw_extractor_and_per_recording_normalization():
    with tempfile.TemporaryDirectory() as tmp:
        data = SyntheticData(Path(tmp))
        try:
            mine = PREP.extract_raw(SYNTH, data.names[:3], "global")
            ref = extract_raw_dataset(data.feat, SYNTH.window, names=data.names[:3])
            for n in data.names[:3]:
                assert np.array_equal(mine[n]["flat"], ref[n]["flat"]) and np.array_equal(mine[n]["lengths"], ref[n]["lengths"])
            norm = PREP.extract_raw(SYNTH, data.names[:3], "per_recording")
            for n in data.names[:3]:
                arr = np.load(data.feat / n)
                T_ = arr.shape[1]
                frames = norm[n]["flat"].reshape(-1, arr.shape[0] * arr.shape[2])[:T_]
                const_col = 1 * arr.shape[2] + 0  # channel 1, joint 0 in the (C, V) flattening of a frame
                keep = np.ones(frames.shape[1], dtype=bool)
                keep[(1 * 3 + 0)] = False
                assert np.allclose(frames[:, keep].mean(axis=0), 0, atol=1e-4)
                assert np.allclose(frames[:, keep].std(axis=0), 1, atol=1e-3)
                assert np.allclose(frames[:, ~keep], 0.0), "a constant channel centers to zero (std floor)"
                pad = norm[n]["flat"].reshape(-1, frames.shape[1])[T_:]
                assert pad.size == 0 or np.all(pad == 0), "terminal padding stays zero after normalization"
                assert not np.array_equal(norm[n]["flat"], mine[n]["flat"])
        finally:
            data.close()
    return True


def test_prepare_cell_artifacts_reuse_and_seeds():
    with tempfile.TemporaryDirectory() as tmp:
        data = SyntheticData(Path(tmp) / "d")
        try:
            out = Path(tmp) / "prep"
            a = PREP.prepare_cell(SYNTH, "global", 111, out / "a", num_codes=8)
            loader = Loader(a)
            assert loader.families() == ["raw_kmeans"] and loader.lookup("raw_kmeans").shape[0] == 8
            assert set(loader.split["train"]).isdisjoint(loader.split["val"]) and loader.split["train"] and loader.split["val"]
            for n in loader.recording_names():
                rec = loader.recording("raw_kmeans", n)
                assert len(rec["codes"]) == len(rec["lengths"]) == loader.manifest["num_windows"]["raw_kmeans"][n]
                assert rec["lengths"].sum() == loader.frame_count(n) and rec["lengths"][:-1].min(initial=8) == 8
            stamp = (a / "manifest.json").stat().st_mtime_ns
            PREP.prepare_cell(SYNTH, "global", 111, out / "a", num_codes=8)
            assert (a / "manifest.json").stat().st_mtime_ns == stamp, "verified artifacts are reused, not rebuilt"
            (a / "manifest.json").unlink()  # an incomplete directory is rebuilt deterministically
            PREP.prepare_cell(SYNTH, "global", 111, out / "a", num_codes=8)
            assert np.array_equal(Loader(a).lookup("raw_kmeans"), loader.lookup("raw_kmeans"))
            b = PREP.prepare_cell(SYNTH, "global", 222, out / "b", num_codes=8)
            c = PREP.prepare_cell(SYNTH, "per_recording", 111, out / "c", num_codes=8)
            lb, lc = Loader(b), Loader(c)
            assert lb.split["train"] == loader.split["train"] and lb.split["val"] == loader.split["val"], "fixed split seed"
            assert not np.array_equal(lb.lookup("raw_kmeans"), loader.lookup("raw_kmeans")), "the vocabulary seed matters"
            assert not np.array_equal(lc.lookup("raw_kmeans"), loader.lookup("raw_kmeans")), "normalization changes tokens"
            from script.week5.raw_kmeans_cross_dataset import run_cross_dataset as X
            assert X.compare_arrays(loader, Loader(a), "raw_kmeans", "raw_kmeans")["passed"]
            assert not X.compare_arrays(loader, lb, "raw_kmeans", "raw_kmeans")["passed"]
            diag = PREP.code_diagnostics(SYNTH, loader, data.gts)
            assert diag["occupied_codes"] > 0 and diag["perplexity"] > 1 and "mi_code_action" in diag
        finally:
            data.close()
    return True


# --- training + dataset-aware decoding ---------------------------------------------------------------------------

def test_training_and_decode_with_non_hugadb_geometry():
    from script.week5.raw_kmeans_cross_dataset import run_cross_dataset as X
    with tempfile.TemporaryDirectory() as tmp:
        data = SyntheticData(Path(tmp) / "d")
        try:
            prep = PREP.prepare_cell(SYNTH, "global", 111, Path(tmp) / "prep", num_codes=8)
            loader, scale = Loader(prep), small_scale()
            cell = X.Cell("synth", "global", 111)
            job = X.make_job(cell)
            jd = Path(tmp) / "job"
            jd.mkdir()
            info = T.run_training_job(job, scale, loader, jd, torch.device("cpu"), None, "id", "impl")
            assert info["exposure"]["accepted_updates"] == scale.horizon(5000) and set(info["roles"]) >= {"selected", "final"}
            row = next(r for r in json.loads((jd / "checkpoints.json").read_text())["ledger"] if r["step"] == info["roles"]["selected"])
            state = T.load_state(jd / "ckpt" / row["file"])["state_dict"]
            out = Path(tmp) / "eval"
            ident = DEC.eval_identity(state_sha=row["tensor_sha256"], job=job, scale=scale, outer_cap=3, spec=SYNTH,
                                      loader=loader, names=loader.recording_names(), prep_hash="p")
            other = DEC.eval_identity(state_sha=row["tensor_sha256"], job=job, scale=scale, outer_cap=3,
                                      spec=replace(SYNTH, num_states=4), loader=loader, names=loader.recording_names(),
                                      prep_hash="p")
            assert ident != other, "the dataset geometry is part of the evaluation identity"
            DEC.decode_state(job, scale, loader, SYNTH, state, 3, torch.device("cpu"), None, out,
                             dict(identity=ident, eval_id="e", job_id="j", family="raw_kmeans", step=info["roles"]["selected"],
                                  ckpt_role="selected", state_sha=row["tensor_sha256"]))
            assert E.eval_complete(out, ident, scale, loader, SYNTH.num_states)
            assert not E.eval_complete(out, ident, scale, loader), "state range is checked against the dataset's classes"
            with np.load(out / "ot.npz") as z:
                assert z["mu"].shape[0] == SYNTH.num_states
            with np.load(out / "predictions.npz") as z:
                assert all(z[k].max() < SYNTH.num_states for k in z.files)
            ladder = json.loads((out / "ladder.json").read_text())["rows"]
            metrics = DEC.rescore(out, SYNTH)
            assert abs(metrics["F1@50"] - ladder[-1]["F1_50"]) < 1e-9 and abs(metrics["MoF"] - ladder[-1]["MoF"]) < 1e-9
            stats = json.loads((out / "info.json").read_text())["segment_stats"]
            assert stats["num_states"] == 3 and stats["mean_pred_segments"] >= 1
        finally:
            data.close()
    return True


# --- runner mechanics --------------------------------------------------------------------------------------------

def _bare_run(tmp: Path):
    from script.week5.contextualizer_calibration import run_calibration as R
    from script.week5.raw_kmeans_cross_dataset import run_cross_dataset as X
    run = object.__new__(X.Run)
    run.args = type("A", (), dict(run_id="t", reference_run="ctx_cal_v1", hours=None, continue_on_gate_failure=False,
                                   datasets=None, smoke=True, resume=False))()
    run.smoke, run.scale, run.device, run.deadline, run.run_dir = True, small_scale(), torch.device("cpu"), None, tmp
    run.specs, run.cells = D.SPECS, X.make_cells()
    run.cell_by_id = {c.cell_id: c for c in run.cells}
    run.impl_hash, run.env = X.source_hash(), {"device": "cpu"}
    run.cell_identity = {c.cell_id: "id" for c in run.cells}
    run.job_infos, run.loaders, run.verified_evals, run.not_run = {}, {}, set(), {}
    run.timings, run.gates, run.blocked, run.stop = {}, {}, {}, False
    run.reference_dir = tmp / "ref"
    try:
        run.qic_dir = R.resolve_source("qic", "qic_v3")
    except SystemExit:
        run.qic_dir = None
    return run


def test_failed_training_not_retried_stale_records_removed_gates_stop():
    from script.week5.raw_kmeans_cross_dataset import run_cross_dataset as X
    with tempfile.TemporaryDirectory() as tmp:
        run = _bare_run(Path(tmp))
        cell = run.cells[0]
        run.prepare = lambda c: object()
        calls = {"n": 0}
        real = T.run_training_job

        def boom(*a, **k):
            calls["n"] += 1
            raise RuntimeError("late failure")

        T.run_training_job = boom
        try:
            assert not run.ensure_eval(cell, "selected") and not run.ensure_eval(cell, "final")
        finally:
            T.run_training_job = real
        assert calls["n"] == 1 and run.not_run[cell.cell_id]["status"] == "failed"
        assert all(run.not_run[f"{cell.cell_id}::{r}"]["status"] == "unavailable" for r in X.ROLES)
        path = run.record_path(cell, "selected")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(status="done", identity="x", step=1, state_sha="s")))
        run.job_infos.clear()
        assert run.verified_record(cell, "selected") is None and not path.exists(), "stale record removed"
        run.write_aggregates()
        assert (run.run_dir / "segmentation.csv").read_text() == ""
        run.args.continue_on_gate_failure = False
        run.gates = {}
        run.smoke = False
        qic = Loader(run.qic_dir) if run.qic_dir else None
        if qic is not None:  # a different seed's HuGaDB tokens differ from qic_v3's raw family -> gate 1 fails and stops
            other = Loader(run.qic_dir)
            other.lookup = lambda family: np.zeros((512, 64), dtype=np.float32)
            real_loader = X.Loader
            X.Loader = lambda path: qic
            try:
                run.gate_tokenizer(other)
            finally:
                X.Loader = real_loader
            assert run.gates["tokenizer_port"]["status"] == "failed" and run.stop
    return True


def test_resume_identity_covers_environment_and_cells():
    run_id_runs = []
    with tempfile.TemporaryDirectory() as tmp:
        run = _bare_run(Path(tmp))
        if run.qic_dir is None:
            return True
        run_id_runs.append(run.resume_identity())
        run.env = {"device": "cuda"}
        assert run.resume_identity() != run_id_runs[0], "switching device must change the resume identity"
        run.env = {"device": "cpu"}
        run.cell_identity = dict(run.cell_identity, **{run.cells[0].cell_id: "changed"})
        assert run.resume_identity() != run_id_runs[0]
    return True


def test_reading_guide_verdicts():
    def rows(ds, norm, values):
        return [dict(dataset=ds, normalization=norm, role="selected", seed=str(s), rung_steps="4000", **{"F1@50": str(v)})
                for s, v in zip((111, 222, 1538574472), values)]

    smq = dict(lara=16.4, babel1=22.4, babel2=27.5, babel3=19.4)
    good = rows("lara", "global", (20, 21, 22)) + rows("lara", "per_recording", (10, 11, 12)) + \
        rows("babel1", "global", (30, 31, 32)) + rows("babel2", "global", (40, 41, 42)) + rows("babel3", "global", (10, 11, 12))
    assert REP.verdict(good, smq)[0] == "transfers"
    bad = rows("lara", "global", (5, 6, 7)) + rows("lara", "per_recording", (5, 6, 7)) + \
        rows("babel1", "global", (30, 31, 32)) + rows("babel2", "global", (40, 41, 42)) + rows("babel3", "global", (30, 31, 32))
    assert REP.verdict(bad, smq)[0] == "does not transfer"
    mixed = rows("lara", "global", (5, 30, 7)) + rows("lara", "per_recording", (20, 21, 22)) + \
        rows("babel1", "global", (30, 31, 32)) + rows("babel2", "global", (10, 11, 12)) + rows("babel3", "global", (30, 31, 10))
    assert REP.verdict(mixed, smq)[0] == "mixed"
    seed_miss = rows("lara", "global", (20, 21, 22)) + rows("lara", "per_recording", (20, 21, 22)) + \
        rows("babel1", "global", (30, 31, 10)) + rows("babel2", "global", (40, 41, 42)) + rows("babel3", "global", (10, 11, 12))
    assert REP.verdict(seed_miss, smq)[0] == "mixed", "one seed below SMQ spoils that group"
    assert REP.verdict(good[:5], smq)[0] == "incomplete"
    return True


def test_tokenizer_port_reproduces_qic_v3_on_real_data():
    """Slow (SMQ_SLOW_TESTS=1): rebuild HuGaDB / global / seed 111 in a fresh process under the LAUNCHER's thread
    settings (8-thread caps) and compare with qic_v3 (gate 1). The preparation is pinned to the thread count that
    built qic_v3, so the launcher's cap must not matter."""
    if not os.environ.get("SMQ_SLOW_TESTS"):
        return True
    import subprocess
    code = (
        "import tempfile, sys; from pathlib import Path\n"
        "from script.week5.contextualizer_calibration import run_calibration as R\n"  # sets the 8-thread caps first
        "from script.week5.raw_kmeans_cross_dataset import datasets as D, prepare as PREP, run_cross_dataset as X\n"
        "from script.quantized_input_comparison.artifacts import Loader\n"
        "with tempfile.TemporaryDirectory() as t:\n"
        "    out = PREP.prepare_cell(D.SPECS['hugadb'], 'global', 111, Path(t) / 'h')\n"
        "    r = X.compare_arrays(Loader(out), Loader(R.resolve_source('qic', 'qic_v3')), 'raw_kmeans', 'raw_kmeans')\n"
        "    print('RESULT', r['passed'], r)\n")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8", OMP_NUM_THREADS="8", PYTHONPATH=str(ROOT))
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT")), out.stderr[-500:])
    assert line.startswith("RESULT True"), line
    return True


def _synthetic_run(tmp: Path, deadline=None, stop=False):
    from script.week5.raw_kmeans_cross_dataset import run_cross_dataset as X
    run = _bare_run(tmp)
    cell = X.Cell("synth", "global", 111)
    run.specs, run.cells, run.cell_by_id = {"synth": SYNTH}, [cell], {cell.cell_id: cell}
    run.cell_identity, run.num_codes = {cell.cell_id: "id"}, 8
    run.deadline, run.stop = deadline, stop
    return run, cell


def test_resume_with_expired_budget_keeps_verified_results_and_invalid_is_final():
    import time
    from script.week5.raw_kmeans_cross_dataset import run_cross_dataset as X
    with tempfile.TemporaryDirectory() as tmp:
        data = SyntheticData(Path(tmp) / "d")
        try:
            run, cell = _synthetic_run(Path(tmp) / "run")
            assert all(run.ensure_eval(cell, role) for role in X.ROLES)
            run.write_aggregates()
            rows = REP.read(run.run_dir / "segmentation.csv")
            assert len(rows) == 2 * len(run.scale.rungs) and not run.not_run
            records = sorted(p.name for p in (run.run_dir / "eval_records").iterdir())
            assert len(records) == 2
            # A later invocation with an already-expired budget and the stop flag set: verification must not
            # depend on the deadline, so completed work is neither deleted nor reported as not trained.
            late, cell = _synthetic_run(Path(tmp) / "run", deadline=time.perf_counter() - 10, stop=True)
            assert late.ensure_train(cell) is True
            assert all(late.verified_record(cell, role) is not None for role in X.ROLES)
            late.write_aggregates()
            assert sorted(p.name for p in (late.run_dir / "eval_records").iterdir()) == records
            assert len(REP.read(late.run_dir / "segmentation.csv")) == len(rows) and not late.not_run
            # A deterministic invalid solve is final: never decoded again on resume.
            again, cell = _synthetic_run(Path(tmp) / "run")
            path = again.record_path(cell, "selected")
            rec = json.loads(path.read_text())
            path.write_text(json.dumps(dict(rec, status="invalid", reason="nonfinite")))
            real = DEC.decode_state

            def forbidden(*a, **k):
                raise AssertionError("an invalid evaluation must not be retried")

            DEC.decode_state = forbidden
            try:
                assert again.ensure_eval(cell, "selected") is False
            finally:
                DEC.decode_state = real
            assert again.not_run[f"{cell.cell_id}::selected"]["status"] == "invalid"
        finally:
            data.close()
    return True


def test_gates_persist_across_invocations():
    with tempfile.TemporaryDirectory() as tmp:
        first = _bare_run(Path(tmp))
        first.gates = {"tokenizer_port": dict(status="passed"), "pipeline_port": dict(status="passed")}
        first.save_gates()
        second = _bare_run(Path(tmp))
        second.gates = {"pipeline_port": dict(status="recomputed")}
        second.load_previous_gates()
        assert second.gates["tokenizer_port"]["status"] == "passed", "an earlier gate outcome is carried forward"
        assert second.gates["pipeline_port"]["status"] == "recomputed", "a gate recomputed now is not overwritten"
        second.gates = {}
        second.smoke = False
        second.args.continue_on_gate_failure = False
        second.cells = []  # anchor cell not selected
        second.ensure_tokenizer_gate()
        assert second.gates["tokenizer_port"]["status"] == "not_applicable_dataset_not_selected"
    return True


def test_pipeline_gate_comparator_against_real_reference_artifacts():
    from script.week5.raw_kmeans_cross_dataset import run_cross_dataset as X
    ref = ROOT / "results" / "week5" / "contextualizer_calibration" / "ctx_cal_v1"
    info_path = ref / "jobs" / X.REFERENCE_JOB / "training_info.json"
    rec_path = ref / "eval_records" / f"{X.REFERENCE_JOB}__selected__cap50.json"
    if not (info_path.exists() and rec_path.exists()):
        return True  # reference run not present on this machine
    info = json.loads(info_path.read_text())
    eval_dir = ref / "evals" / json.loads(rec_path.read_text())["identity"]
    sha = info["role_tensor_sha256"]["selected"]
    same = X.compare_to_reference(info, sha, eval_dir, info, sha, eval_dir)
    assert same["passed"] and same["metrics_exact"] and same["checkpoint_tensors_equal"], same
    assert abs(same["f1_50_last_rung"] - 50.0) < 5e-3 and info["selected_step"] == 4100
    off = X.compare_to_reference(dict(info, selected_step=info["selected_step"] + 100), sha, eval_dir, info, sha, eval_dir)
    assert not off["passed"] and not off["selected_step_equal"]
    return True


ALL_TESTS = [
    test_cells_and_frozen_configuration,
    test_label_loading_and_scoring_gate_on_real_data,
    test_global_extraction_equals_existing_raw_extractor_and_per_recording_normalization,
    test_prepare_cell_artifacts_reuse_and_seeds,
    test_training_and_decode_with_non_hugadb_geometry,
    test_failed_training_not_retried_stale_records_removed_gates_stop,
    test_resume_identity_covers_environment_and_cells,
    test_reading_guide_verdicts,
    test_resume_with_expired_budget_keeps_verified_results_and_invalid_is_final,
    test_gates_persist_across_invocations,
    test_pipeline_gate_comparator_against_real_reference_artifacts,
    test_tokenizer_port_reproduces_qic_v3_on_real_data,
]


def run_all() -> bool:
    ok = True
    for fn in ALL_TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
        except Exception as exc:  # noqa: BLE001
            import traceback
            ok = False
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}", flush=True)
    print(f"{'ALL PASSED' if ok else 'SOME FAILED'} ({len(ALL_TESTS)} tests)")
    return ok


def verify_run_dir(run_dir: Path, cells, specs) -> bool:
    """Plans saved, every cell done or explicit in not_run.csv, no duplicate rows, and every realized evaluation's
    saved predictions rescore to its saved final-rung metrics."""
    run_dir, problems = Path(run_dir), []
    if len(REP.read(run_dir / "planned_cells.csv")) != len(cells):
        problems.append("planned_cells.csv row count")
    not_run = {r["unit"] for r in REP.read(run_dir / "not_run.csv")}
    done_cells = {r["cell_id"] for r in REP.read(run_dir / "cells.csv")}
    seg = REP.read(run_dir / "segmentation.csv")
    keys = [(r["cell_id"], r["role"], r["rung_steps"]) for r in seg]
    if len(keys) != len(set(keys)):
        problems.append("duplicate segmentation rows")
    done_evals = {(r["cell_id"], r["role"]) for r in seg}
    for c in cells:
        if c.cell_id not in done_cells and c.cell_id not in not_run and not any(
                u.startswith(c.cell_id) for u in not_run):
            problems.append(f"cell {c.cell_id} neither done nor explained")
        for role in ("selected", "final"):
            if (c.cell_id, role) not in done_evals and f"{c.cell_id}::{role}" not in not_run:
                problems.append(f"evaluation {c.cell_id}::{role} neither done nor explained")
    checked = 0
    by_identity = {r["identity"]: r for r in seg}
    for identity, r in sorted(by_identity.items()):
        edir = run_dir / "evals" / identity
        ladder = json.loads((edir / "ladder.json").read_text())["rows"]
        last = max(ladder, key=lambda x: x["rung_steps"])
        metrics = DEC.rescore(edir, specs[r["dataset"]])
        for k_csv, k_json in (("MoF", "MoF"), ("Edit", "Edit"), ("F1@10", "F1_10"), ("F1@25", "F1_25"), ("F1@50", "F1_50")):
            if abs(float(metrics[k_csv]) - float(last[k_json])) > 1e-9:
                problems.append(f"{identity} {k_csv} rescoring {metrics[k_csv]} != saved {last[k_json]}")
        checked += 1
    result = dict(ok=not problems, problems=problems, rescored_evaluations=checked, cells_done=len(done_cells),
                  evals_done=len(done_evals), unfinished=sorted(not_run))
    (run_dir / "smoke_verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return not problems


if __name__ == "__main__":
    raise SystemExit(0 if run_all() else 1)
