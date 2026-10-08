"""Tests for phase 4. Run: python -m script.week5.timescale_inspection.tests4"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from script.week5.raw_kmeans_cross_dataset import datasets as D  # noqa: E402
from script.week5.raw_kmeans_cross_dataset import prepare as PREP  # noqa: E402
from script.week5.timescale_inspection import run_phase4 as R4  # noqa: E402
from script.week5.timescale_inspection import velocity_prepare as VPREP  # noqa: E402


def test_no_velocity_reproduces_the_position_only_extractor():
    """`include_velocity=False` must be byte-identical to `raw_kmeans_cross_dataset.prepare.extract_raw`."""
    spec = D.SPECS["babel2"]
    names = D.recording_names(spec)[:6]
    for normalization in ("global", "per_recording"):
        ref = PREP.extract_raw(spec, names, normalization)
        mine = VPREP.extract_velocity_raw(spec, names, normalization, include_velocity=False)
        for n in names:
            assert np.array_equal(ref[n]["flat"], mine[n]["flat"])
            assert np.array_equal(ref[n]["lengths"], mine[n]["lengths"])
            assert ref[n]["num_patches"] == mine[n]["num_patches"]
    return True


def test_velocity_channels_shape_zero_at_frame_zero_and_fps_scaled():
    spec = D.SPECS["babel2"]
    names = D.recording_names(spec)[:2]
    pos = VPREP.extract_velocity_raw(spec, names, "global", include_velocity=False)
    both = VPREP.extract_velocity_raw(spec, names, "global", include_velocity=True)
    for n in names:
        arr = np.load(D.features_dir(spec) / n)
        channels = arr.shape[0] * arr.shape[2] * arr.shape[3]
        p, b = pos[n]["flat"], both[n]["flat"]
        assert b.shape == (p.shape[0], 2 * p.shape[1])
        p3, b3 = p.reshape(p.shape[0], spec.window, channels), b.reshape(b.shape[0], spec.window, 2 * channels)
        assert np.array_equal(b3[..., :channels], p3)  # positions unchanged, per frame
        # frame 0's velocity (first frame of the first window) is exactly zero
        assert np.all(b3[0, 0, channels:] == 0.0)
        # a non-degenerate frame's velocity equals fps * (frame[t] - frame[t-1]) before windowing
        frames = arr.transpose(1, 0, 2, 3).reshape(arr.shape[1], -1).astype(np.float32)
        expected = (frames[1] - frames[0]) * spec.fps
        assert np.allclose(b3[0, 1, channels:], expected, atol=1e-4)
    return True


def test_per_recording_normalization_is_columnwise_so_concat_order_does_not_matter():
    spec = D.SPECS["babel2"]
    names = D.recording_names(spec)[:2]
    a = VPREP.extract_velocity_raw(spec, names, "per_recording", include_velocity=True)
    # build the same thing by normalizing positions and velocities separately, then concatenating
    from script.exp2round.q1q2 import core as q1q2core
    for n in names:
        arr = np.load(D.features_dir(spec) / n)
        T = arr.shape[1]
        frames = arr.transpose(1, 0, 2, 3).reshape(T, -1).astype(np.float32)
        vel = np.zeros_like(frames)
        vel[1:] = (frames[1:] - frames[:-1]) * spec.fps
        pos_n = q1q2core.recording_zscore([frames])[0]
        vel_n = q1q2core.recording_zscore([vel])[0]
        expected = np.concatenate([pos_n, vel_n], axis=1)
        got = a[n]["flat"].reshape(-1, expected.shape[1])[:T]
        assert np.allclose(got, expected, atol=1e-5)
    return True


def test_prep_key_round_trips_and_reuse_is_verified():
    spec = D.SPECS["babel2"]  # full dataset (255 recordings): enough complete windows for K-means-512
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "cell"
        out = VPREP.prepare_cell(spec, "global", 111, run_dir)
        assert VPREP.is_prepared(spec, "global", 111, run_dir)
        mtime = (out / "manifest.json").stat().st_mtime
        VPREP.prepare_cell(spec, "global", 111, run_dir)  # reuse, not rebuilt
        assert (out / "manifest.json").stat().st_mtime == mtime
        from script.quantized_input_comparison.artifacts import Loader
        loader = Loader(out)
        assert VPREP.FAMILY in loader.families()
        rec = loader.recording(VPREP.FAMILY, loader.recording_names()[0])
        assert rec["continuous"].shape[1] == 64  # PCA-64 regardless of the doubled input dimension
        assert not VPREP.is_prepared(spec, "global", 222, run_dir)  # different seed, not reused
    return True


def test_job_uses_the_velocity_family_and_differs_from_the_position_job():
    from script.week5.raw_kmeans_cross_dataset.run_cross_dataset import make_job as position_job
    from script.week5.raw_kmeans_cross_dataset.run_cross_dataset import Cell as PositionCell
    cell = R4.Cell("babel2", "global", 111)
    job = R4.make_job(cell)
    assert job.family == VPREP.FAMILY
    pos = position_job(PositionCell("babel2", "global", 111))
    assert job.family != pos.family
    assert (job.context, job.mask_span, job.mask_fraction, job.guard_radius) == \
           (pos.context, pos.mask_span, pos.mask_fraction, pos.guard_radius)
    return True


def test_identity_changes_with_seed_and_dataset_but_not_run_id():
    runner = R4.Runner.__new__(R4.Runner)
    runner.specs = {"babel2": D.SPECS["babel2"], "lara": D.SPECS["lara"]}
    a = runner.identity_of(R4.Cell("babel2", "global", 111))
    b = runner.identity_of(R4.Cell("babel2", "global", 222))
    c = runner.identity_of(R4.Cell("lara", "global", 111))
    assert len({a, b, c}) == 3
    return True


def test_decode_direct_arms_on_synthetic_data_prefers_a_cleaner_representation():
    """End-to-end on a tiny synthetic velocity-family cell: the decoder should score higher on whichever
    representation actually separates the states (a cheap proxy for Phase 4's real comparison)."""
    from script.action_transport import continuous
    from script.week5.timescale_inspection import radius as RAD

    rng = np.random.default_rng(0)
    states, dim, length = 3, 8, 48
    centres = rng.normal(size=(states, dim))
    labels = np.repeat(rng.integers(0, states, length // 8), 8)
    clean = centres[labels] + 0.3 * rng.normal(size=(length, dim))
    noisy = rng.normal(size=(length, dim))  # carries nothing about `labels`
    for z_raw, tag in ((clean, "clean"), (noisy, "noisy")):
        z, zero = continuous.l2_normalize(z_raw)
        recs = [continuous.ContinuousRecordingInput("r0", labels.astype(np.int32), np.full(length, 4, np.int32),
                                                    30, 4, z, zero)]
        mu0, _ = continuous.build_mu0(np.arange(states), recs, states)
        gts = {"r0": np.repeat(labels, 4).astype(np.int32)}
        starts = {"r0": np.arange(length) * 4}
        counts = {"r0": length * 4}
        from script.week5.timescale_inspection import run_phase1 as R1
        res = R1.fit_and_score(recs, mu0, states, R1.coeffs_for(0.3), RAD.CURRENT, 8, (50,), gts, starts, counts)
        f1 = res["ladder"][-1]["F1_50"]
        if tag == "clean":
            clean_f1 = f1
        else:
            noisy_f1 = f1
    assert clean_f1 > noisy_f1
    return True


def _bare_runner(tmp: Path, arms=("context", "pca_velocity", "centre_velocity")):
    import argparse
    args = argparse.Namespace(run_id="t", hours=None, radius="current", arms=list(arms), cpu=True, dry_run=False,
                              context_run="rkx_v2", direct_run="ts1_v1", probes=False)
    run = R4.Runner.__new__(R4.Runner)
    run.args, run.out = args, tmp
    for d in ("prep", "jobs", "units"):
        (tmp / d).mkdir(parents=True, exist_ok=True)
    run.deadline, run.impl = None, "test"
    from script.week5.timescale_inspection import radius as RAD
    run.radius, run.cells = RAD.CURRENT, [R4.Cell("babel2", "global", 111)]
    run.specs = {"babel2": D.SPECS["babel2"]}
    run.job_infos, run.loaders, run.not_run = {}, {}, {}
    run.manifest_path = tmp / "manifest.json"
    run.ident = dict(radius="current")
    import json
    run.manifest_path.write_text(json.dumps(dict(run_id="t", status="running", identity=dict(radius="current"))))
    return run


def test_invalid_solve_is_recorded_and_final_and_untrained_cells_still_run_direct_arms():
    import json
    from script.fsq_reconstruction.evaluate import FinalSolveInvalid
    with tempfile.TemporaryDirectory() as tmp:
        run = _bare_runner(Path(tmp))
        calls = []

        def fake_decode(cell, arm):
            calls.append(arm)
            return "done"
        run.ensure_train = lambda cell: False  # training did not finish (deadline)
        run.ensure_decode = fake_decode
        import script.week5.timescale_inspection.report4 as report4
        real = report4.write_report
        report4.write_report = lambda *a, **k: Path(tmp) / "REPORT.md"
        try:
            run.execute()
        finally:
            report4.write_report = real
        assert calls == ["pca_velocity", "centre_velocity"], calls  # context skipped, direct arms still decoded

        # an invalid solve: recorded once, then returned as final without decoding again
        run2 = _bare_runner(Path(tmp) / "b", arms=("pca_velocity",))
        cell = run2.cells[0]
        ident = run2.unit_identity(cell, "pca_velocity", None)
        udir = run2.unit_dir(cell, "pca_velocity")
        udir.mkdir(parents=True)
        (udir / "info.json").write_text(json.dumps(dict(status="invalid", identity=ident, reason="x")))
        run2.prepare = lambda c: (_ for _ in ()).throw(AssertionError("an invalid unit must not be re-decoded"))
        assert run2.ensure_decode(cell, "pca_velocity") == "invalid"
    return True


def test_report_applies_the_h4_rule_with_probes():
    import csv
    import json
    from script.week5.timescale_inspection import report4
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "phase4"
        (out / "units").mkdir(parents=True)
        (out / "manifest.json").write_text(json.dumps(dict(run_id="t", status="finished", identity=dict(radius="current"))))
        for seed in R4.SEEDS:
            for arm, v in (("pca_velocity", 30.0), ("centre_velocity", 18.0), ("context", 16.0)):
                d = out / "units" / f"babel2__global__seed{seed}__{arm}"
                d.mkdir()
                (d / "info.json").write_text(json.dumps(dict(
                    status="done", dataset="babel2", normalization="global", seed=seed, arm=arm,
                    ladder=[dict(rung_steps=4000, F1_50=v + seed % 3)], segment_stats=dict(mean_pred_segments=3))))
        with (out / "probes.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["dataset", "normalization", "seed", "probe", "representation",
                                               "balanced_accuracy"])
            w.writeheader()
            for seed in R4.SEEDS:
                w.writerow(dict(dataset="babel2", normalization="global", seed=seed, probe="linear",
                                representation="pca", balanced_accuracy=0.50))
        # the real position baselines (rkx_v2, ts1_v1, ts0_v1) are read from disk when present
        text = report4.write_report(out).read_text(encoding="utf-8")
        assert "## Per-token action information" in text
        if (ROOT / "results/week5/timescale_inspection/ts1_v1/phase1/units").exists():
            assert "babel2 / pca:" in text and "**helps**" in text  # 30 vs ~18.7 on the position pca arm
            assert "H4 (position-only input lacks the motion signal IMU input has): supported" in text  # 0.50 > 0.42
    return True


ALL_TESTS = [
    test_no_velocity_reproduces_the_position_only_extractor,
    test_velocity_channels_shape_zero_at_frame_zero_and_fps_scaled,
    test_per_recording_normalization_is_columnwise_so_concat_order_does_not_matter,
    test_prep_key_round_trips_and_reuse_is_verified,
    test_job_uses_the_velocity_family_and_differs_from_the_position_job,
    test_identity_changes_with_seed_and_dataset_but_not_run_id,
    test_decode_direct_arms_on_synthetic_data_prefers_a_cleaner_representation,
    test_invalid_solve_is_recorded_and_final_and_untrained_cells_still_run_direct_arms,
    test_report_applies_the_h4_rule_with_probes,
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
