"""Validation suite for the frozen-representation study
(`frozen_study.py --validate-only`), per
docs/plans/FROZEN_STUDY_AGENT_INSTRUCTIONS.md §7. Uses synthetic recordings
throughout (no dependence on the real datasets/checkpoints being present on
this machine) except the final identity test, which only needs a
nonexistent stage_b_run name.
"""

from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.action_transport import categorical, config, continuous, transport
from script.action_transport import frozen_study as fs


# --- synthetic fixtures --------------------------------------------------------------

@dataclass
class _FakeMeta:
    name: str
    subject: str
    frame_count: int
    starts: np.ndarray
    lengths: np.ndarray


def _toy_meta(name: str, lengths: np.ndarray, subject: str = "s0") -> _FakeMeta:
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int32)
    return _FakeMeta(name, subject, int(lengths.sum()), starts, lengths.astype(np.int32))


def _toy_rec(name: str, n_windows: int, dim: int, seed: int, window_len: int = 15, fps: int = 60):
    rng = np.random.default_rng(seed)
    z, zero = continuous.l2_normalize(rng.normal(size=(n_windows, dim)))
    lengths = np.full(n_windows, window_len, dtype=np.int32)
    codes = np.arange(n_windows, dtype=np.int32)
    return continuous.ContinuousRecordingInput(name, codes, lengths, fps, window_len, z, zero)


# --- 1. Pooling ------------------------------------------------------------------------

def test_pool_windows_hand_checked_with_edge_truncation():
    pca = np.array([[1., 0.], [0., 1.], [2., 0.], [0., 2.], [3., 0.]])
    lengths = np.array([10, 10, 10, 10, 10])
    out = fs.pool_windows(pca, lengths, h=1)
    assert np.allclose(out[0], (10 * pca[0] + 10 * pca[1]) / 20.0)  # edge: {0,1} only
    assert np.allclose(out[2], (pca[1] * 10 + pca[2] * 10 + pca[3] * 10) / 30.0)  # interior: {1,2,3}
    assert np.allclose(out[4], (pca[3] * 10 + pca[4] * 10) / 20.0)  # edge: {3,4} only


def test_pool_windows_h0_equals_identity():
    pca = np.random.default_rng(0).normal(size=(6, 3))
    lengths = np.array([15, 15, 15, 15, 15, 7])  # includes a terminal partial window
    assert np.allclose(fs.pool_windows(pca, lengths, h=0), pca)


def test_pool_windows_terminal_partial_window_is_weighted_by_its_real_length():
    pca = np.array([[1., 0.], [0., 1.]])
    lengths = np.array([15, 7])  # terminal window shorter than the grid window
    out = fs.pool_windows(pca, lengths, h=1)
    expected = (15 * pca[0] + 7 * pca[1]) / 22.0
    assert np.allclose(out[0], expected)
    assert np.allclose(out[1], expected)


def test_pool_windows_no_averaging_across_recordings():
    # pool_windows operates on one recording's array only -- calling it twice
    # on unrelated arrays must not leak state between calls.
    a = fs.pool_windows(np.ones((3, 2)), np.array([10, 10, 10]), h=1)
    b = fs.pool_windows(np.zeros((3, 2)) + 5.0, np.array([10, 10, 10]), h=1)
    assert np.allclose(a, 1.0) and np.allclose(b, 5.0)


# --- 2. Oracle prototypes ---------------------------------------------------------------

def test_build_oracle_prototypes_hand_computed_1indexed_and_present_only():
    @dataclass
    class _R:
        name: str
        z: np.ndarray
        lengths: np.ndarray

    r1 = _R("a", np.array([[1., 0.], [0., 1.], [1., 0.]]), np.array([10, 10, 10]))
    r2 = _R("b", np.array([[0., 1.]]), np.array([5]))
    wl1, wl2 = np.array([1, 2, 1]), np.array([2])  # 1-indexed labels; class 5 never appears
    mu, classes = fs.build_oracle_prototypes([r1, r2], [wl1, wl2])
    assert list(classes) == [1, 2]  # only present classes, in ascending label-id order
    assert np.allclose(mu[0], [1.0, 0.0])  # class 1: r1 windows 0,2 (weight 10 each) -> [1,0]
    assert np.allclose(mu[1], [0.0, 1.0])  # class 2: r1 window1 (w10) + r2 window0 (w5) -> [0,1]


def test_identity_scoring_maps_state_index_to_label_id():
    classes = np.array([3, 7])  # present labels are not 0/1
    states = np.array([0, 1, 0, 1])
    assert list(classes[states]) == [3, 7, 3, 7]


# --- 3. Ladder ---------------------------------------------------------------------------

def test_ladder_cumulative_rungs_equal_single_solve_with_no_early_stop():
    rec = _toy_rec("r0", 5, 3, seed=2)
    mu, _ = continuous.l2_normalize(np.random.default_rng(3).normal(size=(3, 3)))
    p, q, weights = categorical.recording_tensors(rec, 3, fs.DEVICE)
    cost = torch.as_tensor(continuous.continuous_cost_fn(mu, rec), dtype=torch.float64)
    coeffs = transport.Coeffs(config.SETTING_T.a, config.SETTING_T.beta, config.SETTING_T.lam, config.SETTING_T.eps)
    init = transport.cold_start_log(p, q)
    huge_patience = 10 ** 6
    single = transport.solve_final(init, cost, weights, p, q, coeffs, 60, grad_tol=-1.0, rel_tol=-1.0,
                                   patience=huge_patience, backtrack_max=30)
    step1 = transport.solve_final(init, cost, weights, p, q, coeffs, 30, grad_tol=-1.0, rel_tol=-1.0,
                                  patience=huge_patience, backtrack_max=30)
    step2 = transport.solve_final(step1.log_t, cost, weights, p, q, coeffs, 30, grad_tol=-1.0, rel_tol=-1.0,
                                  patience=huge_patience, backtrack_max=30)
    assert torch.allclose(single.log_t, step2.log_t, atol=1e-12)
    assert single.status == step2.status == "capped"  # tolerances never satisfiable: full budget both ways


def test_converged_recording_is_frozen_and_frac_changed_zero_when_unchanged():
    rec = _toy_rec("r0", 6, 2, seed=1)
    mu = np.array([[1., 0.], [0., 1.]])
    metas = {"r0": _toy_meta("r0", rec.lengths)}
    gts = {"r0": np.ones(int(rec.lengths.sum()), dtype=np.int64)}
    rows, _log_t, status, states_by_rung, hit_deadline = fs.ladder_solve(
        mu, [rec], metas, gts, 60, [500, 1000], None)
    assert hit_deadline is False
    assert status["r0"] == "converged"
    assert np.array_equal(states_by_rung[500]["r0"], states_by_rung[1000]["r0"])
    assert rows[1]["frac_changed"] == 0.0
    assert rows[1]["n_converged"] == 1


def test_deadline_yields_partial_with_completed_rungs_saved():
    calls = {"n": 0}

    def fake_check_deadline(_deadline):
        calls["n"] += 1
        if calls["n"] > 1:
            raise fs.DeadlineExceeded()

    original = fs.check_deadline
    fs.check_deadline = fake_check_deadline
    try:
        rec = _toy_rec("r0", 4, 2, seed=5)
        mu = np.eye(2)
        metas = {"r0": _toy_meta("r0", rec.lengths)}
        gts = {"r0": np.zeros(int(rec.lengths.sum()), dtype=np.int64)}
        rows, _log_t, _status, states_by_rung, hit_deadline = fs.ladder_solve(
            mu, [rec], metas, gts, 60, [500, 1000], None)
        assert hit_deadline is True
        assert len(rows) == 1
        assert 500 in states_by_rung and 1000 not in states_by_rung
    finally:
        fs.check_deadline = original


# --- 4. Scheduler --------------------------------------------------------------------

def test_job_enumeration_counts_and_determinism():
    jobs1, jobs2 = fs.enumerate_jobs(), fs.enumerate_jobs()
    assert len(jobs1) == 98
    ids1, ids2 = [j.job_id for j in jobs1], [j.job_id for j in jobs2]
    assert ids1 == ids2  # deterministic order and naming
    assert len(set(ids1)) == 98  # no collisions
    by_priority = {}
    for j in jobs1:
        by_priority[j.priority] = by_priority.get(j.priority, 0) + 1
    assert by_priority == {"P0": 2, "P1": 16, "P2": 32, "P3": 4, "P4": 4, "P5": 40}


def test_resume_skips_complete_and_reruns_partial():
    jobs = fs.enumerate_jobs()
    status = {"L_hugadb_raw_s111_pca_disc_saved": {"status": "complete"},
             "L_hugadb_raw_s111_pca_oracle": {"status": "partial"}}
    todo_ids = {j.job_id for j in jobs if j.job_id not in ("check_baselines", "pilot")
               and status.get(j.job_id, {}).get("status") != "complete"}
    assert "L_hugadb_raw_s111_pca_disc_saved" not in todo_ids
    assert "L_hugadb_raw_s111_pca_oracle" in todo_ids


def test_smoke_jobs_cover_every_job_kind_once():
    kinds = {j.kind for j in fs.enumerate_smoke_jobs()}
    assert kinds == {"check_baselines", "pilot", "ladder", "fit_ladder", "probe"}


# --- 5. Probe ----------------------------------------------------------------------------

def test_probe_train_test_folds_are_subject_disjoint():
    fold_of = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    subjects = np.array(["s0", "s0", "s1", "s1", "s2", "s2", "s3", "s3"])
    for f in range(4):
        assert set(subjects[fold_of != f]).isdisjoint(set(subjects[fold_of == f]))


def test_probe_subsample_is_deterministic():
    rng1 = np.random.default_rng(fs.PROBE_SEED)
    rng2 = np.random.default_rng(fs.PROBE_SEED)
    assert np.array_equal(rng1.choice(1000, size=50, replace=False), rng2.choice(1000, size=50, replace=False))


def test_frame_weighted_metrics_hand_computed():
    y_true, y_pred = np.array([1, 1, 2, 2]), np.array([1, 2, 2, 2])
    w = np.array([1., 1., 1., 1.])
    acc, f1 = fs.frame_weighted_metrics(y_true, y_pred, w, np.array([1, 2]))
    assert abs(acc - 0.75) < 1e-9
    expected_f1 = float(np.mean([2 * 1.0 * 0.5 / 1.5, 2 * (2 / 3) * 1.0 / ((2 / 3) + 1.0)]))
    assert abs(f1 - expected_f1) < 1e-9


# --- 6. Label separation -----------------------------------------------------------------

def test_label_separation_in_fit_and_representation_code():
    for fn in (fs.run_disc_fit, fs.build_representation, fs.pool_windows):
        src = inspect.getsource(fn)
        for token in ("gts", "window_labels", "core_dataset.gts", ".gt["):
            assert token not in src, f"{fn.__name__} unexpectedly references {token!r}"


# --- 7. Identity -----------------------------------------------------------------------

def test_startup_checks_aborts_on_a_nonexistent_stage_b_run():
    try:
        fs.startup_checks("_nonexistent_stage_b_run_for_tests", fs.STAGE_A_RUN)
        raise AssertionError("must abort for a nonexistent stage_b_run")
    except SystemExit:
        pass


TESTS = [
    test_pool_windows_hand_checked_with_edge_truncation,
    test_pool_windows_h0_equals_identity,
    test_pool_windows_terminal_partial_window_is_weighted_by_its_real_length,
    test_pool_windows_no_averaging_across_recordings,
    test_build_oracle_prototypes_hand_computed_1indexed_and_present_only,
    test_identity_scoring_maps_state_index_to_label_id,
    test_ladder_cumulative_rungs_equal_single_solve_with_no_early_stop,
    test_converged_recording_is_frozen_and_frac_changed_zero_when_unchanged,
    test_deadline_yields_partial_with_completed_rungs_saved,
    test_job_enumeration_counts_and_determinism,
    test_resume_skips_complete_and_reruns_partial,
    test_smoke_jobs_cover_every_job_kind_once,
    test_probe_train_test_folds_are_subject_disjoint,
    test_probe_subsample_is_deterministic,
    test_frame_weighted_metrics_hand_computed,
    test_label_separation_in_fit_and_representation_code,
    test_startup_checks_aborts_on_a_nonexistent_stage_b_run,
]


def run_all() -> None:
    torch.set_num_threads(8)
    passed, failed = 0, []
    for test in TESTS:
        try:
            test()
            print(f"PASS  {test.__name__}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {test.__name__}: {exc}")
            failed.append(test.__name__)
    print(f"\n{passed}/{len(TESTS)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()
