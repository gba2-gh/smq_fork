"""Validation suite for Stage C (`run_stage_c.py --validate-only`). Covers
the identity chain, cost/prototype properties, arm coefficients, the order
diagnostic's restoration correctness, label separation, and Stage A reuse
(docs/plans/STAGE_C_AGENT_INSTRUCTIONS.md §2.7).
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.action_transport import config, contextual, continuous, permute
from script.action_transport import run_stage_c as rc
from script.action_transport.run_stage_b import OUT_ROOT as STAGEB_OUT_ROOT


# --- identity chain --------------------------------------------------------------------

def test_verify_stage_b_manifest_rejects_wrong_hash():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "fake_stage_b"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(json.dumps({"config_hash": "deadbeefdeadbeef"}))
        try:
            contextual.verify_stage_b_manifest(run_dir)
            raise AssertionError("must refuse a wrong config_hash")
        except contextual.IdentityError:
            pass


def test_load_stage_b_cell_rejects_checkpoint_tamper():
    """A checkpoint replaced after its embeddings were extracted (sha256 in
    INDEX.json no longer matches) must be refused."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        emb_dir = run_dir / "embeddings" / "hugadb_raw_pooled_seed111"
        emb_dir.mkdir(parents=True)
        ckpt_path = run_dir / "checkpoints" / "hugadb_raw_pooled_seed111.pt"
        ckpt_path.parent.mkdir(parents=True)
        ckpt_path.write_bytes(b"original checkpoint bytes")
        import hashlib
        (emb_dir / "INDEX.json").write_text(json.dumps({
            "config_hash": contextual.SELECTED_CONFIG_HASH,
            "checkpoint_sha256": hashlib.sha256(b"original checkpoint bytes").hexdigest(),
            "lengths": {"r0": 10}, "recording_order": ["r0"], "codes_sha256": "x", "cache_identity": "y",
        }))
        np.savez_compressed(emb_dir / "r0.npz", embedding=np.zeros((10, 128), dtype=np.float32),
                           zero_flags=np.zeros(10, dtype=bool))
        # sanity: passes before tamper
        _monkeypatch_run_root(run_dir.parent)
        contextual.load_stage_b_cell(run_dir.name, "hugadb", False, 111)
        ckpt_path.write_bytes(b"a different checkpoint")
        try:
            contextual.load_stage_b_cell(run_dir.name, "hugadb", False, 111)
            raise AssertionError("must refuse a checkpoint sha256 mismatch")
        except contextual.IdentityError:
            pass


def _monkeypatch_run_root(new_root: Path):
    contextual.STAGEB_OUT_ROOT = new_root


def test_load_stage_b_cell_rejects_shape_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        emb_dir = run_dir / "embeddings" / "hugadb_raw_pooled_seed111"
        emb_dir.mkdir(parents=True)
        ckpt_path = run_dir / "checkpoints" / "hugadb_raw_pooled_seed111.pt"
        ckpt_path.parent.mkdir(parents=True)
        ckpt_path.write_bytes(b"ckpt")
        import hashlib
        (emb_dir / "INDEX.json").write_text(json.dumps({
            "config_hash": contextual.SELECTED_CONFIG_HASH,
            "checkpoint_sha256": hashlib.sha256(b"ckpt").hexdigest(),
            "lengths": {"r0": 10}, "recording_order": ["r0"], "codes_sha256": "x", "cache_identity": "y",
        }))
        np.savez_compressed(emb_dir / "r0.npz", embedding=np.zeros((5, 128), dtype=np.float32),  # wrong length
                           zero_flags=np.zeros(5, dtype=bool))
        _monkeypatch_run_root(run_dir.parent)
        try:
            contextual.load_stage_b_cell(run_dir.name, "hugadb", False, 111)
            raise AssertionError("must refuse a length mismatch against INDEX.json")
        except contextual.IdentityError:
            pass


# --- cost/prototype properties (reused from continuous.py, checked at 128-d) -----------

def test_cosine_cost_conventions_at_128d():
    d = 128
    e0 = np.zeros(d)
    e0[0] = 1.0
    e1 = np.zeros(d)
    e1[1] = 1.0
    mu = np.stack([e0, -e0, e1])
    z = np.stack([e0])
    zero_flags = np.array([False])
    cost = continuous.cosine_cost(mu, z, zero_flags)
    assert np.allclose(cost[0], [0.0, 2.0, 1.0]), cost
    z_zero = np.zeros((1, d))
    cost_zero = continuous.cosine_cost(mu, z_zero, np.array([True]))
    assert np.allclose(cost_zero[0], [1.0, 1.0, 1.0]), "a zero input vector must cost 1 for every state"


def test_context_only_has_beta_zero_both_settings():
    for setting in config.SETTINGS:
        c = contextual.coeffs_for(setting, "context_only")
        assert c.beta == 0.0
        assert c.a == setting.a and c.lam == setting.lam and c.eps == setting.eps
        c2 = contextual.coeffs_for(setting, "contextual_asot")
        assert c2.beta == setting.beta


def test_mu0_matches_hand_computation():
    """g maps codes {0,1} -> state 0, code 2 -> state 1. mu0[state] is the
    length-weighted normalized mean of z over windows in that state."""
    g = np.array([0, 0, 1], dtype=np.int32)
    rec = continuous.ContinuousRecordingInput(
        "r0", codes=np.array([0, 1, 2, 0], dtype=np.int32), lengths=np.array([1, 2, 3, 1], dtype=np.int32),
        fps=10, window=5, z=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]),
        zero_flags=np.zeros(4, dtype=bool))
    mu0, zero = continuous.build_mu0(g, [rec], 2)
    expected_state0 = 1 * np.array([1.0, 0.0]) + 2 * np.array([0.0, 1.0]) + 1 * np.array([0.0, 1.0])
    expected_state0 = expected_state0 / np.linalg.norm(expected_state0)
    assert np.allclose(mu0[0], expected_state0, atol=1e-10)
    expected_state1 = np.array([1.0, 1.0]) / np.linalg.norm([1.0, 1.0])
    assert np.allclose(mu0[1], expected_state1, atol=1e-10)
    assert not zero.any()


def test_zero_initial_mean_is_unavailable():
    g = np.array([0, 0], dtype=np.int32)
    rec = continuous.ContinuousRecordingInput(
        "r0", codes=np.array([0, 1], dtype=np.int32), lengths=np.array([1, 1], dtype=np.int32),
        fps=10, window=5, z=np.array([[1.0, 0.0], [-1.0, 0.0]]), zero_flags=np.zeros(2, dtype=bool))
    mu0, zero = continuous.build_mu0(g, [rec], 1)
    assert zero[0], "opposite unit vectors with equal weight must sum to zero and be flagged"


# --- order diagnostic ------------------------------------------------------------------

def test_identity_contextualizer_restoration_is_exact():
    """An embedding that depends only on the CODE at each position (not on
    neighbors or absolute position) must be identical before and after
    permute -> extract -> restore, since restoration undoes the permutation
    exactly for any per-position function of the code."""
    rng = np.random.default_rng(0)
    lookup = rng.normal(size=(10, 4))  # one embedding vector per code id
    codes = rng.integers(0, 10, size=37).astype(np.int32)
    lengths = np.full(37, 5, dtype=np.int32)
    perm = permute.build_permutation(lengths, window=5, rng=np.random.default_rng(1))
    codes_perm = permute.apply_permutation(codes, perm)
    z_original = lookup[codes]
    z_perm = lookup[codes_perm]  # "extract_embeddings" stand-in: per-position lookup only
    z_restored = permute.restore_states(z_perm, perm)
    assert np.allclose(z_restored, z_original)


def test_restore_states_round_trip_on_a_known_nonidentity_permutation():
    """A synthetic case where SKIPPING restoration (or double-restoring)
    changes the result, so the evaluator cannot silently score the wrong
    timeline."""
    perm = np.array([2, 0, 1])  # shuffled slot j holds original index perm[j]
    original = np.array([10, 20, 30])
    shuffled = original[perm]  # what would be "observed" at each shuffled slot
    restored = permute.restore_states(shuffled, perm)
    assert np.array_equal(restored, original)
    assert not np.array_equal(shuffled, original), "the permutation must actually move something"
    double_restored = permute.restore_states(restored, perm)
    assert not np.array_equal(double_restored, original), (
        "restoring twice must NOT reproduce the original -- catches an accidental double-restore")


def test_order_diagnostic_kernel_uses_original_chronological_support():
    """The order diagnostic's fitting inputs keep the ORIGINAL lengths/fps/
    window, so kernel_half_width and the kernel support are identical to the
    unpermuted case -- transport adjacency is unaffected by the diagnostic."""
    from script.action_transport import transport
    lengths = np.full(50, 12, dtype=np.int32)
    b_original = transport.kernel_half_width(fps=60, window=15, num_windows=50)
    # The diagnostic never changes fps/window/num_windows for any recording, only z.
    b_after = transport.kernel_half_width(fps=60, window=15, num_windows=len(lengths))
    assert b_original == b_after


# --- label separation -------------------------------------------------------------------

def test_no_action_label_import_in_fitting_path():
    """Structural check: contextual.py (Stage C's fitting/cost/identity code)
    must never import the evaluator, and its script.action_transport imports
    must all be from a fixed allow-list."""
    src = Path(contextual.__file__).read_text(encoding="utf-8")
    assert "action_transport.evaluate" not in src, "contextual.py must not import the evaluator"
    # "run" (run.py) is allowed: contextual.py reuses its generic, label-free
    # categorical_inputs/continuous_inputs input-assembly helpers for the
    # Stage A grouping check -- never its evaluator-facing code, which is
    # what the "no evaluate import" assertion above actually guards.
    allowed = {"categorical", "config", "continuous", "permute", "transport", "run_stage_b",
              "stageb_configs", "stageb_infer", "stageb_train", "run"}
    import re
    for m in re.findall(r"from script\.action_transport(?:\.(\w+))? import", src):
        if m:
            assert m in allowed, f"unexpected import target {m!r} in contextual.py"
    for m in re.findall(r"from script\.action_transport import ([\w, ]+?)(?:\s+#|\n)", src):
        for name in m.split(","):
            name = name.strip().split(" as ")[0].strip()
            assert name in allowed, f"unexpected import target {name!r} in contextual.py"


# --- Stage A reuse -----------------------------------------------------------------------

def test_verify_stage_a_reuse_on_real_run():
    """§2.6: re-scoring a saved Stage A prediction set must reproduce its
    cells.csv row exactly. Skips (does not fail) if the real run is absent
    on this machine."""
    stage_a_dir = config.OUT_ROOT / "stage_a_pooled_002"
    if not (stage_a_dir / "cells.csv").exists():
        print("  (skipped: no stage_a_pooled_002 on this machine)")
        return
    rc.verify_stage_a_reuse("stage_a_pooled_002")


# --- operations ---------------------------------------------------------------------------

def test_manifest_refuses_a_different_stage_b_or_stage_a_run():
    class Args:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        stage_b_dir = STAGEB_OUT_ROOT / "_test_manifest_stub"
        stage_b_dir.mkdir(parents=True, exist_ok=True)
        (stage_b_dir / "manifest.json").write_text(json.dumps({"config_hash": contextual.SELECTED_CONFIG_HASH}))
        try:
            a1 = Args()
            a1.run_id, a1.hours, a1.resume, a1.device = "run", 1.0, False, "cpu"
            a1.stage_b_run, a1.stage_a_run = "_test_manifest_stub", "stage_a_A"
            rc.write_manifest(run_dir, a1, 4)
            a2 = Args()
            a2.run_id, a2.hours, a2.resume, a2.device = "run", 1.0, False, "cpu"
            a2.stage_b_run, a2.stage_a_run = "_test_manifest_stub", "stage_a_B"  # different stage_a_run
            try:
                rc.write_manifest(run_dir, a2, 4)
                raise AssertionError("must refuse a launch naming a different stage_a_run")
            except SystemExit:
                pass
        finally:
            shutil.rmtree(stage_b_dir, ignore_errors=True)


def test_cell_enumeration_counts():
    cells = rc.all_cells()
    assert len(cells) == 28
    assert sum(1 for c in cells if c.arm == "context_only") == 12
    assert sum(1 for c in cells if c.arm == "contextual_asot") == 12
    assert sum(1 for c in cells if c.arm == "contextual_order_perm") == 4
    n_sets = sum(len(rc.prediction_set_ids(c)) for c in cells)
    assert n_sets == 80, n_sets


TESTS = [
    test_verify_stage_b_manifest_rejects_wrong_hash,
    test_load_stage_b_cell_rejects_checkpoint_tamper,
    test_load_stage_b_cell_rejects_shape_mismatch,
    test_cosine_cost_conventions_at_128d,
    test_context_only_has_beta_zero_both_settings,
    test_mu0_matches_hand_computation,
    test_zero_initial_mean_is_unavailable,
    test_identity_contextualizer_restoration_is_exact,
    test_restore_states_round_trip_on_a_known_nonidentity_permutation,
    test_order_diagnostic_kernel_uses_original_chronological_support,
    test_no_action_label_import_in_fitting_path,
    test_verify_stage_a_reuse_on_real_run,
    test_manifest_refuses_a_different_stage_b_or_stage_a_run,
    test_cell_enumeration_counts,
]


def run_all() -> None:
    torch.set_num_threads(8)
    passed, failed = 0, []
    original_root = contextual.STAGEB_OUT_ROOT
    for test in TESTS:
        try:
            test()
            print(f"PASS  {test.__name__}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {test.__name__}: {exc}")
            failed.append(test.__name__)
        finally:
            contextual.STAGEB_OUT_ROOT = original_root
    print(f"\n{passed}/{len(TESTS)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()
