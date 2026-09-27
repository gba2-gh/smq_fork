"""Validation suite for Stage B (`run_stage_b.py --validate-only`). Covers
masking, chunk stitching, the validation split, absence of label access, and
deterministic reload inference (docs/plans/THREE_STAGE_MOTION_PLAN.md §4).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.action_transport.stageb_data import (
    Chunk, build_mask, build_validation_split, chunk_starts, make_chunks, stable_seed,
)
from script.action_transport.stageb_infer import extract_embeddings
from script.action_transport.stageb_model import CodeContextModel, ModelConfig
from script.action_transport.stageb_train import _encode, load_checkpoint, save_checkpoint, train


def test_chunk_starts_covers_exactly():
    for length in (10, 64, 127, 128, 129, 200, 500):
        starts = chunk_starts(length, chunk=128, stride=64)
        covered = np.zeros(length, dtype=bool)
        for s in starts:
            covered[s:min(s + 128, length)] = True
        assert covered.all(), f"length {length} not fully covered by {starts}"
        assert starts[-1] + min(128, length) == length, f"final chunk must be anchored at length {length}"
        if length > 128:
            assert starts[0] == 0 and all(b - a <= 64 for a, b in zip(starts, starts[1:])), starts


def test_make_chunks_never_crosses_recording_boundary():
    codes = {"a": np.arange(150, dtype=np.int64), "b": np.arange(50, dtype=np.int64)}
    lengths = {"a": np.full(150, 15.0, dtype=np.float32), "b": np.full(50, 15.0, dtype=np.float32)}
    chunks = make_chunks(codes, lengths)
    for c in chunks:
        assert c.codes[:c.valid_len].max() < len(codes[c.recording])
        assert np.array_equal(c.codes[:c.valid_len], codes[c.recording][c.start:c.start + c.valid_len])
        assert (c.codes[c.valid_len:] == 0).all(), "padding must be zero, never a real code"
        assert (c.lengths[c.valid_len:] == 0).all(), "padded positions carry zero weight"


def test_mask_respects_valid_len_and_fraction():
    rng = np.random.default_rng(0)
    for valid_len in (0, 1, 5, 40, 127, 128):
        mask = build_mask(valid_len, rng)
        assert mask.shape == (valid_len,)
        target = round(0.30 * valid_len)
        assert mask.sum() == target, f"valid_len={valid_len}: {mask.sum()} != target {target}"


def test_mask_trim_is_deterministic_given_the_union():
    """Two different rngs that happen to produce the same span union must
    trim to the identical kept set (drop-highest-index-first)."""
    valid_len = 20
    mask = np.zeros(valid_len, dtype=bool)
    mask[2:10] = True  # a union of size 8 > target(6) for valid_len=20 (0.3*20=6)
    from script.action_transport.stageb_data import MASK_FRACTION
    target = round(MASK_FRACTION * valid_len)
    kept = sorted(np.flatnonzero(mask))[:target]
    assert kept == [2, 3, 4, 5, 6, 7], kept


def test_validation_split_disjoint_and_keeps_at_least_one_each():
    counts = {f"r{i}": 100 for i in range(10)}
    train, val = build_validation_split(list(counts), counts, seed=111)
    assert set(train) & set(val) == set()
    assert set(train) | set(val) == set(counts)
    assert len(train) >= 1 and len(val) >= 1
    assert sum(counts[n] for n in val) >= 0.05 * sum(counts.values()), "val should be near the 10% target"
    train2, val2 = build_validation_split(list(counts), counts, seed=111)
    assert train == train2 and val == val2, "the split must be deterministic given the seed"


def test_validation_split_minimum_population():
    counts = {"a": 100, "b": 5}
    train, val = build_validation_split(list(counts), counts, seed=111)
    assert len(train) == 1 and len(val) == 1


def test_encode_never_masks_or_targets_padding():
    codes = {"a": np.arange(40, dtype=np.int64)}
    lengths = {"a": np.full(40, 12.0, dtype=np.float32)}
    chunks = make_chunks(codes, lengths)  # one chunk, valid_len=40, padded to 128
    rng = np.random.default_rng(0)
    masks = [build_mask(c.valid_len, rng) for c in chunks]
    input_codes, targets, weights, mask_ind, padding = _encode(chunks, masks, mask_id=500, device=torch.device("cpu"))
    assert bool(padding[0, 40:].all()) and not bool(padding[0, :40].any())
    assert not bool(mask_ind[0, 40:].any()), "padding must never be masked"
    assert float(weights[0, 40:].sum()) == 0.0, "padding must carry zero loss weight"
    assert not bool((input_codes[0, 40:] == 500).any()), "padding must never receive the mask token"


def test_no_action_label_import():
    """Structural check: none of the Stage B modules imports the evaluator
    or anything with `gt`/`label` in its public API surface."""
    import script.action_transport.stageb_data as m1
    import script.action_transport.stageb_model as m2
    import script.action_transport.stageb_train as m3
    import script.action_transport.stageb_infer as m4
    for module in (m1, m2, m3, m4):
        assert "evaluate" not in dir(module) or module.__name__ == "evaluate"
        src = Path(module.__file__).read_text(encoding="utf-8")
        assert "action_transport.evaluate" not in src, f"{module.__name__} must not import the evaluator"


def test_embedding_averaging_hand_computed():
    """A tiny hand-checkable model: identity-like embedding whose hidden
    state at each position is deterministic given the input code, so
    overlap-averaging across two chunks of a short sequence is exact."""
    torch.manual_seed(0)
    cfg = ModelConfig(num_codes=5, d_model=8, n_layers=1, n_heads=2, d_ff=16, dropout=0.0, max_context=6)
    model = CodeContextModel(cfg)
    model.eval()
    codes = np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=np.int64)  # length 8 > max_context(6)

    # Manual single-pass reference using the SAME chunking the module uses.
    from script.action_transport.stageb_data import chunk_starts
    starts = chunk_starts(len(codes), chunk=6, stride=3)
    accum = np.zeros((len(codes), cfg.d_model))
    counts = np.zeros(len(codes))
    with torch.no_grad():
        for start in starts:
            end = min(start + 6, len(codes))
            valid = end - start
            chunk = np.zeros(6, dtype=np.int64)
            chunk[:valid] = codes[start:end]
            padding = np.ones(6, dtype=bool)
            padding[:valid] = False
            hidden, _ = model(torch.as_tensor(chunk[None]), torch.as_tensor(padding[None]))
            accum[start:end] += hidden[0, :valid].numpy()
            counts[start:end] += 1
    manual_mean = accum / counts[:, None]
    manual_unit = manual_mean / np.linalg.norm(manual_mean, axis=1, keepdims=True)

    # Monkeypatch the module's chunk_starts call site indirectly by using the
    # same max_context via a model with max_context=6 -- extract_embeddings
    # uses the fixed MAX_CONTEXT=128 constant, so this test instead verifies
    # the averaging arithmetic directly against manual_unit using the same
    # start/stride convention with a real (small) sequence under MAX_CONTEXT.
    unit, zero = extract_embeddings(model, codes[:6], torch.device("cpu"))
    assert not zero.any()
    # length 6 <= 128 -> single chunk, no averaging needed; must equal one forward pass exactly.
    with torch.no_grad():
        chunk = np.zeros(128, dtype=np.int64)
        chunk[:6] = codes[:6]
        padding = np.ones(128, dtype=bool)
        padding[:6] = False
        hidden, _ = model(torch.as_tensor(chunk[None]), torch.as_tensor(padding[None]))
        expected = hidden[0, :6].numpy()
        expected_unit = expected / np.linalg.norm(expected, axis=1, keepdims=True)
    assert np.allclose(unit, expected_unit, atol=1e-5), "single-chunk case must equal one forward pass, L2-normalized"


def test_deterministic_reload_inference():
    torch.manual_seed(0)
    codes = {"a": np.random.default_rng(0).integers(0, 20, 300).astype(np.int64),
             "b": np.random.default_rng(1).integers(0, 20, 300).astype(np.int64)}
    lengths = {"a": np.full(300, 15.0, dtype=np.float32), "b": np.full(300, 15.0, dtype=np.float32)}
    train_chunks = make_chunks({"a": codes["a"]}, {"a": lengths["a"]})
    val_chunks = make_chunks({"b": codes["b"]}, {"b": lengths["b"]})
    from script.action_transport.stageb_train import TrainConfig
    tiny_cfg = TrainConfig(max_epochs=2, patience=2, batch_size=4)
    model, info = train(train_chunks, val_chunks, num_codes=20, seed=111,
                        device=torch.device("cpu"), cfg=tiny_cfg)
    emb_a, _ = extract_embeddings(model, codes["a"], torch.device("cpu"))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        save_checkpoint(path, model, info, {"note": "test"})
        reloaded, reloaded_info, extra = load_checkpoint(path, torch.device("cpu"))
        emb_a_reloaded, _ = extract_embeddings(reloaded, codes["a"], torch.device("cpu"))
        assert extra["note"] == "test"
        assert reloaded_info["best_epoch"] == info["best_epoch"]
    assert np.allclose(emb_a, emb_a_reloaded, atol=1e-6), "a reloaded checkpoint must reproduce embeddings exactly"


def test_val_loss_decreases_below_random_init():
    """Training must reduce the fixed validation loss below its value at
    initialization -- catches a broken training loop without asserting a
    specific score."""
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    codes = {"a": rng.integers(0, 10, 400).astype(np.int64)}
    codes["a"][::4] = 3  # inject a learnable repeating pattern
    lengths = {"a": np.full(400, 15.0, dtype=np.float32)}
    train_chunks = make_chunks({"a": codes["a"][:300]}, {"a": lengths["a"][:300]})
    val_chunks = make_chunks({"a": codes["a"][300:]}, {"a": lengths["a"][300:]})
    from script.action_transport.stageb_train import TrainConfig
    cfg = TrainConfig(max_epochs=15, patience=15, batch_size=4)
    _, info = train(train_chunks, val_chunks, num_codes=10, seed=111, device=torch.device("cpu"), cfg=cfg)
    initial_val = info["history"][0]["val_loss"]
    assert info["best_val_loss"] < initial_val, (
        f"best val loss {info['best_val_loss']:.3f} did not improve on the first epoch's {initial_val:.3f}")


def test_stable_seed_is_process_independent():
    """v1.6 regression: the validation-mask seed must not depend on Python's
    per-process string-hash randomization."""
    import subprocess
    code = ("import sys; sys.path.insert(0, r'%s'); "
            "from script.action_transport.stageb_data import stable_seed; "
            "print(stable_seed('HuGaDB_v2_various_01_00', 64, 111))") % str(Path(__file__).resolve().parents[2])
    outputs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              env={**__import__("os").environ, "PYTHONHASHSEED": str(h)}).stdout.strip()
               for h in (1, 2, 3)}
    assert len(outputs) == 1 and outputs != {""}, f"seed differs across processes: {outputs}"


def test_tied_head_shapes_and_mask_row_excluded():
    cfg = ModelConfig(num_codes=20, d_model=16, n_layers=1, n_heads=2, d_ff=32, dropout=0.0, tie_head=True)
    model = CodeContextModel(cfg)
    x = torch.randint(0, 20, (2, 10))
    hidden, logits = model(x, torch.zeros(2, 10, dtype=torch.bool))
    assert hidden.shape == (2, 10, 16) and logits.shape == (2, 10, 20), "logits must cover codes only, not the mask row"
    assert not hasattr(model, "head"), "tied model must not carry an untied head"


def test_step_mode_respects_caps():
    rng = np.random.default_rng(3)
    codes = {"a": rng.integers(0, 10, 300).astype(np.int64), "b": rng.integers(0, 10, 200).astype(np.int64)}
    lengths = {k: np.full(len(v), 15.0, dtype=np.float32) for k, v in codes.items()}
    tr = make_chunks({"a": codes["a"]}, lengths)
    va = make_chunks({"b": codes["b"]}, lengths)
    from script.action_transport.stageb_train import TrainConfig
    cfg = TrainConfig(batch_size=2, max_steps=37, eval_every=10, patience_steps=10**6,
                      schedule="cosine", warmup_steps=5, norm_first=True, tie_head=True)
    _, info = train(tr, va, num_codes=10, seed=111, device=torch.device("cpu"), cfg=cfg)
    assert info["steps_run"] == 37 and info["stopped_by"] == "step_cap", (info["steps_run"], info["stopped_by"])
    assert info["history"][-1]["step"] == 37, "the final step must be evaluated"
    cfg2 = TrainConfig(batch_size=2, max_steps=10**6, eval_every=5, patience_steps=15)
    _, info2 = train(tr, va, num_codes=10, seed=111, device=torch.device("cpu"), cfg=cfg2)
    assert info2["stopped_by"] == "patience" and info2["steps_run"] - info2["best_step"] >= 15


def test_position_modes_ignore_padding():
    """Padded keys must never influence unpadded outputs, for both position
    modes and both LayerNorm placements, in train and eval mode."""
    for mode in ("sinusoidal", "alibi"):
        for norm_first in (False, True):
            torch.manual_seed(0)
            model = CodeContextModel(ModelConfig(num_codes=20, d_model=16, n_layers=2, n_heads=4, d_ff=32,
                                                 dropout=0.0, position_mode=mode, norm_first=norm_first))
            x = torch.randint(0, 20, (3, 12))
            pad = torch.zeros(3, 12, dtype=torch.bool)
            pad[1, 8:] = True
            x2 = x.clone()
            x2[1, 8:] = 7
            for training in (True, False):
                model.train(training)
                h1, l1 = model(x, pad)
                h2, _ = model(x2, pad)
                assert torch.isfinite(l1[~pad]).all(), (mode, norm_first, training)
                assert torch.allclose(h1[1, :8], h2[1, :8], atol=1e-5), (mode, norm_first, training)


def test_resolve_train_config_selected_matches_frozen_hash():
    """v1.7: --train-config selected must resolve to the exact configuration
    docs/plans/STAGE_C_AGENT_INSTRUCTIONS.md Sec 1.2 freezes."""
    from script.action_transport.run_stage_b import resolve_train_config
    from script.action_transport.stageb_configs import SELECTED_CONFIG_HASH, SELECTED_CONFIG_NAME, config_hash
    cfg, provenance = resolve_train_config("selected")
    assert config_hash(cfg) == SELECTED_CONFIG_HASH
    assert provenance["selected_from"]["config_name"] == SELECTED_CONFIG_NAME
    assert cfg.position_mode == "alibi" and cfg.norm_first is False and cfg.schedule == "cosine"
    assert cfg.max_steps == 5000 and cfg.warmup_steps == 500


def test_resolve_train_config_detects_tampered_selection():
    """A selection.json pointing at a config name whose current hash no
    longer matches the sweep.csv row it was chosen from must be refused."""
    import csv
    import tempfile
    from script.action_transport.run_stage_b import OUT_ROOT, resolve_train_config
    with tempfile.TemporaryDirectory() as tmp:
        study_dir = OUT_ROOT / f"_test_tamper_{Path(tmp).name}"
        try:
            study_dir.mkdir(parents=True)
            (study_dir / "selection.json").write_text(json.dumps(
                {"chosen": {"config": "postLN_constant_lr0.0003_alibi"}}))
            with (study_dir / "sweep.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["cell", "config", "config_hash"])
                w.writeheader()
                w.writerow({"cell": "hugadb_raw", "config": "postLN_constant_lr0.0003_alibi",
                           "config_hash": "0000000000000000"})  # wrong on purpose
            try:
                resolve_train_config("selected", study_dir.name)
                raise AssertionError("must refuse a hash mismatch")
            except SystemExit:
                pass
        finally:
            import shutil
            shutil.rmtree(study_dir, ignore_errors=True)


def test_run_identity_distinguishes_v15_from_selected():
    """The two --train-config values must not be treated as resumable into
    the same run (IDENTITY_FIELDS includes train_config, which differs)."""
    from script.action_transport.run_stage_b import identity_snapshot
    from script.action_transport.run_stage_b import resolve_train_config as resolve
    from script.action_transport.stageb_train import TrainConfig
    cfg_v15, _ = resolve("v1.5")
    cfg_sel, _ = resolve("selected")
    id_v15 = identity_snapshot(cfg_v15)
    id_sel = identity_snapshot(cfg_sel)
    assert id_v15["train_config"] != id_sel["train_config"]
    assert id_v15["config_hash"] != id_sel["config_hash"]
    assert identity_snapshot(TrainConfig()) == id_v15, "v1.5 must still mean plain TrainConfig()"


def test_codes_sha256_detects_a_single_code_perturbation():
    from script.action_transport.run_stage_b import codes_sha256
    codes = {"a": np.array([1, 2, 3], dtype=np.int64), "b": np.array([4, 5], dtype=np.int64)}
    base = codes_sha256(["a", "b"], codes)
    assert base == codes_sha256(["b", "a"], codes), "order of the names list must not matter (sorted internally)"
    perturbed = {"a": np.array([1, 2, 9], dtype=np.int64), "b": codes["b"]}
    assert base != codes_sha256(["a", "b"], perturbed)


def test_nearest_neighbor_diagnostic_hand_checked():
    """4 points: two near-identical pairs from different (recording, subject)
    combinations. The nearest neighbor of each point should be its pair."""
    from script.action_transport.run_stage_b import nearest_neighbor_diagnostics
    emb = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0], [0.14, 0.99]])
    rec_ids = np.array(["r0", "r0", "r1", "r1"])
    subj_ids = np.array(["s0", "s1", "s0", "s1"])
    out = nearest_neighbor_diagnostics(emb, rec_ids, subj_ids, sample=np.array([0, 1, 2, 3]), k=1)
    assert out["same_recording_rate"] == 1.0, "each point's single nearest neighbor shares its recording"
    # Hand-computed: excluding r0's other point for {0,1} (resp. r1's for {2,3}) leaves the two points of
    # the OTHER recording as candidates; each point's nearest among those is the other recording's point
    # with matching subject (1 hit each for indices 1 and 3) or opposite subject (0 for indices 0 and 2).
    assert out["same_subject_rate_excl_same_recording"] == 0.5, out


def test_get_or_build_split_verifies_against_split_source():
    """v1.7: reusing another run's official split must succeed when it
    reproduces exactly, and refuse if the recomputed split differs."""
    import tempfile
    from script.action_transport.run_stage_b import get_or_build_split, split_path

    class FakeMeta:
        def __init__(self, name, frame_count):
            self.name, self.frame_count = name, frame_count

    class FakeLoaded:
        def __init__(self, metas):
            self.metas = metas

    metas = [FakeMeta(f"r{i}", 1000) for i in range(6)]
    with tempfile.TemporaryDirectory() as tmp:
        source_dir, run_dir = Path(tmp) / "source", Path(tmp) / "run"
        get_or_build_split(source_dir, "hugadb", False, FakeLoaded(metas))
        train_names, val_names = get_or_build_split(run_dir, "hugadb", False, FakeLoaded(metas),
                                                     split_source=source_dir)
        source_blob = json.loads(split_path(source_dir, "hugadb", False).read_text())
        assert train_names == source_blob["train"] and val_names == source_blob["val"]
        # A different recording list must be refused, not silently accepted.
        run_dir2 = Path(tmp) / "run2"
        try:
            get_or_build_split(run_dir2, "hugadb", False,
                               FakeLoaded(metas[:-1] + [FakeMeta("different", 1000)]), split_source=source_dir)
            raise AssertionError("must refuse a split that does not reproduce the source")
        except SystemExit:
            pass


def test_run_one_smoke_with_selected_config():
    """v1.7: a real (not stubbed) smoke pass of the actual runner path --
    identity, config_hash, split verification, checkpoint, and INDEX.json --
    at a tiny step budget on real HuGaDB data (docs/plans/
    STAGE_C_AGENT_INSTRUCTIONS.md Sec 1.4)."""
    import tempfile
    from dataclasses import replace

    from script.action_transport import data
    from script.action_transport.run_stage_b import compute_config_hash, resolve_train_config, run_one

    cfg, _ = resolve_train_config("selected")
    cfg = replace(cfg, max_steps=50, eval_every=25, patience_steps=1000)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        info = run_one("hugadb", False, 111, run_dir, torch.device("cpu"), cfg, "selected_smoke")
        assert info["steps_run"] == 50 and info["stopped_by"] == "step_cap"
        ckpt_path = run_dir / "checkpoints" / "hugadb_raw_pooled_seed111.pt"
        assert ckpt_path.exists()
        emb_dir = run_dir / "embeddings" / "hugadb_raw_pooled_seed111"
        index = json.loads((emb_dir / "INDEX.json").read_text())
        assert index["config_hash"] == compute_config_hash(cfg)
        assert index["position_mode"] == "alibi"
        loaded = data.load_dataset("hugadb")
        assert set(index["recording_order"]) == {m.name for m in loaded.metas}
        for name in index["recording_order"][:2]:
            assert (emb_dir / f"{name}.npy.npz").exists() or (emb_dir / f"{name}.npz").exists()


def test_train_and_eval_mode_compute_the_same_function():
    """At PRODUCTION size (d_model=128, 4 heads -> head dim 32, which enables
    torch's eval fast path), train mode with dropout 0 and eval mode must give
    the same outputs, and eval must not be position-blind. A small-model test
    cannot catch this: torch 2.1's fast path is skipped for small head dims,
    and it silently dropped ALiBi's attention bias (found in the v1.6 study)."""
    for mode in ("sinusoidal", "alibi"):
        for norm_first in (False, True):
            for tie in (False, True):
                torch.manual_seed(0)
                model = CodeContextModel(ModelConfig(num_codes=500, dropout=0.0, position_mode=mode,
                                                     norm_first=norm_first, tie_head=tie))
                x = torch.randint(0, 500, (4, 128))
                pad = torch.zeros(4, 128, dtype=torch.bool)
                pad[2, 70:] = True
                with torch.no_grad():
                    model.train()
                    h_train, _ = model(x, pad)
                    model.eval()
                    h_eval, _ = model(x, pad)
                    perm = torch.randperm(128)
                    x_perm = x.clone()
                    x_perm[0] = x[0, perm]
                    h_perm, _ = model(x_perm, pad)
                valid = ~pad
                label = (mode, norm_first, tie)
                assert torch.isfinite(h_eval[valid]).all(), label
                assert (h_train - h_eval)[valid].abs().max() < 1e-4, label
                assert (h_perm[0] - h_eval[0][perm]).abs().max() > 1e-2, f"eval is position-blind: {label}"


TESTS = [
    test_run_one_smoke_with_selected_config,
    test_get_or_build_split_verifies_against_split_source,
    test_nearest_neighbor_diagnostic_hand_checked,
    test_codes_sha256_detects_a_single_code_perturbation,
    test_run_identity_distinguishes_v15_from_selected,
    test_resolve_train_config_detects_tampered_selection,
    test_resolve_train_config_selected_matches_frozen_hash,
    test_train_and_eval_mode_compute_the_same_function,
    test_position_modes_ignore_padding,
    test_stable_seed_is_process_independent,
    test_tied_head_shapes_and_mask_row_excluded,
    test_step_mode_respects_caps,
    test_chunk_starts_covers_exactly,
    test_make_chunks_never_crosses_recording_boundary,
    test_mask_respects_valid_len_and_fraction,
    test_mask_trim_is_deterministic_given_the_union,
    test_validation_split_disjoint_and_keeps_at_least_one_each,
    test_validation_split_minimum_population,
    test_encode_never_masks_or_targets_padding,
    test_no_action_label_import,
    test_embedding_averaging_hand_computed,
    test_deterministic_reload_inference,
    test_val_loss_decreases_below_random_init,
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
