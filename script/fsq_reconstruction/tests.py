"""Validation suite for the FSQ reconstruction pilot
(docs/plans/FSQ_RECONSTRUCTION_PILOT_PLAN.md Revision 3 §10). Covers FSQ
math parity/round-trips, RNG isolation and matched initialization, the
log(512) categorical cost, extraction determinism, and a real identity
check against data.py -- all on synthetic or tiny real fixtures so this
suite runs in seconds, not hours.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.fsq_reconstruction import evaluate as fq_evaluate
from script.fsq_reconstruction import experiment as fq_experiment
from script.fsq_reconstruction.quantizers import (
    FSQ, FSQSMQModel, PatchFSQAdapter, build_fsq_arm, build_shared_backbone, build_vq_arm, isolated_rng,
    reseed_all,
)

TINY_SMQ_KWARGS = dict(in_channels=6, filters=8, num_layers=1, latent_dim=4, num_actions=512, num_joints=6,
                      num_person=1, patch_size=5, kmeans=False, dead_code_threshold=1, decay=0.5)


# --- FSQ math -------------------------------------------------------------------------

def test_fsq_round_trip_all_512_codes():
    fsq = FSQ((8, 8, 8))
    all_idx = torch.arange(fsq.codebook_size, dtype=torch.int64)
    codes = fsq.indexes_to_codes(all_idx)
    recovered = fsq.codes_to_indexes(codes.to(torch.float32))
    assert torch.equal(recovered, all_idx), "every index must round-trip through indexes_to_codes/codes_to_indexes"
    # every recovered code must be a valid grid point in [-1,1]
    assert float(codes.max()) <= 1.0 + 1e-9 and float(codes.min()) >= -1.0 - 1e-9


def test_fsq_codebook_size_matches_prod_levels():
    fsq = FSQ((8, 8, 8))
    assert fsq.codebook_size == 512
    fsq2 = FSQ((2, 4, 4))
    assert fsq2.codebook_size == 32


def test_fsq_bound_saturates_at_extreme_inputs():
    """Very large |z| must map to the boundary grid point on each
    coordinate. For an even level count (8) the grid is the asymmetric set
    {-1, ..., (L-2)/L} (digits 0..L-1 normalized as (digit-L/2)/(L/2)), so a
    large positive z saturates at (8-2)/8=0.75, not at +1; a large negative
    z saturates at exactly -1. This is the FSQ formula's own asymmetry for
    even L, not a bug in this pilot's math."""
    fsq = FSQ((8, 8, 8))
    z = torch.tensor([[50.0, -50.0, 0.0]])
    q = fsq.quantize(z)
    assert abs(q[0, 0].item() - 0.75) < 1e-6  # saturates at the top grid point for L=8
    assert abs(q[0, 1].item() - (-1.0)) < 1e-6  # saturates at the bottom grid point
    idx = fsq.codes_to_indexes(q)
    assert 0 <= int(idx[0]) < fsq.codebook_size


def test_fsq_straight_through_gradient_is_finite_and_nonzero():
    fsq = FSQ((8, 8, 8))
    z = torch.randn(20, 3, requires_grad=True)
    q = fsq.quantize(z)
    loss = q.sum()
    loss.backward()
    assert torch.isfinite(z.grad).all()
    assert float(z.grad.abs().sum()) > 0.0  # STE must pass a nonzero gradient through the round


def test_patch_fsq_adapter_shapes_and_id_range():
    adapter = PatchFSQAdapter(embedding_dim=4, window=5, levels=(8, 8, 8))
    x = torch.randn(2, 13, 4)  # 13 frames, not a multiple of window=5 -> a padded terminal patch
    mask = torch.ones_like(x)
    quantize, indices, loss, distances = adapter(x, mask)
    assert quantize.shape == x.shape
    assert indices.shape == (2, 13)
    assert float(loss) == 0.0  # FSQ has no commitment term
    assert distances is None  # §5: "make that explicit rather than fabricating distances"
    assert indices.min() >= 0 and indices.max() < adapter.num_embeddings
    for b in range(2):
        p0 = int(indices[b, 0])
        assert (indices[b, :5] == p0).all()  # first patch is one code for all 5 frames


def test_patch_fsq_adapter_gradients_update_adapters():
    adapter = PatchFSQAdapter(embedding_dim=4, window=5, levels=(8, 8, 8))
    x = torch.randn(2, 10, 4, requires_grad=False)
    mask = torch.ones_like(x)
    before = adapter.in_proj.weight.detach().clone()
    quantize, _, _, _ = adapter(x, mask)
    quantize.sum().backward()
    assert adapter.in_proj.weight.grad is not None
    assert torch.isfinite(adapter.in_proj.weight.grad).all()
    assert float(adapter.in_proj.weight.grad.abs().sum()) > 0.0
    assert torch.equal(before, adapter.in_proj.weight.detach())  # backward alone does not change weights


# --- RNG isolation and matched initialization (§4) -------------------------------------

def test_isolated_rng_restores_state_exactly():
    torch.manual_seed(1234)
    before = torch.rand(5)
    torch.manual_seed(1234)
    with isolated_rng(999):
        torch.rand(100)  # burn RNG state inside the isolated block
    after = torch.rand(5)
    assert torch.equal(before, after)


def test_reseed_all_makes_two_independent_draws_identical():
    reseed_all(42)
    a = torch.rand(10)
    reseed_all(42)
    b = torch.rand(10)
    assert torch.equal(a, b)


def test_matched_arms_share_encoder_decoder_but_not_quantizer():
    backbone = build_shared_backbone(TINY_SMQ_KWARGS, seed=111)
    vq_model = build_vq_arm(TINY_SMQ_KWARGS, backbone, init_seed=222)
    fsq_model = build_fsq_arm(TINY_SMQ_KWARGS, backbone, init_seed=333, levels=(2, 4, 4))
    for (k1, v1), (k2, v2) in zip(vq_model.encoder.state_dict().items(), fsq_model.encoder.state_dict().items()):
        assert k1 == k2
        assert torch.equal(v1, v2), f"encoder param {k1} differs between the two matched arms"
    for (k1, v1), (k2, v2) in zip(vq_model.decoder.state_dict().items(), fsq_model.decoder.state_dict().items()):
        assert torch.equal(v1, v2), f"decoder param {k1} differs between the two matched arms"
    # the quantizers must NOT share parameters (different modules entirely)
    assert not hasattr(fsq_model.vq, "_embedding")
    assert vq_model.vq._embedding.shape[0] == TINY_SMQ_KWARGS["num_actions"]


def test_two_backbones_from_the_same_seed_are_identical():
    b1 = build_shared_backbone(TINY_SMQ_KWARGS, seed=111)
    b2 = build_shared_backbone(TINY_SMQ_KWARGS, seed=111)
    assert b1.init_hash == b2.init_hash


def test_vq_eager_init_sets_valid_codebook_before_any_forward():
    backbone = build_shared_backbone(TINY_SMQ_KWARGS, seed=111)
    vq_model = build_vq_arm(TINY_SMQ_KWARGS, backbone, init_seed=222)
    assert bool(vq_model.vq.initted.item()) is True
    assert float(vq_model.vq.cluster_size.min()) == 1.0  # eager init sets every count to 1, not 0


# --- fixed batch order (§4.3) -----------------------------------------------------------

def test_epoch_orders_are_deterministic_given_the_same_seed():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        features_path = Path(tmp)
        for i in range(6):
            np.save(features_path / f"rec{i}.npy", np.zeros((6, 20, 6, 1), dtype=np.float32))
        orders_a = fq_experiment.precompute_epoch_orders(features_path, num_epochs=3, seed=111)
        orders_b = fq_experiment.precompute_epoch_orders(features_path, num_epochs=3, seed=111)
        assert orders_a == orders_b
        orders_c = fq_experiment.precompute_epoch_orders(features_path, num_epochs=3, seed=222)
        assert orders_a != orders_c


# --- log(512) categorical cost (§7) -----------------------------------------------------

def test_emission_cost_uses_log_512_not_log_500():
    theta = np.full((3, 512), 1.0 / 512)
    codes = np.array([0, 1, 2], dtype=np.int32)
    cost = fq_evaluate.emission_cost_k(theta, codes, math.log(512))
    expected = -np.log(1.0 / 512) / math.log(512)
    assert np.allclose(cost, expected)
    cost_wrong_denominator = fq_evaluate.emission_cost_k(theta, codes, math.log(500))
    assert not np.allclose(cost, cost_wrong_denominator)


def test_prior_penalty_uses_the_given_log_k():
    theta = np.full((2, 512), 1.0 / 512)
    p1 = fq_evaluate.prior_penalty_k(theta, a=0.7, eta=60.0, num_codes=512, log_k=math.log(512))
    p2 = fq_evaluate.prior_penalty_k(theta, a=0.7, eta=60.0, num_codes=512, log_k=math.log(500))
    assert p1 != p2


# --- extraction determinism and padding (§4, §6) ---------------------------------------

def _make_toy_smq_model(levels=(2, 4, 4)):
    backbone = build_shared_backbone(TINY_SMQ_KWARGS, seed=111)
    return build_fsq_arm(TINY_SMQ_KWARGS, backbone, init_seed=222, levels=levels)


def test_extract_recording_is_deterministic_and_covers_all_frames():
    import tempfile
    model = _make_toy_smq_model()
    model.eval()
    with tempfile.TemporaryDirectory() as tmp:
        feature_path = Path(tmp) / "rec0.npy"
        arr = np.random.default_rng(0).normal(size=(6, 23, 6, 1)).astype(np.float32)  # 23: not a multiple of 5
        np.save(feature_path, arr)
        out_a = fq_experiment.extract_recording(model, feature_path, window=5, num_features=6, num_joints=6,
                                                num_person=1, device=torch.device("cpu"))
        out_b = fq_experiment.extract_recording(model, feature_path, window=5, num_features=6, num_joints=6,
                                                num_person=1, device=torch.device("cpu"))
        assert np.array_equal(out_a["ids"], out_b["ids"])
        assert np.array_equal(out_a["flat"], out_b["flat"])
        assert out_a["num_patches"] == 5  # ceil(23/5)
        assert int(out_a["lengths"].sum()) == 23  # every real frame accounted for
        assert out_a["lengths"][-1] == 3  # terminal partial patch: 23 - 4*5


def test_per_code_pca_centers_only_uses_occupied_codes():
    import script.action_transport.categorical as categorical
    rec = categorical.RecordingInput("r0", np.array([0, 0, 1], dtype=np.int32),
                                     np.array([10, 10, 10], dtype=np.int32), 60, 5)
    pca_by_name = {"r0": np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)}
    centers, counts = fq_evaluate.per_code_pca_centers([rec], pca_by_name, num_codes=4)
    assert np.allclose(centers[0], [1.0, 0.0])
    assert np.allclose(centers[1], [0.0, 1.0])
    assert np.allclose(centers[2], [0.0, 0.0])  # unoccupied code: zero, not NaN
    assert counts[0] == 20 and counts[1] == 10 and counts[2] == 0


# --- a real identity check against data.py (skips, does not fail, if absent) -----------

def test_check_identity_against_released_meta_on_real_hugadb():
    """§10 check 5 in spirit: saved/recomputed frame counts must agree with
    data.py's loaded dataset. Skips (does not fail) if HuGaDB's released
    checkpoint/provenance gate is unavailable on this machine."""
    features_path = fq_experiment.ROOT / "data" / "hugadb" / "features"
    if not features_path.exists():
        print("  (skipped: no data/hugadb/features on this machine)")
        return
    import os
    names = sorted(os.listdir(features_path))[:2]
    extracted = {}
    for name in names:
        arr = np.load(features_path / name)
        extracted[name] = dict(lengths=np.array([arr.shape[1]]))  # a single "patch" spanning the whole recording
    try:
        result = fq_experiment.check_identity_against_released_meta(
            fq_experiment.PilotConfig(dataset="hugadb"), extracted)
    except SystemExit as exc:
        print(f"  (skipped: {exc})")
        return
    assert set(result["gts_by_name"]) == set(names)


TESTS = [
    test_fsq_round_trip_all_512_codes,
    test_fsq_codebook_size_matches_prod_levels,
    test_fsq_bound_saturates_at_extreme_inputs,
    test_fsq_straight_through_gradient_is_finite_and_nonzero,
    test_patch_fsq_adapter_shapes_and_id_range,
    test_patch_fsq_adapter_gradients_update_adapters,
    test_isolated_rng_restores_state_exactly,
    test_reseed_all_makes_two_independent_draws_identical,
    test_matched_arms_share_encoder_decoder_but_not_quantizer,
    test_two_backbones_from_the_same_seed_are_identical,
    test_vq_eager_init_sets_valid_codebook_before_any_forward,
    test_epoch_orders_are_deterministic_given_the_same_seed,
    test_emission_cost_uses_log_512_not_log_500,
    test_prior_penalty_uses_the_given_log_k,
    test_extract_recording_is_deterministic_and_covers_all_frames,
    test_per_code_pca_centers_only_uses_occupied_codes,
    test_check_identity_against_released_meta_on_real_hugadb,
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
            print(f"FAIL  {test.__name__}: {exc!r}")
            failed.append(test.__name__)
    print(f"\n{passed}/{len(TESTS)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()
