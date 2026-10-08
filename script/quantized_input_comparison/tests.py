"""Preflight (`--preflight`, no full pipeline run) and post-smoke
(`tests.run_real_data_checks(run_dir)`, called by `prepare.py --smoke`)
verification for Task 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from script.quantized_input_comparison import artifacts, config as cfg, extract, families, fit as fitmod, masking


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# --- masking / receptive field ---------------------------------------------------------

def test_masking_disjoint_from_visible():
    starts = np.arange(10) * 15  # window=15, radius=14 -> adjacent windows' supports overlap
    lengths = np.full(10, 15, dtype=np.int32)
    frame_count = 150
    selected = np.zeros(10, dtype=bool)
    selected[3] = True  # mask window 3 only; every other window is visible
    eligible = masking.eligible_loss_positions(starts, lengths, 10, frame_count, selected, radius=14)
    # window 3 support = [3*15-14, 3*15+14+14] = [31,73]; window 2 support = [16,58] overlaps -> ineligible
    _assert(not eligible[3], "adjacent visible window must make an isolated masked target ineligible")


def test_masking_eligible_when_span_masked_together():
    # window=15, radius=14: a window's support spans 43 raw frames, so two
    # windows must be >=3 apart (45 frames of start-separation) before their
    # supports stop overlapping. Masking a wide enough contiguous span
    # (indices 1..7, visible={0,8,9}) leaves its interior (2..6) far enough
    # from both flanking visible windows to become eligible, while the span's
    # own edges (1 and 7) remain adjacent to a visible window and stay
    # ineligible.
    starts = np.arange(10) * 15
    lengths = np.full(10, 15, dtype=np.int32)
    frame_count = 150
    selected = np.zeros(10, dtype=bool)
    selected[1:8] = True
    eligible = masking.eligible_loss_positions(starts, lengths, 10, frame_count, selected, radius=14)
    # Windows must be >=3 apart in index (45 frames of start-separation) to
    # be disjoint at radius 14, so only indices 3..5 (>=3 away from both
    # visible window 0 and visible window 8) are eligible.
    _assert(all(eligible[3:6]), "interior of a wide masked span must be eligible once its neighbors are also masked")
    _assert(not eligible[1] and not eligible[2] and not eligible[6] and not eligible[7],
            "masked positions within 2 windows of a visible window stay ineligible")


def test_masking_zero_eligible_explicit():
    starts = np.arange(3) * 15
    lengths = np.full(3, 15, dtype=np.int32)
    selected = np.array([False, True, False])
    eligible = masking.eligible_loss_positions(starts, lengths, 3, 45, selected, radius=14)
    _assert(eligible.sum() == 0, "an isolated masked target flanked by visible neighbors must be zero-eligible")
    _assert(eligible.dtype == bool and len(eligible) == 3, "eligible mask must be an explicit boolean array")


def test_masking_recording_boundary_clip():
    # A window at the very start/end of a short recording: support clips at
    # 0 / frame_count-1, so a masked terminal window can still be eligible
    # relative to a visible window that is far enough away.
    starts = np.array([0, 15, 100])
    lengths = np.array([15, 15, 5], dtype=np.int32)  # last window terminal, short
    frame_count = 105
    selected = np.array([False, False, True])
    eligible = masking.eligible_loss_positions(starts, lengths, 3, frame_count, selected, radius=14)
    lo, hi = masking.window_support(starts, lengths, frame_count, radius=14)
    _assert(hi[2] == frame_count - 1, "support must clip at the recording boundary, not overrun it")
    _assert(bool(eligible[2]) == bool(hi[1] < lo[2]), "boundary-clipped support must still use exact disjointness")


def test_masking_padded_positions_never_eligible():
    starts = np.array([0, 15, 0])  # last is padding (start=0 reused, but valid_len excludes it)
    lengths = np.array([15, 15, 0], dtype=np.int32)
    selected = np.array([False, False, True])  # a padded index cannot be "selected" by build_mask, but guard anyway
    eligible = masking.eligible_loss_positions(starts, lengths, 2, 30, selected, radius=14)
    _assert(not eligible[2], "positions beyond valid_len must never be eligible")


def test_build_eligible_mask_reproducible():
    starts = np.arange(20) * 15
    lengths = np.full(20, 15, dtype=np.int32)
    rng1 = np.random.default_rng(111)
    rng2 = np.random.default_rng(111)
    sel1, elig1 = masking.build_eligible_mask(starts, lengths, 20, 300, rng1)
    sel2, elig2 = masking.build_eligible_mask(starts, lengths, 20, 300, rng2)
    _assert(np.array_equal(sel1, sel2) and np.array_equal(elig1, elig2), "same seed must reproduce the same mask")


# --- receptive field constant, verified against ms_tcn.py's actual architecture --------

def test_receptive_field_matches_architecture():
    from src.model.ms_tcn import SingleStageModel
    # num_layers=3 dilated layers per stage (dilations 1,2,4, kernel 3) x 2
    # stages -> radius = 2 * (1+2+4) = 14, field = 29. Read num_layers from
    # the actual trained config rather than hardcoding it twice.
    from script.fsq_reconstruction.encoder_sweep import load_source_config
    source_cfg = load_source_config(cfg.SOURCE_RUN)
    dilations = [2 ** i for i in range(source_cfg.num_layers)]
    radius_per_stage = sum(dilations)  # kernel=3 -> (kernel-1)/2 * dilation = dilation
    total_radius = 2 * radius_per_stage  # encoder = 2 stages
    _assert(total_radius == cfg.RECEPTIVE_RADIUS, f"expected radius {cfg.RECEPTIVE_RADIUS}, computed {total_radius}")
    _assert(2 * total_radius + 1 == cfg.RECEPTIVE_FIELD == 29, "receptive field must be 29 frames")
    _assert(cfg.WINDOW + 2 * total_radius == cfg.MAX_PATCH_SUPPORT == 43, "complete patch must depend on 43 frames")


# --- FSQ math sanity (pure, no checkpoint needed) --------------------------------------

def test_fsq_indexes_roundtrip():
    from script.fsq_reconstruction.quantizers import FSQ
    fsq = FSQ((8, 8, 8))
    idx = torch.arange(fsq.codebook_size)
    codes = fsq.indexes_to_codes(idx)
    back = fsq.codes_to_indexes(codes.to(torch.float32))
    _assert(torch.equal(back, idx), "FSQ index -> code -> index round trip must be exact")


def run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"[preflight] {fn.__name__} OK", flush=True)
    print(f"[preflight] {len(fns)} tests passed", flush=True)


# --- real-data / post-smoke checks (called by prepare.py --smoke) ----------------------

def run_real_data_checks(run_dir: Path) -> None:
    device = torch.device("cpu")
    source_cfg = extract.load_source_pilot_config()
    loader = artifacts.Loader(run_dir)
    names = loader.recording_names()
    _assert(len(names) > 0, "smoke run must extract at least one recording")

    # 1) models/EMA buffers unchanged by a forward pass under eval+no_grad
    vq_model, _ = extract.load_frozen_model("vq", source_cfg, device)
    before = {k: v.clone() for k, v in vq_model.state_dict().items()}
    d = source_cfg.smq_kwargs()
    from script.fsq_reconstruction.experiment import DATASET_DEFAULTS, extract_recording
    dd = DATASET_DEFAULTS[source_cfg.dataset]
    feat_path = extract.features_dir(source_cfg) / (names[0] + "")
    if not feat_path.exists():
        feat_path = extract.features_dir(source_cfg) / names[0]
    extract_recording(vq_model, feat_path, dd["patch_size"], dd["num_features"], dd["num_joints"],
                       dd["num_person"], device)
    after = vq_model.state_dict()
    for k, v in before.items():
        _assert(torch.equal(v, after[k]), f"vq {k!r} changed after an eval-mode extraction forward pass")

    # 2) native VQ lookup table equals the live frozen codebook, not just
    # internal self-consistency: index the actual model's embedding table
    # for a sample of code ids and diff against the saved lookup.
    lookup_vq = loader.lookup("vq_native")
    emb = vq_model.vq._embedding.detach().cpu().numpy().reshape(cfg.NUM_CODES, -1).astype(np.float32)
    rng = np.random.default_rng(cfg.SEED)
    sample_ids_check = rng.choice(cfg.NUM_CODES, size=min(32, cfg.NUM_CODES), replace=False)
    _assert(np.allclose(lookup_vq[sample_ids_check], emb[sample_ids_check], atol=1e-6),
            "vq_native lookup must equal the live frozen codebook's embedding table, not just itself")
    # identical ids actually occurring in the data -> identical looked-up vectors
    for name in names[:3]:
        rec = loader.recording("vq_native", name)
        vecs = lookup_vq[rec["codes"]]
        for cid in np.unique(rec["codes"])[:5]:
            idxs = np.flatnonzero(rec["codes"] == cid)
            if len(idxs) > 1:
                _assert(np.allclose(vecs[idxs[0]], vecs[idxs[1]]),
                        "identical native VQ code ids must yield identical quantized vectors")

    # 2b) native FSQ lookup table equals a fresh forward pass through the
    # live frozen adapter's out_proj on sampled grid indices.
    fsq_model, _ = extract.load_frozen_model("fsq", source_cfg, device)
    lookup_fsq = loader.lookup("fsq_native")
    adapter = fsq_model.vq
    idx_sample = torch.as_tensor(sample_ids_check, dtype=torch.long, device=adapter.out_proj.weight.device)
    zhat = adapter.fsq.indexes_to_codes(idx_sample).to(dtype=adapter.out_proj.weight.dtype)
    with torch.no_grad():
        vecs_fsq = adapter.out_proj(zhat).detach().cpu().numpy().astype(np.float32)
    _assert(np.allclose(lookup_fsq[sample_ids_check], vecs_fsq, atol=1e-5),
            "fsq_native lookup must equal a live out_proj forward pass on the same grid indices")

    # 3) K-means lookups equal the fitted centroid VALUES (not just shape):
    # diff the saved family lookup directly against the saved
    # preprocessing/<pop>_kmeans.npz cluster_centers_ for that population.
    for fam in ("vq_kmeans", "fsq_kmeans", "raw_kmeans"):
        lookup = loader.lookup(fam)
        _assert(lookup.shape == (cfg.NUM_CODES, cfg.PCA_DIM), f"{fam} lookup must be [{cfg.NUM_CODES},{cfg.PCA_DIM}]")
        pop_name = cfg.FAMILY_POPULATION[fam]
        kmeans_npz = np.load(run_dir / "preprocessing" / f"{pop_name}_kmeans.npz")
        _assert(np.array_equal(lookup, kmeans_npz["cluster_centers_"].astype(np.float32)),
                f"{fam} lookup must equal the fitted kmeans.cluster_centers_ values, not merely their shape")

    # 4) native dims == 1440
    _assert(loader.lookup("vq_native").shape[1] == cfg.NATIVE_DIM, "native VQ dim must be 1440")
    _assert(loader.lookup("fsq_native").shape[1] == cfg.NATIVE_DIM, "native FSQ dim must be 1440")

    # 5) window/frame alignment: starts + lengths must not exceed frame_count, terminal window shorter
    for name in names:
        rec = loader.recording("vq_native", name)
        fc = loader.frame_count(name)
        _assert(int(rec["starts"][-1] + cfg.WINDOW) >= fc >= int(rec["starts"][-1]),
                f"{name}: terminal window must cover the recording's real frame count")
        _assert(int(rec["lengths"].sum()) == fc, f"{name}: window lengths must sum to the recording's frame count")

    # 6) split disjoint and deterministic
    split = loader.split
    _assert(set(split["train"]).isdisjoint(split["val"]), "train/val split must be disjoint")
    train2, val2 = fitmod.shared_validation_split(split["frame_counts"], cfg.SEED)
    _assert(sorted(train2) == sorted(split["train"]) and sorted(val2) == sorted(split["val"]),
            "split must be deterministic under the same seed")

    # 7) cached artifacts reload identically
    loader2 = artifacts.Loader(run_dir)
    for fam in cfg.FAMILIES:
        _assert(np.array_equal(loader.lookup(fam), loader2.lookup(fam)), f"{fam} lookup must reload identically")
        _assert(np.array_equal(loader.grouping(fam), loader2.grouping(fam)), f"{fam} grouping must reload identically")

    # 8) preprocessing sampled only complete windows
    sample_ids = artifacts.load_sample_ids(run_dir)
    for name, p in sample_ids[:200]:
        rec = loader.recording("raw_kmeans", name)
        _assert(int(rec["lengths"][p]) == cfg.WINDOW, f"sampled window ({name},{p}) must be a complete window")

    # 9) rms is a single shared scalar (per spec), not a per-coordinate array
    for fam in cfg.FAMILIES:
        rms = loader.rms(fam)
        _assert(np.asarray(rms).shape == (), f"{fam} rms.npy must be a scalar, got shape {np.asarray(rms).shape}")
        _assert(float(rms) >= 1.0 - 1e-6 or float(rms) > 0, f"{fam} rms must be a positive scalar (or floored to 1.0)")

    # 10) occupancy exposes both frame-weighted and unweighted (window-count) arrays
    for fam in cfg.FAMILIES:
        occ = loader.occupancy(fam)
        _assert(occ["counts"].shape == (cfg.NUM_CODES,), f"{fam} occupancy counts must be [{cfg.NUM_CODES}]")
        _assert(occ["window_counts"].shape == (cfg.NUM_CODES,), f"{fam} occupancy window_counts must be [{cfg.NUM_CODES}]")
        _assert(occ["window_counts"].dtype == np.int64, f"{fam} window_counts must be integer")
        occupied_by_counts = int((occ["counts"] > 0).sum())
        occupied_by_windows = int((occ["window_counts"] > 0).sum())
        _assert(occupied_by_counts == occupied_by_windows == occ["n_occupied"],
                f"{fam}: frame-weighted and window-count occupancy must agree on which codes are occupied")

    # 11) prepare._load_population (the exact function `--resume` calls to
    # reconstruct preprocessing from saved arrays) uses REAL sklearn
    # StandardScaler/PCA/KMeans objects, not hand-rolled affine-map stand-ins
    # -- checked directly (type identity), not just numerically, since a
    # numeric diff against a separately re-extracted `flat` vector is
    # confounded by ordinary cross-run floating-point nondeterminism in the
    # encoder forward pass itself (different accumulation order per run),
    # which is unrelated to whether the RECONSTRUCTION is correct. A prior
    # version of `_load_population` used `types.SimpleNamespace` stand-ins
    # with a manual `(X - mean) @ components.T` map that computed in a
    # different floating-point precision than sklearn's in-place transform
    # and drifted from the fresh-fit path by ~1e-5; this would have failed
    # the type check below immediately.
    from sklearn.cluster import KMeans as _KMeans
    from sklearn.decomposition import PCA as _PCA
    from sklearn.preprocessing import StandardScaler as _StandardScaler

    from script.quantized_input_comparison import prepare as preparemod
    sample_ids_reloaded = artifacts.load_sample_ids(run_dir)
    for pop_name in cfg.POPULATIONS:
        reloaded_pop = preparemod._load_population(run_dir, pop_name, sample_ids_reloaded)
        _assert(isinstance(reloaded_pop.scaler, _StandardScaler),
                f"{pop_name}: _load_population's reconstructed scaler must be a real sklearn StandardScaler")
        _assert(isinstance(reloaded_pop.pca, _PCA),
                f"{pop_name}: _load_population's reconstructed pca must be a real sklearn PCA")
        _assert(isinstance(reloaded_pop.kmeans, _KMeans),
                f"{pop_name}: _load_population's reconstructed kmeans must be a real sklearn KMeans")

    # And, on the SAME extracted `flat` vector (no cross-run device/order
    # confound), the reconstructed population's transform must reproduce a
    # from-scratch sklearn transform built from the identical saved
    # parameters to near machine precision (not merely `< 1e-3`).
    from script.fsq_reconstruction.experiment import DATASET_DEFAULTS as _DD, extract_recording as _extract_recording

    def _sklearn_transform(pop_name: str, flat: np.ndarray) -> np.ndarray:
        s = np.load(run_dir / "preprocessing" / f"{pop_name}_scaler.npz")
        p = np.load(run_dir / "preprocessing" / f"{pop_name}_pca.npz")
        scaler = _StandardScaler()
        scaler.mean_, scaler.scale_, scaler.var_ = s["mean_"], s["scale_"], s["var_"]
        scaler.n_features_in_ = s["mean_"].shape[0]
        pca = _PCA(n_components=int(p["n_components_"]))
        pca.components_, pca.mean_ = p["components_"], p["mean_"]
        pca.n_components_ = int(p["n_components_"])
        return pca.transform(scaler.transform(flat))

    _dd = _DD[source_cfg.dataset]
    _feat_dir = extract.features_dir(source_cfg)
    _name0 = names[0]
    _vq_flat = _extract_recording(vq_model, _feat_dir / _name0, _dd["patch_size"], _dd["num_features"],
                                   _dd["num_joints"], _dd["num_person"], device)["flat"]
    _fsq_flat = _extract_recording(fsq_model, _feat_dir / _name0, _dd["patch_size"], _dd["num_features"],
                                    _dd["num_joints"], _dd["num_person"], device)["flat"]
    _raw_flat = extract.extract_raw_dataset(_feat_dir, _dd["patch_size"], names=[_name0])[_name0]["flat"]
    real_flat_by_pop = {"vq_encoder": _vq_flat.astype(np.float32), "fsq_encoder": _fsq_flat.astype(np.float32),
                         "raw": _raw_flat.astype(np.float32)}

    max_resume_diff = 0.0
    for pop_name, flat in real_flat_by_pop.items():
        reloaded_pop = preparemod._load_population(run_dir, pop_name, sample_ids_reloaded)
        resumed = fitmod.project_pca(reloaded_pop, flat)
        sk = _sklearn_transform(pop_name, flat)
        diff = float(np.max(np.abs(resumed - sk)))
        max_resume_diff = max(max_resume_diff, diff)
        _assert(diff < 1e-4, f"{pop_name}: resumed transform diverges from an independently-reconstructed "
                              f"sklearn transform (same saved params, same input) by {diff} -- expected near "
                              f"machine precision now that both use real sklearn objects")
    print(f"[smoke] resumed-vs-sklearn transform max abs diff (real data, {_name0}): "
          f"{max_resume_diff:.3e}", flush=True)

    # 12) receptive-field perturbation: perturbing the raw frames that
    # support an ELIGIBLE masked target must leave every VISIBLE window
    # whose support is disjoint from those frames byte-for-byte unchanged
    # (to float32 tolerance) after a fresh forward pass through the real
    # frozen encoder -- and must actually change the perturbed target's own
    # feature (positive control), confirming the receptive field really
    # does cross patch boundaries (contra an earlier, incorrect README claim).
    _test_receptive_field_perturbation_real(loader, vq_model, source_cfg, device, names)

    print(f"[smoke] real-data checks passed for {run_dir}", flush=True)


def _test_receptive_field_perturbation_real(loader, vq_model, source_cfg, device, names) -> None:
    from script.fsq_reconstruction.experiment import DATASET_DEFAULTS, extract_recording
    from script.quantized_input_comparison import masking

    d = DATASET_DEFAULTS[source_cfg.dataset]
    window = d["patch_size"]
    feat_dir = extract.features_dir(source_cfg)

    rng = np.random.default_rng(cfg.SEED)
    chosen = None
    for name in names:
        rec = loader.recording("vq_native", name)
        starts, lengths = rec["starts"], rec["lengths"]
        frame_count = loader.frame_count(name)
        valid_len = int((lengths > 0).sum())
        if valid_len < 5:
            continue
        # A single isolated masked window is never eligible at this
        # window/radius (its immediate neighbors' supports always overlap
        # it -- see test_masking_disjoint_from_visible); mask a width-7
        # contiguous span centered on each candidate instead (as in
        # test_masking_eligible_when_span_masked_together) so its interior
        # can clear the >=3-window-index gap needed for disjoint support.
        for c in range(4, valid_len - 4):
            trial = np.zeros(len(starts), dtype=bool)
            trial[c - 3:c + 4] = True
            elig = masking.eligible_loss_positions(starts, lengths, valid_len, frame_count, trial, radius=cfg.RECEPTIVE_RADIUS)
            if elig[c]:
                chosen = (name, c, starts, lengths, frame_count, valid_len)
                break
        if chosen:
            break
    _assert(chosen is not None, "could not find any eligible masked target across the smoke recordings to "
                                 "perturbation-test -- widen the smoke recording set")
    name, t, starts, lengths, frame_count, valid_len = chosen

    lo, hi = masking.window_support(starts, lengths, frame_count, radius=cfg.RECEPTIVE_RADIUS)
    lo_t, hi_t = int(lo[t]), int(hi[t])

    feat_path = feat_dir / name
    arr = np.load(feat_path)  # (C, T, V, M), raw dtype may be integer (e.g. HuGaDB's int64 sensor units)
    arr_pert = arr.astype(np.float64)
    pert_rng = np.random.default_rng(cfg.SEED + 1)
    arr_pert[:, lo_t:hi_t + 1, :, :] += pert_rng.normal(scale=10.0, size=arr_pert[:, lo_t:hi_t + 1, :, :].shape)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        pert_path = Path(td) / name
        np.save(pert_path, arr_pert)  # `name` already ends in .npy (HuGaDB feature file naming)
        dd = DATASET_DEFAULTS[source_cfg.dataset]
        baseline = extract_recording(vq_model, feat_path, window, dd["num_features"], dd["num_joints"],
                                      dd["num_person"], device)
        perturbed = extract_recording(vq_model, pert_path, window, dd["num_features"], dd["num_joints"],
                                       dd["num_person"], device)

    _assert(not np.allclose(baseline["flat"][t], perturbed["flat"][t], atol=1e-5),
            "positive control failed: perturbing a window's own support did not change its extracted feature "
            "-- the perturbation or the eligible-target selection is not exercising the encoder as expected")

    num_checked = 0
    for w in range(valid_len):
        if w == t:
            continue
        lo_w, hi_w = int(lo[w]), int(hi[w])
        disjoint = hi_w < lo_t or hi_t < lo_w
        if not disjoint:
            continue  # window w's support overlaps the perturbed frames; no invariance claim for it
        num_checked += 1
        _assert(np.allclose(baseline["flat"][w], perturbed["flat"][w], atol=1e-5, rtol=1e-5),
                f"{name}: window {w} (support [{lo_w},{hi_w}], disjoint from perturbed [{lo_t},{hi_t}]) "
                "must be unchanged by a perturbation entirely outside its receptive-field support")
    _assert(num_checked > 0, f"{name}: no disjoint-support visible window found to check invariance against "
                              "-- recording too short for this test")
    print(f"[smoke] receptive-field perturbation check: {name} target window {t} (support [{lo_t},{hi_t}]), "
          f"{num_checked} disjoint-support windows unchanged, target window itself changed as expected", flush=True)
