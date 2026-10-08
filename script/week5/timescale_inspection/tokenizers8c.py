"""Phase 8c preparation: heading-canonical window tokenizers (T6, T6s) and the unweighted T5gh (T5gh-std).

`tokenizers.py` and `tokenizers8.py` are not edited (their hashes are part of the Phase 5-8a identities); the new keys are
registered into `TOK.SPECS` at import time, in-process only. Everything not named here is reused from them.

  T6   `t6_heading_canonical`        T0's window layout (flattened frames x coordinates) on positions rotated about the
                                     vertical axis so the body faces +axis 0 at the window's middle frame. One block.
  T6s  `t6s_heading_canonical_stats` The same canonical coordinates (non-root joints), summarised per window by mean, SD and
                                     mean |frame-to-frame change| (T2's temporal summary). One block.
  T5gh-std `t5_both_std`             Phase 8a's T5gh features (distances + vertical + heading rate) with plain global
                                     standardization: no block weighting.

Gate-only keys (never run as cells): `t6_norot` (rotation disabled; must reproduce T0), `t5_dist_only_std` (distance block
through this module's std path; must reproduce T2).

Rotation of window w: every frame of the window is rotated about axis 2 by -psi_w, with psi_w the facing direction at the
window's middle frame (or the nearest valid frame inside the window; a window with no valid frame is left unrotated and
counted). Absolute heading is removed; gravity (axis 2) and chirality (no reflection) are kept; turning inside the window
is kept. No feature uses a neighbouring window.

Design: docs/plans/week5/TIMESCALE_INSPECTION/PHASE8C_SYMMETRY_TEST.md.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from script.fsq_reconstruction.experiment import write_json_atomic
from script.quantized_input_comparison import artifacts
from script.quantized_input_comparison import fit as fitmod
from script.quantized_input_comparison.families import kmeans_lookup, window_starts
from script.week5.raw_kmeans_cross_dataset import datasets as D
from script.week5.timescale_inspection import tokenizers as TOK
from script.week5.timescale_inspection import tokenizers8 as T8

CANON_KEYS = ("t6_heading_canonical", "t6s_heading_canonical_stats", "t5_both_std")
ALL_DATASETS = ("babel1", "babel2", "babel3", "lara")


def register_8c_tokenizers() -> None:
    T8.register_t5_tokenizers()
    specs = (
        TOK.TokenizerSpec("t6_heading_canonical", "T6", "canon", 64, False, ALL_DATASETS, span=2, selectable=False),
        TOK.TokenizerSpec("t6s_heading_canonical_stats", "T6s", "canon_stats", 64, False, ALL_DATASETS, span=2,
                          selectable=False),
        TOK.TokenizerSpec("t5_both_std", "T5gh-std", "relational+vertical+heading_std", 64, False, ALL_DATASETS, span=2,
                          selectable=False),
        # gate only
        TOK.TokenizerSpec("t6_norot", "T6-norot", "canon_norot", 64, False, ("babel2",), span=2, selectable=False),
        TOK.TokenizerSpec("t5_dist_only_std", "T5-dist-std", "relational_std", 64, False, ("babel2",), span=2,
                          selectable=False),
    )
    for s in specs:
        if s.key not in TOK.SPECS:
            TOK.SPECS[s.key] = s


register_8c_tokenizers()

SOURCE_BLOCKS = {"relational_std": ("dist",), "relational+vertical+heading_std": ("dist", "vertical", "heading")}


# -- heading canonicalization --------------------------------------------------------------------------------------

def window_headings(psi: np.ndarray, valid: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Per window: the heading used for canonicalization and whether the window is rotated. psi_w is psi at the window's
    middle frame, or at the valid frame nearest to it inside the window; a window with no valid frame gets psi_w = 0 and
    rotated = False."""
    T = len(psi)
    P = int(np.ceil(T / window))
    out, rotated = np.zeros(P), np.zeros(P, dtype=bool)
    for p in range(P):
        a, b = p * window, min(T, (p + 1) * window)
        mid = a + (b - a) // 2
        ok = np.flatnonzero(valid[a:b]) + a
        if len(ok):
            out[p] = psi[ok[np.argmin(np.abs(ok - mid))]]
            rotated[p] = True
    return out, rotated


def canonical_positions(pos: np.ndarray, dataset: str, window: int, rotate: bool = True) -> tuple[np.ndarray, dict]:
    """(T, V, 3) positions -> positions with every window rotated about axis 2 by -psi_w, plus per-window counters."""
    sk = T8.skeleton(dataset)
    psi, valid = T8.body_heading(pos, sk)
    psi_w, rotated = window_headings(psi, valid, window)
    stats = dict(windows=int(len(psi_w)), windows_unrotated=int((~rotated).sum()) if rotate else int(len(psi_w)))
    if not rotate:
        return T8.rotate_z(pos, 0.0), stats
    theta = -np.repeat(psi_w, window)[:pos.shape[0]]
    return T8.rotate_z(pos, theta), dict(stats, rotated_mask=rotated)


def canon_flat(pos: np.ndarray, window: int) -> dict:
    """T0's layout: frames (T, C*V) coordinate-major, as `raw_kmeans_cross_dataset.prepare.extract_raw`."""
    arr = np.ascontiguousarray(pos.transpose(2, 0, 1))[..., None]  # (3, T, V, 1)
    T = arr.shape[1]
    return TOK.patch_frames(arr.transpose(1, 0, 2, 3).reshape(T, -1), window)


def canon_stats_features(pos: np.ndarray, dataset: str, window: int) -> np.ndarray:
    """(P, 9(V-1)) mean / SD / mean |delta| of every non-root coordinate, joint-major, over each window's own frames."""
    sk = T8.skeleton(dataset)
    keep = [j for j in range(pos.shape[1]) if j != sk.root]
    return TOK.window_stats(pos[:, keep, :].reshape(pos.shape[0], -1).astype(np.float32), window)


def extract8c(spec: D.DatasetSpec, source: str, names: list[str]) -> tuple[dict[str, dict], dict]:
    if source in SOURCE_BLOCKS:  # the T5 feature path of phase 8a
        out, stats = T8.extract8(spec, {"relational_std": "relational"}.get(source, source.replace("_std", "")), names)
        return out, stats
    rotate = source in ("canon", "canon_stats")
    out, windows, unrotated = {}, 0, 0
    for name in names:
        pos = T8.load_positions(spec, name)
        cpos, st = canonical_positions(pos, spec.name, spec.window, rotate=rotate)
        windows += st["windows"]
        unrotated += st["windows_unrotated"] if rotate else 0
        if source in ("canon", "canon_norot"):
            out[name] = canon_flat(cpos, spec.window)
        elif source == "canon_stats":
            flat = canon_stats_features(cpos, spec.name, spec.window)
            T = pos.shape[0]
            P = int(np.ceil(T / spec.window))
            lengths = np.full(P, spec.window, dtype=np.int32)
            if P * spec.window - T:
                lengths[-1] = spec.window - (P * spec.window - T)
            out[name] = dict(flat=flat, lengths=lengths, num_patches=P)
        else:
            raise ValueError(source)
    return out, dict(windows=windows, windows_unrotated=unrotated,
                     windows_unrotated_share=(unrotated / windows) if windows and rotate else None)


def share_slices(source: str, dataset: str, width: int) -> dict[str, slice]:
    if source in SOURCE_BLOCKS:
        return T8.block_slices(SOURCE_BLOCKS[source], dataset)
    return {"all": slice(0, width)}


# -- cell preparation: `tokenizers8._build` with this module's extractor and an unweighted fit ----------------------

def prepare_cell(dataset: str, tokenizer: str, seed: int, run_dir: Path, window: int | None = None,
                 recording_cap: int | None = None) -> Path:
    key = TOK.make_key(dataset, tokenizer, seed, window, recording_cap)
    spec = TOK.dataset_spec(dataset, window)
    names = D.recording_names(spec, recording_cap)
    counts = D.frame_counts(spec, names)
    run_dir = Path(run_dir)
    if TOK.manifest_ok(run_dir, key, names, counts) and (run_dir / T8.BLOCK_FILE).exists():
        return run_dir
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    with threadpool_limits(limits=key.threads):
        _build(spec, key, run_dir, names, counts)
    return run_dir


def is_prepared(dataset: str, tokenizer: str, seed: int, run_dir: Path, window: int | None = None,
                recording_cap: int | None = None) -> bool:
    return TOK.is_prepared(dataset, tokenizer, seed, run_dir, window, recording_cap) and (Path(run_dir) / T8.BLOCK_FILE).exists()


def _build(spec: D.DatasetSpec, key: TOK.PrepKey, run_dir: Path, names: list[str], counts: dict[str, int]) -> None:
    tok = TOK.SPECS[key.tokenizer]
    family = tok.family
    raw, feat_stats = extract8c(spec, tok.source, names)
    width = int(next(iter(raw.values()))["flat"].shape[1])
    shares = share_slices(tok.source, spec.name, width)
    fit_slices = {"all": slice(0, width)}  # one block: no weighting, ever, in this phase
    frame_counts = {n: int(sum(raw[n]["lengths"])) for n in names}
    if frame_counts != counts:
        raise RuntimeError("windowed frame counts disagree with the feature files' own shapes")
    lengths = fitmod.window_lengths(raw)
    all_ids = fitmod.complete_window_ids(lengths, spec.window)
    sample_ids = fitmod.sample_window_ids(all_ids, key.sample_cap, key.seed)
    artifacts.save_sample_ids(run_dir, sample_ids)
    train_names, val_names = fitmod.shared_validation_split(frame_counts, key.split_seed)
    artifacts.save_split(run_dir, train_names, val_names, frame_counts)
    train_set = set(train_names)
    train_complete = [(n, p) for n, p in all_ids if n in train_set]

    pop, weights = T8.fit_population8(family, raw, sample_ids, key.pca_dim, key.num_codes, key.seed, fit_slices,
                                      whiten=key.whiten)
    assert not weights  # a single block is never reweighted
    artifacts.save_population(run_dir, pop)
    pca_family = {n: fitmod.project_pca(pop, rec["flat"]) for n, rec in raw.items()}
    records = {}
    for name, rec in raw.items():
        pca_feats, ids = fitmod.assign_kmeans(pop, rec["flat"])
        records[name] = dict(codes=ids, continuous=pca_feats, starts=window_starts(rec["num_patches"], spec.window),
                             lengths=rec["lengths"].astype(np.int32))
    lookup = kmeans_lookup(pop)
    stats = fitmod.conditioning_stats({n: r["continuous"] for n, r in records.items()}, sample_ids, train_complete)
    codes_by_name = {n: r["codes"] for n, r in records.items()}
    lengths_by_name = {n: r["lengths"] for n, r in records.items()}
    grouping, _, counts_w = fitmod.initial_grouping(codes_by_name, lengths_by_name, pca_family, key.num_codes,
                                                    spec.num_states, spec.fps, spec.window, key.seed)
    window_counts = np.zeros(key.num_codes, dtype=np.int64)
    for codes in codes_by_name.values():
        np.add.at(window_counts, codes, 1)
    artifacts.save_family(run_dir, family, records, lookup, stats["mean"], stats["rms"], grouping, counts_w, window_counts)
    pre = np.var(pop.scaler.transform(fitmod.gather(raw, sample_ids)), axis=0)  # after standardization, as PCA sees it
    write_json_atomic(run_dir / T8.BLOCK_FILE, dict(
        blocks=list(shares), widths={b: s.stop - s.start for b, s in shares.items()}, weights={},
        pre_pca_variance_share={b: float(pre[s].sum() / pre.sum()) for b, s in shares.items()},
        pca_variance_share=T8.pca_block_share(pop.pca, shares),
        pca_explained_variance_ratio=float(pop.pca.explained_variance_ratio_.sum()),
        kmeans_occupied=int(pop.n_occupied), **feat_stats))
    write_json_atomic(run_dir / "manifest.json", dict(
        schema=TOK.SCHEMA, prep_key=asdict(key), families=[family], recording_names=names,
        recording_frame_counts=frame_counts, num_windows={family: {n: int(len(r["codes"])) for n, r in records.items()}},
        feature_dims={family: int(lookup.shape[1])}, sample_ids_hash=artifacts.sample_ids_hash(sample_ids),
        n_train_recordings=len(train_names), n_val_recordings=len(val_names),
        kmeans_occupied=int(pop.n_occupied), n_fit_windows=int(pop.n_fit_windows)))


# -- gate G8c-2 / G8c-3 (pure functions; the runner records them) -----------------------------------------------------

def reduction_check(dataset: str, recording_cap: int | None = 1) -> dict:
    """G8c-2: with the rotation disabled, T6s's features equal `tokenizers.window_stats` of the root-centred non-root
    coordinates (exact), and T6's flat window equals the T0-layout flattening of the unrotated positions (exact)."""
    spec = TOK.dataset_spec(dataset)
    sk = T8.skeleton(dataset)
    name = D.recording_names(spec, recording_cap)[0]
    pos = T8.load_positions(spec, name)
    cpos, _ = canonical_positions(pos, dataset, spec.window, rotate=False)
    keep = [j for j in range(pos.shape[1]) if j != sk.root]
    # contiguous input: float32 summation order depends on memory layout (the loaded positions are a transposed view)
    direct = TOK.window_stats(np.ascontiguousarray(pos[:, keep, :].reshape(pos.shape[0], -1).astype(np.float32)), spec.window)
    checks = dict(stats_equal=bool(np.array_equal(canon_stats_features(cpos, dataset, spec.window), direct)),
                  flat_equal=bool(np.array_equal(canon_flat(cpos, spec.window)["flat"], canon_flat(pos, spec.window)["flat"])),
                  positions_unchanged=bool(np.array_equal(cpos, pos)))
    return dict(status="passed" if all(checks.values()) else "failed", checks=checks, recording=name)


def _features(pos: np.ndarray, dataset: str, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cpos, st = canonical_positions(pos, dataset, window, rotate=True)
    return canon_flat(cpos, window)["flat"], canon_stats_features(cpos, dataset, window), st["rotated_mask"]


def invariance_check(dataset: str, omega: float = 1.0) -> dict:
    """G8c-3 on the first recording: T6 and T6s are unchanged by a constant rotation about axis 2 (windows that are
    rotated); a turn growing at `omega` rad/s changes them (within-window turning is kept); a mirror image changes them
    (chirality is kept)."""
    spec = TOK.dataset_spec(dataset)
    name = D.recording_names(spec, None)[0]
    pos = T8.load_positions(spec, name)
    W = spec.window
    f0, s0, m0 = _features(pos, dataset, W)
    f1, s1, m1 = _features(T8.rotate_z(pos, 0.65), dataset, W)
    both = m0 & m1
    n = int(both.sum())
    fixed = max(float(np.abs(f0[both] - f1[both]).max()), float(np.abs(s0[both] - s1[both]).max())) if n else float("inf")
    t = np.arange(pos.shape[0]) / spec.fps
    f2, s2, m2 = _features(T8.rotate_z(pos, omega * t), dataset, W)
    turned = float(max(np.abs(f0[both & m2] - f2[both & m2]).max(), np.abs(s0[both & m2] - s2[both & m2]).max()))
    mir = pos.copy()
    mir[..., 0] = -mir[..., 0]
    f3, s3, m3 = _features(mir, dataset, W)
    mirrored = float(max(np.abs(f0[both & m3] - f3[both & m3]).max(), np.abs(s0[both & m3] - s3[both & m3]).max()))
    checks = dict(fixed_rotation_invariant=fixed <= 1e-2, turn_changes_features=turned > 1.0,
                  mirror_changes_features=mirrored > 1.0, windows_tested=n > 5)
    return dict(status="passed" if all(checks.values()) else "failed", checks=checks, recording=name,
                fixed_rotation_max_abs_diff=fixed, turn_max_abs_diff=turned, mirror_max_abs_diff=mirrored, windows_tested=n)
