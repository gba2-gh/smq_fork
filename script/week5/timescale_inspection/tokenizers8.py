"""Phase 8a preparation: T2 relational tokens plus yaw-invariant gravity and heading-rate blocks.

`tokenizers.py` is not edited (its hash is part of every Phase 5-7 identity). The T5 tokenizer keys are registered
into `TOK.SPECS` at import time, in-process only, and their preparation is a copy of `tokenizers.prepare_cell` /
`_build` / `fit_population` with two changes: the per-window input comes from `extract8` (blocks), and appended
blocks are weighted after standardization (multiple factor analysis weighting, anchored on the distance block). With
only the distance block (`t5_dist_only`, used by gate G8a) the path reduces to T2 exactly.

Blocks (all from the window's own frames; none uses the absolute heading):
  dist      T2: per-frame inter-joint distances -> mean, SD, mean |delta| per window
  vertical  root-relative height (axis 2) of every distinct non-root joint, and trunk tilt from +axis 2 ->
            mean, SD, mean |delta| per window
  heading   signed yaw rate of the horizontal left->right body axis (hips + shoulders) -> mean w, mean |w|, SD w,
            from consecutive valid frames inside the window

Design: docs/plans/week5/TIMESCALE_INSPECTION/PHASE8A_GRAVITY_HEADING_TOKENS.md.
"""

from __future__ import annotations

import dataclasses
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from script.fsq_reconstruction.experiment import write_json_atomic
from script.quantized_input_comparison import artifacts
from script.quantized_input_comparison import fit as fitmod
from script.quantized_input_comparison.families import kmeans_lookup, window_starts
from script.week5.raw_kmeans_cross_dataset import datasets as D
from script.week5.timescale_inspection import tokenizers as TOK

VERTICAL_AXIS = 2
VALID_HORIZONTAL_SHARE = 0.3  # the body axis counts as defined if its horizontal part is >= 30% of its length
BLOCK_FILE = "t5_blocks.json"


@dataclass(frozen=True)
class Skeleton:
    root: int
    hips: tuple[int, int]  # (left, right)
    shoulders: tuple[int, int]  # (left, right)
    trunk: tuple[int, int]  # (lower, upper)
    drop: tuple[int, ...]  # duplicate points left out of the height block
    order: tuple  # joints (or left/right pairs) from top to bottom, for gate G8b
    head: int
    feet: tuple[int, int]


SKELETONS = {
    # NTU-25 layout (BABEL): spine base 0, spine-mid 1 (root), neck 2, head 3, shoulders 4/8, hips 12/16, knees 13/17,
    # feet 15/19
    "babel": Skeleton(root=1, hips=(12, 16), shoulders=(4, 8), trunk=(0, 2), drop=(), head=3, feet=(15, 19),
                      order=(3, (4, 8), (12, 16), (13, 17), (15, 19))),
    # LARa annotation tool segments: head 0, head end 1, collars 2/12 (same point), elbows 3/13, femurs 4/14 (hips),
    # feet 5/15, humerus 6/16 (shoulders), tibias 7/17 (knees), toes 8/18, wrists 9/19, wrist ends 10/20,
    # lower back 11, root 21
    "lara": Skeleton(root=21, hips=(4, 14), shoulders=(6, 16), trunk=(21, 0), drop=(12,), head=1, feet=(5, 15),
                     order=(1, 0, (6, 16), 11, (4, 14), (7, 17), (5, 15))),
}

BLOCKS_BY_SOURCE = {"relational": ("dist",), "relational+vertical": ("dist", "vertical"),
                    "relational+heading": ("dist", "heading"), "relational+vertical+heading": ("dist", "vertical", "heading")}
T5_KEYS = ("t5_both", "t5_heading", "t5_gravity")


def register_t5_tokenizers() -> None:
    specs = (
        TOK.TokenizerSpec("t5_gravity", "T5g", "relational+vertical", 64, False, ("babel1", "babel2", "babel3", "lara"),
                          span=2, selectable=False),
        TOK.TokenizerSpec("t5_heading", "T5h", "relational+heading", 64, False, ("babel1", "babel2", "babel3", "lara"),
                          span=2, selectable=False),
        TOK.TokenizerSpec("t5_both", "T5gh", "relational+vertical+heading", 64, False, ("babel1", "babel2", "babel3", "lara"),
                          span=2, selectable=False),
        # gate G8a only: the T5 code path with the distance block alone must reproduce T2
        TOK.TokenizerSpec("t5_dist_only", "T5-dist", "relational", 64, False, ("babel2",), span=2, selectable=False),
    )
    for s in specs:
        if s.key not in TOK.SPECS:
            TOK.SPECS[s.key] = s


register_t5_tokenizers()


def skeleton(dataset: str) -> Skeleton:
    return SKELETONS["lara" if dataset == "lara" else "babel"]


# -- per-frame and per-window features ----------------------------------------------------------------------------

def load_positions(spec: D.DatasetSpec, name: str) -> np.ndarray:
    """(T, V, 3) float32 positions, exactly as `tokenizers.extract_relational` reads them."""
    ch = TOK.position_channels(spec.name)
    arr = np.load(D.features_dir(spec) / name)  # (C, T, V, M)
    if arr.shape[3] != 1:
        raise ValueError("relational features assume one person")
    if ch is not None:
        arr = arr[ch]
    pos = arr[..., 0].transpose(1, 2, 0)
    if pos.shape[2] != 3:
        raise ValueError(f"{spec.name}: expected 3 position channels, found {pos.shape[2]}")
    return pos.astype(np.float32)


def vertical_frames(pos: np.ndarray, sk: Skeleton) -> np.ndarray:
    """(T, H+1): root-relative heights of every distinct non-root joint, then trunk tilt (radians from +axis 2)."""
    keep = [j for j in range(pos.shape[1]) if j != sk.root and j not in sk.drop]
    p = pos.astype(np.float64)
    heights = p[:, keep, VERTICAL_AXIS]
    trunk = p[:, sk.trunk[1]] - p[:, sk.trunk[0]]
    norm = np.linalg.norm(trunk, axis=1)
    tilt = np.arccos(np.clip(trunk[:, VERTICAL_AXIS] / np.maximum(norm, 1e-9), -1.0, 1.0))
    return np.concatenate([heights, tilt[:, None]], axis=1).astype(np.float32)


def body_heading(pos: np.ndarray, sk: Skeleton) -> tuple[np.ndarray, np.ndarray]:
    """Per frame: heading angle psi of the horizontal left->right body axis, and whether it is defined."""
    p = pos.astype(np.float64)
    u = (p[:, sk.hips[1]] - p[:, sk.hips[0]]) + (p[:, sk.shoulders[1]] - p[:, sk.shoulders[0]])
    horiz = [a for a in range(3) if a != VERTICAL_AXIS]
    uh = u[:, horiz]
    nh, n = np.linalg.norm(uh, axis=1), np.linalg.norm(u, axis=1)
    valid = (n > 0) & (nh >= VALID_HORIZONTAL_SHARE * n)
    return np.arctan2(uh[:, 1], uh[:, 0]), valid


def heading_stats(psi: np.ndarray, valid: np.ndarray, window: int, fps: float) -> np.ndarray:
    """(P, 3): mean w, mean |w|, SD w of the yaw rate over consecutive valid frames inside each window (zeros if
    none). Differences never cross a window boundary, so a window's feature uses its own frames only."""
    T = len(psi)
    P = int(np.ceil(T / window))
    out = np.zeros((P, 3), dtype=np.float32)
    d = np.angle(np.exp(1j * (psi[1:] - psi[:-1]))) * fps  # frame t-1 -> t, wrapped to (-pi, pi]
    ok = valid[1:] & valid[:-1]
    for p in range(P):
        a, b = p * window, min(T, (p + 1) * window)
        if b - a < 2:
            continue
        w = d[a:b - 1][ok[a:b - 1]]
        if len(w):
            out[p] = (w.mean(), np.abs(w).mean(), w.std())
    return out


def window_features(pos: np.ndarray, blocks: tuple[str, ...], dataset: str, window: int, fps: float) -> tuple[np.ndarray, dict]:
    """One recording -> (P, D) features and per-recording counters."""
    sk = skeleton(dataset)
    parts, info = [], {}
    if "dist" in blocks:
        parts.append(TOK.window_stats(TOK.frame_distances(pos.astype(np.float32)), window))
    if "vertical" in blocks:
        parts.append(TOK.window_stats(vertical_frames(pos, sk), window))
    if "heading" in blocks:
        psi, valid = body_heading(pos, sk)
        parts.append(heading_stats(psi, valid, window, fps))
        info["heading_frames"], info["heading_undefined"] = int(len(valid)), int((~valid).sum())
    flat = np.concatenate(parts, axis=1).astype(np.float32) if len(parts) > 1 else parts[0]
    if not np.isfinite(flat).all():
        raise RuntimeError("non-finite T5 feature")
    return flat, info


def block_slices(blocks: tuple[str, ...], dataset: str) -> dict[str, slice]:
    sk = SKELETONS["lara" if dataset == "lara" else "babel"]
    V = 22 if dataset == "lara" else 25
    width = dict(dist=3 * V * (V - 1) // 2, vertical=3 * (V - 1 - len(sk.drop) + 1), heading=3)
    out, at = {}, 0
    for b in blocks:
        out[b] = slice(at, at + width[b])
        at += width[b]
    return out


def extract8(spec: D.DatasetSpec, source: str, names: list[str]) -> tuple[dict[str, dict], dict]:
    blocks = BLOCKS_BY_SOURCE[source]
    out, frames, undefined = {}, 0, 0
    for name in names:
        pos = load_positions(spec, name)
        flat, info = window_features(pos, blocks, spec.name, spec.window, spec.fps)
        T = pos.shape[0]
        num_patches = int(np.ceil(T / spec.window))
        lengths = np.full(num_patches, spec.window, dtype=np.int32)
        pad = num_patches * spec.window - T
        if pad:
            lengths[-1] = spec.window - pad
        out[name] = dict(flat=flat, lengths=lengths, num_patches=num_patches)
        frames += info.get("heading_frames", 0)
        undefined += info.get("heading_undefined", 0)
    width = sum(s.stop - s.start for s in block_slices(blocks, spec.name).values())
    if out and next(iter(out.values()))["flat"].shape[1] != width:
        raise RuntimeError("feature width disagrees with the declared block layout")
    stats = dict(heading_frames=frames, heading_undefined=undefined,
                 heading_undefined_share=(undefined / frames) if frames else None)
    return out, stats


# -- population fit: `tokenizers.fit_population` plus block weighting ------------------------------------------------

def fit_population8(name: str, extracted_pop: dict[str, dict], sample_ids: list[tuple[str, int]], pca_dim: int,
                    num_codes: int, seed: int, slices: dict[str, slice], whiten: bool = False):
    """`tokenizers.fit_population` line for line, except that with more than one block each appended block is
    multiplied by s1(dist) / s1(block) after standardization (folded into the scaler's `scale_`, so every later
    `scaler.transform` -- including `fitmod.project_pca` -- applies it)."""
    sample = fitmod.gather(extracted_pop, sample_ids)
    if len(sample) < num_codes:
        raise RuntimeError(f"population {name!r}: only {len(sample)} sampled complete windows, "
                           f"need >= {num_codes} for K-means (K={num_codes})")
    scaler = StandardScaler().fit(sample)
    weights = {}
    if len(slices) > 1:
        std = scaler.transform(sample)
        s1 = {b: float(np.linalg.svd(std[:, sl], compute_uv=False)[0]) for b, sl in slices.items()}
        for b, sl in slices.items():
            if b == "dist":
                continue
            w = s1["dist"] / s1[b]
            scaler.scale_[sl] = scaler.scale_[sl] / w
            weights[b] = dict(weight=w, s1=s1[b])
        weights["dist"] = dict(weight=1.0, s1=s1["dist"])
    sample_std = scaler.transform(sample)
    n_components = min(pca_dim, sample_std.shape[0], sample_std.shape[1])
    pca = PCA(n_components=n_components, whiten=whiten, svd_solver="randomized", random_state=seed)
    sample_pca = pca.fit_transform(sample_std)
    kmeans = KMeans(n_clusters=num_codes, init="k-means++", n_init=5, max_iter=300, tol=1e-4, algorithm="lloyd",
                    random_state=seed)
    kmeans.fit(sample_pca)
    occupied = len(set(kmeans.labels_.tolist()))
    return fitmod.FittedPopulation(name, scaler, pca, kmeans, sample_ids, len(sample), occupied), weights


def pca_block_share(pca: PCA, slices: dict[str, slice]) -> dict[str, float]:
    """Share of the retained PCA variance carried by each block's coordinates."""
    ev = pca.explained_variance_
    tot = float(ev.sum())
    return {b: float((ev[:, None] * pca.components_[:, sl] ** 2).sum() / tot) for b, sl in slices.items()}


# -- cell preparation: `tokenizers.prepare_cell` / `_build` with the two changes above ------------------------------

def prepare_cell(dataset: str, tokenizer: str, seed: int, run_dir: Path, window: int | None = None,
                 recording_cap: int | None = None) -> Path:
    key = TOK.make_key(dataset, tokenizer, seed, window, recording_cap)
    spec = TOK.dataset_spec(dataset, window)
    names = D.recording_names(spec, recording_cap)
    counts = D.frame_counts(spec, names)
    run_dir = Path(run_dir)
    if TOK.manifest_ok(run_dir, key, names, counts) and (run_dir / BLOCK_FILE).exists():
        return run_dir
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    with threadpool_limits(limits=key.threads):
        _build(spec, key, run_dir, names, counts)
    return run_dir


def _build(spec: D.DatasetSpec, key: TOK.PrepKey, run_dir: Path, names: list[str], counts: dict[str, int]) -> None:
    tok = TOK.SPECS[key.tokenizer]
    family = tok.family
    raw, feat_stats = extract8(spec, tok.source, names)
    slices = block_slices(BLOCKS_BY_SOURCE[tok.source], spec.name)
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

    pop, weights = fit_population8(family, raw, sample_ids, key.pca_dim, key.num_codes, key.seed, slices, whiten=key.whiten)
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
    artifacts.save_family(run_dir, family, records, lookup, stats["mean"], stats["rms"], grouping, counts_w,
                          window_counts)
    write_json_atomic(run_dir / BLOCK_FILE, dict(
        blocks=list(slices), widths={b: s.stop - s.start for b, s in slices.items()}, weights=weights,
        pca_variance_share=pca_block_share(pop.pca, slices), pca_explained_variance_ratio=float(pop.pca.explained_variance_ratio_.sum()),
        kmeans_occupied=int(pop.n_occupied), **feat_stats))
    write_json_atomic(run_dir / "manifest.json", dict(
        schema=TOK.SCHEMA, prep_key=asdict(key), families=[family], recording_names=names,
        recording_frame_counts=frame_counts, num_windows={family: {n: int(len(r["codes"])) for n, r in records.items()}},
        feature_dims={family: int(lookup.shape[1])}, sample_ids_hash=artifacts.sample_ids_hash(sample_ids),
        n_train_recordings=len(train_names), n_val_recordings=len(val_names),
        kmeans_occupied=int(pop.n_occupied), n_fit_windows=int(pop.n_fit_windows)))


def is_prepared(dataset: str, tokenizer: str, seed: int, run_dir: Path, window: int | None = None,
                recording_cap: int | None = None) -> bool:
    return TOK.is_prepared(dataset, tokenizer, seed, run_dir, window, recording_cap) and (Path(run_dir) / BLOCK_FILE).exists()


# -- gates G8b / G8c (pure functions; the runner records them) ------------------------------------------------------

def geometry_check(dataset: str, recording_cap: int | None = None, frame_stride: int = 5) -> dict:
    """G8b on every recording of `dataset` (frames subsampled by `frame_stride`)."""
    spec = TOK.dataset_spec(dataset)
    sk = skeleton(dataset)
    names = D.recording_names(spec, recording_cap)
    pos = np.concatenate([load_positions(spec, n)[::frame_stride] for n in names]).astype(np.float64)
    checks, detail = {}, {}
    checks["root_zero"] = bool(np.abs(pos[:, sk.root]).max() == 0.0)
    hf = pos[:, sk.head] - pos[:, list(sk.feet)].mean(1)
    med = np.median(hf, axis=0)
    detail["head_minus_feet_median"] = med.round(2).tolist()
    checks["axis2_vertical"] = bool(int(np.argmax(np.abs(med))) == VERTICAL_AXIS and med[VERTICAL_AXIS] > 0)
    heights = np.median(pos[:, :, VERTICAL_AXIS], axis=0)
    seq = [float(np.mean([heights[j] for j in (o if isinstance(o, tuple) else (o,))])) for o in sk.order]
    detail["order_heights"] = [round(x, 1) for x in seq]
    checks["anatomical_order"] = all(a > b for a, b in zip(seq, seq[1:]))
    for label, (l, r) in (("hips", sk.hips), ("shoulders", sk.shoulders)):
        d = pos[:, r] - pos[:, l]
        share = np.abs(d[:, VERTICAL_AXIS]) / np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        detail[f"{label}_vertical_share_median"] = float(np.median(share))
        checks[f"{label}_horizontal"] = float(np.median(share)) < 0.2
    h, s = pos[:, sk.hips[1]] - pos[:, sk.hips[0]], pos[:, sk.shoulders[1]] - pos[:, sk.shoulders[0]]
    ang = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(h[:, 1], h[:, 0]) - np.arctan2(s[:, 1], s[:, 0]))))))
    detail["hip_shoulder_axis_angle_median_deg"] = float(np.median(ang))
    checks["hip_shoulder_agree"] = float(np.median(ang)) < 15.0
    if dataset == "lara":
        checks["collars_coincide"] = bool(np.abs(pos[:, 2] - pos[:, 12]).max() < 1e-3)
    detail["frames_checked"] = int(len(pos))
    return dict(status="passed" if all(checks.values()) else "failed", checks=checks, detail=detail)


def rotate_z(pos: np.ndarray, theta) -> np.ndarray:
    """Rotate every frame about axis 2 by theta (scalar or per-frame array, radians)."""
    th = np.broadcast_to(np.asarray(theta, dtype=np.float64), (pos.shape[0],))
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    p = pos.astype(np.float64)
    out = p.copy()
    out[..., 0] = c * p[..., 0] - s * p[..., 1]
    out[..., 1] = s * p[..., 0] + c * p[..., 1]
    return out.astype(np.float32)


def invariance_check(dataset: str = "babel2", omega: float = 1.0) -> dict:
    """G8c on the first recording of `dataset`: T5gh features are unchanged by a fixed rotation about axis 2, and a
    rotation growing at `omega` rad/s adds `omega` to mean w in every window whose frames are all valid."""
    spec = TOK.dataset_spec(dataset)
    name = D.recording_names(spec, None)[0]
    pos = load_positions(spec, name)
    blocks = BLOCKS_BY_SOURCE["relational+vertical+heading"]
    base, _ = window_features(pos, blocks, dataset, spec.window, spec.fps)
    rot, _ = window_features(rotate_z(pos, 0.65), blocks, dataset, spec.window, spec.fps)
    fixed_diff = float(np.abs(base - rot).max())
    t = np.arange(pos.shape[0]) / spec.fps
    turn, _ = window_features(rotate_z(pos, omega * t), blocks, dataset, spec.window, spec.fps)
    sl = block_slices(blocks, dataset)["heading"]
    psi, valid = body_heading(pos, skeleton(dataset))
    W = spec.window
    full = [p for p in range(base.shape[0]) if (p + 1) * W <= len(valid) and valid[p * W:(p + 1) * W].all()]
    gain = turn[full, sl.start] - base[full, sl.start]
    turn_err = float(np.abs(gain - omega).max()) if full else float("inf")
    non_heading = [i for i in range(base.shape[1]) if not (sl.start <= i < sl.stop)]
    turn_other = float(np.abs(turn[:, non_heading] - base[:, non_heading]).max())
    checks = dict(fixed_rotation_invariant=fixed_diff <= 1e-3, turn_adds_omega=turn_err <= 1e-3,
                  turn_leaves_other_blocks=turn_other <= 1e-3, windows_tested=len(full) > 0)
    return dict(status="passed" if all(checks.values()) else "failed", checks=checks, recording=name,
                fixed_rotation_max_abs_diff=fixed_diff, turn_max_abs_err=turn_err, turn_other_max_abs_diff=turn_other,
                windows_tested=len(full))
