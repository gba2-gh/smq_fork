"""Window-identifier sampling and per-population preprocessing (standardizer
+ randomized PCA-64 + K-means-512), shared conditioning statistics, and the
seed-111 initial code-to-action grouping.

Populations: "raw" (raw-input windows), "vq_encoder" (VQ arm's
pre-quantization encoder patches), "fsq_encoder" (FSQ arm's, before the
complete FSQ adapter). Each population feeds exactly one K-means family
(raw_kmeans / vq_kmeans / fsq_kmeans); vq_encoder and fsq_encoder additionally
supply the *unprojected* continuous feature for the two native families.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from script.action_transport import categorical, config as at_config
from script.action_transport.stageb_data import build_validation_split
from script.fsq_reconstruction.evaluate import per_code_pca_centers
from script.quantized_input_comparison import config as cfg


def window_lengths(extracted_pop: dict[str, dict]) -> dict[str, np.ndarray]:
    return {name: rec["lengths"] for name, rec in extracted_pop.items()}


def complete_window_ids(lengths_by_name: dict[str, np.ndarray], window: int) -> list[tuple[str, int]]:
    """(recording, patch_index) pairs whose real length equals the full
    window -- the "complete window" population used for fitting. Terminal
    (zero-padded) windows are excluded here but retained elsewhere for
    assignment/evaluation, per spec."""
    ids = []
    for name in sorted(lengths_by_name):
        lengths = lengths_by_name[name]
        for p, length in enumerate(lengths):
            if int(length) == window:
                ids.append((name, p))
    return ids


def sample_window_ids(all_ids: list[tuple[str, int]], cap: int, seed: int) -> list[tuple[str, int]]:
    rng = np.random.default_rng(seed)
    if len(all_ids) <= cap:
        idx = rng.permutation(len(all_ids))
    else:
        idx = rng.choice(len(all_ids), size=cap, replace=False)
    return [all_ids[i] for i in idx]


def gather(extracted_pop: dict[str, dict], ids: list[tuple[str, int]]) -> np.ndarray:
    return np.stack([extracted_pop[name]["flat"][p] for name, p in ids], axis=0).astype(np.float32)


@dataclass
class FittedPopulation:
    name: str
    scaler: StandardScaler
    pca: PCA
    kmeans: KMeans
    sample_ids: list[tuple[str, int]]
    n_fit_windows: int
    n_occupied: int


def fit_population(name: str, extracted_pop: dict[str, dict], sample_ids: list[tuple[str, int]],
                    pca_dim: int, num_codes: int, seed: int) -> FittedPopulation:
    sample = gather(extracted_pop, sample_ids)
    if len(sample) < num_codes:
        raise RuntimeError(f"population {name!r}: only {len(sample)} sampled complete windows, "
                            f"need >= {num_codes} for K-means (K={num_codes})")
    scaler = StandardScaler().fit(sample)
    sample_std = scaler.transform(sample)
    n_components = min(pca_dim, sample_std.shape[0], sample_std.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    sample_pca = pca.fit_transform(sample_std)
    kmeans = KMeans(n_clusters=num_codes, init="k-means++", n_init=5, max_iter=300, tol=1e-4,
                     algorithm="lloyd", random_state=seed)
    kmeans.fit(sample_pca)
    occupied = len(set(kmeans.labels_.tolist()))
    return FittedPopulation(name, scaler, pca, kmeans, sample_ids, len(sample), occupied)


def project_pca(pop: FittedPopulation, flat: np.ndarray) -> np.ndarray:
    return pop.pca.transform(pop.scaler.transform(flat)).astype(np.float32)


def assign_kmeans(pop: FittedPopulation, flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pca_feats = project_pca(pop, flat)
    ids = pop.kmeans.predict(pca_feats).astype(np.int64)
    return pca_feats, ids


def _streamed_pooled_mean(feature_by_name: dict[str, np.ndarray], ids: list[tuple[str, int]]) -> np.ndarray:
    """Per-coordinate mean over `ids`, accumulated one window at a time
    (float64 accumulator, float32/pop-native per-window reads) rather than
    materializing a `[len(ids), d]` float64 array up front -- bounds peak
    memory to O(d) regardless of how many windows are pooled."""
    total = None
    n = 0
    for name, p in ids:
        v = feature_by_name[name][p].astype(np.float64)
        total = v.copy() if total is None else total + v
        n += 1
    if total is None or n == 0:
        raise RuntimeError("conditioning_stats: empty id list, cannot compute a mean")
    return total / n


def _streamed_scalar_rms(feature_by_name: dict[str, np.ndarray], ids: list[tuple[str, int]]) -> float:
    """Single scalar RMS pooled over every coordinate of every window in
    `ids` (not per-coordinate): sqrt(mean(x^2)) over the full
    windows x coordinates population, accumulated incrementally so no full
    float64 array of the training-window population is ever materialized.
    Floored at 1.0 when below 1e-8, per spec ("one scalar shared by
    quantized and continuous inputs, preserving their relative geometry")."""
    sum_sq = 0.0
    count = 0
    for name, p in ids:
        v = feature_by_name[name][p].astype(np.float64)
        sum_sq += float(np.sum(v * v))
        count += v.size
    if count == 0:
        raise RuntimeError("conditioning_stats: empty id list, cannot compute an RMS")
    rms = float(np.sqrt(sum_sq / count))
    return 1.0 if rms < 1e-8 else rms


def conditioning_stats(feature_by_name: dict[str, np.ndarray], sample_ids: list[tuple[str, int]],
                        train_complete_ids: list[tuple[str, int]]) -> dict:
    """`feature_by_name[name]` is [P, d] float32 in the family's own
    continuous-feature space (raw 1440-dim for native families, PCA-64 for
    K-means families). Returns the pooled fitting-window mean (per-coordinate,
    for direct-OT centering) and a single train-split coordinate-RMS SCALAR
    (pooled over every coordinate of every complete training window, not
    one value per coordinate) used to scale both the quantized and the
    continuous inputs identically -- preserving their relative geometry --
    floored at 1.0 when below 1e-8. Both are accumulated incrementally
    (never stacking the full training-window population into one array)."""
    mean = _streamed_pooled_mean(feature_by_name, sample_ids)
    rms = _streamed_scalar_rms(feature_by_name, train_complete_ids)
    return dict(mean=mean.astype(np.float32), rms=np.float32(rms))


def shared_validation_split(frame_counts: dict[str, int], seed: int) -> tuple[list[str], list[str]]:
    names = sorted(frame_counts)
    return build_validation_split(names, frame_counts, seed=seed)


def initial_grouping(codes_by_name: dict[str, np.ndarray], lengths_by_name: dict[str, np.ndarray],
                      pca_by_name: dict[str, np.ndarray], num_codes: int, num_states: int,
                      fps: int, window: int, seed: int):
    """Seed-111 initial code-to-action grouping (frame-weighted per-code
    means in the family's continuous PCA space, via the existing
    `categorical.build_grouping` helper). Returns (grouping, centers,
    counts); raises `categorical.UnavailableError` if fewer than
    `num_states` prototypes are occupied (never silently worked around)."""
    recs = [categorical.RecordingInput(name, np.asarray(codes_by_name[name], dtype=np.int32),
                                        np.asarray(lengths_by_name[name], dtype=np.int32), fps, window)
            for name in sorted(codes_by_name)]
    centers, counts = per_code_pca_centers(recs, pca_by_name, num_codes)
    grouping = categorical.build_grouping(centers, counts, num_states, seed, is_kc_identity=False)
    return grouping, centers, counts
