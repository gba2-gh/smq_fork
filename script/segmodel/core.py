"""Small, label-free M1 model primitives; labels are used only by evaluation helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "segmodel"
SEEDS = (111, 222, 1538574472)
WINDOW = {"hugadb": 15, "lara": 12}
FPS = {"hugadb": 60, "lara": 50}
CLASSES = {"hugadb": 10, "lara": 8}


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def rle(values: np.ndarray):
    cut = np.flatnonzero(values[1:] != values[:-1]) + 1
    edges = np.r_[0, cut, len(values)]
    return np.diff(edges), values[edges[:-1]], edges[:-1]


def per_recording_zscore(arrays):
    out = []
    for x in arrays:
        x = np.asarray(x, dtype=np.float32)
        sd = x.std(0); sd[sd < 1e-8] = 1.
        out.append((x - x.mean(0)) / sd)
    return out


def make_windows(array: np.ndarray, width: int):
    starts = np.arange(0, len(array), width, dtype=np.int32)
    real = np.minimum(width, len(array) - starts).astype(np.int32)
    out = np.zeros((len(starts), width * array.shape[1]), np.float32)
    for j, start in enumerate(starts):
        out[j, :real[j] * array.shape[1]] = array[start:start + real[j]].reshape(-1)
    return out, starts, real


@dataclass
class Prepared:
    transformed: list[np.ndarray]
    real_lengths: list[np.ndarray]
    fit_ids: np.ndarray
    fit_features: np.ndarray
    scaler: StandardScaler
    pca: PCA


def prepare(arrays, width: int, fit_records, seed: int = 111):
    all_windows = [make_windows(x, width) for x in arrays]
    candidates = [(r, j) for r in fit_records for j, n in enumerate(all_windows[r][2]) if n == width]
    rng = np.random.default_rng(seed)
    use = rng.choice(len(candidates), min(10_000, len(candidates)), replace=False)
    ids = np.asarray([candidates[j] for j in np.sort(use)], np.int32)
    sample = np.stack([all_windows[r][0][j] for r, j in ids]).astype(np.float32)
    scaler = StandardScaler().fit(sample)
    sample = scaler.transform(sample)
    pca = PCA(n_components=min(64, *sample.shape), svd_solver="randomized", random_state=seed)
    pca.fit(sample)
    transformed = [pca.transform(scaler.transform(x[0])).astype(np.float32) for x in all_windows]
    return Prepared(transformed, [x[2] for x in all_windows], ids,
                    pca.transform(sample).astype(np.float32), scaler, pca)


def posterior(features: list[np.ndarray], centers: np.ndarray, tau: float):
    out = []
    for x in features:
        d = ((x[:, None] - centers[None]) ** 2).sum(2)
        a = -d / tau; a -= a.max(1, keepdims=True)
        p = np.exp(a); out.append((p / p.sum(1, keepdims=True)).astype(np.float32))
    return out


def base_temperature(features: np.ndarray, centers: np.ndarray):
    d = ((features[:, None] - centers[None]) ** 2).sum(2)
    two = np.partition(d, 1, axis=1)[:, :2]
    gaps = two[:, 1] - two[:, 0]
    gaps = gaps[gaps > 0]
    return None if not len(gaps) else float(np.median(gaps))


def histogram_prefix(posterior: np.ndarray, lengths: np.ndarray):
    return np.vstack([np.zeros((1, posterior.shape[1])), np.cumsum(posterior * lengths[:, None], 0)])


def candidate_costs(prefix: np.ndarray, fps: int, max_windows: int, centroids: np.ndarray):
    """End-indexed best costs/states for segments in window coordinates."""
    n = len(prefix) - 1
    costs = np.full((n + 1, max_windows), np.inf)
    states = np.full_like(costs, -1, dtype=np.int16)
    # Each column contains every segment of one length; this avoids an
    # expensive Python loop during the repeated lambda calibration solves.
    for length in range(1, min(max_windows, n) + 1):
        counts = prefix[length:] - prefix[:-length]
        mass = counts.sum(1)
        hist = np.sqrt(counts / mass[:, None])
        distances = ((hist[:, None, :] - centroids[None, :, :]) ** 2).sum(2)
        state = distances.argmin(1)
        costs[length:, length - 1] = mass / fps * distances[np.arange(len(state)), state]
        states[length:, length - 1] = state
    return costs, states


def dp(costs: np.ndarray, lam: float):
    n, maxw = costs.shape[0] - 1, costs.shape[1]
    best = np.full(n + 1, np.inf); choice = np.zeros(n + 1, np.int32); best[0] = 0.
    for end in range(1, n + 1):
        lengths = np.arange(1, min(maxw, end) + 1)
        values = best[end - lengths] + costs[end, :len(lengths)] + lam
        k = int(values.argmin()); best[end] = values[k]; choice[end] = lengths[k]
    edges = [n]
    while edges[-1]: edges.append(edges[-1] - choice[edges[-1]])
    return np.asarray(edges[::-1], np.int32), float(best[n])


def fit_centroids(posteriors, lengths, n_classes, seed):
    rows = []
    for p, lens in zip(posteriors, lengths):
        pref = histogram_prefix(p, lens)
        for end in range(1, len(p) + 1, 4):
            counts = pref[end] - pref[max(0, end - 4)]
            rows.append(np.sqrt(counts / counts.sum()))
    return KMeans(n_clusters=n_classes, init="k-means++", n_init=5, max_iter=300,
                  tol=1e-4, algorithm="lloyd", random_state=seed).fit(np.asarray(rows)).cluster_centers_


def calibrate_lambda(costs, target: int):
    """Choose λ with frozen centroids using only fitting features and segment count."""
    def count(lam): return sum(len(dp(c, lam)[0]) - 1 for c in costs)
    lo, hi = 0., 1.
    while count(hi) > target and hi < 1e8: hi *= 2
    best = (lo, count(lo))
    for _ in range(40):
        mid = (lo + hi) / 2; value = count(mid)
        if abs(value-target) < abs(best[1]-target): best = (mid, value)
        if value > target: lo = mid
        else: hi = mid
    return best[0], {"target_segments": int(target), "initial_segments": int(best[1]), "within_2pct": abs(best[1]-target) <= .02 * target}


def fit_segmental(posteriors, lengths, width, fps, n_classes, seed, lam, max_seconds=10.):
    centroids = fit_centroids(posteriors, lengths, n_classes, seed)
    max_windows = max(1, int(np.floor(max_seconds * fps / width)))
    trace = []
    for iteration in range(20):
        partitions=[]; total=0.; all_hist=[]; all_weight=[]; all_state=[]
        for p, lens in zip(posteriors, lengths):
            pref=histogram_prefix(p,lens); costs,states=candidate_costs(pref,fps,max_windows,centroids); edges,value=dp(costs,lam)
            total += value; partitions.append((edges, states, pref))
            for a,b in zip(edges[:-1],edges[1:]):
                counts=pref[b]-pref[a]; all_hist.append(np.sqrt(counts/counts.sum())); all_weight.append(counts.sum()/fps); all_state.append(states[b,b-a-1])
        h=np.asarray(all_hist); weights=np.asarray(all_weight); state=np.asarray(all_state)
        next_centroids=centroids.copy(); occupancy=[]
        for c in range(n_classes):
            use=state==c; occupancy.append(int(use.sum()))
            if use.any(): next_centroids[c]=np.average(h[use],axis=0,weights=weights[use])
        trace.append({"round":iteration + 1,"objective":float(total),"state_segments":occupancy,"mean_duration_seconds":float(np.average(weights))})
        if iteration and total > trace[-2]["objective"] + 1e-6*max(1,abs(trace[-2]["objective"])): raise RuntimeError("alternating objective increased")
        centroids=next_centroids
        if iteration and abs(trace[-2]["objective"]-total) <= 1e-6*max(1,abs(trace[-2]["objective"])): break
    labels=[]; edges=[]
    for e, states, _ in partitions:
        labels.append(np.asarray([states[b,b-a-1] for a,b in zip(e[:-1],e[1:])],np.int32)); edges.append(e)
    return centroids, edges, labels, trace
