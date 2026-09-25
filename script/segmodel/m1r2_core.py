"""Corrected Milestone-1 segment objective and reusable experiment primitives."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "segmodel" / "m1r2"
SEEDS = (111, 222, 1538574472)
WINDOW = {"hugadb": 15, "lara": 12}
FPS = {"hugadb": 60, "lara": 50}
MAX_SECONDS = 20


class DeadlineReached(RuntimeError):
    pass


def check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise DeadlineReached("five-hour execution deadline reached")


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def frame_lengths(total: int, width: int) -> np.ndarray:
    starts = np.arange(0, total, width)
    return np.minimum(width, total - starts).astype(np.int32)


def soft_posteriors(features, centers: np.ndarray, temperature: float):
    result = []
    for x in features:
        distance = ((x[:, None] - centers[None]) ** 2).sum(2)
        logits = -distance / temperature
        logits -= logits.max(1, keepdims=True)
        value = np.exp(logits)
        result.append((value / value.sum(1, keepdims=True)).astype(np.float32))
    return result


def base_temperature(features: np.ndarray, centers: np.ndarray) -> float:
    distance = ((features[:, None] - centers[None]) ** 2).sum(2)
    nearest = np.partition(distance, 1, axis=1)[:, :2]
    gaps = nearest[:, 1] - nearest[:, 0]
    gaps = gaps[gaps > 0]
    if not len(gaps):
        raise ValueError("no strictly positive prototype-distance gap")
    return float(np.median(gaps))


def representations(posteriors, mode: str):
    if mode == "soft":
        return [np.sqrt(q).astype(np.float32) for q in posteriors]
    if mode == "hard":
        output = []
        for q in posteriors:
            hard = np.zeros_like(q)
            hard[np.arange(len(q)), q.argmax(1)] = 1
            output.append(hard)
        return output
    raise ValueError(mode)


def one_second_chunks(values, lengths, fps: int, n_states: int, seed: int):
    """Frame-overlap means over exact one-second intervals."""
    rows = []
    for x, n in zip(values, lengths):
        total = int(n.sum())
        starts = np.r_[0, np.cumsum(n[:-1])]
        ends = starts + n
        for left in range(0, total, fps):
            right = min(total, left + fps)
            overlap = np.maximum(0, np.minimum(ends, right) - np.maximum(starts, left))
            use = overlap > 0
            rows.append(np.average(x[use], axis=0, weights=overlap[use]))
    return KMeans(n_clusters=n_states, init="k-means++", n_init=5, max_iter=300,
                  tol=1e-4, algorithm="lloyd", random_state=seed).fit(rows).cluster_centers_


def point_cost_prefix(values: np.ndarray, lengths: np.ndarray, centers: np.ndarray):
    cost = lengths[:, None] * ((values[:, None] - centers[None]) ** 2).sum(2)
    return np.vstack([np.zeros((1, centers.shape[0])), np.cumsum(cost, axis=0)])


def candidate_costs(values, lengths, centers, max_windows: int):
    prefix = point_cost_prefix(values, lengths, centers)
    n = len(values)
    costs = np.full((n + 1, max_windows), np.inf)
    states = np.full((n + 1, max_windows), -1, dtype=np.int16)
    for duration in range(1, min(n, max_windows) + 1):
        by_state = prefix[duration:] - prefix[:-duration]
        state = by_state.argmin(1)
        costs[duration:, duration - 1] = by_state[np.arange(len(state)), state]
        states[duration:, duration - 1] = state
    return costs, states


def decode_costs(costs: np.ndarray, states: np.ndarray, penalty: float):
    n, max_windows = costs.shape[0] - 1, costs.shape[1]
    best = np.full(n + 1, np.inf)
    choice = np.zeros(n + 1, np.int32)
    best[0] = 0
    for end in range(1, n + 1):
        durations = np.arange(1, min(end, max_windows) + 1)
        values = best[end - durations] + costs[end, :len(durations)] + penalty
        pick = int(values.argmin())
        best[end], choice[end] = values[pick], durations[pick]
    edges = [n]
    while edges[-1]:
        edges.append(edges[-1] - choice[edges[-1]])
    edges = np.asarray(edges[::-1], dtype=np.int32)
    labels = np.asarray([states[b, b-a-1] for a, b in zip(edges[:-1], edges[1:])], dtype=np.int32)
    return edges, labels, float(best[n])


def decode_all(values, lengths, centers, penalty: float, max_windows: int, deadline: float):
    edges, labels, objective = [], [], 0.0
    for x, n in zip(values, lengths):
        check_deadline(deadline)
        costs, states = candidate_costs(x, n, centers, max_windows)
        e, z, value = decode_costs(costs, states, penalty)
        edges.append(e); labels.append(z); objective += value
    return edges, labels, objective


def action_runs(labels) -> int:
    return sum(0 if not len(z) else 1 + int(np.count_nonzero(z[1:] != z[:-1])) for z in labels)


def update_centers(values, lengths, edges, labels, old_centers):
    sums = np.zeros_like(old_centers, dtype=np.float64)
    weights = np.zeros(len(old_centers), dtype=np.float64)
    occupancy = np.zeros(len(old_centers), dtype=np.int64)
    for x, n, e, z in zip(values, lengths, edges, labels):
        prefix_x = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x * n[:, None], axis=0)])
        prefix_n = np.r_[0, np.cumsum(n)]
        for a, b, state in zip(e[:-1], e[1:], z):
            sums[state] += prefix_x[b] - prefix_x[a]
            weights[state] += prefix_n[b] - prefix_n[a]
            occupancy[state] += 1
    centers = old_centers.copy()
    used = weights > 0
    centers[used] = sums[used] / weights[used, None]
    return centers, occupancy


def assigned_objective(values, lengths, centers, edges, labels, penalty):
    total = penalty * sum(len(z) for z in labels)
    for x, n, e, z in zip(values, lengths, edges, labels):
        for a, b, state in zip(e[:-1], e[1:], z):
            total += float((n[a:b, None] * (x[a:b] - centers[state]) ** 2).sum())
    return total


def alternate(values, lengths, initial, penalty, max_windows, deadline, max_rounds=60):
    centers = initial.copy()
    trace = []
    previous = None
    converged = False
    for iteration in range(max_rounds):
        edges, labels, decoded = decode_all(values, lengths, centers, penalty, max_windows, deadline)
        centers, occupancy = update_centers(values, lengths, edges, labels, centers)
        objective = assigned_objective(values, lengths, centers, edges, labels, penalty)
        if objective > decoded + 1e-7 * max(1, abs(decoded)):
            raise RuntimeError("centroid update increased objective")
        improvement = None if previous is None else (previous - objective) / max(1, abs(previous))
        trace.append({"round": iteration + 1, "objective": objective,
                      "decoded_objective": decoded, "relative_improvement": improvement,
                      "state_blocks": occupancy.tolist(), "action_runs": action_runs(labels)})
        if previous is not None and objective > previous + 1e-7 * max(1, abs(previous)):
            raise RuntimeError("alternating objective increased")
        if improvement is not None and improvement <= 1e-6:
            converged = True
            break
        previous = objective
    # Required consistency: decode once more with the final updated centroids.
    edges, labels, objective = decode_all(values, lengths, centers, penalty, max_windows, deadline)
    return centers, edges, labels, trace, converged, objective


def expand(edges, labels, frame_counts):
    predictions = []
    for e, z, n in zip(edges, labels, frame_counts):
        cumulative = np.r_[0, np.cumsum(n)]
        result = np.empty(int(cumulative[-1]), dtype=np.int32)
        for a, b, state in zip(e[:-1], e[1:], z):
            result[cumulative[a]:cumulative[b]] = state
        predictions.append(result)
    return predictions


def run_durations(predictions, fps):
    durations = []
    for prediction in predictions:
        edges = np.r_[0, np.flatnonzero(prediction[1:] != prediction[:-1]) + 1, len(prediction)]
        durations.extend(np.diff(edges) / fps)
    return np.asarray(durations)


def permute_rows(posteriors, seed: int):
    sizes = [len(q) for q in posteriors]
    flat = np.concatenate(posteriors)
    flat = flat[np.random.default_rng(seed).permutation(len(flat))]
    cuts = np.cumsum(sizes)[:-1]
    return [x.copy() for x in np.split(flat, cuts)]


@dataclass
class LambdaPoint:
    penalty: float
    blocks: int
    runs: int
    mean_duration: float
    edges: list[np.ndarray]
    labels: list[np.ndarray]
    objective: float


def lambda_curve(values, lengths, centers, max_windows, fps, deadline, points=13):
    cached = {}
    frozen = []
    for x, n in zip(values, lengths):
        check_deadline(deadline)
        frozen.append(candidate_costs(x, n, centers, max_windows))
    def evaluate(value):
        if value not in cached:
            e, z, objective = [], [], 0.0
            for costs, states in frozen:
                check_deadline(deadline)
                edge, label, part = decode_costs(costs, states, value)
                e.append(edge); z.append(label); objective += part
            total_seconds = sum(n.sum() for n in lengths) / fps
            cached[value] = LambdaPoint(value, sum(len(v) for v in z), action_runs(z),
                                        total_seconds / action_runs(z), e, z, objective)
        return cached[value]
    zero = evaluate(0.0)
    scale = max(zero.objective / max(1, zero.blocks), np.finfo(float).eps)
    high = scale
    minimum = sum(int(np.ceil(len(x) / max_windows)) for x in values)
    while evaluate(high).blocks > minimum and high < scale * 2**40:
        high *= 4
    positive = np.geomspace(scale / 1024, high, points - 1)
    return [zero] + [evaluate(float(value)) for value in positive]


def json_array(value) -> np.ndarray:
    return np.asarray([json.dumps(item, sort_keys=True) for item in value])
