"""Shared, label-auditable primitives for the Q1/Q2 diagnostic round."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    balanced_accuracy_score,
    confusion_matrix,
    mutual_info_score,
    normalized_mutual_info_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from script.repro.d_error_decomposition import rle, score

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "exp2round" / "q1q2"
SEEDS = (111, 222, 1538574472)
PERMUTATION_SEEDS = tuple(range(111, 131))
VOCAB_SIZES = (10, 20, 50, 100, 500, 1000)
WINDOWS = {"hugadb": (60, 30, 15), "lara": (50, 25, 12)}
NUM_ACTIONS = {"hugadb": 10, "lara": 8}
CHECKPOINTS = {
    "hugadb": ROOT / "models" / "pretrained" / "hugadb.model",
    "lara": ROOT / "models" / "pretrained" / "lara.model",
}
EXPECTED_HASH_PREFIX = {"hugadb": "22dfd5639c718177", "lara": "a45392ad30e27bda"}
D7_REFERENCE = {"hugadb": 44.85, "lara": 41.66}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def subject_id(dataset: str, name: str) -> str:
    if dataset == "hugadb":
        match = re.search(r"_(\d{2})_\d{2}(?:\.npy)?$", name)
    elif dataset == "lara":
        match = re.search(r"_(S\d+)_", name)
    else:
        raise ValueError(dataset)
    if not match:
        raise ValueError(f"cannot parse {dataset} subject from {name}")
    return match.group(1)


def majority(values: np.ndarray) -> int:
    labels, counts = np.unique(values, return_counts=True)
    return int(labels[np.argmax(counts)])


def entropy(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def conditional_entropy(target: np.ndarray, given: np.ndarray) -> float:
    return entropy(target) - float(mutual_info_score(target, given))


def conditional_mi(units: np.ndarray, subjects: np.ndarray, actions: np.ndarray) -> float:
    total = len(units)
    value = 0.0
    for action in np.unique(actions):
        mask = actions == action
        value += mask.sum() / total * mutual_info_score(units[mask], subjects[mask])
    return float(value)


def information_values(units: np.ndarray, actions: np.ndarray,
                       subjects: np.ndarray) -> dict[str, float | int | str]:
    return {
        "mi_unit_action": float(mutual_info_score(units, actions)),
        "mi_unit_subject": float(mutual_info_score(units, subjects)),
        "cmi_unit_subject_given_action": conditional_mi(units, subjects, actions),
        "h_unit": entropy(units),
        "h_action_given_unit": conditional_entropy(actions, units),
        "h_subject_given_unit": conditional_entropy(subjects, units),
        "nmi_unit_action": float(normalized_mutual_info_score(actions, units,
                                                                average_method="arithmetic")),
        "nmi_unit_subject": float(normalized_mutual_info_score(subjects, units,
                                                                 average_method="arithmetic")),
        "ami_unit_action": float(adjusted_mutual_info_score(actions, units,
                                                               average_method="arithmetic")),
        "ami_unit_subject": float(adjusted_mutual_info_score(subjects, units,
                                                                average_method="arithmetic")),
        "n_windows": int(len(units)),
        "occupied_units": int(len(np.unique(units))),
        "occupancy": json.dumps(np.bincount(units).tolist(), separators=(",", ":")),
    }


def permutation_references(units: np.ndarray, actions: np.ndarray,
                           subjects: np.ndarray) -> dict[str, tuple[float, float]]:
    marginal_keys = (
        "mi_unit_action", "mi_unit_subject", "h_unit", "h_action_given_unit",
        "h_subject_given_unit", "nmi_unit_action", "nmi_unit_subject",
        "ami_unit_action", "ami_unit_subject",
    )
    values = {key: [] for key in marginal_keys}
    conditional = []
    for seed in PERMUTATION_SEEDS:
        rng = np.random.default_rng(seed)
        permuted = rng.permutation(units)
        metrics = information_values(permuted, actions, subjects)
        for key in marginal_keys:
            values[key].append(float(metrics[key]))
        within = units.copy()
        for action in np.unique(actions):
            idx = np.flatnonzero(actions == action)
            within[idx] = rng.permutation(within[idx])
        conditional.append(conditional_mi(within, subjects, actions))
    result = {
        key: (float(np.mean(data)), float(np.std(data)))
        for key, data in values.items()
    }
    result["cmi_unit_subject_given_action"] = (
        float(np.mean(conditional)), float(np.std(conditional))
    )
    return result


def map_predictions(gts: Sequence[np.ndarray], predictions: Sequence[np.ndarray],
                    method: str) -> list[np.ndarray]:
    gt = np.concatenate(gts)
    pred = np.concatenate(predictions)
    gt_labels = np.unique(gt)
    pred_labels = np.unique(pred)
    counts = np.zeros((len(gt_labels), len(pred_labels)), dtype=np.int64)
    for row, gt_label in enumerate(gt_labels):
        mask = gt == gt_label
        for col, pred_label in enumerate(pred_labels):
            counts[row, col] = np.count_nonzero(pred[mask] == pred_label)
    mapping: dict[int, int] = {}
    if method == "hungarian":
        rows, cols = linear_sum_assignment(-counts)
        mapping.update({int(pred_labels[col]): int(gt_labels[row])
                        for row, col in zip(rows, cols)})
        sentinel = int(gt_labels.min()) - 1
        mapping.update({int(label): sentinel for label in pred_labels if int(label) not in mapping})
    elif method == "many_to_one":
        mapping.update({int(pred_labels[col]): int(gt_labels[np.argmax(counts[:, col])])
                        for col in range(len(pred_labels))})
    else:
        raise ValueError(method)
    return [np.array([mapping[int(value)] for value in sequence], dtype=np.int32)
            for sequence in predictions]


def majority_code_predictions(gts: Sequence[np.ndarray], codes: Sequence[np.ndarray]) -> list[np.ndarray]:
    output = []
    for gt, code in zip(gts, codes):
        prediction = np.empty(len(gt), dtype=np.int32)
        lengths, _, starts = rle(gt)
        for length, start in zip(lengths, starts):
            prediction[start:start + length] = majority(code[start:start + length])
        output.append(prediction)
    return output


def score_mapped(gts: Sequence[np.ndarray], predictions: Sequence[np.ndarray],
                 method: str = "hungarian") -> dict[str, float]:
    return score(gts, map_predictions(gts, predictions, method))


def make_folds(groups: Sequence[str], lengths: Sequence[int], n_folds: int = 4,
               seed: int = 111) -> list[int]:
    totals: dict[str, int] = {}
    for group, length in zip(groups, lengths):
        totals[group] = totals.get(group, 0) + int(length)
    rng = np.random.default_rng(seed)
    tie = {group: value for group, value in zip(totals, rng.random(len(totals)))}
    ordered = sorted(totals, key=lambda group: (-totals[group], tie[group]))
    fold_totals = [0] * n_folds
    assignment: dict[str, int] = {}
    for group in ordered:
        fold = int(np.argmin(fold_totals))
        assignment[group] = fold
        fold_totals[fold] += totals[group]
    return [assignment[group] for group in groups]


@dataclass
class Dataset:
    name: str
    names: list[str]
    gts: list[np.ndarray]
    codes: list[np.ndarray]
    arrays: list[np.ndarray]
    subjects: list[str]
    patch_size: int
    num_actions: int
    cache_identity: str


def load_dataset(dataset: str, representation: str = "latent") -> Dataset:
    pred_path = ROOT / "results" / "preds" / f"{dataset}_pretrained.npz"
    pred = np.load(pred_path, allow_pickle=True)
    names = [str(value) for value in pred["names"]]
    gts = [np.asarray(value, dtype=np.int32) for value in pred["gt"]]
    subjects = [subject_id(dataset, name) for name in names]
    if representation == "raw":
        arrays = []
        for name in names:
            raw = np.load(ROOT / "data" / dataset / "features" / name)
            arrays.append(np.moveaxis(raw, 1, 0).reshape(raw.shape[1], -1).astype(np.float32))
        codes = [np.asarray(value, dtype=np.int32) for value in pred["pred"]]
        identity = f"raw:{sha256_file(pred_path)}"
    else:
        cache = OUT / "cache" / dataset
        if not (cache / "_meta.npz").exists():
            cache = ROOT / "results" / "seg" / "cache" / dataset
        meta = np.load(cache / "_meta.npz", allow_pickle=True)
        meta_names = [str(value) for value in meta["names"]]
        normalized = [Path(name).stem for name in names]
        if meta_names != normalized:
            raise ValueError(f"{dataset} cache names do not match prediction dump")
        expected = sha256_file(CHECKPOINTS[dataset])
        if str(meta["ckpt_sha256"]) != expected:
            raise ValueError(f"{dataset} latent cache checkpoint hash mismatch")
        arrays = [np.load(cache / f"{name}.npy", mmap_mode="r") for name in meta_names]
        codes = [np.asarray(value, dtype=np.int32) for value in meta["codes"]]
        identity = f"{cache}:{expected}"
    for name, gt, code, array in zip(names, gts, codes, arrays):
        if len(gt) != len(code) or len(gt) != len(array):
            raise ValueError(f"frame alignment failed for {name}")
    return Dataset(dataset, names, gts, codes, arrays, subjects,
                   int(pred["patch_size"]), int(pred["num_actions"]), identity)


def recording_zscore(arrays: Sequence[np.ndarray]) -> list[np.ndarray]:
    result = []
    for array in arrays:
        value = np.asarray(array, dtype=np.float32)
        mean = value.mean(axis=0, keepdims=True)
        sd = value.std(axis=0, keepdims=True)
        sd[sd < 1e-8] = 1.0
        result.append((value - mean) / sd)
    return result


def window_inventory(arrays: Sequence[np.ndarray], window: int) -> tuple[list[tuple[int, int]], int]:
    complete = []
    terminal = 0
    for sequence, array in enumerate(arrays):
        for start in range(0, len(array) - window + 1, window):
            complete.append((sequence, start))
        terminal += int(len(array) % window != 0)
    return complete, terminal


def flatten_window(array: np.ndarray, start: int, window: int) -> np.ndarray:
    block = np.zeros((window, array.shape[1]), dtype=np.float32)
    piece = np.asarray(array[start:min(start + window, len(array))], dtype=np.float32)
    block[:len(piece)] = piece
    return block.reshape(-1)


@dataclass
class PreparedWindows:
    transformed: list[np.ndarray]
    fit_features: np.ndarray
    fit_ids: np.ndarray
    scaler: StandardScaler
    pca: PCA
    terminal_windows: int


def prepare_windows(arrays: Sequence[np.ndarray], window: int,
                    fit_sequences: Iterable[int], max_sample: int = 10_000,
                    seed: int = 111) -> PreparedWindows:
    fit_set = set(int(value) for value in fit_sequences)
    inventory, terminal = window_inventory(arrays, window)
    candidates = [(sequence, start) for sequence, start in inventory if sequence in fit_set]
    if not candidates:
        raise ValueError("no complete fitting windows")
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(candidates), size=min(max_sample, len(candidates)), replace=False))
    ids = np.asarray([candidates[index] for index in chosen], dtype=np.int32)
    width = window * arrays[0].shape[1]
    sample = np.empty((len(ids), width), dtype=np.float32)
    for row, (sequence, start) in enumerate(ids):
        sample[row] = flatten_window(arrays[int(sequence)], int(start), window)
    scaler = StandardScaler(copy=False).fit(sample)
    scaler.transform(sample)
    components = min(64, sample.shape[0], sample.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    fit_features = pca.fit_transform(sample).astype(np.float32, copy=False)
    transformed = []
    for array in arrays:
        starts = list(range(0, len(array), window))
        output = np.empty((len(starts), components), dtype=np.float32)
        for offset in range(0, len(starts), 256):
            batch_starts = starts[offset:offset + 256]
            batch = np.stack([flatten_window(array, start, window) for start in batch_starts])
            output[offset:offset + len(batch)] = pca.transform(scaler.transform(batch))
        transformed.append(output)
    return PreparedWindows(transformed, fit_features, ids, scaler, pca, terminal)


def fit_vocabulary(prepared: PreparedWindows, num_units: int, seed: int) -> KMeans:
    if num_units > len(prepared.fit_features):
        raise ValueError(f"K={num_units} exceeds fitting sample")
    model = KMeans(n_clusters=num_units, init="k-means++", n_init=5, max_iter=300,
                   tol=1e-4, algorithm="lloyd", random_state=seed)
    model.fit(prepared.fit_features)
    return model


def assign_windows(model: KMeans, transformed: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [model.predict(value).astype(np.int32) for value in transformed]


def window_metadata(data: Dataset, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions, subjects, sequences = [], [], []
    for sequence, (gt, subject) in enumerate(zip(data.gts, data.subjects)):
        for start in range(0, len(gt), window):
            actions.append(majority(gt[start:min(start + window, len(gt))]))
            subjects.append(subject)
            sequences.append(sequence)
    return np.asarray(actions), np.asarray(subjects), np.asarray(sequences, dtype=np.int32)


def segment_features(data: Dataset, window: int, mode: str,
                     transformed: Sequence[np.ndarray], num_units: int | None = None,
                     assignments: Sequence[np.ndarray] | None = None,
                     centers: np.ndarray | None = None, temperature: float | None = None):
    features, labels, lengths, seq_ids = [], [], [], []
    for sequence, gt in enumerate(data.gts):
        seg_lengths, seg_labels, starts = rle(gt)
        for length, label, start in zip(seg_lengths, seg_labels, starts):
            end = int(start + length)
            first, last = int(start // window), int((end - 1) // window)
            indices = np.arange(first, last + 1)
            weights = np.array([
                max(0, min(end, (index + 1) * window) - max(int(start), index * window))
                for index in indices
            ], dtype=np.float64)
            if mode == "continuous":
                value = np.average(transformed[sequence][indices], axis=0, weights=weights)
            elif mode == "hard":
                value = np.zeros(int(num_units), dtype=np.float64)
                np.add.at(value, assignments[sequence][indices], weights)
                value /= value.sum()
                value = np.sqrt(value)
            elif mode == "soft":
                x = transformed[sequence][indices]
                distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
                probabilities = softmax(-distances / float(temperature), axis=1)
                value = np.average(probabilities, axis=0, weights=weights)
                value /= value.sum()
                value = np.sqrt(value)
            else:
                raise ValueError(mode)
            features.append(value)
            labels.append(int(label))
            lengths.append(int(length))
            seq_ids.append(sequence)
    return (np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int32),
            np.asarray(lengths, dtype=np.int32), np.asarray(seq_ids, dtype=np.int32))


def segment_predictions(data: Dataset, segment_labels: np.ndarray) -> list[np.ndarray]:
    output, offset = [], 0
    for gt in data.gts:
        lengths, _, starts = rle(gt)
        prediction = np.empty(len(gt), dtype=np.int32)
        for length, start in zip(lengths, starts):
            prediction[start:start + length] = segment_labels[offset]
            offset += 1
        output.append(prediction)
    if offset != len(segment_labels):
        raise AssertionError("unused segment labels")
    return output


def cluster_segments(data: Dataset, features: np.ndarray, lengths: np.ndarray,
                     seed: int, weighted: bool = False):
    model = KMeans(n_clusters=data.num_actions, init="k-means++", n_init=5,
                   max_iter=300, tol=1e-4, algorithm="lloyd", random_state=seed)
    labels = model.fit_predict(features, sample_weight=lengths if weighted else None)
    predictions = segment_predictions(data, labels)
    return score_mapped(data.gts, predictions), labels, model


def base_temperature(model: KMeans, fit_features: np.ndarray) -> float | None:
    distances = model.transform(fit_features) ** 2
    nearest = np.partition(distances, 1, axis=1)[:, :2]
    gaps = nearest[:, 1] - nearest[:, 0]
    positive = gaps[gaps > 0]
    return float(np.median(positive)) if len(positive) else None


def supervised_metrics(true_segments: np.ndarray, pred_segments: np.ndarray,
                       lengths: np.ndarray, classes: np.ndarray) -> dict:
    recalls = recall_score(true_segments, pred_segments, labels=classes,
                           average=None, zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(true_segments, pred_segments) * 100),
        "frame_mof": float(np.average(true_segments == pred_segments, weights=lengths) * 100),
        "per_class_recall": json.dumps(dict(zip(map(str, classes), map(float, recalls)))),
        "confusion": json.dumps(confusion_matrix(true_segments, pred_segments,
                                                  labels=classes).tolist()),
    }


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_npz(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def elapsed(started: float) -> float:
    return float(time.perf_counter() - started)
