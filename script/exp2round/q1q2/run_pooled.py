"""Run pooled Q1/Q2 information, clustering, and restricted representation controls."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from script.exp2round.q1q2.core import (
    OUT, SEEDS, VOCAB_SIZES, WINDOWS, Dataset, assign_windows, atomic_json,
    base_temperature, cluster_segments, elapsed, fit_vocabulary,
    information_values, load_dataset, majority, map_predictions,
    permutation_references, prepare_windows, recording_zscore, rle, save_npz,
    score, score_mapped, segment_features, segment_predictions, window_metadata,
)
from sklearn.cluster import KMeans


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def completed_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    keys = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            keys.add(tuple(row.get(name) for name in
                           ("dataset", "representation", "window", "num_units",
                            "seed", "readout", "temperature_multiplier")))
    return keys


def cell_key(row: dict) -> tuple:
    return tuple(row.get(name) for name in
                 ("dataset", "representation", "window", "num_units",
                  "seed", "readout", "temperature_multiplier"))


def flatten_assignments(assignments):
    return np.concatenate(assignments).astype(np.int32, copy=False)


def majority_unit_predictions(data: Dataset, window: int, assignments):
    features, _, _, _ = segment_features(data, window, "hard", [],
                                         num_units=max(map(np.max, assignments)) + 1,
                                         assignments=assignments)
    return segment_predictions(data, np.argmax(features, axis=1).astype(np.int32))


def permutation_assignments(assignments, seed: int):
    lengths = [len(value) for value in assignments]
    flat = flatten_assignments(assignments).copy()
    np.random.default_rng(seed + 10_000_000).shuffle(flat)
    result, offset = [], 0
    for length in lengths:
        result.append(flat[offset:offset + length])
        offset += length
    return result


def within_subject_score(data: Dataset, features: np.ndarray, segment_seq: np.ndarray,
                         seed: int):
    segment_subjects = np.asarray([data.subjects[index] for index in segment_seq])
    labels = np.full(len(features), -1, dtype=np.int32)
    unavailable = []
    for subject in np.unique(segment_subjects):
        idx = np.flatnonzero(segment_subjects == subject)
        if len(idx) < data.num_actions:
            unavailable.append(str(subject))
            continue
        model = KMeans(n_clusters=data.num_actions, init="k-means++", n_init=5,
                       max_iter=300, tol=1e-4, algorithm="lloyd", random_state=seed)
        labels[idx] = model.fit_predict(features[idx])
    raw_predictions = segment_predictions(data, labels)
    mapped = [None] * len(data.names)
    for subject in np.unique(data.subjects):
        seq_idx = [i for i, value in enumerate(data.subjects) if value == subject]
        if subject in unavailable:
            continue
        subject_mapped = map_predictions([data.gts[i] for i in seq_idx],
                                         [raw_predictions[i] for i in seq_idx], "hungarian")
        for index, prediction in zip(seq_idx, subject_mapped):
            mapped[index] = prediction
    available = [index for index, value in enumerate(mapped) if value is not None]
    metrics = score([data.gts[index] for index in available],
                    [mapped[index] for index in available]) if available else {}
    return metrics, sum(len(data.gts[index]) for index in available) / sum(map(len, data.gts)), unavailable


def save_preprocessing(dataset: str, representation: str, window: int, prepared) -> None:
    path = OUT / "artifacts" / dataset / representation / f"w{window}_preprocess.npz"
    save_npz(path, fit_ids=prepared.fit_ids, scaler_mean=prepared.scaler.mean_,
             scaler_scale=prepared.scaler.scale_, pca_components=prepared.pca.components_,
             pca_mean=prepared.pca.mean_,
             pca_explained_variance_ratio=prepared.pca.explained_variance_ratio_)


def run(dataset: str, representation: str, windows: list[int], vocab_sizes: list[int],
        seeds: list[int], deadline: float | None, temperature_multipliers: list[float]) -> None:
    gate_path = OUT / f"gate_{dataset}.json"
    if not gate_path.exists() or not json.loads(gate_path.read_text())["passed"]:
        raise RuntimeError(f"{dataset} provenance gate has not passed")
    data = load_dataset(dataset, "raw" if representation == "raw" else "latent")
    if representation == "per_recording":
        data.arrays = recording_zscore(data.arrays)
    output = OUT / "pooled_cells.jsonl"
    information_output = OUT / "information.jsonl"
    completed = completed_keys(output)
    restricted = representation != "latent"
    for window in windows:
        if deadline is not None and time.time() >= deadline:
            return
        prep_started = time.perf_counter()
        prepared = prepare_windows(data.arrays, window, range(len(data.names)))
        save_preprocessing(dataset, representation, window, prepared)
        actions, subjects, _ = window_metadata(data, window)
        if sum(len(value) for value in prepared.transformed) != len(actions):
            raise AssertionError("window metadata and features differ")
        if not restricted:
            continuous, _, lengths, _ = segment_features(
                data, window, "continuous", prepared.transformed
            )
            for seed in seeds:
                row = {"dataset": dataset, "representation": representation,
                       "window": window, "num_units": None, "seed": seed,
                       "readout": "continuous", "temperature_multiplier": None}
                if cell_key(row) not in completed:
                    started = time.perf_counter()
                    metrics, labels, _ = cluster_segments(data, continuous, lengths, seed)
                    row.update(metrics)
                    row.update(status="complete", runtime_seconds=elapsed(started),
                               preprocessing_seconds=elapsed(prep_started),
                               n_segments=len(continuous), pca_dim=continuous.shape[1],
                               explained_variance=float(prepared.pca.explained_variance_ratio_.sum()),
                               cache_identity=data.cache_identity)
                    append_jsonl(output, row); completed.add(cell_key(row))
        for num_units in vocab_sizes:
            for seed in seeds:
                if deadline is not None and time.time() >= deadline:
                    return
                hard_key = (dataset, representation, window, num_units, seed, "hard", None)
                artifact = OUT / "artifacts" / dataset / representation / f"w{window}_k{num_units}_s{seed}.npz"
                started = time.perf_counter()
                vocabulary = fit_vocabulary(prepared, num_units, seed)
                assignments = assign_windows(vocabulary, prepared.transformed)
                flat = flatten_assignments(assignments)
                info = information_values(flat, actions, subjects)
                references = permutation_references(flat, actions, subjects)
                info_row = {"dataset": dataset, "representation": representation,
                            "window": window, "num_units": num_units, "seed": seed,
                            **info}
                for metric, (mean, sd) in references.items():
                    info_row[f"{metric}_permutation_mean"] = mean
                    info_row[f"{metric}_permutation_sd"] = sd
                info_row.update(terminal_windows=prepared.terminal_windows,
                                status="complete", cache_identity=data.cache_identity)
                append_jsonl(information_output, info_row)
                object_assignments = np.empty(len(assignments), dtype=object)
                object_assignments[:] = assignments
                save_npz(artifact, centers=vocabulary.cluster_centers_,
                         assignments=object_assignments, inertia=vocabulary.inertia_,
                         n_iter=vocabulary.n_iter_)
                hard, _, lengths, segment_seq = segment_features(
                    data, window, "hard", prepared.transformed, num_units, assignments
                )
                if hard_key not in completed:
                    metrics, segment_labels, _ = cluster_segments(data, hard, lengths, seed)
                    row = {"dataset": dataset, "representation": representation,
                           "window": window, "num_units": num_units, "seed": seed,
                           "readout": "hard", "temperature_multiplier": None, **metrics,
                           "status": "complete", "runtime_seconds": elapsed(started),
                           "n_iter": int(vocabulary.n_iter_), "inertia": float(vocabulary.inertia_),
                           "fit_windows": len(prepared.fit_features), "all_windows": len(flat),
                           "terminal_windows": prepared.terminal_windows,
                           "pca_dim": prepared.fit_features.shape[1],
                           "explained_variance": float(prepared.pca.explained_variance_ratio_.sum()),
                           "n_segments": len(hard), "cache_identity": data.cache_identity}
                    append_jsonl(output, row); completed.add(cell_key(row))
                if restricted:
                    continue
                # Duration-weighted segment clustering.
                weighted_row = {"dataset": dataset, "representation": representation,
                                "window": window, "num_units": num_units, "seed": seed,
                                "readout": "hard_length_weighted", "temperature_multiplier": None}
                if cell_key(weighted_row) not in completed:
                    weighted_metrics, _, _ = cluster_segments(data, hard, lengths, seed, weighted=True)
                    weighted_row.update(weighted_metrics, status="complete",
                                        runtime_seconds=elapsed(started), cache_identity=data.cache_identity)
                    append_jsonl(output, weighted_row); completed.add(cell_key(weighted_row))
                # Majority-unit mappings.
                majority_predictions = segment_predictions(data, np.argmax(hard, axis=1).astype(np.int32))
                for mapping in ("hungarian", "many_to_one"):
                    majority_row = {"dataset": dataset, "representation": representation,
                                    "window": window, "num_units": num_units, "seed": seed,
                                    "readout": f"majority_unit_{mapping}",
                                    "temperature_multiplier": None}
                    if cell_key(majority_row) not in completed:
                        majority_row.update(score_mapped(data.gts, majority_predictions, mapping),
                                            status="complete", runtime_seconds=elapsed(started),
                                            cache_identity=data.cache_identity)
                        append_jsonl(output, majority_row); completed.add(cell_key(majority_row))
                # Count-preserving permutation control.
                permuted = permutation_assignments(assignments, seed)
                perm_features, _, perm_lengths, _ = segment_features(
                    data, window, "hard", prepared.transformed, num_units, permuted
                )
                perm_row = {"dataset": dataset, "representation": representation,
                            "window": window, "num_units": num_units, "seed": seed,
                            "readout": "permuted_hard", "temperature_multiplier": None}
                if cell_key(perm_row) not in completed:
                    perm_metrics, _, _ = cluster_segments(data, perm_features, perm_lengths, seed)
                    perm_row.update(perm_metrics, status="complete", runtime_seconds=elapsed(started),
                                    cache_identity=data.cache_identity)
                    append_jsonl(output, perm_row); completed.add(cell_key(perm_row))
                # Soft pooled readouts are only specified at full and quarter patch.
                if window in (data.patch_size, data.patch_size // 4):
                    temperature = base_temperature(vocabulary, prepared.fit_features)
                    for multiplier in temperature_multipliers:
                        soft_row = {"dataset": dataset, "representation": representation,
                                    "window": window, "num_units": num_units, "seed": seed,
                                    "readout": "soft", "temperature_multiplier": multiplier}
                        if cell_key(soft_row) in completed:
                            continue
                        if temperature is None:
                            soft_row.update(status="invalid", reason="no positive distance gap")
                        else:
                            soft, _, soft_lengths, _ = segment_features(
                                data, window, "soft", prepared.transformed,
                                centers=vocabulary.cluster_centers_,
                                temperature=temperature * multiplier,
                            )
                            soft_metrics, _, _ = cluster_segments(data, soft, soft_lengths, seed)
                            soft_row.update(soft_metrics, status="complete",
                                            base_temperature=temperature,
                                            runtime_seconds=elapsed(started),
                                            cache_identity=data.cache_identity)
                        append_jsonl(output, soft_row); completed.add(cell_key(soft_row))
                # Within-subject diagnostic only for restricted Task-D cells.
                if num_units in (10, 100, 1000) and window in (data.patch_size, data.patch_size // 4):
                    within_row = {"dataset": dataset, "representation": representation,
                                  "window": window, "num_units": num_units, "seed": seed,
                                  "readout": "within_subject", "temperature_multiplier": None}
                    if cell_key(within_row) not in completed:
                        within_metrics, coverage, unavailable = within_subject_score(
                            data, hard, segment_seq, seed
                        )
                        within_row.update(within_metrics, status="complete",
                                          frame_coverage=coverage,
                                          unavailable_subjects=json.dumps(unavailable),
                                          runtime_seconds=elapsed(started),
                                          cache_identity=data.cache_identity)
                        append_jsonl(output, within_row); completed.add(cell_key(within_row))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["hugadb", "lara"])
    parser.add_argument("--representations", nargs="+", default=["latent"],
                        choices=["latent", "raw", "per_recording"])
    parser.add_argument("--windows", nargs="+", type=int)
    parser.add_argument("--vocab_sizes", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--temperatures", nargs="+", type=float, default=[1.0])
    parser.add_argument("--deadline_epoch", type=float)
    args = parser.parse_args()
    for dataset in args.datasets:
        for representation in args.representations:
            windows = args.windows or list(WINDOWS[dataset])
            vocab = args.vocab_sizes or list(VOCAB_SIZES)
            if representation != "latent":
                windows = [value for value in windows
                           if value in (WINDOWS[dataset][0], WINDOWS[dataset][-1])]
                vocab = [value for value in vocab if value in (10, 100, 1000)]
            run(dataset, representation, windows, vocab, args.seeds,
                args.deadline_epoch, args.temperatures)


if __name__ == "__main__":
    main()
