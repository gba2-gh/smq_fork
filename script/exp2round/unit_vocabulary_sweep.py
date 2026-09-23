"""Independent latent-unit vocabulary sweep described in the exp2round brief.

The ``--gate1_only`` mode runs only W=patch, K_u=10.  It is deliberately an
early stopping point: the full grid must not be launched if this wiring check
is more than 1.5 MoF from the preregistered HuGaDB reference of 46.42.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.repro.d_error_decomposition import global_map, rle, score


SEEDS = (111, 222, 1538574472)
GATE1_REFERENCE = 46.42
GATE1_TOLERANCE = 1.5


def subject_id(dataset, name):
    if dataset == "hugadb":
        match = re.search(r"_(\d{2})_\d{2}(?:\.npy)?$", name)
        return match.group(1) if match else None
    if dataset == "lara":
        match = re.search(r"_(S\d+)_", name)
        return match.group(1) if match else None
    return None


def majority_label(values):
    labels, counts = np.unique(values, return_counts=True)
    return labels[np.argmax(counts)]


def majority_codes_on_gt_segments(gt, unit_frames):
    out = np.empty_like(unit_frames)
    lengths, _, starts = rle(gt)
    for start, length in zip(starts, lengths):
        end = start + length
        out[start:end] = majority_label(unit_frames[start:end])
    return out


def prepare_window_features(latents, window, pca_seed):
    """Standardise complete windows and project all assignment windows to PCA-64.

    Incomplete final windows are excluded from scaler/PCA fitting as required,
    but are zero-padded and transformed for assignment so frame predictions can
    still satisfy len(pred) == len(gt).
    """
    raw_windows = []
    complete_mask = []
    counts = []
    for latent in latents:
        latent = np.asarray(latent, dtype=np.float32)
        length = len(latent)
        complete = length // window
        sequence_windows = []
        if complete:
            sequence_windows.extend(
                latent[:complete * window].reshape(complete, -1)
            )
        complete_mask.extend([True] * complete)
        if length % window:
            padded = np.zeros((window,) + latent.shape[1:], dtype=np.float32)
            padded[:length % window] = latent[complete * window:]
            sequence_windows.append(padded.reshape(-1))
            complete_mask.append(False)
        raw_windows.extend(sequence_windows)
        counts.append(len(sequence_windows))

    raw = np.asarray(raw_windows, dtype=np.float32)
    complete_mask = np.asarray(complete_mask, dtype=bool)
    scaler = StandardScaler(copy=False)
    scaler.fit(raw[complete_mask])
    scaled = scaler.transform(raw)
    components = min(64, scaled[complete_mask].shape[0], scaled.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=pca_seed)
    pca.fit(scaled[complete_mask])
    projected = pca.transform(scaled).astype(np.float32, copy=False)
    return projected, complete_mask, counts, float(pca.explained_variance_ratio_.sum())


def split_by_counts(values, counts):
    result = []
    offset = 0
    for count in counts:
        result.append(values[offset:offset + count])
        offset += count
    if offset != len(values):
        raise AssertionError("window count does not match flattened values")
    return result


def expand_window_labels(labels_by_sequence, lengths, window):
    predictions = []
    for labels, length in zip(labels_by_sequence, lengths):
        prediction = np.repeat(labels, window)[:length]
        if len(prediction) != length:
            raise AssertionError("window labels do not cover the sequence")
        predictions.append(prediction.astype(np.int32, copy=False))
    return predictions


def segment_histogram_features(gts, unit_frames, num_units):
    features = []
    segment_lengths = []
    segment_subject_sequence = []
    for sequence_index, (gt, units) in enumerate(zip(gts, unit_frames)):
        if len(gt) != len(units):
            raise AssertionError("len(pred) != len(gt)")
        lengths, _, starts = rle(gt)
        for start, length in zip(starts, lengths):
            end = start + length
            values, counts = np.unique(units[start:end], return_counts=True)
            hist = np.zeros(num_units, dtype=np.float32)
            hist[values.astype(int)] = counts / length
            features.append(np.sqrt(hist))
            segment_lengths.append(int(length))
            segment_subject_sequence.append(sequence_index)
    return (
        np.stack(features),
        np.asarray(segment_lengths),
        np.asarray(segment_subject_sequence),
    )


def expand_segment_labels(gts, segment_labels):
    predictions = []
    index = 0
    for gt in gts:
        prediction = np.empty(len(gt), dtype=np.int32)
        lengths, _, starts = rle(gt)
        for start, length in zip(starts, lengths):
            prediction[start:start + length] = segment_labels[index]
            index += 1
        predictions.append(prediction)
    if index != len(segment_labels):
        raise AssertionError("segment labels were not consumed exactly once")
    return predictions


def score_histogram_readout(gts, unit_frames, num_units, num_actions, seed,
                            sample_weight=None):
    features, lengths, segment_sequences = segment_histogram_features(
        gts, unit_frames, num_units
    )
    model = KMeans(
        n_clusters=num_actions,
        n_init=5,
        max_iter=300,
        random_state=seed,
    )
    labels = model.fit_predict(features, sample_weight=sample_weight)
    predictions = expand_segment_labels(gts, labels)
    mapped = global_map(gts, predictions)
    return score(gts, mapped), labels, lengths, segment_sequences


def score_majority_readout(gts, unit_frames):
    predictions = [
        majority_codes_on_gt_segments(gt, units)
        for gt, units in zip(gts, unit_frames)
    ]
    return score(gts, global_map(gts, predictions))


def continuous_segment_features(gts, window_features, window):
    features = []
    lengths = []
    sequence_indices = []
    for sequence_index, (gt, sequence_features) in enumerate(zip(gts, window_features)):
        segment_lengths, _, starts = rle(gt)
        for start, length in zip(starts, segment_lengths):
            end = start + length
            indices = np.arange(start, end) // window
            features.append(sequence_features[indices].mean(axis=0))
            lengths.append(int(length))
            sequence_indices.append(sequence_index)
    return np.stack(features), np.asarray(lengths), np.asarray(sequence_indices)


def score_continuous_readout(gts, window_features, window, num_actions, seed):
    features, lengths, segment_sequences = continuous_segment_features(
        gts, window_features, window
    )
    model = KMeans(
        n_clusters=num_actions, n_init=5, max_iter=300, random_state=seed
    )
    labels = model.fit_predict(features)
    predictions = expand_segment_labels(gts, labels)
    return score(gts, global_map(gts, predictions)), labels, lengths, segment_sequences


def window_diagnostics(gts, window_labels, names, dataset, window):
    actions, units, subjects = [], [], []
    for gt, labels, name in zip(gts, window_labels, names):
        subject = subject_id(dataset, name)
        for index, unit in enumerate(labels):
            start = index * window
            end = min(start + window, len(gt))
            actions.append(majority_label(gt[start:end]))
            units.append(unit)
            subjects.append(subject)
    result = {
        "nmi_unit_action": float(normalized_mutual_info_score(actions, units)),
        "nmi_unit_subject": None,
    }
    if dataset != "babel1" and all(subject is not None for subject in subjects):
        result["nmi_unit_subject"] = float(
            normalized_mutual_info_score(subjects, units)
        )
    return result


def mean_windows_per_segment(gts, window):
    counts = []
    for gt in gts:
        lengths, _, starts = rle(gt)
        for start, length in zip(starts, lengths):
            end = start + length
            first = start // window
            last = (end - 1) // window
            counts.append(last - first + 1)
    return float(np.mean(counts))


def fit_unit_vocabulary(features, fit_mask, num_units, seed):
    kwargs = dict(n_clusters=num_units, n_init=5, max_iter=300, random_state=seed)
    if num_units >= 500:
        model = MiniBatchKMeans(batch_size=4096, **kwargs)
    else:
        model = KMeans(**kwargs)
    model.fit(features[fit_mask])
    return model.predict(features)


def make_cell(seed, window, num_units, readout, metrics, variance,
              units_per_segment, diagnostics, segment_subject_nmi,
              runtime_seconds, representation="pca64"):
    return {
        "seed": seed,
        "window": window,
        "num_units": num_units,
        "readout": readout,
        **metrics,
        "representation": representation,
        "pca_retained_variance": variance,
        "mean_units_per_gt_segment": units_per_segment,
        **diagnostics,
        "nmi_segment_cluster_subject": segment_subject_nmi,
        "runtime_seconds": runtime_seconds,
    }


def segment_subject_nmi(dataset, names, segment_sequences, labels):
    subjects = [subject_id(dataset, names[index]) for index in segment_sequences]
    if dataset == "babel1":
        return None
    if not all(value is not None for value in subjects):
        return None
    return float(normalized_mutual_info_score(subjects, labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gate1_only", action="store_true")
    parser.add_argument("--skip_no_pca_control", action="store_true")
    parser.add_argument("--windows", type=int, nargs="+", default=None,
                        help="Optional explicit window subset for chunked runs")
    parser.add_argument("--vocab_sizes", type=int, nargs="+", default=None,
                        help="Optional explicit K_u subset for chunked runs")
    args = parser.parse_args()

    started = time.perf_counter()
    dump = np.load(args.latents, allow_pickle=True)
    dataset = str(dump["dataset"])
    latents = [np.asarray(value) for value in dump["latent"]]
    gts = [np.asarray(value) for value in dump["gt"]]
    cached_preds = [np.asarray(value) for value in dump["pred"]]
    names = [str(value) for value in dump["names"]]
    patch_size = int(dump["patch_size"])
    num_actions = int(dump["num_actions"])

    majority_reference = score(
        gts,
        global_map(
            gts,
            [majority_codes_on_gt_segments(gt, pred)
             for gt, pred in zip(gts, cached_preds)],
        ),
    )
    result = {
        "dataset": dataset,
        "checkpoint": str(dump["ckpt"]),
        "latents": str(args.latents),
        "gate0": json.loads(str(dump["gate0_json"])),
        "d7_gt_boundaries_majority_code": majority_reference,
        "tail_policy": "incomplete windows excluded from fitting; zero-padded for assignment",
        "cells": [],
    }

    windows = ([patch_size] if args.gate1_only else
               (args.windows or [patch_size, patch_size // 2, patch_size // 4]))
    vocab_sizes = ([10] if args.gate1_only else
                   (args.vocab_sizes or [10, 20, 50, 100, 500, 1000]))
    for window in windows:
        features, fit_mask, counts, variance = prepare_window_features(
            latents, window, SEEDS[-1]
        )
        features_by_sequence = split_by_counts(features, counts)
        units_per_segment = mean_windows_per_segment(gts, window)

        if not args.gate1_only:
            for seed in SEEDS:
                cell_started = time.perf_counter()
                metrics, labels, _, segment_sequences = score_continuous_readout(
                    gts, features_by_sequence, window, num_actions, seed
                )
                result["cells"].append(make_cell(
                    seed, window, None, "continuous_mean", metrics, variance,
                    units_per_segment,
                    {"nmi_unit_action": None, "nmi_unit_subject": None},
                    segment_subject_nmi(dataset, names, segment_sequences, labels),
                    time.perf_counter() - cell_started,
                ))

        for num_units in vocab_sizes:
            if num_units > int(fit_mask.sum()):
                continue
            for seed in SEEDS:
                cell_started = time.perf_counter()
                labels = fit_unit_vocabulary(features, fit_mask, num_units, seed)
                labels_by_sequence = split_by_counts(labels, counts)
                unit_frames = expand_window_labels(
                    labels_by_sequence, [len(gt) for gt in gts], window
                )
                metrics, segment_labels, _, segment_sequences = score_histogram_readout(
                    gts, unit_frames, num_units, num_actions, seed
                )
                diagnostics = window_diagnostics(
                    gts, labels_by_sequence, names, dataset, window
                )
                result["cells"].append(make_cell(
                    seed, window, num_units, "histogram", metrics, variance,
                    units_per_segment, diagnostics,
                    segment_subject_nmi(dataset, names, segment_sequences, segment_labels),
                    time.perf_counter() - cell_started,
                ))

                if not args.gate1_only:
                    majority_started = time.perf_counter()
                    majority_metrics = score_majority_readout(gts, unit_frames)
                    result["cells"].append(make_cell(
                        seed, window, num_units, "majority_unit", majority_metrics,
                        variance, units_per_segment, diagnostics, None,
                        time.perf_counter() - majority_started,
                    ))

                    shuffle_started = time.perf_counter()
                    rng = np.random.default_rng(seed + 10_000_000)
                    probabilities = np.bincount(labels, minlength=num_units) / len(labels)
                    shuffled = rng.choice(num_units, size=len(labels), p=probabilities)
                    shuffled_by_sequence = split_by_counts(shuffled, counts)
                    shuffled_frames = expand_window_labels(
                        shuffled_by_sequence, [len(gt) for gt in gts], window
                    )
                    shuffled_metrics, shuffled_segment_labels, _, shuffled_sequences = (
                        score_histogram_readout(
                            gts, shuffled_frames, num_units, num_actions, seed
                        )
                    )
                    shuffled_diagnostics = window_diagnostics(
                        gts, shuffled_by_sequence, names, dataset, window
                    )
                    result["cells"].append(make_cell(
                        seed, window, num_units, "shuffled_histogram",
                        shuffled_metrics, variance, units_per_segment,
                        shuffled_diagnostics,
                        segment_subject_nmi(
                            dataset, names, shuffled_sequences, shuffled_segment_labels
                        ),
                        time.perf_counter() - shuffle_started,
                    ))

                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

        if (not args.gate1_only and window == patch_size
                and not args.skip_no_pca_control):
            # Required PCA sensitivity control: standardised raw windows,
            # K_u=100 at W=patch.  Reconstruct the standardised vectors once.
            raw_windows = []
            raw_complete = []
            raw_counts = []
            for latent in latents:
                complete = len(latent) // window
                sequence = list(np.asarray(latent[:complete * window], np.float32)
                                .reshape(complete, -1))
                raw_complete.extend([True] * complete)
                if len(latent) % window:
                    padded = np.zeros((window,) + latent.shape[1:], dtype=np.float32)
                    padded[:len(latent) % window] = latent[complete * window:]
                    sequence.append(padded.reshape(-1))
                    raw_complete.append(False)
                raw_windows.extend(sequence)
                raw_counts.append(len(sequence))
            raw_features = np.asarray(raw_windows, dtype=np.float32)
            raw_complete = np.asarray(raw_complete, dtype=bool)
            raw_scaler = StandardScaler(copy=False).fit(raw_features[raw_complete])
            raw_features = raw_scaler.transform(raw_features)
            for seed in SEEDS:
                cell_started = time.perf_counter()
                labels = fit_unit_vocabulary(raw_features, raw_complete, 100, seed)
                labels_by_sequence = split_by_counts(labels, raw_counts)
                unit_frames = expand_window_labels(
                    labels_by_sequence, [len(gt) for gt in gts], window
                )
                metrics, segment_labels, _, segment_sequences = score_histogram_readout(
                    gts, unit_frames, 100, num_actions, seed
                )
                diagnostics = window_diagnostics(
                    gts, labels_by_sequence, names, dataset, window
                )
                result["cells"].append(make_cell(
                    seed, window, 100, "histogram_no_pca", metrics, None,
                    units_per_segment, diagnostics,
                    segment_subject_nmi(dataset, names, segment_sequences, segment_labels),
                    time.perf_counter() - cell_started,
                    representation="standardized_raw",
                ))
            del raw_features, raw_windows

    if not args.gate1_only:
        primary = [cell for cell in result["cells"] if cell["readout"] == "histogram"]
        candidates = {}
        for cell in primary:
            candidates.setdefault((cell["window"], cell["num_units"]), []).append(cell["MoF"])
        best_window, best_num_units = max(
            candidates, key=lambda key: np.mean(candidates[key])
        )
        result["best_primary_cell"] = {
            "window": best_window,
            "num_units": best_num_units,
            "mean_mof": float(np.mean(candidates[(best_window, best_num_units)])),
        }
        features, fit_mask, counts, variance = prepare_window_features(
            latents, best_window, SEEDS[-1]
        )
        units_per_segment = mean_windows_per_segment(gts, best_window)
        for seed in SEEDS:
            cell_started = time.perf_counter()
            labels = fit_unit_vocabulary(features, fit_mask, best_num_units, seed)
            labels_by_sequence = split_by_counts(labels, counts)
            unit_frames = expand_window_labels(
                labels_by_sequence, [len(gt) for gt in gts], best_window
            )
            _, lengths, _ = segment_histogram_features(
                gts, unit_frames, best_num_units
            )
            metrics, segment_labels, _, segment_sequences = score_histogram_readout(
                gts, unit_frames, best_num_units, num_actions, seed,
                sample_weight=lengths,
            )
            diagnostics = window_diagnostics(
                gts, labels_by_sequence, names, dataset, best_window
            )
            result["cells"].append(make_cell(
                seed, best_window, best_num_units, "histogram_length_weighted",
                metrics, variance, units_per_segment, diagnostics,
                segment_subject_nmi(dataset, names, segment_sequences, segment_labels),
                time.perf_counter() - cell_started,
            ))

    if args.gate1_only:
        mofs = [cell["MoF"] for cell in result["cells"]]
        mean_mof = float(np.mean(mofs))
        result["gate1"] = {
            "reference_mof": GATE1_REFERENCE,
            "tolerance_mof": GATE1_TOLERANCE,
            "observed_mof_by_seed": mofs,
            "observed_mean_mof": mean_mof,
            "absolute_difference": abs(mean_mof - GATE1_REFERENCE),
            "passed": abs(mean_mof - GATE1_REFERENCE) <= GATE1_TOLERANCE,
        }

    result["runtime_seconds"] = time.perf_counter() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if args.gate1_only and not result["gate1"]["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
