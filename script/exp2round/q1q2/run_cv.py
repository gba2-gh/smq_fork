"""Cross-validated supervised readout and held-subject unit-reuse diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from script.exp2round.q1q2.core import (
    OUT, SEEDS, VOCAB_SIZES, WINDOWS, assign_windows, base_temperature,
    fit_vocabulary, load_dataset, make_folds, prepare_windows, save_npz,
    segment_features, supervised_metrics,
)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def fit_readout(x_train, y_train, x_test):
    present = sorted(set(map(int, np.unique(y_train))))
    if len(present) < 2:
        return None, False, "fewer than two training classes"
    model = LogisticRegression(
        penalty="l2", C=1.0, fit_intercept=True, solver="lbfgs",
        max_iter=2000, tol=1e-4, class_weight="balanced", multi_class="auto",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    if not converged:
        return model.predict(x_test), False, "logistic regression did not converge"
    return model.predict(x_test), True, None


def supports(labels: np.ndarray, classes: np.ndarray) -> str:
    values = {str(int(label)): int(np.count_nonzero(labels == label)) for label in classes}
    return json.dumps(values, separators=(",", ":"))


def unit_reuse_rows(dataset, grouping, window, num_units, seed, fold, data,
                    assignments, test_sequences):
    held_subjects = sorted({data.subjects[index] for index in test_sequences})
    rates = np.zeros((num_units, len(held_subjects)), dtype=np.float64)
    for subject_index, subject in enumerate(held_subjects):
        sequences = [index for index in test_sequences if data.subjects[index] == subject]
        total = 0
        counts = np.zeros(num_units, dtype=np.int64)
        for sequence in sequences:
            complete = len(data.arrays[sequence]) // window
            values = assignments[sequence][:complete]
            counts += np.bincount(values, minlength=num_units)
            total += complete
        if total:
            rates[:, subject_index] = counts / total
    used_counts = np.count_nonzero(rates > 0, axis=1)
    threshold = int(np.ceil(len(held_subjects) / 2))
    rows = []
    for unit in range(num_units):
        if rates[unit].sum() == 0:
            entropy = normalized = None
        else:
            probability = rates[unit] / rates[unit].sum()
            positive = probability[probability > 0]
            entropy = float(-(positive * np.log(positive)).sum())
            normalized = entropy / np.log(len(held_subjects)) if len(held_subjects) > 1 else None
        rows.append({
            "dataset": dataset, "grouping": grouping, "window": window,
            "num_units": num_units, "seed": seed, "fold": fold, "unit": unit,
            "n_held_subjects": len(held_subjects), "half_subject_threshold": threshold,
            "subjects_using_unit": int(used_counts[unit]),
            "used_by_at_least_half": bool(used_counts[unit] >= threshold),
            "subject_distribution_entropy": entropy,
            "normalized_subject_distribution_entropy": normalized,
            "unused": bool(used_counts[unit] == 0),
        })
    return rows


def write_readout(rows_path, dataset, grouping, window, num_units, seed, readout,
                  multiplier, fold, y_train, y_test, pred, lengths_test,
                  train_sequences, test_sequences, classes, status, reason,
                  runtime, temperature=None):
    row = {
        "dataset": dataset, "grouping": grouping, "window": window,
        "num_units": num_units, "seed": seed, "readout": readout,
        "temperature_multiplier": multiplier, "fold": fold,
        "level": "fold", "status": status, "reason": reason,
        "n_train_segments": len(y_train), "n_test_segments": len(y_test),
        "n_train_recordings": len(train_sequences),
        "n_test_recordings": len(test_sequences),
        "train_class_support": supports(y_train, classes),
        "test_class_support": supports(y_test, classes),
        "missing_train_classes": json.dumps(sorted(set(map(int, classes)) - set(map(int, y_train)))),
        "runtime_seconds": runtime, "base_temperature": temperature,
    }
    if pred is not None:
        row.update(supervised_metrics(y_test, pred, lengths_test, classes))
    append_jsonl(rows_path, row)


def run_configuration(dataset: str, grouping: str, window: int, num_units: int,
                      seed: int, multipliers: list[float], continuous: bool,
                      deadline: float | None) -> None:
    data = load_dataset(dataset)
    groups = data.names if grouping == "recording" else data.subjects
    sequence_folds = np.asarray(make_folds(groups, [len(value) for value in data.gts]))
    fold_manifest = OUT / "artifacts" / dataset / "folds"
    save_npz(fold_manifest / f"{grouping}.npz", names=np.asarray(data.names),
             subjects=np.asarray(data.subjects), folds=sequence_folds)
    rows_path = OUT / "supervised_readouts.jsonl"
    reuse_path = OUT / "unit_reuse.jsonl"
    classes = np.unique(np.concatenate(data.gts)).astype(np.int32)
    if len(classes) != data.num_actions:
        raise ValueError(f"{dataset}: observed {len(classes)} classes, expected {data.num_actions}")
    aggregate: dict[tuple, dict[str, list]] = {}
    for fold in range(4):
        if deadline is not None and time.time() >= deadline:
            return
        started = time.perf_counter()
        train_sequences = np.flatnonzero(sequence_folds != fold)
        test_sequences = np.flatnonzero(sequence_folds == fold)
        prep_path = OUT / "artifacts" / dataset / "cv" / grouping / f"w{window}_f{fold}.npz"
        projected_path = OUT / "artifacts" / dataset / "cv" / grouping / \
            f"w{window}_f{fold}_projected.npz"
        if projected_path.exists():
            cached = np.load(projected_path, allow_pickle=True)
            prepared = SimpleNamespace(
                transformed=[np.asarray(value) for value in cached["transformed"]],
                fit_features=np.asarray(cached["fit_features"]),
                fit_ids=np.asarray(cached["fit_ids"]),
                terminal_windows=int(cached["terminal_windows"]),
            )
        else:
            prepared = prepare_windows(data.arrays, window, train_sequences)
            save_npz(prep_path, fit_ids=prepared.fit_ids, scaler_mean=prepared.scaler.mean_,
                     scaler_scale=prepared.scaler.scale_, pca_components=prepared.pca.components_,
                     pca_mean=prepared.pca.mean_,
                     pca_explained_variance_ratio=prepared.pca.explained_variance_ratio_)
            object_transformed = np.empty(len(prepared.transformed), dtype=object)
            object_transformed[:] = prepared.transformed
            save_npz(projected_path, fit_ids=prepared.fit_ids,
                     fit_features=prepared.fit_features,
                     transformed=object_transformed,
                     terminal_windows=prepared.terminal_windows)
        if continuous:
            features, labels, lengths, segment_seq = segment_features(
                data, window, "continuous", prepared.transformed
            )
            train = np.isin(segment_seq, train_sequences)
            test = np.isin(segment_seq, test_sequences)
            pred, valid, reason = fit_readout(features[train], labels[train], features[test])
            write_readout(rows_path, dataset, grouping, window, None, None,
                          "continuous", None, fold, labels[train], labels[test], pred,
                          lengths[test], train_sequences, test_sequences, classes,
                          "complete" if valid else "invalid", reason,
                          time.perf_counter() - started)
            if pred is not None:
                key = ("continuous", None)
                bucket = aggregate.setdefault(key, {"true": [], "pred": [], "lengths": [],
                                                    "valid": [], "reasons": []})
                bucket["true"].append(labels[test]); bucket["pred"].append(pred)
                bucket["lengths"].append(lengths[test])
                bucket["valid"].append(valid); bucket["reasons"].append(reason)
        vocabulary = fit_vocabulary(prepared, num_units, seed)
        assignments = assign_windows(vocabulary, prepared.transformed)
        artifact = OUT / "artifacts" / dataset / "cv" / grouping / \
            f"w{window}_k{num_units}_s{seed}_f{fold}.npz"
        object_assignments = np.empty(len(assignments), dtype=object)
        object_assignments[:] = assignments
        save_npz(artifact, centers=vocabulary.cluster_centers_, assignments=object_assignments,
                 inertia=vocabulary.inertia_, n_iter=vocabulary.n_iter_)
        hard, labels, lengths, segment_seq = segment_features(
            data, window, "hard", prepared.transformed, num_units, assignments
        )
        train = np.isin(segment_seq, train_sequences)
        test = np.isin(segment_seq, test_sequences)
        pred, valid, reason = fit_readout(hard[train], labels[train], hard[test])
        write_readout(rows_path, dataset, grouping, window, num_units, seed, "hard", None,
                      fold, labels[train], labels[test], pred, lengths[test],
                      train_sequences, test_sequences, classes,
                      "complete" if valid else "invalid", reason,
                      time.perf_counter() - started)
        if pred is not None:
            bucket = aggregate.setdefault(("hard", None),
                                          {"true": [], "pred": [], "lengths": [],
                                           "valid": [], "reasons": []})
            bucket["true"].append(labels[test]); bucket["pred"].append(pred)
            bucket["lengths"].append(lengths[test])
            bucket["valid"].append(valid); bucket["reasons"].append(reason)
        temperature = base_temperature(vocabulary, prepared.fit_features)
        if temperature is not None:
            for multiplier in multipliers:
                soft, _, _, _ = segment_features(
                    data, window, "soft", prepared.transformed,
                    centers=vocabulary.cluster_centers_, temperature=temperature * multiplier
                )
                pred, valid, reason = fit_readout(soft[train], labels[train], soft[test])
                write_readout(rows_path, dataset, grouping, window, num_units, seed, "soft",
                              multiplier, fold, labels[train], labels[test], pred,
                              lengths[test], train_sequences, test_sequences, classes,
                              "complete" if valid else "invalid", reason,
                              time.perf_counter() - started, temperature)
                if pred is not None:
                    bucket = aggregate.setdefault(("soft", multiplier),
                                                  {"true": [], "pred": [], "lengths": [],
                                                   "valid": [], "reasons": []})
                    bucket["true"].append(labels[test]); bucket["pred"].append(pred)
                    bucket["lengths"].append(lengths[test])
                    bucket["valid"].append(valid); bucket["reasons"].append(reason)
        else:
            for multiplier in multipliers:
                write_readout(rows_path, dataset, grouping, window, num_units, seed, "soft",
                              multiplier, fold, labels[train], labels[test], None,
                              lengths[test], train_sequences, test_sequences, classes,
                              "invalid", "no positive distance gap",
                              time.perf_counter() - started)
        if grouping == "subject":
            for row in unit_reuse_rows(dataset, grouping, window, num_units, seed,
                                       fold, data, assignments, test_sequences):
                append_jsonl(reuse_path, row)
    for (readout, multiplier), bucket in aggregate.items():
        truth = np.concatenate(bucket["true"])
        prediction = np.concatenate(bucket["pred"])
        lengths = np.concatenate(bucket["lengths"])
        row = {
            "dataset": dataset, "grouping": grouping, "window": window,
            "num_units": None if readout == "continuous" else num_units,
            "seed": None if readout == "continuous" else seed,
            "readout": readout, "temperature_multiplier": multiplier,
            "fold": None, "level": "pooled_oof",
            "status": "complete" if all(bucket["valid"]) else "invalid",
            "reason": None if all(bucket["valid"]) else
                      "; ".join(sorted({reason for reason in bucket["reasons"] if reason})),
            "n_test_segments": len(truth),
            "test_class_support": supports(truth, classes),
            **supervised_metrics(truth, prediction, lengths, classes),
        }
        append_jsonl(rows_path, row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["hugadb", "lara"])
    parser.add_argument("--groupings", nargs="+", default=["recording", "subject"])
    parser.add_argument("--windows", nargs="+", type=int)
    parser.add_argument("--vocab_sizes", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--temperatures", nargs="+", type=float, default=[1.0])
    parser.add_argument("--skip_continuous", action="store_true")
    parser.add_argument("--deadline_epoch", type=float)
    args = parser.parse_args()
    for dataset in args.datasets:
        windows = args.windows or [WINDOWS[dataset][0], WINDOWS[dataset][-1]]
        for grouping in args.groupings:
            for window in windows:
                first = True
                for num_units in (args.vocab_sizes or list(VOCAB_SIZES)):
                    for seed in args.seeds:
                        run_configuration(dataset, grouping, window, num_units, seed,
                                          args.temperatures,
                                          continuous=(first and not args.skip_continuous),
                                          deadline=args.deadline_epoch)
                        first = False


if __name__ == "__main__":
    main()
