"""Complete 0.5x/2x soft-temperature sensitivities from cached fits."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from script.exp2round.q1q2.core import (
    OUT, SEEDS, VOCAB_SIZES, WINDOWS, base_temperature, cluster_segments,
    flatten_window, load_dataset, make_folds, save_npz, segment_features,
    supervised_metrics,
)
from script.exp2round.q1q2.run_cv import fit_readout, supports, write_readout


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def load_rows(path: Path):
    return [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []


def pooled_projection(data, window):
    path = OUT / "artifacts" / data.name / "latent" / f"w{window}_preprocess.npz"
    cache = np.load(path)
    scaler_mean, scaler_scale = cache["scaler_mean"], cache["scaler_scale"]
    pca_mean, components = cache["pca_mean"], cache["pca_components"]
    transformed = []
    for array in data.arrays:
        starts = list(range(0, len(array), window))
        result = np.empty((len(starts), len(components)), dtype=np.float32)
        for offset in range(0, len(starts), 256):
            selected = starts[offset:offset + 256]
            raw = np.stack([flatten_window(array, start, window) for start in selected])
            scaled = (raw - scaler_mean) / scaler_scale
            result[offset:offset + len(selected)] = (scaled - pca_mean) @ components.T
        transformed.append(result)
    return transformed


def run_pooled() -> None:
    path = OUT / "pooled_cells.jsonl"
    rows = load_rows(path)
    done = {(r.get("dataset"), r.get("window"), r.get("num_units"), r.get("seed"),
             r.get("readout"), r.get("temperature_multiplier")) for r in rows
            if r.get("representation") == "latent"}
    base = {(r["dataset"], r["window"], r["num_units"], r["seed"]):
            r.get("base_temperature") for r in rows
            if r.get("representation") == "latent" and r.get("readout") == "soft"
            and r.get("temperature_multiplier") == 1.0}
    for dataset in ("hugadb", "lara"):
        data = load_dataset(dataset)
        for window in (data.patch_size, data.patch_size // 4):
            transformed = pooled_projection(data, window)
            for num_units in VOCAB_SIZES:
                for seed in SEEDS:
                    artifact = np.load(
                        OUT / "artifacts" / dataset / "latent" /
                        f"w{window}_k{num_units}_s{seed}.npz", allow_pickle=True
                    )
                    centers = artifact["centers"]
                    temperature = base.get((dataset, window, num_units, seed))
                    if temperature is None:
                        continue
                    for multiplier in (0.5, 2.0):
                        key = (dataset, window, num_units, seed, "soft", multiplier)
                        if key in done:
                            continue
                        started = time.perf_counter()
                        features, _, lengths, _ = segment_features(
                            data, window, "soft", transformed, centers=centers,
                            temperature=temperature * multiplier,
                        )
                        metrics, _, _ = cluster_segments(data, features, lengths, seed)
                        append_jsonl(path, {
                            "dataset": dataset, "representation": "latent", "window": window,
                            "num_units": num_units, "seed": seed, "readout": "soft",
                            "temperature_multiplier": multiplier, **metrics,
                            "status": "complete", "base_temperature": temperature,
                            "runtime_seconds": time.perf_counter() - started,
                            "cache_identity": data.cache_identity,
                        })


def run_cv() -> None:
    rows_path = OUT / "supervised_readouts.jsonl"
    existing = load_rows(rows_path)
    done = {(r.get("dataset"), r.get("grouping"), r.get("window"), r.get("num_units"),
             r.get("seed"), r.get("readout"), r.get("temperature_multiplier"),
             r.get("fold"), r.get("level")) for r in existing}
    for dataset in ("hugadb", "lara"):
        data = load_dataset(dataset)
        classes = np.unique(np.concatenate(data.gts)).astype(np.int32)
        for grouping in ("recording", "subject"):
            groups = data.names if grouping == "recording" else data.subjects
            folds = np.asarray(make_folds(groups, [len(value) for value in data.gts]))
            for window in (data.patch_size, data.patch_size // 4):
                for num_units in VOCAB_SIZES:
                    for seed in SEEDS:
                        aggregates = {m: {"true": [], "pred": [], "lengths": [],
                                          "valid": [], "reasons": []} for m in (0.5, 2.0)}
                        for fold in range(4):
                            projected = np.load(
                                OUT / "artifacts" / dataset / "cv" / grouping /
                                f"w{window}_f{fold}_projected.npz", allow_pickle=True
                            )
                            transformed = [np.asarray(value) for value in projected["transformed"]]
                            fit_features = np.asarray(projected["fit_features"])
                            artifact = np.load(
                                OUT / "artifacts" / dataset / "cv" / grouping /
                                f"w{window}_k{num_units}_s{seed}_f{fold}.npz", allow_pickle=True
                            )
                            centers = artifact["centers"]
                            model = SimpleNamespace(
                                transform=lambda x, c=centers:
                                np.sqrt(((x[:, None, :] - c[None, :, :]) ** 2).sum(axis=2))
                            )
                            temperature = base_temperature(model, fit_features)
                            train_sequences = np.flatnonzero(folds != fold)
                            test_sequences = np.flatnonzero(folds == fold)
                            for multiplier in (0.5, 2.0):
                                key = (dataset, grouping, window, num_units, seed, "soft",
                                       multiplier, fold, "fold")
                                if key in done:
                                    continue
                                started = time.perf_counter()
                                features, labels, lengths, segment_seq = segment_features(
                                    data, window, "soft", transformed, centers=centers,
                                    temperature=temperature * multiplier,
                                )
                                train = np.isin(segment_seq, train_sequences)
                                test = np.isin(segment_seq, test_sequences)
                                pred, valid, reason = fit_readout(
                                    features[train], labels[train], features[test]
                                )
                                write_readout(
                                    rows_path, dataset, grouping, window, num_units, seed,
                                    "soft", multiplier, fold, labels[train], labels[test], pred,
                                    lengths[test], train_sequences, test_sequences, classes,
                                    "complete" if valid else "invalid", reason,
                                    time.perf_counter() - started, temperature,
                                )
                                if pred is not None:
                                    bucket = aggregates[multiplier]
                                    bucket["true"].append(labels[test]); bucket["pred"].append(pred)
                                    bucket["lengths"].append(lengths[test])
                                    bucket["valid"].append(valid); bucket["reasons"].append(reason)
                        for multiplier, bucket in aggregates.items():
                            key = (dataset, grouping, window, num_units, seed, "soft",
                                   multiplier, None, "pooled_oof")
                            if key in done or len(bucket["true"]) != 4:
                                continue
                            truth, pred, lengths = (np.concatenate(bucket[name])
                                                    for name in ("true", "pred", "lengths"))
                            append_jsonl(rows_path, {
                                "dataset": dataset, "grouping": grouping, "window": window,
                                "num_units": num_units, "seed": seed, "readout": "soft",
                                "temperature_multiplier": multiplier, "fold": None,
                                "level": "pooled_oof",
                                "status": "complete" if all(bucket["valid"]) else "invalid",
                                "reason": None if all(bucket["valid"]) else "; ".join(sorted({
                                    reason for reason in bucket["reasons"] if reason})),
                                "n_test_segments": len(truth),
                                "test_class_support": supports(truth, classes),
                                **supervised_metrics(truth, pred, lengths, classes),
                            })


if __name__ == "__main__":
    run_pooled()
    run_cv()
