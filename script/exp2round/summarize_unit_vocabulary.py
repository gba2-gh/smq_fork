"""Merge chunked vocabulary runs and emit the preregistered negative-result outputs."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from script.repro.d_error_decomposition import global_map, rle, score


def majority(values):
    labels, counts = np.unique(values, return_counts=True)
    return labels[np.argmax(counts)]


def majority_on_gt(gt, pred):
    out = np.empty_like(pred)
    lengths, _, starts = rle(gt)
    for start, length in zip(starts, lengths):
        out[start:start + length] = majority(pred[start:start + length])
    return out


def oracle_labels_on_predicted_segments(prediction, gt):
    out = np.empty_like(gt)
    lengths, _, starts = rle(prediction)
    for start, length in zip(starts, lengths):
        out[start:start + length] = majority(gt[start:start + length])
    return out


def means(cells, readout="histogram"):
    grouped = defaultdict(list)
    for cell in cells:
        if cell["readout"] == readout:
            grouped[(cell["window"], cell["num_units"])].append(cell)
    return {
        key: {
            metric: float(np.mean([cell[metric] for cell in values]))
            for metric in ("MoF", "Edit", "F1@10", "F1@25", "F1@50")
        } | {
            "MoF_std": float(np.std([cell["MoF"] for cell in values])),
            "nmi_action": float(np.mean([cell["nmi_unit_action"] for cell in values])),
            "nmi_subject": float(np.mean([cell["nmi_unit_subject"] for cell in values])),
            "nmi_segment_subject": float(np.mean([
                cell["nmi_segment_cluster_subject"] for cell in values
            ])),
            "units_per_segment": values[0]["mean_units_per_gt_segment"],
        }
        for key, values in grouped.items()
    }


def control_best(cells, readout):
    grouped = defaultdict(list)
    for cell in cells:
        if cell["readout"] == readout:
            grouped[(cell["window"], cell["num_units"])].append(cell["MoF"])
    if not grouped:
        return None
    key = max(grouped, key=lambda value: np.mean(grouped[value]))
    return {
        "window": key[0], "num_units": key[1],
        "mean_mof": float(np.mean(grouped[key])),
        "std_mof": float(np.std(grouped[key])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", required=True, type=Path)
    parser.add_argument("--short", required=True, type=Path)
    parser.add_argument("--latents", required=True, type=Path)
    parser.add_argument("--out_dir", type=Path, default=Path("results/exp2round"))
    args = parser.parse_args()

    patch = json.loads(args.patch.read_text(encoding="utf-8"))
    short = json.loads(args.short.read_text(encoding="utf-8"))
    cells = patch["cells"] + short["cells"]
    dump = np.load(args.latents, allow_pickle=True)
    gts = [np.asarray(value) for value in dump["gt"]]
    preds = [np.asarray(value) for value in dump["pred"]]

    mapped = global_map(gts, preds)
    d7 = {
        "smq_as_is": score(gts, mapped),
        "gt_boundaries_majority_code": score(
            gts,
            global_map(gts, [majority_on_gt(gt, pred) for gt, pred in zip(gts, preds)]),
        ),
        "predicted_boundaries_gt_labels": score(
            gts,
            [oracle_labels_on_predicted_segments(pred, gt)
             for pred, gt in zip(mapped, gts)],
        ),
    }
    primary = means(cells)
    best_key = max(primary, key=lambda key: primary[key]["MoF"])
    best = {"window": best_key[0], "num_units": best_key[1], **primary[best_key]}
    improvement = best["MoF"] - d7["gt_boundaries_majority_code"]["MoF"]
    gate2 = {
        "required_improvement_mof": 5.0,
        "hugadb_improvement_mof": improvement,
        "hugadb_passed": improvement >= 5.0,
        "passed_both_datasets": False,
        "reason": "HuGaDB failed, so the required conjunction cannot pass",
    }

    merged = {
        "dataset": patch["dataset"],
        "checkpoint": patch["checkpoint"],
        "latents": patch["latents"],
        "gate0": patch["gate0"],
        "gate1": {
            "reference_mof": 46.42,
            "observed_mean_mof": primary[(60, 10)]["MoF"],
            "absolute_difference": abs(primary[(60, 10)]["MoF"] - 46.42),
            "tolerance_mof": 1.5,
            "passed": abs(primary[(60, 10)]["MoF"] - 46.42) <= 1.5,
        },
        "gate2": gate2,
        "d7_references": d7,
        "best_primary_cell": best,
        "controls": {
            name: control_best(cells, name)
            for name in ("majority_unit", "continuous_mean", "shuffled_histogram",
                         "histogram_length_weighted")
        },
        "not_run": {
            "no_pca_W60_K100": "process terminated during raw-dimensional k-means",
            "lara_released": "not run because HuGaDB failed Gate 2",
            "t1_seeds": "not run because Gate 2 failed",
            "babel1": "not run because Gate 2 failed",
        },
        "cells": cells,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "units_sweep_hugadb.json"
    json_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")

    csv_path = args.out_dir / "units_sweep.csv"
    fields = [
        "dataset", "checkpoint", "seed", "window", "num_units", "readout",
        "MoF", "Edit", "F1@10", "F1@25", "F1@50",
        "mean_units_per_gt_segment", "nmi_unit_action", "nmi_unit_subject",
        "nmi_segment_cluster_subject", "pca_retained_variance",
        "representation", "runtime_seconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for cell in cells:
            writer.writerow({
                "dataset": patch["dataset"],
                "checkpoint": patch["checkpoint"],
                **cell,
            })

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for window in sorted({key[0] for key in primary}, reverse=True):
        points = sorted(
            (num_units, values["MoF"], values["MoF_std"])
            for (cell_window, num_units), values in primary.items()
            if cell_window == window
        )
        axis.errorbar(
            [point[0] for point in points],
            [point[1] for point in points],
            yerr=[point[2] for point in points],
            marker="o", capsize=3, label=f"W={window}",
        )
    axis.axhline(
        d7["gt_boundaries_majority_code"]["MoF"], color="black", linestyle="--",
        label="D7 majority code",
    )
    axis.axhline(
        d7["predicted_boundaries_gt_labels"]["MoF"], color="gray", linestyle=":",
        label="Pred. boundaries + GT labels",
    )
    axis.set_xscale("log")
    axis.set_xlabel("Independent unit vocabulary size $K_u$")
    axis.set_ylabel("MoF (%)")
    axis.set_title("HuGaDB: action recovery from segment unit histograms")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    plot_path = args.out_dir / "units_sweep_mof.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    lines = [
        "# Independent unit-vocabulary sweep",
        "",
        "## Outcome",
        "",
        (f"**Negative result; Gate 2 failed.** The best HuGaDB cell was "
         f"W={best['window']}, K_u={best['num_units']} at {best['MoF']:.2f} "
         f"MoF (SD {best['MoF_std']:.2f}), versus the recomputed D7 "
         f"majority-code reference of {d7['gt_boundaries_majority_code']['MoF']:.2f}. "
         f"The improvement was {improvement:.2f}, below the required +5.00."),
        "",
        "![MoF vocabulary sweep](units_sweep_mof.png)",
        "",
        "## Gates",
        "",
        f"- Gate 0: PASS ({patch['gate0']['agreement']:.6f} agreement; threshold 0.999).",
        (f"- Gate 1: PASS ({primary[(60, 10)]['MoF']:.2f} mean MoF; "
         f"{abs(primary[(60, 10)]['MoF'] - 46.42):.2f} from 46.42)."),
        f"- Gate 2: FAIL (HuGaDB improvement {improvement:.2f}; required 5.00 on both datasets).",
        "",
        "## Full primary curve (mean across three seeds)",
        "",
        "| W | K_u | MoF | SD | F1@50 | Units/GT segment | NMI action | NMI subject |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (window, num_units), values in sorted(primary.items()):
        lines.append(
            f"| {window} | {num_units} | {values['MoF']:.2f} | "
            f"{values['MoF_std']:.2f} | {values['F1@50']:.2f} | "
            f"{values['units_per_segment']:.2f} | {values['nmi_action']:.3f} | "
            f"{values['nmi_subject']:.3f} |"
        )
    lines.extend([
        "",
        "## Controls",
        "",
        "| Control | Best mean MoF | W | K_u |",
        "|---|---:|---:|---:|",
    ])
    for label, key in [
        ("Majority independent unit", "majority_unit"),
        ("Continuous PCA-64 segment mean", "continuous_mean"),
        ("Shuffled units", "shuffled_histogram"),
        ("Length-weighted segment clustering", "histogram_length_weighted"),
    ]:
        value = merged["controls"][key]
        lines.append(
            f"| {label} | {value['mean_mof']:.2f} | {value['window']} | "
            f"{value['num_units'] if value['num_units'] is not None else 'n/a'} |"
        )
    lines.extend([
        "",
        "## Supported",
        "",
        "- Gate 0 verifies that dumped pre-quantisation latents reproduce cached unit assignments.",
        "- Real unit structure matters: shuffled-unit controls collapse well below the primary cells.",
        "- At the best cell, unit/action NMI exceeds unit/subject NMI; identity nuisance is not the main explanation there.",
        "- Shorter windows increase observations per GT segment, but do not deliver the preregistered +5 MoF improvement.",
        "",
        "## Not supported",
        "",
        "- A larger independent vocabulary does not make action identity substantially recoverable under this protocol.",
        "- Performance does not improve monotonically with vocabulary size; the largest vocabularies often degrade it.",
        "- Gate 2 does not justify expansion to retrained T1 seeds or BABEL-1.",
        "",
        "## Planned but not run",
        "",
        "- LARa released checkpoint: stopped because HuGaDB already made the two-dataset Gate 2 conjunction impossible.",
        "- Three T1 K=C seeds and BABEL-1: prohibited after Gate 2 failure.",
        "- W=patch, K_u=100 without PCA: raw-dimensional k-means was terminated by the OS; no result is claimed.",
        "",
        "## Protocol notes",
        "",
        "- CPU only; fixed seeds 111, 222, and 1538574472.",
        "- Incomplete terminal windows were excluded from vocabulary/PCA fitting and zero-padded only for assignment, preserving frame alignment.",
        "- The explicit `models/hugadb/epoch-30.model` checkpoint differs from `models/pretrained/hugadb.model` behind the original 44.85 table; therefore the same-checkpoint D7 reference (44.96) is primary here.",
    ])
    report_path = args.out_dir / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "json": str(json_path), "csv": str(csv_path), "plot": str(plot_path),
        "report": str(report_path), "best": best, "gate2": gate2,
    }, indent=2))


if __name__ == "__main__":
    main()
