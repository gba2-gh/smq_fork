"""
E1 Part A -- Ground-truth duration structure (experiment.md).

For each dataset, iterates every annotated segment in the ground truth,
records its length in frames, and divides by that dataset's patch size
(60 / 50 / 30 frames for HuGaDB / LARa / BABEL respectively, confirmed
against Stage 0's codebase map: main.py's get_dataset_defaults()).

No trained model is needed -- this only reads groundTruth/*.txt.

Outputs, per dataset, under figures/e1_part_a/<dataset>/:
    - segment_length_hist.svg  (log-x histogram of segment length in patches)
    - per_class_stats.csv      (mean/std/min/max/p5/p95, in patches, per class)
    - summary.txt              (fraction < 1 patch, fraction > 5 patches)
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Per-dataset patch size in frames, from main.py's get_dataset_defaults()
PATCH_SIZE = {
    "hugadb": 60,
    "lara": 50,
    "babel1": 30,
    "babel2": 30,
    "babel3": 30,
}


def get_segments(gt_path):
    """Yields (label, length_in_frames) for every contiguous run in every
    groundTruth/*.txt file under gt_path."""
    for file in sorted(gt_path.glob("*.txt")):
        with open(file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            continue
        label = lines[0]
        run = 1
        for i in range(1, len(lines)):
            if lines[i] == label:
                run += 1
            else:
                yield label, run
                label = lines[i]
                run = 1
        yield label, run


def analyze_dataset(dataset, patch_size, out_root):
    gt_path = Path(f"data/{dataset}/groundTruth")
    if not gt_path.is_dir():
        print(f"[{dataset}] no groundTruth dir, skipping")
        return

    per_class_lengths = defaultdict(list)
    all_lengths_patches = []

    for label, length_frames in get_segments(gt_path):
        length_patches = length_frames / patch_size
        per_class_lengths[label].append(length_patches)
        all_lengths_patches.append(length_patches)

    if not all_lengths_patches:
        print(f"[{dataset}] no segments found, skipping")
        return

    all_lengths_patches = np.array(all_lengths_patches)
    out_dir = out_root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Histogram, log-x axis ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(np.log10(all_lengths_patches.min()), np.log10(all_lengths_patches.max()), 40)
    ax.hist(all_lengths_patches, bins=bins)
    ax.set_xscale("log")
    ax.set_xlabel("Segment length (patches)")
    ax.set_ylabel("Count")
    ax.set_title(f"{dataset}: ground-truth segment length (patch_size={patch_size} frames)")
    fig.tight_layout()
    fig.savefig(out_dir / "segment_length_hist.svg")
    plt.close(fig)

    # --- Per-class table ---
    rows = ["class,n,mean,std,min,max,p5,p95"]
    for label in sorted(per_class_lengths):
        vals = np.array(per_class_lengths[label])
        rows.append(",".join(str(v) for v in [
            label, len(vals), vals.mean(), vals.std(), vals.min(), vals.max(),
            np.percentile(vals, 5), np.percentile(vals, 95),
        ]))
    (out_dir / "per_class_stats.csv").write_text("\n".join(rows))

    # --- Overall fractions ---
    frac_short = float((all_lengths_patches < 1.0).mean())
    frac_long = float((all_lengths_patches > 5.0).mean())
    summary = (
        f"dataset: {dataset}\n"
        f"patch_size: {patch_size} frames\n"
        f"n_segments: {len(all_lengths_patches)}\n"
        f"mean_length_patches: {all_lengths_patches.mean():.4f}\n"
        f"median_length_patches: {np.median(all_lengths_patches):.4f}\n"
        f"fraction_shorter_than_1_patch: {frac_short:.4f}\n"
        f"fraction_longer_than_5_patches: {frac_long:.4f}\n"
    )
    (out_dir / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    out_root = Path("figures/e1_part_a")
    for dataset, patch_size in PATCH_SIZE.items():
        analyze_dataset(dataset, patch_size, out_root)
