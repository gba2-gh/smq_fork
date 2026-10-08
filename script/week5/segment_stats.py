"""Ground-truth segment-length statistics per dataset.

A segment is a maximal run of identical frame labels in one groundTruth file
(one label per line, one line per frame). Reports the median segment length and
the share of segments shorter than 1 s, plus the mask span (in tokens) expressed
as a fraction of the median segment.

Usage: python script/week5/segment_stats.py [--span 8]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {  # name: (fps, token window in frames)
    "hugadb": (60, 15),
    "lara": (50, 12),
    "babel1": (30, 8),
    "babel2": (30, 8),
    "babel3": (30, 8),
}


def segment_lengths(gt_dir: Path) -> np.ndarray:
    """Frame length of every segment in every recording."""
    out = []
    for path in sorted(gt_dir.glob("*.txt")):
        labels = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if not labels:
            continue
        starts = [0] + [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]] + [len(labels)]
        out.extend(np.diff(starts))
    return np.asarray(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--span", type=int, default=8, help="mask span in tokens")
    args = parser.parse_args()
    header = f"{'dataset':<8} {'segments':>8} {'median s':>9} {'mean s':>7} {'<1 s':>6} {'<2 s':>6} {'span s':>7} {'span/median':>11}"
    print(header)
    print("-" * len(header))
    for name, (fps, window) in DATASETS.items():
        lengths = segment_lengths(ROOT / "data" / name / "groundTruth")
        median_s = np.median(lengths) / fps
        span_s = args.span * window / fps
        print(f"{name:<8} {len(lengths):>8} {median_s:>9.2f} {lengths.mean() / fps:>7.2f} "
              f"{np.mean(lengths < fps):>6.1%} {np.mean(lengths < 2 * fps):>6.1%} {span_s:>7.2f} {span_s / median_s:>11.2f}")


if __name__ == "__main__":
    main()
