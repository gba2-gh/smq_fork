"""
Identify the exact JSD convention used by HVQ / HiST-VQ.

HiST-VQ, Sec 4.1: "JSD measures how closely the predicted segment length
distribution matches the ground truth distribution for each video by comparing
their histograms (with 20-frame bins) using the Jensen-Shannon Distance. These
distances are averaged per activity and then combined into a frame-weighted
average across all activities."

That leaves three things underspecified, which this script sweeps:
  1. log base for the JS divergence (2 -> distance in [0,1]; e -> [0, .8326])
  2. the bin grid: how far the 20-frame bins run before the overflow bin, and
     whether that grid is fixed dataset-wide or set per video
  3. the frame weight used to combine activities

Published SMQ row of HiST-VQ Tab. 3 (x100, lower better):
    HuGaDB 87.1 | LARa 74.2 | BABEL-1 72.8 | BABEL-2 77.0 | BABEL-3 81.9
Those five targets are enough to pin the convention down.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TARGET = {"hugadb": 87.1, "lara": 74.2, "babel1": 72.8, "babel2": 77.0, "babel3": 81.9}


def runs(labels):
    """Segment lengths (run-length encoding) of a frame-wise label array."""
    if len(labels) == 0:
        return np.array([])
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(labels)]))
    return np.diff(bounds)


def js_distance(p, q, base):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / p.sum() if p.sum() else p
    q = q / q.sum() if q.sum() else q
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    div = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    div = div / np.log(base)
    return float(np.sqrt(max(div, 0.0)))


def activity_of(dataset, name):
    """The HVQ notion of 'activity' as it exists in each dataset's file names."""
    if dataset == "lara":
        return name[:3]                      # scenario L01 / L02 / L03
    if dataset == "hugadb":
        return name.split("_")[2]            # all files are 'various'
    return "all"                             # BABEL has no activity grouping


def compute(preds_file, bin_width, n_bins, grid, base, weight):
    d = np.load(preds_file, allow_pickle=True)
    dataset = str(d["dataset"])
    gts, prs, names = list(d["gt"]), list(d["pred"]), list(d["names"])

    if grid == "fixed":
        edges = np.arange(0, bin_width * n_bins + 1, bin_width, dtype=float)
        edges = np.append(edges, np.inf)

    per_act = {}
    for gt, pr, nm in zip(gts, prs, names):
        g, p = runs(gt), runs(pr)
        if len(g) == 0 or len(p) == 0:
            continue
        if grid == "per_video":
            hi = max(g.max(), p.max())
            e = np.arange(0, hi + bin_width, bin_width, dtype=float)
            if len(e) < 2:
                e = np.array([0.0, bin_width])
            e = np.append(e, np.inf)
        else:
            e = edges
        hg, _ = np.histogram(g, bins=e)
        hp, _ = np.histogram(p, bins=e)
        act = activity_of(dataset, str(nm))
        per_act.setdefault(act, {"jsd": [], "frames": 0})
        per_act[act]["jsd"].append(js_distance(hg, hp, base))
        per_act[act]["frames"] += len(gt)

    acts = sorted(per_act)
    means = np.array([np.mean(per_act[a]["jsd"]) for a in acts])
    if weight == "frames":
        w = np.array([per_act[a]["frames"] for a in acts], float)
    else:
        w = np.array([len(per_act[a]["jsd"]) for a in acts], float)
    return float(100.0 * np.average(means, weights=w))


if __name__ == "__main__":
    pred_dir = Path("results/preds")
    datasets = ["hugadb", "lara", "babel1", "babel2", "babel3"]

    grids = ["fixed", "per_video"]
    n_bins_opts = [5, 10, 15, 20, 25, 30, 50, 100]
    bases = [2.0, np.e]
    weights = ["frames"]

    rows = []
    for grid, nb, base, weight in itertools.product(grids, n_bins_opts, bases, weights):
        if grid == "per_video" and nb != n_bins_opts[0]:
            continue  # n_bins is unused when the grid is per-video
        vals = {}
        for ds in datasets:
            f = pred_dir / f"{ds}_pretrained.npz"
            if f.exists():
                vals[ds] = compute(f, 20, nb, grid, base, weight)
        err = np.mean([abs(vals[k] - TARGET[k]) for k in vals])
        rows.append(dict(grid=grid, n_bins=(None if grid == "per_video" else nb),
                         base=("2" if base == 2.0 else "e"), weight=weight,
                         mean_abs_err=err, **{k: round(v, 2) for k, v in vals.items()}))

    rows.sort(key=lambda r: r["mean_abs_err"])
    print(f"{'grid':<10}{'bins':<6}{'base':<6}{'wt':<8}{'MAE':<8}" +
          "".join(f"{d:<10}" for d in datasets))
    print(f"{'TARGET':<10}{'':<6}{'':<6}{'':<8}{'':<8}" +
          "".join(f"{TARGET[d]:<10}" for d in datasets))
    print("-" * 96)
    for r in rows:
        print(f"{r['grid']:<10}{str(r['n_bins']):<6}{r['base']:<6}{r['weight']:<8}"
              f"{r['mean_abs_err']:<8.2f}" + "".join(f"{r.get(d, float('nan')):<10}" for d in datasets))
    Path("results/e1_jsd").mkdir(parents=True, exist_ok=True)
    Path("results/e1_jsd/convention_sweep.json").write_text(json.dumps(rows, indent=2))
