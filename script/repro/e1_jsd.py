"""
E1 -- Segment-length bias (JSD) + segment/sequence length histograms.

Convention (identified by script/repro/jsd_sweep.py against the five published
SMQ values in HiST-VQ Tab. 3, and confirmed exactly on HuGaDB and LARa):

  * per video, take the run-length encoding of the frame-wise ground truth and
    of the frame-wise prediction -> two lists of segment lengths in frames
  * histogram both on a shared 20-frame grid running from 0 to the longest
    segment in that video, plus a final overflow bin
  * normalise each histogram to sum to 1
  * JS distance = sqrt(JS divergence) with log base 2, so the value lies in
    [0, 1]  (base e caps at .8326 and cannot produce HVQ's published 97.4)
  * mean over the videos of an activity, then a frame-weighted mean over
    activities.  Activity = the grouping present in the file names:
    LARa scenario (L01/L02/L03); HuGaDB is a single category ('various');
    BABEL has none.
  * report x100, lower is better.

Three reference levels are computed alongside, because a raw JSD near 87 is
uninterpretable without knowing what 0 would even mean:

  sparsity_floor    -- split each video's GT segments into two disjoint random
                       halves and score one against the other. Two samples from
                       the *same* distribution, at the sample size this metric
                       actually sees. Nothing can score below this.
  matched_count_null-- draw n_pred lengths from the activity's pooled GT length
                       distribution and score against that video's GT. This is
                       "correct duration statistics, wrong placement, same
                       number of segments as SMQ predicted".
  patch_ceiling     -- majority-vote the GT labels inside each patch, then score
                       that against the raw GT. The best score ANY method
                       emitting one label per patch can obtain. This isolates
                       how much of SMQ's JSD is forced by the patch size.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BIN_W = 20
RNG = np.random.default_rng(0)

# dataviz reference palette, categorical slots 1-3 (light mode)
C_GT, C_PRED, C_CEIL = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def rle(a):
    a = np.asarray(a)
    if a.size == 0:
        return np.array([]), np.array([])
    ch = np.flatnonzero(a[1:] != a[:-1]) + 1
    b = np.concatenate(([0], ch, [len(a)]))
    return np.diff(b), a[b[:-1]]


def js_distance(p, q, base=2.0):
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.sum() == 0 or q.sum() == 0:
        return np.nan
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        k = a > 0
        return float(np.sum(a[k] * np.log(a[k] / b[k])))

    return float(np.sqrt(max((0.5 * kl(p, m) + 0.5 * kl(q, m)) / np.log(base), 0.0)))


def pair_jsd(len_a, len_b):
    """JS distance between two segment-length lists, on their shared 20f grid."""
    if len(len_a) == 0 or len(len_b) == 0:
        return np.nan
    hi = max(np.max(len_a), np.max(len_b))
    edges = np.append(np.arange(0, hi + BIN_W, BIN_W, dtype=float), np.inf)
    ha, _ = np.histogram(len_a, bins=edges)
    hb, _ = np.histogram(len_b, bins=edges)
    return js_distance(ha, hb)


def activity_of(dataset, name):
    if dataset == "lara":
        return name[:3]
    if dataset == "hugadb":
        return name.split("_")[2]
    return "all"


def patch_quantize(gt, patch):
    """Majority GT label per patch, expanded back to frames."""
    out = np.empty_like(gt)
    for s in range(0, len(gt), patch):
        e = min(s + patch, len(gt))
        vals, cnt = np.unique(gt[s:e], return_counts=True)
        out[s:e] = vals[np.argmax(cnt)]
    return out


def frame_weighted(per_act):
    acts = sorted(per_act)
    means = np.array([np.nanmean(per_act[a]["jsd"]) for a in acts])
    w = np.array([per_act[a]["frames"] for a in acts], float)
    keep = ~np.isnan(means)
    return float(100.0 * np.average(means[keep], weights=w[keep]))


def run(preds_file, n_null=50):
    d = np.load(preds_file, allow_pickle=True)
    dataset, patch = str(d["dataset"]), int(d["patch_size"])
    gts, prs, names = list(d["gt"]), list(d["pred"]), [str(n) for n in d["names"]]

    # pooled GT lengths per activity, for the matched-count null
    pool = {}
    for gt, nm in zip(gts, names):
        pool.setdefault(activity_of(dataset, nm), []).append(rle(gt)[0])
    pool = {k: np.concatenate(v) for k, v in pool.items()}

    acc = {k: {} for k in ("jsd", "floor", "null", "ceiling")}
    all_gt_len, all_pr_len, all_ceil_len, seq_len = [], [], [], []

    for gt, pr, nm in zip(gts, prs, names):
        act = activity_of(dataset, nm)
        gl, _ = rle(gt)
        pl, _ = rle(pr)
        cl, _ = rle(patch_quantize(np.asarray(gt), patch))
        seq_len.append(len(gt))
        all_gt_len.append(gl)
        all_pr_len.append(pl)
        all_ceil_len.append(cl)

        for key, val in (("jsd", pair_jsd(gl, pl)), ("ceiling", pair_jsd(gl, cl))):
            acc[key].setdefault(act, {"jsd": [], "frames": 0})
            acc[key][act]["jsd"].append(val)
            acc[key][act]["frames"] += len(gt)

        # sparsity floor: two disjoint halves of this video's own GT segments
        fl = []
        if len(gl) >= 4:
            for _ in range(n_null):
                idx = RNG.permutation(len(gl))
                h = len(gl) // 2
                fl.append(pair_jsd(gl[idx[:h]], gl[idx[h:2 * h]]))
        acc["floor"].setdefault(act, {"jsd": [], "frames": 0})
        acc["floor"][act]["jsd"].append(np.nanmean(fl) if fl else np.nan)
        acc["floor"][act]["frames"] += len(gt)

        # matched-count null: right duration statistics, right segment count
        nl = [pair_jsd(gl, RNG.choice(pool[act], size=len(pl), replace=True))
              for _ in range(n_null)] if len(pl) else []
        acc["null"].setdefault(act, {"jsd": [], "frames": 0})
        acc["null"][act]["jsd"].append(np.nanmean(nl) if nl else np.nan)
        acc["null"][act]["frames"] += len(gt)

    gt_pool = np.concatenate(all_gt_len)
    pr_pool = np.concatenate(all_pr_len)
    ceil_pool = np.concatenate(all_ceil_len)

    res = dict(
        dataset=dataset, ckpt=str(d["ckpt"]), patch_size=patch,
        n_sequences=len(gts),
        jsd_x100=round(frame_weighted(acc["jsd"]), 2),
        sparsity_floor_x100=round(frame_weighted(acc["floor"]), 2),
        matched_count_null_x100=round(frame_weighted(acc["null"]), 2),
        patch_ceiling_x100=round(frame_weighted(acc["ceiling"]), 2),
        wasserstein1_frames=round(float(wasserstein_distance(gt_pool, pr_pool)), 2),
        wasserstein1_over_mean_gt=round(float(
            wasserstein_distance(gt_pool, pr_pool) / gt_pool.mean()), 4),
        n_gt_segments=int(len(gt_pool)), n_pred_segments=int(len(pr_pool)),
        gt_median_len=float(np.median(gt_pool)), pred_median_len=float(np.median(pr_pool)),
        gt_mean_len=round(float(gt_pool.mean()), 1), pred_mean_len=round(float(pr_pool.mean()), 1),
        seg_ratio_pred_over_gt=round(len(pr_pool) / len(gt_pool), 2),
        gt_frac_below_one_patch=round(float((gt_pool < patch).mean()), 4),
        seq_len_median=float(np.median(seq_len)), seq_len_mean=round(float(np.mean(seq_len)), 1),
        seq_len_min=int(np.min(seq_len)), seq_len_max=int(np.max(seq_len)),
    )
    return res, gt_pool, pr_pool, ceil_pool, np.array(seq_len)


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.grid(axis="y", color=GRID, lw=.6, alpha=.7)
    ax.set_axisbelow(True)


def fig_seglen(ds, res, gt, pr, ceil, out, n_bins=12):
    """GT vs SMQ segment-length histogram on the metric's own 20-frame grid."""
    hi = BIN_W * n_bins
    edges = np.arange(0, hi + BIN_W, BIN_W, dtype=float)

    def dens(x):
        h, _ = np.histogram(np.clip(x, 0, hi + 1), bins=np.append(edges, np.inf))
        return h / h.sum() * 100

    hg, hp, hc = dens(gt), dens(pr), dens(ceil)
    x = np.arange(len(hg))
    w = .27

    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    ax.bar(x - w, hg, w, color=C_GT, label="Ground truth", zorder=3)
    ax.bar(x, hp, w, color=C_PRED, label="SMQ prediction", zorder=3)
    ax.bar(x + w, hc, w, color=C_CEIL,
           label=f"GT quantized to {res['patch_size']}f patches (floor)", zorder=3)
    _style(ax)
    lbl = [str(int(e)) for e in edges[:-1]] + [str(int(hi)) + "+"]
    ax.set_xticks(x)
    ax.set_xticklabels(lbl, fontsize=8)
    ax.set_xlabel("segment length (frames, 20-frame bins)", color=INK2, fontsize=9)
    ax.set_ylabel("% of segments", color=INK2, fontsize=9)
    ax.set_title(f"{ds} - segment-length distribution   "
                 f"(JSD {res['jsd_x100']:.1f}, patch floor {res['patch_ceiling_x100']:.1f})",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(out.with_suffix("." + ext), dpi=200, transparent=False)
    plt.close(fig)


def fig_seqlen(ds, seq, patch, out):
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.hist(seq, bins=40, color=C_GT, zorder=3)
    _style(ax)
    med = float(np.median(seq))
    ax.axvline(med, color=C_PRED, lw=2, zorder=4)
    ax.annotate(f"median {int(med)} f = {med / patch:.1f} patches",
                xy=(med, ax.get_ylim()[1] * .85), xytext=(8, 0),
                textcoords="offset points", color=C_PRED, fontsize=9, va="center")
    ax.set_xlabel("sequence length (frames)", color=INK2, fontsize=9)
    ax.set_ylabel("sequences", color=INK2, fontsize=9)
    ax.set_title(f"{ds} - sequence lengths (n={len(seq)})", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(out.with_suffix("." + ext), dpi=200, transparent=False)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("figures/repro_e1_jsd"))
    ap.add_argument("--csv", type=Path, default=Path("results/e1_jsd/jsd_summary.csv"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in args.preds:
        res, gt, pr, ceil, seq = run(f)
        tag = f.stem
        fig_seglen(tag, res, gt, pr, ceil, args.outdir / f"{tag}_segment_lengths")
        fig_seqlen(tag, seq, res["patch_size"], args.outdir / f"{tag}_sequence_lengths")
        rows.append(res)
        print(json.dumps(res))

    with open(args.csv, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\n[e1] wrote {args.csv} and figures to {args.outdir}")
