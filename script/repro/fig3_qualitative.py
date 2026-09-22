"""
Reproduce Figure 3 of the SMQ paper (Gokay et al., ICCV 2025) -- the SMQ row
only, as requested; the CTE / TOT / ASOT rows are not reproduced.

Panels follow the paper: (a) HuGaDB  (b) LARa  (c) BABEL Subset-1
(d) BABEL Subset-3.  Each panel shows the ground-truth strip above the SMQ
prediction strip, with time on the x axis.

Predictions come from the cached dumps and are mapped to ground-truth labels
with the SAME dataset-level (global) Hungarian assignment the published
evaluation uses -- so the strips correspond exactly to the reported MoF.

Colour is assigned per panel, in fixed slot order over the ground-truth class
ids present, so one action keeps one colour within a panel. Every panel carries
a legend and segments wide enough are labelled in place, so identity is never
carried by colour alone; a thin surface-coloured gap separates adjacent fills.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model.eval_utils import create_correspondences, read_mapping_file

# dataviz reference palette, categorical slots 1-8 (light mode), fixed order
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK2 = "#ffffff", "#0b0b0b", "#52514e"

# The paper's own examples where identifiable: the HuGaDB panel is the sequence
# the text describes ("it mixes standing up and sitting down"); the LARa and
# BABEL-1 sequences are the ones the official repo shows as qualitative results.
PANELS = [
    ("a", "hugadb", "HuGaDB_v2_various_01_00", "HuGaDB", 60),
    ("b", "lara",   "L02_S15_R04",             "LARa",   50),
    ("c", "babel1", "929",                     "BABEL Subset-1", 30),
    ("d", "babel3", None,                      "BABEL Subset-3", 30),
]


def rle(a):
    a = np.asarray(a)
    ch = np.flatnonzero(a[1:] != a[:-1]) + 1
    b = np.concatenate(([0], ch, [len(a)]))
    return list(zip(b[:-1], b[1:], a[b[:-1]]))


def load_panel(dataset, seq, preds_dir, data_root):
    d = np.load(preds_dir / f"{dataset}_pretrained.npz", allow_pickle=True)
    names = [str(n) for n in d["names"]]
    gts, prs = list(d["gt"]), list(d["pred"])

    # dataset-level Hungarian, exactly as in evaluate_global_hungarian
    _, pr2gt = create_correspondences(np.concatenate(gts), np.concatenate(prs),
                                      mapping=True)

    if seq is None:
        # Representative, not cherry-picked: among sequences with >= 3 GT
        # classes, take the one whose frame agreement is closest to this
        # dataset's published (global-Hungarian) MoF.
        mapped_all = [np.vectorize(pr2gt.get)(p) for p in prs]
        mof = float(np.concatenate([g == m for g, m in zip(gts, mapped_all)]).mean())
        cand = [(abs(float((np.asarray(g) == m).mean()) - mof), n)
                for g, m, n in zip(gts, mapped_all, names)
                if len(np.unique(g)) >= 3 and len(g) >= 150]
        seq = Path(sorted(cand)[0][1]).stem
    i = names.index(seq + ".npy")
    gt, pred = np.asarray(gts[i]), np.vectorize(pr2gt.get)(prs[i])

    mapping = read_mapping_file(data_root / dataset / "mapping" / "mapping.txt")
    return seq, gt, pred, mapping


def draw_strip(ax, segm, colors, y, h, total, labels=None, min_frac=0.09):
    for s, e, lab in rle(segm):
        ax.add_patch(plt.Rectangle((s / total, y), (e - s) / total, h,
                                   facecolor=colors[lab], edgecolor=SURFACE,
                                   linewidth=1.2, zorder=3))
        if labels is not None and (e - s) / total >= min_frac:
            ax.text((s + e) / 2 / total, y + h / 2, labels[lab],
                    ha="center", va="center", fontsize=7.5, color=SURFACE,
                    zorder=4, fontweight="medium")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_dir", type=Path, default=Path("results/preds"))
    ap.add_argument("--data_root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("figures/repro_fig3/fig3_smq"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(9.0, 6.6))

    for ax, (tag, ds, seq, title, fps) in zip(axes, PANELS):
        seq, gt, pred, mapping = load_panel(ds, seq, args.preds_dir, args.data_root)
        present = sorted(set(np.unique(gt)) | set(np.unique(pred)))
        colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(present)}
        names = {c: mapping.get(int(c), str(c)) for c in present}
        total = len(gt)

        draw_strip(ax, gt, colors, 0.56, 0.34, total, labels=names)
        draw_strip(ax, pred, colors, 0.10, 0.34, total)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.27, 0.73])
        ax.set_yticklabels(["SMQ", "GT"], fontsize=10, color=INK)
        ax.tick_params(length=0, colors=INK2)
        ax.spines[:].set_visible(False)

        secs = total / fps
        ticks = np.linspace(0, 1, 6)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t * secs:.0f}" for t in ticks], fontsize=8, color=INK2)
        if tag == PANELS[-1][0]:
            ax.set_xlabel("time (s)", fontsize=8, color=INK2, labelpad=1)

        agree = 100.0 * float((gt == pred).mean())
        ax.set_title(f"({tag}) {title}   {seq}   "
                     f"{len(rle(gt))} GT segments vs {len(rle(pred))} predicted   "
                     f"frame agreement {agree:.1f}%",
                     fontsize=10, color=INK, loc="left", pad=6)
        ax.legend(handles=[Patch(facecolor=colors[c], label=names[c]) for c in present],
                  loc="upper left", bbox_to_anchor=(0, -0.38), ncol=4,
                  frameon=False, fontsize=8, labelcolor=INK2, handlelength=1.1,
                  handleheight=1.1, columnspacing=1.2)

    fig.suptitle("SMQ qualitative segmentation vs ground truth "
                 "(reproduction of Fig. 3, SMQ row only)",
                 fontsize=12, color=INK, x=0.012, ha="left", y=0.985)
    fig.subplots_adjust(top=0.90, bottom=0.05, left=0.075, right=0.985, hspace=1.45)
    for ext in ("svg", "png"):
        fig.savefig(args.out.with_suffix("." + ext), dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"[fig3] wrote {args.out}.svg / .png")


if __name__ == "__main__":
    main()
