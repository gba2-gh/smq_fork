"""
Minimal figures for the primary contrasts.
  fig_exp1_retrieval.png : common-anchor class-balanced precision@10 (levels, then paired differences)
  fig_exp2_f1.png        : F1@50 levels and the paired contrasts against the +2 point threshold
Colours follow the entity, not its rank (documented reference palette slots; the palette validator
could not be run because node is unavailable, so every bar is also directly labelled).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEG = ROOT / "results" / "seg"
FIG = ROOT / "figures" / "seg"
FIG.mkdir(parents=True, exist_ok=True)

COL = dict(fixed="#2a78d6", oracle="#eb6834", random="#1baf7a", dp="#4a3aa7", dp_random="#1baf7a", smq="#8a8985")
NAME = dict(fixed="Fixed 1 s", oracle="Oracle", random="Random\n(oracle-matched)", dp="DP (label-free)",
            dp_random="Random\n(DP-matched)", smq="Original SMQ")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
DS = [d for d in ("lara", "babel1", "babel2", "babel3") if (SEG / f"exp1_summary_{d}.json").exists()]
TITLE = dict(lara="LARa (16 participants)", babel1="BABEL-1 (1,960 rec.)", babel2="BABEL-2 (255 rec.)",
             babel3="BABEL-3 (183 rec.)")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig1():
    S = {d: json.load(open(SEG / f"exp1_summary_{d}.json")) for d in DS}
    fig, axes = plt.subplots(2, len(DS), figsize=(3.6 * len(DS), 6.4), squeeze=False)
    for j, d in enumerate(DS):
        ax = axes[0, j]
        style(ax)
        conds = ["fixed", "oracle", "random"]
        x = 0
        ticks, labels = [], []
        for mode, mlabel in (("cont", "continuous"), ("q", "quantized")):
            for c in conds:
                e = S[d]["conditions"][c][f"ca_{mode}_p"]
                ax.bar(x, e["point"], width=0.8, color=COL[c])
                ax.errorbar(x, e["point"], yerr=[[e["point"] - e["lo"]], [e["hi"] - e["point"]]], color=INK, lw=1, capsize=2)
                ax.text(x, e["hi"] + 0.005, f"{e['point']:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
                x += 1
            ticks.append(x - 2)
            labels.append(mlabel)
            x += 1
        ch = S[d]["conditions"]["fixed"]["ca_q_chance"]["point"]
        ax.axhline(ch, color=MUTED, lw=1, ls="--")
        ax.text(x - 0.5, ch, " chance", va="bottom", ha="right", fontsize=7, color=MUTED)
        ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(TITLE[d], fontsize=10, color=INK)
        if j == 0:
            ax.set_ylabel("class-balanced precision@10\n(common anchors; own-group excluded)", fontsize=8, color=MUTED)
        # paired differences
        ax = axes[1, j]
        style(ax)
        rows = []
        for a, b in (("oracle", "fixed"), ("oracle", "random"), ("random", "fixed")):
            for mode in ("cont", "q"):
                rows.append((f"{a} − {b}\n{'cont.' if mode == 'cont' else 'quant.'}",
                             S[d]["paired_contrasts"][f"{a}-{b}"][f"ca_{mode}_p"], a))
        for i, (lab, e, a) in enumerate(rows):
            ax.errorbar(i, e["point"], yerr=[[e["point"] - e["lo"]], [e["hi"] - e["point"]]], fmt="o", color=COL[a],
                        ecolor=COL[a], capsize=3, ms=5, lw=1.5)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r[0].replace("\n", " ") for r in rows], fontsize=7, rotation=40, ha="right")
        if j == 0:
            unit = "participants" if d == "lara" else "recordings"
            ax.set_ylabel(f"paired difference [95% interval]\n(resampling of {unit})", fontsize=8, color=MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[c]) for c in ("fixed", "oracle", "random")]
    fig.legend(handles, ["Fixed 1 s", "Oracle boundaries", "Random (oracle-matched durations)"], loc="lower center", ncol=3,
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "fig_exp1_retrieval.png", dpi=160)
    plt.close(fig)


def fig2():
    ds2 = [d for d in DS if (SEG / f"exp2_summary_{d}.json").exists()]
    if not ds2:
        return
    S = {d: json.load(open(SEG / f"exp2_summary_{d}.json")) for d in ds2}
    fig, axes = plt.subplots(2, len(ds2), figsize=(3.6 * len(ds2), 6.4), squeeze=False)
    for j, d in enumerate(ds2):
        ax = axes[0, j]; style(ax)
        for i, c in enumerate(("smq", "fixed", "dp", "dp_random")):
            e = S[d]["conditions"][c]["F1_50_h"]
            ax.bar(i, e["point"], color=COL[c], width=0.75)
            ax.errorbar(i, e["point"], yerr=[[e["point"] - e["lo"]], [e["hi"] - e["point"]]], color=INK, lw=1, capsize=2)
            ax.text(i, e["hi"] + 0.4, f"{e['point']:.1f}", ha="center", fontsize=7, color=INK)
        ax.set_xticks(range(4)); ax.set_xticklabels([NAME[c] for c in ("smq", "fixed", "dp", "dp_random")], fontsize=6)
        ax.set_title(TITLE[d], fontsize=10, color=INK)
        if j == 0:
            ax.set_ylabel("segmental F1@50 (fold-level Hungarian)", fontsize=8, color=MUTED)
        ax = axes[1, j]; style(ax)
        rows = [("DP − Fixed", "dp-fixed", "dp"), ("DP − DP-random", "dp-dp_random", "dp"),
                ("DP-random − Fixed", "dp_random-fixed", "dp_random")]
        for i, (lab, key, c) in enumerate(rows):
            e = S[d]["paired_contrasts"][key]["F1_50_h"]
            ax.errorbar(i, e["point"], yerr=[[e["point"] - e["lo"]], [e["hi"] - e["point"]]], fmt="o", color=COL[c], capsize=3, ms=5, lw=1.5)
        ax.axhline(0, color=MUTED, lw=1)
        ax.axhline(2.0, color=INK, lw=1, ls="--")
        ax.text(2.4, 2.0, "+2 pt target", fontsize=7, va="bottom", ha="right", color=INK)
        ax.set_xticks(range(3)); ax.set_xticklabels([r[0] for r in rows], fontsize=7)
        if j == 0:
            ax.set_ylabel("paired F1@50 difference, points\n[95% interval]", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIG / "fig_exp2_f1.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    print("figures written to", FIG)
