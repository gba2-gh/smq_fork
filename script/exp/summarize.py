"""
Aggregate T1-T4 analysis JSON into tables and figures, applying the
pre-registered effect rule (results/PREREG_T1-T4.md):

    effect  <=>  |mean_A - mean_B| > max(sd_A, sd_B)  and the mean +- sd bands do not overlap

Works on partial results: conditions with fewer than 3 seeds are shown with
their n and are never given a verdict.

    python script/exp/summarize.py            -> results/exp/SUMMARY.md, results/exp/summary_*.csv,
                                                  figures/exp/*.svg|png
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = Path("results/exp/analysis")
FIG = Path("figures/exp")
DATASETS = ["hugadb", "lara", "babel1"]
C = {"hugadb": 10, "lara": 8, "babel1": 5}
SEEDS = ["1538574472", "111", "222"]
# T1 and T3 were concluded at 2 seeds, while the effect rule was pre-registered
# at n = 3 (see the report's Deviations section). Verdicts computed from 2 seeds
# are provisional, and every cell prints its own n.
MIN_SEEDS = 2
NAME = {"hugadb": "HuGaDB", "lara": "LARa", "babel1": "BABEL-1"}

# dataviz reference palette, categorical slots in fixed order (light mode)
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def load(path):
    try:
        r = json.loads(path.read_text())
        return None if r.get("failed") else r
    except Exception:
        return None


def get(r, dotted):
    for k in dotted.split("."):
        if r is None or k not in r:
            return None
        r = r[k]
    return r


def stats(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return None
    return dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"), n=len(v))


def fmt(s, digits=1):
    if s is None:
        return "—"
    if s["n"] < 2:
        return f"{s['mean']:.{digits}f} (n={s['n']})"
    return f"{s['mean']:.{digits}f} ± {s['sd']:.{digits}f} (n={s['n']})"


def verdict(a, b, lower_is_better=False):
    """Effect rule. Returns '', 'no effect', '▲ effect' or '▼ effect' (▲ = b better than a)."""
    if a is None or b is None or a["n"] < MIN_SEEDS or b["n"] < MIN_SEEDS:
        return ""
    diff = b["mean"] - a["mean"]
    thr = max(a["sd"], b["sd"])
    overlap = (a["mean"] - a["sd"] <= b["mean"] + b["sd"]) and (b["mean"] - b["sd"] <= a["mean"] + a["sd"])
    if abs(diff) > thr and not overlap:
        better = diff < 0 if lower_is_better else diff > 0
        return "▲ effect" if better else "▼ effect"
    return "no effect"


def seeds_of(folder, pattern):
    out = {}
    for f in sorted(folder.glob("*.json")):
        m = re.fullmatch(pattern, f.stem)
        if m:
            out[f.stem] = load(f)
    return out


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.grid(axis="y", color=GRID, lw=.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------- T1 --
def t1(md, rows):
    md.append("## T1 — codebook size decoupled from class count\n")
    md.append("Mean ± sd over the seeds that ran (n per cell; T1 was concluded at 2 seeds). "
              "Verdict columns compare each K against K = C "
              "(same readout) under the effect rule; the probe verdict compares SMQ's code probe "
              "against the raw-feature k-means control at the same K.\n")
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2))
    for col, ds in enumerate(DATASETS):
        base = None
        ks = [C[ds] * m for m in (1, 2, 4, 8, 16)]
        md.append(f"\n### {NAME[ds]} (C = {C[ds]})\n")
        md.append("| K | thr | (a) MoF | (c1) geom MoF | (c2) temp MoF | (c1) F1@50 | (c2) F1@50 | (c2) JSD "
                  "| code probe | k-means code probe | latent probe | perplexity | c1 MoF vs K=C | c2 MoF vs K=C | c2 F1@50 vs K=C | probe vs k-means |")
        md.append("|" + "---|" * 16)
        series = defaultdict(list)
        for K in ks:
            runs = [load(A / "T1" / ds / f"K{K}_s{s}.json") for s in SEEDS]
            km = [load(A / "T1_kmeans" / ds / f"K{K}_s{s}.json") for s in SEEDS]
            st = {k: stats([get(r, k) for r in runs]) for k in
                  ("a_bijective.MoF", "c1_geometry_merge.MoF", "c2_temporal_merge.MoF",
                   "c1_geometry_merge.F1@50", "c2_temporal_merge.F1@50", "c2_temporal_merge.JSD",
                   "probe_code_onehot", "probe_latent", "codebook.perplexity")}
            kst = {k: stats([get(r, k) for r in km]) for k in
                   ("probe_code_onehot", "c1_geometry_merge.MoF", "c2_temporal_merge.MoF")}
            if K == C[ds]:
                base = st
            thr = max(1, round(10 * C[ds] / K))
            md.append(f"| {K} | {thr} | {fmt(st['a_bijective.MoF'])} | {fmt(st['c1_geometry_merge.MoF'])} | "
                      f"{fmt(st['c2_temporal_merge.MoF'])} | {fmt(st['c1_geometry_merge.F1@50'])} | "
                      f"{fmt(st['c2_temporal_merge.F1@50'])} | {fmt(st['c2_temporal_merge.JSD'])} | "
                      f"{fmt(st['probe_code_onehot'])} | {fmt(kst['probe_code_onehot'])} | {fmt(st['probe_latent'])} | "
                      f"{fmt(st['codebook.perplexity'])} | "
                      f"{'' if K == C[ds] else verdict(base['c1_geometry_merge.MoF'], st['c1_geometry_merge.MoF'])} | "
                      f"{'' if K == C[ds] else verdict(base['c2_temporal_merge.MoF'], st['c2_temporal_merge.MoF'])} | "
                      f"{'' if K == C[ds] else verdict(base['c2_temporal_merge.F1@50'], st['c2_temporal_merge.F1@50'])} | "
                      f"{verdict(kst['probe_code_onehot'], st['probe_code_onehot'])} |")
            for key, s in list(st.items()) + [("km." + k, v) for k, v in kst.items()]:
                series[key].append(s)
                rows.append(dict(exp="T1", dataset=ds, K=K, metric=key,
                                 mean=None if s is None else s["mean"],
                                 sd=None if s is None else s["sd"], n=0 if s is None else s["n"]))

        churn = [load(A / "T1" / ds / f"K{16 * C[ds]}_s{SEEDS[0]}_thr10.json")]
        if churn[0]:
            md.append(f"\nChurn check (K = {16 * C[ds]}, published threshold 10, seed {SEEDS[0]}): "
                      f"(c2) MoF {churn[0]['c2_temporal_merge']['MoF']:.1f}, code probe "
                      f"{churn[0]['probe_code_onehot']:.1f}, perplexity {churn[0]['codebook']['perplexity']:.1f}.")
        raw = load(A / "T1_kmeans" / ds / "raw_patch_mean_probe.json")

        x = np.arange(len(ks))

        def line(ax, key, color, label, marker="o"):
            m = [np.nan if s is None else s["mean"] for s in series[key]]
            e = [0 if s is None or np.isnan(s["sd"]) else s["sd"] for s in series[key]]
            ax.errorbar(x, m, yerr=e, color=color, lw=2, marker=marker, ms=5, capsize=3, label=label)

        ax = axes[0, col]
        line(ax, "probe_latent", S1, "SMQ latent (pre-quant.)")
        line(ax, "probe_code_onehot", S2, "SMQ code")
        line(ax, "km.probe_code_onehot", S3, "k-means code (control)", marker="s")
        if raw:
            ax.axhline(raw["probe_raw_patch_mean"], color=INK2, lw=1, ls="--")
            ax.text(x[-1], raw["probe_raw_patch_mean"], "raw features", color=INK2, fontsize=7,
                    ha="right", va="bottom")
        ax.set_title(f"{NAME[ds]} — patch→class linear probe", fontsize=10, color=INK, loc="left")
        ax.set_ylabel("accuracy (%)", fontsize=8, color=INK2)
        _style(ax)
        ax.set_xticks(x); ax.set_xticklabels([str(k) for k in ks])

        ax = axes[1, col]
        line(ax, "a_bijective.MoF", INK2, "(a) as published")
        line(ax, "c1_geometry_merge.MoF", S1, "(c1) geometry merge")
        line(ax, "c2_temporal_merge.MoF", S2, "(c2) temporal merge")
        line(ax, "km.c2_temporal_merge.MoF", S3, "k-means (c2)", marker="s")
        ax.set_title(f"{NAME[ds]} — MoF after merging to C", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("codebook size K", fontsize=8, color=INK2)
        ax.set_ylabel("MoF (%)", fontsize=8, color=INK2)
        _style(ax)
        ax.set_xticks(x); ax.set_xticklabels([str(k) for k in ks])
    axes[0, 0].legend(frameon=False, fontsize=7, labelcolor=INK2)
    axes[1, 0].legend(frameon=False, fontsize=7, labelcolor=INK2)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(FIG / f"t1_codebook_size.{ext}", dpi=180)
    plt.close(fig)
    md.append("\n![T1](../../figures/exp/t1_codebook_size.png)\n")


# ---------------------------------------------------------------------- T2 --
def t2(md, rows):
    md.append("## T2 — temporal-consistency objective (K = C)\n")
    md.append("Baseline = T1 K = C runs (same runner). Verdicts vs baseline under the effect rule.\n")
    keys = [("a_bijective.MoF", False), ("a_bijective.Edit", False), ("a_bijective.F1@50", False),
            ("a_bijective.JSD", True), ("probe_code_onehot", False), ("nmi_code_vs_gt_patch", False),
            ("codebook.perplexity", False)]
    for ds in DATASETS:
        md.append(f"\n### {NAME[ds]}\n")
        md.append("| condition | " + " | ".join(k.split(".")[-1] for k, _ in keys) + " |")
        md.append("|" + "---|" * (len(keys) + 1))
        base = {k: stats([get(load(A / "T1" / ds / f"K{C[ds]}_s{s}.json"), k) for s in SEEDS]) for k, _ in keys}
        md.append("| baseline | " + " | ".join(fmt(base[k], 3 if "nmi" in k else 1) for k, _ in keys) + " |")
        for cond in ("tc1x", "tc10x", "tconly"):
            st = {k: stats([get(load(A / "T2" / ds / f"{cond}_s{s}.json"), k) for s in SEEDS]) for k, _ in keys}
            md.append(f"| {cond} | " + " | ".join(
                f"{fmt(st[k], 3 if 'nmi' in k else 1)} {verdict(base[k], st[k], lib)}" for k, lib in keys) + " |")
            for k, _ in keys:
                s = st[k]
                rows.append(dict(exp="T2", dataset=ds, K=cond, metric=k, mean=None if s is None else s["mean"],
                                 sd=None if s is None else s["sd"], n=0 if s is None else s["n"]))


# ---------------------------------------------------------------------- T3 --
def t3(md, rows):
    md.append("## T3 — duration-aware decoding (post-hoc)\n")
    md.append("Δ = decoder − argmax on the same checkpoint. Released checkpoints: single values. "
              "T1 K = C seeds: mean ± sd of the per-seed Δ; the verdict compares decoded vs argmax "
              "across the 3 seeds under the effect rule.\n")
    metrics = [("MoF", False), ("Edit", False), ("F1@50", False), ("JSD", True)]
    for ds in DATASETS:
        md.append(f"\n### {NAME[ds]}\n")
        md.append("| decoder | source | " + " | ".join(f"Δ{m}" for m, _ in metrics) + " | segments pred/GT |")
        md.append("|" + "---|" * (len(metrics) + 3))
        pre = load(A / "T3" / "pretrained" / f"{ds}.json")
        seeds = [load(A / "T3" / "T1" / ds / f"K{C[ds]}_s{s}.json") for s in SEEDS]
        dec_keys = [k for k in (pre or next((s for s in seeds if s), {}) or {}) if k.startswith(("hmm_beta", "hsmm_beta"))]
        if pre:
            md.append(f"| argmax | released | " + " | ".join(f"({pre['argmax'][m]:.1f})" for m, _ in metrics)
                      + f" | {pre['argmax']['segments_pred_over_gt']:.2f} |")
        for dk in dec_keys:
            if pre and dk in pre:
                md.append(f"| {dk} | released | " + " | ".join(
                    f"{pre[dk][m] - pre['argmax'][m]:+.1f}" for m, _ in metrics)
                          + f" | {pre[dk]['segments_pred_over_gt']:.2f} |")
            ok = [s for s in seeds if s and dk in s]
            if ok:
                cells = []
                for m, lib in metrics:
                    d = stats([s[dk][m] - s["argmax"][m] for s in ok])
                    v = verdict(stats([s["argmax"][m] for s in ok]), stats([s[dk][m] for s in ok]), lib)
                    cells.append(f"{d['mean']:+.1f} ± {0 if np.isnan(d['sd']) else d['sd']:.1f} {v}")
                    rows.append(dict(exp="T3", dataset=ds, K=dk, metric="delta_" + m, mean=d["mean"],
                                     sd=d["sd"], n=d["n"]))
                seg = np.mean([s[dk]["segments_pred_over_gt"] for s in ok])
                md.append(f"| {dk} | T1 seeds (n={len(ok)}) | " + " | ".join(cells) + f" | {seg:.2f} |")


# ---------------------------------------------------------------------- T4 --
def t4(md, rows):
    md.append("## T4 — patch duration vs frame rate\n")
    md.append("1 s @ native = T1 K = C runs. Durations in seconds; JSD at native-frame resolution. "
              "Verdicts vs the 1 s native condition.\n")
    conds = {"hugadb": ["P0.5s", "base", "P2s", "R2", "R4"], "lara": ["P0.5s", "base", "P2s", "R2", "R5"],
             "babel1": ["P0.5s", "base", "P2s", "R2", "R3"]}
    keys = [("a_bijective.MoF", False), ("a_bijective.Edit", False), ("a_bijective.F1@50", False),
            ("a_bijective.JSD", True), ("durations_s.pred_mean", None), ("durations_s.gt_mean", None),
            ("patch_grid_oracle.MoF", None), ("probe_code_onehot", False)]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    for col, ds in enumerate(DATASETS):
        md.append(f"\n### {NAME[ds]}\n")
        md.append("| condition | patch (s) | fps | " + " | ".join(k for k, _ in keys) + " |")
        md.append("|" + "---|" * (len(keys) + 3))
        base = {k: stats([get(load(A / "T1" / ds / f"K{C[ds]}_s{s}.json"), k) for s in SEEDS]) for k, _ in keys}
        pts = []
        for cond in conds[ds]:
            if cond == "base":
                runs = [load(A / "T1" / ds / f"K{C[ds]}_s{s}.json") for s in SEEDS]
            else:
                runs = [load(A / "T4" / ds / f"{cond}_s{s}.json") for s in SEEDS]
            st = {k: stats([get(r, k) for r in runs]) for k, _ in keys}
            any_r = next((r for r in runs if r), None)
            ps = f"{any_r['patch_seconds']:.2f}" if any_r else "—"
            fps_v = ({"hugadb": 60, "lara": 50, "babel1": 30}[ds] / any_r["rate"]) if any_r else None
            md.append(f"| {'1s native' if cond == 'base' else cond} | {ps} | "
                      f"{'—' if fps_v is None else f'{fps_v:g}'} | " + " | ".join(
                f"{fmt(st[k], 2 if 'durations' in k else 1)}"
                f"{'' if cond == 'base' or lib is None else ' ' + verdict(base[k], st[k], lib)}" for k, lib in keys) + " |")
            for k, _ in keys:
                s = st[k]
                rows.append(dict(exp="T4", dataset=ds, K=cond, metric=k, mean=None if s is None else s["mean"],
                                 sd=None if s is None else s["sd"], n=0 if s is None else s["n"]))
            if cond in ("P0.5s", "base", "P2s") and any_r and st["durations_s.pred_mean"]:
                pts.append((any_r["patch_seconds"], st["durations_s.pred_mean"]))
        ax = axes[col]
        if pts:
            xs = [p[0] for p in pts]
            ax.errorbar(xs, [p[1]["mean"] for p in pts],
                        yerr=[0 if np.isnan(p[1]["sd"]) else p[1]["sd"] for p in pts],
                        color=S2, lw=2, marker="o", capsize=3, label="predicted mean segment")
            if base and base["durations_s.gt_mean"]:
                ax.axhline(base["durations_s.gt_mean"]["mean"], color=S1, lw=2, label="GT mean segment")
            ax.plot(xs, xs, color=INK2, lw=1, ls="--", label="= one patch")
            ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels([f"{x:g}" for x in xs])
        ax.set_title(f"{NAME[ds]} — Sweep P", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("patch duration (s)", fontsize=8, color=INK2)
        ax.set_ylabel("segment duration (s)", fontsize=8, color=INK2)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=7, labelcolor=INK2)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(FIG / f"t4_patch_duration.{ext}", dpi=180)
    plt.close(fig)
    md.append("\n![T4](../../figures/exp/t4_patch_duration.png)\n")


def main():
    md = ["# T1–T4 results summary (auto-generated)\n",
          "Generated by `script/exp/summarize.py` from `results/exp/analysis/`. "
          "Pre-registration: `results/PREREG_T1-T4.md`. Effect rule: |Δ mean| > max(sd) and "
          "mean ± sd bands do not overlap. **Deviation:** the rule was pre-registered at n = 3; "
          "T1 and T3 were concluded at n = 2 (T4 ran all 3 seeds), so their verdicts are "
          "provisional. T2 was not run — see the report.\n"]
    rows = []
    for fn in (t1, t2, t3, t4):
        fn(md, rows)
    Path("results/exp").mkdir(parents=True, exist_ok=True)
    Path("results/exp/SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    with open("results/exp/summary_long.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["exp", "dataset", "K", "metric", "mean", "sd", "n"])
        w.writeheader()
        w.writerows(rows)
    print("[summarize] wrote results/exp/SUMMARY.md, results/exp/summary_long.csv, figures/exp/")


if __name__ == "__main__":
    main()
