"""Why randomized SVD: does the PCA solver change the tokens? Token level only (no contextualizer, no OT).

For each anchor dataset and seed, on exactly the fitting sample the anchor uses (<= 10k complete windows, the cell's
seed, global standardization), PCA-64 is fitted with sklearn's `randomized` (the anchor), `full` (exact LAPACK SVD)
and `arpack` (exact truncated) solvers, at the anchor's whitening. Reported per cell:

  fit seconds                 per solver
  subspace cosine (min/mean)  cosines of the principal angles between the randomized and the exact 64-d subspaces
                              (1.0 = the same subspace; the smallest is the worst direction)
  variance captured           explained-variance ratio summed over the 64 components, per solver
  K-means-512 codes           K-means fitted on each solver's sample projection (identical K-means seed), then every
                              window of the dataset assigned: adjusted Rand index (ARI) with the randomized codes
                              (code ids are arbitrary, so raw id agreement is meaningless)
  K-means seed reference      ARI between the anchor's codes and the anchor refitted with another K-means seed, i.e.
                              how much the code partition moves anyway when nothing but K-means' initialization changes

If the solver ARI is at or above the K-means seed reference, the solver choice is below the noise floor of the tokenizer
and decoding `pca_full` (stage S of `run_dimred`) is not informative.

Cumulative explained variance: `--curves-only` fits just the PCA solvers (seconds per cell, no K-means) and writes
`curves.json`, `cumvar_mean.png` (one curve per dataset, mean over seeds) and one figure per dataset, `cumvar_<dataset>.png`: one column per seed, the randomized and exact
cumulative explained-variance curves over the 64 components (top) and their difference (bottom, since the two curves
agree to ~1e-5 and overlap at full scale). The full check records the same ratios in `solver_check.json` and also
refreshes `curves.json` and the figures. `--plots-only` redraws from `curves.json`.

  python -m script.week5.dimred_ablation.solver_check [--datasets babel2,babel3,hugadb,lara] [--seeds 111,222,1538574472]
  python -m script.week5.dimred_ablation.solver_check --curves-only
"""

from __future__ import annotations

import os

_CAP = str(min(8, os.cpu_count() or 8))
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_var] = _CAP

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from script.fsq_reconstruction.experiment import write_json_atomic  # noqa: E402
from script.quantized_input_comparison import fit as fitmod  # noqa: E402
from script.week5.dimred_ablation import prepare as PR  # noqa: E402
from script.week5.raw_kmeans_cross_dataset import datasets as D  # noqa: E402

SOLVERS = ("randomized", "full", "arpack")
SEEDS = (111, 222, 1538574472)
DIM, K = 64, 512
OUT = ROOT / "results" / "week5" / "dimred_ablation" / "solver_check"


def kmeans(x: np.ndarray, seed: int) -> KMeans:
    return KMeans(n_clusters=K, init="k-means++", n_init=5, max_iter=300, tol=1e-4, algorithm="lloyd",
                  random_state=seed).fit(x)


def check_cell(dataset: str, seed: int, raw: dict[str, dict]) -> dict:
    a = PR.ANCHORS[dataset]
    window = D.SPECS[dataset].window
    ids = fitmod.sample_window_ids(fitmod.complete_window_ids(fitmod.window_lengths(raw), window), 10_000, seed)
    sample = fitmod.gather(raw, ids)
    scaler = StandardScaler().fit(sample)
    sample_std = scaler.transform(sample)
    all_std = {n: scaler.transform(r["flat"]) for n, r in raw.items()}
    fits, codes, out = {}, {}, dict(dataset=dataset, seed=seed, whiten=a.whiten, input_dim=int(sample.shape[1]),
                                     n_fit_windows=len(sample))
    for solver in SOLVERS:
        t0 = time.perf_counter()
        pca = PCA(n_components=DIM, whiten=a.whiten, svd_solver=solver, random_state=seed)
        proj = pca.fit_transform(sample_std)
        out[f"fit_seconds_{solver}"] = time.perf_counter() - t0
        out[f"variance_captured_{solver}"] = float(pca.explained_variance_ratio_.sum())
        out[f"ratio_{solver}"] = [float(v) for v in pca.explained_variance_ratio_]
        km = kmeans(proj, seed)
        codes[solver] = np.concatenate([km.predict(pca.transform(all_std[n]).astype(np.float32)) for n in sorted(raw)])
        fits[solver] = pca
        if solver == "randomized":  # the K-means-initialization reference, on the anchor's own projection
            km2 = kmeans(proj, seed + 1)
            ref = np.concatenate([km2.predict(pca.transform(all_std[n]).astype(np.float32)) for n in sorted(raw)])
            out["kmeans_seed_reference_ari"] = float(adjusted_rand_score(codes["randomized"], ref))
    base = fits["randomized"].components_.astype(np.float64)
    for solver in SOLVERS[1:]:
        other = fits[solver].components_.astype(np.float64)
        cos = np.linalg.svd(base @ other.T, compute_uv=False)  # cosines of the principal angles
        out[f"subspace_cos_min_{solver}"] = float(cos.min())
        out[f"subspace_cos_mean_{solver}"] = float(cos.mean())
        out[f"code_ari_{solver}"] = float(adjusted_rand_score(codes["randomized"], codes[solver]))
    out["n_windows"] = int(len(codes["randomized"]))
    return out


def curves_cell(dataset: str, seed: int, raw: dict[str, dict]) -> dict:
    """PCA-64 explained-variance ratios per solver on the anchor's fitting sample; no K-means, no codes."""
    a = PR.ANCHORS[dataset]
    window = D.SPECS[dataset].window
    ids = fitmod.sample_window_ids(fitmod.complete_window_ids(fitmod.window_lengths(raw), window), 10_000, seed)
    sample_std = StandardScaler().fit_transform(fitmod.gather(raw, ids))
    out = dict(dataset=dataset, seed=seed, input_dim=int(sample_std.shape[1]), n_fit_windows=int(sample_std.shape[0]))
    for solver in SOLVERS:
        pca = PCA(n_components=DIM, whiten=a.whiten, svd_solver=solver, random_state=seed).fit(sample_std)
        out[f"ratio_{solver}"] = [float(v) for v in pca.explained_variance_ratio_]
    return out


def save_curves(cells: list[dict]) -> None:
    write_json_atomic(OUT / "curves.json", dict(cells=[{k: v for k, v in c.items() if k in
                                                         ("dataset", "seed", "input_dim", "n_fit_windows") or k.startswith("ratio_")}
                                                        for c in cells]))


def plot_curves(cells: list[dict]) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for ds in dict.fromkeys(c["dataset"] for c in cells):
        cs = sorted((c for c in cells if c["dataset"] == ds), key=lambda c: c["seed"])
        fig, axes = plt.subplots(2, len(cs), figsize=(4.6 * len(cs), 6.2), sharex=True, squeeze=False)
        x = np.arange(1, DIM + 1)
        for j, c in enumerate(cs):
            rand, full = np.cumsum(c["ratio_randomized"]), np.cumsum(c["ratio_full"])
            top, bot = axes[0, j], axes[1, j]
            top.plot(x, full, color="black", lw=2.2, label="exact (full SVD)")
            top.plot(x, rand, color="tab:orange", lw=1.4, ls="--", label="randomized")
            top.set_title(f"seed {c['seed']}")
            top.set_ylim(min(full.min(), rand.min()) * 0.98, 1.0)
            top.grid(alpha=0.3)
            bot.axhline(0, color="gray", lw=0.8)
            bot.plot(x, rand - full, color="tab:orange", lw=1.4)
            bot.set_xlabel("number of components")
            bot.grid(alpha=0.3)
            bot.text(0.98, 0.04, f"total {rand[-1]:.5f} vs {full[-1]:.5f}\nmax |diff| {np.abs(rand - full).max():.1e}",
                     transform=bot.transAxes, ha="right", va="bottom", fontsize=8)
            if j == 0:
                top.set_ylabel("cumulative explained variance ratio")
                bot.set_ylabel("randomized − exact")
                top.legend(loc="lower right", fontsize=8)
        fig.suptitle(f"{ds}: PCA-64 fitted on {cs[0]['n_fit_windows']:,} windows x {cs[0]['input_dim']} features, randomized vs exact SVD", fontsize=11)
        fig.tight_layout()
        path = OUT / f"cumvar_{ds}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_mean_curves(cells: list[dict]) -> Path:
    """One curve per dataset: the cumulative explained variance of the anchor's randomized solver (solid) and of the
    exact SVD (dashed), averaged over the seeds (band: min-max over seeds). Bottom: mean of randomized minus exact."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(1, DIM + 1)
    fig, (top, bot) = plt.subplots(2, 1, figsize=(7.2, 7.4), sharex=True)
    for i, ds in enumerate(dict.fromkeys(c["dataset"] for c in cells)):
        cs = [c for c in cells if c["dataset"] == ds]
        rand = np.stack([np.cumsum(c["ratio_randomized"]) for c in cs])
        full = np.stack([np.cumsum(c["ratio_full"]) for c in cs])
        color = f"C{i}"
        top.plot(x, rand.mean(0), color=color, lw=2.4, label=f"{ds}: {cs[0]['n_fit_windows']:,} windows x {cs[0]['input_dim']} features, {rand[:, -1].mean():.3f} at 64")
        top.plot(x, full.mean(0), color="black", lw=1.0, ls="--", label="exact SVD" if i == 0 else None)
        top.fill_between(x, rand.min(0), rand.max(0), color=color, alpha=0.2, lw=0)
        bot.plot(x, (rand - full).mean(0), color=color, lw=1.6, label=ds)
    top.set_ylabel("cumulative explained variance ratio")
    top.set_title(f"PCA-64 on the fitting sample (<= 10k windows), mean of {len(cs)} seeds\n"
                  "solid: randomized SVD, dashed: exact SVD, band: min-max over seeds", fontsize=10)
    top.legend(loc="lower right", fontsize=9)
    top.grid(alpha=0.3)
    bot.axhline(0, color="gray", lw=0.8)
    bot.set_xlabel("number of components")
    bot.set_ylabel("randomized − exact SVD (mean over seeds)")
    bot.legend(loc="lower left", fontsize=9)
    bot.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT / "cumvar_mean.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def table(rows: list[dict]) -> list[str]:
    lines = ["| dataset | seed | fit s (rand / full / arpack) | variance captured (rand / full) | subspace cos min / mean "
             "(full) | codes ARI vs full | codes ARI vs arpack | K-means seed reference ARI |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['seed']} | {r['fit_seconds_randomized']:.2f} / {r['fit_seconds_full']:.2f} / "
            f"{r['fit_seconds_arpack']:.2f} | {r['variance_captured_randomized']:.5f} / {r['variance_captured_full']:.5f} | "
            f"{r['subspace_cos_min_full']:.6f} / {r['subspace_cos_mean_full']:.6f} | "
            f"{r['code_ari_full']:.4f} | {r['code_ari_arpack']:.4f} | "
            f"{r['kmeans_seed_reference_ari']:.4f} |")
    return lines


def summary(rows: list[dict]) -> list[str]:
    lines = ["| dataset | variance captured, randomized − full | mean subspace cos | codes ARI vs full | K-means seed "
             "reference ARI | full / randomized fit time |", "|---|---|---|---|---|---|"]
    for ds in dict.fromkeys(r["dataset"] for r in rows):
        rs = [r for r in rows if r["dataset"] == ds]
        m = lambda k: float(np.mean([r[k] for r in rs]))  # noqa: E731
        lines.append(f"| {ds} | {m('variance_captured_randomized') - m('variance_captured_full'):+.1e} | "
                     f"{m('subspace_cos_mean_full'):.5f} | {m('code_ari_full'):.3f} | {m('kmeans_seed_reference_ari'):.3f} | "
                     f"{m('fit_seconds_full') / m('fit_seconds_randomized'):.1f}x |")
    return lines


def write_report(rows: list[dict]) -> Path:
    text = ["# PCA solver check (token level)", "", __doc__.split("\n\n")[1], ""] + table(rows) + [""] + summary(rows)
    (OUT / "REPORT.md").write_text("\n".join(text), encoding="utf-8")
    return OUT / "REPORT.md"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="babel2,babel3,hugadb,lara")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--report-only", action="store_true", help="rewrite REPORT.md from the saved json")
    ap.add_argument("--curves-only", action="store_true", help="fit only the PCA solvers, save curves.json and plots")
    ap.add_argument("--plots-only", action="store_true", help="redraw the plots from curves.json")
    args = ap.parse_args(argv)
    if args.plots_only:
        cells = json.loads((OUT / "curves.json").read_text())["cells"]
        for path in plot_curves(cells) + [plot_mean_curves(cells)]:
            print(path)
        return
    if args.curves_only:
        OUT.mkdir(parents=True, exist_ok=True)
        cells = []
        for ds in args.datasets.split(","):
            raw = PR.extract(ds, D.recording_names(D.SPECS[ds]), None)
            for seed in map(int, args.seeds.split(",")):
                with threadpool_limits(limits=PR.TOK.PREP_THREADS):
                    cells.append(curves_cell(ds, seed, raw))
                print(f"[{ds} seed {seed}] curves fitted", flush=True)
            save_curves(cells)
        for path in plot_curves(cells) + [plot_mean_curves(cells)]:
            print(path)
        return
    if args.report_only:
        rows = json.loads((OUT / "solver_check.json").read_text())["rows"]
        print(write_report(rows))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in args.datasets.split(","):
        names = D.recording_names(D.SPECS[ds])
        raw = PR.extract(ds, names, None)
        for seed in map(int, args.seeds.split(",")):
            with threadpool_limits(limits=PR.TOK.PREP_THREADS):
                r = check_cell(ds, seed, raw)
            rows.append(r)
            print(f"[{ds} seed {seed}] subspace cos min {r['subspace_cos_min_full']:.6f}, codes ARI vs full "
                  f"{r['code_ari_full']:.4f} (K-means seed reference {r['kmeans_seed_reference_ari']:.4f}), "
                  f"fit {r['fit_seconds_randomized']:.2f}s vs {r['fit_seconds_full']:.2f}s", flush=True)
            write_json_atomic(OUT / "solver_check.json", dict(rows=rows))
    print(write_report(rows))
    save_curves(rows)
    for path in plot_curves(rows) + [plot_mean_curves(rows)]:
        print(path)


if __name__ == "__main__":
    main()
