"""Does the 10k-window fitting cap change the tokens? Token level only (no contextualizer, no OT).

For HuGaDB and LARa (the datasets with more than 10k complete windows; BABEL-2/3 have about 10k in total, so their
cap is already "all"), the anchor tokenizer -- global standardization, randomized PCA-64, K-means-512 (k-means++,
n_init 5, max_iter 300, tol 1e-4) -- is fitted on nested fitting samples of 10,000, 40,000 and all complete windows of
the TRAINING recordings (the pipeline's shared train / validation split, seed 111), each with two K-means seeds. Every
fit is evaluated on the complete windows of the held-out VALIDATION recordings, which no fit sees:

  relative error     mean squared error between a held-out window and its centroid, mapped back to the standardized
                     input space, divided by the held-out windows' total variance (lower is better; comparable across
                     sizes because it is measured in the same space)
  seed ARI           adjusted Rand index between the codes of the two K-means seeds at the same size: the instability
                     of the tokenizer at that size (higher = more stable)
  cross-size ARI     mean ARI between codes fitted at two different sizes (all four seed pairings), read against the
                     seed ARI of the smaller size: if it is no lower, the size change moves the codes no more than
                     K-means' own randomness does
  codes used         distinct codes among the held-out windows

The 10k row here is fitted on training recordings only, so it is not the anchor's saved tokens (those use all recordings).

  python -m script.week5.dimred_ablation.sample_size_check --hours 1
  python -m script.week5.dimred_ablation.sample_size_check --report-only
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_var] = "16"

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

DATASETS = ("hugadb", "lara")
KMEANS_SEEDS = (111, 112)
PCA_SEED, SPLIT_SEED, SAMPLE_SEED = 111, 111, 111
K, DIM = 512, 64
THREADS = 16


def out_dir(tag: str) -> Path:
    return ROOT / "results" / "week5" / "dimred_ablation" / tag


def build_data(ds: str, recording_cap: int | None):
    spec = D.SPECS[ds]
    names = D.recording_names(spec, recording_cap)
    raw = PR.extract(ds, names, None)
    frame_counts = {n: int(sum(raw[n]["lengths"])) for n in names}
    train_names, val_names = fitmod.shared_validation_split(frame_counts, SPLIT_SEED)
    train_set = set(train_names)
    ids = fitmod.complete_window_ids(fitmod.window_lengths(raw), spec.window)
    train_ids = [i for i in ids if i[0] in train_set]
    val_ids = [i for i in ids if i[0] not in train_set]
    perm = np.random.default_rng(SAMPLE_SEED).permutation(len(train_ids))  # nested samples: prefixes of one permutation
    train_ids = [train_ids[i] for i in perm]
    return raw, train_ids, fitmod.gather(raw, val_ids), len(val_names)


def sizes_for(n_train: int, requested: list[str]) -> list[tuple[str, int]]:
    out = []
    for s in requested:
        n = n_train if s == "all" else min(int(s), n_train)
        if n not in [m for _, m in out]:
            out.append(("all" if n == n_train else str(n), n))
    return out


def fit_eval(raw, train_ids, n: int, seed: int, val_x: np.ndarray, whiten: bool) -> tuple[dict, np.ndarray]:
    t0 = time.perf_counter()
    sample = fitmod.gather(raw, train_ids[:n])
    scaler = StandardScaler().fit(sample)
    std = scaler.transform(sample)
    pca = PCA(n_components=min(DIM, *std.shape), whiten=whiten, svd_solver="randomized", random_state=PCA_SEED)
    proj = pca.fit_transform(std)
    km = KMeans(n_clusters=K, init="k-means++", n_init=5, max_iter=300, tol=1e-4, algorithm="lloyd",
                random_state=seed).fit(proj)
    fit_seconds = time.perf_counter() - t0
    val_std = scaler.transform(val_x)
    codes = km.predict(pca.transform(val_std).astype(np.float32)).astype(np.int32)
    centroid_std = pca.inverse_transform(km.cluster_centers_)  # whiten-aware: back into the standardized space
    err = float(((val_std - centroid_std[codes]) ** 2).sum(1).mean())
    total = float(((val_std - val_std.mean(0)) ** 2).sum(1).mean())
    return dict(n=n, seed=seed, fit_seconds=fit_seconds, relative_error=err / total,
                codes_used=int(len(np.unique(codes))), n_val=int(len(val_x)), kmeans_iters=int(km.n_iter_),
                train_inertia_per_window=float(km.inertia_ / n)), codes


def run(args) -> None:
    out = out_dir(args.tag)
    (out / "codes").mkdir(parents=True, exist_ok=True)
    deadline = time.perf_counter() + args.hours * 3600 if args.hours else None
    rows_path = out / "fits.json"
    rows = json.loads(rows_path.read_text())["rows"] if rows_path.exists() else []
    done = {(r["dataset"], r["size"], r["seed"]) for r in rows}
    last = {}  # dataset -> seconds of the previous fit, for the budget guard
    for ds in args.datasets.split(","):
        raw, train_ids, val_x, n_val_rec = build_data(ds, args.recording_cap)
        whiten = PR.ANCHORS[ds].whiten
        sizes = sizes_for(len(train_ids), args.sizes.split(","))
        print(f"[{ds}] {len(train_ids)} training windows, {len(val_x)} held-out windows "
              f"({n_val_rec} validation recordings); sizes {[f'{l}={n}' for l, n in sizes]}", flush=True)
        for label, n in sizes:
            for seed in KMEANS_SEEDS:
                if (ds, label, seed) in done:
                    continue
                projected = last.get(ds, 0.0) * max(n / max(last.get(f"{ds}_n", n), 1), 1.0)
                if deadline is not None and time.perf_counter() + projected > deadline:
                    print(f"[{ds}] budget reached before size {label} seed {seed}; stopping (completed fits are kept)",
                          flush=True)
                    write_json_atomic(rows_path, dict(rows=rows))
                    return
                with threadpool_limits(limits=THREADS):
                    row, codes = fit_eval(raw, train_ids, n, seed, val_x, whiten)
                last[ds], last[f"{ds}_n"] = row["fit_seconds"], n
                row.update(dataset=ds, size=label)
                np.savez_compressed(out / "codes" / f"{ds}__{label}__s{seed}.npz", codes=codes)
                rows.append(row)
                done.add((ds, label, seed))
                write_json_atomic(rows_path, dict(rows=rows))
                print(f"[{ds}] n={n} ({label}) kmeans seed {seed}: fit {row['fit_seconds']:.0f}s, relative error "
                      f"{row['relative_error']:.4f}, {row['codes_used']} codes used on held-out windows", flush=True)


def summarize(tag: str) -> Path:
    out = out_dir(tag)
    rows = json.loads((out / "fits.json").read_text())["rows"]
    lines = ["# Fitting-sample size check (token level)", "", __doc__.split("\n\n")[0], "",
             "| dataset | fitting windows | fit s (mean) | held-out relative error | seed ARI | codes used |",
             "|---|---|---|---|---|---|"]
    cross = ["", "| dataset | pair | cross-size ARI | seed ARI of the smaller size | read |", "|---|---|---|---|---|"]
    for ds in dict.fromkeys(r["dataset"] for r in rows):
        by_size = {}
        for r in rows:
            if r["dataset"] == ds:
                by_size.setdefault(r["size"], {})[r["seed"]] = r

        def codes(size, seed):
            return np.load(out / "codes" / f"{ds}__{size}__s{seed}.npz")["codes"]

        seed_ari = {}
        for size, d in by_size.items():
            s = sorted(d)
            if len(s) >= 2:
                seed_ari[size] = adjusted_rand_score(codes(size, s[0]), codes(size, s[1]))
            n = d[s[0]]["n"]
            lines.append(f"| {ds} | {n} ({size}) | {np.mean([x['fit_seconds'] for x in d.values()]):.0f} | "
                         f"{np.mean([x['relative_error'] for x in d.values()]):.4f} | "
                         f"{seed_ari.get(size, float('nan')):.3f} | {int(np.mean([x['codes_used'] for x in d.values()]))} |")
        order = sorted(by_size, key=lambda k: by_size[k][min(by_size[k])]["n"])
        for i, a in enumerate(order):
            for b in order[i + 1:]:
                vals = [adjusted_rand_score(codes(a, sa), codes(b, sb)) for sa in by_size[a] for sb in by_size[b]]
                x, ref = float(np.mean(vals)), seed_ari.get(a, float("nan"))
                cross.append(f"| {ds} | {a} vs {b} | {x:.3f} | {ref:.3f} | "
                             + ("size moves the codes no more than seed noise" if x >= ref - 0.02 else
                                "size moves the codes more than seed noise") + " |")
    text = "\n".join(lines + cross) + "\n"
    path = out / "REPORT.md"
    path.write_text(text, encoding="utf-8")
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--sizes", default="10000,40000,all")
    ap.add_argument("--hours", type=float, default=None, help="elapsed-time budget; completed fits are kept")
    ap.add_argument("--tag", default="sample_size_check")
    ap.add_argument("--recording-cap", type=int, default=None, help="smoke tests only")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)
    if not args.report_only:
        run(args)
    print(summarize(args.tag))


if __name__ == "__main__":
    main()
