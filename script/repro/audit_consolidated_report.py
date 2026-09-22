"""Read-only checks of existing experiment artifacts; writes a separate audit JSON.

Run from the repository root with the smq Python environment. No model training.
"""
import ast
import csv
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from script.exp.analyze_run import hungarian_map, score, jsd
from script.exp.t3_hsmm import hsmm_viterbi
from script.repro.e1_jsd import pair_jsd


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get(obj, key):
    for part in key.split("."):
        obj = obj[part]
    return obj


def main():
    out = {}
    base = ROOT / "results/exp/analysis"
    classes = dict(hugadb=10, lara=8, babel1=5)
    out["artifact_counts"] = {
        exp: len(list((base / exp).rglob("*.json")))
        for exp in ("T1", "T1_kmeans", "T2", "T3", "T4")
    }
    failures = []
    checked = 0
    for row in csv.DictReader((ROOT / "results/exp/summary_long.csv").open()):
        exp, ds, cond, key = (row[k] for k in ("exp", "dataset", "K", "metric"))
        if exp == "T1":
            folder = "T1_kmeans" if key.startswith("km.") else "T1"
            key = key.removeprefix("km.")
            files = sorted((base / folder / ds).glob(f"K{cond}_s*.json"))
            files = [f for f in files if "_thr" not in f.stem]
        elif exp == "T2":
            files = sorted((base / exp / ds).glob(f"{cond}_s*.json"))
        elif exp == "T3":
            files = sorted((base / exp / "T1" / ds).glob("*.json"))
        else:
            folder = base / ("T1" if cond == "base" else "T4") / ds
            pattern = f"K{classes[ds]}_s*.json" if cond == "base" else f"{cond}_s*.json"
            files = sorted(folder.glob(pattern))
        vals = []
        for f in files:
            r = read(f)
            if exp == "T3":
                metric = key.removeprefix("delta_")
                vals.append(r[cond][metric] - r["argmax"][metric])
            else:
                vals.append(get(r, key))
        ok = len(vals) == int(row["n"])
        if vals:
            ok &= math.isclose(statistics.mean(vals), float(row["mean"]), abs_tol=1e-9)
            if len(vals) > 1:
                ok &= math.isclose(statistics.stdev(vals), float(row["sd"]), abs_tol=1e-9)
        if not ok:
            failures.append(row)
        checked += 1
    out["summary_csv"] = dict(rows_checked=checked, mismatches=failures)

    configs = [read(f) for f in (ROOT / "models/exp").rglob("config.json")
               if "/T1/" in f.as_posix() or "/T4/" in f.as_posix()]
    out["training_configs"] = dict(count=len(configs),
        microbatch_nonnull=sum(c["micro_batch_size"] is not None for c in configs),
        tc_nonzero=sum(c["tc_weight"] != 0 for c in configs),
        epochs=sorted(set(c["epoch"] for c in configs)))
    out["e0_seed_stats"] = {}
    for ds in ("hugadb", "lara"):
        rs = [read(ROOT / f"results/e0/{ds}_seed{s}.json")["global"]
              for s in ("Default", "111", "222")]
        out["e0_seed_stats"][ds] = {
            k: dict(mean=statistics.mean(r[k] for r in rs),
                    sample_sd=statistics.stdev(r[k] for r in rs),
                    population_sd=statistics.pstdev(r[k] for r in rs)) for k in rs[0]
        }

    out["released_prediction_rescore"] = {}
    jsd_reference = {r["dataset"]: float(r["jsd_x100"]) for r in csv.DictReader(
        (ROOT / "results/e1_jsd/jsd_summary.csv").open())}
    out["released_many_to_one_rescore"] = {}
    for ds in ("hugadb", "lara", "babel1", "babel2", "babel3"):
        with np.load(ROOT / f"results/preds/{ds}_pretrained.npz", allow_pickle=True) as d:
            gts = [np.asarray(g, dtype=int) for g in d["gt"]]
            prs = [np.asarray(p, dtype=int) for p in d["pred"]]
            mapped = hungarian_map(gts, prs)
            metrics = score(gts, mapped)
            expected = read(ROOT / f"results/e0/{ds}_pretrained.json")["global"]
            assert all(abs(metrics[k] - expected[k]) <= .00051 for k in metrics)
            metrics["JSD"] = jsd(gts, mapped, [str(n) for n in d["names"]], ds, 1)
            assert abs(metrics["JSD"] - jsd_reference[ds]) <= .0051
            if ds in classes:
                ground, predicted = np.concatenate(gts), np.concatenate(prs)
                lut = np.zeros(int(predicted.max()) + 1, dtype=int)
                for code in np.unique(predicted):
                    labels, counts = np.unique(ground[predicted == code], return_counts=True)
                    lut[code] = labels[counts.argmax()]
                many = score(gts, [lut[p] for p in prs])
                expected_many = read(base / "pretrained" / f"{ds}.json")["b_many_to_one"]
                assert many == expected_many
                out["released_many_to_one_rescore"][ds] = many
            # Count boundaries excluded by D2's early continue.
            skipped_gt = skipped_pred = skipped_seqs = 0
            for g, p in zip(gts, mapped):
                ng, npred = int(np.count_nonzero(np.diff(g))), int(np.count_nonzero(np.diff(p)))
                if ng == 0 or npred == 0:
                    skipped_seqs += 1
                    skipped_gt += ng
                    skipped_pred += npred
            metrics["D2_skipped"] = dict(sequences=skipped_seqs, gt_boundaries=skipped_gt,
                                          predicted_boundaries=skipped_pred)
            out["released_prediction_rescore"][ds] = metrics
        print(f"Rescored {ds}", flush=True)

    identity_checks = 0
    for ds, c in classes.items():
        for f in (base / "T1" / ds).glob(f"K{c}_s*.json"):
            r = read(f)
            for merged in ("c1_geometry_merge", "c2_temporal_merge"):
                assert all(r[merged][k] == v for k, v in r["a_bijective"].items())
            identity_checks += 1
    out["K_equals_C_identity_checks"] = identity_checks

    # Execute the actual small diagnostic function without importing GPU dependencies.
    tree = ast.parse((ROOT / "script/repro/d_boundary_vs_label.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "boundary_f1")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "boundary_f1", "exec"), namespace)
    out["boundary_duplicate_counterexample"] = dict(
        predicted=[9, 11], ground_truth=[10], tolerance=1,
        implemented_f1=namespace["boundary_f1"](np.array([9, 11]), np.array([10]), 1),
        one_to_one_f1=100 * 2 / 3)
    out["JSD_identical_lengths"] = pair_jsd(np.array([20, 40, 60]), np.array([20, 40, 60]))

    # Independent exhaustive check of the HSMM recurrence, including transitions/durations.
    rng = np.random.default_rng(42)
    for _ in range(40):
        p, k = int(rng.integers(2, 7)), int(rng.integers(2, 4))
        e = rng.normal(size=(p, k))
        a = rng.normal(size=(k, k))
        dur = rng.normal(size=(k, 100))

        def objective(path):
            changes = [0] + [i for i in range(1, p) if path[i] != path[i-1]] + [p]
            total = -math.log(k) + sum(e[i, path[i]] for i in range(p))
            for idx, (start, end) in enumerate(zip(changes, changes[1:])):
                total += dur[path[start], end-start-1]
                if idx:
                    total += a[path[start-1], path[start]]
            return total

        decoded = hsmm_viterbi(e, dur, a)
        best = max(objective(path) for path in itertools.product(range(k), repeat=p))
        assert math.isclose(objective(decoded), best, abs_tol=1e-9)
    out["hsmm_exhaustive_cases_passed"] = 40
    assert not failures
    destination = ROOT / "results/CONSOLIDATED_AUDIT.json"
    destination.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
