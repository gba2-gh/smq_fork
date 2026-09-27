"""Generate REPORT.md from a run's saved artifacts only (cells.csv,
not_run.csv, manifest.json, planned_cells.csv) -- no refitting (instructions
§7). Works on v1.3 runs (fewer columns) and v1.3.1 runs.

Usage:
    python -m script.action_transport.report --run-id stage_a_pooled_002
    python -m script.action_transport.report --run-id stage_a_pooled_002 --compare-to stage_a_pooled_001
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.action_transport import config

METRICS = ["MoF", "Edit", "F1@10", "F1@25", "F1@50"]
ARMS = ["no_temporal", "no_temporal_filter", "categorical_asot", "categorical_asot_kc", "continuous_asot"]
LABEL = {"no_temporal": "A0", "no_temporal_filter": "A0+filter", "categorical_asot": "A1",
         "categorical_asot_kc": "A1-KC", "continuous_asot": "A-cont", "permuted_asot": "A1-permuted",
         config.SMQ_BASELINE_ARM: "SMQ"}
DIAGNOSTICS = [("run_ratio", "pred/GT runs"), ("pred_run_median_s", "median run (s)"),
               ("gt_run_median_s", "GT median run (s)"), ("occupied_states", "states used"),
               ("max_state_fraction", "largest state"), ("mi_state_subject", "MI(state;subject)")]


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def normalize_row(r: dict) -> dict:
    """Fill v1.3 rows (no setting/scope columns, filter rows under arm
    no_temporal) with the v1.3.1 conventions."""
    r = dict(r)
    pid = r["prediction_set_id"]
    r.setdefault("setting", config.setting_of(pid))
    r.setdefault("scope", "pooled")
    if "_filter__" in pid:
        r["arm"] = "no_temporal_filter"
    return r


def num(value) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def fmt(values: list[float], digits: int = 2) -> str:
    if not values:
        return "--"
    if len(values) == 1:
        return f"{values[0]:.{digits}f} (n=1)"
    return f"{statistics.mean(values):.{digits}f} ± {statistics.pstdev(values):.{digits}f} (n={len(values)})"


def group(rows: list[dict], scope: str) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for r in rows:
        if r["scope"] != scope or r["status"] in ("invalid", "unavailable") or num(r.get("MoF")) is None:
            continue
        out.setdefault((r["dataset"], r["normalize"], r["arm"], r["setting"]), []).append(r)
    return out


def values(groups, key, field) -> list[float]:
    return [v for v in (num(r.get(field)) for r in groups.get(key, [])) if v is not None]


def table(groups: dict, arms: list[str], setting: str, field: str, baselines: dict | None = None,
          digits: int = 2) -> str:
    header = (["SMQ"] if baselines is not None else []) + [LABEL[a] for a in arms]
    lines = ["| Dataset | Norm | " + " | ".join(header) + " |", "|---|---|" + "---|" * len(header)]
    for dataset in config.DATASETS:
        for normalize in ("False", "True"):
            cells = []
            if baselines is not None:
                b = baselines.get(dataset)
                cells.append(f"{num(b[field]):.{digits}f}" if b and num(b.get(field)) is not None else "--")
            present = False
            for arm in arms:
                vals = values(groups, (dataset, normalize, arm, setting), field)
                present |= bool(vals)
                cells.append(fmt(vals, digits) if vals else "not run")
            if present:
                lines.append(f"| {dataset} | {'on' if normalize == 'True' else 'off'} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def paired_differences(groups: dict, arm_a: str, arm_b: str, setting: str, field: str = "F1@50") -> str:
    """Seed-paired differences arm_a - arm_b (same dataset/normalize/seed)."""
    lines = [f"| Dataset | Norm | {LABEL[arm_a]} - {LABEL[arm_b]} per seed | mean | wins |", "|---|---|---|---|---|"]
    for dataset in config.DATASETS:
        for normalize in ("False", "True"):
            a = {r["seed"]: num(r[field]) for r in groups.get((dataset, normalize, arm_a, setting), [])}
            b = {r["seed"]: num(r[field]) for r in groups.get((dataset, normalize, arm_b, setting), [])}
            seeds = sorted(set(a) & set(b))
            if not seeds:
                continue
            diffs = [a[s] - b[s] for s in seeds]
            lines.append(f"| {dataset} | {'on' if normalize == 'True' else 'off'} | "
                         + ", ".join(f"{d:+.2f}" for d in diffs)
                         + f" | {statistics.mean(diffs):+.2f} | {sum(d > 0 for d in diffs)}/{len(diffs)} |")
    return "\n".join(lines)


def convergence_table(rows: list[dict]) -> str:
    lines = ["| Arm | Setting | fit outer: converged / capped / invalid | final solves (recordings): "
             "converged / capped / invalid | max residual | median outer iters |", "|---|---|---|---|---|---|"]
    by: dict[tuple, list[dict]] = {}
    for r in rows:
        if r["arm"] in (config.SMQ_BASELINE_ARM, "no_temporal_filter") or r["scope"] == "oof":
            continue
        by.setdefault((r["arm"], r["setting"]), []).append(r)
    for (arm, setting), rs in sorted(by.items()):
        fit = Counter(r.get("fit_outer_status") or "?" for r in rs)
        conv = sum(int(num(r.get("final_n_converged")) or 0) for r in rs)
        cap = sum(int(num(r.get("final_n_capped")) or 0) for r in rs)
        inv = sum(int(num(r.get("final_n_invalid")) or 0) for r in rs)
        res = [v for v in (num(r.get("final_residual_max")) for r in rs) if v is not None]
        iters = [v for v in (num(r.get("fit_outer_iterations")) for r in rs) if v is not None]
        lines.append(f"| {LABEL.get(arm, arm)} | {setting} | {fit.get('converged', 0)} / {fit.get('capped', 0)} / "
                     f"{fit.get('invalid', 0)} | {conv} / {cap} / {inv} | "
                     f"{max(res):.1e} | {statistics.median(iters):.0f} |" if res and iters else
                     f"| {LABEL.get(arm, arm)} | {setting} | {fit.get('converged', 0)} / {fit.get('capped', 0)} / "
                     f"{fit.get('invalid', 0)} | {conv} / {cap} / {inv} | -- | -- |")
    return "\n".join(lines)


def comparison_section(out: list[str], groups: dict, baselines: dict, scope_label: str) -> None:
    for setting in ("T", "E"):
        tag = "primary" if setting == "T" else "secondary sensitivity"
        out += [f"### {scope_label}: F1@50, setting {setting} ({tag})", "",
                table(groups, ARMS, setting, "F1@50", baselines), ""]
    out += ["#### Paired seed differences (F1@50, setting T)", "",
            "1. **Claim** (fine vs class-matched vocabulary, same decoder):", "",
            paired_differences(groups, "categorical_asot", "categorical_asot_kc", "T"), "",
            "2. **Temporal assignment** (transport vs the fixed smoother):", "",
            paired_differences(groups, "categorical_asot", "no_temporal_filter", "T"), "",
            "3. **Discretization** (hard codes vs continuous features, same decoder):", "",
            paired_differences(groups, "categorical_asot", "continuous_asot", "T"), ""]


def compare_runs(rows: list[dict], other: list[dict]) -> str:
    theirs = {r["prediction_set_id"]: r for r in other}
    shared = [r for r in rows if r["prediction_set_id"] in theirs]
    if not shared:
        return "No shared prediction sets."
    worst = {m: 0.0 for m in METRICS}
    changed = []
    for r in shared:
        o = theirs[r["prediction_set_id"]]
        for m in METRICS:
            a, b = num(r.get(m)), num(o.get(m))
            if a is None or b is None:
                continue
            delta = abs(a - b)
            worst[m] = max(worst[m], delta)
            if delta > 0.01:
                changed.append((r["prediction_set_id"], m, b, a))
    lines = [f"{len(shared)} shared prediction sets. Largest absolute metric change: "
             + ", ".join(f"{m} {v:.2f}" for m, v in worst.items()) + "."]
    if changed:
        lines += ["", f"{len(changed)} metric values changed by more than 0.01 points "
                  "(expected only where near-tied states flip under the exact-log fix):", "",
                  "| Prediction set | Metric | before | after |", "|---|---|---|---|"]
        lines += [f"| {pid} | {m} | {b:.2f} | {a:.2f} |" for pid, m, b, a in changed[:40]]
        if len(changed) > 40:
            lines.append(f"| ... {len(changed) - 40} more | | | |")
    return "\n".join(lines)


def build_report(run_dir: Path, compare_dir: Path | None) -> str:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows = [normalize_row(r) for r in load_csv(run_dir / "cells.csv")]
    not_run = load_csv(run_dir / "not_run.csv")
    planned = load_csv(run_dir / "planned_cells.csv")
    planned_ids = {pid for p in planned for pid in p["prediction_set_ids"].split(";") if pid}
    done_ids = {r["prediction_set_id"] for r in rows}
    protocol = manifest.get("protocol", "pooled")
    baselines = {r["dataset"]: r for r in rows if r["arm"] == config.SMQ_BASELINE_ARM}
    launches = manifest.get("launches", [])

    out = [f"# Stage A report -- {manifest['run_id']}", "",
           f"Generated from saved artifacts only (no refitting). Runner protocol version "
           f"{manifest.get('protocol_version')}, protocol `{protocol}`"
           + (f", after pooled run `{manifest['after_pooled_run']}`" if manifest.get("after_pooled_run") else "")
           + f", {len(launches) or 1} launch(es)"
           + (f" on {', '.join(sorted({l.get('host', '?') for l in launches}))}" if launches else "") + ".", "",
           f"**Coverage:** {len(done_ids & planned_ids)}/{len(planned_ids)} planned prediction sets scored; "
           f"{len(not_run)} fit(s) not run.", ""]
    if not_run:
        out += ["| Not-run cell | Reason |", "|---|---|"]
        out += [f"| {n['cell_key']} | {n['reason']} |" for n in not_run]
        out.append("")

    out += ["## Summary of what this run can and cannot show", "",
            "- F1@50 is the declared focal metric; all five metrics are tabulated below.",
            "- Setting T is primary. E re-decodes the same T-fitted parameters with different "
            "coefficients; it is a bundled sensitivity, not an E-fitted model.",
            "- A0+filter is one fixed smoother. Beating it is not superiority over all temporal models.",
            "- A-cont and A1 use different cost geometries under equal coefficients.",
            "- SMQ is a system comparison (different tokenizer and resolution), not an isolated K intervention.",
            "- Seed SD describes initialization sensitivity, not sampling uncertainty.", ""]

    if baselines:
        out += ["## SMQ baseline (same checkpoint, same evaluator)", "",
                "| Dataset | MoF | Edit | F1@10 | F1@25 | F1@50 | pred/GT runs |", "|---|---|---|---|---|---|---|"]
        for d, b in sorted(baselines.items()):
            out.append(f"| {d} | " + " | ".join(f"{num(b[m]):.2f}" for m in METRICS)
                       + f" | {num(b.get('run_ratio')) or 0:.2f} |")
        out += ["", "Provenance check: the segmodel audit reported 41.99 / 24.30 (HuGaDB MoF / F1@50) and "
                "37.38 / 16.39 (LARa) for the same codes and mapping.", ""]
    else:
        out += ["## SMQ baseline", "", "**Not scored in this run** (v1.3 runner). Q1/Q2 same-checkpoint values, "
                "cited for orientation only: HuGaDB 42.0 / 24.3, LARa 37.4 / 16.4 (MoF / F1@50).", ""]

    scopes = [("pooled", "Pooled")] if protocol == "pooled" else [("oof", "Out-of-fold"), ("fold", "Per fold")]
    for scope, label in scopes:
        groups = group(rows, scope)
        if not groups:
            continue
        out += [f"## Comparisons ({label})", ""]
        comparison_section(out, groups, baselines, label)
        out += [f"### All five metrics, setting T ({label})", ""]
        for metric in METRICS:
            out += [f"**{metric}**", "", table(groups, ARMS, "T", metric, baselines), ""]
        if any(num(r.get("run_ratio")) is not None for rs in groups.values() for r in rs):
            out += [f"### Segmentation diagnostics, setting T ({label})", "",
                    "Predicted runs merge adjacent equal raw states. A pred/GT ratio far above 1 means "
                    "over-segmentation; states used / largest state expose collapse; MI(state;subject) "
                    "is descriptive only (subject IDs are metadata).", ""]
            for field, name in DIAGNOSTICS:
                out += [f"**{name}**", "", table(groups, ARMS, "T", field, baselines, digits=3), ""]

    perm = [r for r in rows if r["arm"] == "permuted_asot"]
    if perm:
        pooled_groups = group(rows, "pooled")
        out += ["## Permutation adjacency diagnostic (not a primary comparison)", "",
                "A1 refit on within-recording permuted codes (seed 111), states restored to the original "
                "timeline before scoring. Compare with A1 seed 111 in the same row: the gap is what true "
                "temporal adjacency contributes, including emission-refitting effects.", "",
                "| Dataset | Norm | Setting | A1 seed 111 F1@50 | A1-permuted F1@50 | A1 run ratio | permuted run ratio |",
                "|---|---|---|---|---|---|---|"]
        for r in sorted(perm, key=lambda x: (x["dataset"], x["normalize"], x["setting"])):
            a1 = [x for x in pooled_groups.get((r["dataset"], r["normalize"], "categorical_asot", r["setting"]), [])
                  if str(x["seed"]) == str(config.SAMPLING_SEED)]
            a1_f1 = f"{num(a1[0]['F1@50']):.2f}" if a1 else "--"
            a1_rr = f"{num(a1[0].get('run_ratio')):.2f}" if a1 and num(a1[0].get("run_ratio")) else "--"
            p_rr = f"{num(r.get('run_ratio')):.2f}" if num(r.get("run_ratio")) else "--"
            out.append(f"| {r['dataset']} | {'on' if r['normalize'] == 'True' else 'off'} | {r['setting']} | "
                       f"{a1_f1} | {num(r['F1@50']):.2f} | {a1_rr} | {p_rr} |")
        out.append("")

    out += ["## Convergence", "",
            "Fit outer status counts prediction-set rows (each fit appears once per setting). Final solves "
            "count recordings. `capped` = declared cap reached with a finite, nonincreasing objective; "
            "capped results are included in every table above.", "", convergence_table(rows), ""]

    if compare_dir is not None:
        other = [normalize_row(r) for r in load_csv(compare_dir / "cells.csv")]
        out += [f"## Reproducibility against `{compare_dir.name}`", "", compare_runs(rows, other), ""]

    out += ["## Limitations", "",
            f"- Protocol `{protocol}` only" + ("; subject-disjoint transfer not run." if protocol == "pooled" else "."),
            "- The frozen encoder is transductive; per-recording normalization and inference are offline.",
            "- Local temporal consistency is not a learned action grammar.",
            "- Every planned arm and setting is shown above or marked not run; nothing was selected by score.", ""]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--compare-to", default=None, help="another run id to check reproducibility against")
    args = parser.parse_args()
    run_dir = config.OUT_ROOT / args.run_id
    if not (run_dir / "cells.csv").exists():
        raise SystemExit(f"no cells.csv at {run_dir}")
    compare_dir = config.OUT_ROOT / args.compare_to if args.compare_to else None
    if compare_dir is not None and not (compare_dir / "cells.csv").exists():
        raise SystemExit(f"no cells.csv at {compare_dir}")
    out_path = run_dir / "REPORT.md"
    out_path.write_text(build_report(run_dir, compare_dir), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
