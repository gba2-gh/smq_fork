"""Generate REPORT.md for a Stage C run from saved artifacts only (no
refitting), per docs/plans/STAGE_C_AGENT_INSTRUCTIONS.md §2.6.

Usage:
    python -m script.action_transport.report_stage_c --run-id stage_c_pooled_001
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.action_transport import config
from script.action_transport.report import fmt, load_csv, num
from script.action_transport.run_stage_c import ARM_LABELS, OUT_ROOT

METRICS = ["MoF", "Edit", "F1@10", "F1@25", "F1@50"]
STAGEA_ARMS = ["no_temporal", "no_temporal_filter", "categorical_asot", "categorical_asot_kc", "continuous_asot"]
STAGEC_ARMS = ["context_only", "context_only_filter", "contextual_asot"]
DIAGNOSTICS = [("run_ratio", "pred/GT runs"), ("pred_run_median_s", "median run (s)"),
               ("occupied_states", "states used"), ("max_state_fraction", "largest state"),
               ("mi_state_subject", "MI(state;subject)")]


def group(rows: list[dict]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for r in rows:
        if r.get("status") in ("invalid", "unavailable", "invalid_input") or num(r.get("MoF")) is None:
            continue
        out.setdefault((r["dataset"], r["normalize"], r["arm"], r["setting"]), []).append(r)
    return out


def mean_by(groups: dict, key: tuple, field: str) -> float | None:
    vals = [v for v in (num(r.get(field)) for r in groups.get(key, [])) if v is not None]
    return statistics.mean(vals) if vals else None


def wins(groups_a: dict, key_a: tuple, groups_b: dict, key_b: tuple, field: str = "F1@50") -> tuple[int, int]:
    a = {r["seed"]: num(r[field]) for r in groups_a.get(key_a, [])}
    b = {r["seed"]: num(r[field]) for r in groups_b.get(key_b, [])}
    seeds = sorted(set(a) & set(b))
    return sum(a[s] > b[s] for s in seeds), len(seeds)


def primary_bar_table(c_groups: dict, a_groups: dict, smq: dict) -> tuple[list[str], bool]:
    """Plan v1.7's operationalized primary bar (THREE_STAGE_MOTION_PLAN.md
    §4 "Version 1.7"): contextual_asot's 3-seed mean F1@50 must beat both
    A-cont and SMQ, per (dataset, normalize), under T."""
    out = ["| Dataset | Norm | contextual_asot mean F1@50 | A-cont mean F1@50 | beats A-cont | "
          "SMQ F1@50 | beats SMQ | wins vs A-cont | wins vs SMQ |",
          "|---|---|---|---|---|---|---|---|---|"]
    all_hold = True
    for dataset in config.DATASETS:
        for normalize in ("False", "True"):
            key_ctx = (dataset, normalize, "contextual_asot", "T")
            key_cont = (dataset, normalize, "continuous_asot", "T")
            m_ctx = mean_by(c_groups, key_ctx, "F1@50")
            m_cont = mean_by(a_groups, key_cont, "F1@50")
            smq_val = num(smq.get(dataset, {}).get("F1@50"))
            beats_cont = m_ctx is not None and m_cont is not None and m_ctx > m_cont
            beats_smq = m_ctx is not None and smq_val is not None and m_ctx > smq_val
            all_hold &= bool(beats_cont) and bool(beats_smq)
            w_cont = wins(c_groups, key_ctx, a_groups, key_cont)
            a_seed_rows = {r["seed"]: num(r["F1@50"]) for r in c_groups.get(key_ctx, [])}
            w_smq = (sum(v > smq_val for v in a_seed_rows.values()), len(a_seed_rows)) if smq_val is not None else (0, 0)
            ctx_cell = f"{m_ctx:.2f}" if m_ctx is not None else "not run"
            cont_cell = f"{m_cont:.2f}" if m_cont is not None else "not run"
            smq_cell = f"{smq_val:.2f}" if smq_val is not None else "not run"
            out.append(f"| {dataset} | {'on' if normalize == 'True' else 'off'} | {ctx_cell} | {cont_cell} | "
                      f"{'YES' if beats_cont else 'no'} | {smq_cell} | {'YES' if beats_smq else 'no'} | "
                      f"{w_cont[0]}/{w_cont[1]} | {w_smq[0]}/{w_smq[1]} |")
    return out, all_hold


def partial_result_table(c_groups: dict) -> tuple[list[str], bool]:
    """Plan v1.7: context-only's median-run-duration ratio must lie in
    [0.5, 2] in every (dataset, normalize) for the partial-result clause."""
    out = ["| Dataset | Norm | context_only pred/GT median run ratio (mean of 3 seeds) | in [0.5, 2] |",
          "|---|---|---|---|"]
    all_hold = True
    for dataset in config.DATASETS:
        for normalize in ("False", "True"):
            rows = c_groups.get((dataset, normalize, "context_only", "T"), [])
            ratios = [num(r.get("pred_run_median_s")) / num(r.get("gt_run_median_s"))
                     for r in rows if num(r.get("pred_run_median_s")) and num(r.get("gt_run_median_s"))]
            mean_ratio = statistics.mean(ratios) if ratios else None
            ok = mean_ratio is not None and 0.5 <= mean_ratio <= 2.0
            all_hold &= bool(ok)
            ratio_cell = f"{mean_ratio:.2f}" if mean_ratio is not None else "not run"
            out.append(f"| {dataset} | {'on' if normalize == 'True' else 'off'} | {ratio_cell} | "
                      f"{'yes' if ok else 'no'} |")
    return out, all_hold


def table(groups: dict, arms: list[str], setting: str, field: str, digits: int = 2) -> str:
    header = [ARM_LABELS.get(a, a) for a in arms]
    lines = ["| Dataset | Norm | " + " | ".join(header) + " |", "|---|---|" + "---|" * len(header)]
    for dataset in config.DATASETS:
        for normalize in ("False", "True"):
            present, cells = False, []
            for arm in arms:
                vals = [v for v in (num(r.get(field)) for r in groups.get((dataset, normalize, arm, setting), []))
                       if v is not None]
                present |= bool(vals)
                cells.append(fmt(vals, digits) if vals else "not run")
            if present:
                lines.append(f"| {dataset} | {'on' if normalize == 'True' else 'off'} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def paired_differences(groups: dict, arm_a: str, arm_b: str, setting: str, field: str = "F1@50") -> str:
    lines = [f"| Dataset | Norm | {ARM_LABELS.get(arm_a, arm_a)} - {ARM_LABELS.get(arm_b, arm_b)} per seed | "
            "mean | wins |", "|---|---|---|---|---|"]
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


def order_diagnostic_table(rows: list[dict], c_groups: dict) -> str:
    perm = [r for r in rows if r["arm"] == "contextual_order_perm"]
    lines = ["| Dataset | Norm | Setting | contextual_asot (seed 111) F1@50 | order-diagnostic F1@50 |",
            "|---|---|---|---|---|"]
    for r in sorted(perm, key=lambda x: (x["dataset"], x["normalize"], x["setting"])):
        original = [x for x in c_groups.get((r["dataset"], r["normalize"], "contextual_asot", r["setting"]), [])
                   if str(x["seed"]) == str(config.SAMPLING_SEED)]
        orig_f1 = f"{num(original[0]['F1@50']):.2f}" if original else "--"
        lines.append(f"| {r['dataset']} | {'on' if r['normalize'] == 'True' else 'off'} | {r['setting']} | "
                    f"{orig_f1} | {num(r['F1@50']):.2f} |")
    return "\n".join(lines)


def stage_b_summary(stage_b_run: str) -> list[str]:
    results_path = config.OUT_ROOT.parent / "action_transport_stage_b" / stage_b_run / "results.json"
    if not results_path.exists():
        return ["Not found."]
    results = json.loads(results_path.read_text())
    lines = ["| Cell | stopped_by | steps_run | model_acc | copy_acc | margin |", "|---|---|---|---|---|---|"]
    for key, v in sorted(results.items()):
        acc, copy = v.get("model_masked_acc"), v.get("diagnostics", {}).get("neighbor_baseline_acc")
        margin = acc - copy if acc is not None and copy is not None else None
        lines.append(f"| {key} | {v.get('stopped_by')} | {v.get('steps_run')} | "
                     f"{acc:.3f} | {copy:.3f} | {margin:+.3f} |" if margin is not None else
                     f"| {key} | {v.get('stopped_by')} | {v.get('steps_run')} | -- | -- | -- |")
    return lines


def build_report(run_dir: Path, stage_a_dir: Path) -> str:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows = load_csv(run_dir / "cells.csv")
    not_run = load_csv(run_dir / "not_run.csv")
    planned = load_csv(run_dir / "planned_cells.csv")
    planned_ids = {pid for p in planned for pid in p["prediction_set_ids"].split(";") if pid}
    done_ids = {r["prediction_set_id"] for r in rows}
    c_groups = group(rows)

    a_rows = load_csv(stage_a_dir / "cells.csv")
    a_groups = {}
    for r in a_rows:
        r = dict(r)
        r.setdefault("setting", config.setting_of(r["prediction_set_id"]))
        if r.get("status") in ("invalid", "unavailable") or num(r.get("MoF")) is None:
            continue
        a_groups.setdefault((r["dataset"], r["normalize"], r["arm"], r["setting"]), []).append(r)
    smq = {r["dataset"]: r for r in a_rows if r["arm"] == config.SMQ_BASELINE_ARM}

    bar_lines, bar_met = primary_bar_table(c_groups, a_groups, smq)
    partial_lines, partial_met = partial_result_table(c_groups)
    if bar_met:
        verdict = "**Primary bar met**: contextual ASOT beats both A-cont and SMQ on F1@50, under T, in every (dataset, normalization)."
    elif partial_met:
        verdict = "**Partial result**: the primary bar is not met, but context alone removes most of Stage A's fragmentation (context, not transport, is doing the work)."
    else:
        verdict = "**Neither the primary bar nor the partial-result clause is met.** Per plan §4 v1.7, the discrete-code-plus-transport direction is not supported by this evidence."

    out = [f"# Stage C report -- {manifest['run_id']}", "",
          f"Generated from saved artifacts only (no refitting). Runner version {manifest.get('stagec_version')}, "
          f"stage_b_run `{manifest.get('stage_b_run')}`, stage_a_run `{manifest.get('stage_a_run')}`.", "",
          f"**Coverage:** {len(done_ids & planned_ids)}/{len(planned_ids)} planned prediction sets scored; "
          f"{len(not_run)} fit(s) not run.", ""]
    if not_run:
        out += ["| Not-run cell | Reason |", "|---|---|"]
        out += [f"| {n['cell_key']} | {n['reason']} |" for n in not_run]
        out.append("")

    out += ["## 1. Precommitted decision (plan §4, v1.7)", "", verdict, "",
           "### Primary bar: contextual ASOT vs A-cont and SMQ (F1@50, T, pooled, mean of 3 seeds)", ""]
    out += bar_lines + ["", "### Partial-result clause: context-only fragmentation ratio in [0.5, 2]", ""]
    out += partial_lines + [""]

    out += ["## 2. Comparison matrix (representation x temporal treatment), setting T, F1@50", "",
           "| Representation | No temporal | Fixed filter | Temporal transport |", "|---|---|---|---|"]
    for dataset in config.DATASETS:
        for normalize in ("False", "True"):
            def cell(arm, grp):
                vals = [v for v in (num(r.get("F1@50")) for r in grp.get((dataset, normalize, arm, "T"), []))
                       if v is not None]
                return fmt(vals) if vals else "not run"
            out.append(f"| Hard K=500 ({dataset}/{'norm' if normalize == 'True' else 'raw'}) | "
                      f"{cell('no_temporal', a_groups)} | {cell('no_temporal_filter', a_groups)} | "
                      f"{cell('categorical_asot', a_groups)} |")
            out.append(f"| B-code context ({dataset}/{'norm' if normalize == 'True' else 'raw'}) | "
                      f"{cell('context_only', c_groups)} | {cell('context_only_filter', c_groups)} | "
                      f"{cell('contextual_asot', c_groups)} |")
    out.append("")

    out += ["## 3. Paired seed differences (F1@50, setting T)", "",
           "**Adding transport to context** (contextual_asot - context_only):", "",
           paired_differences(c_groups, "contextual_asot", "context_only", "T"), "",
           "**Transport vs the fixed smoother on context** (contextual_asot - context_only_filter):", "",
           paired_differences(c_groups, "contextual_asot", "context_only_filter", "T"), ""]

    out += ["## 4. All five metrics (setting T)", ""]
    for metric in METRICS:
        out += [f"**{metric}**", "", table(c_groups, STAGEC_ARMS, "T", metric), ""]

    out += ["## 5. Fragmentation diagnostics (setting T)", ""]
    for field, name in DIAGNOSTICS:
        out += [f"**{name}**", "", table(c_groups, STAGEC_ARMS, "T", field, digits=3), ""]

    out += ["## 6. Contextual-order diagnostic (seed 111, pooled only)", "",
           "Stage B's frozen checkpoint fed permuted codes, embeddings restored to original order, "
           "prototypes refit, transport with the original chronological kernel.", "",
           order_diagnostic_table(rows, c_groups), ""]

    out += ["## 7. `stage_b_003` summary", ""] + stage_b_summary(manifest.get("stage_b_run", "")) + [""]

    out += ["## 8. Limitations", "",
           "- No K=C contextualizer exists, so beating A1-KC cannot isolate vocabulary size from context.",
           "- The frozen encoder is transductive; per-recording normalization and inference are offline.",
           "- Equal T/E coefficients do not calibrate the 128-d contextual cosine cost the same way as the "
           "64-d PCA cosine or the categorical cost.",
           "- Pooled protocol only; subject-disjoint Stage C was not run.",
           "- One selected contextualizer, chosen by a label-free tuning criterion on one seed.", ""]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_dir = OUT_ROOT / args.run_id
    if not (run_dir / "cells.csv").exists():
        raise SystemExit(f"no cells.csv at {run_dir}")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    stage_a_dir = config.OUT_ROOT / manifest["stage_a_run"]
    out_path = run_dir / "REPORT.md"
    out_path.write_text(build_report(run_dir, stage_a_dir), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
