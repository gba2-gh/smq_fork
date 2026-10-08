"""Run outputs: manifest tables, gate evaluation (Section 10 of the instructions) and REPORT.md."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from script.compositional_dynamics import config as C

# Declared before any scored run (instructions Section 10).
GATES = (
    ("m1_ordered", "b0_centre", 5.0, "Ordered minus matched B0-centre"),
    ("m1_ordered", "m1_bag", 3.0, "Ordered minus bag"),
    ("m1_ordered", "occupancy", 3.0, "Ordered minus occupancy"),
)
MOF_FLOOR = -2.0


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def load_cells(run_dir: Path) -> dict:
    cells = {}
    for d in sorted((run_dir / "cells").glob("*__seed*")):
        info = _read(d / "info.json")
        if not info:
            continue
        done = info["status"] == "complete"
        cells[(info["arm"], info["seed"], info.get("split", "primary"))] = dict(info=info, scores=_read(d / "scores.json") if done else None,
                                                   diag=_read(d / "diagnostics.json") if done else None,
                                                   training=_read(d / "training.json") if done else None)
    return cells


def primary(cell: dict, pop: str = "eval") -> dict | None:
    sc = cell["scores"]
    return sc[pop][str(sc["primary_endpoint_steps"])]["metrics"] if sc else None


def evaluate_gate(cells: dict, seeds, smoke: bool, verification: dict | None) -> dict:
    cells = {(a, s): c for (a, s, sp), c in cells.items() if sp == "primary"}
    if smoke:
        return dict(verdict="not computed (smoke run)", comparisons=[])
    comps, missing = [], []
    for a, b, f1_min, label in GATES:
        diffs = {}
        for s in seeds:
            ca, cb = cells.get((a, s)), cells.get((b, s))
            ma, mb = (primary(ca) if ca else None), (primary(cb) if cb else None)
            if ma is None or mb is None:
                missing.append(f"{a}/{b} seed {s}")
                continue
            diffs[s] = dict(F1_50=ma["F1@50"] - mb["F1@50"], MoF=ma["MoF"] - mb["MoF"])
        row = dict(label=label, a=a, b=b, per_seed=diffs, f1_min=f1_min)
        if len(diffs) == len(seeds):
            f1 = [d["F1_50"] for d in diffs.values()]
            mof = [d["MoF"] for d in diffs.values()]
            row.update(mean_F1_50=float(np.mean(f1)), mean_MoF=float(np.mean(mof)), sd_F1_50=float(np.std(f1)),
                       sd_MoF=float(np.std(mof)),
                       passed=bool(np.mean(f1) >= f1_min and np.mean(mof) >= MOF_FLOOR and all(x > 0 for x in f1)))
        comps.append(row)
    ok_verif = bool(verification and verification.get("passed") and verification.get("code_hash") == C.code_hash())
    if missing or any("passed" not in r for r in comps) or not ok_verif or len(seeds) != len(C.SEEDS):
        why = ([f"missing/invalid cells: {sorted(set(missing))}"] if missing else [])
        why += [] if ok_verif else ["verification (tests) not recorded as passed for this code hash"]
        if len(seeds) != len(C.SEEDS):
            why.append("not all three seeds were requested")
        return dict(verdict="INCONCLUSIVE", reasons=why, comparisons=comps)
    return dict(verdict="PASSED" if all(r["passed"] for r in comps) else "FAILED", comparisons=comps)


def fold_means(cells: dict, arm: str, seed: int) -> dict | None:
    """Mean evaluation metrics over the subject-disjoint folds; needs every fold complete."""
    ms = []
    for k in range(C.SUBJECT_FOLDS):
        c = cells.get((arm, seed, f"fold{k}"))
        m = primary(c) if c else None
        if m is None:
            return None
        ms.append(m)
    return {key: float(np.mean([m[key] for m in ms])) for key in ms[0]}


def secondary_sections(cells: dict, seeds, gate: dict) -> list:
    """Secondary readouts. They are descriptive: they cannot rescue a failed gate or overturn a passed one."""
    L = []
    mult = {}
    for arm in C.ARMS:
        for s in seeds:
            c = cells.get((arm, s, "primary"))
            ms = c and c["scores"] and c["scores"].get("multiplier_sensitivity")
            if ms:
                mult[(arm, s)] = dict(ms, **{"1.0": primary(c)})
    if mult:
        L += ["", "## Secondary: final-decode cost multipliers (primary split, frozen models; multiplier 1 is the primary result)", "",
              "| arm | seed | multiplier | MoF | F1@50 |", "|---|---|---|---|---|"]
        for (arm, s), d in sorted(mult.items()):
            for k in sorted(d, key=float):
                L.append(f"| {arm} | {s} | {k} | {d[k]['MoF']:.2f} | {d[k]['F1@50']:.2f} |")
    fm = {(a, s): fold_means(cells, a, s) for a in C.ARMS for s in seeds}
    if any(v for v in fm.values()):
        L += ["", "## Secondary: subject-disjoint folds (mean over 4 folds of 4 held-out subjects; one Hungarian mapping per fold)", "",
              "The tokenizer was still fitted on all recordings, so this is not fully inductive.", "",
              "| arm | seed | MoF | Edit | F1@50 |", "|---|---|---|---|---|"]
        for (a, s), v in sorted(fm.items()):
            L.append(f"| {a} | {s} | " + (" | ".join(f"{v[k]:.2f}" for k in ("MoF", "Edit", "F1@50")) if v else "incomplete | | ") + " |")
        L += ["", "Descriptive comparisons (same thresholds as the gate, not a gate):", ""]
        agree = []
        for a, b, f1_min, label in GATES:
            diffs = [(fm[(a, s)]["F1@50"] - fm[(b, s)]["F1@50"], fm[(a, s)]["MoF"] - fm[(b, s)]["MoF"]) for s in seeds if fm.get((a, s)) and fm.get((b, s))]
            if len(diffs) != len(seeds):
                L.append(f"- {label}: incomplete")
                continue
            f1, mof = [d[0] for d in diffs], [d[1] for d in diffs]
            ok = np.mean(f1) >= f1_min and np.mean(mof) >= MOF_FLOOR and all(x > 0 for x in f1)
            agree.append(ok)
            L.append(f"- {label}: mean F1@50 {np.mean(f1):+.2f}, mean MoF {np.mean(mof):+.2f}, per-seed F1 {', '.join(f'{x:+.2f}' for x in f1)} -> {'meets' if ok else 'does not meet'} the thresholds")
        prim = [r.get("passed") for r in gate.get("comparisons", []) if "passed" in r]
        if prim and len(agree) == len(prim) and prim != agree:
            L.append("- **WARNING: the subject-disjoint readout disagrees with the primary gate on at least one comparison.** Report both; neither overrides the other.")
    return L


def write_csv(path: Path, rows: list) -> None:
    fields = sorted({k for r in rows for k in r}) if rows else ["empty"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def finalize(run_dir: Path, arms, seeds, scale: C.Scale, not_run: list) -> None:
    cells = load_cells(run_dir)
    rows, tr, trs = [], [], []
    for (arm, seed, split), c in sorted(cells.items()):
        info = c["info"]
        row = dict(arm=arm, seed=seed, split=split, status=info["status"], seconds=info.get("seconds"), n_fit=info.get("n_fit"),
                   n_eval=info.get("n_eval"))
        for pop in ("eval", "fit"):
            m = primary(c, pop) if c["scores"] else None
            for k in ("MoF", "Edit", "F1@10", "F1@25", "F1@50"):
                row[f"{pop}_{k}"] = None if m is None else m[k]
        if c["diag"]:
            for pop in ("fit", "eval"):
                sv = c["diag"]["solver"][pop]
                trs.append(dict(arm=arm, seed=seed, split=split, population=pop, status_counts=json.dumps(sv["status_counts"]),
                                steps_total=sv["steps_total"], residual_max=sv["residual_max"]))
        for rf in (c["training"] or {}).get("refreshes", []):
            tr.append(dict(arm=arm, seed=seed, split=split, **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in rf.items()}))
        rows.append(row)
    write_csv(run_dir / "cells.csv", rows)
    write_csv(run_dir / "training.csv", tr)
    write_csv(run_dir / "transport.csv", trs)
    write_csv(run_dir / "not_run.csv", not_run)
    verification = _read(C.RESULTS / "verification.json")
    gate = evaluate_gate(cells, seeds, bool(scale.smoke_fit), verification)
    parity = _read(C.RESULTS / "parity" / "parity.json")
    (run_dir / "gates.json").write_text(json.dumps(dict(
        gate=gate, verification=verification,
        parity=parity and {k: parity[k] for k in ("exact_match", "max_abs_diff", "seed")}), indent=2, default=str))
    L = [f"# Milestone 1 run {run_dir.name}", "",
         f"Scale `{scale.name}`" + (" (**SMOKE: plumbing check, not evidence**)" if scale.smoke_fit else ""), "",
         "## Evaluation population, final-solver endpoint (one Hungarian mapping fitted globally over the evaluation recordings)", "",
         "| arm | seed | status | MoF | Edit | F1@10 | F1@25 | F1@50 |", "|---|---|---|---|---|---|---|---|"]
    for r in [r for r in rows if r["split"] == "primary"]:
        v = [r[f"eval_{k}"] for k in ("MoF", "Edit", "F1@10", "F1@25", "F1@50")]
        L.append(f"| {r['arm']} | {r['seed']} | {r['status']} | " + " | ".join("-" if x is None else f"{x:.2f}" for x in v) + " |")
    L += ["", "## Frozen feasibility gate (declared before the run)", "", f"**{gate['verdict']}**", ""]
    L += [f"- {reason}" for reason in gate.get("reasons", [])]
    for r in gate["comparisons"]:
        per = ", ".join(f"{s}: F1 {d['F1_50']:+.2f} / MoF {d['MoF']:+.2f}" for s, d in r["per_seed"].items())
        tail = ("; incomplete" if "passed" not in r else
                f"; mean F1 {r['mean_F1_50']:+.2f}, mean MoF {r['mean_MoF']:+.2f}, SD {r['sd_F1_50']:.2f}/{r['sd_MoF']:.2f} -> "
                f"{'pass' if r['passed'] else 'fail'}")
        L.append(f"- {r['label']} (need mean F1@50 >= +{r['f1_min']:g}, mean MoF >= {MOF_FLOOR:g}, all seeds positive): {per}{tail}")
    L += secondary_sections(cells, seeds, gate)
    if not_run:
        L += ["", "## Not run", ""] + [f"- {n['arm']} seed {n['seed']} {n.get('split', 'primary')}: {n['reason']}" for n in not_run]
    L += ["", "Three initialization seeds on one fixed split are not confidence intervals. Details: cells.csv, training.csv, "
          "transport.csv, per-cell diagnostics.json."]
    (run_dir / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[report] wrote {run_dir / 'REPORT.md'}; gate: {gate['verdict']}")
