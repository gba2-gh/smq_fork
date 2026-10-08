"""Markdown report for one cross-dataset run, written only from saved artifacts."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

PUBLISHED_F1_50 = {  # published F1@50 (SMQ paper Table 1-2, HiST-VQ paper Table 1-2, MASQ paper), different protocols
    "hugadb": dict(ASOT=3.0, HiST_VQ=28.3, MASQ=39.5),
    "lara": dict(ASOT=5.7, HiST_VQ=19.3, MASQ=22.2),
    "babel1": dict(ASOT=24.5, HiST_VQ=22.7, MASQ=21.0),
    "babel2": dict(ASOT=23.4, HiST_VQ=30.4, MASQ=28.4),
    "babel3": dict(ASOT=14.3, HiST_VQ=24.1, MASQ=24.5),
}
ORDER = ("hugadb", "babel1", "babel2", "babel3", "lara")


def read(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fmt(x, d: int = 2) -> str:
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return "n/a"


def table(headers: list[str], rows: list[list]) -> str:
    if not rows:
        return "_no rows_\n"
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def group_values(seg: list[dict], dataset: str, norm: str, role: str, first: bool = False) -> dict[int, float]:
    """seed -> F1@50 at the last (or first) rung."""
    rows = [r for r in seg if r["dataset"] == dataset and r["normalization"] == norm and r["role"] == role]
    out = {}
    for seed in sorted({int(r["seed"]) for r in rows}):
        mine = [r for r in rows if int(r["seed"]) == seed]
        pick = min(mine, key=lambda r: int(r["rung_steps"])) if first else max(mine, key=lambda r: int(r["rung_steps"]))
        out[seed] = float(pick["F1@50"])
    return out


def verdict(seg: list[dict], smq: dict[str, float]) -> tuple[str, list[str]]:
    """The declared reading guide, applied to the validation-selected checkpoint at the last rung."""
    notes = []

    def above(ds, norm):
        v = group_values(seg, ds, norm, "selected")
        if len(v) < 3 or ds not in smq:
            return None, None
        mean = statistics.mean(v.values())
        return bool(mean > smq[ds] and all(x > smq[ds] for x in v.values())), bool(mean < smq[ds])

    lara = [above("lara", n) for n in ("global", "per_recording")]
    babel = [above(d, "global") for d in ("babel1", "babel2", "babel3")]
    if any(a is None for a, _ in lara + babel):
        return "incomplete", ["not every LARa normalization and BABEL subset has three verified seeds"]
    lara_above, babel_above = sum(a for a, _ in lara), sum(a for a, _ in babel)
    lara_below_both, babel_below = all(b for _, b in lara), sum(b for _, b in babel)
    notes.append(f"LARa groups above SMQ (mean and every seed): {lara_above}/2; BABEL subsets above SMQ: {babel_above}/3; "
                 f"LARa below SMQ under both normalizations: {lara_below_both}; BABEL subsets below SMQ (mean): {babel_below}/3")
    if lara_above >= 1 and babel_above >= 2:
        return "transfers", notes
    if lara_below_both or babel_below >= 2:
        return "does not transfer", notes
    return "mixed", notes


def write_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    seg, cells = read(run_dir / "segmentation.csv"), read(run_dir / "cells.csv")
    not_run = read(run_dir / "not_run.csv")
    manifest = json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").exists() else {}
    gates = json.loads((run_dir / "gates.json").read_text()) if (run_dir / "gates.json").exists() else {}
    smoke = bool(manifest.get("scale", {}).get("smoke"))
    smq = {ds: g["smq_local_f1_50"] for ds, g in gates.get("label_loading_and_scoring", {}).items() if "smq_local_f1_50" in g}
    lines = [f"# Raw K-means tokens across datasets — {manifest.get('run_id', run_dir.name)}", ""]
    if smoke:
        lines += ["> **SMOKE RUN — shrunken budgets on small recording subsets. Not a reproduction, not scientific "
                  "evidence.**", ""]
    lines += [f"Manifest: `manifest.json` (status `{manifest.get('status')}`, implementation "
              f"`{manifest.get('implementation_hash')}`). Cells with verified training: {len(cells)}/"
              f"{manifest.get('cells')}; unfinished units: {len(not_run)} (`not_run.csv`).", "",
              "Frozen HuGaDB configuration on every dataset (raw patches, K-means-512, reference contextualizer, guard 0, "
              "5,000 updates, outer cap 50). Only window, frame rate and class count differ by dataset. Tokenizer and "
              "OT fitting pool all recordings, validation included, so decoding is transductive; HuGaDB settings were "
              "chosen on HuGaDB development scores; LARa is not untouched (it informed earlier choices).", ""]

    lines += ["## Gates", ""]
    g3 = gates.get("label_loading_and_scoring", {})
    lines += [table(["dataset", "labels = saved SMQ ground truth", "max |metric diff| vs saved SMQ scores", "passed"],
                    [[ds, g.get("labels_equal_saved_gt"), fmt(max((abs(v) for v in g.get("metric_diffs", {}).values()),
                                                              default=float("nan")), 4), g.get("passed")]
                     for ds, g in g3.items()])]
    for name, label in (("tokenizer_port", "Gate 1 — tokenizer port vs qic_v3"), ("pipeline_port", "Gate 2 — pipeline port vs ctx_cal_v1")):
        g = gates.get(name, {})
        lines += [f"**{label}:** {g.get('status', 'not run')}"
                  + (f" (F1@50 {fmt(g.get('f1_50_last_rung'))} vs {fmt(g.get('reference_f1_50_last_rung'))})"
                     if "f1_50_last_rung" in g else ""), ""]
    if any(g.get("status") == "failed" for g in (gates.get("tokenizer_port", {}), gates.get("pipeline_port", {}))):
        lines += ["**A port-regression gate failed; dependent cells are not interpreted.** See `gates.json` and "
                  "`gate_mismatch_environment.json`.", ""]

    lines += ["## F1@50 by dataset (mean ± sd over three paired seeds)", "",
              "Validation-selected checkpoint (primary) and final 5,000-update checkpoint (secondary), last final-solve "
              "rung (4,000 in a full run). SMQ is the released checkpoint re-scored locally; the published numbers use "
              "different protocols (end-to-end training, 1-second patches, K = number of classes).", ""]
    rows = []
    for ds in ORDER:
        for norm in ("global", "per_recording"):
            sel, fin = group_values(seg, ds, norm, "selected"), group_values(seg, ds, norm, "final")
            if not sel and not fin:
                continue
            ms = lambda v: f"{statistics.mean(v.values()):.2f} ± {statistics.stdev(v.values()) if len(v) > 1 else 0:.2f}" if v else "n/a"
            pub = PUBLISHED_F1_50.get(ds, {})
            rows.append([ds, norm, len(sel), ms(sel), " / ".join(fmt(sel.get(s)) for s in sorted(sel)), ms(fin),
                         fmt(smq.get(ds)), fmt(pub.get("ASOT")), fmt(pub.get("HiST_VQ")), fmt(pub.get("MASQ"))])
    lines += [table(["dataset", "normalization", "seeds", "selected", "selected per seed", "final", "SMQ (local)",
                     "ASOT (pub.)", "HiST-VQ (pub.)", "MASQ (pub.)"], rows)]

    result, notes = verdict(seg, smq)
    lines += ["## Declared reading", "", f"**{result}** (selected checkpoint, last rung, mean of three seeds)."
              + (" _Smoke budgets: this label carries no information._" if smoke else ""), ""]
    lines += [f"- {n}" for n in notes] + ["", "The rule was fixed before scoring; an intermediate outcome is reported as mixed, "
                                           "and the normalization factor and checkpoint policy are shown as they came out.", ""]

    other = []
    for ds in ORDER:
        for norm in ("global", "per_recording"):
            for role in ("selected", "final"):
                rows_ = [r for r in seg if r["dataset"] == ds and r["normalization"] == norm and r["role"] == role]
                if not rows_:
                    continue
                last = {}
                for r in rows_:
                    k = int(r["seed"])
                    if k not in last or int(r["rung_steps"]) > int(last[k]["rung_steps"]):
                        last[k] = r
                m = lambda key: fmt(statistics.mean(float(r[key]) for r in last.values()))
                other.append([ds, norm, role, m("MoF"), m("Edit"), m("F1@10"), m("F1@25"), m("F1@50"),
                              fmt(statistics.mean(float(r.get("seg_mean_pred_segments", "nan")) for r in last.values()), 1),
                              fmt(statistics.mean(float(r.get("seg_mean_gt_segments", "nan")) for r in last.values()), 1),
                              sum("capped" in r["ot_status"] for r in last.values())])
    lines += ["## All metrics (mean over seeds, last rung)", "",
              table(["dataset", "norm", "checkpoint", "MoF", "Edit", "F1@10", "F1@25", "F1@50", "pred segments / rec.",
                     "GT segments / rec.", "outer-capped fits"], other)]

    lines += ["## Per-cell diagnostics", "",
              table(["cell", "selected step", "val loss", "valid-token fraction", "eligible / update", "occupied codes",
                     "perplexity", "singleton codes", "MI code-action", "MI code-subject"],
                    [[c["cell_id"], c["selected_step"], fmt(c["best_val_loss"], 3), fmt(c["valid_token_fraction"], 3),
                      fmt(c["eligible_per_update"], 0), c.get("code_occupied_codes", "n/a"),
                      fmt(c.get("code_perplexity"), 0), c.get("code_singleton_codes", "n/a"),
                      fmt(c.get("code_mi_code_action"), 3), fmt(c.get("code_mi_code_subject"), 3)] for c in cells]),
              "The subject probe is a label-aware diagnostic of what the codes carry; it never entered selection.", ""]

    lines += ["## Missing cells and limits", "",
              table(["unit", "status", "reason"], [[r["unit"], r["status"], r["reason"]] for r in not_run])
              if not_run else "All planned units completed.\n",
              "Limits: three paired seeds (each drives the vocabulary and the contextualizer); the outer OT fit is "
              "budget-capped in essentially every fit; BABEL clips are short (mostly padding in a 128-token context); "
              "published numbers come from different protocols. This run does not tune anything on LARa or BABEL.", ""]
    path = run_dir / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
