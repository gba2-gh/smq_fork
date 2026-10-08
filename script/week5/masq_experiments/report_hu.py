"""REPORT_HU.md for the HuGaDB-focused follow-up: the variants V1, V2a, V2b against the references from masq_v1.

The references (S0, M-pub, M-JL on HuGaDB, seeds 111/222/1538574472) are read from masq_v1: same code, settings and seeds,
so the pairing is by seed. Comparisons use report.reading (bar: mean >= 3 with every seed in the same direction, three seeds).

Declared decision rule (docs/plans/week5/masq_experiments/TIER0_RESULTS_AND_HUGADB_FOCUS.md): the MASQ track continues only if
a variant passes the bar over M-JL (joint dropout alone) on BOTH MoF and F1@50; otherwise it stops on HuGaDB.
"""
from __future__ import annotations

import statistics as st
from pathlib import Path

from script.week5.masq_experiments import config as C
from script.week5.masq_experiments import report as R

REF_RUN = "masq_v1"
REF_ARMS = ("S0", "M-pub", "M-JL")
DATASET = "hugadb"
DECISION_RULE = ("The MASQ track continues only if a variant passes the bar over M-JL on both MoF and F1@50 (mean difference >= 3 "
                 "with all three seeds in the same direction). Otherwise it stops on HuGaDB: no other dataset is run.")


def decide(kinds: dict[str, dict[str, str]]) -> tuple[str, list[str]]:
    """kinds: variant -> {'MoF': kind, 'F1@50': kind} of (variant - M-JL). Returns (decision, variants that add value)."""
    winners = [v for v, k in kinds.items() if k.get("MoF") == "pass" and k.get("F1@50") == "pass"]
    return ("continue" if winners else "stop"), winners


def write_report_hu(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    run = R.load_run(run_dir)
    man = run["manifest"]
    smoke = bool(man.get("smoke"))
    cells = {k: v for k, v in run["cells"].items() if k[1] == DATASET and k[0] in C.VARIANT_ARMS}
    ref = R.load_run(C.run_dir(REF_RUN))
    for k, v in ref["cells"].items():
        if k[0] in REF_ARMS and k[1] == DATASET:
            cells[k] = v
    arms = [a for a in REF_ARMS + C.VARIANT_ARMS if any(k[0] == a for k in cells)]
    L = [f"# HuGaDB-focused follow-up (velocity-term variants): {run['run_id']}", ""]
    if smoke:
        L += ["**SMOKE RUN: 2 epochs on 3 batches per epoch. Plumbing check, not a result.**", ""]
    L += [f"Status `{man.get('status')}`. Variant code `{man.get('variant_hash')}`, training code `{man.get('train_hash')}`. References S0, M-pub and "
          f"M-JL are read from `{REF_RUN}` (same code, settings and seeds). Global Hungarian mapping, final epoch, MoF and F1@50 co-primary; "
          f"bar: mean difference >= {C.BAR:g} with every seed in the same direction.", "",
          f"**Decision rule (declared before the run):** {DECISION_RULE}", ""]

    rows = []
    for arm in arms:
        v = {m: R._vals(cells, arm, DATASET, "global", m) for m in C.METRICS}
        seeds = [s for s in C.SEEDS if s in v["MoF"]]
        src = "masq_v1" if arm in REF_ARMS else run["run_id"]
        rows.append([arm, src, len(seeds)] + [R._mean_sd(v[m].values()) for m in C.METRICS]
                    + [" / ".join(f"{v['MoF'][s]:.1f}" for s in seeds), " / ".join(f"{v['F1@50'][s]:.1f}" for s in seeds)])
    L += ["## Cells (HuGaDB, global Hungarian; seed mean ± population SD)", "",
          R.table(["arm", "from", "n seeds"] + list(C.METRICS) + ["MoF per seed", "F1@50 per seed"], rows),
          f"Published MASQ (single run): " + ", ".join(f"{m} {C.MASQ_PUBLISHED[DATASET][m]}" for m in C.METRICS) + ".", ""]

    calib = []
    for arm in C.VARIANT_ARMS:
        for seed in C.SEEDS:
            t = R._load(C.model_dir(run["run_id"], C.Unit(arm, DATASET, seed)) / "training_info.json")
            if t and t.get("calibration"):
                c = t["calibration"]
                calib.append([arm, seed, c["mode"], c["rho"], f"{c['rec_init']:.4g}", f"{c['term_init']:.4g}", f"{c['weight']:.4g}"])
    if calib:
        L += ["## Calibration (label-free, at initialization: weight x term = rho x reconstruction)", "",
              R.table(["arm", "seed", "term", "rho", "reconstruction(init)", "term(init)", "weight"], calib)]

    L += ["## Comparisons (A minus B, paired by seed)", ""]
    kinds, every = {}, []
    for a in C.VARIANT_ARMS:
        if not any(k[0] == a for k in cells):
            continue
        for b, why in (("M-JL", "does the extra term add anything to joint dropout alone? (decision comparison)"),
                       ("S0", "total effect over our SMQ retrain"), ("M-pub", "against MASQ's published-weight velocity loss (scale hypothesis)")):
            parts = []
            for m in ("MoF", "F1@50"):
                d = R._paired(cells, a, b, DATASET, m)
                rd = R.reading(d)
                parts.append(f"{m} {R._fmt(d)} -> {rd}")
                if b == "M-JL":
                    kinds.setdefault(a, {})[m] = R.reading_kind(rd)
            L.append(f"- **{a} − {b}** ({why}): " + "; ".join(parts))
            for m in C.METRICS:
                d = R._paired(cells, a, b, DATASET, m)
                if d:
                    every.append([f"{a} − {b}", m, R._fmt(d), len(d), R.reading(d)])
    L += ["", "### Every metric", "", R.table(["A − B", "metric", "Δ (per seed)", "n", "reading"], every)]

    L += ["## Training (final epoch; 'extra' = the variant's added term, weighted)", ""]
    rows = []
    for arm in arms:
        for seed in C.SEEDS:
            c = cells.get((arm, DATASET, seed))
            if c and c["train"]:
                last = c["train"]["per_epoch"][-1]
                rows.append([arm, seed, f"{last['rec']:.1f}", f"{last['commit']:.1f}", f"{last['vel']:.1f}", c["train"]["dead_code_replacements_total"], f"{c['train']['seconds']:.0f}"])
    L += [R.table(["arm", "seed", "rec", "commit", "extra / vel", "dead-code replacements", "seconds"], rows)]

    L += ["## Decision (computed from the declared rule)", ""]
    n_complete = all(all(len(R._paired(cells, a, "M-JL", DATASET, m)) == 3 for m in ("MoF", "F1@50")) for a in kinds) if kinds else False
    if smoke:
        L += ["Not computed for a smoke run.", ""]
    elif not kinds or not n_complete:
        L += ["Not computed: needs every variant with three seeds against M-JL.", ""]
    else:
        decision, winners = decide(kinds)
        if decision == "continue":
            L += [f"**Continue.** Passing the bar over M-JL on both MoF and F1@50: {', '.join(winners)}.", ""]
        else:
            L += ["**Stop on HuGaDB.** No variant passes the bar over joint dropout alone on both MoF and F1@50, so the velocity "
                  "term adds nothing measurable in either reading; the MASQ track is not extended to other datasets.", ""]
    L += ["## Caveats", "", R.CAVEATS, "",
          "The variants are our readings of an ambiguous specification (equation 5 versus the paper's description of a smoothness "
          "prior); neither is known to be the authors' implementation. rho values are declared choices, all of which are reported.", ""]
    path = run_dir / "REPORT_HU.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
