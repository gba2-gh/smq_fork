"""Phase 8c report, written only from saved files.

Cells (span 2, `context`, last rung, paired by seed):
  T6 / T6s / T5gh-std          this phase's units  (phase8c/units/<ds>__<tok>__seed<s>__context)
  T5gh (weighted)              phase8a/units (seeds 111, 222) and this phase's seed 1538574472
  T2                           ts5_v1 phase5 (BABEL-2/3) and phase6 (LARa) units
  T0                           ts5_v1 phase6 S1a units (BABEL-2/3)

Declared comparisons (PHASE8C_SYMMETRY_TEST.md, Revision 2): Q1 T6 vs T0; Q2 T6s vs T2; Q3 T5gh-std vs weighted T5gh.
T6 vs T2 and T5gh-std vs T2 are descriptive. Every comparison is read only with three seeds; "within the bar" means
comparable performance, never equivalence.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

from script.week5.timescale_inspection import report4 as R4
from script.week5.timescale_inspection import report5 as R5r
from script.week5.timescale_inspection import report8a as R8

ROOT = R4.ROOT
BAR = R4.BAR
METRICS = R5r.ALL_METRICS
LABEL = R5r.METRIC_LABELS
SEEDS = (111, 222, 1538574472)
DATASETS = ("babel2", "babel3", "lara")
NAME = {"t0": "T0", "t2": "T2", "t6": "T6", "t6s": "T6s", "t5std": "T5gh-std", "t5w": "T5gh (weighted)"}
TOK = {"t6": "t6_heading_canonical", "t6s": "t6s_heading_canonical_stats", "t5std": "t5_both_std", "t5w": "t5_both"}
# (id, label, A, B, datasets, declared?)
COMPARISONS = (
    ("Q1", "T6 vs T0: does removing absolute heading (keeping gravity and chirality) help?", "t6", "t0", ("babel2", "babel3"), True),
    ("Q2", "T6s vs T2: at the same temporal summary, canonical coordinates vs pairwise distances", "t6s", "t2",
     ("babel2", "babel3", "lara"), True),
    ("Q3", "T5gh-std vs weighted T5gh: did 8a's block weighting cause its loss?", "t5std", "t5w", ("babel2", "babel3"), True),
    ("D1", "T6 vs T2 (descriptive: invariance, temporal summary, layout and codebook all differ)", "t6", "t2", DATASETS, False),
    ("D2", "T5gh-std vs T2 (descriptive)", "t5std", "t2", ("babel2", "babel3"), False),
)


def analysis_hash() -> str:
    h = hashlib.sha256()
    for name in ("report8c.py", "tokenizers8c.py"):
        h.update(name.encode())
        h.update((Path(__file__).parent / name).read_bytes())
    return h.hexdigest()[:16]


def read_unit(path: Path) -> dict | None:
    if not path.exists():
        return None
    i = json.loads(path.read_text())
    if i.get("status") != "done":
        return None
    return dict(R5r._norm_row(max(i["ladder"], key=lambda r: r["rung_steps"])), pred_segs=i["segment_stats"]["mean_pred_segments"],
                gt_segs=i["segment_stats"]["mean_gt_segments"])


def cells(out: Path, source_root: Path) -> dict:
    """{kind: {dataset: {seed: metrics}}} for t0, t2, t6, t6s, t5std, t5w."""
    out, src = Path(out), Path(source_root)
    res = {k: {ds: {} for ds in DATASETS} for k in NAME}
    for ds in DATASETS:
        for s in SEEDS:
            res["t2"][ds][s] = R8.t2_reference(src, ds, s)
            if ds != "lara":
                res["t0"][ds][s] = read_unit(src / "phase6" / "units" / f"{ds}__t0_raw__seed{s}__span2__context" / "info.json")
                a = read_unit(src / "phase8a" / "units" / f"{ds}__t5_both__seed{s}__context" / "info.json")
                b = read_unit(out / "units" / f"{ds}__t5_both__seed{s}__context" / "info.json")
                res["t5w"][ds][s] = a or b
            for k in ("t6", "t6s", "t5std"):
                res[k][ds][s] = read_unit(out / "units" / f"{ds}__{TOK[k]}__seed{s}__context" / "info.json")
    return {k: {ds: {s: v for s, v in d.items() if v is not None} for ds, d in per.items()} for k, per in res.items()}


def paired(a: dict, b: dict, met: str = "F1_50"):
    ss = [s for s in SEEDS if s in a and s in b]
    if not ss:
        return None
    d = [a[s][met] - b[s][met] for s in ss]
    return statistics.mean(d), d, ss


def fmt_d(p) -> str:
    return f"{p[0]:+.2f} ({' / '.join(f'{x:+.1f}' for x in p[1])})"


def reading(p) -> str:
    if p is None:
        return "no cells"
    if len(p[2]) < 3:
        return f"incomplete ({len(p[2])} seed{'s' if len(p[2]) != 1 else ''})"
    if p[0] >= BAR and all(x > 0 for x in p[1]):
        return "**A better (passes the bar)**"
    if p[0] <= -BAR and all(x < 0 for x in p[1]):
        return "**B better (reverse passes the bar)**"
    return "within the bar: comparable performance, not equivalence"


def class_rows(out: Path, source_root: Path, have: dict) -> list[list]:
    import statistics as st
    rows = []
    for ds in ("babel2", "babel3"):
        for k in ("t6", "t6s", "t5std", "t5w"):
            seeds = sorted(have[k][ds], key=SEEDS.index)
            ref, mine = [], []
            for s in seeds:
                phase, cid = R8.t2_cell_id(ds, s)
                a = R8._load_preds(Path(source_root) / "phase7" / "s0" / f"{phase}__{cid}__context" / "predictions.npz")
                own = Path(out) / "units" / f"{ds}__{TOK[k]}__seed{s}__context" / "predictions.npz"
                old = Path(source_root) / "phase8a" / "units" / f"{ds}__t5_both__seed{s}__context" / "predictions.npz"
                b = R8._load_preds(own if own.exists() else old)
                if a is None or b is None:
                    continue
                ref.append(R8.class_scores(ds, a))
                mine.append(R8.class_scores(ds, b))
            if not mine:
                continue
            for n in [c for c in mine[0] if c != "MoF"]:
                rr, rm = st.mean(x[n][0] for x in ref), st.mean(x[n][0] for x in mine)
                rows.append([ds, NAME[k], n + (" *" if n in R8.CLASS_FOCUS[ds] else ""), f"{mine[0][n][2]:.1f}", len(mine),
                             f"{rr:.1f}", f"{rm:.1f}", f"{rm - rr:+.1f}", f"{st.mean(x[n][1] for x in ref):.1f}",
                             f"{st.mean(x[n][1] for x in mine):.1f}"])
    return rows


def write_report(out: Path, source_root: Path) -> Path:
    out = Path(out)
    m = json.loads((out / "manifest.json").read_text())
    L = [f"# Time-scale inspection, phase 8c: which invariance? — {m['run_id']}", "",
         f"Status `{m.get('status')}`. Rules declared in PHASE8C_SYMMETRY_TEST.md (Revision 2) before the run. Primary metric F1@50 "
         f"at the last rung, mask span 2, paired by seed; the bar is a mean gain >= {BAR:g} with every seed in the same direction and is "
         "read only with three seeds. 'Within the bar' means comparable performance, not equivalence.", ""]
    g = json.loads((out / "gates.json").read_text()) if (out / "gates.json").exists() else {}
    if g:
        L += ["## Gates", "", R4.table(["gate", "status", "detail"], [
            [k, v.get("status"), str(v.get("checks", v.get("reason", "")))[:300]] for k, v in sorted(g.items())]), ""]
    if m.get("not_run"):
        L += ["## Blocked", ""] + [f"- `{k}`: {v}" for k, v in sorted(m["not_run"].items())] + [""]

    have = cells(out, source_root)
    rows = []
    for ds in DATASETS:
        for k in ("t0", "t2", "t6", "t6s", "t5std", "t5w"):
            v = have[k][ds]
            if not v:
                continue
            seeds = sorted(v, key=SEEDS.index)
            f = [v[s]["F1_50"] for s in seeds]
            rows.append([ds, NAME[k], " / ".join(f"{s}:{v[s]['F1_50']:.1f}" for s in seeds), len(f), f"{statistics.mean(f):.2f}",
                         f"{statistics.mean(v[s]['pred_segs'] for s in seeds):.1f}", f"{statistics.mean(v[s]['gt_segs'] for s in seeds):.1f}"])
    L += ["## Cells (F1@50, span 2, `context`)", "",
          R4.table(["dataset", "tokenizer", "per seed", "n", "mean", "pred segs/rec", "gt segs/rec"], rows)]

    L += ["## Declared comparisons and descriptive ones (A minus B, paired by seed)", ""]
    allm = []
    for cid, text, a, b, dss, declared in COMPARISONS:
        L += [f"### {cid}{'' if declared else ' (descriptive)'}: {text}", ""]
        for ds in dss:
            p = paired(have[a][ds], have[b][ds])
            L.append(f"- **{ds}**: {fmt_d(p) if p else 'no common cells'} over {len(p[2]) if p else 0} seeds -> {reading(p)}")
            for met in METRICS:
                pm = paired(have[a][ds], have[b][ds], met)
                if pm:
                    allm.append([cid, ds, f"{NAME[a]} − {NAME[b]}", LABEL[met], fmt_d(pm), len(pm[2]), reading(pm)])
        L.append("")
    L += ["### LARa non-inferiority (no loss >= 3 F1@50 or MoF against T2)", ""]
    for k in ("t6", "t6s"):
        p, mf = paired(have[k]["lara"], have["t2"]["lara"]), paired(have[k]["lara"], have["t2"]["lara"], "MoF")
        if p is None:
            L.append(f"- {NAME[k]}: no LARa cells")
            continue
        loss = p[0] <= -BAR or (mf is not None and mf[0] <= -BAR)
        L.append(f"- {NAME[k]}: F1@50 {fmt_d(p)}, MoF {fmt_d(mf)} over {len(p[2])} seeds -> {'**LOSS >= 3**' if loss else 'no loss >= 3 points'}")
    L += ["", "## Every metric", "", R4.table(["id", "dataset", "A − B", "metric", "Δ (per seed)", "n", "reading"], allm)]

    L += ["## Per-class frame recall and precision against T2 (uses labels; analysis only)", "",
          "Mean over the common seeds, global Hungarian mapping as the evaluator. * marks 8a's targeted classes. T0 predictions at span 2 "
          "are not saved, so T0 is not in this table. Small classes are noisy under a one-to-one mapping.", ""]
    try:
        L += [R4.table(["dataset", "tokenizer", "class", "frame share %", "n seeds", "T2 recall", "recall", "Δ recall", "T2 precision", "precision"],
                       class_rows(out, source_root, have))]
    except Exception as exc:  # the report must still be written
        L += [f"_per-class table failed: {type(exc).__name__}: {exc}_", ""]

    drows = []
    for p in sorted((out / "prep").glob("*/*/t5_blocks.json")):
        b = json.loads(p.read_text())
        parts = p.parent.name.split("__")
        share = " / ".join(f"{k} {v:.3f}" for k, v in b["pca_variance_share"].items())
        pre = " / ".join(f"{k} {v:.3f}" for k, v in b.get("pre_pca_variance_share", {}).items()) or "—"
        drows.append([parts[0], parts[1], parts[2].replace("seed", ""), pre, share, b["kmeans_occupied"],
                      f"{b['pca_explained_variance_ratio']:.3f}",
                      f"{b['windows_unrotated_share']:.4f}" if b.get("windows_unrotated_share") is not None else "—"])
    L += ["## Tokenizer diagnostics (label-free)", "",
          R4.table(["dataset", "tokenizer", "seed", "pre-PCA variance share by block", "share of PCA-64 variance by block", "occupied codes",
                    "PCA-64 explained", "windows left unrotated"], drows), ""]
    path = out / "REPORT.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
