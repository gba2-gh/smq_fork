"""Markdown report for the dimensionality-reduction ablation, written only from saved files.

Every variant is read against the same dataset's `base` (the anchor re-run in this ablation, same code path), paired by
seed. The decision bar is the timescale-inspection one: a variant differs from `base` when its mean paired F1@50
difference is at least BAR points and every seed moves in that direction; anything else is "no clear difference".

  python -m script.week5.dimred_ablation.report --run-id dr_v1
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from script.week5.dimred_ablation import prepare as PR
from script.week5.dimred_ablation import reducers as RD

ROOT = Path(__file__).resolve().parents[3]
SEEDS = (111, 222, 1538574472)
BAR = 3.0
DATASETS = ("babel2", "babel3", "hugadb", "lara")
METRICS = ("MoF", "Edit", "F1_10", "F1_25", "F1_50")
TS5 = ROOT / "results" / "week5" / "timescale_inspection" / "ts5_v1"
ANCHOR_UNITS = {  # the anchor's own saved units, per seed (HuGaDB's anchor is task2_v1, seed 111, read as a constant)
    "babel2": lambda s: TS5 / "phase5" / "units" / f"babel2__t2_relational__seed{s}__span2__context" / "info.json",
    "lara": lambda s: TS5 / "phase6" / "units" / f"lara__t2_relational__seed{s}__span2__context" / "info.json",
    "babel3": lambda s: TS5 / "phase5" / "units" / f"babel3__t1_whiten__seed{s}__pca" / "info.json",
}
SMQ = {"hugadb": 24.3, "lara": 16.4, "babel2": 27.5, "babel3": 19.4}


def last_row(info: dict) -> dict:
    return info["ladder"][-1]


def load_units(out: Path) -> dict:
    """{(dataset, variant): {seed: unit info}} for every finished unit."""
    res: dict = {}
    for path in sorted((out / "units").glob("*/info.json")):
        info = json.loads(path.read_text())
        if info.get("status") != "done":
            continue
        res.setdefault((info["dataset"], info["variant"]), {})[int(info["seed"])] = info
    return res


def load_diag(out: Path) -> dict:
    res: dict = {}
    for path in sorted((out / "diagnostics").glob("*.json")):
        d = json.loads(path.read_text())
        ds, variant, seed = d["cell"].split("__")
        res.setdefault((ds, variant), {})[int(seed.removeprefix("seed"))] = d
    return res


def mean_sd(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    return f"{statistics.mean(xs):.2f}" + (f" ± {statistics.stdev(xs):.2f}" if len(xs) > 1 else "")


def paired(v: dict[int, float], base: dict[int, float]):
    seeds = sorted(set(v) & set(base))
    if not seeds:
        return None
    d = [v[s] - base[s] for s in seeds]
    return statistics.mean(d), d


def verdict(p, n_expected: int) -> str:
    if p is None or len(p[1]) < n_expected:
        return "incomplete"
    if p[0] >= BAR and all(x > 0 for x in p[1]):
        return "**better**"
    if p[0] <= -BAR and all(x < 0 for x in p[1]):
        return "**worse**"
    return "no clear difference"


def delta_text(p) -> str:
    return "n/a" if p is None else f"{p[0]:+.2f} ({' / '.join(f'{x:+.1f}' for x in p[1])})"


def mean_of(rows: dict, key, sub=None) -> str:
    vals = [(r[sub] if sub else r).get(key) for r in rows.values()]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return f"{statistics.mean(vals):.3f}" if vals else "n/a"


def anchor_lines(ds: str, units: dict) -> list[str]:
    a = PR.ANCHORS[ds]
    base = {s: last_row(i)["F1_50"] for s, i in units.get((ds, "base"), {}).items()}
    lines = [f"Anchor: {a.source} features, {a.window}-frame windows, "
             f"{'whitened ' if a.whiten else ''}PCA-64, K-means-512, arm `{a.arm}`"
             + (f" (guard {a.guard}, span {a.span})" if a.arm == "context" else "")
             + f". Source: {a.reference}, F1@50 {a.reference_f1_50:.2f}; SMQ {SMQ[ds]:.1f}.", ""]
    if ds in ANCHOR_UNITS:
        orig = {}
        for s in SEEDS:
            p = ANCHOR_UNITS[ds](s)
            if p.exists():
                orig[s] = last_row(json.loads(p.read_text()))["F1_50"]
        pairs = [f"seed {s}: {orig[s]:.2f} -> {base[s]:.2f}" for s in SEEDS if s in orig and s in base]
        lines.append("Re-run of the anchor (original unit -> this run's `base`): "
                     + ("; ".join(pairs) if pairs else "not available yet") + ".")
    else:
        lines.append(f"Re-run of the anchor: seed 111 {a.reference_f1_50:.2f} (task2_v1) -> "
                     + (f"{base[111]:.2f}" if 111 in base else "n/a") + " (this run's `base`; seeds 222 / 1538574472 "
                     "are new).")
    lines.append("")
    return lines


def dataset_table(ds: str, units: dict, diag: dict, n_seeds: int) -> list[str]:
    base = {s: last_row(i)["F1_50"] for s, i in units.get((ds, "base"), {}).items()}
    head = ("| stage | variant | dims | retained var | codes used | NMI(code;action) | NMI(code;recording) |"
            + (" NMI(code;subject) |" if ds in ("hugadb", "lara") else "")
            + " F1@50 | Δ F1@50 vs base (per seed) | reading | MoF | Edit | F1@10 | F1@25 | segments pred/gt |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for v in RD.VARIANTS:
        rows = units.get((ds, v.key), {})
        dg = diag.get((ds, v.key), {})
        if not rows and not dg:
            continue
        f1 = {s: last_row(i)["F1_50"] for s, i in rows.items()}
        p = None if v.key == "base" else paired(f1, base)
        dims = mean_of(dg, "out_dim", "reducer")
        seg = [i["segment_stats"] for i in rows.values()]
        seg_text = (f"{statistics.mean(x['mean_pred_segments'] for x in seg):.1f} / "
                    f"{statistics.mean(x['mean_gt_segments'] for x in seg):.1f}") if seg else "n/a"
        cells = [v.stage, f"{v.key} ({v.label})", dims.split(".")[0] if dims != "n/a" else dims,
                 mean_of(dg, "retained_variance", "reducer"), mean_of(dg, "kmeans_occupied", "reducer").split(".")[0],
                 mean_of(dg, "nmi_code_action"), mean_of(dg, "nmi_code_recording")]
        if ds in ("hugadb", "lara"):
            cells.append(mean_of(dg, "nmi_code_subject"))
        cells += [mean_sd(list(f1.values())) + (f" (n={len(f1)})" if len(f1) != n_seeds else ""),
                  "reference" if v.key == "base" else delta_text(p),
                  "—" if v.key == "base" else verdict(p, n_seeds)]
        for m in ("MoF", "Edit", "F1_10", "F1_25"):
            cells.append(mean_sd([last_row(i)[m] for i in rows.values()]))
        cells.append(seg_text)
        lines.append("| " + " | ".join(cells) + " |")
    return lines + [""]


def summary_matrix(units: dict, datasets) -> list[str]:
    lines = ["| variant | " + " | ".join(datasets) + " |", "|---|" + "---|" * len(datasets)]
    for v in RD.VARIANTS:
        if v.key == "base":
            row = [mean_sd([last_row(i)["F1_50"] for i in units.get((ds, "base"), {}).values()]) for ds in datasets]
            lines.append(f"| base F1@50 | " + " | ".join(row) + " |")
            continue
        row = []
        for ds in datasets:
            base = {s: last_row(i)["F1_50"] for s, i in units.get((ds, "base"), {}).items()}
            f1 = {s: last_row(i)["F1_50"] for s, i in units.get((ds, v.key), {}).items()}
            p = paired(f1, base)
            row.append("n/a" if p is None else f"{p[0]:+.2f}" + (" ✓" if verdict(p, len(SEEDS)) == "**better**" else
                                                                 " ✗" if verdict(p, len(SEEDS)) == "**worse**" else ""))
        lines.append(f"| {v.key} | " + " | ".join(row) + " |")
    return lines + ["", "✓ / ✗: better / worse than `base` by the bar (mean ≥ 3.0 F1@50 and every seed the same sign).", ""]


READING = """## How to read this

* Only two things differ between variants: **which subspace** the window vector is reduced to, and **how its axes are
  scaled**. A rotation inside the kept subspace changes nothing downstream (K-means, centroid lookup and the
  mean-centred cosine OT cost are all rotation-invariant), so e.g. ICA would equal whitened PCA and is not run.
* For the `context` anchors (HuGaDB, LARa, BABEL-2) the reduction reaches the result only through the **K-means
  partition** (and the initial code grouping): the contextualizer sees code identities. For BABEL-3 (`pca` arm) the
  reduced vector **is** the OT feature, so the reduction acts on the cost directly.
* `sfa64` is whitened by construction: read it against `whiten_flip` where the anchor is unwhitened (HuGaDB, LARa,
  BABEL-2) and against `base` on BABEL-3. The same holds for `whiten_half`, which sits between the two.
* `identity` keeps every coordinate without scaling; on BABEL-3 it is also unwhitened, so its difference from `base`
  mixes "no reduction" with "no whitening" (`whiten_flip` isolates the second).
* `wrpca64` changes only the axes (fitted on recording-centred windows); `rc_pca64` also removes each recording's mean
  from the data. NMI(code; recording / subject) shows whether either actually reduced recording identity in the codes.
* NMI columns are evaluation-only descriptions of the tokens, computed after preparation; nothing is fitted on them.
"""


def gate_lines(out: Path) -> list[str]:
    path = out / "gates.json"
    if not path.exists():
        return ["Gates: not run.", ""]
    g = json.loads(path.read_text())
    lines = ["| gate | status | detail |", "|---|---|---|"]
    for name, res in sorted(g.items()):
        detail = res.get("reason", "")
        if "checks" in res:
            failed = [k for k, ok in res["checks"].items() if not ok]
            detail = ("all checks equal" if not failed else "differs: " + ", ".join(failed))
            if res.get("code_agreement") is not None:
                detail += (f"; codes differ in {res.get('windows_differing', 'n/a')} window(s) "
                           f"(agreement {res['code_agreement']:.6f}); max centroid diff {res['lookup_max_abs_diff']:.2e}")
        lines.append(f"| {name} | {res.get('status')} | {detail} |")
    return lines + [""]


def write_report(out: Path) -> Path:
    out = Path(out)
    m = json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").exists() else {}
    units, diag = load_units(out), load_diag(out)
    n_seeds = 1 if m.get("smoke") else len(SEEDS)
    datasets = [ds for ds in DATASETS if any(k[0] == ds for k in list(units) + list(diag))]
    lines = [f"# Dimensionality-reduction ablation — `{out.name}`", "",
             f"Status: {m.get('status', 'unknown')}" + (" (SMOKE: plumbing check, not a result)" if m.get("smoke") else "")
             + f"; last rung of the OT ladder; seeds {', '.join(map(str, SEEDS[:n_seeds]))}.", "",
             "## Gates", ""] + gate_lines(out)
    lines += ["## F1@50 difference from `base`, all datasets", ""] + summary_matrix(units, datasets)
    for ds in datasets:
        lines += [f"## {ds}", ""] + anchor_lines(ds, units) + dataset_table(ds, units, diag, n_seeds)
    lines.append(READING)
    if m.get("not_run"):
        lines += ["## Not run", ""] + [f"* {k}: {v}" for k, v in sorted(m["not_run"].items())] + [""]
    path = out / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args(argv)
    print(write_report(ROOT / "results" / "week5" / "dimred_ablation" / args.run_id))


if __name__ == "__main__":
    main()
