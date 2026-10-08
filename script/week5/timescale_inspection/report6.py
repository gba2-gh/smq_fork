"""Markdown report for phase 6 (confirming the span rule), written only from saved files.

Merges Phase 5's (`<run>/phase5/units`) and Phase 6's (`<run>/phase6/units`) unit files into one table, keyed
(dataset, tokenizer, arm, window, span); a key present in both is an error (a run should never duplicate a cell Phase
5 already has). T0 baselines (`context` from `rkx_v2`, `centre`/`pca` from `ts1_v1`) are added through
`report5.Results.f1`. Reading rules: docs/plans/week5/TIMESCALE_INSPECTION/PHASE6_SPAN_CONFIRMATION.md.
"""

from __future__ import annotations

import json
import math
import statistics
from fractions import Fraction
from pathlib import Path

from script.week5.raw_kmeans_cross_dataset import datasets as D
from script.week5.timescale_inspection import report4 as R4
from script.week5.timescale_inspection import report5 as R5r
from script.week5.timescale_inspection import tokenizers as TOK

ROOT = R4.ROOT
BAR = R4.BAR
BABEL3_DECLARED_SPAN2_GAIN = 3.90  # PHASE6_SPAN_CONFIRMATION.md H8b, from Phase 5 (3 seeds)
MEDIAN_SEGMENT_S = {"babel1": 1.20, "babel2": 1.70, "babel3": 1.77, "lara": 2.2, "hugadb": 3.47}  # script/week5/segment_stats.py (verified 2026-10-03)


def span_seconds(ds: str, window: int, span: int) -> float:
    return span * window / D.SPECS[ds].fps


def span_ratio(ds: str, window: int, span: int) -> float:
    return span_seconds(ds, window, span) / MEDIAN_SEGMENT_S[ds]


def sign_test_p(diffs: list[float]) -> float:
    """Exact two-sided sign-test p-value: P(at least `max(pos, neg)` of n non-zero signs agree), binomial(n, 0.5)."""
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return 1.0
    pos = sum(1 for d in nz if d > 0)
    k = max(pos, n - pos)

    def choose(a, b):
        return math.comb(a, b)

    tail = Fraction(sum(choose(n, i) for i in range(k, n + 1)), 2 ** n)
    return float(min(1.0, 2 * tail))


def load_phase_units(out: Path) -> dict:
    rows: dict = {}
    for path in sorted(Path(out).glob("units/*/info.json")):
        try:
            info = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if info.get("status") != "done":
            continue
        last = max(info["ladder"], key=lambda r: r["rung_steps"])
        key = (info["dataset"], info["tokenizer"], info["arm"], info["window"], info["span"])
        rows.setdefault(key, {})[info["seed"]] = dict(last, segment_stats=info["segment_stats"], ot_status=info["ot_status"],
                                                      outer_iterations=info["outer_iterations"])
    return rows


def merge_units(phase5_dir: Path, phase6_dir: Path) -> dict:
    a, b = load_phase_units(phase5_dir), load_phase_units(phase6_dir)
    merged: dict = {k: dict(v) for k, v in a.items()}
    for key, seeds in b.items():
        existing = merged.setdefault(key, {})
        clash = set(existing) & set(seeds)
        if clash:
            raise ValueError(f"phase6 duplicates phase5 at {key}, seeds {sorted(clash)}: a run should never "
                             f"re-decode a cell phase5 already has")
        existing.update(seeds)
    return merged


class Results6:
    def __init__(self, phase5_dir: Path, phase6_dir: Path, context_run: str, direct_run: str):
        self.units = merge_units(phase5_dir, phase6_dir)
        self.ctx = R4.load_context_baseline(context_run)
        self.direct = R4.load_direct_baseline(direct_run)

    def f1(self, ds: str, tok: str, arm: str, window: int | None = None, span: int | None = None) -> dict[int, float]:
        window = window or D.SPECS[ds].window
        span = span or TOK.SPECS[tok].span
        return {s: m["F1_50"] for s, m in self.metrics(ds, tok, arm, window, span).items()}

    def row(self, ds: str, tok: str, arm: str, window: int | None = None, span: int | None = None) -> dict:
        window = window or D.SPECS[ds].window
        span = span or TOK.SPECS[tok].span
        return self.units.get((ds, tok, arm, window, span), {})

    def metrics(self, ds: str, tok: str, arm: str, window: int | None = None, span: int | None = None) -> dict:
        """Every one of MoF/Edit/F1@10/F1@25/F1@50 per seed, from the merged phase5+phase6 units; falls back to the
        T0 baseline (rkx_v2/ts1_v1) at window/span defaults, exactly as `f1` does."""
        window = window or D.SPECS[ds].window
        span = span or TOK.SPECS[tok].span
        out = {s: R5r._norm_row(r) for s, r in self.row(ds, tok, arm, window, span).items()}
        if tok == "t0_raw" and window == D.SPECS[ds].window and span == 8:
            # T0 at the original seeds is read from rkx_v2 (context) / ts1_v1 (centre, pca); T0 at the new seeds
            # (333, 444) was run in phase 6 through the generalized path, which gate G1 showed reproduces rkx_v2's
            # tokens. Both are merged; a seed present in both sources is an error, never silently resolved.
            src = self.ctx.get((ds, "global")) if arm == "context" else self.direct.get((ds, "global", arm))
            base = {s: R5r._norm_row(row) for s, row in (src or {}).items()}
            clash = set(base) & set(out)
            if clash:
                raise ValueError(f"T0 {ds}/{arm}: seeds {sorted(clash)} exist both as a baseline and as a phase-6 unit")
            out = {**base, **out}
        return out

    def segs(self, ds: str, tok: str, arm: str, span: int | None = None) -> tuple[float, float] | None:
        rows = self.row(ds, tok, arm, span=span)
        if not rows:
            return None
        pred = statistics.mean(r["segment_stats"]["mean_pred_segments"] for r in rows.values())
        gt = statistics.mean(r["segment_stats"]["mean_gt_segments"] for r in rows.values())
        return pred, gt

    def early_stopped(self, ds: str, tok: str, arm: str, span: int | None = None) -> list[int]:
        return [s for s, r in self.row(ds, tok, arm, span=span).items()
                if r.get("ot_status") == "converged" and r.get("outer_iterations", 999) < 50]


def paired(v: dict[int, float], base: dict[int, float], min_seeds: int = 3):
    seeds = sorted(set(v) & set(base))
    if len(seeds) < min_seeds:
        return None
    d = [v[s] - base[s] for s in seeds]
    return statistics.mean(d), d, seeds


def passes_bar(p) -> bool:
    return p is not None and p[0] >= BAR and all(x > 0 for x in p[1])


def delta_text(p) -> str:
    return "n/a" if p is None else f"{p[0]:+.2f} ({' / '.join(f'{x:+.1f}' for x in p[1])})"


def headline_row(res: Results6, ds: str, tok: str, arm: str, span: int) -> list:
    v = res.f1(ds, tok, arm, span=span)
    if not v:
        return [ds, TOK.SPECS[tok].label if tok != "t0_raw" else "T0", arm, span, "n/a", "n/a", "n/a (<5 seeds)"]
    seeds = sorted(v)
    mean_sd = f"{statistics.mean(v.values()):.2f} ± {statistics.stdev(v.values()) if len(v) > 1 else 0:.2f}"
    return [ds, TOK.SPECS[tok].label if tok != "t0_raw" else "T0", arm, span,
            " / ".join(f"{v[s]:.1f}" for s in seeds), mean_sd, str(len(v))]


def headline_section(res: Results6) -> tuple[list[str], dict]:
    L = ["## Headline: BABEL-2/3, 5 seeds, tokenizer x span (`context`)", "",
         "Original seeds 111/222/1538574472 plus the new seeds 333/444 (S1a extends T0 span 2 to all 3 original "
         "seeds; S1b adds both tokenizers x both spans at the new seeds).", ""]
    rows = []
    for ds in ("babel2", "babel3"):
        for tok in ("t0_raw", "t2_relational"):
            for span in (8, 2):
                rows.append(headline_row(res, ds, tok, "context", span))
    L += [R4.table(["dataset", "tokenizer", "arm", "span", "F1@50 per seed (sorted: 111/222/333/444/1538574472)",
                    "mean ± sd", "n seeds"], rows)]
    L += ["### `centre` (no-contextualizer) at span 8, 5 seeds where available", ""]
    crows = []
    for ds in ("babel2", "babel3"):
        for tok in ("t0_raw", "t2_relational"):
            crows.append(headline_row(res, ds, tok, "centre", 8))
    L += [R4.table(["dataset", "tokenizer", "arm", "span", "F1@50 per seed (sorted: 111/222/333/444/1538574472)", "mean ± sd", "n seeds"], crows)]

    verdicts = {}
    L += ["### H2'-confirm", ""]
    for ds in ("babel2", "babel3"):
        v8, v2 = res.f1(ds, "t2_relational", "context", span=8), res.f1(ds, "t2_relational", "context", span=2)
        p = paired(v2, v8, min_seeds=5)
        if p is None:
            L.append(f"- {ds}: not enough seeds finished on both sides yet (have {len(set(v8) & set(v2))}/5).")
            continue
        mean_d, diffs, seeds = p
        pos = sum(1 for d in diffs if d > 0)
        sp = sign_test_p(diffs)
        ok = passes_bar(p)
        verdicts[ds] = ok
        L.append(f"- {ds}: T2 span 2 - T2 span 8 = {delta_text(p)} over {len(seeds)} seeds -> "
                 f"**{'beats span 8 by the bar' if ok else 'does not beat span 8 by the bar'}** "
                 f"({pos}/{len(diffs)} seeds positive, sign-test p={sp:.4f})")
    h2_confirm = ("confirmed" if verdicts.get("babel2") and any(verdicts.get(d) for d in ("babel3",)) else
                 "confirmed on BABEL-2 only" if verdicts.get("babel2") else
                 "not confirmed" if verdicts else "undetermined")
    L += ["", f"**H2'-confirm: {h2_confirm}.**", ""]
    return L, verdicts


def h9_section(res: Results6) -> list[str]:
    L = ["## H9: does the span effect need the tokenizer?", "",
         "Using the 2x2 (tokenizer x span) at the seeds both sides have.", ""]
    for ds in ("babel2", "babel3"):
        cells = {(tok, sp): res.f1(ds, tok, "context", span=sp) for tok in ("t0_raw", "t2_relational") for sp in (8, 2)}
        t0_span = paired(cells[("t0_raw", 2)], cells[("t0_raw", 8)])
        t2_span = paired(cells[("t2_relational", 2)], cells[("t2_relational", 8)])
        tok_at_8 = paired(cells[("t2_relational", 8)], cells[("t0_raw", 8)])
        tok_at_2 = paired(cells[("t2_relational", 2)], cells[("t0_raw", 2)])
        L += [f"### {ds}", "",
              f"- span effect on T0 (span2 - span8): {delta_text(t0_span)}",
              f"- span effect on T2 (span2 - span8): {delta_text(t2_span)}",
              f"- tokenizer effect at span 8 (T2 - T0): {delta_text(tok_at_8)}",
              f"- tokenizer effect at span 2 (T2 - T0): {delta_text(tok_at_2)}"]
        if t0_span is not None and t2_span is not None:
            seeds = sorted(set(cells[("t0_raw", 2)]) & set(cells[("t0_raw", 8)]) &
                           set(cells[("t2_relational", 2)]) & set(cells[("t2_relational", 8)]))
            inter = [(cells[("t2_relational", 2)][s] - cells[("t2_relational", 8)][s]) -
                    (cells[("t0_raw", 2)][s] - cells[("t0_raw", 8)][s]) for s in seeds]
            L.append(f"- interaction ((T2 span2-span8) - (T0 span2-span8)), paired by seed: "
                     f"{statistics.mean(inter):+.2f} ({' / '.join(f'{x:+.1f}' for x in inter)})" if inter else
                     "- interaction: not enough seeds")
            if passes_bar(t0_span):
                label = "span effect is tokenizer-independent (T0 span 2 beats T0 span 8 by the bar)"
            elif t0_span[0] >= -BAR and t0_span[0] <= BAR:
                needs = passes_bar(t2_span)
                label = ("needs relational tokens (T0 span 2 is within the bar of T0 span 8; T2's span effect "
                        "passes)" if needs else "neither tokenizer shows a span effect by the bar")
            elif t0_span[0] <= -BAR and all(x < 0 for x in t0_span[1]):
                label = "reverses on T0 (T0 span 2 loses to T0 span 8 by the bar)"
            else:
                label = "mixed (does not fit a declared label)"
            L.append(f"- **{label}**")
        L.append("")
    return L


def span_rule_table(res: Results6) -> list[str]:
    L = ["## Span-rule table: F1@50 and predicted vs ground-truth segments, by span ratio", "", "`context` arm only.", ""]
    rows = []
    cells = [("babel2", "t0_raw"), ("babel2", "t2_relational"), ("babel3", "t0_raw"), ("babel3", "t2_relational"),
            ("lara", "t2_relational"), ("hugadb", "t0_raw")]
    entries = []
    for ds, tok in cells:
        spec = D.SPECS[ds]
        for span in sorted({2, 4, 8, 16, TOK.SPECS[tok].span}):
            v = res.f1(ds, tok, "context", span=span)
            if not v:
                continue
            segs = res.segs(ds, tok, "context", span=span)
            entries.append((span_ratio(ds, spec.window, span), ds, tok, span, v, segs))
    for ratio, ds, tok, span, v, segs in sorted(entries, key=lambda e: (e[1], e[0])):
        mean_sd = f"{statistics.mean(v.values()):.2f} ± {statistics.stdev(v.values()) if len(v) > 1 else 0:.2f}"
        rows.append([ds, TOK.SPECS[tok].label if tok != "t0_raw" else "T0", span, f"{span_seconds(ds, D.SPECS[ds].window, span):.2f}",
                    f"{ratio:.2f}", mean_sd, str(len(v)), R4.fmt(segs[0], 1) if segs else "n/a",
                    R4.fmt(segs[1], 1) if segs else "n/a"])
    L += [R4.table(["dataset", "tokenizer", "span", "mask (s)", "span ratio", "F1@50 mean ± sd", "n seeds",
                    "pred segs/rec", "gt segs/rec"], rows), ""]
    return L


def h8_section(res: Results6) -> list[str]:
    L = ["## H8a: HuGaDB negative control, and H8b: LARa out-of-sample", ""]
    h8a_v8 = res.f1("hugadb", "t0_raw", "context", span=8)
    for span, label in ((2, "span 2"), (4, "span 4")):
        v = res.f1("hugadb", "t0_raw", "context", span=span)
        p = paired(v, h8a_v8)
        if p is None:
            L.append(f"- HuGaDB {label}: not enough seeds yet.")
            continue
        if passes_bar(p):
            verdict = "**gains by the bar -> the ratio-1 threshold looks wrong** (a short span would help everywhere)"
        elif p[0] <= -BAR and all(x < 0 for x in p[1]):
            verdict = "**loses by the bar -> consistent with the smoothing mechanism** (not a pass/fail; HuGaDB needs the smoothing a longer span gives, setting a lower limit on the useful span)"
        else:
            verdict = "within the bar -> no effect either way"
        L.append(f"- HuGaDB {label} vs span 8: {delta_text(p)} -> {verdict}")
    L.append("")
    lara_v8 = res.f1("lara", "t2_relational", "context", span=8)
    for span, label in ((2, "span 2"), (4, "span 4")):
        v = res.f1("lara", "t2_relational", "context", span=span)
        p = paired(v, lara_v8)
        if p is None:
            L.append(f"- LARa {label}: not enough seeds yet.")
            continue
        # The declared reference is BABEL-3's span-2 gain as it stood when the rule was written (Phase 5, 3 seeds:
        # +3.90; PHASE6_SPAN_CONFIRMATION.md line 106). An earlier version of this code substituted the updated
        # 5-seed gain, which moved the threshold after the fact and flipped the verdict; the declared value is used.
        p3 = paired(res.f1("babel3", "t2_relational", "context", span=2), res.f1("babel3", "t2_relational", "context", span=8))
        if p[0] <= -BAR and all(x < 0 for x in p[1]):
            verdict = "**refuted (short spans hurt on LARa)**"
        elif p[0] >= BAR and p[0] > BABEL3_DECLARED_SPAN2_GAIN:
            verdict = (f"**threshold looks wrong** (gains >= {BAR:g} and more than BABEL-3's declared span-2 gain of "
                       f"{BABEL3_DECLARED_SPAN2_GAIN:+.2f})")
        else:
            verdict = "consistent with the prediction (a modest gain at most, no loss)"
        if p3:
            verdict += f"; for information, BABEL-3's gain with 5 seeds is {p3[0]:+.2f} (not the declared reference)"
        L.append(f"- LARa {label} vs span 8: {delta_text(p)} -> {verdict}")
    L.append("")
    return L


def early_stop_notes(res: Results6) -> list[str]:
    hits = []
    for key, rows in res.units.items():
        ds, tok, arm, window, span = key
        for s, r in rows.items():
            if r.get("ot_status") == "converged" and r.get("outer_iterations", 999) < 50:
                hits.append(f"{ds}/{tok}/{arm} span{span} seed{s}: converged at {r['outer_iterations']} iterations")
    if not hits:
        return []
    return ["## OT fits that stopped before the 50-iteration cap (may explain seed-level outliers)", ""] + \
           [f"- {h}" for h in sorted(hits)] + [""]


def smq_reference_section(res: Results6) -> list[str]:
    L = ["## Reference: SMQ's local re-score (not a declared criterion)", "", R4.table(
        ["dataset", "SMQ F1@50 (released checkpoint, global Hungarian)"],
        [[ds, R4.fmt(v, 1)] for ds, v in sorted(R5r.SMQ_REFERENCE.items())]), ""]
    headline = res.f1("babel2", "t2_relational", "context", span=2)
    if headline:
        L.append(f"- Pre-declared headline (BABEL-2, T2, span 2, `context`): {statistics.mean(headline.values()):.2f} "
                 f"mean over {len(headline)} seeds, vs SMQ {R5r.SMQ_REFERENCE['babel2']}.")
    headline3 = res.f1("babel3", "t2_relational", "context", span=2)
    if headline3:
        L.append(f"- Same cell on BABEL-3: {statistics.mean(headline3.values()):.2f} mean over {len(headline3)} "
                 f"seeds, vs SMQ {R5r.SMQ_REFERENCE['babel3']}.")
    L.append("")
    return L



def all_metrics_section(res: Results6) -> list[str]:
    """H2'-confirm and the HuGaDB/LARa span tests, re-run on MoF, Edit, F1@10 and F1@25 as well as the declared
    F1@50, read from the identical saved rows."""
    cells = [
        ("babel2 T2 span2 vs span8", "babel2", "t2_relational", "context", 2, "babel2", "t2_relational", "context", 8),
        ("babel3 T2 span2 vs span8", "babel3", "t2_relational", "context", 2, "babel3", "t2_relational", "context", 8),
        ("hugadb T0 span2 vs span8", "hugadb", "t0_raw", "context", 2, "hugadb", "t0_raw", "context", 8),
        ("hugadb T0 span4 vs span8", "hugadb", "t0_raw", "context", 4, "hugadb", "t0_raw", "context", 8),
        ("lara T2 span2 vs span8", "lara", "t2_relational", "context", 2, "lara", "t2_relational", "context", 8),
        ("lara T2 span4 vs span8", "lara", "t2_relational", "context", 4, "lara", "t2_relational", "context", 8),
    ]
    rows = []
    for label, ds, tok, arm, sp_n, ds_b, tok_b, arm_b, sp_b in cells:
        new_m, base_m = res.metrics(ds, tok, arm, span=sp_n), res.metrics(ds_b, tok_b, arm_b, span=sp_b)
        if not new_m or not base_m:
            continue
        for m in R5r.ALL_METRICS:
            v = {s: row[m] for s, row in new_m.items()}
            base = {s: row[m] for s, row in base_m.items()}
            p = paired(v, base)
            rows.append([label, R5r.METRIC_LABELS[m], f"{statistics.mean(v.values()):.2f}" if v else "n/a",
                        f"{statistics.mean(base.values()):.2f}" if base else "n/a", delta_text(p),
                        "beats by bar" if passes_bar(p) else ("n/a" if p is None else "no")])
    if not rows:
        return []
    return ["## Verification on every metric, not F1@50 alone", "",
            "F1@50 is the pre-declared metric for H2'-confirm, H8a and H8b. This re-runs the same paired span "
            "comparisons on MoF, Edit, F1@10 and F1@25, read from the identical saved rows.", "",
            R4.table(["comparison", "metric", "new mean", "base mean", "Δ (paired)", "bar"], rows), ""]


def write_report(phase6_dir: Path, phase5_dir: Path | None = None, context_run: str = "rkx_v2",
                 direct_run: str = "ts1_v1") -> Path:
    phase6_dir = Path(phase6_dir)
    phase5_dir = Path(phase5_dir) if phase5_dir is not None else phase6_dir.parent / "phase5"
    manifest = json.loads((phase6_dir / "manifest.json").read_text())
    res = Results6(phase5_dir, phase6_dir, context_run, direct_run)
    L = [f"# Time-scale inspection, phase 6: confirming the span rule — {manifest['run_id']}", "",
         f"Status `{manifest.get('status')}`. Reuses `{phase5_dir}` read-only for any preparation it already built. "
         "The primary number is F1@50 (selected checkpoint, last rung), paired by seed; the bar is >= 3 points on the "
         "mean with every seed in the same direction. With 5 seeds the report also gives the positive-seed count and "
         "an exact two-sided sign-test p-value; neither is part of a verdict.", ""]
    gates_path = phase6_dir / "gates.json"
    if gates_path.exists():
        g = json.loads(gates_path.read_text())
        def detail(k, v):
            if v.get("diffs"):
                return ", ".join(f"{m} {d}" for m, d in v["diffs"].items())
            if "checks" in v:
                return f"checks {v['checks']}"
            if k == "g3" and "median_extent_channels_3_5" in v:
                return (f"median across-joint extent: channels 3-5 {v['median_extent_channels_3_5']:.3f} vs 0-2 "
                       f"{v['median_extent_channels_0_2']:.3f}" + (" (inherited from phase5)" if "inherited_from" in v else ""))
            return v.get("reason", "")

        grows = [[k, v.get("status"), detail(k, v)] for k, v in sorted(g.items())]
        L += ["## Gates", "", R4.table(["gate", "status", "detail"], grows), ""]
    if manifest.get("not_run"):
        L += ["## Blocked by a gate", ""] + [f"- `{k}`: {v}" for k, v in sorted(manifest["not_run"].items())] + [""]
    head_lines, _ = headline_section(res)
    L += head_lines
    L += h9_section(res)
    L += span_rule_table(res)
    L += h8_section(res)
    L += all_metrics_section(res)
    L += early_stop_notes(res)
    L += smq_reference_section(res)
    path = phase6_dir / "REPORT.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
