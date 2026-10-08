"""Markdown report for phase 5 (tokenizer, then span and window), written only from saved files.

Baselines are read, not rebuilt: T0 `context` from `rkx_v2`'s saved results, T0 `centre`/`pca` from Phase 1's
`inputs` stage (`ts1_v1`), per-token probes of T0 from Phase 0 (`ts0_v1`). Reading rules (H6, H7, H2') are those declared
in PHASE5_TOKENIZER.md, applied mechanically; F1@50 alone decides H6.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from script.week5.timescale_inspection import report4 as R4
from script.week5.timescale_inspection import tokenizers as TOK

ROOT = R4.ROOT
SEEDS = R4.SEEDS
BAR = R4.BAR
ARMS = ("context", "centre", "pca")
SMQ_REFERENCE = {"hugadb": 24.3, "lara": 16.4, "babel2": 27.5, "babel3": 19.4}
HUGADB_T0_CONTEXT_SEED111 = 50.00
HUGADB_FLAG = 6.0
DATASETS = ("babel2", "babel3", "lara", "hugadb")
NEW_TOKENIZERS = tuple(k for k in TOK.SPECS if k != "t0_raw")
CONTEXT_COMPARABLE = ("t1_whiten", "t2_relational", "t3_lara_pos256", "t3b_lara_pos64")  # T4 `context` has other masking
PROBES = ("linear", "nn", "nn_white")
MEDIAN_SEGMENT_S = {"babel1": 1.20, "babel2": 1.70, "babel3": 1.77, "lara": 2.2, "hugadb": 3.47}  # script/week5/segment_stats.py (verified 2026-10-03)
ALL_METRICS = ("MoF", "Edit", "F1_10", "F1_25", "F1_50")
METRIC_LABELS = {"MoF": "MoF", "Edit": "Edit", "F1_10": "F1@10", "F1_25": "F1@25", "F1_50": "F1@50"}


def _norm_row(row: dict) -> dict:
    """A unit's ladder row (keys MoF/Edit/F1_10/F1_25/F1_50) or a segmentation.csv/Phase-1 row (keys
    MoF/Edit/F1@10/F1@25/F1@50, string-valued) -> {MoF, Edit, F1_10, F1_25, F1_50} as floats."""
    return {m: float(row[m] if m in row else row[METRIC_LABELS[m]]) for m in ALL_METRICS}



def span_seconds(ds: str, window: int, span: int) -> float:
    from script.week5.raw_kmeans_cross_dataset import datasets as D
    return span * window / D.SPECS[ds].fps


def span_ratio(ds: str, window: int, span: int) -> float:
    return span_seconds(ds, window, span) / MEDIAN_SEGMENT_S[ds]


def default_window(ds: str) -> int:
    from script.week5.raw_kmeans_cross_dataset import datasets as D
    return D.SPECS[ds].window


def load_units(out: Path) -> dict:
    """(dataset, tokenizer, arm, window, span) -> {seed: last ladder row + segment_stats}."""
    rows: dict = {}
    for path in sorted((Path(out) / "units").glob("*/info.json")):
        try:
            info = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if info.get("status") != "done":
            continue
        last = max(info["ladder"], key=lambda r: r["rung_steps"])
        key = (info["dataset"], info["tokenizer"], info["arm"], info["window"], info["span"])
        rows.setdefault(key, {})[info["seed"]] = dict(last, segment_stats=info["segment_stats"],
                                                      ot_status=info.get("ot_status"),
                                                      outer_iterations=info.get("outer_iterations"))
    return rows


class Results:
    def __init__(self, out: Path, context_run: str, direct_run: str):
        self.units = load_units(out)
        self.ctx = R4.load_context_baseline(context_run)
        self.direct = R4.load_direct_baseline(direct_run)

    def f1(self, ds: str, tok: str, arm: str, window: int | None = None, span: int | None = None) -> dict[int, float]:
        window = window or default_window(ds)
        span = span or TOK.SPECS[tok].span
        if tok == "t0_raw" and window == default_window(ds) and span == 8:
            if arm == "context":
                return R4.f1s(self.ctx.get((ds, "global")))
            return R4.f1s(self.direct.get((ds, "global", arm)))
        return {s: float(r["F1_50"]) for s, r in self.units.get((ds, tok, arm, window, span), {}).items()}

    def metrics(self, ds: str, tok: str, arm: str, window: int | None = None,
               span: int | None = None) -> dict[int, dict]:
        """Every one of MoF/Edit/F1@10/F1@25/F1@50, per seed -- the same three sources `f1` reads, so the
        verification table below is read from the identical saved rows, not recomputed."""
        window = window or default_window(ds)
        span = span or TOK.SPECS[tok].span
        if tok == "t0_raw" and window == default_window(ds) and span == 8:
            src = self.ctx.get((ds, "global")) if arm == "context" else self.direct.get((ds, "global", arm))
            return {s: _norm_row(row) for s, row in (src or {}).items()}
        return {s: _norm_row(r) for s, r in self.units.get((ds, tok, arm, window, span), {}).items()}

    def segs(self, ds: str, tok: str, arm: str) -> tuple[float, float] | None:
        rows = self.units.get((ds, tok, arm, default_window(ds), TOK.SPECS[tok].span), {})
        if not rows:
            return None
        return (statistics.mean(r["segment_stats"]["mean_pred_segments"] for r in rows.values()),
                statistics.mean(r["segment_stats"]["mean_gt_segments"] for r in rows.values()))

    def early_stopped(self, ds: str, tok: str, arm: str) -> list[int]:
        """Seeds whose OT fit converged in fewer than 50 outer iterations (the cap) -- these cells sometimes land
        several points away from the seeds that ran to the cap, which explains most of the seed spread on the
        no-contextualizer arms (see the Phase 6 plan)."""
        rows = self.units.get((ds, tok, arm, default_window(ds), TOK.SPECS[tok].span), {})
        return sorted(s for s, r in rows.items() if r.get("ot_status") == "converged" and (r.get("outer_iterations") or 999) < 50)


def paired(v: dict[int, float], base: dict[int, float]):
    """(mean difference, per-seed differences) over the seeds both sides have; None unless all 3 seeds."""
    seeds = sorted(set(v) & set(base))
    if len(seeds) < len(SEEDS):
        return None
    d = [v[s] - base[s] for s in seeds]
    return statistics.mean(d), d


def passes_bar(p) -> bool:
    return p is not None and p[0] >= BAR and all(x > 0 for x in p[1])


def delta_text(p) -> str:
    return "n/a" if p is None else f"{p[0]:+.2f} ({' / '.join(f'{x:+.1f}' for x in p[1])})"


def probe_label(rises: list[bool]) -> str:
    """How many of the three probes rise over T0: all / some / none (a disagreement stays visible)."""
    if not rises:
        return "n/a"
    return "all rise" if all(rises) else ("none rise" if not any(rises) else f"some rise ({sum(rises)}/{len(rises)})")


def gate_lines(out: Path) -> list[str]:
    path = out / "gates.json"
    if not path.exists():
        return ["_no gates recorded_"]
    g = json.loads(path.read_text())
    rows = []
    for k, v in sorted(g.items()):
        detail = (", ".join(f"{m} {d:.4f}" for m, d in v["diff"].items()) if "diff" in v else
                  (f"checks {v['checks']}" if "checks" in v else v.get("reason", "")))
        if k == "g3" and "median_extent_channels_3_5" in v:
            detail = f"median across-joint extent: channels 3-5 {v['median_extent_channels_3_5']:.3f} vs 0-2 {v['median_extent_channels_0_2']:.3f}"
        rows.append([k, v.get("status"), detail])
    return [R4.table(["gate", "status", "detail"], rows)]


def dataset_table(res: Results, ds: str) -> list[str]:
    rows = []
    for tok in ("t0_raw",) + tuple(k for k in NEW_TOKENIZERS if ds in TOK.SPECS[k].datasets):
        for arm in ARMS:
            v = res.f1(ds, tok, arm)
            if not v:
                continue
            base = res.f1(ds, "t0_raw", arm)
            p = None if tok == "t0_raw" or (tok == "t4_smq" and arm == "context") else paired(v, base)
            note = "" if tok != "t4_smq" or arm != "context" else " (guard 14, span 16: not comparable with T0)"
            segs = res.segs(ds, tok, arm) if tok != "t0_raw" else None
            stopped = res.early_stopped(ds, tok, arm) if tok != "t0_raw" else []
            flag = f" (early stop: seed(s) {', '.join(map(str, stopped))})" if stopped else ""
            rows.append([TOK.SPECS[tok].label + " " + tok.split("_", 1)[1] if tok != "t0_raw" else "T0 raw (baseline)", arm + note,
                         " / ".join(R4.fmt(v.get(s_)) for s_ in SEEDS) + flag, R4.mean_sd(v),
                         "—" if tok == "t0_raw" else delta_text(p), R4.fmt(segs[0], 1) if segs else "—",
                         R4.fmt(segs[1], 1) if segs else "—"])
    if not rows:
        return []
    return [f"### {ds}", "", R4.table(["tokenizer", "arm", "F1@50 per seed (111 / 222 / 1538574472)", "mean ± sd",
                                      "Δ vs T0, same arm (paired)", "pred segs/rec", "gt segs/rec"], rows)]


def h6_lines(res: Results) -> tuple[list[str], dict]:
    L, supported = [], {}
    for ds in ("babel2", "lara"):
        for tok in NEW_TOKENIZERS:
            if ds not in TOK.SPECS[tok].datasets:
                continue
            arms = ["centre"] + (["context"] if tok in CONTEXT_COMPARABLE else [])
            for arm in arms:
                p = paired(res.f1(ds, tok, arm), res.f1(ds, "t0_raw", arm))
                if p is None:
                    continue
                ok = passes_bar(p)
                fragile = ok and min(p[1]) < 1.0
                supported.setdefault(ds, {}).setdefault(tok, []).append((arm, ok))
                verdict = "passes, fragile (smallest per-seed gain < 1 point)" if fragile else                     ("beats T0 by the bar" if ok else "does not beat T0 by the bar")
                L.append(f"- {ds} / {TOK.SPECS[tok].label} {arm}: Δ vs T0 {delta_text(p)} -> **{verdict}**")
    return L, supported


def h7_lines(res: Results, supported: dict) -> list[str]:
    L = []
    for ds in ("babel2", "babel3", "lara"):
        t0_gap = None
        c, m = res.f1(ds, "t0_raw", "context"), res.f1(ds, "t0_raw", "centre")
        if len(set(c) & set(m)) == len(SEEDS):
            t0_gap = {s: c[s] - m[s] for s in SEEDS}
        for tok in CONTEXT_COMPARABLE:
            if ds not in TOK.SPECS[tok].datasets:
                continue
            c, m = res.f1(ds, tok, "context"), res.f1(ds, tok, "centre")
            if len(set(c) & set(m)) < len(SEEDS):
                continue
            gap = {s: c[s] - m[s] for s in SEEDS}
            mg = statistics.mean(gap.values())
            kind = ("helps" if all(x >= BAR for x in gap.values()) else "hurts" if all(x <= -BAR for x in gap.values()) else "neutral")
            moved = (mg - statistics.mean(t0_gap.values())) if t0_gap else None
            h6_here = any(ok for _, ok in supported.get(ds, {}).get(tok, [])) if ds in supported else False
            L.append(f"- {ds} / {TOK.SPECS[tok].label}: context − centre = {mg:+.2f} ({' / '.join(f'{x:+.1f}' for x in gap.values())}) "
                     f"-> **{kind}**; T0's gap {statistics.mean(t0_gap.values()):+.2f}" if t0_gap else
                     f"- {ds} / {TOK.SPECS[tok].label}: context − centre = {mg:+.2f} -> **{kind}**")
            if moved is not None:
                L[-1] += f"; moved by {moved:+.2f}" + (f" -> **H7 supported on this tokenizer** (it also satisfies H6)" if (moved >= BAR and h6_here) else "")
    for ds in ("babel2", "babel3", "lara", "hugadb"):
        c, m = res.f1(ds, "t4_smq", "context"), res.f1(ds, "t4_smq", "centre")
        seeds = sorted(set(c) & set(m))
        if seeds:
            L.append(f"- {ds} / T4 (descriptive; other masking): context − centre = "
                     f"{statistics.mean(c[s] - m[s] for s in seeds):+.2f} over {len(seeds)} seed(s)")
    return L


def probe_tables(out: Path, probe_run: str) -> list[str]:
    """Per (dataset, tokenizer): the three probes on the PCA feature against T0, and the all/some/none label."""
    cells = {}
    for path in sorted((out / "probes").glob("*.json")):
        d = json.loads(path.read_text())
        if "cell" not in d and d.get("tokenizer") != "t0_raw":
            continue
        for r in d["rows"]:
            if r["representation"] != "pca":
                continue
            cells.setdefault((d["dataset"], d["tokenizer"]), {}).setdefault(r["probe"], []).append(r["balanced_accuracy"])
    if not cells:
        return []
    phase0 = R4.probe_means(ROOT / "results" / "week5" / "timescale_inspection" / probe_run / "phase0" / "probes.csv")
    rows = []
    for (ds, tok), probes in sorted(cells.items()):
        if tok == "t0_raw":
            continue
        base = {p: (phase0.get((ds, "global", p, "pca"), (None, 0))[0] if p != "nn_white" else None) for p in PROBES}
        white0 = cells.get((ds, "t0_raw"), {}).get("nn_white")
        if white0:
            base["nn_white"] = statistics.mean(white0)
        rises, row = [], [ds, TOK.SPECS[tok].label]
        for p in PROBES:
            new = statistics.mean(probes[p]) if probes.get(p) else None
            row.append(f"{R4.fmt(base[p], 3)} → {R4.fmt(new, 3)}")
            if new is not None and base[p] is not None:
                rises.append(new > base[p])
        rows.append(row + [probe_label(rises), str(len(probes.get("linear", [])))])
    return ["## Per-token probes on the PCA feature (balanced accuracy, T0 → new; mechanism label, not part of H6)", "",
            "For T1 (whitening), the linear probe and the whitened 1-NN probe are invariant to PCA whitening by "
            "construction -- both standardize the feature before fitting, so whitening the PCA space ahead of that "
            "changes nothing. Only the plain 1-NN probe can move for T1, and T1's label is computed from it alone "
            "(the table still shows all three; the invariant two read unchanged, not 'risen').", "",
            R4.table(["dataset", "tokenizer", "linear", "1-NN", "1-NN whitened", "label", "seeds probed"], rows)]


def span_window_sections(res: Results, out: Path, selected: str | None) -> list[str]:
    L = []
    if not selected:
        return ["## Span and window", "", "_not run: the selection (stage D) was not reached or stage A is incomplete._", ""]
    L += ["## Span (stage D) and window (stage E) on the selected tokenizer", "",
          f"Selected on BABEL-2 only, frozen in `selection.json`: **{selected}** ({TOK.SPECS[selected].label}). Span 8 and window "
          "default come from stage A (or the T0 baselines).", ""]
    verdict = {}
    rows = []
    for ds in ("babel2", "babel3"):
        base = res.f1(ds, selected, "context")
        for span in (2, 4, 8):
            v = res.f1(ds, selected, "context", span=span)
            if not v:
                continue
            p = None if span == 8 else paired(v, base)
            rows.append([ds, span, " / ".join(R4.fmt(v.get(s)) for s in SEEDS), R4.mean_sd(v), "—" if span == 8 else delta_text(p)])
            if p is not None:
                verdict.setdefault(ds, []).append((span, p))
    if rows:
        L += ["### Span (`context`, F1@50)", "", R4.table(["dataset", "mask span", "per seed", "mean ± sd", "Δ vs span 8 (paired)"], rows)]
    h2 = "undetermined"
    per_dataset_lines = []
    for ds in ("babel2", "babel3"):
        cells = verdict.get(ds, [])
        beat_ds = [sp for sp, p in cells if passes_bar(p)]
        if cells:
            per_dataset_lines.append(f"- {ds}: " + (f"span(s) {', '.join(map(str, beat_ds))} beat span 8 by the bar"
                                                    if beat_ds else "no shorter span beats span 8 by the bar") +
                                     " (" + "; ".join(f"span {sp} Δ {delta_text(p)}" for sp, p in cells) + ")")
    if per_dataset_lines:
        L += ["", "Per-dataset bar result (before the combined verdict):", ""] + per_dataset_lines
    dev = verdict.get("babel2", [])
    if dev:
        beat = [sp for sp, p in dev if passes_bar(p)]
        conf = verdict.get("babel3", [])
        same_dir = bool(conf) and any(sp in beat and p[0] > 0 for sp, p in conf)
        beat_babel3_alone = [sp for sp, p in conf if passes_bar(p)]
        h2 = ("supported (spans " + ", ".join(map(str, beat)) + " beat span 8 on BABEL-2; BABEL-3 moves the same way)" if beat and same_dir
              else "BABEL-2 beats span 8 by the bar but BABEL-3 does not move the same way" if beat
              else "not supported (no shorter span beats span 8 by the bar on BABEL-2)")
        if beat_babel3_alone:
            h2 += f"; BABEL-3 also beats span 8 by the bar on its own at span(s) {', '.join(map(str, beat_babel3_alone))}"
    diag = []
    for path in sorted((out / "span_diag").glob("*.json")):
        d = json.loads(path.read_text())
        diag.append(d)
    if diag:
        agg: dict = {}
        for d in diag:
            agg.setdefault((d["dataset"], d["span"]), []).append(d)
        drows = [[ds, sp, len(v), R4.fmt(statistics.mean(x["share_targets_crossing"] for x in v), 3),
                  R4.fmt(statistics.mean(x["within_loss"] for x in v), 3), R4.fmt(statistics.mean(x["crossing_loss"] for x in v), 3),
                  R4.fmt(statistics.mean(x["within_targets"] + x["crossing_targets"] for x in v), 0)]
                 for (ds, sp), v in sorted(agg.items())]
        L += ["### Masked-loss boundary split, recomputed per span (diagnostic)", "",
              R4.table(["dataset", "span", "seeds", "share of scored targets crossing a boundary", "within loss", "crossing loss",
                        "scored targets"], drows)]
    wrows = []
    for ds in ("babel2", "babel3"):
        for w in (4, default_window(ds), 15):
            for arm in ("context", "centre"):
                v = res.f1(ds, selected, arm, window=w)
                if v:
                    mask_s = span_seconds(ds, w, TOK.SPECS[selected].span)
                    wrows.append([ds, w, f"{mask_s:.2f}", f"{span_ratio(ds, w, TOK.SPECS[selected].span):.2f}", arm,
                                 " / ".join(R4.fmt(v.get(s)) for s in SEEDS), R4.mean_sd(v)])
    if wrows:
        L += ["### Window (F1@50)", "",
              "The OT radius in seconds is nearly constant across these windows (about 0.93s / 0.80s / 1.00s for "
              "windows 4/8/15 at the current rule), so it is a minor confound here. The larger confounds are the mask "
              "duration and the context length in seconds, both of which change with the window at a fixed span; "
              "the mask-duration and span-ratio columns make that explicit.", "",
              R4.table(["dataset", "window (frames)", "mask duration (s)", "span ratio", "arm", "per seed",
                        "mean ± sd"], wrows)]
    L += [f"**H2′ (a shorter mask span helps, on the selected tokenizer): {h2}.**", ""]
    return L



def all_metrics_rows(res: Results, cells: list[tuple]) -> list[list]:
    """cells: list of (label, ds, tok_new, arm_new, ds_base, tok_base, arm_base[, span_new, span_base]).
    One row per metric per comparison, with the paired delta and the bar verdict computed on that metric, exactly as
    `paired`/`passes_bar` do for F1@50 elsewhere -- so this table checks whether the F1@50 story also holds on
    MoF, Edit, F1@10 and F1@25, not only on the single declared metric."""
    rows = []
    for cell in cells:
        label, ds, tok_n, arm_n, ds_b, tok_b, arm_b = cell[:7]
        span_n = cell[7] if len(cell) > 7 else None
        span_b = cell[8] if len(cell) > 8 else None
        new_m = res.metrics(ds, tok_n, arm_n, span=span_n)
        base_m = res.metrics(ds_b, tok_b, arm_b, span=span_b)
        if not new_m or not base_m:
            continue
        for m in ALL_METRICS:
            v = {s: row[m] for s, row in new_m.items()}
            base = {s: row[m] for s, row in base_m.items()}
            p = paired(v, base)
            rows.append([label, METRIC_LABELS[m], R4.mean_sd(v), R4.mean_sd(base), delta_text(p),
                        "beats by bar" if passes_bar(p) else ("n/a" if p is None else "no")])
    return rows


def all_metrics_section(res: Results, selected: str | None) -> list[str]:
    """Every key comparison this report makes on F1@50 alone, re-run on all five metrics and shown side by side.
    F1@50 is still the declared, pre-registered metric that decides H6/H7/H2'; this section is a check that the
    pattern is not an artifact of that one choice."""
    cells = [
        ("babel2 T2 centre vs T0", "babel2", "t2_relational", "centre", "babel2", "t0_raw", "centre"),
        ("babel2 T2 context vs T0", "babel2", "t2_relational", "context", "babel2", "t0_raw", "context"),
        ("babel3 T2 context vs T0", "babel3", "t2_relational", "context", "babel3", "t0_raw", "context"),
        ("lara T2 centre vs T0", "lara", "t2_relational", "centre", "lara", "t0_raw", "centre"),
        ("lara T2 context vs T0", "lara", "t2_relational", "context", "lara", "t0_raw", "context"),
        ("lara T3b context vs T0", "lara", "t3b_lara_pos64", "context", "lara", "t0_raw", "context"),
        ("lara T4 centre vs T0", "lara", "t4_smq", "centre", "lara", "t0_raw", "centre"),
    ]
    if selected:
        cells += [
            (f"babel2 {TOK.SPECS[selected].label} span 2 vs span 8", "babel2", selected, "context", "babel2", selected,
             "context", 2, 8),
            (f"babel2 {TOK.SPECS[selected].label} span 4 vs span 8", "babel2", selected, "context", "babel2", selected,
             "context", 4, 8),
            (f"babel3 {TOK.SPECS[selected].label} span 2 vs span 8", "babel3", selected, "context", "babel3", selected,
             "context", 2, 8),
        ]
    rows = all_metrics_rows(res, cells)
    if not rows:
        return []
    return ["## Verification on every metric, not F1@50 alone", "",
            "F1@50 is the pre-declared metric for every verdict above (H6, H7, H2'). This table re-runs the same "
            "paired comparisons on MoF, Edit, F1@10 and F1@25 too, read from the identical saved rows, so a result "
            "driven by the choice of F1@50 would show up here as a metric that moves the other way.", "",
            R4.table(["comparison", "metric", "new mean ± sd", "base mean ± sd", "Δ (paired)", "bar"], rows), ""]


def t3_lines(res: Results) -> list[str]:
    L = []
    for arm in ARMS:
        t3, t3b, t0 = (res.f1("lara", k, arm) for k in ("t3_lara_pos256", "t3b_lara_pos64", "t0_raw"))
        a, b = paired(t3, t3b), paired(t3b, t0)
        if a is not None or b is not None:
            L.append(f"- LARa / {arm}: truncation effect (T3 − T3b) {delta_text(a)}; orientation effect (T3b − T0) {delta_text(b)}")
    return L


def best_vs_smq(res: Results) -> list[str]:
    """SMQ's released checkpoint, re-scored here with dataset-level (`global`) Hungarian matching
    (`results/e0/*_pretrained.json`), against the best arm found in this phase. The "best" column is selected after
    the fact from every arm x tokenizer combination with a full seed count, so it is optimistic -- it is not a single
    pre-declared comparison. HuGaDB's single-seed `context` rows are excluded from "best" (3 seeds are required there
    too), so HuGaDB is reported only where it has one."""
    rows, n_considered = [], 0
    for ds in DATASETS:
        best = None
        for tok in ("t0_raw",) + NEW_TOKENIZERS:
            for arm in ARMS:
                v = res.f1(ds, tok, arm)
                if len(v) >= len(SEEDS):
                    n_considered += 1
                    m = statistics.mean(v.values())
                    if best is None or m > best[0]:
                        best = (m, f"{TOK.SPECS[tok].label} {arm}")
        if best:
            rows.append([ds, R4.fmt(SMQ_REFERENCE[ds], 1), R4.fmt(best[0], 2), best[1]])
    L = [f"## Reference: SMQ's released checkpoint (re-scored here, global Hungarian) vs. the best-of-{n_considered} "
         "arm found here (optimistic; not a pre-declared comparison)", ""]
    if rows:
        L += [R4.table(["dataset", "SMQ F1@50 (released checkpoint)", f"best-of-{n_considered} here (mean, 3 seeds)",
                        "which"], rows), ""]
    return L if rows else []


def write_report(out: Path, context_run: str = "rkx_v2", direct_run: str = "ts1_v1", probe_run: str = "ts0_v1") -> Path:
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    res = Results(out, context_run, direct_run)
    sel_path = out / "selection.json"
    selected = json.loads(sel_path.read_text())["selected"] if sel_path.exists() else None
    L = [f"# Time-scale inspection, phase 5: the tokenizer — {manifest['run_id']}", "",
         f"Status `{manifest.get('status')}`. Radius `{manifest['identity']['radius']}`, scale `{manifest['identity']['scale']}`. "
         f"Baselines are read, not rebuilt: T0 `context` from `{context_run}`, T0 `centre`/`pca` from `{direct_run}` Phase 1 `inputs`. "
         "The primary number is F1@50 (selected checkpoint, last rung); the bar is ≥ 3 points on the mean with all 3 seeds in the "
         "same direction. T4's `context` arm is trained with guard 14 / span 16 and is not comparable with T0's.", ""]
    if manifest.get("not_run"):
        L += ["## Blocked by a gate", ""] + [f"- `{k}`: {v}" for k, v in sorted(manifest["not_run"].items())] + [""]
    L += ["## Gates", ""] + gate_lines(out)
    L += ["## F1@50 by dataset", ""]
    for ds in DATASETS:
        L += dataset_table(res, ds)
    if any(res.f1("hugadb", t, "context") for t in NEW_TOKENIZERS):
        L += ["### HuGaDB transfer check (descriptive, seed 111 only)", ""]
        for tok in NEW_TOKENIZERS:
            v = res.f1("hugadb", tok, "context").get(111)
            if v is not None:
                flag = "FLAGGED" if v < HUGADB_T0_CONTEXT_SEED111 - HUGADB_FLAG else "ok"
                L.append(f"- {TOK.SPECS[tok].label} `context` {v:.2f} vs T0 {HUGADB_T0_CONTEXT_SEED111:.2f}: **{flag}** "
                         f"(flag below −{HUGADB_FLAG:g})" + (" — different masking" if tok == "t4_smq" else ""))
        L.append("")
    h6, supported = h6_lines(res)
    L += ["## H6: the tokenizer limits skeleton datasets (F1@50 alone decides)", ""]
    L += h6 or ["- not enough seeds finished yet."]
    wins = sorted({f"{ds}/{TOK.SPECS[t].label}" for ds, d in supported.items() for t, v in d.items() if any(ok for _, ok in v)})
    L += ["", f"**H6: {'supported by ' + ', '.join(wins) if wins else ('not supported' if h6 else 'undetermined')}.**", ""]
    L += ["## H7: was the contextualizer's failure conditional on the tokens?", ""] + (h7_lines(res, supported) or ["- not enough seeds finished yet."]) + [""]
    t3 = t3_lines(res)
    if t3:
        L += ["## LARa: what T3 and T3b separate", ""] + t3 + [""]
    L += probe_tables(out, probe_run)
    L += all_metrics_section(res, selected)
    L += span_window_sections(res, out, selected)
    L += best_vs_smq(res)
    path = out / "REPORT.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
