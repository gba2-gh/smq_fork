"""Phase 7 criteria (stage K) and report, written only from saved files.

Sources of frame predictions and scores per (dataset, span, seed), `context` arm:
  BABEL-2/3, LARa (T2) and HuGaDB spans 2/4 (T0): phase7/s0/<phase>__<cell>__context (re-decodes; only those that
      reproduced their saved ladder exactly are used)
  HuGaDB span 8 (T0): rkx_v2/evals/<identity>/predictions.npz via rkx_v2/eval_records, scores from segmentation.csv
  BABEL-1 (T2): phase7/units (the held-out cells, decoded with predictions saved)
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import numpy as np

from script.fsq_reconstruction.experiment import write_json_atomic
from script.quantized_input_comparison.artifacts import Loader
from script.week5.raw_kmeans_cross_dataset import datasets as D
from script.week5.timescale_inspection import criteria7 as K
from script.week5.timescale_inspection import report4 as R4
from script.week5.timescale_inspection import report5 as R5r
from script.week5.timescale_inspection import run_phase7 as R7  # registers the BABEL-1 tokenizer keys
from script.week5.timescale_inspection import tokenizers as TOK

ROOT = R4.ROOT
BAR = R4.BAR
METRICS = R5r.ALL_METRICS
CRITERIA_DATASETS = ("babel1", "babel2", "babel3", "lara", "hugadb")
TOKENIZER_FOR = {"babel1": "t2_relational_b1", "babel2": "t2_relational", "babel3": "t2_relational",
                 "lara": "t2_relational", "hugadb": "t0_raw"}
SMQ_BABEL1_F1_50 = 22.38  # results/e0/babel1_pretrained.json, global
SEED_ORDER = (111, 222, 333, 444, 1538574472)


def _npz(path: Path, suffix: str = "") -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {(k[:-len(suffix)] if suffix and k.endswith(suffix) else k): z[k] for k in z.files if not suffix or k.endswith(suffix)}


def last(ladder):
    return max(ladder, key=lambda r: r["rung_steps"])


def collect(out: Path, source_root: Path, context_run: str) -> dict:
    """(dataset, span) -> {seed: dict(preds={name: frames}, metrics={...}, pred_segs, gt_segs)}"""
    data: dict = {}
    for info_path in sorted((Path(out) / "s0").glob("*/info.json")):
        i = json.loads(info_path.read_text())
        if i.get("status") != "done" or not i.get("reproduced"):
            continue
        data.setdefault((i["dataset"], i["span"]), {})[i["seed"]] = dict(
            preds=_npz(info_path.parent / "predictions.npz"), metrics=R5r._norm_row(last(i["ladder"])))
    for info_path in sorted((Path(out) / "units").glob(f"{R7.HELDOUT}__t2_relational_b1__*__context/info.json")):
        i = json.loads(info_path.read_text())
        if i.get("status") != "done" or i["window"] != D.SPECS[R7.HELDOUT].window or not (info_path.parent / "predictions.npz").exists():
            continue
        data.setdefault((R7.HELDOUT, i["span"]), {})[i["seed"]] = dict(
            preds=_npz(info_path.parent / "predictions.npz"), metrics=R5r._norm_row(last(i["ladder"])))
    rkx = ROOT / "results" / "week5" / "raw_kmeans_cross_dataset" / context_run
    base = R4.load_context_baseline(context_run).get(("hugadb", "global"), {})
    for seed, row in base.items():
        rec = rkx / "eval_records" / f"hugadb__global__seed{seed}__selected.json"
        if not rec.exists():
            continue
        r = json.loads(rec.read_text())
        pred = rkx / "evals" / r["identity"] / "predictions.npz"
        if pred.exists():
            data.setdefault(("hugadb", 8), {})[seed] = dict(preds=_npz(pred, "__frame_pred"), metrics=R5r._norm_row(row))
    return data


def prep_dirs(out: Path, source_root: Path, ds: str, seed: int) -> list[Path]:
    tok = TOKENIZER_FOR[ds]
    pid = f"{ds}__{tok}__seed{seed}"
    return [source_root / "phase5" / "prep" / tok / pid, source_root / "phase6" / "prep" / tok / pid, Path(out) / "prep" / tok / pid]


DECLARED_SPANS = {"babel1": (2, 8), "babel2": (2, 4, 8), "babel3": (2, 4, 8), "lara": (2, 4, 8), "hugadb": (2, 4, 8)}
MIN_COMMON_SEEDS = 3
CRITERIA = ("K1", "K3", "K4", "always_shortest", "always_8")


def analysis_hash() -> str:
    """Hash of the selector and verdict code, recorded with every criteria.json so the declared rules are reproducible."""
    import hashlib
    h = hashlib.sha256()
    for name in ("criteria7.py", "report7.py"):
        h.update(name.encode())
        h.update((Path(__file__).parent / name).read_bytes())
    return h.hexdigest()[:16]


def feature_tau(out: Path, source_root: Path, ds: str, seeds: list[int]) -> tuple[float, dict]:
    spec, tok = D.SPECS[ds], TOKENIZER_FOR[ds]
    per_seed = {}
    for s in seeds:
        d = next((p for p in prep_dirs(out, source_root, ds, s) if (p / "manifest.json").exists()), None)
        if d is None:
            per_seed[s] = dict(tau_s=float("nan"), n_defined=0, n_total=0, missing_preparation=True)
            continue
        L = Loader(d)
        fam = TOK.SPECS[tok].family
        per_seed[s] = K.feature_tau_coverage({n: L.recording(fam, n)["continuous"] for n in L.recording_names()},
                                             spec.window, spec.fps)
    vals = [v["tau_s"] for v in per_seed.values() if not np.isnan(v["tau_s"])]
    return (float(np.median(vals)) if len(vals) == len(seeds) else float("nan")), per_seed


def completeness(data: dict, ds: str) -> tuple[list[int], str | None]:
    """The common seeds of the declared spans, or a reason the dataset is incomplete. Requires every declared span,
    >= MIN_COMMON_SEEDS common seeds, and identical recording coverage (names and frame counts) in every cell."""
    spans = DECLARED_SPANS[ds]
    missing = [sp for sp in spans if (ds, sp) not in data]
    if missing:
        return [], f"declared span(s) {missing} missing (absent, or no re-decode reproduced)"
    seeds = sorted(set.intersection(*(set(data[(ds, sp)]) for sp in spans)))
    if len(seeds) < MIN_COMMON_SEEDS:
        return seeds, f"only {len(seeds)} seed(s) common to every declared span (need {MIN_COMMON_SEEDS})"
    ref = {n: len(v) for n, v in data[(ds, spans[0])][seeds[0]]["preds"].items()}
    for sp in spans:
        for s in seeds:
            cov = {n: len(v) for n, v in data[(ds, sp)][s]["preds"].items()}
            if cov != ref:
                return seeds, f"recording coverage differs at span {sp}, seed {s}"
    return seeds, None


def compute_criteria(out: Path, source_root: Path, context_run: str = "rkx_v2") -> Path:
    """Exploratory, retrospective screening of the declared selectors. Datasets that are not complete are reported as
    incomplete; a selector can then fail (wrong where data exist) but never pass."""
    data = collect(out, source_root, context_run)
    result: dict = dict(datasets={}, criteria=CRITERIA, analysis_hash=analysis_hash(), declared_spans=DECLARED_SPANS,
                        min_common_seeds=MIN_COMMON_SEEDS)
    for ds in CRITERIA_DATASETS:
        spans = list(DECLARED_SPANS[ds])
        seeds, reason = completeness(data, ds)
        if reason:
            result["datasets"][ds] = dict(complete=False, reason=reason, spans=spans, seeds=seeds)
            continue
        spec = D.SPECS[ds]
        preds = {sp: {s: data[(ds, sp)][s]["preds"] for s in seeds} for sp in spans}
        f1 = {sp: statistics.mean(data[(ds, sp)][s]["metrics"]["F1_50"] for s in seeds) for sp in spans}
        k1 = {sp: K.k1_stability(preds[sp]) for sp in spans}
        med = {sp: K.median_pred_duration(preds[sp], spec.fps) for sp in spans}
        tau, tau_seeds = feature_tau(out, source_root, ds, seeds)
        picks = dict(K1=K.pick_k1(k1), K3=K.pick_largest_fitting(spans, med, spec.window, spec.fps),
                     K4=None if np.isnan(tau) else K.pick_largest_fitting(spans, tau, spec.window, spec.fps),
                     always_shortest=min(spans), always_8=8)
        result["datasets"][ds] = dict(complete=True, spans=spans, seeds=seeds, f1_by_span=f1,
                                      acceptable=sorted(K.acceptable_spans(f1)), k1_by_span=k1,
                                      median_pred_duration_s=med, tau_s=tau, tau_by_seed=tau_seeds,
                                      occupancy={sp: K.occupancy(preds[sp]) for sp in spans},
                                      mask_seconds={sp: K.mask_seconds(sp, spec.window, spec.fps) for sp in spans},
                                      picks=picks)
    complete = {d: v["complete"] for d, v in result["datasets"].items()}
    result["gate"] = {}
    for crit in CRITERIA:
        picks = {d: v["picks"][crit] for d, v in result["datasets"].items() if v["complete"]}
        acc = {d: set(v["acceptable"]) for d, v in result["datasets"].items() if v["complete"]}
        result["gate"][crit] = K.gate_declared(picks, acc, complete)
    path = Path(out) / "criteria.json"
    write_json_atomic(path, json.loads(json.dumps(result, default=lambda o: sorted(o) if isinstance(o, set) else str(o))))
    return path


# -- the report ------------------------------------------------------------------------------------------------
def unit_metrics(out: Path, tok: str, arm: str, span: int) -> dict[int, dict]:
    res = {}
    for p in sorted((Path(out) / "units").glob(f"{R7.HELDOUT}__{tok}__*__{arm}/info.json")):
        i = json.loads(p.read_text())
        if i.get("status") == "done" and i["span"] == span and i["window"] == D.SPECS[R7.HELDOUT].window:
            res[i["seed"]] = dict(R5r._norm_row(last(i["ladder"])), pred_segs=i["segment_stats"]["mean_pred_segments"],
                                  gt_segs=i["segment_stats"]["mean_gt_segments"])
    return res


def heldout_cells(out: Path, context_run: str, direct_run: str) -> dict:
    ctx = R4.load_context_baseline(context_run).get((R7.HELDOUT, "global"), {})
    cen = R4.load_direct_baseline(direct_run).get((R7.HELDOUT, "global", "centre"), {})
    t0_8 = {s: R5r._norm_row(r) for s, r in ctx.items()}
    new = unit_metrics(out, "t0_raw_b1", "context", 8)
    clash = set(t0_8) & set(new)
    if clash:
        raise ValueError(f"BABEL-1 T0 span 8: seeds {sorted(clash)} in both rkx_v2 and phase7")
    t0_8.update(new)
    return {"T0 span 8 context": t0_8, "T0 span 2 context": unit_metrics(out, "t0_raw_b1", "context", 2),
            "T0 centre": {s: R5r._norm_row(r) for s, r in cen.items()},
            "T2 span 8 context": unit_metrics(out, "t2_relational_b1", "context", 8),
            "T2 span 2 context": unit_metrics(out, "t2_relational_b1", "context", 2),
            "T2 centre": unit_metrics(out, "t2_relational_b1", "centre", 8)}


def paired(a: dict, b: dict, metric: str = "F1_50", seeds=None):
    common = sorted((set(a) & set(b)) if seeds is None else (set(a) & set(b) & set(seeds)))
    if len(common) < 3:
        return None
    d = [a[s][metric] - b[s][metric] for s in common]
    return statistics.mean(d), d, common


def verdict(p) -> str:
    if p is None:
        return "not enough seeds"
    ok = p[0] >= BAR and all(x > 0 for x in p[1])
    return f"{p[0]:+.2f} ({' / '.join(f'{x:+.1f}' for x in p[1])}) over {len(p[2])} seeds -> **{'passes the bar' if ok else 'does not pass the bar'}**"


def passes(p) -> bool:
    return p is not None and p[0] >= BAR and all(x > 0 for x in p[1])


def write_report(out: Path, source_root: Path, context_run: str = "rkx_v2", direct_run: str = "ts1_v1") -> Path:
    out = Path(out)
    m = json.loads((out / "manifest.json").read_text())
    L = [f"# Time-scale inspection, phase 7: BABEL-1 prospective evaluation and label-free span screening — {m['run_id']}", "",
         f"Status `{m.get('status')}`. Rules declared in PHASE7_LABEL_FREE_SCALE.md before the run. Primary metric F1@50 "
         "at the last rung; bar >= 3 points on the mean with every seed in the same direction.", ""]
    g = json.loads((out / "gates.json").read_text()) if (out / "gates.json").exists() else {}
    if g:
        L += ["## Gates", "", R4.table(["gate", "status", "detail"],
                                       [[k, v.get("status"), str(v.get("checks", v.get("reason", "")))] for k, v in sorted(g.items())]), ""]
    if m.get("not_run"):
        L += ["## Blocked", ""] + [f"- `{k}`: {v}" for k, v in sorted(m["not_run"].items())] + [""]

    cells = heldout_cells(out, context_run, direct_run)
    rows = []
    for name, v in cells.items():
        if not v:
            continue
        seeds = sorted(v, key=SEED_ORDER.index)
        f = [v[s]["F1_50"] for s in seeds]
        segs = [v[s].get("pred_segs") for s in seeds if v[s].get("pred_segs") is not None]
        gts = [v[s].get("gt_segs") for s in seeds if v[s].get("gt_segs") is not None]
        rows.append([name, " / ".join(f"{s}:{v[s]['F1_50']:.1f}" for s in seeds),
                     f"{statistics.mean(f):.2f} ± {statistics.stdev(f) if len(f) > 1 else 0:.2f}", len(f),
                     f"{statistics.mean(segs):.1f}" if segs else "—", f"{statistics.mean(gts):.1f}" if gts else "—"])
    L += ["## Part H: prospective evaluation of the frozen T2 / short-span recipe on BABEL-1", "",
          "**Not an untouched test set.** BABEL-1's T0 `context`/`centre` scores, oracle ceilings and ground-truth duration "
          "statistics were already examined in rkx_v2, Phase 0 and Phase 1, and 3 of its 1,960 recordings are byte-identical "
          "to BABEL-2 recordings and 7 to BABEL-3 recordings. What is new is that the T2 / short-span recipe was frozen on "
          "BABEL-2/3 and LARa before it was ever run here.", "",
          f"SMQ released checkpoint on BABEL-1 (global Hungarian): F1@50 {SMQ_BABEL1_F1_50}. T0 span 8 at seeds "
          "111/222/1538574472 is rkx_v2; at 333/444 it is this phase (gate G7).", "",
          R4.table(["cell", "F1@50 per seed", "mean ± sd", "n seeds", "pred segs/rec", "gt segs/rec"], rows)]
    t2s2, t2s8, t0s8, t0s2, t2c = (cells[k] for k in ("T2 span 2 context", "T2 span 8 context", "T0 span 8 context",
                                                      "T0 span 2 context", "T2 centre"))
    orig = R7.ORIGINAL_SEEDS
    L += ["### Declared comparisons", "",
          f"- **P1** (T2 span 2 vs T2 span 8), original seeds: {verdict(paired(t2s2, t2s8, seeds=orig))}",
          f"- **P1**, all seeds: {verdict(paired(t2s2, t2s8))}",
          f"- **P2** (T2 span 2 vs T0 span 8, the original pipeline), original seeds: {verdict(paired(t2s2, t0s8, seeds=orig))}",
          f"- **P2**, all seeds: {verdict(paired(t2s2, t0s8))}",
          f"- **P3** (descriptive; T0 span 2 vs T0 span 8): {verdict(paired(t0s2, t0s8, seeds=orig))}",
          f"- **P4** (does contextualization add value? T2 span 2 `context` vs T2 `centre`, same tokens, no context): "
          f"{verdict(paired(t2s2, t2c, seeds=orig))}", ""]
    common = sorted(set(t2s2) & set(t2s8) & set(t0s2) & set(t0s8) & set(orig))
    if len(common) >= 3:
        span_t0 = [t0s2[s]["F1_50"] - t0s8[s]["F1_50"] for s in common]
        span_t2 = [t2s2[s]["F1_50"] - t2s8[s]["F1_50"] for s in common]
        inter = [b - a for a, b in zip(span_t0, span_t2)]

        def fmt(xs):
            return f"{statistics.mean(xs):+.2f} ({' / '.join(f'{x:+.1f}' for x in xs)})"

        L += [f"### Tokenizer x span factorial (the {len(common)} seeds with all four cells; T0 span 2 has no seeds 333/444)", "",
              f"- span effect on T0 (span 2 - span 8): {fmt(span_t0)}",
              f"- span effect on T2 (span 2 - span 8): {fmt(span_t2)}",
              f"- interaction (T2 span effect - T0 span effect): {fmt(inter)}", ""]
    vrows = []
    for label, a_, b_ in (("P1 T2 span2 vs span8", t2s2, t2s8), ("P2 T2 span2 vs T0 span8", t2s2, t0s8),
                          ("P4 T2 span2 context vs T2 centre", t2s2, t2c)):
        for met in METRICS:
            p = paired(a_, b_, met, seeds=orig)
            if p:
                vrows.append([label, R5r.METRIC_LABELS[met], f"{p[0]:+.2f} ({' / '.join(f'{x:+.1f}' for x in p[1])})",
                              "passes" if passes(p) else "no"])
    if vrows:
        L += ["### Every metric (same paired comparisons, original seeds)", "",
              R4.table(["comparison", "metric", "Δ (paired)", "bar"], vrows)]

    s0 = [json.loads(p.read_text()) for p in sorted((out / "s0").glob("*/info.json"))]
    nrep = sum(1 for i in s0 if i.get("reproduced"))
    L += ["## Part S0: re-decodes with predictions saved", "",
          f"{nrep}/{len(s0)} re-decodes reproduced their saved ladder exactly (every rung, every metric); "
          f"{len(R7.s0_items())} were planned. Only reproduced units enter the screening."]
    bad = [f"- {i['source_unit']}: max diff {i['max_abs_diff']}" for i in s0 if not i.get("reproduced")]
    L += (["", "Not reproduced (excluded):"] + bad if bad else []) + [""]

    cpath = out / "criteria.json"
    if not cpath.exists():
        L += ["## Part K", "", "_criteria not computed (see the run log)._", ""]
    else:
        c = json.loads(cpath.read_text())
        now = analysis_hash()
        stale = "" if c.get("analysis_hash") == now else f" — **differs from the current code (`{now}`): recompute before reading**"
        L += ["## Part K: label-free span selectors — exploratory, retrospective screening", "",
              "These selectors use no labels, but the spans they are scored against were chosen by F1@50, and choosing "
              "among K1/K3/K4 by how well they score is itself a label-based selection. Any 'pass' here is a candidate "
              "for a prospective test, not evidence of label-free selection.", "",
              f"Analysis hash (criteria7.py + report7.py): `{c.get('analysis_hash')}`{stale}", ""]
        crow, inc = [], []
        for ds, v in c["datasets"].items():
            if not v["complete"]:
                inc.append(f"- **{ds}: incomplete** — {v['reason']}")
                continue
            for sp in v["spans"]:
                k, occ = str(sp), v["occupancy"][str(sp)]
                crow.append([ds, sp, f"{v['mask_seconds'][k]:.2f}", f"{v['f1_by_span'][k]:.2f}",
                             "yes" if sp in v["acceptable"] else "", f"{v['k1_by_span'][k]:.2f}",
                             f"{v['median_pred_duration_s'][k]:.2f}", f"{occ['mean_runs_per_recording']:.1f}",
                             f"{occ['mean_states_per_recording']:.2f}", f"{occ['share_single_state_recordings']:.2f}",
                             len(v["seeds"])])
        if crow:
            L += [R4.table(["dataset", "span", "mask (s)", "F1@50", "acceptable", "K1 stability", "K3 median pred seg (s)",
                            "pred runs/rec", "states/rec", "share 1-state recs", "seeds"], crow), ""]
        trow = []
        for ds, v in c["datasets"].items():
            if v["complete"]:
                cov = "; ".join(f"seed {s}: {t['n_defined']}/{t['n_total']} recordings" for s, t in v["tau_by_seed"].items())
                trow.append([ds, f"{v['tau_s']:.2f}" if v["tau_s"] == v["tau_s"] else "undefined", cov])
        if trow:
            L += ["K4 coverage (recordings where the decorrelation time is defined):", "",
                  R4.table(["dataset", "K4 tau (s)", "coverage"], trow), ""]
        if inc:
            L += ["Incomplete datasets (a selector cannot pass while any declared dataset is incomplete):", ""] + inc + [""]
        prow = []
        for crit in c["criteria"]:
            gd = c["gate"][crit]
            cellv = []
            for d in CRITERIA_DATASETS:
                v = c["datasets"].get(d, {})
                if not v.get("complete"):
                    cellv.append("incomplete")
                else:
                    pk = v["picks"][crit]
                    mark = " ✓" if gd["per_dataset"].get(d) else " ✗"
                    cellv.append(f"{pk if pk is not None else 'undefined'}{mark}")
            prow.append([crit] + cellv + [f"{gd['n_right']}/{gd['n_declared']}", f"**{gd['status']}**"])
        L += ["### Picks and gate", "", R4.table(["selector"] + list(CRITERIA_DATASETS) + ["right", "status"], prow), "",
              "K1 = cross-seed segmentation stability (argmax; can score 100 on collapsed segmentations, hence the "
              "occupancy columns); K3 = largest span whose mask fits inside the model's own median predicted segment "
              "(over-merging can justify the long span that caused it); K4 = largest span whose mask fits inside the "
              "feature decorrelation time (persistence need not mark action boundaries). An undefined pick counts as "
              "wrong. Constants are references.", ""]
    path = out / "REPORT.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
