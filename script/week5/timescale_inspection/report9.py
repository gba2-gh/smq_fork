"""Phase 9 report, written only from saved files (PHASE9_LEARNED_UNITS.md, Revision 3).

Cells (span 2, last rung, paired by seed): R is ts5_v1's T2 `context` unit (read). G, I1q, I1 (T2-code grouping "tg"; I1
also with the I2-code grouping "ig") and I2 (both) are this phase's units. Every comparison is read only with three
seeds; "within the bar" means comparable performance, never equivalence; Q1a-Q1c are effects of those specific contrasts,
not of "geometry" or "continuous input" in general.
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
NAME = {"R": "R (T2 ids)", "g": "G (geometry-init ids)", "i1q": "I1q (centroid)", "i1": "I1 (continuous)", "i2": "I2 (refined targets)"}
# (id, text, (arm, tag), (arm, tag), declared)
COMPARISONS = (
    ("Q1a", "G vs R: initial geometry (same parameterization)", ("g", "tg"), ("R", "tg"), True),
    ("Q1b", "I1q vs G: free table vs input tied to the centroid (same geometric start)", ("i1q", "tg"), ("g", "tg"), True),
    ("Q1c", "I1 vs I1q: detail inside the K-means cell (actual features vs centroid)", ("i1", "tg"), ("i1q", "tg"), True),
    ("Q2", "I2 vs I1, both with the I2-code grouping: do refined contextual targets help? (primary)", ("i2", "ig"), ("i1", "ig"), True),
    ("Q2r", "I2 vs I1, both with the T2-code grouping (robustness)", ("i2", "tg"), ("i1", "tg"), True),
    ("Q3a", "I2 (I2-code grouping) vs R (descriptive: the total effect)", ("i2", "ig"), ("R", "tg"), False),
    ("Q3b", "I2 (T2-code grouping) vs R (descriptive)", ("i2", "tg"), ("R", "tg"), False),
    ("D1", "I1: I2-code grouping vs T2-code grouping (OT-initialization effect, descriptive)", ("i1", "ig"), ("i1", "tg"), False),
    ("D2", "I2: I2-code grouping vs T2-code grouping (OT-initialization effect, descriptive)", ("i2", "ig"), ("i2", "tg"), False),
)
PROSPECTIVE = (
    ("Q4a", "I1 vs R on BABEL-1 (prospective; BABEL-1 is not untouched)", ("i1", "tg"), ("R", "tg"), True),
    ("Q4b", "I2 (I2-code grouping) vs R on BABEL-1 (prospective)", ("i2", "ig"), ("R", "tg"), True),
    ("Q2", "I2 vs I1, both with the I2-code grouping", ("i2", "ig"), ("i1", "ig"), True),
    ("D1", "I1: I2-code grouping vs T2-code grouping (descriptive)", ("i1", "ig"), ("i1", "tg"), False),
)


def analysis_hash() -> str:
    h = hashlib.sha256()
    for name in ("report9.py", "units9.py"):
        h.update(name.encode())
        h.update((Path(__file__).parent / name).read_bytes())
    return h.hexdigest()[:16]


def read_info(path: Path) -> dict | None:
    if not path.exists():
        return None
    i = json.loads(path.read_text())
    if i.get("status") != "done":
        return None
    return dict(R5r._norm_row(max(i["ladder"], key=lambda r: r["rung_steps"])), pred_segs=i["segment_stats"]["mean_pred_segments"],
                gt_segs=i["segment_stats"]["mean_gt_segments"], nn=i.get("nn_same_recording"))


def reference(source_root: Path, ds: str, seed: int) -> dict | None:
    if ds == "babel1":
        return read_info(Path(source_root) / "phase7" / "units" / f"babel1__t2_relational_b1__seed{seed}__span2__context" / "info.json")
    return R8.t2_reference(source_root, ds, seed)


def ref_predictions(source_root: Path, ds: str, seed: int):
    if ds == "babel1":
        return R8._load_preds(Path(source_root) / "phase7" / "units" / f"babel1__t2_relational_b1__seed{seed}__span2__context" / "predictions.npz")
    phase, cid = R8.t2_cell_id(ds, seed)
    return R8._load_preds(Path(source_root) / "phase7" / "s0" / f"{phase}__{cid}__context" / "predictions.npz")


def cells(out: Path, source_root: Path, datasets) -> dict:
    """{(arm, tag): {dataset: {seed: metrics}}}, with R under ('R', 'tg')."""
    res: dict = {}
    for ds in datasets:
        for s in SEEDS:
            ref = reference(source_root, ds, s)
            if ref:
                res.setdefault(("R", "tg"), {}).setdefault(ds, {})[s] = ref
            for p in (Path(out) / "units").glob(f"{ds}__*__seed{s}__*/info.json"):
                parts = p.parent.name.split("__")
                arm, tag = parts[1], parts[-1]
                info = read_info(p)
                if info and arm in ("g", "i1q", "i1", "i2"):
                    res.setdefault((arm, tag), {}).setdefault(ds, {})[s] = info
    return res


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


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def f(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def class_rows(out: Path, source_root: Path, datasets) -> list[list]:
    rows = []
    for ds in datasets:
        for arm, tag in (("i1", "tg"), ("i2", "ig")):
            ref, mine = [], []
            for s in SEEDS:
                a = ref_predictions(source_root, ds, s)
                b = R8._load_preds(Path(out) / "units" / f"{ds}__{arm}__seed{s}__{tag}" / "predictions.npz")
                if a is None or b is None:
                    continue
                ref.append(R8.class_scores(ds, a))
                mine.append(R8.class_scores(ds, b))
            if not mine:
                continue
            for n in [c for c in mine[0] if c != "MoF"]:
                rr, rm = statistics.mean(x[n][0] for x in ref), statistics.mean(x[n][0] for x in mine)
                rows.append([ds, f"{arm}/{tag}", n, f"{mine[0][n][2]:.1f}", len(mine), f"{rr:.1f}", f"{rm:.1f}", f"{rm - rr:+.1f}"])
    return rows


def write_report(out: Path, source_root: Path, prospective: bool = False) -> Path:
    out = Path(out)
    m = json.loads((out / "manifest.json").read_text())
    datasets = ("babel1",) if prospective else ("babel2", "babel3", "lara")
    comps = PROSPECTIVE if prospective else COMPARISONS
    L = [f"# Time-scale inspection, phase 9{'p (prospective BABEL-1)' if prospective else ''}: iterative contextual target refinement — {m['run_id']}", "",
         f"Status `{m.get('status')}`. Rules declared in PHASE9_LEARNED_UNITS.md (Revision 3) before the run. Primary metric F1@50 at the "
         f"last rung, mask span 2, paired by seed; the bar is a mean gain >= {BAR:g} with every seed in the same direction, read only with "
         "three seeds. 'Within the bar' means comparable performance, not equivalence. This phase tests contextual target refinement; it "
         "cannot show discovery of local motion units (every arm reads T2's PCA-64 summaries and I2's codes are contextual).", ""]
    g = json.loads((out / "gates.json").read_text()) if (out / "gates.json").exists() else {}
    if g:
        L += ["## Gates", "", R4.table(["gate", "status", "detail"], [[k, v.get("status"), str(v.get("checks", v.get("reason", "")))[:260]]
                                                                      for k, v in sorted(g.items())]), ""]
    if m.get("not_run"):
        L += ["## Not run", ""] + [f"- `{k}`: {v}" for k, v in sorted(m["not_run"].items())] + [""]

    have = cells(out, source_root, datasets)
    rows = []
    for ds in datasets:
        for key in (("R", "tg"), ("g", "tg"), ("i1q", "tg"), ("i1", "tg"), ("i1", "ig"), ("i2", "tg"), ("i2", "ig")):
            v = have.get(key, {}).get(ds)
            if not v:
                continue
            seeds = sorted(v, key=SEEDS.index)
            fs = [v[s]["F1_50"] for s in seeds]
            rows.append([ds, NAME[key[0]], key[1] if key[0] != "R" else "t2 (native)", " / ".join(f"{s}:{v[s]['F1_50']:.1f}" for s in seeds), len(fs),
                         f"{statistics.mean(fs):.2f}", f"{statistics.mean(v[s]['pred_segs'] for s in seeds):.1f}",
                         f"{statistics.mean(v[s]['gt_segs'] for s in seeds):.1f}", f(_mean([v[s].get('nn') for s in seeds]))])
    L += ["## Cells (F1@50, span 2). `tg` = T2-code grouping initializes OT, `ig` = I2-code grouping", "",
          R4.table(["dataset", "arm", "grouping", "per seed", "n", "mean", "pred segs/rec", "gt segs/rec", "same-recording NN share"], rows)]

    L += ["## Comparisons (A minus B, paired by seed)", ""]
    allm = []
    for cid, text, a, b, declared in comps:
        L += [f"### {cid}{'' if declared else ' (descriptive)'}: {text}", ""]
        for ds in datasets:
            A, B = have.get(a, {}).get(ds, {}), have.get(b, {}).get(ds, {})
            p = paired(A, B)
            L.append(f"- **{ds}**: {fmt_d(p) if p else 'no common cells'} over {len(p[2]) if p else 0} seeds -> {reading(p)}")
            for met in METRICS:
                pm = paired(A, B, met)
                if pm:
                    allm.append([cid, ds, f"{a[0]}/{a[1]} − {b[0]}/{b[1]}", LABEL[met], fmt_d(pm), len(pm[2]), reading(pm)])
        L.append("")
    L += ["## Every metric", "", R4.table(["id", "dataset", "A − B", "metric", "Δ (per seed)", "n", "reading"], allm)]

    trows = []
    for p in sorted((out / "jobs").glob("*/training_info.json")):
        i = json.loads(p.read_text())
        parts = p.parent.name.split("__")
        mv = i.get("masked_val") or {}
        trows.append([parts[0], parts[1], parts[2].replace("seed", ""), i["roles"]["selected"], f(i["best_val_loss"]), f(mv.get("val_accuracy")),
                      f"{i['seconds']:.0f}"])
    L += ["## Training (validation loss is comparable only within an arm; masked accuracy is the difficulty of the prediction task)", "",
          R4.table(["dataset", "arm", "seed", "selected step", "val loss", "masked-code val accuracy", "seconds"], trows), ""]

    drows = []
    for ds in datasets:
        dj = out / "diagnostics" / f"{ds}.json"
        seed_files = sorted((out / "diagnostics").glob(f"{ds}__seed*.json"))
        if not seed_files:
            continue
        per = [json.loads(p.read_text()) for p in seed_files]
        agg = json.loads(dj.read_text()) if dj.exists() else {"layers": {}, "t2_cross_seed_nmi": None}
        t2 = [d["t2"] for d in per]
        drows.append([ds, "T2 codes (reference)", f(_mean([x["occupied"] for x in t2]), 0), f(_mean([x["perplexity"] for x in t2]), 1),
                      f"{f(agg.get('t2_cross_seed_nmi'))} (chance {f(agg.get('chance_nmi'))}, {agg.get('windows', '?')} windows)", f(_mean([x["recording_nmi"] for x in t2])), f(_mean([x["subject_nmi"] for x in t2])),
                      f(_mean([x["nn_same_recording"] for x in t2]))])
        for k in range(1, 5):
            ls = [d[f"layer{k}"] for d in per]
            drows.append([ds, f"I1 layer {k}{' (targets)' if k == 2 else ''}", f(_mean([x["occupied"] for x in ls]), 0),
                          f(_mean([x["perplexity"] for x in ls]), 1), f((agg.get("layers", {}).get(str(k)) or {}).get("cross_seed_nmi")),
                          f(_mean([x["recording_nmi"] for x in ls])), f(_mean([x["subject_nmi"] for x in ls])),
                          f(_mean([x["nn_same_recording"] for x in ls]))])
    L += ["## Layer diagnostics (label-free; they choose nothing — layer 2 was fixed in advance)", "",
          "High cross-seed agreement is not read as quality: stable recording identity or coarse posture would also score highly. "
          "Association with recording and subject (normalized mutual information of codes with that metadata; nearest-neighbour share "
          "ignoring temporal neighbours) shows whether units separate recordings rather than actions. With 512 codes and few windows "
          "the NMI of unrelated labelings is far above 0 (finite-sample bias), so cross-seed NMI is read only against the chance value "
          "shown for T2 (one seed's codes shuffled across windows) and against T2 at the same window count.", "",
          R4.table(["dataset", "codes", "occupied", "perplexity", "cross-seed NMI", "recording NMI", "subject NMI", "same-recording NN share"], drows), ""]

    L += ["## Per-class frame recall against R (uses labels; analysis only)", "", "Global Hungarian mapping as the evaluator; small classes are noisy.", ""]
    try:
        L += [R4.table(["dataset", "arm/grouping", "class", "frame share %", "n seeds", "R recall", "recall", "Δ recall"], class_rows(out, source_root, datasets))]
    except Exception as exc:  # the report must still be written
        L += [f"_per-class table failed: {type(exc).__name__}: {exc}_", ""]
    path = out / "REPORT.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
