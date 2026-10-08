"""Phase 8a report, written only from saved files, plus the declared third-seed rule.

Sources: this phase's units (`phase8a/units/<ds>__<tok>__seed<s>__context`, with frame predictions); the T2 span-2
references, read from ts5_v1 (BABEL-2/3: phase5/units, LARa: phase6/units; their frame predictions from phase7/s0).

The third-seed rule (declared in PHASE8A_GRAVITY_HEADING_TOKENS.md): a (dataset, tokenizer) gets the third seed only if
its two-seed mean F1@50 at the last rung is >= PROMISE points above T2's mean on the same two seeds. The decisions
are written to `third_seed.json` before any third-seed job starts. Two-seed cells are screening, never the bar.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

from script.week5.raw_kmeans_cross_dataset import datasets as D
from script.week5.timescale_inspection import report4 as R4
from script.week5.timescale_inspection import report5 as R5r

ROOT = R4.ROOT
BAR = R4.BAR
PROMISE = 1.0  # F1@50 points over T2 on the same screening seeds
SCREEN_SEEDS = (111, 222)
THIRD_SEED = 1538574472
METRICS = R5r.ALL_METRICS
LABEL = R5r.METRIC_LABELS
TOKENS = ("t5_both", "t5_heading", "t5_gravity")
TOKEN_LABEL = {"t5_both": "T5gh", "t5_heading": "T5h", "t5_gravity": "T5g", "t2_relational": "T2"}
DATASETS = ("babel2", "babel3", "lara")
SEED_ORDER = (111, 222, 1538574472)
CLASS_FOCUS = {"babel2": ("stand_up", "kick"), "babel3": ("wave",), "lara": ()}


def analysis_hash() -> str:
    h = hashlib.sha256()
    for name in ("report8a.py", "tokenizers8.py"):
        h.update(name.encode())
        h.update((Path(__file__).parent / name).read_bytes())
    return h.hexdigest()[:16]


def last(ladder):
    return max(ladder, key=lambda r: r["rung_steps"])


def t2_cell_id(ds: str, seed: int) -> tuple[str, str]:
    return ("phase6" if ds == "lara" else "phase5"), f"{ds}__t2_relational__seed{seed}__span2"


def t2_reference(source_root: Path, ds: str, seed: int) -> dict | None:
    phase, cid = t2_cell_id(ds, seed)
    p = Path(source_root) / phase / "units" / f"{cid}__context" / "info.json"
    if not p.exists():
        return None
    i = json.loads(p.read_text())
    if i.get("status") != "done":
        return None
    return dict(R5r._norm_row(last(i["ladder"])), pred_segs=i["segment_stats"]["mean_pred_segments"],
                gt_segs=i["segment_stats"]["mean_gt_segments"])


def t5_cells(out: Path) -> dict:
    """(dataset, tokenizer) -> {seed: metrics}."""
    res: dict = {}
    for p in sorted((Path(out) / "units").glob("*__context/info.json")):
        i = json.loads(p.read_text())
        if i.get("status") != "done" or i["tokenizer"] not in TOKENS:
            continue
        res.setdefault((i["dataset"], i["tokenizer"]), {})[i["seed"]] = dict(
            R5r._norm_row(last(i["ladder"])), pred_segs=i["segment_stats"]["mean_pred_segments"],
            gt_segs=i["segment_stats"]["mean_gt_segments"])
    return res


def paired(a: dict, b: dict, met: str = "F1_50", seeds=None):
    ss = [s for s in SEED_ORDER if s in a and s in b and (seeds is None or s in seeds)]
    if not ss:
        return None
    d = [a[s][met] - b[s][met] for s in ss]
    return statistics.mean(d), d, ss


def t2_cells(source_root: Path, seeds=SEED_ORDER) -> dict:
    return {ds: {s: r for s in seeds if (r := t2_reference(source_root, ds, s)) is not None} for ds in DATASETS}


def third_seed_decisions(out: Path, source_root: Path, screen_seeds=SCREEN_SEEDS) -> dict:
    """Per 'dataset|tokenizer': whether the third seed runs, with the values the decision used."""
    cells, t2 = t5_cells(out), t2_cells(source_root, screen_seeds)
    dec = {}
    for (ds, tok), v in sorted(cells.items()):
        pr = paired(v, t2[ds], "F1_50", seeds=screen_seeds)
        have = pr is not None and len(pr[2]) == len(screen_seeds)
        gain = pr[0] if pr else None
        dec[f"{ds}|{tok}"] = dict(run_third_seed=bool(have and gain >= PROMISE), screening_seeds=list(pr[2]) if pr else [],
                                  gain_f1_50=gain, per_seed=pr[1] if pr else [], threshold=PROMISE,
                                  reason=("promising" if have and gain >= PROMISE else
                                          "below threshold" if have else "screening seeds incomplete"))
    return dec


# -- per-class diagnostic (uses labels; analysis only) -----------------------------------------------------------------

def _classes(ds: str) -> dict[str, int]:
    out = {}
    for line in (ROOT / "data" / ds / "mapping" / "mapping.txt").read_text().split("\n"):
        if line.strip():
            out[line.split(maxsplit=1)[1].strip()] = len(out)
    return out


def _load_preds(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k.replace(".npy", "").replace("__frame_pred", ""): z[k] for k in z.files}


def class_scores(ds: str, preds: dict[str, np.ndarray]) -> dict:
    """Per-class frame recall and precision after the dataset-level (global) Hungarian mapping, as the evaluator."""
    from scipy.optimize import linear_sum_assignment
    cls = _classes(ds)
    P, G = [], []
    for name in sorted(preds):
        g = [ln.strip() for ln in (ROOT / "data" / ds / "groundTruth" / f"{name}.txt").read_text().split("\n") if ln.strip()]
        if len(g) != len(preds[name]):
            raise ValueError(f"{ds}/{name}: {len(preds[name])} predicted frames, {len(g)} ground-truth frames")
        P.append(np.asarray(preds[name], int))
        G.append(np.array([cls[x] for x in g]))
    p, g = np.concatenate(P), np.concatenate(G)
    C = np.zeros((p.max() + 1, len(cls)))
    np.add.at(C, (p, g), 1)
    r, c = linear_sum_assignment(-C)
    m = np.full(p.max() + 1, -1)
    m[r] = c
    q = m[p]
    out = {"MoF": 100.0 * float((q == g).mean())}
    for n, i in cls.items():
        tp = float(((q == i) & (g == i)).sum())
        out[n] = (100 * tp / max(1, (g == i).sum()), 100 * tp / max(1, (q == i).sum()), 100.0 * float((g == i).mean()))
    return out


def class_rows(out: Path, source_root: Path, cells: dict) -> list[list]:
    rows = []
    for ds in DATASETS:
        for tok in TOKENS:
            seeds = sorted(cells.get((ds, tok), {}), key=SEED_ORDER.index)
            ref, mine = [], []
            for s in seeds:
                phase, cid = t2_cell_id(ds, s)
                a = _load_preds(Path(source_root) / "phase7" / "s0" / f"{phase}__{cid}__context" / "predictions.npz")
                b = _load_preds(Path(out) / "units" / f"{ds}__{tok}__seed{s}__context" / "predictions.npz")
                if a is None or b is None:
                    continue
                ref.append(class_scores(ds, a))
                mine.append(class_scores(ds, b))
            if not mine:
                continue
            for n in [k for k in mine[0] if k != "MoF"]:
                r_t2 = statistics.mean(x[n][0] for x in ref)
                r_t5 = statistics.mean(x[n][0] for x in mine)
                p_t2 = statistics.mean(x[n][1] for x in ref)
                p_t5 = statistics.mean(x[n][1] for x in mine)
                rows.append([ds, TOKEN_LABEL[tok], n + (" *" if n in CLASS_FOCUS[ds] else ""), f"{mine[0][n][2]:.1f}", len(mine),
                             f"{r_t2:.1f}", f"{r_t5:.1f}", f"{r_t5 - r_t2:+.1f}", f"{p_t2:.1f}", f"{p_t5:.1f}"])
    return rows


# -- the report --------------------------------------------------------------------------------------------------------

def fmt_d(p) -> str:
    return f"{p[0]:+.2f} ({' / '.join(f'{x:+.1f}' for x in p[1])})"


def label_for(pr, n_seeds: int) -> str:
    if pr is None:
        return "no cell"
    if n_seeds >= 3:
        return "**passes the bar**" if pr[0] >= BAR and all(x > 0 for x in pr[1]) else "does not pass the bar"
    return f"screening only ({n_seeds} seed{'s' if n_seeds != 1 else ''})"


def write_report(out: Path, source_root: Path) -> Path:
    out = Path(out)
    m = json.loads((out / "manifest.json").read_text())
    L = [f"# Time-scale inspection, phase 8a: gravity- and heading-aware relational tokens — {m['run_id']}", "",
         f"Status `{m.get('status')}`. Rules declared in PHASE8A_GRAVITY_HEADING_TOKENS.md before the run. Primary metric "
         f"F1@50 at the last rung, paired with T2 (span 2, same seeds) from ts5_v1. The bar (mean gain >= {BAR:g}, every "
         "seed positive) applies only to cells with three seeds; two-seed cells are screening.", ""]
    g = json.loads((out / "gates.json").read_text()) if (out / "gates.json").exists() else {}
    if g:
        L += ["## Gates", "", R4.table(["gate", "status", "detail"], [
            [k, v.get("status"), str(v.get("checks", v.get("reason", ""))) + (f" {v['detail']}" if "detail" in v else "")]
            for k, v in sorted(g.items())]), ""]
    if m.get("not_run"):
        L += ["## Blocked", ""] + [f"- `{k}`: {v}" for k, v in sorted(m["not_run"].items())] + [""]

    cells, t2 = t5_cells(out), t2_cells(source_root)
    rows, allm = [], []
    for ds in DATASETS:
        for tok in TOKENS:
            v = cells.get((ds, tok))
            if not v:
                continue
            seeds = sorted(v, key=SEED_ORDER.index)
            pr = paired(v, t2[ds])
            f = [v[s]["F1_50"] for s in seeds]
            n = len(pr[2]) if pr else 0
            rows.append([ds, TOKEN_LABEL[tok], " / ".join(f"{s}:{v[s]['F1_50']:.1f}" for s in seeds),
                         f"{statistics.mean(f):.2f}", " / ".join(f"{s}:{t2[ds][s]['F1_50']:.1f}" for s in pr[2]) if pr else "—",
                         f"{statistics.mean(t2[ds][s]['F1_50'] for s in pr[2]):.2f}" if pr else "—",
                         fmt_d(pr) if pr else "—", n, label_for(pr, n),
                         f"{statistics.mean(v[s]['pred_segs'] for s in seeds):.1f}", f"{statistics.mean(v[s]['gt_segs'] for s in seeds):.1f}"])
            for met in METRICS:
                pm = paired(v, t2[ds], met)
                if pm:
                    ok = "passes" if (pm[0] >= BAR and all(x > 0 for x in pm[1]) and len(pm[2]) >= 3) else \
                        ("screening" if len(pm[2]) < 3 else "no")
                    allm.append([ds, TOKEN_LABEL[tok], LABEL[met], fmt_d(pm), len(pm[2]), ok])
    L += ["## T5 vs T2 (span 2, `context`, global Hungarian, last rung)", "",
          R4.table(["dataset", "tokenizer", "T5 F1@50 per seed", "T5 mean", "T2 per seed (paired)", "T2 mean (paired)",
                    "Δ F1@50 (per seed)", "n", "reading", "pred segs/rec", "gt segs/rec"], rows)]

    L += ["### Q1: is T5gh better than T2?", ""]
    for ds in DATASETS:
        v = cells.get((ds, "t5_both"))
        pr = paired(v, t2[ds]) if v else None
        if pr is None:
            L.append(f"- **{ds}**: no T5gh cell")
            continue
        n = len(pr[2])
        if ds == "babel2":
            L.append(f"- **babel2** (primary; stand_up + kick = 19% of segments): {fmt_d(pr)} over {n} seeds -> {label_for(pr, n)}")
        else:
            mof = paired(v, t2[ds], "MoF")
            loss = pr[0] <= -BAR or (mof is not None and mof[0] <= -BAR)
            L.append(f"- **{ds}** (non-inferiority; no gain predicted): F1@50 {fmt_d(pr)}, MoF {fmt_d(mof) if mof else 'n/a'} over {n} seeds "
                     f"-> {'**LOSS >= 3**' if loss else 'no loss >= 3 points'}")
    L += [""]

    dec = json.loads((out / "third_seed.json").read_text())["decisions"] if (out / "third_seed.json").exists() else None
    if dec is not None:
        L += ["## Third-seed rule", "",
              f"A cell gets the third seed only if its two-seed mean F1@50 is >= {PROMISE:g} above T2's on the same seeds. "
              "Decided from the saved screening units before any third-seed job started.", "",
              R4.table(["dataset / tokenizer", "screening Δ F1@50", "per seed", "decision", "reason"], [
                  [k.replace("|", " / "), f"{d['gain_f1_50']:+.2f}" if d["gain_f1_50"] is not None else "n/a",
                   " / ".join(f"{x:+.1f}" for x in d["per_seed"]), "third seed" if d["run_third_seed"] else "no third seed", d["reason"]]
                  for k, d in sorted(dec.items())]), ""]

    L += ["## Every metric (paired with T2 on the common seeds)", "", R4.table(["dataset", "tokenizer", "metric", "Δ (per seed)", "n", "bar"], allm)]

    fact = []
    for ds in ("babel2", "babel3"):
        common = [s for s in SEED_ORDER if all(s in cells.get((ds, t), {}) for t in TOKENS) and s in t2[ds]]
        if len(common) < 2:
            continue
        d = {t: [cells[(ds, t)][s]["F1_50"] - t2[ds][s]["F1_50"] for s in common] for t in TOKENS}
        inter = [a - b - c for a, b, c in zip(d["t5_both"], d["t5_gravity"], d["t5_heading"])]
        fmt = lambda xs: f"{statistics.mean(xs):+.2f} ({' / '.join(f'{x:+.1f}' for x in xs)})"
        fact.append([ds, ",".join(map(str, common)), fmt(d["t5_gravity"]), fmt(d["t5_heading"]), fmt(d["t5_both"]), fmt(inter)])
    L += ["## Q3: which channel? (F1@50 minus T2, common seeds; descriptive)", "",
          R4.table(["dataset", "seeds", "vertical only (T5g)", "heading only (T5h)", "both (T5gh)", "interaction (gh − g − h)"], fact), ""]

    L += ["## Q2: per-class frame recall and precision, T2 vs T5 (uses labels; analysis only)", "",
          "Mean over the common seeds; global Hungarian mapping as the evaluator (MoF is reproduced by the same mapping). "
          "* marks the classes the plan predicts to benefit. Small classes are noisy under a one-to-one mapping.", ""]
    try:
        cr = class_rows(out, source_root, cells)
        L += [R4.table(["dataset", "tokenizer", "class", "frame share %", "n seeds", "T2 recall", "T5 recall", "Δ recall", "T2 precision",
                        "T5 precision"], cr)]
    except Exception as exc:  # the report must still be written
        L += [f"_per-class table failed: {type(exc).__name__}: {exc}_", ""]

    drows = []
    for p in sorted((out / "prep").glob("*/*/t5_blocks.json")):
        b = json.loads(p.read_text())
        parts = p.parent.name.split("__")
        wts = {k: round(v["weight"], 3) for k, v in b["weights"].items() if k != "dist"}
        drows.append([parts[0], TOKEN_LABEL.get(parts[1], parts[1]), parts[2].replace("seed", ""),
                      " / ".join(f"{k} {v:.3f}" for k, v in b["pca_variance_share"].items()), wts or "—", b["kmeans_occupied"],
                      f"{b['pca_explained_variance_ratio']:.3f}",
                      f"{b['heading_undefined_share']:.3f}" if b.get("heading_undefined_share") is not None else "—"])
    L += ["## Tokenizer diagnostics (label-free)", "",
          R4.table(["dataset", "tokenizer", "seed", "share of PCA-64 variance by block", "block weight vs dist", "occupied codes",
                    "PCA-64 explained", "heading undefined share"], drows), ""]
    path = out / "REPORT.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
