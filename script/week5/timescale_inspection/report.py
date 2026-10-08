"""Markdown report for phase 0, written only from the saved CSV files."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

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


def mean_over_seeds(rows: list[dict], keys: tuple, value: str) -> dict:
    acc = defaultdict(list)
    for r in rows:
        try:
            acc[tuple(r[k] for k in keys)].append(float(r[value]))
        except (TypeError, ValueError):
            pass
    return {k: (statistics.mean(v), len(v)) for k, v in acc.items()}


def by_dataset(rows):
    return sorted(rows, key=lambda r: ORDER.index(r["dataset"]) if r["dataset"] in ORDER else 99)


def write_report(out: Path) -> Path:
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    L = [f"# Time-scale inspection, phase 0 — {manifest['run_id']}", "",
         f"Source run `{manifest['source_run']}` (snapshot of {len(manifest['source_records'])} verified decodes; the "
         f"source may still be running). Implementation `{manifest['implementation_hash']}`. Everything below is "
         "label-aware diagnostics; nothing here selects a setting.", ""]

    grid = read(out / "grid_ceilings.csv")
    if grid:
        L += ["## 0.1 Token-grid ceilings (perfect token labels)", "",
              table(["dataset", "window (frames)", "token (s)", "current", "MoF", "Edit", "F1@10", "F1@25", "F1@50",
                     "identity mapping"],
                    [[r["dataset"], r["window_frames"], r["token_seconds"], r["current"], r["MoF"], r["Edit"],
                      r["F1@10"], r["F1@25"], r["F1@50"], r["mapping_is_identity"]] for r in by_dataset(grid)])]

    smooth = read(out / "smoothing_ceilings.csv")
    if smooth:
        L += ["## 0.2 Smoothing ceilings (mode filter on perfect token labels, current grid)", "",
              "`fixed` radii are in tokens (seconds shown); `relative` is floor(fraction × tokens in the recording), "
              "`relative_floor1` floors it at 1 token. `current` marks the radius the OT kernel uses today.", "",
              table(["dataset", "rule", "value", "radius (s)", "current", "median radius (tokens)", "radius 0 share",
                     "MoF", "Edit", "F1@50"],
                    [[r["dataset"], r["rule"], r["value"], r["radius_seconds"], r.get("current", ""),
                      fmt(r.get("median_radius_tokens"), 1), fmt(r.get("share_radius_zero")), r["MoF"], r["Edit"],
                      r["F1@50"]] for r in by_dataset(smooth)])]

    summ = read(out / "profile_summary.csv")
    if summ:
        rows = []
        for r in by_dataset(summ):
            rows.append([r["dataset"], r["normalization"], r["seed"], r["role"], r["F1@50"], r["saved_last_rung_F1_50"],
                         fmt(r["gt_segments_per_rec"], 1), fmt(r["pred_segments_per_rec"], 1), fmt(r["gt_median_s"]),
                         fmt(r["pred_median_s"]), fmt(r["gt_share_under_1s"]), fmt(r["pred_share_under_1s"]),
                         fmt(r.get("share_errors_involving_none")), r.get("no_none_F1@50", "")])
        L += ["## 0.3 Failure profile of the saved predictions", "",
              "`F1@50` is recomputed from saved predictions and must equal the saved last-rung value. "
              "`no none` drops ground-truth `none` frames from both sequences before scoring (this splices "
              "segments, so read it as indicative).", "",
              table(["dataset", "norm", "seed", "checkpoint", "F1@50", "saved", "GT segs/rec", "pred segs/rec",
                     "GT median s", "pred median s", "GT <1 s", "pred <1 s", "errors involving none",
                     "F1@50 no none"], rows)]
        et = read(out / "profile_error_types.csv")
        sel = [r for r in et if r["role"] == "selected"]
        if sel:
            agg = defaultdict(lambda: defaultdict(list))
            for r in sel:
                for k in ("segments", "hit", "boundary", "merged", "wrong_label"):
                    agg[(r["dataset"], r["normalization"], r["length_bin"])][k].append(float(r[k]))
            rows = [[d, n, b, fmt(statistics.mean(v["segments"]), 0)] +
                    [fmt(statistics.mean(v[k])) for k in ("hit", "boundary", "merged", "wrong_label")]
                    for (d, n, b), v in sorted(agg.items(), key=lambda kv: (ORDER.index(kv[0][0]), kv[0][1], kv[0][2]))]
            L += ["### Ground-truth segments by outcome (selected checkpoint, mean over seeds)", "",
                  "`hit`: a same-label predicted segment with IoU ≥ 0.5. Otherwise, by the majority predicted label "
                  "over the segment: `boundary` (own label, wrong extent), `merged` (a neighbour's label), "
                  "`wrong_label` (any other).", "",
                  table(["dataset", "norm", "GT length", "segments", "hit", "boundary", "merged", "wrong label"], rows)]
        pc = read(out / "profile_per_class.csv")
        sel = [r for r in pc if r["role"] == "selected"]
        if sel:
            keys = ("dataset", "normalization", "label", "name")
            rec = mean_over_seeds(sel, keys, "frame_recall")
            prec = mean_over_seeds(sel, keys, "frame_precision")
            hit = mean_over_seeds(sel, keys, "segment_hit_rate")
            share = mean_over_seeds(sel, keys, "frame_share")
            med = mean_over_seeds(sel, keys, "median_segment_s")
            rows = [[k[0], k[1], k[3], fmt(share[k][0]), fmt(med[k][0]), fmt(rec[k][0]), fmt(prec[k][0]), fmt(hit[k][0])]
                    for k in sorted(rec, key=lambda k: (ORDER.index(k[0]), k[1], int(k[2])))]
            L += ["### Per class (selected checkpoint, mean over seeds)", "",
                  table(["dataset", "norm", "class", "frame share", "median GT segment s", "frame recall",
                         "frame precision", "segment hit rate"], rows)]
        tt = read(out / "profile_tertiles.csv")
        sel = [r for r in tt if r["role"] == "selected"]
        if sel:
            f1 = mean_over_seeds(sel, ("dataset", "normalization", "tertile"), "F1@50")
            ml = mean_over_seeds(sel, ("dataset", "normalization", "tertile"), "median_length_s")
            rows = [[k[0], k[1], k[2], fmt(ml[k][0], 1), fmt(f1[k][0])]
                    for k in sorted(f1, key=lambda k: (ORDER.index(k[0]), k[1], k[2]))]
            L += ["### By recording length tertile (selected checkpoint, mean over seeds)", "",
                  table(["dataset", "norm", "tertile", "median length s", "F1@50"], rows)]

    pr = read(out / "probes.csv")
    if pr:
        keys = ("dataset", "normalization", "probe", "representation")
        ba = mean_over_seeds(pr, keys, "balanced_accuracy")
        reps = ("pca", "centre", "onehot", "context")
        groups = sorted({k[:3] for k in ba}, key=lambda k: (ORDER.index(k[0]), k[1], k[2]))
        rows = [[*g, *[fmt(ba.get((*g, rep), (float("nan"),))[0], 3) for rep in reps],
                 ba.get((*g, "context"), (0, 0))[1]] for g in groups]
        L += ["## 0.4 Action information per token (balanced accuracy, grouped split, mean over seeds)", "",
              "`pca`: the patch's PCA-64 feature; `centre`: its K-means centre; `onehot`: the code (the `nn` row is "
              "the per-code majority label); `context`: the selected checkpoint's contextual embedding.", "",
              table(["dataset", "norm", "probe", *reps, "seeds"], rows)]

    ms = read(out / "masked_loss_split.csv")
    if ms:
        rows = [[r["dataset"], r["normalization"], r["seed"], r["selected_step"], fmt(r["share_targets_crossing"]),
                 fmt(r["within_loss"], 3), fmt(r["crossing_loss"], 3), fmt(r["within_top1"], 3),
                 fmt(r["crossing_top1"], 3), fmt(r["val_loss_all"], 4), fmt(r["saved_best_val_loss"], 4)]
                for r in by_dataset(ms)]
        L += ["## 0.4 Masked-prediction loss by boundary crossing (validation masks, selected checkpoint)", "",
              "A target is `crossing` when the contiguous masked run containing it spans a ground-truth label change. "
              "`all` must match the saved validation loss of the selected step.", "",
              table(["dataset", "norm", "seed", "step", "share crossing", "loss within", "loss crossing", "top-1 within",
                     "top-1 crossing", "loss all", "saved"], rows)]

    path = out / "REPORT.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
