"""Markdown report for phase 4 (input representation), written only from saved files.

Compares the velocity-family arms (this phase) against the position-only baseline:
`context` from `rkx_v2`'s own saved results, `pca`/`centre` from the Phase 1
`inputs` stage (`ts1_v1` by default) at the current radius -- neither rebuilt
here, so the comparison is against the exact numbers reported earlier.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SEEDS = (111, 222, 1538574472)
BAR = 3.0


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


def load_velocity(out: Path) -> dict:
    """(dataset, normalization, arm) -> {seed: last ladder row + segment_stats}."""
    rows: dict = {}
    for path in sorted((out / "units").glob("*/info.json")):
        try:
            info = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if info.get("status") != "done":
            continue
        last = max(info["ladder"], key=lambda r: r["rung_steps"])
        key = (info["dataset"], info["normalization"], info["arm"])
        rows.setdefault(key, {})[info["seed"]] = dict(last, segment_stats=info["segment_stats"])
    return rows


def load_context_baseline(run_id: str) -> dict:
    """(dataset, normalization) -> {seed: last 'selected' row}, from the position-only run's segmentation.csv."""
    import csv
    path = ROOT / "results" / "week5" / "raw_kmeans_cross_dataset" / run_id / "segmentation.csv"
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    out: dict = {}
    for r in rows:
        if r["role"] != "selected":
            continue
        key = (r["dataset"], r["normalization"])
        cur = out.setdefault(key, {}).get(int(r["seed"]))
        if cur is None or int(r["rung_steps"]) > int(cur["rung_steps"]):
            out[key][int(r["seed"])] = r
    return out


def load_direct_baseline(run_id: str) -> dict:
    """(dataset, normalization, arm in {pca, centre}) -> {seed: last ladder row}, from Phase 1's `inputs` stage
    at the current radius (`ts1_v1` by default)."""
    out: dict = {}
    base = ROOT / "results" / "week5" / "timescale_inspection" / run_id / "phase1" / "units"
    if not base.exists():
        return out
    for arm in ("pca", "centre"):
        for path in sorted(base.glob(f"*__current__beta0.3__in-{arm}/info.json")):
            try:
                info = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if info.get("status") != "done":
                continue
            last = max(info["ladder"], key=lambda r: r["rung_steps"])
            key = (info["dataset"], info["normalization"], arm)
            out.setdefault(key, {})[info["seed"]] = last
    return out


def f1s(d: dict | None) -> dict[int, float]:
    if not d:
        return {}
    return {s: float(v["F1_50"] if "F1_50" in v else v["F1@50"]) for s, v in d.items()}


def mean_sd(v: dict[int, float]) -> str:
    return f"{statistics.mean(v.values()):.2f} ± {statistics.stdev(v.values()) if len(v) > 1 else 0:.2f}" if v else "n/a"


def delta(v: dict[int, float], base: dict[int, float]) -> tuple[str, bool | None]:
    seeds = sorted(set(v) & set(base))
    if len(seeds) < 3:
        return "n/a", None
    d = [v[s] - base[s] for s in seeds]
    return f"{statistics.mean(d):+.2f} ({' / '.join(f'{x:+.1f}' for x in d)})", all(x > 0 for x in d)


def probe_means(path: Path) -> dict:
    """(dataset, normalization, probe, representation) -> (mean balanced accuracy, seeds)."""
    import csv
    if not path.exists() or path.stat().st_size == 0:
        return {}
    acc: dict = {}
    for r in csv.DictReader(path.open(newline="", encoding="utf-8")):
        acc.setdefault((r["dataset"], r["normalization"], r["probe"], r["representation"]), []).append(
            float(r["balanced_accuracy"]))
    return {k: (statistics.mean(v), len(v)) for k, v in acc.items()}


def probe_section(out: Path, probe_run: str, datasets) -> tuple[list[str], dict]:
    vel = probe_means(out / "probes.csv")
    pos = probe_means(ROOT / "results" / "week5" / "timescale_inspection" / probe_run / "phase0" / "probes.csv")
    if not vel:
        return [], {}
    rows, gains = [], {}
    for ds in datasets:
        for probe in ("linear", "nn"):
            for rep in ("pca", "centre", "onehot", "context"):
                a, b = pos.get((ds, "global", probe, rep)), vel.get((ds, "global", probe, rep))
                if a is None and b is None:
                    continue
                d = (b[0] - a[0]) if a and b else None
                rows.append([ds, probe, rep, fmt(a[0] if a else None, 3), fmt(b[0] if b else None, 3),
                             f"{d:+.3f}" if d is not None else "n/a", f"{a[1] if a else 0}/{b[1] if b else 0}"])
                if rep == "pca" and probe == "linear" and d is not None:
                    gains[ds] = d
    L = ["## Per-token action information (Phase 0.4 probes, balanced accuracy, mean over seeds)", "",
         f"Position-only numbers from `{probe_run}`'s Phase 0 `probes.csv`; +velocity from this run. Same grouped split, "
         "caps and classifiers. The H4 rule uses the linear probe on the PCA feature.", "",
         table(["dataset", "probe", "representation", "position", "+velocity", "Δ", "seeds (pos/vel)"], rows)]
    return L, gains


def write_report(out: Path, context_run: str = "rkx_v2", direct_run: str = "ts1_v1",
                 datasets: tuple[str, ...] = ("babel2", "babel3", "lara"), probe_run: str = "ts0_v1") -> Path:
    out = Path(out)
    manifest = json.loads(out.joinpath("manifest.json").read_text())
    velocity = load_velocity(out)
    ctx_base = load_context_baseline(context_run)
    direct_base = load_direct_baseline(direct_run)
    datasets = [d for d in datasets if (d, "global") in ctx_base or any(k[0] == d for k in velocity)]

    L = [f"# Time-scale inspection, phase 4: input representation — {manifest['run_id']}", "",
         f"Status `{manifest.get('status')}`. Radius: `{manifest['identity']['radius']}`. Positions-plus-velocity "
         f"family, built fresh per cell; the position-only baseline is **not rebuilt** here -- `context` comes from "
         f"`{context_run}`'s saved results, `pca`/`centre` from `{direct_run}`'s Phase 1 `inputs` stage, both at the "
         "same radius. HuGaDB is not rebuilt (it is already derivative IMU data); its position-only numbers from "
         "`rkx_v2`/Phase 1 stand as the control.", ""]

    verdicts = []
    for ds in datasets:
        norm = "global"
        rows = []
        for arm, label, base in (("context", "context", f1s(ctx_base.get((ds, norm)))),
                                  ("pca_velocity", "pca", f1s(direct_base.get((ds, norm, "pca")))),
                                  ("centre_velocity", "centre", f1s(direct_base.get((ds, norm, "centre"))))):
            vel = velocity.get((ds, norm, arm), {})
            v = {s: r["F1_50"] for s, r in vel.items()}
            if not v and not base:
                continue
            d, all_pos = delta(v, base) if v and base else ("n/a", None)
            segs = statistics.mean(r["segment_stats"]["mean_pred_segments"] for r in vel.values()) if vel else None
            rows.append([label, len(base), " / ".join(fmt(base.get(s)) for s in SEEDS), mean_sd(base),
                        len(v), " / ".join(fmt(v.get(s)) for s in SEEDS), mean_sd(v), d,
                        fmt(segs, 1) if segs is not None else "n/a"])
            if len(v) == 3 and len(base) == 3:
                mdiff = statistics.mean(v.values()) - statistics.mean(base.values())
                tag = "helps" if (mdiff >= BAR and all_pos) else ("hurts" if (mdiff <= -BAR and all_pos is False and
                                                                              all(base[s] - v[s] > 0 for s in SEEDS))
                                                                   else "no clear effect")
                verdicts.append(f"{ds} / {label}: position {mean_sd(base)} -> +velocity {mean_sd(v)} "
                                f"(Δ {mdiff:+.2f}) -> **{tag}** (bar ±{BAR:g}, paired, all 3 seeds)")
        if rows:
            L += [f"## {ds}", "",
                  table(["arm", "pos seeds", "position per seed", "position mean±sd", "vel seeds",
                        "+velocity per seed", "+velocity mean±sd", "Δ (paired)", "+velocity pred segs/rec"], rows)]

    probe_lines, gains = probe_section(out, probe_run, datasets)
    L += probe_lines
    L += ["## H4 reading (bar: ≥ 3 F1@50 points on the mean, all 3 seeds in the same direction)", ""]
    L += [f"- {v}" for v in verdicts] if verdicts else ["- not enough seeds finished on both sides yet."]
    helps = {v.split(" / ")[0] for v in verdicts if "**helps**" in v}
    f1_ok = bool(helps & {"babel2", "lara"})
    probe_ok = any(gains.get(d, 0) > 0 for d in helps & {"babel2", "lara"})
    if not verdicts:
        h4 = "undetermined"
    else:
        h4 = "supported" if (f1_ok and probe_ok) else ("partly supported (F1@50 only, probe did not rise)" if f1_ok
                                                     else "not supported")
    L += ["", f"**H4 (position-only input lacks the motion signal IMU input has): {h4}.** Rule: velocity helps "
          "F1@50 on BABEL-2 or LARa by the bar in some arm, and the linear probe on the PCA feature gains action "
          "information on that dataset. BABEL-3 is reported but not part of the declared rule.", ""]
    path = out / "REPORT.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
