"""Markdown report for phase 1, written only from the saved unit records."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ORDER = ("hugadb", "babel1", "babel2", "babel3", "lara")
SEEDS = (111, 222, 1538574472)
SETTING_ORDER = ("current", "fixed_0", "fixed_1", "fixed_2", "fixed_3", "fixed_4", "rel_2", "rel_4", "rel_8",
                 "sec_0.25", "sec_0.5", "sec_1.0")
BAR = 3.0  # F1@50 points, declared in the plan


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


def load_units(out: Path) -> list[dict]:
    units = []
    for path in sorted((out / "units").glob("*/info.json")):
        try:
            info = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if info.get("status") == "done":
            last = max(info["ladder"], key=lambda r: r["rung_steps"])
            units.append(dict(info, last=last))
    return units


def index(units: list[dict]) -> dict:
    return {(u["dataset"], u["normalization"], u.get("input", "context"), u["setting"], round(u["beta"], 3),
             u["seed"]): u for u in units}


def cell(idx: dict, ds: str, setting: str, beta: float = 0.3, norm: str = "global",
         inp: str = "context") -> dict[int, dict]:
    return {s: idx[(ds, norm, inp, setting, beta, s)] for s in SEEDS if (ds, norm, inp, setting, beta, s) in idx}


def f1s(units: dict[int, dict]) -> dict[int, float]:
    return {s: float(u["last"]["F1_50"]) for s, u in units.items()}


def mean_sd(v: dict[int, float]) -> str:
    return f"{statistics.mean(v.values()):.2f} ± {statistics.stdev(v.values()) if len(v) > 1 else 0:.2f}" if v else "n/a"


def delta(v: dict[int, float], base: dict[int, float]) -> tuple[str, bool | None]:
    """Paired by seed. Returns text and whether every seed is at least BAR above the baseline mean difference."""
    seeds = sorted(set(v) & set(base))
    if len(seeds) < 3:
        return "n/a", None
    d = [v[s] - base[s] for s in seeds]
    return f"{statistics.mean(d):+.2f} ({' / '.join(f'{x:+.1f}' for x in d)})", all(x > 0 for x in d)


def settings_rows(idx: dict, ds: str, names: list[str], norm: str = "global") -> list[list]:
    base = f1s(cell(idx, ds, "current", norm=norm))
    rows = []
    for name in names:
        units = cell(idx, ds, name, norm=norm)
        if not units:
            continue
        v = f1s(units)
        d, _ = delta(v, base)
        mean = lambda key: statistics.mean(float(u["last"][key]) for u in units.values())
        segs = statistics.mean(u["segment_stats"]["mean_pred_segments"] for u in units.values())
        gt = statistics.mean(u["segment_stats"]["mean_gt_segments"] for u in units.values())
        used = statistics.mean(u["segment_stats"]["mean_pred_states_used"] for u in units.values())
        alias = "yes" if all(u.get("aliased_of") for u in units.values()) else ""
        rows.append([name, alias, len(v), mean_sd(v), " / ".join(fmt(v.get(s)) for s in SEEDS), d,
                     fmt(mean("MoF")), fmt(mean("Edit")), fmt(mean("F1_10")), fmt(segs, 1), fmt(gt, 1), fmt(used, 2),
                     sum("capped" in u["ot_status"] for u in units.values())])
    return rows


HEAD = ["setting", "= current", "seeds", "F1@50 mean ± sd", "per seed (111 / 222 / 1538574472)",
        "Δ vs current, paired (per seed)", "MoF", "Edit", "F1@10", "pred segs / rec", "GT segs / rec", "states used",
        "capped fits"]


def reading(idx: dict, selection: dict | None) -> list[str]:
    notes = []
    dev = "babel2"
    base = f1s(cell(idx, dev, "current"))
    best_name, best = None, None
    for name in SETTING_ORDER[1:]:
        v = f1s(cell(idx, dev, name))
        if len(v) == 3 and len(base) == 3:
            m = statistics.mean(v.values()) - statistics.mean(base.values())
            if best is None or m > best:
                best_name, best = name, m
    if best_name is None:
        return ["The BABEL-2 screen is not finished (three seeds of every setting are needed); no verdict."]
    v = f1s(cell(idx, dev, best_name))
    _, all_pos = delta(v, base)
    h1_dev = best >= BAR and bool(all_pos)
    notes.append(f"BABEL-2 (development): best setting `{best_name}` is {best:+.2f} F1@50 over the current radius on the "
                 f"mean; all three seeds above: {all_pos}. Bar: ≥ {BAR:g} on the mean and every seed → "
                 f"{'met' if h1_dev else 'not met'}.")
    b3 = f1s(cell(idx, "babel3", best_name))
    b3base = f1s(cell(idx, "babel3", "current"))
    if len(b3) == 3 and len(b3base) == 3:
        d3 = statistics.mean(b3.values()) - statistics.mean(b3base.values())
        notes.append(f"BABEL-3 (held out from the choice, same setting): {d3:+.2f} on the mean → "
                     f"{'same direction' if d3 > 0 else 'opposite direction'}.")
        h1 = h1_dev and d3 > 0
    else:
        h1 = None
        notes.append("BABEL-3 screen incomplete.")
    notes.append("**H1 (radius too wide for BABEL): " + {True: "supported", False: "not supported", None: "undetermined"}[h1] + ".**")
    if selection:
        chosen = selection["chosen"]
        for ds in ("hugadb", "babel1", "lara"):
            base_ds = f1s(cell(idx, ds, "current"))
            for fam in ("sec", "rel"):
                v = f1s(cell(idx, ds, chosen[fam]))
                if len(v) == 3 and len(base_ds) == 3:
                    d = statistics.mean(v.values()) - statistics.mean(base_ds.values())
                    flag = ""
                    if ds == "hugadb":
                        flag = f" → transfer check (drop ≤ {BAR:g}): {'passed' if d >= -BAR else 'FAILED'}"
                    notes.append(f"{ds}, frozen `{chosen[fam]}`: {d:+.2f} vs current{flag}")
    return notes


def inputs_section(idx: dict, selection: dict | None) -> list[str]:
    """No-contextualizer inputs (`pca`, `centre`) against the contextual embedding, same radius, paired by seed."""
    keys = sorted({(k[0], k[1]) for k in idx if k[2] != "context"},
                  key=lambda k: (ORDER.index(k[0]) if k[0] in ORDER else 99, k[1]))
    if not keys:
        return []
    radii = ["current"] + ([selection["chosen"]["sec"], selection["chosen"]["rel"]] if selection else [])
    L = ["## No-contextualizer inputs", "",
         "`pca`: the patch's PCA-64 feature, mean-subtracted and L2-normalized (the repository's `direct_continuous` arm). "
         "`centre`: the K-means centre of its code, same treatment (`direct_quantized`). `context`: the selected "
         "checkpoint's contextual embedding (the `rkx_v2` decode). The OT fit, solver settings and scorer are identical; "
         "no checkpoint is involved for `pca` and `centre`. Δ is paired by seed against `context` at the same radius "
         "(the context column at a non-current radius exists only if the screen/confirm stage decoded it).", ""]
    verdicts = []
    for ds, norm in keys:
        rows = []
        for radius in radii:
            ctx = f1s(cell(idx, ds, radius, norm=norm))
            for inp in ("context", "centre", "pca"):
                units = cell(idx, ds, radius, norm=norm, inp=inp)
                if not units:
                    continue
                v = f1s(units)
                d, _ = delta(v, ctx) if inp != "context" else ("", None)
                mean = lambda key: statistics.mean(float(u["last"][key]) for u in units.values())
                st = lambda key: statistics.mean(u["segment_stats"][key] for u in units.values())
                rows.append([radius, inp, len(v), mean_sd(v), " / ".join(fmt(v.get(s)) for s in SEEDS), d,
                             fmt(mean("MoF")), fmt(mean("Edit")), fmt(st("mean_pred_segments"), 1),
                             fmt(st("mean_gt_segments"), 1), fmt(st("mean_pred_states_used"), 2)])
        if rows:
            L += [f"### {ds}" + (f" ({norm})" if ds == "lara" else ""), "",
                  table(["radius", "input", "seeds", "F1@50 mean ± sd", "per seed (111 / 222 / 1538574472)",
                         "Δ vs context, paired", "MoF", "Edit", "pred segs / rec", "GT segs / rec", "states used"], rows)]
        c = f1s(cell(idx, ds, "current", norm=norm))
        q = f1s(cell(idx, ds, "current", norm=norm, inp="centre"))
        p = f1s(cell(idx, ds, "current", norm=norm, inp="pca"))
        if len(c) == 3 and len(q) == 3 and len(p) == 3:
            cq, c_all = delta(c, q)
            qp, q_all = delta(q, p)
            helps = statistics.mean(c.values()) - statistics.mean(q.values()) >= BAR and bool(c_all)
            costs = statistics.mean(p.values()) - statistics.mean(q.values()) >= BAR and all(
                p[s] > q[s] for s in SEEDS)
            verdicts.append(f"{ds}{' ' + norm if ds == 'lara' else ''}: context − centre = {cq} → the contextualizer "
                            f"{'adds' if helps else 'does not clearly add'} ≥ {BAR:g} points; pca − centre = "
                            f"{statistics.mean(p.values()) - statistics.mean(q.values()):+.2f} → quantization "
                            f"{'costs' if costs else 'does not clearly cost'} ≥ {BAR:g} points.")
    if verdicts:
        L += ["### Reading at the current radius (bar: ≥ 3 F1@50 points on the mean and in every seed)", ""] +              [f"- {v}" for v in verdicts] + [""]
    return L


def write_report(out: Path) -> Path:
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    units = load_units(out)
    idx = index(units)
    gates = json.loads((out / "gates.json").read_text()) if (out / "gates.json").exists() else {}
    selection = json.loads((out / "selection.json").read_text()) if (out / "selection.json").exists() else None
    L = [f"# Time-scale inspection, phase 1: OT kernel radius — {manifest['run_id']}", "",
         f"Status `{manifest.get('status')}`; {len(units)} finished units. Source run: "
         f"`{manifest['identity']['source_run']}` (validation-selected checkpoints, last final-solve rung = 4,000, "
         f"outer cap {manifest['identity']['outer_cap']}). Only the kernel radius (and, in the beta table, β) changes; "
         "a = 0.7, λ = 0.05, ε = 0.07 and 25 inner steps are frozen. **BABEL-2 is the development set**: settings "
         "are chosen on it only, and every other dataset is reported with those settings frozen.", ""]
    L += ["## Gates", "", table(["gate", "status", "cell", "max |Δ| vs source ladder", "F1@50 now / source"],
                                [[k.split(":")[-1], g.get("status"), g.get("cell", ""), g.get("max_abs_diff", ""),
                                  f"{g.get('f1_50_now', '')} / {g.get('f1_50_source', '')}"] for k, g in gates.items()]),
          "A setting whose half-width equals the current one on every recording runs the identical solve; it reuses "
          "the source run's decode (`= current` column) once these gates pass.", ""]
    for ds in ("babel2", "babel3"):
        rows = settings_rows(idx, ds, list(SETTING_ORDER))
        if rows:
            L += [f"## Screen — {ds}" + (" (development)" if ds == "babel2" else " (held out from selection)"), "",
                  table(HEAD, rows)]
    if selection:
        L += ["## Selection (from BABEL-2 only)", "", f"Rule: {selection['rule']}.", "",
              table(["family", "setting", "BABEL-2 mean F1@50"],
                    [[fam, name, fmt(m)] for fam, ms in selection["development_means"].items() for name, m in ms.items()]),
              f"Chosen: fixed-seconds `{selection['chosen']['sec']}`, relative `{selection['chosen']['rel']}`.", ""]
        for ds in ("babel1", "hugadb", "lara"):
            names = ["current", selection["chosen"]["sec"], selection["chosen"]["rel"]]
            rows = settings_rows(idx, ds, names)
            if rows:
                L += [f"## Confirm — {ds} (settings frozen)", "", table(HEAD, rows)]
        brows = []
        for name in (selection["chosen"]["sec"], selection["chosen"]["rel"]):
            for beta in (0.1, 0.3, 0.5):
                v = f1s(cell(idx, "babel2", name, beta))
                if v:
                    brows.append([name, beta, len(v), mean_sd(v), " / ".join(fmt(v.get(s)) for s in SEEDS)])
        if any(r[1] != 0.3 for r in brows):
            L += ["## β cross-check — babel2", "", table(["radius", "β", "seeds", "F1@50 mean ± sd", "per seed"], brows)]
    L += inputs_section(idx, selection)
    L += ["## Declared reading", ""] + [f"- {n}" for n in reading(idx, selection)] + [
        "", "The 3-point bar comes from the observed seed-to-seed spread (up to about 6 points on HuGaDB and BABEL-2). "
        "A cross-dataset rule additionally needs one family within 1 point of the best setting on every dataset; "
        "compare the confirm tables. Decision point 1 is yours.", ""]
    path = out / "REPORT.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
