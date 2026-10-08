"""REPORT.md and scores.csv for a run: cells, references, comparisons R1-R5, G5 and the verdict (computed, not hand-written)."""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

from script.week5.masq_experiments import config as C

M = C.METRICS
CAVEATS = (
    "All results here are *development* results (protocol P7): every dataset informed earlier decisions and no untouched "
    "evaluation set exists. The published MASQ numbers are single runs from a paper that is not peer reviewed; our M arms "
    "are a reimplementation from the paper text with the SMQ backbone and defaults, so a gap to the published numbers may "
    "come from undocumented details. These experiments cannot show *why* MASQ works, only whether its training changes "
    "reproduce and which component matters; they say nothing about our OT decoder on MASQ latents (Tier 1).")


# -- pure logic (unit-tested) ---------------------------------------------------------------------------------------------

def reading(diffs: list[float] | None) -> str:
    """Paired differences over seeds -> the declared reading (bar: mean >= 3, every seed in the same direction)."""
    if not diffs:
        return "no cells"
    if len(diffs) < 3:
        return f"incomplete ({len(diffs)} seed{'s' if len(diffs) != 1 else ''})"
    mean = st.mean(diffs)
    if mean >= C.BAR and all(d > 0 for d in diffs):
        return "passes the bar"
    if mean <= -C.BAR and all(d < 0 for d in diffs):
        return "reverse passes the bar"
    return "within the bar: comparable performance, not equivalence"


def reading_kind(text: str) -> str:
    if text.startswith("passes"):
        return "pass"
    if text.startswith("reverse"):
        return "reverse"
    if text.startswith("within"):
        return "within"
    return "incomplete"


def reproduced(mean_by_metric: dict, published: dict) -> bool:
    """R2: the 3-seed mean within 3 points of the published value on both MoF and F1@50."""
    return all(abs(mean_by_metric[m] - published[m]) <= C.BAR for m in ("MoF", "F1@50"))


def verdict(r1_mof: dict[str, str], r2: dict[str, bool]) -> tuple[str, str]:
    """r1_mof: dataset -> reading kind of M-pub minus S0 on MoF; r2: dataset -> reproduced. Plan section B.7.

    replicated = R1 passes on MoF in >= 3 of 5 datasets including LARa (and R2 holds somewhere); partly = 1-2 passes, or >= 3
    passes without LARa, or the replicated pattern with R2 failing everywhere; not replicated = no pass.
    """
    passes = [d for d, k in r1_mof.items() if k == "pass"]
    detail = f"R1 passes on MoF in {len(passes)} of {len(r1_mof)} datasets ({', '.join(passes) or 'none'}); R2 reproduced on {sum(r2.values())}"
    if not passes:
        return "not replicated", detail
    if len(passes) >= 3 and "lara" in passes and any(r2.values()):
        return "replicated", detail
    return "partly replicated", detail


def g5(s0_by_dataset: dict[str, dict[str, float]]) -> dict[str, dict]:
    """S0 means on HuGaDB and LARa within 2 SD of the earlier retrained spread (MoF and F1@50)."""
    out = {}
    for ds, spread in C.SMQ_RETRAINED_SPREAD.items():
        if ds not in s0_by_dataset:
            continue
        checks = {m: abs(s0_by_dataset[ds][m] - mu) <= 2 * sd for m, (mu, sd) in spread.items()}
        out[ds] = dict(passed=all(checks.values()), checks=checks)
    return out


# -- loading --------------------------------------------------------------------------------------------------------------

def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def load_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    manifest = _load(run_dir / "manifest.json") or {}
    run_id = manifest.get("run_id", run_dir.name)
    cells = {}
    for p in sorted((run_dir / "units").glob("*/scores.json")):
        unit = C.parse_unit(p.parent.name)
        s = json.loads(p.read_text())
        cells[(unit.arm, unit.dataset, unit.seed)] = dict(
            scores=s, diag=_load(p.parent / "diagnostics.json"),
            train=_load(C.model_dir(run_id, unit) / "training_info.json"))
    return dict(manifest=manifest, cells=cells, run_id=run_id)


def _vals(cells, arm, ds, key="global", metric="F1@50"):
    return {seed: c["scores"][key][metric] for (a, d, seed), c in cells.items() if a == arm and d == ds}


def _paired(cells, a, b, ds, metric, key="global"):
    va, vb = _vals(cells, a, ds, key, metric), _vals(cells, b, ds, key, metric)
    seeds = [s for s in C.SEEDS if s in va and s in vb]
    return [va[s] - vb[s] for s in seeds]


def _fmt(diffs) -> str:
    return "—" if not diffs else f"{st.mean(diffs):+.2f} ({' / '.join(f'{d:+.1f}' for d in diffs)})"


def table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _mean_sd(xs):
    xs = list(xs)
    if not xs:
        return "—"
    return f"{st.mean(xs):.2f}" + (f" ± {st.pstdev(xs):.2f}" if len(xs) > 1 else "")


def e0_reference(ds: str) -> dict | None:
    p = C.ROOT / "results" / "e0" / f"{ds}_pretrained.json"
    return json.loads(p.read_text())["global"] if p.exists() else None


# -- report ---------------------------------------------------------------------------------------------------------------

def write_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    run = load_run(run_dir)
    man, cells, rid = run["manifest"], run["cells"], run["run_id"]
    smoke = bool(man.get("smoke"))
    arms_present = [a for a in C.ARMS if any(k[0] == a for k in cells)]
    datasets = [d for d in C.DATASET_ORDER if any(k[1] == d for k in cells)]
    L = [f"# MASQ replication (Tier 0): {rid}", ""]
    if smoke:
        L += ["**SMOKE RUN: 2 epochs on 3 batches per epoch. These numbers are plumbing checks, not results.**", ""]
    L += [f"Status `{man.get('status')}`. Training code `{man.get('train_hash')}`, scoring code `{man.get('score_hash')}`; analysis hash at launch `{man.get('analysis_hash_at_launch')}`"
          f" (now `{C.analysis_hash()}`" + (")" if man.get("analysis_hash_at_launch") == C.analysis_hash() else ", CHANGED since launch)") + ".",
          "Checkpoint: final epoch (30 for results; 2 for smoke), global Hungarian mapping (primary, as in the published tables), MoF and F1@50 co-primary. "
          f"Bar: mean difference >= {C.BAR:g} with every seed in the same direction, read only with three seeds.", ""]
    gates = _load(run_dir / "gates.json")
    if gates:
        L += ["## Gates", "", table(["gate", "status"], [[k, v.get("status")] for k, v in gates.items() if isinstance(v, dict) and "status" in v])]
    if man.get("not_run"):
        L += ["## Not run", ""] + [f"- `{k}`: {v}" for k, v in sorted(man["not_run"].items())] + [""]

    # cells
    L += ["## Cells (global Hungarian; seed mean ± SD, SD = population SD over seeds)", ""]
    csv_rows = []
    for ds in datasets:
        rows = []
        for arm in arms_present:
            v = {m: _vals(cells, arm, ds, "global", m) for m in M}
            if not v["MoF"]:
                continue
            seeds = [s for s in C.SEEDS if s in v["MoF"]]
            ex = [c for (a, d, s), c in cells.items() if a == arm and d == ds]
            segs = _mean_sd(c["scores"]["segments"]["mean_pred"] for c in ex)
            occ = _mean_sd(c["scores"]["codes"]["occupied"] for c in ex)
            rows.append([arm, len(seeds)] + [f"{_mean_sd(v[m].values())}" for m in M] + [segs, f"{_mean_sd(c['scores']['segments']['mean_gt'] for c in ex)}", occ])
            for s in seeds:
                csv_rows.append([rid, arm, ds, s] + [round(v[m][s], 4) for m in M]
                                + [round(cells[(arm, ds, s)]["scores"]["local"][m], 4) for m in M])
        L += [f"### {C.DATASET_NAMES[ds]}", "", table(["arm", "n seeds"] + list(M) + ["pred segs/rec", "gt segs/rec", "occupied codes"], rows)]
        ref = [["MASQ published (single run)"] + [C.MASQ_PUBLISHED[ds][m] for m in M]]
        e0 = e0_reference(ds)
        if e0:
            ref.append(["SMQ released checkpoint (re-scored, e0)"] + [round(e0[m], 2) for m in M])
        if ds in C.SMQ_RETRAINED_SPREAD:
            sp = C.SMQ_RETRAINED_SPREAD[ds]
            ref.append(["SMQ retrained, earlier 3 seeds (mean ± SD)"] + [f"{sp[m][0]} ± {sp[m][1]}" if m in sp else "—" for m in M])
        L += [table(["reference"] + list(M), ref)]
    L += ["### Per-sequence (local) Hungarian, reported separately and never mixed with the global scores", ""]
    for ds in datasets:
        rows = [[arm] + [_mean_sd(_vals(cells, arm, ds, "local", m).values()) for m in M] for arm in arms_present if _vals(cells, arm, ds, "local", "MoF")]
        L += [f"**{C.DATASET_NAMES[ds]}**", "", table(["arm"] + list(M), rows)]

    # comparisons
    def block(cid, text, a, b):
        out = [f"### {cid}: {text}", ""]
        kinds = {}
        for ds in datasets:
            parts = []
            for m in ("MoF", "F1@50"):
                d = _paired(cells, a, b, ds, m)
                rd = reading(d)
                parts.append(f"{m} {_fmt(d)} -> {rd}")
                if m == "MoF":
                    kinds[ds] = reading_kind(rd)
            out.append(f"- **{ds}**: " + "; ".join(parts))
        return out + [""], kinds

    L += ["## Comparisons (A minus B, paired by seed; MoF and F1@50 co-primary)", ""]
    s0_means = {ds: {m: st.mean(_vals(cells, "S0", ds, "global", m).values()) for m in ("MoF", "F1@50")}
                for ds in datasets if len(_vals(cells, "S0", ds)) == 3}
    g5res = g5(s0_means)
    flag = {ds: ("" if r["passed"] else " [control outside the expected spread: read with caution]") for ds, r in g5res.items()}
    r1, kinds = block("R1", "M-pub minus S0: does MASQ's training change improve SMQ?", "M-pub", "S0")
    L += r1
    for ds, f in flag.items():
        if f:
            L.append(f"- G5 flag on {ds}:{f}")
    r2 = {}
    L += ["### R2: M-pub mean vs published MASQ (reproduced if within 3 points on MoF and F1@50)", ""]
    for ds in datasets:
        mp = {m: _vals(cells, "M-pub", ds, "global", m) for m in ("MoF", "F1@50")}
        if len(mp["MoF"]) == 3:
            means = {m: st.mean(mp[m].values()) for m in mp}
            r2[ds] = reproduced(means, C.MASQ_PUBLISHED[ds])
            L.append(f"- **{ds}**: ours MoF {means['MoF']:.1f} / F1@50 {means['F1@50']:.1f}; published {C.MASQ_PUBLISHED[ds]['MoF']} / "
                     f"{C.MASQ_PUBLISHED[ds]['F1@50']} -> {'reproduced' if r2[ds] else 'not reproduced'}")
        else:
            L.append(f"- **{ds}**: incomplete")
    L += ["", "### R3: S0 (our retrained SMQ) vs the references", ""]
    for ds in datasets:
        if _vals(cells, "S0", ds):
            e0 = e0_reference(ds)
            row = f"- **{ds}**: S0 MoF {_mean_sd(_vals(cells, 'S0', ds, 'global', 'MoF').values())}, F1@50 {_mean_sd(_vals(cells, 'S0', ds, 'global', 'F1@50').values())}"
            if e0:
                row += f"; released SMQ {e0['MoF']:.1f} / {e0['F1@50']:.1f}"
            if ds in g5res:
                row += f"; G5 {'passed' if g5res[ds]['passed'] else 'FAILED'} {g5res[ds]['checks']}"
            L.append(row)
    L.append("")
    if any(k[0] in ("M-JL", "M-fix") for k in cells):
        for cid, text, a, b in (("R4a", "M-pub minus M-JL: the velocity loss's contribution", "M-pub", "M-JL"),
                                ("R4b", "M-fix minus M-pub: does the published per-dataset p matter?", "M-fix", "M-pub")):
            blk, _ = block(cid, text, a, b)
            L += blk
    else:
        L += ["### R4: not run (stage 2)", ""]

    # every metric
    allm = []
    for cid, a, b in (("R1", "M-pub", "S0"), ("R4a", "M-pub", "M-JL"), ("R4b", "M-fix", "M-pub")):
        for ds in datasets:
            for m in M:
                d = _paired(cells, a, b, ds, m)
                if d:
                    allm.append([cid, ds, f"{a} - {b}", m, _fmt(d), len(d), reading(d)])
    L += ["## Every metric", "", table(["id", "dataset", "A − B", "metric", "Δ (per seed)", "n", "reading"], allm)]

    # identity (R5)
    L += ["## R5: identity diagnostics (descriptive; subject only for HuGaDB and LARa)", "",
          "Code AMI = chance-corrected mutual information of native codes with the metadata. Latent NN share = same-recording share "
          "of nearest neighbours of patch-mean latents (adjacent patches excluded). The per-class block uses labels (analysis only).", ""]
    rows = []
    for ds in datasets:
        for arm in arms_present:
            ex = [c["diag"] for (a, d, s), c in cells.items() if a == arm and d == ds and c["diag"]]
            if not ex:
                continue
            fm = lambda k: _mean_sd(e[k] for e in ex if e.get(k) is not None)  # noqa: E731
            pc = _mean_sd(e["per_class"]["frame_weighted_mean"] for e in ex if e.get("per_class") and e["per_class"]["frame_weighted_mean"] is not None)
            rows.append([ds, arm, len(ex), fm("recording_ami"), fm("subject_ami"), pc, fm("latent_same_recording_nn_share")])
    L += [table(["dataset", "arm", "n", "recording AMI", "subject AMI", "per-class subject AMI (labels)", "latent NN share"], rows)]
    ident = []
    for ds in datasets:
        for m in ("recording_ami", "subject_ami", "latent_same_recording_nn_share"):
            a = {s: c["diag"][m] for (arm, d, s), c in cells.items() if arm == "M-pub" and d == ds and c["diag"] and c["diag"].get(m) is not None}
            b = {s: c["diag"][m] for (arm, d, s), c in cells.items() if arm == "S0" and d == ds and c["diag"] and c["diag"].get(m) is not None}
            ss = [s for s in C.SEEDS if s in a and s in b]
            if ss:
                ident.append([ds, m, _fmt([a[s] - b[s] for s in ss]), len(ss)])
    L += ["M-pub minus S0 (per seed):", "", table(["dataset", "measure", "Δ", "n"], ident)]

    # training
    rows = []
    for (arm, ds, seed), c in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        t = c["train"]
        if t:
            last = t["per_epoch"][-1]
            rows.append([ds, arm, seed, f"{last['rec']:.5f}", f"{last['commit']:.5f}", f"{last['vel']:.5f}", t["dead_code_replacements_total"],
                         t["ema_cluster_size_nonzero"], f"{t['seconds']:.0f}"])
    L += ["## Training (final epoch losses)", "", table(["dataset", "arm", "seed", "rec", "commit", "vel", "dead-code replacements", "EMA codes in use", "seconds"], rows)]

    # verdict
    L += ["## Verdict (computed from the declared rule)", ""]
    if smoke:
        L += ["Not computed for a smoke run.", ""]
    elif len(r2) == len(C.DATASETS) and len(kinds) == len(C.DATASETS) and all(k != "incomplete" for k in kinds.values()):
        v, detail = verdict(kinds, r2)
        L += [f"**{v}.** {detail}.", "",
              "Rule: replicated = R1 passes on MoF in >= 3 of 5 datasets including LARa and R2 holds somewhere; partly = 1-2 passes, "
              "or >= 3 passes without LARa, or the replicated pattern with R2 failing everywhere; not replicated = no pass.", ""]
    else:
        L += ["Not computed: the verdict needs S0 and M-pub with three seeds on all five datasets (Stage 1 incomplete).", ""]
    L += ["## Caveats", "", CAVEATS, ""]
    path = run_dir / "REPORT.md"
    path.write_text("\n".join(L), encoding="utf-8")
    with open(run_dir / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "arm", "dataset", "seed"] + [f"global_{m}" for m in M] + [f"local_{m}" for m in M])
        w.writerows(csv_rows)
    return path
