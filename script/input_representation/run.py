"""Input-representation test driver. Modes: preflight, smoke, dry-run, run (cells in fresh child processes) and the child `cell`.

Per (variant, seed): prep (new variants only) -> b0 (matched held-out B0 decode via the Milestone 1 machinery, new variants
only; T2's reference cells are read from the Milestone 1 run) -> probe (label-using diagnostics). Then REPORT.md with the
decision rules declared in docs/plans/INPUT_REPRESENTATION_TEST_PLAN.md."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
_CAP = str(min(8, os.cpu_count() or 8))
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, _CAP)

import numpy as np  # noqa: E402

from script.input_representation import config as C  # noqa: E402

STAGES = ("prep", "b0", "probe")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    os.replace(tmp, path)


def read(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def patch_compositional(variant: str) -> None:
    """Point the Milestone 1 data/train code at this variant's preparation, without editing it."""
    from script.compositional_dynamics import config as CD
    CD.FAMILY = C.family(variant)
    CD.prep_dir = lambda seed: C.prep_dir(variant, seed)


def b0_dir(run_dir: Path, variant: str) -> Path:
    return run_dir / f"b0__{variant}"


def run_stage(run_dir: Path, variant: str, seed: int, stage: str, smoke: bool, device: str, reuse: bool) -> int:
    t0 = time.perf_counter()
    if stage == "prep":
        from script.input_representation import features
        d = features.prepare(variant, seed)
        print(f"[prep] {variant} seed {seed}: {d} ({time.perf_counter() - t0:.0f}s)", flush=True)
        return 0
    if stage == "b0":
        from script.compositional_dynamics import config as CD
        from script.compositional_dynamics import run as CR
        from script.input_representation import features
        features.install()
        patch_compositional(variant)
        scale = CD.SMOKE if smoke else CD.FULL
        return CR.run_cell(b0_dir(run_dir, variant), seed, "b0_centre", scale, "cpu", None, reuse)
    if stage == "probe":
        out = run_dir / "probes" / f"{variant}__seed{seed}.json"
        if reuse and (read(out) or {}).get("code_hash") == C.code_hash() and (read(out) or {}).get("smoke") == smoke:
            print(f"[probe] {variant} seed {seed}: reused", flush=True)
            return 0
        from script.input_representation import probes
        w = probes.load_windows(C.prep_dir(variant, seed), C.family(variant))
        res = probes.run_probes(w, C.SMOKE_PROBE_CAP if smoke else C.PROBE_CAP)
        write_json(out, dict(res, variant=variant, seed=seed, smoke=smoke, code_hash=C.code_hash(), n_windows=int(len(w["y"])),
                             seconds=time.perf_counter() - t0))
        print(f"[probe] {variant} seed {seed}: action {res['action_probe_continuous']:.1f} / code table {res['action_probe_code_table']:.1f} "
              f"/ subject {res['subject_probe_accuracy']:.1f} (chance {res['subject_probe_chance']:.1f}) ({time.perf_counter() - t0:.0f}s)", flush=True)
        return 0
    raise ValueError(stage)


# --- report -----------------------------------------------------------------------------------------------------

def b0_metrics(run_dir: Path, variant: str, seed: int):
    """(MoF, F1@50) of the matched B0 on the 44 evaluation recordings, final solver endpoint."""
    d = (C.REF_RUN / "cells" / f"b0_centre__seed{seed}") if variant == "t2" else (b0_dir(run_dir, variant) / "cells" / f"b0_centre__seed{seed}")
    sc = read(d / "scores.json")
    if not sc:
        return None
    m = sc["eval"][str(sc["primary_endpoint_steps"])]["metrics"]
    return m["MoF"], m["F1@50"]


def decide(rows: dict, smoke: bool) -> dict:
    if smoke:
        return dict(verdict="not computed (smoke run)")
    comp = {}
    for v in C.NEW_VARIANTS:
        d = [(rows[(v, s)]["MoF"] - rows[("t2", s)]["MoF"], rows[(v, s)]["F1_50"] - rows[("t2", s)]["F1_50"])
             for s in C.SEEDS if (v, s) in rows and ("t2", s) in rows and "MoF" in rows[(v, s)] and "MoF" in rows[("t2", s)]]
        if len(d) != len(C.SEEDS):
            return dict(verdict="INCONCLUSIVE", reason=f"{v}: missing seeds")
        mof, f1 = [x[0] for x in d], [x[1] for x in d]
        comp[v] = dict(mean_MoF=float(np.mean(mof)), mean_F1_50=float(np.mean(f1)), per_seed_MoF=mof, per_seed_F1_50=f1,
                       passes=bool(np.mean(mof) >= C.BAR and np.mean(f1) >= C.BAR and min(mof) > 0 and min(f1) > 0))
    winners = [v for v, c in comp.items() if c["passes"]]
    if winners:
        best = max(winners, key=lambda v: comp[v]["mean_MoF"] + comp[v]["mean_F1_50"])
        return dict(verdict="ADOPT_INPUT", adopt=best, comparisons=comp)
    gaps = {}
    for v in C.VARIANTS:
        pr = [rows.get((v, s), {}).get("action_probe_continuous") for s in C.SEEDS]
        mf = [rows.get((v, s), {}).get("MoF") for s in C.SEEDS]
        if None in pr or None in mf:
            return dict(verdict="INCONCLUSIVE", reason=f"{v}: missing probes or decodes", comparisons=comp)
        gaps[v] = float(np.mean(pr) - np.mean(mf))
    best = max(gaps, key=gaps.get)
    verdict = "ORGANISATION_PROBLEM" if gaps[best] >= C.GAP_THRESHOLD else "REPRESENTATION_LIMIT"
    return dict(verdict=verdict, probe_minus_mof=gaps, best_input=best, comparisons=comp)


def finalize(run_dir: Path, smoke: bool) -> None:
    rows = {}
    for v in C.VARIANTS:
        for s in C.SEEDS:
            r = dict(read(run_dir / "probes" / f"{v}__seed{s}.json") or {})
            m = b0_metrics(run_dir, v, s)
            if m:
                r["MoF"], r["F1_50"] = m
            if r:
                rows[(v, s)] = r
    dec = decide(rows, smoke)
    L = [f"# Input-representation test {run_dir.name}", "", "SMOKE RUN: plumbing only, not evidence." if smoke else "",
         "LARa, held-out matched B0 decode (44 evaluation recordings, final endpoint; T2 = Milestone 1 reference cells), "
         "label-using probes as diagnostics.", "",
         "| input | seed | B0 MoF | B0 F1@50 | action probe (continuous) | action probe (code table) | majority | subject probe (chance) |",
         "|---|---|---|---|---|---|---|---|"]
    f = lambda r, k: "-" if r.get(k) is None else f"{r[k]:.1f}"
    for (v, s), r in sorted(rows.items()):
        L.append(f"| {v} | {s} | {f(r, 'MoF')} | {f(r, 'F1_50')} | {f(r, 'action_probe_continuous')} | {f(r, 'action_probe_code_table')} | "
                 f"{f(r, 'majority_baseline')} | {f(r, 'subject_probe_accuracy')} ({f(r, 'subject_probe_chance')}) |")
    L += ["", "## Decision (declared rules)", "", f"**{dec['verdict']}**", ""]
    for v, c in dec.get("comparisons", {}).items():
        L.append(f"- {v} − t2: mean MoF {c['mean_MoF']:+.2f}, mean F1@50 {c['mean_F1_50']:+.2f}, per-seed MoF "
                 f"{', '.join(f'{x:+.2f}' for x in c['per_seed_MoF'])}, F1@50 {', '.join(f'{x:+.2f}' for x in c['per_seed_F1_50'])} -> "
                 f"{'passes' if c['passes'] else 'does not pass'} (bar +{C.BAR:g} on both, all seeds positive)")
    if "probe_minus_mof" in dec:
        L.append(f"- supervised probe minus unsupervised MoF (mean over seeds): " + ", ".join(f"{k} {v:+.1f}" for k, v in dec["probe_minus_mof"].items())
                 + f"; threshold {C.GAP_THRESHOLD:g}")
    if dec.get("reason"):
        L.append(f"- {dec['reason']}")
    L += ["", "The probes use action labels and subject metadata and are upper-bound diagnostics, never a selection signal. "
          "t2_centered is not heading-invariant, so a win could come from orientation rather than posture."]
    (run_dir / "REPORT.md").write_text("\n".join(x for x in L), encoding="utf-8")
    write_json(run_dir / "decision.json", dict(decision=dec, rows={f"{v}__{s}": r for (v, s), r in rows.items()}, smoke=smoke))
    print(f"[report] {run_dir / 'REPORT.md'}: {dec['verdict']}")


def plan(variants, seeds):
    out = []
    for s in seeds:
        for v in variants:
            if v in C.NEW_VARIANTS:
                out += [(v, s, "prep"), (v, s, "b0")]
            out.append((v, s, "probe"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="script.input_representation.run")
    sub = ap.add_subparsers(dest="mode")
    for m in ("preflight", "dry-run", "smoke", "run"):
        p = sub.add_parser(m)
        p.add_argument("--run-id", default=None)
        p.add_argument("--variants", default=",".join(C.VARIANTS))
        p.add_argument("--seeds", default=",".join(map(str, C.SEEDS)))
        p.add_argument("--hours", type=float, default=None)
        p.add_argument("--reuse-completed", action="store_true")
    c = sub.add_parser("cell")
    c.add_argument("--run-id", required=True)
    c.add_argument("--variant", required=True)
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--stage", required=True)
    c.add_argument("--smoke", action="store_true")
    c.add_argument("--reuse-completed", action="store_true")
    a = ap.parse_args(argv)
    if a.mode is None:
        ap.print_help()
        return 0
    if a.mode == "cell":
        return run_stage(C.RESULTS / a.run_id, a.variant, a.seed, a.stage, a.smoke, "cpu", a.reuse_completed)
    variants = [x for x in a.variants.split(",") if x]
    seeds = [int(x) for x in a.seeds.split(",") if x]
    if a.mode == "preflight":
        problems = []
        for s in seeds:
            if not (C.REF_RUN / "cells" / f"b0_centre__seed{s}" / "scores.json").exists():
                problems.append(f"missing T2 reference cell for seed {s}")
            if not C.prep_dir("t2", s).exists():
                problems.append(f"missing T2 preparation for seed {s}")
        from script.input_representation import features
        from script.week5.timescale_inspection import tokenizers as TOK
        features.install()
        spec = TOK.dataset_spec(C.DATASET)
        names = sorted(__import__("script.week5.raw_kmeans_cross_dataset.datasets", fromlist=["x"]).recording_names(spec, None))[:2]
        for src in features.NEW_SOURCES:
            x = features.extract_new(spec, src, names)
            print(f"{src}: windows x dim = {x[names[0]]['flat'].shape}")
        print(json.dumps(dict(ok=not problems, problems=problems), indent=2))
        return 1 if problems else 0
    if a.mode == "dry-run":
        n_new = sum(1 for v in variants if v in C.NEW_VARIANTS) * len(seeds)
        print(f"stages: {len(plan(variants, seeds))}; B0 decodes ~9 min and prep ~2-3 min each for {n_new} new cells; "
              f"probes ~1-2 min each -> about {n_new * 12 / 60 + len(variants) * len(seeds) * 1.5 / 60:.1f} h (estimate, unmeasured until smoke)")
        return 0
    if not a.run_id:
        print("--run-id is required", file=sys.stderr)
        return 2
    smoke = a.mode == "smoke"
    if smoke != ("smoke" in a.run_id):
        print("smoke run ids must contain 'smoke' and full run ids must not", file=sys.stderr)
        return 2
    if smoke:
        variants, seeds = ["t2_bodyscale"], [111]
    run_dir = C.RESULTS / a.run_id
    start, bad = time.perf_counter(), 0
    for v, s, stage in plan(variants, seeds):
        if a.hours and time.perf_counter() - start > a.hours * 3600:
            print(f"[run] cap reached; not launching {v} {s} {stage}", flush=True)
            continue
        cmd = [sys.executable, "-m", "script.input_representation.run", "cell", "--run-id", a.run_id, "--variant", v, "--seed", str(s), "--stage", stage]
        if smoke:
            cmd.append("--smoke")
        if a.reuse_completed:
            cmd.append("--reuse-completed")
        code = subprocess.run(cmd, cwd=str(C.ROOT)).returncode
        bad = max(bad, code)
        if code and stage == "prep":
            print(f"[run] prep failed for {v} {s}; skipping its later stages", flush=True)
            continue
    write_json(run_dir / "manifest.json", dict(run_id=a.run_id, variants=variants, seeds=seeds, smoke=smoke, code_hash=C.code_hash(),
                                                finished=datetime.now().isoformat(timespec="seconds")))
    finalize(run_dir, smoke)
    return bad


if __name__ == "__main__":
    sys.exit(main())
