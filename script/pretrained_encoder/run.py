"""Pretrained-encoder test driver (BABEL-2): preflight, smoke, dry-run, run (stages in fresh child processes) and the child `cell`.

Per (input, seed): prep (mb only) -> b0 on each of 5 recording-disjoint folds (the Milestone 1 matched B0 machinery, fitted on the
other four folds and decoded on the held-out fold) -> probe (label-using diagnostics). Then REPORT.md with the declared rule."""
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

from script.pretrained_encoder import config as C  # noqa: E402


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    os.replace(tmp, path)


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


# --- folds and the Milestone 1 overrides -----------------------------------------------------------------------------

def make_folds(names, n_folds: int = C.N_FOLDS, seed: int = C.FOLD_SEED) -> list:
    """Label-free recording-disjoint folds: recordings shuffled with the fixed seed, dealt round-robin."""
    names = sorted(names)
    order = np.random.default_rng(seed).permutation(len(names))
    return [sorted(names[i] for i in order[k::n_folds]) for k in range(n_folds)]


def split_names(loader, split: str):
    names = sorted(loader.recording_names())
    folds = make_folds(names)
    held = set(folds[int(split[4:])])
    return [n for n in names if n not in held], [n for n in names if n in held]


def apply_overrides(inp: str) -> None:
    """Point the Milestone 1 data/train/evaluate code at BABEL-2 and this input's preparation, at run time and in this process only."""
    from script.compositional_dynamics import config as CD
    from script.compositional_dynamics import data as Dm
    CD.DATASET, CD.WINDOW, CD.FPS, CD.NUM_STATES = C.DATASET, C.WINDOW, C.FPS, C.NUM_STATES
    CD.FAMILY = C.family(inp)
    CD.prep_dir = lambda seed: C.prep_dir(inp, seed)
    Dm.split_names = split_names
    Dm.subject_of = lambda name: name              # BABEL has no subject IDs: the "subject" association is the recording association


def b0_dir(run_dir: Path, inp: str) -> Path:
    return run_dir / f"b0__{inp}"


def run_stage(run_dir: Path, inp: str, seed: int, stage: str, split: str, smoke: bool, reuse: bool) -> int:
    t0 = time.perf_counter()
    if stage == "prep":
        from script.pretrained_encoder import features
        d = features.prepare(seed)
        print(f"[prep] mb seed {seed}: {d} ({time.perf_counter() - t0:.0f}s)", flush=True)
        return 0
    if stage == "b0":
        from script.compositional_dynamics import config as CD
        from script.compositional_dynamics import run as CR
        apply_overrides(inp)
        return CR.run_cell(b0_dir(run_dir, inp), seed, "b0_centre", CD.SMOKE if smoke else CD.FULL, "cpu", None, reuse, split)
    if stage == "probe":
        out = run_dir / "probes" / f"{inp}__seed{seed}.json"
        old = read(out) or {}
        if reuse and old.get("code_hash") == C.code_hash() and old.get("smoke") == smoke and old.get("done"):
            print(f"[probe] {inp} seed {seed}: reused", flush=True)
            return 0
        res = probe(inp, seed, C.SMOKE_PROBE_CAP if smoke else C.PROBE_CAP)
        write_json(out, dict(res, input=inp, seed=seed, smoke=smoke, code_hash=C.code_hash(), done=True, seconds=time.perf_counter() - t0))
        print(f"[probe] {inp} seed {seed}: action {res['action_probe_continuous']:.1f} / code table {res['action_probe_code_table']:.1f} "
              f"(majority {res['majority_baseline']:.1f}) ({time.perf_counter() - t0:.0f}s)", flush=True)
        return 0
    raise ValueError(stage)


# --- label-using probes (diagnostics only) -------------------------------------------------------------------------

def probe(inp: str, seed: int, cap: int) -> dict:
    from script.input_representation import probes as P
    from script.quantized_input_comparison.artifacts import Loader
    from script.week5.raw_kmeans_cross_dataset import datasets as D
    loader = Loader(C.prep_dir(inp, seed))
    names = sorted(loader.recording_names())
    spec = D.SPECS[C.DATASET]
    gts = D.load_gts(spec, names)
    n_classes = int(max(g.max() for g in gts.values())) + 1
    X, codes, lens, y, rec = [], [], [], [], []
    for i, n in enumerate(names):
        r = loader.recording(C.family(inp), n)
        X.append(r["continuous"]); codes.append(r["codes"]); lens.append(r["lengths"].astype(np.float64))
        y.append(P.window_majority(gts[n], r["starts"].astype(np.int64), r["lengths"].astype(np.int64), n_classes)); rec.append(np.full(len(r["codes"]), i))
    X, codes, lens, y, rec = (np.concatenate(a) for a in (X, codes, lens, y, rec))
    folds = make_folds(names)
    lr, ct, mj = [], [], []
    for k, held in enumerate(folds):
        test = np.isin(rec, [names.index(n) for n in held])
        tr = ~test
        pred = P._fit_predict(X[tr], y[tr], X[test], cap, C.PROBE_SEED + k)
        lr.append(P._weighted_acc(pred, y[test], lens[test]))
        table = np.zeros((512, n_classes))
        np.add.at(table, (codes[tr], y[tr]), lens[tr])
        glob = np.bincount(y[tr], weights=lens[tr], minlength=n_classes).argmax()
        lookup = np.where(table.sum(1) > 0, table.argmax(1), glob)
        ct.append(P._weighted_acc(lookup[codes[test]], y[test], lens[test]))
        mj.append(P._weighted_acc(np.full(test.sum(), glob), y[test], lens[test]))
    return dict(action_probe_continuous=float(np.mean(lr)), action_probe_continuous_per_fold=lr, action_probe_code_table=float(np.mean(ct)),
                majority_baseline=float(np.mean(mj)), probe_cap=cap, n_windows=int(len(y)))


# --- report -------------------------------------------------------------------------------------------------------

def fold_metrics(run_dir: Path, inp: str, seed: int, folds=C.FOLDS):
    vals = []
    for f in folds:
        sc = read(b0_dir(run_dir, inp) / "cells" / f"b0_centre__seed{seed}__{f}" / "scores.json")
        if not sc:
            return None
        m = sc["eval"][str(sc["primary_endpoint_steps"])]
        vals.append((m["metrics"]["MoF"], m["metrics"]["F1@50"], 100.0 * float(np.mean(list(m["per_class_recall"].values())))))
    return tuple(float(np.mean([v[i] for v in vals])) for i in range(3))


def decide(rows: dict, smoke: bool, verified: bool) -> dict:
    """Declared rule (plan Section 4); rows[(input, seed)] = dict(MoF, F1_50, probe)."""
    if smoke:
        return dict(verdict="not computed (smoke run)")
    if not verified or any(("MoF" not in rows.get((i, s), {}) or "probe" not in rows.get((i, s), {})) for i in C.INPUTS for s in C.SEEDS):
        return dict(verdict="INCONCLUSIVE")
    mof = [rows[("mb", s)]["MoF"] - rows[("t2", s)]["MoF"] for s in C.SEEDS]
    f1 = [rows[("mb", s)]["F1_50"] - rows[("t2", s)]["F1_50"] for s in C.SEEDS]
    probe_gain = float(np.mean([rows[("mb", s)]["probe"] - rows[("t2", s)]["probe"] for s in C.SEEDS]))
    out = dict(mb_minus_t2_MoF=mof, mb_minus_t2_F1_50=f1, probe_gain=probe_gain)
    adopt = all(np.mean(x) >= C.BAR and min(x) > 0 for x in (mof, f1))
    out["verdict"] = "ADOPT_ENCODER" if adopt else ("INFORMATION_ONLY" if probe_gain >= C.PROBE_GAIN else "STOP")
    return out


def finalize(run_dir: Path, smoke: bool, seeds, inputs, folds) -> None:
    rows = {}
    for inp in inputs:
        for s in seeds:
            fm = fold_metrics(run_dir, inp, s, folds)
            pr = read(run_dir / "probes" / f"{inp}__seed{s}.json")
            r = {}
            if fm:
                r.update(MoF=fm[0], F1_50=fm[1], bal=fm[2])
            if pr:
                r.update(probe=pr["action_probe_continuous"], code_table=pr["action_probe_code_table"], majority=pr["majority_baseline"])
            if r:
                rows[(inp, s)] = r
    ver = read(C.RESULTS / "verification.json")
    verified = bool(ver and ver.get("passed") and ver.get("code_hash") == C.code_hash())
    dec = decide(rows, smoke, verified)
    f = lambda r, k: "-" if k not in r else f"{r[k]:.1f}"
    L = [f"# Pretrained-encoder test {run_dir.name}", "", "SMOKE RUN: plumbing only, not evidence." if smoke else "",
         "BABEL-2; matched B0 (C = 5) fitted on four recording folds and decoded on the held-out fold, mean over the folds run; final endpoint; "
         "one Hungarian mapping per fold. Probes use labels and are diagnostics only.", "",
         f"Folds run: {', '.join(folds)}.", "",
         "| input | seed | B0 MoF | B0 F1@50 | balanced acc | action probe (continuous) | code table | majority |", "|---|---|---|---|---|---|---|---|"]
    for (inp, s), r in sorted(rows.items()):
        L.append(f"| {inp} | {s} | {f(r, 'MoF')} | {f(r, 'F1_50')} | {f(r, 'bal')} | {f(r, 'probe')} | {f(r, 'code_table')} | {f(r, 'majority')} |")
    L += ["", "## Decision (declared rules)", "", f"**{dec['verdict']}**", ""]
    if "mb_minus_t2_MoF" in dec:
        L.append(f"- mb − t2 MoF per seed: {', '.join(f'{x:+.2f}' for x in dec['mb_minus_t2_MoF'])} (mean {np.mean(dec['mb_minus_t2_MoF']):+.2f}); "
                 f"F1@50: {', '.join(f'{x:+.2f}' for x in dec['mb_minus_t2_F1_50'])} (mean {np.mean(dec['mb_minus_t2_F1_50']):+.2f}); bar +{C.BAR:g} on both, all seeds positive")
        L.append(f"- supervised probe gain mb − t2 (mean over seeds): {dec['probe_gain']:+.1f} (INFORMATION_ONLY needs ≥ {C.PROBE_GAIN:g})")
    if dec["verdict"] == "INCONCLUSIVE":
        L.append(f"- missing seeds/folds/probes or verification not recorded as passed for this code hash (verification ok: {verified})")
    L += ["", "One encoder, one checkpoint, one pooling, one view, frozen. The encoder saw AMASS motions (including BABEL) without labels; the tokenizer is fitted on all "
          "recordings. Recording-disjoint folds share performers (BABEL has no subject IDs). Three seeds are not confidence intervals."]
    (run_dir / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    write_json(run_dir / "decision.json", dict(decision=dec, rows={f"{i}__{s}": r for (i, s), r in rows.items()}, smoke=smoke))
    print(f"[report] {run_dir / 'REPORT.md'}: {dec['verdict']}")


def plan(inputs, seeds, folds):
    out = []
    for s in seeds:
        for inp in inputs:
            if inp == "mb":
                out.append((inp, s, "prep", "primary"))
            out += [(inp, s, "b0", f) for f in folds]
            out.append((inp, s, "probe", "primary"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="script.pretrained_encoder.run")
    sub = ap.add_subparsers(dest="mode")
    for m in ("preflight", "dry-run", "smoke", "run"):
        p = sub.add_parser(m)
        p.add_argument("--run-id", default=None)
        p.add_argument("--inputs", default=",".join(C.INPUTS))
        p.add_argument("--seeds", default=",".join(map(str, C.SEEDS)))
        p.add_argument("--folds", default=",".join(C.FOLDS))
        p.add_argument("--hours", type=float, default=None)
        p.add_argument("--reuse-completed", action="store_true")
    c = sub.add_parser("cell")
    c.add_argument("--run-id", required=True)
    c.add_argument("--input", required=True)
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--stage", required=True)
    c.add_argument("--split", default="primary")
    c.add_argument("--smoke", action="store_true")
    c.add_argument("--reuse-completed", action="store_true")
    a = ap.parse_args(argv)
    if a.mode is None:
        ap.print_help()
        return 0
    if a.mode == "cell":
        return run_stage(C.RESULTS / a.run_id, a.input, a.seed, a.stage, a.split, a.smoke, a.reuse_completed)
    inputs, seeds, folds = [x for x in a.inputs.split(",") if x], [int(x) for x in a.seeds.split(",") if x], [x for x in a.folds.split(",") if x]
    if a.mode == "preflight":
        problems = [f"missing T2 preparation for seed {s}" for s in seeds if not (C.prep_dir("t2", s) / "manifest.json").exists()]
        from script.pretrained_encoder import encoder as E
        from script.pretrained_encoder import features as F
        try:
            F.checkpoint_ok()
        except Exception as exc:  # noqa: BLE001
            problems.append(f"checkpoint: {exc}")
        from script.week5.raw_kmeans_cross_dataset import datasets as D
        import torch
        spec = D.SPECS[C.DATASET]
        names = D.recording_names(spec, None)
        n = names[0]
        arr = np.load(D.features_dir(spec) / n)
        feat = F.frame_features(arr[..., 0].transpose(1, 2, 0), torch.device("cpu"))
        print(f"{n}: frames {arr.shape[1]} -> features {feat.shape}, finite {bool(np.isfinite(feat).all())}; folds sizes {[len(f) for f in make_folds(names)]}")
        print(json.dumps(dict(ok=not problems, problems=problems), indent=2))
        return 1 if problems else 0
    if a.mode == "dry-run":
        n_b0 = len(inputs) * len(seeds) * len(folds)
        print(f"{len(plan(inputs, seeds, folds))} stages; {n_b0} B0 decodes at ~2-3 min each plus feature extraction (~1 min once) and probes "
              f"-> about {(n_b0 * 2.5 + len(inputs) * len(seeds) * 1.0) / 60:.1f} h (estimate, unmeasured until smoke)")
        return 0
    if not a.run_id:
        print("--run-id is required", file=sys.stderr)
        return 2
    smoke = a.mode == "smoke"
    if smoke != ("smoke" in a.run_id):
        print("smoke run ids must contain 'smoke' and full run ids must not", file=sys.stderr)
        return 2
    if smoke:
        seeds, folds = [111], ["fold0"]
    run_dir = C.RESULTS / a.run_id
    start, bad = time.perf_counter(), 0
    skip_seed_input = set()
    for inp, s, stage, split in plan(inputs, seeds, folds):
        if (inp, s) in skip_seed_input:
            continue
        if a.hours and time.perf_counter() - start > a.hours * 3600:
            print(f"[run] cap reached; not launching {inp} {s} {stage} {split}", flush=True)
            continue
        cmd = [sys.executable, "-m", "script.pretrained_encoder.run", "cell", "--run-id", a.run_id, "--input", inp, "--seed", str(s),
               "--stage", stage, "--split", split]
        if smoke:
            cmd.append("--smoke")
        if a.reuse_completed:
            cmd.append("--reuse-completed")
        code = subprocess.run(cmd, cwd=str(C.ROOT)).returncode
        bad = max(bad, code)
        if code and stage == "prep":
            skip_seed_input.add((inp, s))
            print(f"[run] prep failed for {inp} seed {s}; skipping its later stages", flush=True)
    write_json(run_dir / "manifest.json", dict(run_id=a.run_id, inputs=inputs, seeds=seeds, folds=folds, smoke=smoke, code_hash=C.code_hash(),
                                                finished=datetime.now().isoformat(timespec="seconds")))
    finalize(run_dir, smoke, seeds, inputs, folds)
    return bad


if __name__ == "__main__":
    sys.exit(main())
