"""Self-labelled embedding test driver (BABEL-2, stage 1): preflight, smoke, dry-run, run, and the child `cell` / `bank` stages.

Cells are (arm, seed, fold) in fresh child processes; completed cells are reused by identity; interrupted cells restart. REPORT.md
applies the rules declared in docs/plans/SELF_LABELLED_EMBEDDING_PLAN.md (Section 6)."""
from __future__ import annotations

import argparse
import hashlib
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

from script.self_labelled import config as C  # noqa: E402

SMOKE_FIT, SMOKE_EVAL = 40, 10


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


def cell_dir(run_dir: Path, arm: str, seed: int, fold: str) -> Path:
    return run_dir / "cells" / f"{arm}__seed{seed}__{fold}"


def identity(arm: str, seed: int, fold: str, smoke: bool) -> str:
    payload = dict(code=C.code_hash(), arm=arm, seed=seed, fold=fold, smoke=smoke, updates=C.SMOKE_UPDATES if smoke else C.UPDATES)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def completed(d: Path, ident: str) -> bool:
    info = read(d / "info.json")
    return bool(info) and info.get("status") == "complete" and info.get("identity") == ident and (d / "scores.json").exists()


def subset(names: list, n: int, seed: int) -> list:
    return sorted(np.random.default_rng([seed, 99]).choice(names, min(n, len(names)), replace=False).tolist())


def stage_bank() -> int:
    from script.self_labelled import augment
    t0 = time.perf_counter()
    names = __import__("script.self_labelled.evaluate", fromlist=["x"]).all_names()
    augment.load_bank(names)
    augment.canonical_windows(names)
    print(f"[bank] {len(names)} recordings x {C.N_VARIANTS} variants ready ({time.perf_counter() - t0:.0f}s)", flush=True)
    return 0


def run_cell(run_dir: Path, arm: str, seed: int, fold: str, smoke: bool, device: str, deadline_s, reuse: bool) -> int:
    import torch
    from script.self_labelled import augment, evaluate as EV, train as TR
    d = cell_dir(run_dir, arm, seed, fold)
    ident = identity(arm, seed, fold, smoke)
    if reuse and completed(d, ident):
        print(f"[cell] {arm} seed {seed} {fold}: reused", flush=True)
        return 0
    t0 = time.perf_counter()
    deadline = t0 + deadline_s if deadline_s else None
    base = dict(identity=ident, arm=arm, seed=seed, fold=fold, smoke=smoke, code_hash=C.code_hash(), bank=str(augment.bank_dir()),
                started=datetime.now().isoformat(timespec="seconds"))
    write_json(d / "info.json", dict(status="running", **base))
    try:
        names = EV.all_names()
        fit_names, eval_names = EV.fold_split(names, fold)
        if arm == "F0":
            # historical, declared transductive baseline: the pretrained-encoder cell, verified against the identity the current code
            # would give it, plus the shortcut diagnostic on its decoder inputs
            if EV.f0_cell_dir(seed, fold, run_dir) is None:
                from script.pretrained_encoder import run as PR
                if PR.run_stage(run_dir, "mb", seed, "prep", "primary", False, True) or PR.run_stage(run_dir, "mb", seed, "b0", fold, False, True):
                    raise RuntimeError("F0 cell build failed")
            ver = EV.f0_verify(seed, fold, run_dir)
            if not ver["ok"]:
                write_json(d / "info.json", dict(status="invalid", reason=f"F0 provenance: {ver['reason']}", seconds=time.perf_counter() - t0, **base))
                print(f"[cell] F0 seed {seed} {fold}: INVALID provenance ({ver['reason']})", flush=True)
                return 4
            e, diag = EV.f0_scores(seed, fold, run_dir)
            out = dict(arm=arm, seed=seed, fold=fold, source=ver, n_eval=len(eval_names), transductive_tokenization=True,
                       eval=dict(metrics=e["metrics"], per_class_recall=e["per_class_recall"],
                                 balanced_accuracy=100.0 * float(np.mean(list(e["per_class_recall"].values()))),
                                 association=diag["association"]["eval"]["after"]),
                       nn_share_eval=EV.nn_share(EV.f0_vectors(seed, eval_names)))
            write_json(d / "scores.json", out)
            write_json(d / "info.json", dict(status="complete", seconds=time.perf_counter() - t0, **base))
            print(f"[cell] F0 seed {seed} {fold}: verified (MoF {e['metrics']['MoF']:.1f} F1@50 {e['metrics']['F1@50']:.1f})", flush=True)
            return 0
        if smoke:
            fit_names, eval_names = subset(fit_names, SMOKE_FIT, seed), subset(eval_names, SMOKE_EVAL, seed)
        lay = EV.layout(fit_names + eval_names)
        out = dict(arm=arm, seed=seed, fold=fold, n_fit=len(fit_names), n_eval=len(eval_names))
        if arm == "F1":
            z = EV.f1_inputs(seed, fit_names, eval_names)
            out["eval"] = EV.save_decode(d / "decode_primary.npz", EV.decode(z, fit_names, eval_names, lay, seed, smoke, deadline=deadline))
            out["nn_share_eval"] = EV.nn_share({n: z[n] for n in eval_names})
        else:
            canon = augment.canonical_windows(fit_names + eval_names)
            bank = {} if arm == "L_noaug" else augment.load_bank(fit_names, build_missing=smoke)
            dev = torch.device(device)
            try:
                enc, protos, (mean, std), log = TR.train(fit_names, canon, bank, {n: lay[n]["lengths"] for n in fit_names}, arm, seed, dev,
                                                         updates=C.SMOKE_UPDATES if smoke else C.UPDATES, ot_steps=C.OT_INNER_STEPS, deadline=deadline)
            except TR.OTFailure as exc:
                write_json(d / "info.json", dict(status="invalid", reason=f"OTFailure: {exc}", seconds=time.perf_counter() - t0, **base))
                print(f"[cell] {arm} seed {seed} {fold}: INVALID ({exc})", flush=True)
                return 4
            write_json(d / "training_log.json", {k: v for k, v in log.items()})
            k = C.COLLAPSE_WINDOW
            out["training"] = dict(seconds=log["seconds"], collapse=log["collapse"], collapsed=log["collapsed"],
                                   first_loss=float(np.mean(log["loss"][:k])), final_loss=float(np.mean(log["loss"][-k:])),
                                   **{f"{key}_end": float(np.mean(log[key][-k:])) for key in
                                      ("usage_entropy", "min_state_share", "max_state_share", "hard_effective_states", "hard_min_share",
                                       "embedding_effective_rank", "max_prototype_cos")})
            z = TR.embed_all(enc, canon, mean, std, dev)
            out["eval"] = EV.save_decode(d / "decode_primary.npz", EV.decode(z, fit_names, eval_names, lay, seed, smoke, deadline=deadline))
            sec = EV.save_decode(d / "decode_own_prototypes.npz", EV.decode(z, fit_names, eval_names, lay, seed, smoke, mu_fixed=protos.cpu().numpy(),
                                                                             deadline=deadline))
            out["secondary_own_prototypes"] = {key: sec[key] for key in ("metrics", "balanced_accuracy", "per_class_recall", "solver")}
            out["nn_share_eval"] = EV.nn_share({n: z[n] for n in eval_names})
            out["final_embedding_effective_rank_fit"] = TR.effective_rank(torch.as_tensor(np.concatenate([z[n] for n in fit_names])))
            torch.save(dict(encoder=enc.state_dict(), prototypes=protos.cpu(), mean=mean, std=std), d / "model.pt")
            np.savez_compressed(d / "embeddings.npz", **{n: z[n].astype(np.float32) for n in fit_names + eval_names})
        write_json(d / "scores.json", out)
        write_json(d / "info.json", dict(status="complete", seconds=time.perf_counter() - t0, finished=datetime.now().isoformat(timespec="seconds"), **base))
        m = out["eval"]["metrics"]
        print(f"[cell] {arm} seed {seed} {fold}: MoF {m['MoF']:.1f} F1@50 {m['F1@50']:.1f} ({time.perf_counter() - t0:.0f}s)", flush=True)
        return 0
    except TimeoutError:
        write_json(d / "info.json", dict(status="interrupted", seconds=time.perf_counter() - t0, **base))
        print(f"[cell] {arm} seed {seed} {fold}: INTERRUPTED (not a result)", flush=True)
        return 3
    except Exception as exc:  # noqa: BLE001
        from script.action_transport.budget import DeadlineExceeded
        write_json(d / "info.json", dict(status="interrupted" if isinstance(exc, DeadlineExceeded) else "failed", reason=f"{type(exc).__name__}: {exc}",
                                         seconds=time.perf_counter() - t0, **base))
        print(f"[cell] {arm} seed {seed} {fold}: {type(exc).__name__}: {exc}", flush=True)
        return 3 if isinstance(exc, DeadlineExceeded) else 4


# --- decision rules and report ----------------------------------------------------------------------------------

def dev_rule(d_mof: list, d_f1: list) -> dict:
    """Development rule: each mean >= FLOOR and at least one mean >= GAIN with >= MIN_SEEDS_POS seeds positive on that metric."""
    mof, f1 = np.mean(d_mof), np.mean(d_f1)
    gain = any(m >= C.GAIN and int(np.sum(np.asarray(d) > 0)) >= C.MIN_SEEDS_POS for m, d in ((mof, d_mof), (f1, d_f1)))
    return dict(holds=bool(mof >= C.FLOOR and f1 >= C.FLOOR and gain), mean_MoF=float(mof), mean_F1_50=float(f1),
                per_seed_MoF=[float(x) for x in d_mof], per_seed_F1_50=[float(x) for x in d_f1])


def valid_cell(run_dir: Path, arm: str, seed: int, fold: str):
    """The cell's scores if it is complete AND its recorded identity equals the identity the current code assigns; else None."""
    d = cell_dir(run_dir, arm, seed, fold)
    info, sc = read(d / "info.json"), read(d / "scores.json")
    if not info or not sc or info.get("status") != "complete" or info.get("identity") != identity(arm, seed, fold, False) or "eval" not in sc:
        return None
    return sc


def seed_metrics(run_dir: Path, arm: str, seed: int, folds):
    """Fold means of (MoF, F1@50, balanced acc, recording NMI, shared-context NN share, support-disjoint NN share among eligible
    queries, collapsed folds); None unless EVERY fold's cell is valid."""
    vals, collapsed = [], 0
    for f in folds:
        sc = valid_cell(run_dir, arm, seed, f)
        if sc is None:
            return None
        e, nn = sc["eval"], sc.get("nn_share_eval") or {}
        rec_nmi = e["association"]["recording"]["nmi"]
        sd = (nn.get("support_disjoint") or {}).get("share_eligible")
        vals.append((e["metrics"]["MoF"], e["metrics"]["F1@50"], e["balanced_accuracy"], rec_nmi,
                     (nn.get("shared_context") or {}).get("share_all", np.nan), np.nan if sd is None else sd))
        collapsed += int(bool(sc.get("training", {}).get("collapsed")))
    return tuple(float(np.nanmean([v[i] for v in vals])) if not all(np.isnan(v[i]) for v in vals) else float("nan") for i in range(6)) + (collapsed,)


def decide(table: dict, seeds, folds, arms, verified: bool, smoke: bool) -> dict:
    """Declared rule (plan Section 6), computed only on the complete declared grid: all five seeds, all five folds, all five arms, every
    cell valid for the current code, and a passing test record. A collapsed L_full cell blocks adoption."""
    if smoke:
        return dict(verdict="not computed (smoke run)")
    reasons = []
    if list(seeds) != list(C.SEEDS):
        reasons.append(f"seeds {list(seeds)} != declared {list(C.SEEDS)}")
    if list(folds) != list(C.FOLDS):
        reasons.append(f"folds {list(folds)} != declared {list(C.FOLDS)}")
    if not set(C.ARMS) <= set(arms):
        reasons.append(f"arms {sorted(arms)} do not include all of {list(C.ARMS)}")
    if not verified:
        reasons.append("verification not recorded as passed for this code hash")
    missing = [f"{a}/{s}" for a in C.ARMS for s in C.SEEDS if table.get((a, s)) is None]
    if missing:
        reasons.append(f"incomplete or invalid cells for {missing[:10]}")
    if reasons:
        return dict(verdict="INCONCLUSIVE", reasons=reasons)
    out = {}
    for name, other in (("vs_F0", "F0"), ("vs_F1", "F1"), ("vs_L_noeq", "L_noeq"), ("vs_L_noaug", "L_noaug")):
        out[name] = dev_rule([table[("L_full", s)][0] - table[(other, s)][0] for s in seeds], [table[("L_full", s)][1] - table[(other, s)][1] for s in seeds])
    collapsed_full = int(sum(table[("L_full", s)][6] for s in seeds))
    adopt = out["vs_F0"]["holds"] and collapsed_full == 0
    learning = out["vs_F1"]["holds"]
    out.update(adopt=adopt, learning_claim=learning, equipartition_claim=out["vs_L_noeq"]["holds"], augmentation_claim=out["vs_L_noaug"]["holds"],
               collapsed_L_full_cells=collapsed_full,
               verdict=("STAGE2_ELIGIBLE" if adopt and learning else ("ADOPT_BUT_NOT_LEARNING" if adopt else
                                                                    ("BLOCKED_BY_COLLAPSE" if out["vs_F0"]["holds"] else "STOP"))))
    return out


def finalize(run_dir: Path, seeds, folds, smoke: bool, arms) -> None:
    table = {(a, s): seed_metrics(run_dir, a, s, folds) for a in arms for s in seeds} if not smoke else {}
    ver = read(C.RESULTS / "verification.json")
    verified = bool(ver and ver.get("passed") and ver.get("code_hash") == C.code_hash())
    dec = decide(table, seeds, folds, arms, verified, smoke)
    f = lambda v, i: "-" if v is None or v[i] != v[i] else f"{v[i]:.2f}"
    L = [f"# Self-labelled embedding test {run_dir.name}", "", "SMOKE RUN: plumbing only, not evidence." if smoke else "",
         "BABEL-2; frozen-MotionBERT window features; matched B0 decode on held-out recording folds (mean over folds), final endpoint, one Hungarian "
         "mapping per fold. `F0` = frozen MotionBERT K-means codes from pe_babel2_001 (verified identity; declared transductive tokenization). "
         "`F1` and the learned arms fit every preprocessing step on the fold's fitting recordings only.", "",
         "| arm | seed | MoF | F1@50 | balanced acc | recording NMI | NN share (shared context) | NN share (support-disjoint, eligible) | collapsed folds |",
         "|---|---|---|---|---|---|---|---|---|"]
    for (a, s), v in sorted(table.items()):
        L.append(f"| {a} | {s} | {f(v, 0)} | {f(v, 1)} | {f(v, 2)} | {f(v, 3)} | {f(v, 4)} | {f(v, 5)} | {'-' if v is None else v[6]} |")
    L += ["", "## Decision (rules declared in the plan)", "", f"**{dec['verdict']}**", ""]
    for r in dec.get("reasons", []):
        L.append(f"- {r}")
    for k in ("vs_F0", "vs_F1", "vs_L_noeq", "vs_L_noaug"):
        if k in dec:
            r = dec[k]
            L.append(f"- L_full {k}: mean MoF {r['mean_MoF']:+.2f}, mean F1@50 {r['mean_F1_50']:+.2f}; per-seed MoF "
                     f"{', '.join(f'{x:+.1f}' for x in r['per_seed_MoF'])}, F1@50 {', '.join(f'{x:+.1f}' for x in r['per_seed_F1_50'])} -> "
                     f"{'rule holds' if r['holds'] else 'rule fails'}")
    if "adopt" in dec:
        L.append(f"- adopt (vs F0, no collapsed L_full cell): {dec['adopt']}; collapsed L_full cells: {dec['collapsed_L_full_cells']}; "
                 f"learning claim (vs F1): {dec['learning_claim']}; equipartition claim: {dec['equipartition_claim']}; augmentation claim: {dec['augmentation_claim']}")
    L += ["", "Development data (BABEL-2 was diagnosed with labels in earlier tests). Five seeds measure initialization and batch-order sensitivity on the same "
          "recordings. The shortcut diagnostics are reported, not gated; the shared-context neighbour share can still reflect shared MotionBERT source frames, "
          "and the support-disjoint share is computed only over queries that have an eligible same-recording candidate. Batch equipartition does not by itself "
          "prevent recording-based states: several recordings can occupy different states while the batch stays balanced."]
    (run_dir / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    write_json(run_dir / "decision.json", dict(decision=dec, table={f"{a}__{s}": v for (a, s), v in table.items()}, smoke=smoke))
    print(f"[report] {run_dir / 'REPORT.md'}: {dec['verdict']}")


# --- driver ------------------------------------------------------------------------------------------------------------

def child(cmd: list) -> int:
    return subprocess.run([sys.executable, "-m", "script.self_labelled.run"] + cmd, cwd=str(C.ROOT)).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="script.self_labelled.run")
    sub = ap.add_subparsers(dest="mode")
    for m in ("preflight", "dry-run", "smoke", "run"):
        p = sub.add_parser(m)
        p.add_argument("--run-id", default=None)
        p.add_argument("--arms", default=",".join(C.ARMS))
        p.add_argument("--seeds", default=",".join(map(str, C.SEEDS)))
        p.add_argument("--folds", default=",".join(C.FOLDS))
        p.add_argument("--device", default="cuda")
        p.add_argument("--hours", type=float, default=None)
        p.add_argument("--reuse-completed", action="store_true")
    sub.add_parser("bank")
    c = sub.add_parser("cell")
    c.add_argument("--run-id", required=True)
    c.add_argument("--arm", required=True)
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--fold", required=True)
    c.add_argument("--device", default="cuda")
    c.add_argument("--smoke", action="store_true")
    c.add_argument("--deadline-s", type=float, default=None)
    c.add_argument("--reuse-completed", action="store_true")
    a = ap.parse_args(argv)
    if a.mode is None:
        ap.print_help()
        return 0
    if a.mode == "bank":
        return stage_bank()
    if a.mode == "cell":
        return run_cell(C.RESULTS / a.run_id, a.arm, a.seed, a.fold, a.smoke, a.device, a.deadline_s, a.reuse_completed)
    arms, seeds, folds = [x for x in a.arms.split(",") if x], [int(x) for x in a.seeds.split(",") if x], [x for x in a.folds.split(",") if x]
    if a.mode == "preflight":
        import torch
        from script.self_labelled import evaluate as EV
        problems = []
        names = EV.all_names()
        from script.pretrained_encoder import config as PC
        from script.quantized_input_comparison.artifacts import Loader
        for s in C.PE_SEEDS:
            if sorted(Loader(PC.prep_dir("mb", s)).recording_names()) != names:
                problems.append(f"recording names of the mb preparation (seed {s}) differ from the dataset listing")
            for fo in folds:
                if EV.f0_scores(s, fo, C.RESULTS / "none")[0] is None:
                    problems.append(f"missing F0 cell seed {s} {fo}")
        for fo in folds:
            fit, ev = EV.fold_split(names, fo)
            if set(fit) & set(ev) or len(fit) + len(ev) != len(names):
                problems.append(f"{fo} is not a partition")
        print(f"{len(names)} recordings; cuda {torch.cuda.is_available()}; receptive field +-{__import__('script.self_labelled.model', fromlist=['x']).receptive_field_windows()} windows")
        print(json.dumps(dict(ok=not problems, problems=problems), indent=2))
        return 1 if problems else 0
    if a.mode == "dry-run":
        n = {arm: len(seeds) * len(folds) for arm in arms}
        print(f"cells: {sum(n.values())}; rough per-cell minutes: F0 ~0 (reused for 3 seeds, ~2 for new seeds), F1 ~2, L_* ~9 (training ~5 + decodes ~4; measured per-update 0.1 s in smoke)")
        est = 2 * 2 * len(folds) + len(seeds) * len(folds) * 2 + sum(len(seeds) * len(folds) * 9.0 for x in arms if x in C.LEARNED)
        print(f"projected ~{est / 60:.1f} h for arms {arms}")
        return 0
    if not a.run_id:
        print("--run-id is required", file=sys.stderr)
        return 2
    smoke = a.mode == "smoke"
    if smoke != ("smoke" in a.run_id):
        print("smoke run ids must contain 'smoke' and full run ids must not", file=sys.stderr)
        return 2
    if smoke:
        arms, seeds, folds = [x for x in arms if x in ("F1", "L_full", "L_noeq", "L_noaug")][:4], [111], ["fold0"]
    run_dir = C.RESULTS / a.run_id
    if not smoke and child(["bank"]):
        print("bank build failed", file=sys.stderr)
        return 4
    start, bad = time.perf_counter(), 0
    order = [(arm, s, fo) for arm in [x for x in C.ARMS if x in arms] for s in seeds for fo in folds]
    for arm, s, fo in order:
        remaining = None if a.hours is None else a.hours * 3600 - (time.perf_counter() - start)
        if remaining is not None and remaining <= 0:
            print(f"[run] cap reached; not launching {arm} {s} {fo}", flush=True)
            continue
        cmd = ["cell", "--run-id", a.run_id, "--arm", arm, "--seed", str(s), "--fold", fo, "--device", a.device]
        if smoke:
            cmd.append("--smoke")
        if remaining is not None:
            cmd += ["--deadline-s", f"{remaining:.0f}"]
        if a.reuse_completed:
            cmd.append("--reuse-completed")
        bad = max(bad, child(cmd))
    write_json(run_dir / "manifest.json", dict(run_id=a.run_id, arms=arms, seeds=seeds, folds=folds, smoke=smoke, code_hash=C.code_hash(),
                                                finished=datetime.now().isoformat(timespec="seconds")))
    finalize(run_dir, seeds, folds, smoke, arms)
    return bad


if __name__ == "__main__":
    sys.exit(main())
