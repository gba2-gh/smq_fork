"""Unit-to-action grouping test driver: preflight, smoke, dry-run, run (cells in fresh child processes) and the child `cell`.

One cell = (seed, split): fit the M=24 state B0 on the fitting recordings, decode fitting and evaluation recordings afresh under
the frozen prototypes, learn the label-free groupings from the fitting plans only, relabel and score. The Milestone 1 machinery
(script/compositional_dynamics) is reused with its state count set to M at run time, without editing its code paths."""
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
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from script.unit_grouping import config as C  # noqa: E402
from script.unit_grouping import grouping as G  # noqa: E402

EXIT_DONE, EXIT_INTERRUPTED, EXIT_INVALID, EXIT_FAILED = 0, 3, 4, 1


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


def cell_dir(run_dir: Path, seed: int, split: str) -> Path:
    return run_dir / "cells" / (f"seed{seed}" + ("" if split == "primary" else f"__{split}"))


def identity(seed: int, split: str, scale, source_hash: str) -> str:
    import hashlib
    payload = dict(code=C.code_hash(), seed=seed, split=split, scale=scale.__dict__, source=source_hash, M=C.NUM_STATES,
                   h=C.HORIZON, n_random=C.N_RANDOM)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def completed(d: Path, ident: str) -> bool:
    info = read(d / "info.json")
    return bool(info) and info.get("status") == "complete" and info.get("identity") == ident and (d / "scores.json").exists()


# --- scoring helpers --------------------------------------------------------------------------------------------

def oracle_from_confusion(conf: dict) -> tuple:
    """conf: state -> {class: frames} (int keys, from evaluate.score_states). Returns (state -> majority class, oracle
    many-to-one MoF %): an upper bound on MoF for hard relabellings of these fixed states only."""
    assign = {int(s): max(row, key=row.get) for s, row in conf.items()}
    tot = sum(sum(row.values()) for row in conf.values())
    return assign, 100.0 * sum(max(row.values()) for row in conf.values()) / tot


def group_summary(groups: np.ndarray, conf: dict, oracle_assign: dict, mass: np.ndarray) -> dict:
    """Many-to-one view of a grouping on the evaluation frames (diagnostic only)."""
    merged: dict = {}
    for s, g in enumerate(groups):
        for c, n in conf.get(s, {}).items():
            merged.setdefault(int(g), {}).setdefault(int(c), 0)
            merged[int(g)][int(c)] += n
    tot = sum(sum(v.values()) for v in merged.values())
    maj = {g: max(v, key=v.get) for g, v in merged.items()}
    states = range(len(groups))
    return dict(many_to_one_mof=100.0 * sum(max(v.values()) for v in merged.values()) / tot,
                classes_with_group=len(set(maj.values())),
                ari_vs_oracle=float(adjusted_rand_score([oracle_assign.get(s, -1) for s in states], groups)),
                group_sizes=np.bincount(groups, minlength=C.NUM_ACTIONS).tolist(),
                group_mass=[float(mass[groups == g].sum() / mass.sum()) for g in range(C.NUM_ACTIONS)])


def score_arm(states: dict, data, gts, Ev) -> dict:
    sc = Ev.score_states(states, data.eval, gts, data.starts, data.frame_counts)
    rec = sc["per_class_recall"]
    return dict(metrics=sc["metrics"], balanced_accuracy=100.0 * float(np.mean(list(rec.values()))), per_class_recall=rec,
                pred_over_gt=sc["pred_runs"] / sc["gt_runs"], mapping=sc["mapping"],
                association=Ev.association(states, data.eval, gts, data.starts, data.frame_counts))


# --- one cell ----------------------------------------------------------------------------------------------------

def run_cell(run_dir: Path, seed: int, split: str, scale_name: str, deadline_s, reuse: bool) -> int:
    from script.action_transport.budget import DeadlineExceeded
    from script.compositional_dynamics import config as CD
    from script.compositional_dynamics import data as Dm, decode as Dc, evaluate as Ev, train as Tr

    scale = CD.SCALES[scale_name]
    CD.NUM_STATES = C.NUM_STATES                      # the Milestone 1 fit/decode code reads this at call time
    d = cell_dir(run_dir, seed, split)
    ident = identity(seed, split, scale, Dm.source_identity(seed)["recordings_and_all"])
    if reuse and completed(d, ident):
        print(f"[cell] seed {seed} {split}: reused", flush=True)
        return EXIT_DONE
    deadline = time.perf_counter() + deadline_s if deadline_s else None
    t0 = time.perf_counter()
    base = dict(identity=ident, seed=seed, split=split, scale=scale_name, smoke=bool(scale.smoke_fit), M=C.NUM_STATES,
                started=datetime.now().isoformat(timespec="seconds"), code_hash=C.code_hash())
    write_json(d / "info.json", dict(status="running", **base))
    try:
        data = Dm.load_cell_data(seed, scale, split)
        fit = Tr.fit_b0(data, scale, deadline)
        t_fit = time.perf_counter() - t0
        cost = {"fit": Tr.b0_costs(data, data.fit, fit.mu), "eval": Tr.b0_costs(data, data.eval, fit.mu)}
        final = scale.rungs[-1]
        # fresh decodes under the FROZEN prototypes; the fitting plans below come from here, not from the alternating fit
        dec_fit = Dc.decode_ladder(cost["fit"], data.fit, scale.rungs, deadline)
        dec_eval = Dc.decode_ladder(cost["eval"], data.eval, scale.rungs, deadline)
        t_dec = time.perf_counter() - t0 - t_fit
        plans = [dec_fit["responsibilities"][n] for n in data.fit.names]
        lengths = [data.fit.lengths[data.fit.slice_of(r)] for r in range(len(data.fit.names))]
        mass = G.state_mass(plans, lengths)
        groups = {"group_temporal": G.spectral_groups(G.affinity(plans), C.NUM_ACTIONS, seed),
                  "group_shuffled": G.spectral_groups(G.affinity(G.shuffle_rows(plans, seed)), C.NUM_ACTIONS, seed),
                  "group_appearance": G.appearance_groups(fit.mu, mass, C.NUM_ACTIONS, seed)}
        sizes = np.bincount(groups["group_temporal"], minlength=C.NUM_ACTIONS)
        randoms = G.size_matched_random(sizes, C.N_RANDOM, seed)
        # --- scoring (labels enter only here)
        gts = Ev.load_labels(data.eval_names)
        eval_states = dec_eval["states"][final]
        raw = Ev.score_states(eval_states, data.eval, gts, data.starts, data.frame_counts)
        oracle_assign, oracle_mof = oracle_from_confusion(raw["confusion"])
        arms = {}
        for name, g in groups.items():
            arms[name] = score_arm(G.relabel(eval_states, g), data, gts, Ev)
            arms[name].update(group_summary(g, raw["confusion"], oracle_assign, mass))
        rnd_m = [Ev.score_states(G.relabel(eval_states, g), data.eval, gts, data.starts, data.frame_counts)["metrics"] for g in randoms]
        rnd = dict(MoF=[m["MoF"] for m in rnd_m], F1_50=[m["F1@50"] for m in rnd_m],
                   p95_MoF=float(np.percentile([m["MoF"] for m in rnd_m], C.RANDOM_PERCENTILE)),
                   p95_F1_50=float(np.percentile([m["F1@50"] for m in rnd_m], C.RANDOM_PERCENTILE)))
        for mult in C.MULTIPLIERS:                      # smoothing sensitivity on the frozen 24-state costs
            dm = Dc.decode_ladder(cost["eval"], data.eval, scale.rungs, deadline, multiplier=mult)["states"][final]
            for name, g in groups.items():
                arms[name].setdefault("multipliers", {})[str(mult)] = Ev.score_states(G.relabel(dm, g), data.eval, gts, data.starts, data.frame_counts)["metrics"]
        scores = dict(primary_endpoint_steps=final, arms=arms, random=rnd, raw_states=dict(metrics=raw["metrics"], oracle_many_to_one_mof=oracle_mof,
                      oracle_assignment=oracle_assign), groups={k: v.tolist() for k, v in groups.items()})
        write_json(d / "scores.json", scores)
        np.savez_compressed(d / "plans_fit.npz", **{n: p.astype(np.float16) for n, p in zip(data.fit.names, plans)})
        np.save(d / "prototypes.npy", fit.mu)
        info = dict(base, status="complete", n_fit=len(data.fit_names), n_eval=len(data.eval_names), b0_outer_status=fit.outer_status,
                    b0_outer_iterations=fit.outer_iterations, solver_fit=dec_fit["status_counts"], solver_eval=dec_eval["status_counts"],
                    timings=dict(fit_s=t_fit, decode_s=t_dec, total_s=time.perf_counter() - t0), finished=datetime.now().isoformat(timespec="seconds"))
        write_json(d / "info.json", info)
        print(f"[cell] seed {seed} {split}: complete in {info['timings']['total_s']:.0f}s", flush=True)
        return EXIT_DONE
    except DeadlineExceeded:
        write_json(d / "info.json", dict(base, status="interrupted", seconds=time.perf_counter() - t0))
        print(f"[cell] seed {seed} {split}: INTERRUPTED (not a result)", flush=True)
        return EXIT_INTERRUPTED
    except Tr.InvalidSolve as exc:
        write_json(d / "info.json", dict(base, status="invalid", reason=str(exc), seconds=time.perf_counter() - t0))
        print(f"[cell] seed {seed} {split}: INVALID ({exc})", flush=True)
        return EXIT_INVALID


# --- decision rule and report ---------------------------------------------------------------------------------------

def _diffs(M, a, b, seeds):
    return [(M[(a, s)][0] - M[(b, s)][0], M[(a, s)][1] - M[(b, s)][1]) for s in seeds]


def _bar(x) -> bool:
    return bool(np.mean(x) >= C.BAR and min(x) > 0)


def verdict(M: dict, P95: dict, seeds, complete: bool) -> dict:
    """M[(arm, seed)] = (MoF, F1@50) for arms b0_c and the three groupings; P95[seed] = (p95 MoF, p95 F1@50) of the
    size-matched random groupings. Declared rule of the plan, Section 5."""
    if not complete:
        return dict(verdict="INCONCLUSIVE")
    d = _diffs(M, "group_temporal", "b0_c", seeds)
    a = _bar([x[0] for x in d]) and _bar([x[1] for x in d])
    ctrl = {}
    for name in ("group_appearance", "group_shuffled"):
        dd = _diffs(M, "group_temporal", name, seeds)
        mof, f1 = [x[0] for x in dd], [x[1] for x in dd]
        ctrl[name] = bool((_bar(mof) and np.mean(f1) >= C.FLOOR) or (_bar(f1) and np.mean(mof) >= C.FLOOR))
    dcrit = all(M[("group_temporal", s)][0] > P95[s][0] and M[("group_temporal", s)][1] > P95[s][1] for s in seeds)
    if not a:
        v = "STOP"
    elif not dcrit:
        v = "PRIOR_MASS"
    elif not (ctrl["group_appearance"] and ctrl["group_shuffled"]):
        v = "OVERCOMPLETE_ONLY"
    else:
        v = "TEMPORAL_GROUPING"
    return dict(verdict=v, a_vs_b0=a, b_vs_appearance=ctrl["group_appearance"], c_vs_shuffled=ctrl["group_shuffled"], d_vs_random_p95=dcrit,
                vs_b0_mean=[float(np.mean([x[0] for x in d])), float(np.mean([x[1] for x in d]))])


def b0_reference(seed: int, split: str):
    sc = read(C.reference_cell(seed, split) / "scores.json")
    if not sc:
        return None
    e = sc["eval"][str(sc["primary_endpoint_steps"])]
    return (e["metrics"]["MoF"], e["metrics"]["F1@50"]), 100.0 * float(np.mean(list(e["per_class_recall"].values())))


def finalize(run_dir: Path, smoke: bool, seeds, splits, not_run) -> None:
    cells = {}
    for sp in splits:
        for s in seeds:
            d = cell_dir(run_dir, s, sp)
            info, sc = read(d / "info.json"), read(d / "scores.json")
            cells[(s, sp)] = dict(info=info, scores=sc if info and info.get("status") == "complete" else None)

    def table(split_list):
        M, P95, ok = {}, {}, True
        for s in seeds:
            vals = [(cells.get((s, sp)), b0_reference(s, sp)) for sp in split_list]
            if any(c is None or c["scores"] is None or r is None for c, r in vals):
                ok = False
                continue
            M[("b0_c", s)] = tuple(np.mean([r[0][i] for _, r in vals]) for i in (0, 1))
            for arm in C.ARMS:
                M[(arm, s)] = tuple(np.mean([c["scores"]["arms"][arm]["metrics"][k] for c, _ in vals]) for k in ("MoF", "F1@50"))
            P95[s] = tuple(np.mean([c["scores"]["random"][k] for c, _ in vals]) for k in ("p95_MoF", "p95_F1_50"))
        return M, P95, ok

    ver = read(C.RESULTS / "verification.json")
    ver_ok = bool(ver and ver.get("passed") and ver.get("code_hash") == C.code_hash())
    M, P95, ok = table(["primary"])
    gate = verdict(M, P95, seeds, ok and ver_ok and not smoke and len(seeds) == len(C.SEEDS))
    if smoke:
        gate = dict(verdict="not computed (smoke run)")
    fold_list = [s for s in splits if s.startswith("fold")]
    Mf, Pf, okf = table(fold_list) if fold_list else ({}, {}, False)
    folds = verdict(Mf, Pf, seeds, okf and len(fold_list) == 4 and not smoke) if fold_list and not smoke else None
    L = [f"# Unit-grouping test {run_dir.name}", "", "SMOKE RUN: plumbing only, not evidence." if smoke else "",
         f"M = {C.NUM_STATES} states grouped into C = {C.NUM_ACTIONS}; LARa; held-out evaluation recordings; final solver endpoint; one Hungarian mapping "
         "over the evaluation recordings. `b0_c` = Milestone 1 C-state B0 cells (read, not re-run).", "",
         "## Primary split (mean over seeds in parentheses where shown)", "",
         "| arm | seed | MoF | F1@50 | balanced acc | classes with a group | ARI vs oracle | pred/true segments |", "|---|---|---|---|---|---|---|---|"]
    for s in seeds:
        r = b0_reference(s, "primary")
        if r:
            L.append(f"| b0_c | {s} | {r[0][0]:.1f} | {r[0][1]:.1f} | {r[1]:.1f} | - | - | - |")
        c = cells.get((s, "primary"))
        if c and c["scores"]:
            for arm in C.ARMS:
                a = c["scores"]["arms"][arm]
                L.append(f"| {arm} | {s} | {a['metrics']['MoF']:.1f} | {a['metrics']['F1@50']:.1f} | {a['balanced_accuracy']:.1f} | {a['classes_with_group']} | "
                         f"{a['ari_vs_oracle']:.2f} | {a['pred_over_gt']:.2f} |")
            rn = c["scores"]["random"]
            L.append(f"| group_random (100, size-matched) | {s} | mean {np.mean(rn['MoF']):.1f} / p95 {rn['p95_MoF']:.1f} | mean {np.mean(rn['F1_50']):.1f} / p95 {rn['p95_F1_50']:.1f} | - | - | - | - |")
            rs = c["scores"]["raw_states"]
            L.append(f"| (24 raw states: oracle many-to-one MoF bound) | {s} | {rs['oracle_many_to_one_mof']:.1f} | - | - | - | - | - |")
    L += ["", "## Decision (declared rule, primary split)", "", f"**{gate['verdict']}**", ""]
    for k in ("a_vs_b0", "b_vs_appearance", "c_vs_shuffled", "d_vs_random_p95"):
        if k in gate:
            L.append(f"- {k}: {'holds' if gate[k] else 'fails'}")
    if gate["verdict"] == "INCONCLUSIVE":
        L.append(f"- missing seeds/cells or verification not recorded as passed for this code hash (verification ok: {ver_ok})")
    if folds:
        L += ["", "## Secondary: subject-disjoint folds (descriptive, mean over 4 folds)", "", f"Rule applied descriptively: **{folds['verdict']}**."]
        if gate["verdict"] != folds["verdict"] and gate["verdict"] not in ("INCONCLUSIVE",):
            L.append("- **WARNING: the fold readout disagrees with the primary gate. Report both; neither overrides the other.**")
    mult_rows = []
    for s in seeds:
        c = cells.get((s, "primary"))
        if c and c["scores"]:
            for arm in C.ARMS:
                m = c["scores"]["arms"][arm].get("multipliers", {})
                mult_rows.append(f"| {arm} | {s} | " + " | ".join(f"{m[k]['MoF']:.1f}/{m[k]['F1@50']:.1f}" for k in sorted(m, key=float)) + " |")
    if mult_rows:
        L += ["", "## Smoothing sensitivity (MoF/F1@50 at emission multipliers 0.5 and 2; multiplier 1 is the table above)", "",
              "| arm | seed | x0.5 | x2 |", "|---|---|---|---|"] + mult_rows
    if not_run:
        L += ["", "## Not run", ""] + [f"- seed {n['seed']} {n['split']}: {n['reason']}" for n in not_run]
    L += ["", "Scope: temporal grouping of transport states (symmetric affinity, reversal-invariant), not ordered composition and not grouping of raw tokens. "
          "The oracle is an upper bound on MoF for hard relabellings of the fixed 24 states only. Three seeds on one split are not confidence intervals."]
    (run_dir / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    write_json(run_dir / "gates.json", dict(primary=gate, folds=folds, verification=ver))
    print(f"[report] {run_dir / 'REPORT.md'}: {gate['verdict']}")


# --- driver ---------------------------------------------------------------------------------------------------------

def estimate_cell_seconds() -> float | None:
    """Projected seconds for one primary cell: three times the historical C-state B0 (transport cost grows with the state
    count) plus the multiplier decodes. Folds are about 0.85 of this."""
    from script.compositional_dynamics import run as CR
    try:
        h = CR.historical_b0_seconds()
    except Exception:  # noqa: BLE001
        return None
    return 3.0 * (h["fit"] * C_FIT_RATIO + h["ladder"] * 1.0) + 2 * 3.0 * h["ladder"] * 44 / 439 + 120.0


C_FIT_RATIO = 395 / 439


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="script.unit_grouping.run")
    sub = ap.add_subparsers(dest="mode")
    for m in ("preflight", "dry-run", "smoke", "run"):
        p = sub.add_parser(m)
        p.add_argument("--run-id", default=None)
        p.add_argument("--seeds", default=",".join(map(str, C.SEEDS)))
        p.add_argument("--splits", default="primary")
        p.add_argument("--hours", type=float, default=None)
        p.add_argument("--reuse-completed", action="store_true")
    c = sub.add_parser("cell")
    c.add_argument("--run-id", required=True)
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--split", default="primary")
    c.add_argument("--scale", default="full")
    c.add_argument("--deadline-s", type=float, default=None)
    c.add_argument("--reuse-completed", action="store_true")
    a = ap.parse_args(argv)
    if a.mode is None:
        ap.print_help()
        return 0
    if a.mode == "cell":
        return run_cell(C.RESULTS / a.run_id, a.seed, a.split, a.scale, a.deadline_s, a.reuse_completed)
    seeds = [int(x) for x in a.seeds.split(",") if x]
    splits = [x for x in a.splits.split(",") if x]
    if not set(splits) <= set(C.SPLITS) or not set(seeds) <= set(C.SEEDS):
        print("unknown seed or split", file=sys.stderr)
        return 2
    if a.mode == "preflight":
        problems = [f"missing reference cell {C.reference_cell(s, sp)}" for s in seeds for sp in splits if not (C.reference_cell(s, sp) / "scores.json").exists()]
        from script.compositional_dynamics import data as Dm
        for s in seeds:
            try:
                f = Dm.verify_source(s)
                print(f"seed {s}: {f['n_fit']} fit / {f['n_eval']} eval, source hash {f['identity']['recordings_and_all']}")
            except Exception as exc:  # noqa: BLE001
                problems.append(f"seed {s}: {exc}")
        print(json.dumps(dict(ok=not problems, problems=problems), indent=2))
        return 1 if problems else 0
    if a.mode == "dry-run":
        e = estimate_cell_seconds()
        n_p, n_f = len(seeds) * ("primary" in splits), len(seeds) * len([s for s in splits if s.startswith("fold")])
        print("projected seconds per primary cell:", None if e is None else round(e), "(unmeasured until a full cell completes)")
        if e:
            print(f"projected total: {(n_p * e + n_f * 0.85 * e) / 3600:.1f} h for splits {splits}, seeds {seeds}")
        return 0
    if not a.run_id:
        print("--run-id is required", file=sys.stderr)
        return 2
    smoke = a.mode == "smoke"
    if smoke != ("smoke" in a.run_id):
        print("smoke run ids must contain 'smoke' and full run ids must not", file=sys.stderr)
        return 2
    if smoke:
        seeds = [111]
    run_dir = C.RESULTS / a.run_id
    est = None if smoke else estimate_cell_seconds()
    start, not_run, worst = time.perf_counter(), [], 0
    for sp in splits:
        for s in seeds:
            remaining = None if a.hours is None else a.hours * 3600 - (time.perf_counter() - start)
            if remaining is not None and (remaining <= 0 or (est and est * (1 if sp == "primary" else 0.85) > remaining)):
                not_run.append(dict(seed=s, split=sp, reason=f"deadline: {remaining:.0f}s left"))
                print(f"[run] not launching seed {s} {sp}: projected over the remaining budget", flush=True)
                continue
            cmd = [sys.executable, "-m", "script.unit_grouping.run", "cell", "--run-id", a.run_id, "--seed", str(s), "--split", sp,
                   "--scale", "smoke" if smoke else "full"]
            if remaining is not None:
                cmd += ["--deadline-s", f"{remaining:.0f}"]
            if a.reuse_completed:
                cmd.append("--reuse-completed")
            code = subprocess.run(cmd, cwd=str(C.ROOT)).returncode
            if code:
                worst = max(worst, code)
                not_run.append(dict(seed=s, split=sp, reason=f"child exit code {code}"))
    write_json(run_dir / "manifest.json", dict(run_id=a.run_id, seeds=seeds, splits=splits, smoke=smoke, code_hash=C.code_hash(), hours=a.hours,
                                                finished=datetime.now().isoformat(timespec="seconds")))
    finalize(run_dir, smoke, seeds, splits, not_run)
    return worst


if __name__ == "__main__":
    sys.exit(main())
