"""Residual co-occurrence test on the saved LARa unit-grouping fits (run ug_lara_001): driver, cells and report.

Each cell (seed, split) reuses the saved 24-state prototypes and fitting plans, re-decodes the evaluation recordings under the
frozen prototypes, recomputes the three original groupings (parity check against the saved run), adds the residual grouping and
its own size-matched random partitions, and scores everything with the same functions as the original test."""
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
from script.unit_grouping import residual as RS  # noqa: E402
from script.unit_grouping import run as UG  # noqa: E402

SOURCE_RUN = C.RESULTS / "ug_lara_001"
ARMS = ("group_residual", "group_temporal", "group_shuffled", "group_appearance")


def source_cell(seed: int, split: str) -> Path:
    return UG.cell_dir(SOURCE_RUN, seed, split)


def run_cell(run_dir: Path, seed: int, split: str, smoke: bool) -> int:
    from script.compositional_dynamics import config as CD
    from script.compositional_dynamics import data as Dm, decode as Dc, evaluate as Ev, train as Tr
    t0 = time.perf_counter()
    CD.NUM_STATES = C.NUM_STATES
    src = source_cell(seed, split)
    saved = json.loads((src / "scores.json").read_text())
    scale = CD.SMOKE if smoke else CD.FULL
    data = Dm.load_cell_data(seed, CD.FULL, split)            # the full populations the saved fits used
    eval_names = data.eval_names[:3] if smoke else data.eval_names
    if smoke:
        data.eval = Dm.make_population(eval_names, [data.eval.codes[data.eval.slice_of(r)] for r in range(3)],
                                       [data.eval.lengths[data.eval.slice_of(r)] for r in range(3)], data.eval.hist_len)
    mu = np.load(src / "prototypes.npy")
    z = np.load(src / "plans_fit.npz")
    plans = [z[n].astype(np.float64) for n in data.fit.names]
    plans = [p / p.sum(axis=1, keepdims=True) for p in plans]              # float16 storage: renormalize rows
    lengths = [data.fit.lengths[data.fit.slice_of(r)] for r in range(len(data.fit.names))]
    mass = G.state_mass(plans, lengths)
    A = G.affinity(plans)
    E = RS.expected_shuffled_affinity(plans)
    groups = {"group_residual": G.spectral_groups(RS.residual_affinity(A, E), C.NUM_ACTIONS, seed),
              "group_temporal": G.spectral_groups(A, C.NUM_ACTIONS, seed),
              "group_shuffled": G.spectral_groups(G.affinity(G.shuffle_rows(plans, seed)), C.NUM_ACTIONS, seed),
              "group_appearance": G.appearance_groups(mu, mass, C.NUM_ACTIONS, seed)}
    cost = Tr.b0_costs(data, data.eval, mu)
    final = scale.rungs[-1] if smoke else CD.FULL.rungs[-1]
    dec = Dc.decode_ladder(cost, data.eval, (scale.rungs if smoke else CD.FULL.rungs), None)
    states = dec["states"][final]
    gts = Ev.load_labels(eval_names)
    raw = Ev.score_states(states, data.eval, gts, data.starts, data.frame_counts)
    oracle_assign, oracle_mof = UG.oracle_from_confusion(raw["confusion"])
    arms = {}
    for name, g in groups.items():
        arms[name] = UG.score_arm(G.relabel(states, g), data, gts, Ev)
        arms[name].update(UG.group_summary(g, raw["confusion"], oracle_assign, mass))
    sizes = np.bincount(groups["group_residual"], minlength=C.NUM_ACTIONS)
    rnd = [Ev.score_states(G.relabel(states, g), data.eval, gts, data.starts, data.frame_counts)["metrics"]
           for g in G.size_matched_random(sizes, C.N_RANDOM, seed)]
    random = dict(p95_MoF=float(np.percentile([m["MoF"] for m in rnd], C.RANDOM_PERCENTILE)),
                  p95_F1_50=float(np.percentile([m["F1@50"] for m in rnd], C.RANDOM_PERCENTILE)),
                  mean_MoF=float(np.mean([m["MoF"] for m in rnd])), mean_F1_50=float(np.mean([m["F1@50"] for m in rnd])))
    parity = {name: dict(same_groups=bool(np.array_equal(groups[name], np.asarray(saved["groups"][name]))),
                         saved=saved["arms"][name]["metrics"], recomputed=arms[name]["metrics"],
                         exact=bool(all(saved["arms"][name]["metrics"][k] == arms[name]["metrics"][k] for k in ("MoF", "F1@50"))))
              for name in ("group_temporal", "group_shuffled", "group_appearance")}
    out = dict(arms=arms, random=random, parity=parity, oracle_many_to_one_mof=oracle_mof,
               residual_share=RS.residual_share(A, E), ari_residual_vs_temporal=float(adjusted_rand_score(groups["group_temporal"], groups["group_residual"])),
               groups={k: v.tolist() for k, v in groups.items()}, smoke=smoke, seed=seed, split=split, code_hash=C.code_hash(),
               seconds=time.perf_counter() - t0, finished=datetime.now().isoformat(timespec="seconds"))
    d = run_dir / "cells" / (f"seed{seed}" + ("" if split == "primary" else f"__{split}"))
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "scores.json.tmp"
    tmp.write_text(json.dumps(out, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    os.replace(tmp, d / "scores.json")
    print(f"[cell] seed {seed} {split}: residual MoF {arms['group_residual']['metrics']['MoF']:.1f} F1@50 {arms['group_residual']['metrics']['F1@50']:.1f} "
          f"| residual share {out['residual_share']:.3f} | parity exact {all(p['exact'] for p in parity.values())} ({out['seconds']:.0f}s)", flush=True)
    return 0


def verdict(M: dict, P95: dict, seeds, complete: bool) -> dict:
    """The unit-grouping v2 rule with the residual grouping as the hypothesis arm (plan Section 4)."""
    if not complete:
        return dict(verdict="INCONCLUSIVE")
    Mt = {(("group_temporal" if a == "group_residual" else ("group_temporal_original" if a == "group_temporal" else a)), s): v
          for (a, s), v in M.items()}
    out = UG.verdict(Mt, P95, seeds, True)
    names = {"TEMPORAL_GROUPING": "RESIDUAL_GROUPING"}
    out["verdict"] = names.get(out["verdict"], out["verdict"])
    d = [(M[("group_residual", s)][0] - M[("group_temporal", s)][0], M[("group_residual", s)][1] - M[("group_temporal", s)][1]) for s in seeds]
    out["vs_temporal_mean"] = [float(np.mean([x[0] for x in d])), float(np.mean([x[1] for x in d]))]
    return out


def finalize(run_dir: Path, seeds, splits, smoke: bool) -> None:
    cells = {}
    for sp in splits:
        for s in seeds:
            p = run_dir / "cells" / (f"seed{s}" + ("" if sp == "primary" else f"__{sp}")) / "scores.json"
            cells[(s, sp)] = json.loads(p.read_text()) if p.exists() else None

    def table(split_list):
        M, P95, ok = {}, {}, True
        for s in seeds:
            vals = [(cells.get((s, sp)), UG.b0_reference(s, sp)) for sp in split_list]
            if any(c is None or r is None for c, r in vals):
                ok = False
                continue
            M[("b0_c", s)] = tuple(np.mean([r[0][i] for _, r in vals]) for i in (0, 1))
            for a in ARMS:
                M[(a, s)] = tuple(np.mean([c["arms"][a]["metrics"][k] for c, _ in vals]) for k in ("MoF", "F1@50"))
            P95[s] = tuple(np.mean([c["random"][k] for c, _ in vals]) for k in ("p95_MoF", "p95_F1_50"))
        return M, P95, ok

    ver = json.loads((C.RESULTS / "verification.json").read_text()) if (C.RESULTS / "verification.json").exists() else {}
    ver_ok = bool(ver.get("passed") and ver.get("code_hash") == C.code_hash())
    M, P95, ok = table(["primary"])
    gate = dict(verdict="not computed (smoke run)") if smoke else verdict(M, P95, seeds, ok and ver_ok and len(seeds) == len(C.SEEDS))
    folds = [s for s in splits if s.startswith("fold")]
    Mf, Pf, okf = table(folds) if folds else ({}, {}, False)
    fold_gate = verdict(Mf, Pf, seeds, okf and len(folds) == 4) if folds and not smoke else None
    L = [f"# Residual co-occurrence grouping {run_dir.name}", "", "SMOKE RUN: plumbing only, not evidence." if smoke else "",
         "LARa; the 24 states, prototypes and fitting plans of ug_lara_001, re-decoded on the held-out recordings; groupings into 8 actions; "
         "`b0_c` = Milestone 1 8-state B0.", "",
         "| seed | arm | MoF | F1@50 | balanced acc | ARI vs oracle | classes with a group |", "|---|---|---|---|---|---|---|"]
    for s in seeds:
        c = cells.get((s, "primary"))
        r = UG.b0_reference(s, "primary")
        if r:
            L.append(f"| {s} | b0_c | {r[0][0]:.1f} | {r[0][1]:.1f} | {r[1]:.1f} | - | - |")
        if c:
            for a in ARMS:
                x = c["arms"][a]
                L.append(f"| {s} | {a} | {x['metrics']['MoF']:.1f} | {x['metrics']['F1@50']:.1f} | {x['balanced_accuracy']:.1f} | {x['ari_vs_oracle']:.2f} | {x['classes_with_group']} |")
            L.append(f"| {s} | random (size-matched to residual) | mean {c['random']['mean_MoF']:.1f} / p95 {c['random']['p95_MoF']:.1f} | "
                     f"mean {c['random']['mean_F1_50']:.1f} / p95 {c['random']['p95_F1_50']:.1f} | - | - | - |")
            L.append(f"| {s} | residual share of affinity / ARI residual vs temporal / parity exact | {c['residual_share']:.3f} | {c['ari_residual_vs_temporal']:.2f} | "
                     f"{all(p['exact'] for p in c['parity'].values())} | | |")
    L += ["", "## Decision (declared rule, primary split)", "", f"**{gate['verdict']}**", ""]
    for k in ("a_vs_b0", "b_vs_appearance", "c_vs_shuffled", "d_vs_random_p95"):
        if k in gate:
            L.append(f"- {k}: {'holds' if gate[k] else 'fails'}")
    if "vs_temporal_mean" in gate:
        L.append(f"- residual minus temporal (mean MoF, F1@50): {gate['vs_temporal_mean'][0]:+.2f}, {gate['vs_temporal_mean'][1]:+.2f} (reported, not gated)")
    if gate["verdict"] == "INCONCLUSIVE":
        L.append(f"- missing cells or verification not recorded for this code hash (verification ok: {ver_ok})")
    if fold_gate:
        L += ["", f"Subject-disjoint folds, rule applied descriptively: **{fold_gate['verdict']}**."]
        if fold_gate["verdict"] != gate["verdict"]:
            L.append("- **WARNING: the fold readout disagrees with the primary result.**")
    (run_dir / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    (run_dir / "gates.json").write_text(json.dumps(dict(primary=gate, folds=fold_gate), indent=2, default=str))
    print(f"[report] {run_dir / 'REPORT.md'}: {gate['verdict']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="script.unit_grouping.residual_run")
    sub = ap.add_subparsers(dest="mode")
    for m in ("smoke", "run"):
        p = sub.add_parser(m)
        p.add_argument("--run-id", required=True)
        p.add_argument("--seeds", default=",".join(map(str, C.SEEDS)))
        p.add_argument("--splits", default="primary")
    c = sub.add_parser("cell")
    c.add_argument("--run-id", required=True)
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--split", default="primary")
    c.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.mode is None:
        ap.print_help()
        return 0
    if a.mode == "cell":
        return run_cell(C.RESULTS / a.run_id, a.seed, a.split, a.smoke)
    smoke = a.mode == "smoke"
    if smoke != ("smoke" in a.run_id):
        print("smoke run ids must contain 'smoke' and full run ids must not", file=sys.stderr)
        return 2
    seeds = [111] if smoke else [int(x) for x in a.seeds.split(",") if x]
    splits = ["primary"] if smoke else [x for x in a.splits.split(",") if x]
    run_dir = C.RESULTS / a.run_id
    bad = 0
    for sp in splits:
        for s in seeds:
            cmd = [sys.executable, "-m", "script.unit_grouping.residual_run", "cell", "--run-id", a.run_id, "--seed", str(s), "--split", sp]
            if smoke:
                cmd.append("--smoke")
            bad = max(bad, subprocess.run(cmd, cwd=str(C.ROOT)).returncode)
    finalize(run_dir, seeds, splits, smoke)
    return bad


if __name__ == "__main__":
    sys.exit(main())
