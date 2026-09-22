"""
Experiment 1 -- oracle boundary diagnostic (fixed vs oracle vs duration-matched random).

Usage:  python script/seg/exp1.py --dataset lara [--weighting duration]
Writes  results/seg/raw/exp1_<dataset>[_dur].pkl
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from script.seg import partition as P  # noqa: E402
from script.seg import pipeline as PL  # noqa: E402
from script.seg import represent as R  # noqa: E402

OUT = ROOT / "results" / "seg" / "raw"
OUT.mkdir(parents=True, exist_ok=True)


def build_partitions(data):
    fixed = [P.fixed_edges(int(t), data.W) for t in data.T]
    oracle = [P.oracle_edges(l) for l in data.labels]
    rand = []
    for r_ in range(PL.N_RANDOM):
        rand.append([P.random_edges(P.durations(oracle[i]), np.random.default_rng([PL.RANDOM_SEED_BASE + r_, i]))
                     for i in range(data.n)])
    return fixed, oracle, rand


def partition_checks(data, fixed, oracle, rand):
    chk = dict(n_recordings=data.n)
    for name, ed in (("fixed", fixed), ("oracle", oracle)):
        for i in range(data.n):
            P.check_partition(ed[i], int(data.T[i]))
            P.frame_segment_ids(ed[i], int(data.T[i]))
    for r_ in range(len(rand)):
        for i in range(data.n):
            P.check_partition(rand[r_][i], int(data.T[i]))
            assert sorted(P.durations(rand[r_][i])) == sorted(P.durations(oracle[i])), "duration multiset differs"
            assert len(rand[r_][i]) == len(oracle[i]), "segment count differs"
    chk["all_partitions_cover_each_frame_exactly_once"] = True
    chk["random_and_oracle_share_count_and_duration_multiset"] = True
    chk["oracle_matches_gt_run_count"] = all(
        len(oracle[i]) - 1 == len(np.flatnonzero(np.diff(data.labels[i]) != 0)) + 1 for i in range(data.n))
    # null informativeness per recording
    ni = []
    for i in range(data.n):
        ni.append(P.null_informativeness(P.durations(oracle[i]), [rand[r_][i] for r_ in range(len(rand))], oracle[i]))
    chk["null_informativeness"] = ni
    n_lt4 = sum(x["n_segments"] < 4 for x in ni)
    n_single = sum(x["n_segments"] == 1 for x in ni)
    n_ident = sum(x["frac_randomisations_identical_to_oracle"] >= 0.5 for x in ni)
    n_lowcv = sum(x["duration_cv"] < 0.25 for x in ni)
    chk["null_summary"] = dict(recordings=data.n, single_segment=n_single, fewer_than_4_segments=n_lt4,
                               majority_of_randomisations_identical_to_oracle=n_ident,
                               duration_cv_below_025=n_lowcv)
    return chk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--weighting", default="segment", choices=["segment", "duration"])
    args = ap.parse_args()
    t0 = time.time()
    data = PL.Data(args.dataset)
    fold_of_rec, nf = data.make_folds()
    anchors = data.make_anchors()
    fixed, oracle, rand = build_partitions(data)
    chk = partition_checks(data, fixed, oracle, rand)
    chk["anchors_sha256"] = PL.anchors_hash(anchors, data)
    chk["n_anchors"] = int(sum(len(a) for a in anchors))
    print(f"{args.dataset}: n={data.n} folds={nf} anchors={chk['n_anchors']} null={chk['null_summary']}", flush=True)

    conds = {"fixed": fixed, "oracle": oracle}
    for r_ in range(PL.N_RANDOM):
        conds[f"random{r_}"] = rand[r_]
    results = {c: {s: {} for s in PL.KMEANS_SEEDS} for c in conds}
    results["smq"] = {}

    for fold in range(nf):
        ev = np.flatnonzero(fold_of_rec == fold)
        fit = np.flatnonzero(fold_of_rec != fold)
        assert not set(ev) & set(fit)
        if data.ds == "lara":
            assert not set(data.group[ev]) & set(data.group[fit]), "participant leakage between fit and eval"
        mu, sd = R.fit_scaler([data.z[r] for r in fit])          # frame-level, fitting recordings only
        for cname, ed in conds.items():
            out = PL.fit_and_score_condition(data, ed, fit, ev, mu, sd, anchors, PL.KMEANS_SEEDS,
                                             do_f1=(cname == "fixed"), weighting=args.weighting)
            for s, res in out.items():
                PL.merge_fold_results(results[cname][s], res, ev, data.n, fold)
        sm = PL.score_original_smq(data, fit, ev, anchors, do_f1=True)
        PL.merge_fold_results(results["smq"], sm, ev, data.n, fold)
        print(f"  fold {fold} done ({time.time() - t0:.0f}s)", flush=True)

    # implementation check: identical anchor query/gallery populations in every condition
    ref = results["fixed"][PL.KMEANS_SEEDS[0]]
    same = True
    for c in results:
        rs = results[c] if c == "smq" else results[c][PL.KMEANS_SEEDS[0]]
        for key in ("ca_q_n", "ca_cont_n"):
            if key in rs and key in ref:
                same &= bool(np.allclose(rs[key], ref[key]))
    chk["common_anchor_identical_queries_across_conditions"] = bool(same)
    assert same, "common-anchor query populations differ between conditions"

    meta = dict(dataset=args.dataset, weighting=args.weighting, fold_of_rec=fold_of_rec, n_folds=nf,
                names=data.names, group=data.group, group_kind=data.group_kind, T=data.T, fps=data.fps,
                class_names=data.class_names, labels_n_frames=[int(len(l)) for l in data.labels],
                checks=chk, none_class=data.none_class)
    suffix = "" if args.weighting == "segment" else "_dur"
    with open(OUT / f"exp1_{args.dataset}{suffix}.pkl", "wb") as f:
        pickle.dump(dict(meta=meta, results=results), f, protocol=4)
    print(f"saved exp1_{args.dataset}{suffix}.pkl  total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
