"""
Experiment 2 -- one label-free segmentation method: exact linear-fit dynamic programme.

Per fold: fit scaler on fitting recordings; choose lambda on UNLABELLED fitting recordings so
the DP event rate matches the one-second grid's rate; segment every recording with that lambda;
compare Fixed (reused from Experiment 1) / DP / DP-random (matched count+durations) / original SMQ.

Usage: python script/seg/exp2.py --dataset lara
"""
import argparse
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

RAW = ROOT / "results" / "seg" / "raw"
MIN_SEC, MAX_SEC = 0.25, 10.0
RATE_TOL = 0.02
LAMBDA_SUBSAMPLE = None      # None = all fitting recordings (see PROTOCOL.md deviation D1)
RANDOM_SEED_BASE_DP = 2000


def bounds_frames(fps):
    return int(np.ceil(MIN_SEC * fps)), int(np.floor(MAX_SEC * fps))


def standardised(z, mu, sd):
    return (z - mu) / sd


def select_lambda(costs, T_list, fps, W, Lmin, Lmax):
    """Bisection on log(lambda) so that segments/second equals the one-second grid's rate."""
    secs = sum(T_list) / fps
    target = sum(int(np.ceil(t / W)) for t in T_list) / secs

    def rate(lam):
        n = 0
        for c in costs:
            e, _ = P.dp_segment(c, Lmin, Lmax, lam)
            n += len(e) - 1
        return n / secs

    lo, hi = 1e-4, 1e4
    trace = []
    r_lo, r_hi = rate(lo), rate(hi)
    assert r_lo >= target >= r_hi, (r_lo, target, r_hi)
    best = None
    for _ in range(60):
        mid = float(np.sqrt(lo * hi))
        r = rate(mid)
        trace.append((mid, r))
        if best is None or abs(r - target) < abs(best[1] - target):
            best = (mid, r)
        if abs(r / target - 1) <= RATE_TOL:
            break
        if r > target:
            lo = mid            # too many segments -> larger penalty
        else:
            hi = mid
    lam, r = best
    return lam, dict(target_rate=target, achieved_rate_subsample=r, matched=bool(abs(r / target - 1) <= RATE_TOL),
                     iterations=len(trace))


def segment_recordings(data, recs, mu, sd, Lmin, Lmax, lam):
    edges = {}
    diag = {}
    for r in recs:
        c = P.linear_fit_cost_end_indexed(standardised(data.z[r], mu, sd), Lmin, Lmax)
        e, info = P.dp_segment(c, Lmin, Lmax, lam)
        P.check_partition(e, int(data.T[r]))
        d = np.diff(e)
        edges[r] = e
        diag[r] = dict(fallback_single=bool(info["fallback_single"]),
                       n_segments=int(len(d)), n_at_min=int((d == Lmin).sum()), n_at_max=int((d == Lmax).sum()),
                       fit_cost=float(P.partition_cost(c, e, Lmin)) if not info["fallback_single"] else float("nan"))
    return edges, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    t0 = time.time()
    ds = args.dataset
    data = PL.Data(ds)
    fold_of_rec, nf = data.make_folds()
    anchors = data.make_anchors()
    e1 = pickle.load(open(RAW / f"exp1_{ds}.pkl", "rb"))
    assert np.array_equal(e1["meta"]["fold_of_rec"], fold_of_rec), "fold assignment differs from Experiment 1"
    from script.seg.exp1 import build_partitions  # noqa: E402
    fixed, oracle, _ = build_partitions(data)
    Lmin, Lmax = bounds_frames(data.fps)
    print(f"{ds}: Lmin={Lmin} Lmax={Lmax} folds={nf}", flush=True)

    results = {"dp": {s: {} for s in PL.KMEANS_SEEDS}}
    for r_ in range(PL.N_RANDOM):
        results[f"dprand{r_}"] = {s: {} for s in PL.KMEANS_SEEDS}
    dp_edges_all = [None] * data.n
    dp_diag = [None] * data.n
    fold_info = {}

    for fold in range(nf):
        ev = np.flatnonzero(fold_of_rec == fold)
        fit = np.flatnonzero(fold_of_rec != fold)
        mu, sd = R.fit_scaler([data.z[r] for r in fit])
        sub = fit if LAMBDA_SUBSAMPLE is None else fit[np.linspace(0, len(fit) - 1, min(LAMBDA_SUBSAMPLE, len(fit))).astype(int)]
        costs = [P.linear_fit_cost_end_indexed(standardised(data.z[r], mu, sd), Lmin, Lmax) for r in sub]
        lam, linfo = select_lambda(costs, [int(data.T[r]) for r in sub], data.fps, data.W, Lmin, Lmax)   # unlabelled only
        del costs
        edges, diag = segment_recordings(data, list(fit) + list(ev), mu, sd, Lmin, Lmax, lam)
        secs_fit = sum(data.T[r] for r in fit) / data.fps
        secs_ev = sum(data.T[r] for r in ev) / data.fps
        linfo.update(lam=lam,
                     grid_rate_fit=sum(int(np.ceil(data.T[r] / data.W)) for r in fit) / secs_fit,
                     dp_rate_fit=sum(diag[r]["n_segments"] for r in fit) / secs_fit,
                     dp_rate_eval=sum(diag[r]["n_segments"] for r in ev) / secs_ev,
                     grid_rate_eval=sum(int(np.ceil(data.T[r] / data.W)) for r in ev) / secs_ev)
        fold_info[fold] = linfo
        print(f"  fold {fold}: lambda={lam:.4g} {linfo}  ({time.time() - t0:.0f}s)", flush=True)
        for r in ev:
            dp_edges_all[r] = edges[r]; dp_diag[r] = diag[r]

        cond_edges = {"dp": [edges.get(i) for i in range(data.n)]}
        for r_ in range(PL.N_RANDOM):
            cond_edges[f"dprand{r_}"] = [
                P.random_edges(P.durations(edges[i]), np.random.default_rng([RANDOM_SEED_BASE_DP + r_, i]))
                if i in edges else None for i in range(data.n)]
        for cname, ed in cond_edges.items():
            out = PL.fit_and_score_condition(data, ed, fit, ev, mu, sd, anchors, PL.KMEANS_SEEDS, do_f1=True)
            for s, res in out.items():
                PL.merge_fold_results(results[cname][s], res, ev, data.n, fold)
        print(f"  fold {fold} scored ({time.time() - t0:.0f}s)", flush=True)

    meta = dict(dataset=ds, fold_of_rec=fold_of_rec, names=data.names, group=data.group, T=data.T, fps=data.fps,
                Lmin=Lmin, Lmax=Lmax, fold_info=fold_info, dp_diag=dp_diag,
                dp_durations=[np.diff(e) for e in dp_edges_all], dp_edges=dp_edges_all,
                min_sec=MIN_SEC, max_sec=MAX_SEC, rate_tol=RATE_TOL)
    with open(RAW / f"exp2_{ds}.pkl", "wb") as f:
        pickle.dump(dict(meta=meta, results=results), f, protocol=4)
    print(f"saved exp2_{ds}.pkl total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
