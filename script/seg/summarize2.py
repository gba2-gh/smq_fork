"""
Summaries with paired resampling for Experiment 2 (label-free DP segmentation).

  python script/seg/summarize2.py --dataset lara
Writes results/seg/exp2_summary_<dataset>.json and results/seg/per_recording_exp2_<dataset>.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from script.seg import pipeline as PL  # noqa: E402
from script.seg import stats as S  # noqa: E402
from script.seg.summarize import OUT, load, metrics_for  # noqa: E402

N_RANDOM = PL.N_RANDOM


def summarize_exp2(ds):
    e1, e2 = load(f"exp1_{ds}.pkl"), load(f"exp2_{ds}.pkl")
    meta, m2 = e1["meta"], e2["meta"]
    assert np.array_equal(meta["fold_of_rec"], m2["fold_of_rec"])
    r1, r2 = e1["results"], e2["results"]
    W = S.make_weights(meta["group"])           # same draws as Experiment 1 (same group order and seed)
    stacks = {
        "smq": S.stack_condition(r1["smq"]),
        "fixed": S.stack_condition(r1["fixed"]),
        "oracle": S.stack_condition(r1["oracle"]),
        "dp": S.stack_condition(r2["dp"]),
        "dp_random": S.stack_condition([r2[f"dprand{r}"] for r in range(N_RANDOM)]),
    }
    classes = {"ca": S.eval_classes(stacks["fixed"], "ca_q")}
    fold = meta["fold_of_rec"]
    M_ = {c: metrics_for(stacks[c], W, fold, classes, do_f1=(c != "oracle")) for c in stacks}
    summary = dict(dataset=ds, group_kind=meta["group_kind"], n_groups=int(len(np.unique(meta["group"]))),
                   n_recordings=len(meta["names"]), n_draws=S.B_DRAWS, Lmin_frames=m2["Lmin"], Lmax_frames=m2["Lmax"],
                   folds=m2["fold_info"])
    tab = {}
    for c in stacks:
        tab[c] = {}
        for k, x in M_[c].items():
            p, lo, hi = S.ci(x.mean(1))
            e = dict(point=p, lo=lo, hi=hi)
            pm = x[0]
            if c == "dp_random":
                e["sd_across_randomisations"] = float(pm.reshape(N_RANDOM, -1).mean(1).std(ddof=1))
                e["sd_across_kmeans_seeds"] = float(pm.reshape(N_RANDOM, -1).std(1, ddof=1).mean())
            elif c != "smq":
                e["sd_across_kmeans_seeds"] = float(pm.std(ddof=1))
            tab[c][k] = e
    summary["conditions"] = tab
    pairs = [("dp", "fixed"), ("dp", "dp_random"), ("dp_random", "fixed"), ("dp", "smq"), ("fixed", "smq"),
             ("oracle", "fixed")]
    con = {}
    for a, b in pairs:
        con[f"{a}-{b}"] = {}
        for k in M_[a]:
            if k in M_[b]:
                p, lo, hi = S.ci(M_[a][k].mean(1) - M_[b][k].mean(1))
                con[f"{a}-{b}"][k] = dict(point=p, lo=lo, hi=hi)
    summary["paired_contrasts"] = con

    # ---- continuation rule (PROTOCOL.md section 5)
    def rule(tag):
        d, dr, rf = con["dp-fixed"], con["dp-dp_random"], con["dp_random-fixed"]
        f1, ed = d[f"F1_50_{tag}"], d[f"Edit_{tag}"]
        mof = d["cont_mof_hungarian" if tag == "h" else "cont_mof_transfer"]
        a = f1["point"] >= 2.0 and f1["lo"] > 0
        b = ed["point"] >= 0 and 100 * mof["point"] >= -1.0
        c1 = rf[f"F1_50_{tag}"]["point"] < 0.5 * f1["point"]
        c2 = dr[f"F1_50_{tag}"]["lo"] > 0
        return dict(dF1_50=f1, dEdit=ed,
                    dMoF_points=dict(point=100 * mof["point"], lo=100 * mof["lo"], hi=100 * mof["hi"]),
                    dp_random_minus_fixed_F1_50=rf[f"F1_50_{tag}"], dp_minus_dp_random_F1_50=dr[f"F1_50_{tag}"],
                    a_gain_ge_2pts_and_ci_excludes_0=bool(a), b_edit_ge_0_and_mof_ge_minus1=bool(b),
                    c_random_explains_less_than_half=bool(c1), c_dp_beats_random_ci_excludes_0=bool(c2),
                    ADVANCE=bool(a and b and c1 and c2))
    summary["continuation_rule"] = dict(hungarian_mapping=rule("h"), train_fitted_mapping=rule("t"))

    # ---- fraction of the oracle advantage recovered (common-anchor retrieval, NMI)
    rec = {}
    for k in ("ca_cont_lift", "ca_q_lift", "cont_nmi", "cont_mof_transfer"):
        g_dp, g_or = con["dp-fixed"][k], con["oracle-fixed"][k]
        rec[k] = dict(dp_minus_fixed=g_dp, oracle_minus_fixed=g_or,
                      ratio_point=(g_dp["point"] / g_or["point"]) if abs(g_or["point"]) > 1e-9 else None)
    summary["oracle_advantage_recovered_by_dp"] = rec

    # ---- descriptive duration / saturation (no success criterion)
    fps = meta["fps"]
    T = np.array(meta["T"])
    dur = np.concatenate([np.asarray(d) for d in m2["dp_durations"]]) / fps
    diag = m2["dp_diag"]
    nseg = sum(d["n_segments"] for d in diag)
    labels = PL.Data(ds).labels
    gt_d = np.concatenate([np.diff(np.flatnonzero(np.concatenate([[True], np.diff(l) != 0, [True]]))) for l in labels]) / fps
    summary["descriptives"] = dict(
        dp_segments=int(nseg), dp_rate_per_s=float(nseg / (T.sum() / fps)),
        dp_duration_s=dict(mean=float(dur.mean()), median=float(np.median(dur)),
                           p10=float(np.percentile(dur, 10)), p90=float(np.percentile(dur, 90))),
        gt_duration_s=dict(mean=float(gt_d.mean()), median=float(np.median(gt_d)),
                           p10=float(np.percentile(gt_d, 10)), p90=float(np.percentile(gt_d, 90))),
        frac_segments_at_min_bound=float(sum(d["n_at_min"] for d in diag) / nseg),
        frac_segments_at_max_bound=float(sum(d["n_at_max"] for d in diag) / nseg),
        frac_recordings_fallback_single=float(np.mean([d["fallback_single"] for d in diag])),
        lambda_per_fold={int(k): v["lam"] for k, v in m2["fold_info"].items()},
        rate_matched_all_folds=bool(all(v["matched"] for v in m2["fold_info"].values())))
    (OUT / f"exp2_summary_{ds}.json").write_text(json.dumps(summary, indent=1, default=float))

    rows = []
    for i, name in enumerate(meta["names"]):
        for c in ("smq", "fixed", "dp", "dp_random"):
            a = stacks[c]
            r = dict(dataset=ds, recording=name, group=int(meta["group"][i]), fold=int(meta["fold_of_rec"][i]),
                     condition=c, n_frames=int(T[i]), n_segments=float(a["nseg"][i].mean()),
                     frames_correct_hungarian_fold_map=float(a["hcorrect"][i].mean()),
                     frames_correct_transfer_map=float(a["tcorrect"][i].mean()),
                     edit_hungarian=float(a["edit_h"][i].mean()), edit_transfer=float(a["edit_t"][i].mean()))
            for j, ov in enumerate((10, 25, 50)):
                for tag in ("h", "t"):
                    f = a[f"f1_{tag}"][i]                                    # (M, 3, 3)
                    r[f"F1_{ov}_tp_{tag}"] = float(f[:, j, 0].mean())
                    r[f"F1_{ov}_fp_{tag}"] = float(f[:, j, 1].mean())
                    r[f"F1_{ov}_fn_{tag}"] = float(f[:, j, 2].mean())
            for j, tol in enumerate((0.5, 1.0)):
                for nm in ("bcode", "bcut"):
                    for q, lab in enumerate(("tp", "fp", "fn")):
                        r[f"{nm}_{lab}_tol{tol}"] = float(a[nm][i][:, j, q].mean())
            if c == "dp":
                r["dp_n_at_min"], r["dp_n_at_max"] = diag[i]["n_at_min"], diag[i]["n_at_max"]
            rows.append(r)
    keys = sorted({k for r in rows for k in r},
                  key=lambda k: (k not in ("dataset", "recording", "group", "fold", "condition", "n_frames"), k))
    with open(OUT / f"per_recording_exp2_{ds}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    s = summarize_exp2(a.dataset)
    for c, t in s["conditions"].items():
        print(c, {k: round(v["point"], 2) for k, v in t.items() if k in (
            "F1_50_h", "F1_25_h", "F1_10_h", "Edit_h", "cont_mof_hungarian", "ca_cont_p", "ca_q_p")})
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk.startswith(("a_", "b_", "c_", "ADV"))}
                      for k, v in s["continuation_rule"].items()}, indent=1))
    print(s["descriptives"])
