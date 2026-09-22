"""
Summaries with paired resampling for Experiment 1 (and shared helpers for Experiment 2).

  python script/seg/summarize.py exp1 --dataset lara
Writes results/seg/exp1_summary_<dataset>.json and results/seg/per_recording_exp1_<dataset>.csv
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from script.seg import stats as S  # noqa: E402

RAW = ROOT / "results" / "seg" / "raw"
OUT = ROOT / "results" / "seg"

RETR_SETS = ["ca", "seg80", "seg00"]


def load(name):
    with open(RAW / name, "rb") as f:
        return pickle.load(f)


def condition_stacks(results, conds):
    st = {}
    for c in conds:
        if c == "random":
            st[c] = S.stack_condition([results[f"random{r}"] for r in range(sum(k.startswith("random") for k in results))])
        else:
            st[c] = S.stack_condition(results[c])
    return st


def member_axis_summary(x, kind, n_seeds=3):
    """x: (B+1, M). point estimate = mean over members at draw 0; per-draw value = mean over members."""
    return x.mean(1)


def metrics_for(st_c, W, fold_of_rec, retr_classes, do_f1=False, mof_kinds=True):
    """All metrics of one condition for every draw: dict name -> (B+1, M)."""
    out = {}
    for setname in RETR_SETS:
        for mode in ("cont", "q"):
            pre = f"{setname}_{mode}"
            if f"{pre}_S" not in st_c:
                continue
            r = S.cb_retrieval(W, st_c, pre, retr_classes[setname] if setname == "ca" else S.eval_classes(st_c, pre))
            for k in ("p", "chance", "lift"):
                out[f"{pre}_{k}"] = r[k]
    out.update({f"cont_{k}": v for k, v in S.contingency_metrics(W, st_c, fold_of_rec).items()})
    for kind, name in (("bcode", "bcode"), ("bcut", "bcut")):
        for j, tol in enumerate((0.5, 1.0)):
            b = S.boundary_metrics(W, st_c, name)
            for k in ("P", "R", "F1"):
                out[f"{name}_{k}_tol{tol}"] = b[k][..., j]
    if do_f1 and "f1_h" in st_c:
        for tag in ("h", "t"):
            f = S.f1_metrics(W, st_c, tag)
            for k, v in f.items():
                out[f"{k}_{tag}"] = v
    return out


def all_metrics(res_all, meta, conds, W, do_f1):
    fold = meta["fold_of_rec"]
    st = condition_stacks(res_all, conds)
    # common-anchor classes: identical for all conditions (anchors and labels are shared)
    retr_classes = {"ca": S.eval_classes(st["fixed"], "ca_q")}
    return {c: metrics_for(st[c], W, fold, retr_classes, do_f1=do_f1) for c in conds}, st, retr_classes


def cell(x, kind="mean"):
    """(point, lo, hi) of a (B+1,) array."""
    return S.ci(x)


def summarize_dataset(ds, weighting="segment"):
    suffix = "" if weighting == "segment" else "_dur"
    d = load(f"exp1_{ds}{suffix}.pkl")
    meta, res = d["meta"], d["results"]
    conds = ["fixed", "oracle", "random", "smq"]
    W = S.make_weights(meta["group"])
    M_, st, classes = all_metrics(res, meta, conds, W, do_f1=True)
    summary = dict(dataset=ds, weighting=weighting, group_kind=meta["group_kind"],
                   n_groups=int(len(np.unique(meta["group"]))), n_recordings=len(meta["names"]),
                   n_draws=S.B_DRAWS, class_names=meta["class_names"], checks={k: v for k, v in meta["checks"].items()
                                                                              if k != "null_informativeness"})
    tab = {}
    for c in conds:
        tab[c] = {}
        for k, x in M_[c].items():
            m = x.mean(1)                                   # mean over seeds/randomisations, per draw
            p, lo, hi = S.ci(m)
            entry = dict(point=p, lo=lo, hi=hi)
            # separate variability sources at the point sample
            pm = x[0]
            if c == "random":
                per_rand = pm.reshape(10, -1).mean(1)       # (R,) seed-averaged
                entry["sd_across_randomisations"] = float(per_rand.std(ddof=1))
                entry["sd_across_kmeans_seeds"] = float(pm.reshape(10, -1).std(1, ddof=1).mean())
            elif c != "smq":
                entry["sd_across_kmeans_seeds"] = float(pm.std(ddof=1))
            tab[c][k] = entry
    summary["conditions"] = tab

    # paired contrasts
    contrasts = [("oracle", "fixed"), ("oracle", "random"), ("random", "fixed")]
    con = {}
    for a, b in contrasts:
        con[f"{a}-{b}"] = {}
        for k in M_[a]:
            if k in M_[b]:
                dlt = M_[a][k].mean(1) - M_[b][k].mean(1)
                p, lo, hi = S.ci(dlt)
                con[f"{a}-{b}"][k] = dict(point=p, lo=lo, hi=hi)
    summary["paired_contrasts"] = con

    # informative-null subset sensitivity (recordings with >=4 oracle segments and duration CV >= 0.25)
    ni = d["meta"]["checks"]["null_informativeness"]
    sub = np.array([(x["n_segments"] >= 4 and x["duration_cv"] >= 0.25) for x in ni], dtype=float)
    summary["informative_null_subset"] = dict(n_recordings=int(sub.sum()))
    if sub.sum() >= 10:
        Ws = S.make_weights(meta["group"], subset=sub)
        Ms, _, _ = all_metrics(res, meta, ["fixed", "oracle", "random"], Ws, do_f1=False)
        sc = {}
        for a, b in contrasts:
            sc[f"{a}-{b}"] = {}
            for k in ("ca_cont_p", "ca_q_p", "ca_cont_lift", "ca_q_lift", "cont_nmi", "cont_mof_transfer"):
                dlt = Ms[a][k].mean(1) - Ms[b][k].mean(1)
                p, lo, hi = S.ci(dlt)
                sc[f"{a}-{b}"][k] = dict(point=p, lo=lo, hi=hi)
        summary["informative_null_subset"]["paired_contrasts"] = sc

    # per-class views (point sample)
    pc = {}
    for c in conds:
        pc[c] = {}
        for setname in RETR_SETS:
            for mode in ("cont", "q"):
                pre = f"{setname}_{mode}"
                if f"{pre}_S" not in st[c]:
                    continue
                Sv = st[c][f"{pre}_S"].sum(0).mean(0); nv = st[c][f"{pre}_n"].sum(0).mean(0)
                pc[c][pre] = dict(n_queries=nv.tolist(),
                                  precision=[float(a / b) if b > 0 else None for a, b in zip(Sv, nv)],
                                  chance=[float(a / b) if b > 0 else None
                                          for a, b in zip(st[c][f"{pre}_ch"].sum(0).mean(0), nv)])
    summary["per_class"] = pc

    # segment count / duration descriptives for each partition (seconds)
    fps = meta["fps"]
    for c in ("fixed", "oracle", "random"):
        key = "nseg"
        arr = st[c][key][:, :, ] if False else st[c]["nseg"]
        summary.setdefault("descriptives", {})[c] = dict(
            segments_per_recording_mean=float(arr.mean()),
            segments_per_second=float(arr.mean(1).sum() / (np.array(meta["T"]).sum() / fps)))
    (OUT / f"exp1_summary_{ds}{suffix}.json").write_text(json.dumps(summary, indent=1, default=float))

    # per-recording CSV (machine readable; point sample, means over seeds/randomisations)
    import csv
    rows = []
    for i, name in enumerate(meta["names"]):
        for c in ("fixed", "oracle", "random", "smq"):
            r = dict(dataset=ds, recording=name, group=int(meta["group"][i]), fold=int(meta["fold_of_rec"][i]),
                     condition=c, n_frames=int(meta["T"][i]))
            a = st[c]
            r["n_segments"] = float(a["nseg"][i].mean())
            r["frames_correct_transfer_map"] = float(a["tcorrect"][i].mean())
            r["frames_correct_hungarian_fold_map"] = float(a["hcorrect"][i].mean())
            for setname in RETR_SETS:
                for mode in ("cont", "q"):
                    pre = f"{setname}_{mode}"
                    if f"{pre}_S" in a:
                        r[f"{pre}_sum_precision"] = float(a[f"{pre}_S"][i].mean(0).sum())
                        r[f"{pre}_n_queries"] = float(a[f"{pre}_n"][i].mean(0).sum())
            rows.append(r)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("dataset", "recording", "group", "fold", "condition", "n_frames"), k))
    with open(OUT / f"per_recording_exp1_{ds}{suffix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["exp1"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--weighting", default="segment")
    a = ap.parse_args()
    s = summarize_dataset(a.dataset, a.weighting)
    for c, t in s["conditions"].items():
        print(c, {k: round(v["point"], 3) for k, v in t.items() if k in (
            "ca_cont_p", "ca_q_p", "ca_cont_chance", "cont_nmi", "cont_mof_transfer", "cont_mof_hungarian")})
    for k, t in s["paired_contrasts"].items():
        print(k, {m: (round(v["point"], 3), round(v["lo"], 3), round(v["hi"], 3)) for m, v in t.items()
                  if m in ("ca_cont_p", "ca_q_p", "cont_nmi")})
