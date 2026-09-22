"""
Temporal-error decomposition for SMQ.  Post-hoc only: everything here runs on
the cached code sequences, so no retraining and no GPU.

It answers "where do the temporal errors come from" by separating four
mechanisms that the published metrics pool together:

D1  Patch-grid oracle.  Majority-vote the GT labels inside each patch and score
    that against the GT.  This is the best score ANY method emitting one label
    per patch can reach.  SMQ's distance below it is the error that is NOT
    caused by the patch grid.

D2  Boundary budget.  For every predicted boundary, distance to the nearest GT
    boundary; and for every GT boundary, distance to the nearest predicted one.
    Split into: on-grid hits (within half a patch -- as close as the grid
    permits), spurious predicted boundaries, and missed GT boundaries.

D3  Flicker.  Fraction of predicted runs exactly one patch long, and how many of
    those are "sandwiched" (identical code immediately before and after) --
    pure quantization flicker that a mode filter removes.  Then actually apply a
    mode filter over the patch-level code sequence at several widths and
    re-score, to measure how much is recoverable with no retraining at all.

D4  Duration law.  SMQ assigns each patch independently, so run lengths should
    be geometric.  Compare the predicted self-transition probability and the
    resulting geometric law against the GT duration distribution.

D5  Prior mismatch.  Code-usage distribution vs GT class-prior distribution.
    Hungarian matching is 1:1, so a code that fires twice as often as its
    matched class costs MoF no matter how well-placed its boundaries are.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model.eval_utils import create_correspondences, edit_score, f_score
from script.repro.e1_jsd import pair_jsd, activity_of, frame_weighted, patch_quantize

OVERLAPS = [.1, .25, .5]


def rle(a):
    a = np.asarray(a)
    ch = np.flatnonzero(a[1:] != a[:-1]) + 1
    b = np.concatenate(([0], ch, [len(a)]))
    return np.diff(b), a[b[:-1]], b[:-1]


def boundaries(a):
    a = np.asarray(a)
    return np.flatnonzero(a[1:] != a[:-1]) + 1


def score(gt_list, mapped_list):
    correct = total = 0
    edit = 0.0
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
    for gt, pm in zip(gt_list, mapped_list):
        correct += int((gt == pm).sum())
        total += len(gt)
        edit += edit_score(pm, gt)
        for i, o in enumerate(OVERLAPS):
            a, b, c = f_score(pm, gt, o)
            tp[i] += a; fp[i] += b; fn[i] += c
    f1 = []
    for i in range(3):
        p = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) else 0.0
        r = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) else 0.0
        f1.append((2 * p * r / (p + r) if (p + r) else 0.0) * 100)
    return dict(MoF=round(100 * correct / total, 2), Edit=round(edit / len(gt_list), 2),
                **{f"F1@{int(o*100)}": round(v, 2) for o, v in zip(OVERLAPS, f1)})


def global_map(gts, prs):
    _, pr2gt = create_correspondences(np.concatenate(gts), np.concatenate(prs),
                                      mapping=True)
    return [np.vectorize(pr2gt.get)(p) for p in prs]


def mode_filter_patches(pred, patch, width):
    """Mode filter over the PATCH-level code sequence, expanded back to frames."""
    codes = pred[::patch]
    n = len(codes)
    if width <= 1 or n == 0:
        return pred.copy()
    out = codes.copy()
    h = width // 2
    for i in range(n):
        w = codes[max(0, i - h): min(n, i + h + 1)]
        vals, cnt = np.unique(w, return_counts=True)
        best = vals[cnt == cnt.max()]
        # tie -> keep the current code, so the filter only ever removes flicker
        out[i] = codes[i] if codes[i] in best else best[0]
    return np.repeat(out, patch)[:len(pred)]


def jsd_of(gts, preds, names, dataset):
    acc = {}
    for gt, pr, nm in zip(gts, preds, names):
        act = activity_of(dataset, nm)
        acc.setdefault(act, {"jsd": [], "frames": 0})
        acc[act]["jsd"].append(pair_jsd(rle(gt)[0], rle(pr)[0]))
        acc[act]["frames"] += len(gt)
    return round(frame_weighted(acc), 2)


def analyse(preds_file):
    d = np.load(preds_file, allow_pickle=True)
    dataset, patch = str(d["dataset"]), int(d["patch_size"])
    gts = [np.asarray(g) for g in d["gt"]]
    prs = [np.asarray(p) for p in d["pred"]]
    names = [str(n) for n in d["names"]]
    mapped = global_map(gts, prs)

    out = {"dataset": dataset, "ckpt": str(d["ckpt"]), "patch_size": patch}
    out["smq"] = score(gts, mapped)
    out["smq"]["JSD"] = jsd_of(gts, mapped, names, dataset)

    # ---- D1: patch-grid oracle -------------------------------------------
    oracle = [patch_quantize(g, patch) for g in gts]
    out["d1_patch_grid_oracle"] = score(gts, oracle)
    out["d1_patch_grid_oracle"]["JSD"] = jsd_of(gts, oracle, names, dataset)

    # ---- D2: boundary budget ---------------------------------------------
    tol = patch // 2
    hit_p = spur = 0
    hit_g = miss = 0
    disp = []
    for g, p in zip(gts, mapped):
        bg, bp = boundaries(g), boundaries(p)
        if len(bg) == 0 or len(bp) == 0:
            continue
        dp = np.abs(bp[:, None] - bg[None, :]).min(axis=1)
        dg = np.abs(bg[:, None] - bp[None, :]).min(axis=1)
        hit_p += int((dp <= tol).sum()); spur += int((dp > tol).sum())
        hit_g += int((dg <= tol).sum()); miss += int((dg > tol).sum())
        disp.append(dp)
    disp = np.concatenate(disp)
    out["d2_boundaries"] = dict(
        tolerance_frames=tol,
        n_pred=int(hit_p + spur), n_gt=int(hit_g + miss),
        pred_within_tol_pct=round(100 * hit_p / (hit_p + spur), 1),
        gt_covered_pct=round(100 * hit_g / (hit_g + miss), 1),
        spurious_pct=round(100 * spur / (hit_p + spur), 1),
        missed_pct=round(100 * miss / (hit_g + miss), 1),
        median_displacement_frames=float(np.median(disp)),
    )

    # ---- D3: flicker + mode filter ---------------------------------------
    one_patch = sandwiched = total_runs = 0
    for p in prs:
        L, V, S = rle(p)
        total_runs += len(L)
        for i, (l, v) in enumerate(zip(L, V)):
            if l == patch:
                one_patch += 1
                if 0 < i < len(L) - 1 and V[i - 1] == V[i + 1]:
                    sandwiched += 1
    out["d3_flicker"] = dict(
        pct_runs_one_patch=round(100 * one_patch / total_runs, 1),
        pct_runs_sandwiched_single=round(100 * sandwiched / total_runs, 1),
    )
    out["d3_mode_filter"] = {}
    for w in (3, 5, 7):
        filt = [mode_filter_patches(p, patch, w) for p in prs]
        fm = global_map(gts, filt)
        s = score(gts, fm)
        s["JSD"] = jsd_of(gts, fm, names, dataset)
        s["segments_vs_gt"] = round(
            sum(len(rle(x)[0]) for x in fm) / sum(len(rle(g)[0]) for g in gts), 2)
        out["d3_mode_filter"][f"width_{w}"] = s

    # ---- D4: duration law -------------------------------------------------
    codes = [p[::patch] for p in prs]
    stay = tot = 0
    for c in codes:
        if len(c) > 1:
            stay += int((c[1:] == c[:-1]).sum()); tot += len(c) - 1
    p_stay = stay / tot
    gt_len = np.concatenate([rle(g)[0] for g in gts])
    pr_len = np.concatenate([rle(p)[0] for p in prs])
    out["d4_duration"] = dict(
        p_self_transition=round(p_stay, 4),
        geometric_mean_patches=round(1 / (1 - p_stay), 2),
        pred_mean_patches=round(float(pr_len.mean()) / patch, 2),
        gt_mean_patches=round(float(gt_len.mean()) / patch, 2),
        gt_cv=round(float(gt_len.std() / gt_len.mean()), 2),
        pred_cv=round(float(pr_len.std() / pr_len.mean()), 2),
        geometric_cv=round(float(np.sqrt(p_stay) ), 2),
    )

    # ---- D5: prior mismatch ----------------------------------------------
    gt_cnt = Counter(np.concatenate(gts).tolist())
    pr_cnt = Counter(np.concatenate(mapped).tolist())
    keys = sorted(set(gt_cnt) | set(pr_cnt))
    g = np.array([gt_cnt.get(k, 0) for k in keys], float); g /= g.sum()
    p = np.array([pr_cnt.get(k, 0) for k in keys], float); p /= p.sum()
    ent = lambda x: float(-(x[x > 0] * np.log2(x[x > 0])).sum())
    out["d5_prior"] = dict(
        gt_entropy_bits=round(ent(g), 2), pred_entropy_bits=round(ent(p), 2),
        max_entropy_bits=round(float(np.log2(len(keys))), 2),
        total_variation=round(float(0.5 * np.abs(g - p).sum()), 3),
        mof_upper_bound_from_prior=round(float(100 * np.minimum(g, p).sum()), 2),
    )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/diagnostics/error_decomposition.json"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    all_res = []
    for f in args.preds:
        r = analyse(f)
        all_res.append(r)
        print(json.dumps(r, indent=2))
        print()
    args.out.write_text(json.dumps(all_res, indent=2))
    print(f"[diag] wrote {args.out}")
