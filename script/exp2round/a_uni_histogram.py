#1. A cheap check before building anything. Represent each ground-truth segment by its unit histogram, cluster these into C without labels,
 #and Hungarian-match. Compare against D7's majority SMQ code on the same segments (37.5–44.9 MoF). If the histograms don't 
 #clearly beat that, no decoder will help and the problem is the units.

import numpy as np
import argparse
import json
from pathlib import Path
from script.repro.d_error_decomposition import rle, boundaries, score, global_map
#from script.repro.d_boundary_vs_label import relabel_by_majority
from script.repro.dump_predictions import DEFAULTS
from sklearn.cluster import KMeans

def relabel_by_majority(segmentation, gt):
    """Give every run of `segmentation` the majority GT label inside it."""
    out = np.empty_like(gt)
    L, _, S = rle(segmentation)
    for s, l in zip(S, L):
        e = s + l
        v, c = np.unique(gt[s:e], return_counts=True)
        out[s:e] = v[np.argmax(c)]
    return out


def codes_on_gt_segments(gt, pred_codes):
    """Label every GT segment with the majority SMQ code inside it."""
    out = np.empty_like(pred_codes)
    L, _, S = rle(gt)
    for s, l in zip(S, L):
        e = s + l
        v, c = np.unique(pred_codes[s:e], return_counts=True)
        out[s:e] = v[np.argmax(c)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DEFAULTS))
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--preds", required=True, type=Path)
    ap.add_argument("--data_root", type=Path, default=Path("data"))
    ap.add_argument("--max_seqs_latent", type=int, default=120)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    d = np.load(args.preds, allow_pickle=True)
    patch = int(d["patch_size"])
    gts = [np.asarray(g) for g in d["gt"]] # gt segments
    prs = [np.asarray(p) for p in d["pred"]] #pred segments
    names = [str(n) for n in d["names"]]
    mapped = global_map(gts, prs)

    res = {"dataset": args.dataset, "ckpt": str(args.ckpt), "patch_size": patch}

    # ---------------- D7 ----------------
    res["d7_smq"] = score(gts, mapped)
    res["d7_smq_bnd_oracle_lab"] = score(gts, [relabel_by_majority(p, g) for p, g in zip(mapped, gts)])
    gt_seg_codes = [codes_on_gt_segments(g, p) for g, p in zip(gts, prs)]
    res["d7_gt_bnd_smq_lab"] = score(gts, global_map(gts, gt_seg_codes))

    print(json.dumps(res, indent=2))
    out = args.out or Path(f"results/exp2round/d_uni_histogram_{args.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))


    #exp2 A
    out = np.empty_like(pred_codes)
    L, _, S = rle(gts)
    for l, in zip(S,L):
        e = s+ l
        unit_hist = np.zeros(1000)

        value,count = np.unique(pred_codes[s:e], return_counts = True)
        #unit_hist[v] = np.sqrt(c/s)  for v, c in zip(value, count) #unit histogram divided by segmetn lenght, root sqrt hellinger dist
        unit_hist[v] = c/s for v, c in zip(value, count)
        unit_hist = np.squrt(unit_hist)
        out[s:e] = unit_hist
    return out

    K = 10
    km=KMeans(n_clusters=K,n_init=5,max_iter=100,random_state=seed).fit(out)
    centers=km.cluster_centers_.astype(np.float32) 
    kmean_pred = km.labels_
    mapped = global_map(gts, kmean_pred)
    score = score(gts, mapped)
    print(score)

    #km = MiniBatchKMeans(n_clusters=K, batch_size=4096, n_init=3, max_iter=100,random_state=seed).fit(d["X"])
    #labels = km.labels_.astype(np.int16)

                           