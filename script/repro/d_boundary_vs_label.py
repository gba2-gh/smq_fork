"""
D6 / D7 -- separate "the boundaries are wrong" from "the labels are wrong", and
ask whether boundary information is even present in the encoder's latents.

D7  Oracle ablation grid.  Four segmentations, all scored with the repo's own
    metrics, differing only in which half of the problem is solved for free:

      smq                 : SMQ boundaries + SMQ labels                (published)
      smq_bnd_oracle_lab  : SMQ boundaries, each predicted segment relabelled
                            with its majority GT label. Pure boundary quality.
      gt_bnd_smq_lab      : GT boundaries, each GT segment labelled by the
                            majority SMQ code inside it, then globally
                            Hungarian-matched. Pure label quality.
      patch_grid_oracle   : both solved, at patch resolution (the ceiling).

    smq_bnd_oracle_lab tells you how much MoF the existing boundaries could
    support; gt_bnd_smq_lab tells you how much the codes could support if
    boundaries were free.

D6  Is boundary information present in the latents?  Take the encoder's
    pre-quantisation latent z_t, build a change signal, and measure how well its
    peaks coincide with GT boundaries (ROC AUC + top-N boundary F1).  The same
    signal is computed on the raw input features as a reference, so we can see
    whether the encoder adds boundary information or washes it out.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model.eval_utils import create_correspondences
from script.repro.d_error_decomposition import rle, boundaries, score, global_map
from script.repro.dump_predictions import build_model, DEFAULTS


# ----------------------------------------------------------------- D7 --------
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


# ----------------------------------------------------------------- D6 --------
def change_signal(x, half):
    """Mean-vs-mean contrast between the window before t and the window after t.

    x: (T, D). Returns (T,) with large values where the representation changes.
    """
    T = x.shape[0]
    cs = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)], axis=0)

    def wmean(a, b):          # mean of rows [a, b)
        a = np.clip(a, 0, T); b = np.clip(b, 0, T)
        n = np.maximum(b - a, 1)[:, None]
        return (cs[b] - cs[a]) / n

    t = np.arange(T)
    left = wmean(t - half, t)
    right = wmean(t, t + half)
    return np.linalg.norm(right - left, axis=1)


def roc_auc(scores, labels):
    """AUC via rank statistic; labels is a boolean array."""
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def peak_pick(sig, n, min_dist):
    """n highest-scoring positions, greedily enforcing a minimum separation."""
    picked = []
    for i in np.argsort(sig)[::-1]:
        if len(picked) >= n:
            break
        if all(abs(i - p) >= min_dist for p in picked):
            picked.append(int(i))
    return np.array(sorted(picked))


def boundary_f1(pred_b, gt_b, tol):
    if len(pred_b) == 0 or len(gt_b) == 0:
        return 0.0
    d = np.abs(pred_b[:, None] - gt_b[None, :])
    tp = int((d.min(axis=1) <= tol).sum())
    prec = tp / len(pred_b)
    rec = int((d.min(axis=0) <= tol).sum()) / len(gt_b)
    return round(100 * (2 * prec * rec / (prec + rec) if (prec + rec) else 0.0), 2)


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
    gts = [np.asarray(g) for g in d["gt"]]
    prs = [np.asarray(p) for p in d["pred"]]
    names = [str(n) for n in d["names"]]
    mapped = global_map(gts, prs)

    res = {"dataset": args.dataset, "ckpt": str(args.ckpt), "patch_size": patch}

    # ---------------- D7 ----------------
    res["d7_smq"] = score(gts, mapped)
    res["d7_smq_bnd_oracle_lab"] = score(
        gts, [relabel_by_majority(p, g) for p, g in zip(mapped, gts)])
    gt_seg_codes = [codes_on_gt_segments(g, p) for g, p in zip(gts, prs)]
    res["d7_gt_bnd_smq_lab"] = score(gts, global_map(gts, gt_seg_codes))

    # ---------------- D6 ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = args.data_root / args.dataset
    model, _, _ = build_model(args.dataset, root / "groundTruth")
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device).eval()

    tol = patch // 2
    auc_lat, auc_raw, f1_lat, f1_raw, f1_smq = [], [], [], [], []
    idx = np.linspace(0, len(names) - 1, min(args.max_seqs_latent, len(names))).astype(int)

    with torch.no_grad():
        for i in idx:
            gt = gts[i]
            if len(gt) < 4 * patch or len(np.unique(gt)) < 2:
                continue
            feats = np.load(root / "features" / names[i])
            x = torch.tensor(feats, dtype=torch.float).unsqueeze(0).to(device)
            mask = torch.ones(x.size(), device=device)
            _ = model(x, mask)
            N, C, T, V, M = x.size()
            # encoder latents, repacked to (T, V*M*latent_dim) -- the exact
            # tensor the quantiser sees
            lat = model.latent.view(N * M, V, model.latent_dim, T)
            lat = lat.permute(0, 3, 1, 2).reshape(T, -1).cpu().numpy()
            raw = feats.transpose(1, 0, 2, 3).reshape(T, -1)

            gb = boundaries(gt)
            lab = np.zeros(T, bool)
            for b in gb:
                lab[max(0, b - tol):min(T, b + tol + 1)] = True

            s_lat = change_signal(lat, patch // 2)
            s_raw = change_signal(raw, patch // 2)
            auc_lat.append(roc_auc(s_lat, lab))
            auc_raw.append(roc_auc(s_raw, lab))
            n = len(gb)
            f1_lat.append(boundary_f1(peak_pick(s_lat, n, patch // 2), gb, tol))
            f1_raw.append(boundary_f1(peak_pick(s_raw, n, patch // 2), gb, tol))
            f1_smq.append(boundary_f1(boundaries(mapped[i]), gb, tol))

    res["d6_boundary_info"] = dict(
        n_sequences=len(auc_lat), tolerance_frames=tol,
        auc_latent_change=round(float(np.nanmean(auc_lat)), 3),
        auc_raw_feature_change=round(float(np.nanmean(auc_raw)), 3),
        boundary_f1_latent_topN=round(float(np.mean(f1_lat)), 2),
        boundary_f1_raw_topN=round(float(np.mean(f1_raw)), 2),
        boundary_f1_smq_actual=round(float(np.mean(f1_smq)), 2),
    )

    print(json.dumps(res, indent=2))
    out = args.out or Path(f"results/diagnostics/bnd_vs_label_{args.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
