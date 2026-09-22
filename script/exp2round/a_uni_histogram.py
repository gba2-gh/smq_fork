import numpy as nu
from script.repro.d_error_decomposition import rle, boundaries, score, global_map
from script.repro.d_boundary_vs_label import relabel_by_majority

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
    gts = [np.asarray(g) for g in d["gt"]]
    prs = [np.asarray(p) for p in d["pred"]]
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