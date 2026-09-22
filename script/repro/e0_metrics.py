"""
E0 -- reproduce published SMQ numbers from a dumped prediction file.

Uses the repo's OWN metric code (src/model/eval_utils.py) unchanged, so the
numbers are exactly what `main.py --action=eval` prints; this script only
replaces the forward pass with the cached dump.

Reports both protocols the repo implements:
  local  : Hungarian matching computed independently per sequence
  global : one dataset-level Hungarian matching applied to every sequence
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model.eval_utils import create_correspondences, edit_score, f_score

OVERLAPS = [.1, .25, .5]


def _f1_from_counts(tp, fp, fn):
    out = np.zeros(len(OVERLAPS))
    for i in range(len(OVERLAPS)):
        p = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) else 0.0
        r = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) else 0.0
        out[i] = (2 * p * r / (p + r) if (p + r) else 0.0) * 100.0
    return out


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
    mof = 100.0 * correct / total if total else 0.0
    return dict(MoF=mof, Edit=edit / len(gt_list),
                **{f"F1@{int(o*100)}": v for o, v in zip(OVERLAPS, _f1_from_counts(tp, fp, fn))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    d = np.load(args.preds, allow_pickle=True)
    gt_all = list(d["gt"])
    pred_all = list(d["pred"])

    # --- local (per-sequence) Hungarian ---
    local_mapped = [create_correspondences(gt, pr) for gt, pr in zip(gt_all, pred_all)]
    local = score(gt_all, local_mapped)

    # --- global (dataset-level) Hungarian ---
    _, pr2gt = create_correspondences(np.concatenate(gt_all),
                                      np.concatenate(pred_all), mapping=True)
    global_mapped = [np.vectorize(pr2gt.get)(pr) for pr in pred_all]
    glob = score(gt_all, global_mapped)

    res = dict(dataset=str(d["dataset"]), ckpt=str(d["ckpt"]),
               n_sequences=len(gt_all), num_codes=int(d["num_actions"]),
               patch_size=int(d["patch_size"]), local=local, **{"global": glob})

    print(json.dumps(res, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
