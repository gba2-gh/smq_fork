"""
Run a trained SMQ checkpoint over every sequence of a dataset and dump, for
each sequence:  ground-truth frame labels + raw (unmapped) patch-code indices.

Everything downstream (E0 metrics, JSD, qualitative figures, diagnostics)
reads this .npz so the network is only run once per checkpoint.

Usage:
  python script/repro/dump_predictions.py --dataset hugadb \
      --ckpt models/pretrained/hugadb.model --out results/preds/hugadb_pretrained.npz
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.model.smq import SMQModel
from src.model.utils import get_num_actions
from src.model.eval_utils import read_mapping_file

# Same per-dataset config as main.py::get_dataset_defaults
DEFAULTS = {
    "hugadb": dict(num_features=6, num_joints=6,  num_person=1, patch_size=60),
    "lara":   dict(num_features=6, num_joints=22, num_person=1, patch_size=50),
    "babel1": dict(num_features=3, num_joints=25, num_person=1, patch_size=30),
    "babel2": dict(num_features=3, num_joints=25, num_person=1, patch_size=30),
    "babel3": dict(num_features=3, num_joints=25, num_person=1, patch_size=30),
}


def build_model(dataset, gt_path, num_f_maps=128, num_layers=3, latent_dim=16,
                patch_size=None, decay=0.5):
    cfg = DEFAULTS[dataset]
    ps = patch_size if patch_size is not None else cfg["patch_size"]
    num_actions = get_num_actions(gt_path)
    model = SMQModel(
        in_channels=cfg["num_features"], filters=num_f_maps, num_layers=num_layers,
        latent_dim=latent_dim, num_actions=num_actions, num_joints=cfg["num_joints"],
        num_person=cfg["num_person"], patch_size=ps, kmeans=False,
        kmeans_metric="euclidean", sampling_quantile=0.5,
        replacement_strategy="representative", decay=decay,
    )
    return model, num_actions, ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DEFAULTS))
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data_root", type=Path, default=Path("data"))
    ap.add_argument("--patch_size", type=int, default=None)
    ap.add_argument("--num_f_maps", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--latent_dim", type=int, default=16)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = args.data_root / args.dataset
    features_path, gt_path = root / "features", root / "groundTruth"
    mapping_file = root / "mapping" / "mapping.txt"

    model, num_actions, ps = build_model(args.dataset, gt_path,
                                         args.num_f_maps, args.num_layers,
                                         args.latent_dim, args.patch_size)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device).eval()

    mapping = read_mapping_file(mapping_file)
    actions_dict = {v: k for k, v in mapping.items()}

    vids = sorted(os.listdir(features_path))
    gts, preds, names = [], [], []
    with torch.no_grad():
        for vid in vids:
            feats = np.load(features_path / vid)
            gt = np.array([actions_dict[a] for a in
                           open(gt_path / vid.replace("npy", "txt")).read().splitlines()])
            x = torch.tensor(feats, dtype=torch.float).unsqueeze(0).to(device)
            mask = torch.ones(x.size(), device=device)
            _ = model(x, mask)
            pred = model.indices.squeeze(0).cpu().numpy()
            gts.append(gt.astype(np.int16))
            preds.append(pred.astype(np.int16))
            names.append(vid)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        names=np.array(names),
        gt=np.array(gts, dtype=object),
        pred=np.array(preds, dtype=object),
        patch_size=ps,
        num_actions=num_actions,
        dataset=args.dataset,
        ckpt=str(args.ckpt),
        allow_pickle=True,
    )
    print(f"[dump] {args.dataset} ckpt={args.ckpt} -> {args.out} "
          f"({len(vids)} sequences, K={num_actions}, patch={ps})")


if __name__ == "__main__":
    main()
