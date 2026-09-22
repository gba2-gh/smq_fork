"""
Dump everything the T1-T4 analyses need from one checkpoint, in one forward
pass per sequence (eval mode, no grad):

  names, gt (frame labels), T (lengths)
  codes      : patch-level code index per sequence  (frame-level = repeat by W)
  patch_dist : patch x K distance to every code      (T3 decoding)
  codebook   : K x W x D                             (T1 geometry merge)
  probe_*    : patch-mean pre-quantisation latents + majority GT label, for an
               evenly spaced subset of sequences     (latent probe)

Codebook size K and patch size W are read from the checkpoint itself, so the
dump cannot silently disagree with how the model was trained.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from main import get_dataset_defaults
from src.model.smq import SMQModel
from src.model.eval_utils import read_mapping_file


def ragged(arrays):
    """1-D object array of per-sequence arrays (np.array would stack equal-length
    sequences, e.g. LARa's, into a 2-D object array)."""
    out = np.empty(len(arrays), dtype=object)
    for i, a in enumerate(arrays):
        out[i] = a
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data_root", type=Path, default=Path("data"))
    ap.add_argument("--probe_seqs", type=int, default=250)
    ap.add_argument("--gpu_mem_fraction", type=float, default=0.5)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)
    torch.set_num_threads(4)

    state = torch.load(args.ckpt, map_location="cpu")
    K, W, D = state["vq._embedding"].shape
    cfg = get_dataset_defaults(args.dataset)
    model = SMQModel(in_channels=cfg["num_features"], filters=128, num_layers=3,
                     latent_dim=16, num_actions=K, num_joints=cfg["num_joints"],
                     num_person=1, patch_size=W)
    model.load_state_dict(state)
    model.to(dev).eval()

    root = args.data_root / args.dataset
    mapping = read_mapping_file(root / "mapping" / "mapping.txt")
    actions = {v: k for k, v in mapping.items()}
    vids = sorted(os.listdir(root / "features"))
    probe_idx = set(np.linspace(0, len(vids) - 1, min(args.probe_seqs, len(vids))).astype(int).tolist())

    captured = {}
    model.vq.register_forward_hook(lambda m, i, o: captured.__setitem__("dist", o[3]))

    gts, codes, dists, lens = [], [], [], []
    p_lat, p_lab, p_grp = [], [], []
    with torch.no_grad():
        for i, vid in enumerate(vids):
            feats = np.load(root / "features" / vid)
            gt = np.array([actions[a] for a in
                           (root / "groundTruth" / vid.replace(".npy", ".txt")).read_text().splitlines()])
            x = torch.tensor(feats, dtype=torch.float, device=dev).unsqueeze(0)
            _ = model(x, torch.ones_like(x))
            T = x.shape[2]
            code_frames = model.indices.squeeze(0).cpu().numpy()
            dist = -captured["dist"].squeeze(0)[::W].cpu().numpy()          # (P, K), positive
            gts.append(gt.astype(np.int16))
            codes.append(code_frames[::W].astype(np.int16))
            dists.append(dist.astype(np.float32))
            lens.append(T)

            if i in probe_idx:
                V = x.shape[3]
                lat = model.latent.view(1, V, model.latent_dim, T).permute(0, 3, 1, 2).reshape(T, -1)
                P = int(np.ceil(T / W))
                pad = P * W - T
                lat = torch.nn.functional.pad(lat, (0, 0, 0, pad))
                cnt = torch.full((P, 1), float(W), device=dev)
                cnt[-1, 0] = W - pad
                lat_mean = lat.view(P, W, -1).sum(1) / cnt
                p_lat.append(lat_mean.cpu().numpy().astype(np.float16))
                lab = np.empty(P, np.int16)
                for p in range(P):
                    v, c = np.unique(gt[p * W:(p + 1) * W], return_counts=True)
                    lab[p] = v[np.argmax(c)]
                p_lab.append(lab)
                p_grp.append(np.full(P, i, np.int32))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, dataset=args.dataset, ckpt=str(args.ckpt), K=K, W=W,
        names=np.array(vids), T=np.array(lens),
        gt=ragged(gts), codes=ragged(codes), patch_dist=ragged(dists),
        codebook=state["vq._embedding"].numpy().astype(np.float32),
        probe_latent=np.concatenate(p_lat), probe_label=np.concatenate(p_lab),
        probe_group=np.concatenate(p_grp))
    print(f"[dump_run] {args.dataset} K={K} W={W} {len(vids)} sequences -> {args.out}")


if __name__ == "__main__":
    main()
