"""Dump pre-quantisation SMQ encoder latents and verify their quantisation.

This is intentionally CPU-only.  Gate 0 reconstructs checkpoint-sized patches
from the saved (T, V, D) latents, quantises them against the checkpoint
codebook, and requires >= 0.999 frame agreement with a cached prediction dump.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.repro.dump_predictions import DEFAULTS, build_model
from src.model.eval_utils import read_mapping_file
from src.model.motion_quantizer import euclidean_dist


def encoder_latent_tvd(model, sequence_length):
    """Convert model.latent from (M*V, D, T) to (T, V, D)."""
    if model.num_person != 1:
        raise ValueError("dump format (T,V,D) currently requires num_person == 1")
    latent = model.latent.detach().cpu()
    expected = model.num_joints * model.num_person
    if latent.shape != (expected, model.latent_dim, sequence_length):
        raise ValueError(f"unexpected encoder latent shape {tuple(latent.shape)}")
    return latent.reshape(model.num_joints, model.latent_dim, sequence_length).permute(2, 0, 1)


def quantise_dumped_latent(latent_tvd, codebook, patch_size):
    """Reproduce SMQ patch assignment from a dumped (T,V,D) latent."""
    latent = torch.as_tensor(latent_tvd, dtype=torch.float32).reshape(len(latent_tvd), -1)
    remainder = len(latent) % patch_size
    if remainder:
        latent = torch.nn.functional.pad(latent, (0, 0, 0, patch_size - remainder))
    patches = latent.reshape(-1, patch_size, latent.shape[-1])
    labels = torch.argmin(euclidean_dist(patches, codebook), dim=1)
    return labels.repeat_interleave(patch_size)[:len(latent_tvd)].numpy().astype(np.int16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DEFAULTS))
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--preds", required=True, type=Path,
                        help="Cached prediction dump for Gate 0")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--gate_threshold", type=float, default=0.999)
    args = parser.parse_args()

    started = time.perf_counter()
    root = args.data_root / args.dataset
    features_path = root / "features"
    gt_path = root / "groundTruth"
    mapping_file = root / "mapping" / "mapping.txt"

    model, num_actions, patch_size = build_model(args.dataset, gt_path)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.cpu().eval()
    codebook = model.vq._embedding.detach().cpu()

    cached = np.load(args.preds, allow_pickle=True)
    cached_names = [str(name) for name in cached["names"]]
    cached_preds = [np.asarray(pred) for pred in cached["pred"]]
    if int(cached["patch_size"]) != patch_size:
        raise ValueError("cached predictions and checkpoint use different patch sizes")

    mapping = read_mapping_file(mapping_file)
    actions_dict = {value: key for key, value in mapping.items()}
    names = sorted(os.listdir(features_path))
    if names != cached_names:
        raise ValueError("cached prediction names/order do not match dataset features")

    latents, gts, preds = [], [], []
    matches = total = 0
    with torch.no_grad():
        for name, cached_pred in zip(names, cached_preds):
            features = np.load(features_path / name)
            gt = np.array([
                actions_dict[action]
                for action in (gt_path / name.replace("npy", "txt")).read_text().splitlines()
            ], dtype=np.int16)
            x = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
            mask = torch.ones_like(x)
            model(x, mask)

            latent = encoder_latent_tvd(model, len(gt)).numpy().astype(np.float32)
            reconstructed = quantise_dumped_latent(latent, codebook, patch_size)
            model_pred = model.indices.squeeze(0).cpu().numpy().astype(np.int16)
            if not (len(gt) == len(cached_pred) == len(model_pred) == len(reconstructed)):
                raise ValueError(f"length mismatch for {name}")
            if not np.array_equal(model_pred, reconstructed):
                raise RuntimeError(f"direct and reconstructed quantisation disagree for {name}")

            matches += int(np.sum(reconstructed == cached_pred))
            total += len(gt)
            latents.append(latent)
            gts.append(gt)
            preds.append(cached_pred.astype(np.int16))

    agreement = matches / total
    elapsed = time.perf_counter() - started
    gate = {
        "name": "gate_0",
        "agreement": agreement,
        "threshold": args.gate_threshold,
        "passed": agreement >= args.gate_threshold,
        "matching_frames": matches,
        "total_frames": total,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        latent=np.array(latents, dtype=object),
        gt=np.array(gts, dtype=object),
        pred=np.array(preds, dtype=object),
        names=np.array(names),
        patch_size=patch_size,
        num_actions=num_actions,
        dataset=args.dataset,
        ckpt=str(args.ckpt),
        cached_preds=str(args.preds),
        gate0_json=json.dumps(gate),
        runtime_seconds=elapsed,
    )
    print(json.dumps({**gate, "out": str(args.out), "runtime_seconds": elapsed}, indent=2))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
