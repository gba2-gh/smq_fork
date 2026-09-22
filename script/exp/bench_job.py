"""
Short resource probe for one training configuration: runs a few real training
steps (same loss as model.Trainer.train) under the same caps the job runner
uses, then reports peak GPU memory and seconds per batch.  Stops after
--batches steps or --max_seconds, whichever comes first.
"""
import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import torch
from torch import optim

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from batch_gen import BatchGenerator
from main import get_dataset_defaults
from model import Trainer
from src.model.utils import get_num_actions, distance_joints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num_actions", type=int, default=None)
    ap.add_argument("--dead_code_threshold", type=int, default=10)
    ap.add_argument("--tc_weight", type=float, default=0.0)
    ap.add_argument("--patch_size", type=int, default=None)
    ap.add_argument("--data_root", type=Path, default=Path("data"))
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--max_seconds", type=float, default=120)
    ap.add_argument("--gpu_mem_fraction", type=float, default=0.5)
    ap.add_argument("--grad_checkpoint", action="store_true")
    ap.add_argument("--worst_case", action="store_true",
                    help="Put the longest sequences first so the first batches are the peak-memory ones")
    args = ap.parse_args()

    if sys.platform == "win32":  # below-normal priority, as the runner uses
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    torch.set_num_threads(4)
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)

    cfg = get_dataset_defaults(args.dataset)
    root = args.data_root / args.dataset
    K = args.num_actions or get_num_actions(root / "groundTruth")
    ps = args.patch_size or cfg["patch_size"]
    tr = Trainer(in_channels=cfg["num_features"], filters=128, num_layers=3, latent_dim=16,
                 num_actions=K, num_joints=cfg["num_joints"], num_person=1, patch_size=ps,
                 kmeans=False, kmeans_metric="euclidean", sampling_quantile=0.5,
                 replacement_strategy="representative", decay=0.5,
                 dead_code_threshold=args.dead_code_threshold, tc_weight=args.tc_weight,
                 grad_checkpoint=args.grad_checkpoint)
    model = tr.model.to(dev).train()
    opt = optim.Adam(model.parameters(), lr=5e-4)
    bg = BatchGenerator(root / "features", 1, cfg["num_features"], cfg["num_joints"], 1)
    bg.read_data()
    if args.worst_case:
        import numpy as np
        bg.list_of_examples.sort(
            key=lambda f: -np.load(root / "features" / f, mmap_mode="r").shape[1])

    times, t_start = [], time.time()
    for b in range(args.batches):
        if not bg.has_next() or time.time() - t_start > args.max_seconds:
            break
        t0 = time.time()
        x, m = bg.next_batch(cfg["batch_size"])
        x, m = x.to(dev), m.to(dev)
        opt.zero_grad()
        rec = model(x, m)
        loss = 0.001 * torch.mean((distance_joints(x) - distance_joints(rec)) ** 2) \
            + model.commit_loss + model.vq.tc_loss
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        print(f"  batch {b + 1}: {times[-1]:.2f}s  peak reserved "
              f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GB", flush=True)

    steady = times[1:] or times
    n_batches = bg.num_batches(cfg["batch_size"])
    print(f"[bench] {args.dataset} K={K} thr={args.dead_code_threshold} tc={args.tc_weight} ckpt={args.grad_checkpoint} "
          f"patch={ps}: peak reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GB "
          f"(cap {args.gpu_mem_fraction * 24:.1f} GB), steady {sum(steady) / len(steady):.2f} s/batch "
          f"-> est. {sum(steady) / len(steady) * n_batches * 30 / 60:.0f} min for 30 epochs")


if __name__ == "__main__":
    main()
