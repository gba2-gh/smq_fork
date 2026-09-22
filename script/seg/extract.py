"""
Extract frozen-encoder frame-level latents for the segmentation-first programme.

For each dataset, the RELEASED checkpoint (models/pretrained/<ds>.model) is run
once per recording in eval mode. We store

  <cache>/<ds>/<name>.npy   float32 (T, D)  pre-quantisation latent (the VQ input),
                                            one row per native frame, D = joints*16
  <cache>/<ds>/_meta.npz    names, original SMQ frame-level codes, codebook (K,W,D),
                            checkpoint sha256

and verify that the frame-level original-SMQ codes agree exactly with the cached
E0 predictions (results/preds/<ds>_pretrained.npz).
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from main import get_dataset_defaults  # noqa: E402
from src.model.smq import SMQModel  # noqa: E402
from script.seg.common import DATASETS  # noqa: E402


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--cache", type=Path, default=ROOT / "results" / "seg" / "cache")
    ap.add_argument("--ckpt_dir", type=Path, default=ROOT / "models" / "pretrained")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ds in args.datasets:
        ck = args.ckpt_dir / f"{ds}.model"
        state = torch.load(ck, map_location="cpu")
        K, W, D = state["vq._embedding"].shape
        cfg = get_dataset_defaults(ds)
        assert W == cfg["patch_size"] == DATASETS[ds]["patch"], (W, cfg)
        model = SMQModel(in_channels=cfg["num_features"], filters=128, num_layers=3, latent_dim=16,
                         num_actions=K, num_joints=cfg["num_joints"], num_person=1, patch_size=W)
        model.load_state_dict(state)
        model.to(dev).eval()
        cap = {}
        model.vq.register_forward_hook(lambda m, i, o: cap.__setitem__("z", i[0].detach()))

        out = args.cache / ds
        out.mkdir(parents=True, exist_ok=True)
        names = sorted(p.stem for p in (ROOT / "data" / ds / "features").glob("*.npy"))
        codes = []
        with torch.no_grad():
            for n in names:
                f = np.load(ROOT / "data" / ds / "features" / f"{n}.npy")
                x = torch.tensor(f, dtype=torch.float, device=dev).unsqueeze(0)
                _ = model(x, torch.ones_like(x))
                z = cap["z"][0].cpu().numpy().astype(np.float32)  # (T, D)
                assert z.shape == (f.shape[1], D), (z.shape, f.shape, D)
                np.save(out / f"{n}.npy", z)
                codes.append(model.indices[0].cpu().numpy().astype(np.int16))

        # verify against cached E0 predictions
        d = np.load(ROOT / "results" / "preds" / f"{ds}_pretrained.npz", allow_pickle=True)
        pn = [str(s).replace(".npy", "") for s in d["names"]]
        assert pn == names, "recording order differs from E0 dump"
        mism = sum(int((np.asarray(p) != c).sum()) for p, c in zip(d["pred"], codes))
        tot = sum(len(c) for c in codes)
        print(f"{ds}: K={K} W={W} D={D}  n={len(names)}  code mismatches vs E0 dump: {mism}/{tot}")
        arr = np.empty(len(codes), dtype=object)
        for i, c in enumerate(codes):
            arr[i] = c
        np.savez(out / "_meta.npz", names=np.array(names), codes=arr,
                 codebook=state["vq._embedding"].numpy(), ckpt_sha256=sha256(ck), mismatch=mism, total=tot)


if __name__ == "__main__":
    main()
