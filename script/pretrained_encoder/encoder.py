"""Frozen MotionBERT (pretraining checkpoint MB_release, no action labels) as a per-frame feature extractor for BABEL skeletons.

Pipeline per recording (all label-free):
  NTU-25 joints (BABEL, mm, z up, centred at spine-mid) -> H36M-17 by joint correspondence -> metres -> AMASS camera convention
  used in MotionBERT's own AMASS pretraining (tools/convert_amass.py: v @ [[1,0,0],[0,0,1],[0,-1,0]] * 0.298)
  -> 2D input (x, y, confidence 1) -> resample 30 -> 60 fps (MotionBERT's AMASS frame rate) -> sliding 243-frame clips (stride 81,
  overlaps averaged) -> representation [F, 17, 512] -> back to 30 fps -> mean over joints -> 512-d per frame.
The 3D head output is used only by the feasibility check (lifting error against the true depth), never as a feature."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MB_DIR = ROOT / "external" / "motionbert"
CHECKPOINT = MB_DIR / "checkpoint" / "pretrain" / "MB_release" / "latest_epoch.bin"
CHECKPOINT_SHA256 = "02a9ee017e5c9d688b5dd41e0055bddb75b6eee7f5eed326a8cdc6e5bf08eaf8"

# H36M-17 order: pelvis, Rhip, Rknee, Rankle, Lhip, Lknee, Lankle, spine, thorax, neck/nose, head, Lsho, Lelb, Lwri, Rsho, Relb, Rwri
NTU_TO_H36M = [0, 16, 17, 18, 12, 13, 14, 1, 20, 2, 3, 4, 5, 6, 8, 9, 10]
REAL2CAM = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
SCALE = 0.298
CLIP, STRIDE, SRC_FPS, MB_FPS = 243, 81, 30, 60


def load_model(device):
    sys.path.insert(0, str(MB_DIR))
    from lib.model.DSTformer import DSTformer
    model = DSTformer(dim_in=3, dim_out=3, dim_feat=512, dim_rep=512, depth=5, num_heads=8, mlp_ratio=2, num_joints=17,
                      maxlen=243, att_fuse=True)
    sd = torch.load(CHECKPOINT, map_location="cpu")["model_pos"]
    model.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}, strict=True)
    return model.to(device).eval()


def to_camera(ntu: np.ndarray, convention: str = "amass") -> np.ndarray:
    """ntu (T, 25, 3) mm -> (T, 17, 3) MotionBERT camera coordinates. `convention` alternatives exist only for the feasibility check."""
    j = ntu[:, NTU_TO_H36M, :].astype(np.float32) / 1000.0
    if convention == "amass":
        cam = j @ REAL2CAM
    elif convention == "amass_flipx":
        cam = (j @ REAL2CAM) * np.array([-1, 1, 1], dtype=np.float32)
    elif convention == "yup":
        cam = j * np.array([1, -1, 1], dtype=np.float32)
    else:
        raise ValueError(convention)
    return cam * SCALE


def resample(x: np.ndarray, factor: float) -> np.ndarray:
    T = x.shape[0]
    n = max(1, int(round(T * factor)))
    src = np.linspace(0, T - 1, n)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (src - lo)[:, None, None]
    return (1 - w) * x[lo] + w * x[hi]


def crop_scale_clip(x2d: np.ndarray):
    """MotionBERT's crop_scale (lib/utils/utils_data.py) with scale_range [1, 1]: x, y of one clip mapped to [-1, 1] by the clip's
    square bounding box. Returns (normalized clip, scale) so the 3D head output can be put back in camera units (z / scale)."""
    xy = x2d[..., :2]
    xmin, ymin = xy[..., 0].min(), xy[..., 1].min()
    scale = max(xy[..., 0].max() - xmin, xy[..., 1].max() - ymin)
    if scale <= 0:
        return np.zeros_like(x2d), 1.0
    xs = (xmin + xy[..., 0].max() - scale) / 2
    ys = (ymin + xy[..., 1].max() - scale) / 2
    out = x2d.copy()
    out[..., 0] = ((xy[..., 0] - xs) / scale - 0.5) * 2
    out[..., 1] = ((xy[..., 1] - ys) / scale - 0.5) * 2
    return np.clip(out, -1, 1), float(scale)


@torch.no_grad()
def run_clips(model, x2d: np.ndarray, device, want_3d: bool = False):
    """x2d (F, 17, 3) at 60 fps -> (rep [F, 17, 512], pred3d root-relative in camera units [F, 17, 3] or None); each clip is
    crop-scaled as in pretraining and overlapping clips are averaged."""
    F_ = x2d.shape[0]
    starts = list(range(0, max(1, F_ - CLIP + 1), STRIDE))
    if starts[-1] + CLIP < F_:
        starts.append(F_ - CLIP)
    rep = np.zeros((F_, 17, 512), dtype=np.float64)
    p3 = np.zeros((F_, 17, 3), dtype=np.float64) if want_3d else None
    cnt = np.zeros(F_)
    for s in starts:
        e = min(F_, s + CLIP)
        # MotionBERT's AMASS pretraining feeds raw camera-unit x, y with c = 1 (no crop_scale; MB_pretrain.yaml sets no
        # scale_range_pretrain), so the same is done here. crop_scale_clip is kept only as a documented alternative.
        clip = torch.as_tensor(x2d[s:e][None], dtype=torch.float32, device=device)
        r = model.get_representation(clip)[0]
        rep[s:e] += r.cpu().numpy()
        if want_3d:
            out = model.head(r).cpu().numpy().astype(np.float64)
            p3[s:e] += out - out[:, :1]
        cnt[s:e] += 1
    rep /= cnt[:, None, None]
    if want_3d:
        p3 /= cnt[:, None, None]
    return rep, p3


def recording_features(model, ntu: np.ndarray, device, convention: str = "amass", want_3d: bool = False):
    cam = to_camera(ntu, convention)
    x2d = np.concatenate([cam[..., :2], np.ones_like(cam[..., :1])], axis=-1)
    up = resample(x2d, MB_FPS / SRC_FPS)
    rep, p3 = run_clips(model, up, device, want_3d)
    T = ntu.shape[0]
    idx = np.clip(np.round(np.arange(T) * (MB_FPS / SRC_FPS)).astype(int), 0, len(rep) - 1)
    feat = rep[idx].mean(axis=1).astype(np.float32)               # (T, 512): mean over joints
    return (feat, cam, p3[idx] if want_3d else None)
