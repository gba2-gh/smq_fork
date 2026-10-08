"""Exact 3D nuisance augmentation of NTU-25 skeletons and the precomputed bank of augmented MotionBERT window features.

A variant is: body scale, then a mirror (x flipped AND left/right joints swapped, so the result is a valid body), then the input
view rotation about the vertical axis (replacing the fixed 270 degree view). All three act on the 3D skeleton before MotionBERT,
so time is preserved exactly and window k of every variant covers the same frames as window k of the canonical features."""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

from script.self_labelled import config as C

# NTU-25 left/right pairs: shoulder, elbow, wrist, hand, hip, knee, ankle, foot, hand tip, thumb
LR_PAIRS = [(4, 8), (5, 9), (6, 10), (7, 11), (12, 16), (13, 17), (14, 18), (15, 19), (21, 23), (22, 24)]
SWAP = np.arange(25)
for _a, _b in LR_PAIRS:
    SWAP[_a], SWAP[_b] = _b, _a


def rotation_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def mirror(ntu: np.ndarray) -> np.ndarray:
    out = ntu[:, SWAP, :].copy()
    out[..., 0] *= -1.0
    return out


def draw_params(name: str, variant: int) -> dict:
    rng = np.random.default_rng([C.BANK_SEED, zlib.crc32(name.encode()), variant])
    return dict(angle=float(rng.uniform(0.0, 360.0)), do_mirror=bool(rng.random() < C.MIRROR_P),
                scale=float(rng.uniform(*C.SCALE_RANGE)))


def apply(ntu: np.ndarray, angle: float, do_mirror: bool, scale: float) -> np.ndarray:
    """ntu (T, 25, 3) in mm. Returns the skeleton to which the fixed view rotation `angle` has still to be applied by the encoder."""
    out = ntu * scale
    return mirror(out) if do_mirror else out


def bank_identity() -> str:
    """Everything a cached bank entry depends on: the augmentation and extraction code, the checkpoint, and the declared settings.
    Any change gives a new bank directory, so stale features are never reused."""
    import hashlib
    import json
    from script.pretrained_encoder import encoder as E
    h = hashlib.sha256()
    for f in (C.PKG / "augment.py", C.ROOT / "script/pretrained_encoder/encoder.py", C.ROOT / "script/pretrained_encoder/features.py",
              C.ROOT / "script/week5/timescale_inspection/tokenizers.py"):
        h.update(f.name.encode() + f.read_bytes())
    h.update(json.dumps(dict(ckpt=E.CHECKPOINT_SHA256, seed=C.BANK_SEED, n=C.N_VARIANTS, scale=C.SCALE_RANGE, mirror=C.MIRROR_P,
                             window=C.WINDOW, clip=E.CLIP, stride=E.STRIDE), sort_keys=True).encode())
    return h.hexdigest()[:12]


def bank_dir(root: Path | None = None) -> Path:
    return (root or C.RESULTS) / "aug_bank" / bank_identity()


def build_recording(name: str, ntu: np.ndarray, device=None) -> np.ndarray:
    """(N_VARIANTS, P, 1536) float16 window features of one recording's augmented variants."""
    from script.pretrained_encoder import features as F
    out = []
    for v in range(C.N_VARIANTS):
        p = draw_params(name, v)
        frame = F.frame_features(apply(ntu, **p), device, deg=p["angle"])
        out.append(F.window_features(frame, C.WINDOW).astype(np.float16))
    return np.stack(out)


def load_bank(names: list, build_missing: bool = True, root: Path | None = None) -> dict:
    """name -> (N_VARIANTS, P, 1536) float16, built on demand and cached per recording."""
    from script.week5.raw_kmeans_cross_dataset import datasets as D
    d = bank_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    spec = D.SPECS[C.DATASET]
    out = {}
    for n in names:
        p = d / (Path(n).stem + ".npy")
        if not p.exists():
            if not build_missing:
                raise FileNotFoundError(p)
            arr = np.load(D.features_dir(spec) / n)[..., 0].transpose(1, 2, 0)
            tmp = p.with_suffix(".tmp.npy")
            np.save(tmp, build_recording(n, arr))
            tmp.replace(p)
        out[n] = np.load(p)
    return out


def canonical_windows(names: list) -> dict:
    """name -> (P, 1536) float32: the baseline's canonical window features (270 degree view, no mirror, scale 1)."""
    from script.pretrained_encoder import features as F
    from script.week5.raw_kmeans_cross_dataset import datasets as D
    spec = D.SPECS[C.DATASET]
    out = {}
    for n in names:
        p = F.cache_path(n)
        if p.exists():
            frame = np.load(p)
        else:
            frame = F.frame_features(np.load(D.features_dir(spec) / n)[..., 0].transpose(1, 2, 0))
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, frame)
        out[n] = F.window_features(frame, C.WINDOW).astype(np.float32)
    return out
