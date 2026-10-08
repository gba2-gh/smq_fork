"""Frozen-checkpoint loading and per-recording extraction for the five
families. Reuses `script.fsq_reconstruction.experiment.extract_dataset`
(VQ/FSQ pre-quantization encoder patches + native code ids, eval mode,
no_grad, one unpadded recording per batch) and
`script.fsq_reconstruction.encoder_sweep.extract_raw_dataset` (raw-input
windows on the identical patch grid) verbatim -- Task 1 does not
reimplement either.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from src.model.smq import SMQModel

from script.fsq_reconstruction.experiment import DATASET_DEFAULTS, PilotConfig, extract_dataset
from script.fsq_reconstruction.encoder_sweep import extract_raw_dataset, load_source_config
from script.fsq_reconstruction.quantizers import FSQSMQModel
from script.quantized_input_comparison import config as cfg


def load_source_pilot_config() -> PilotConfig:
    return load_source_config(cfg.SOURCE_RUN)


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_model(arm: str, source_cfg: PilotConfig, device: torch.device):
    """Loads the epoch-30 checkpoint for `arm` ("vq" or "fsq"), sets
    eval() and freezes every parameter (including the VQ codebook and its
    EMA buffers, which are ordinary buffers -- `.eval()` alone is not
    enough: `SkeletonMotionQuantizer.forward`'s EMA-update branch is gated
    on `self.training`, so eval mode already skips it; `requires_grad_(False)`
    additionally guarantees no optimizer could touch the parameters)."""
    kwargs = source_cfg.smq_kwargs()
    model = SMQModel(**kwargs) if arm == "vq" else FSQSMQModel(fsq_levels=source_cfg.levels, **kwargs)
    path = cfg.checkpoint_path(arm)
    state = torch.load(path, map_location=device)
    before = {k: v.clone() for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # Verify the loaded state dict is exactly what load_state_dict consumed
    # (buffers included), i.e. nothing was silently dropped/renamed.
    after = model.state_dict()
    for k, v in before.items():
        if not torch.equal(v.cpu(), after[k].cpu()):
            raise RuntimeError(f"{arm} checkpoint tensor {k!r} did not load verbatim")
    return model, sha256_of_file(path)


def features_dir(source_cfg: PilotConfig) -> Path:
    return cfg.ROOT / "data" / source_cfg.dataset / "features"


def recording_names(source_cfg: PilotConfig, cap: int | None = None) -> list[str]:
    names = sorted(os.listdir(features_dir(source_cfg)))
    return names[:cap] if cap else names


def recording_frame_counts(source_cfg: PilotConfig, names: list[str]) -> dict[str, int]:
    """Per-recording raw frame count (T, from the (C, T, V, M) feature
    array's shape), read via a memory-mapped shape lookup -- no windowing,
    no population extraction, no full-array load. Cheap enough to compute
    before deciding whether an existing run_dir is reusable under --resume,
    so a changed data directory (same recording names, different content)
    is caught even when the full extraction hasn't run yet."""
    fpath = features_dir(source_cfg)
    counts = {}
    for name in names:
        arr = np.load(fpath / name, mmap_mode="r")
        counts[name] = int(arr.shape[1])
    return counts


def load_models_and_names(source_cfg: PilotConfig, device: torch.device, recording_cap: int | None = None) -> dict:
    """Loads the two frozen models and the (possibly capped) recording name
    list -- the small, persistent state needed throughout preparation. Does
    NOT extract any per-window feature population (those are the large,
    per-population arrays that must be extracted, consumed and released one
    population at a time; see `extract_population`)."""
    d = DATASET_DEFAULTS[source_cfg.dataset]
    names = recording_names(source_cfg, recording_cap)
    vq_model, vq_hash = load_frozen_model("vq", source_cfg, device)
    fsq_model, fsq_hash = load_frozen_model("fsq", source_cfg, device)
    return dict(vq_model=vq_model, fsq_model=fsq_model, vq_hash=vq_hash, fsq_hash=fsq_hash,
                names=names, window=d["patch_size"])


def extract_population(pop_name: str, source_cfg: PilotConfig, model, device: torch.device,
                        names: list[str]) -> dict[str, dict]:
    """Extracts exactly ONE of the three continuous-feature populations
    ("raw", "vq_encoder", "fsq_encoder") over `names`, and only that one --
    the caller is expected to consume it (fit + build the families that
    depend on it + save) and then drop the reference before extracting the
    next population, so at most one population's full per-window arrays are
    resident at a time (never all three simultaneously)."""
    d = DATASET_DEFAULTS[source_cfg.dataset]
    fpath = features_dir(source_cfg)
    if pop_name == "raw":
        # Bounded: pass the (possibly capped) name list straight through
        # instead of loading the whole dataset and subsetting afterwards, so
        # --smoke stays memory/time-bounded to `recording_cap` recordings.
        return extract_raw_dataset(fpath, d["patch_size"], names=names)
    if model is None:
        raise ValueError(f"extract_population({pop_name!r}) requires a frozen model")
    from script.fsq_reconstruction.experiment import extract_recording
    out = {}
    for name in names:
        out[name] = extract_recording(model, fpath / name, d["patch_size"], d["num_features"], d["num_joints"],
                                       d["num_person"], device)
    return out
