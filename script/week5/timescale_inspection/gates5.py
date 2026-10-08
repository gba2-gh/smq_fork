"""Phase 5 gates. A failing gate blocks only what depends on it (see PHASE5_TOKENIZER.md):

  G1  the generalized preparation reproduces `rkx_v2`'s raw_kmeans artifacts (BABEL-2, seed 111): codes, features, grouping
      exactly, K-means centres to 1e-4 -> everything
  G2  SMQ's released checkpoint loads verbatim and its own codes reproduce results/e0/{dataset}_pretrained.json
      `global` within 0.005 on every metric, via the exact producer `script/repro/e0_metrics.py` -> T4 on that dataset
  G3  LARa's channels 3-5 have the larger across-joint extent (they are positions, not orientations) -> T2/T3/T3b on LARa
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from script.quantized_input_comparison.artifacts import Loader
from script.week5.raw_kmeans_cross_dataset import datasets as D
from script.week5.timescale_inspection import tokenizers as TOK

ROOT = D.ROOT
LOOKUP_ATOL = 1e-4
G2_TOLERANCE = 0.005
G2_METRICS = ("MoF", "Edit", "F1@10", "F1@25", "F1@50")


def gate_g1(prep_dir: Path, rkx_run: str = "rkx_v2", dataset: str = "babel2", seed: int = 111,
            recording_cap: int | None = None) -> dict:
    """Build T0 through the generalized path and compare every artifact with the saved rkx_v2 cell."""
    ref_dir = ROOT / "results" / "week5" / "raw_kmeans_cross_dataset" / rkx_run / "prep" / f"{dataset}__global__seed{seed}"
    if not ref_dir.exists():
        return dict(status="failed", reason=f"reference artifacts not found: {ref_dir}")
    TOK.prepare_cell(dataset, "t0_raw", seed, prep_dir, recording_cap=recording_cap)
    a, b = Loader(ref_dir), Loader(prep_dir)
    fam_a, fam_b = "raw_kmeans", TOK.SPECS["t0_raw"].family
    checks: dict[str, bool] = {}
    names = sorted(a.recording_names())
    checks["same_recordings"] = names == sorted(b.recording_names())
    if checks["same_recordings"]:
        checks["codes"] = all(np.array_equal(a.recording(fam_a, n)["codes"], b.recording(fam_b, n)["codes"]) for n in names)
        checks["continuous"] = all(np.array_equal(a.recording(fam_a, n)["continuous"], b.recording(fam_b, n)["continuous"])
                                   for n in names)
        checks["starts_lengths"] = all(
            np.array_equal(a.recording(fam_a, n)["starts"], b.recording(fam_b, n)["starts"]) and
            np.array_equal(a.recording(fam_a, n)["lengths"], b.recording(fam_b, n)["lengths"]) for n in names)
    # K-means centres are not bit-reproducible even between two runs of the original preparation (measured max
    # difference 7.6e-6 on BABEL-2 seed 111, with identical codes and PCA features), so the centres are compared to a
    # float32-level tolerance; everything that decides a code or a decode input is compared exactly.
    lookup_diff = float(np.abs(a.lookup(fam_a).astype(np.float64) - b.lookup(fam_b).astype(np.float64)).max())
    checks["lookup"] = lookup_diff <= LOOKUP_ATOL
    checks["mean"] = np.array_equal(a.mean(fam_a), b.mean(fam_b))
    checks["rms"] = float(a.rms(fam_a)) == float(b.rms(fam_b))
    checks["grouping"] = np.array_equal(a.grouping(fam_a), b.grouping(fam_b))
    checks["split"] = (json.loads((ref_dir / "split.json").read_text()) == json.loads((Path(prep_dir) / "split.json").read_text()))
    ok = all(checks.values())
    return dict(status="passed" if ok else "failed", checks=checks, lookup_max_abs_diff=lookup_diff, dataset=dataset, seed=seed, reference=str(ref_dir))


def smq_global_scores(dataset: str, device: torch.device) -> tuple[dict, dict]:
    """Replays `dump_predictions` (same loop) and `e0_metrics.main`'s global branch (same functions)."""
    from script.repro import e0_metrics as e0
    from src.model.eval_utils import create_correspondences, read_mapping_file
    model, state = TOK.load_smq(dataset, device)
    loaded = model.state_dict()
    tensors_equal = (set(loaded) == set(state) and all(torch.equal(loaded[k].cpu(), state[k].cpu()) for k in state))
    root = ROOT / "data" / dataset
    features_path, gt_path = root / "features", root / "groundTruth"
    mapping = read_mapping_file(root / "mapping" / "mapping.txt")
    actions_dict = {v: k for k, v in mapping.items()}
    gts, preds = [], []
    with torch.no_grad():
        for vid in sorted(os.listdir(features_path)):
            feats = np.load(features_path / vid)
            gt = np.array([actions_dict[a] for a in open(gt_path / vid.replace("npy", "txt")).read().splitlines()])
            x = torch.tensor(feats, dtype=torch.float).unsqueeze(0).to(device)
            model(x, torch.ones(x.size(), device=device))
            gts.append(gt.astype(np.int16))
            preds.append(model.indices.squeeze(0).cpu().numpy().astype(np.int16))
    _, pr2gt = create_correspondences(np.concatenate(gts), np.concatenate(preds), mapping=True)
    mapped = [np.vectorize(pr2gt.get)(pr) for pr in preds]
    return e0.score(gts, mapped), dict(tensors_equal=bool(tensors_equal), n_sequences=len(gts))


def gate_g2(dataset: str, device: torch.device) -> dict:
    saved_path = ROOT / "results" / "e0" / f"{dataset}_pretrained.json"
    if not saved_path.exists():
        return dict(status="failed", reason=f"{saved_path} not found")
    saved = json.loads(saved_path.read_text())["global"]
    got, info = smq_global_scores(dataset, device)
    diff = {m: abs(float(got[m]) - float(saved[m])) for m in G2_METRICS}
    ok = info["tensors_equal"] and all(v <= G2_TOLERANCE for v in diff.values())
    return dict(status="passed" if ok else "failed", dataset=dataset, max_abs_diff=max(diff.values()), diff=diff,
                recomputed={m: float(got[m]) for m in G2_METRICS}, saved={m: float(saved[m]) for m in G2_METRICS}, **info)


def lara_extents(max_recordings: int = 30) -> dict:
    """Median across-joint extent (max - min over joints, median over frames) of every channel, over an even
    sample of recordings."""
    spec = D.SPECS["lara"]
    names = D.recording_names(spec)
    step = max(1, len(names) // max_recordings)
    per_channel = None
    for n in names[::step]:
        arr = np.load(D.features_dir(spec) / n)  # (C, T, V, M)
        ext = np.median(arr.max(axis=2) - arr.min(axis=2), axis=1)[:, 0]  # (C,)
        per_channel = ext[None] if per_channel is None else np.vstack([per_channel, ext[None]])
    med = np.median(per_channel, axis=0)
    return dict(per_channel=[float(x) for x in med], recordings=int(per_channel.shape[0]))


def gate_g3() -> dict:
    ext = lara_extents()
    pos, ori = ext["per_channel"][3:6], ext["per_channel"][0:3]
    ok = float(np.median(pos)) > float(np.median(ori))
    return dict(status="passed" if ok else "failed", median_extent_channels_3_5=float(np.median(pos)),
                median_extent_channels_0_2=float(np.median(ori)), **ext)
