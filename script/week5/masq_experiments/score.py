"""One scoring unit: the prediction loop and metrics of gates5.smq_global_scores / e0_metrics, plus patch latents.

Labels are read here (evaluation only). The checkpoint is always the final epoch.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from script.repro import e0_metrics as e0
from script.week5.masq_experiments import config as C
from script.week5.masq_experiments.model_masq import build_masq
from script.week5.masq_experiments.train_masq import data_paths
from src.model.eval_utils import create_correspondences, read_mapping_file
from src.model.utils import get_num_actions


def _segments(a: np.ndarray) -> int:
    return int(1 + (np.diff(a) != 0).sum()) if len(a) else 0


def _patch_means(latent: torch.Tensor, patch: int) -> np.ndarray:
    """latent (N*M*V, Z, T) for one recording (N = M = 1) -> (patches, V*Z), averaged over non-overlapping patches."""
    nvm, Z, T = latent.shape
    per_frame = latent.permute(2, 0, 1).reshape(T, nvm * Z)  # (T, V*Z), joint-major channels, as the quantizer sees them
    out = [per_frame[s:s + patch].mean(dim=0) for s in range(0, T, patch)]
    return torch.stack(out).float().cpu().numpy()


def predict(model, dataset: str, device: torch.device, with_latents: bool = False):
    """Frame-level codes for every recording (sorted by file name), in eval mode with an all-ones mask."""
    paths = data_paths(dataset)
    mapping = read_mapping_file(paths["mapping"])
    actions_dict = {v: k for k, v in mapping.items()}
    patch = C.DATASETS[dataset]["patch_size"]
    names, gts, preds, lats = [], [], [], []
    model.eval()
    with torch.no_grad():
        for vid in sorted(os.listdir(paths["features"])):
            feats = np.load(paths["features"] / vid)
            gt = np.array([actions_dict[a] for a in open(paths["gt"] / vid.replace("npy", "txt")).read().splitlines()])
            x = torch.tensor(feats, dtype=torch.float).unsqueeze(0).to(device)
            model(x, torch.ones(x.size(), device=device))
            names.append(vid)
            gts.append(gt.astype(np.int16))
            preds.append(model.indices.squeeze(0).cpu().numpy().astype(np.int16))
            if with_latents:
                lats.append(_patch_means(model.latent, patch))
    return names, gts, preds, lats


def metrics_from(gts, preds) -> dict:
    _, pr2gt = create_correspondences(np.concatenate(gts), np.concatenate(preds), mapping=True)
    glob = e0.score(gts, [np.vectorize(pr2gt.get)(p) for p in preds])
    local = e0.score(gts, [create_correspondences(g, p) for g, p in zip(gts, preds)])
    return dict(global_=glob, local=local)


def code_stats(preds, num_codes: int) -> dict:
    counts = np.bincount(np.concatenate(preds).astype(np.int64), minlength=num_codes).astype(np.float64)
    p = counts / counts.sum()
    nz = p[p > 0]
    return dict(num_codes=int(num_codes), occupied=int((counts > 0).sum()), perplexity=float(np.exp(-(nz * np.log(nz)).sum())))


def score_unit(unit: C.Unit, ckpt_path: Path, out_dir: Path, device: torch.device, identity: str | None = None,
               smoke: bool = False) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_actions = get_num_actions(data_paths(unit.dataset)["gt"])
    model = build_masq(unit.dataset, num_actions, jlsd_p=unit.jlsd_p)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    t0 = time.perf_counter()
    names, gts, preds, lats = predict(model, unit.dataset, device, with_latents=True)
    m = metrics_from(gts, preds)
    res = {"global": m["global_"], "local": m["local"], "n_sequences": len(gts),
           "segments": dict(mean_pred=float(np.mean([_segments(p) for p in preds])), mean_gt=float(np.mean([_segments(g) for g in gts]))),
           "codes": code_stats(preds, num_actions), "identity": identity, "smoke": smoke, "unit_id": unit.unit_id,
           "seconds": time.perf_counter() - t0, "seconds_per_sequence": (time.perf_counter() - t0) / max(len(gts), 1)}
    obj = lambda xs: np.array(xs, dtype=object)  # noqa: E731
    npz = out_dir / "predictions.npz"
    tmp = out_dir / "predictions.tmp.npz"
    np.savez(tmp, names=np.array(names), gt=obj(gts), pred=obj(preds), patch_latents=obj(lats))
    os.replace(tmp, npz)
    C.write_json_atomic(out_dir / "scores.json", res)
    return res


def released_checkpoint_scores(dataset: str, device: torch.device) -> dict:
    """Gate G4: the released SMQ checkpoint, loaded into MASQModel(jlsd_p=0), through this module's scorer."""
    num_actions = get_num_actions(data_paths(dataset)["gt"])
    model = build_masq(dataset, num_actions, jlsd_p=0.0)
    result = model.load_state_dict(torch.load(C.ROOT / "models" / "pretrained" / f"{dataset}.model", map_location=device))
    model.to(device)
    names, gts, preds, _ = predict(model, dataset, device)
    m = metrics_from(gts, preds)
    return dict(global_=m["global_"], local=m["local"], missing=list(result.missing_keys), unexpected=list(result.unexpected_keys))
