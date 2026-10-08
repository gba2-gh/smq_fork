"""Assembles the five families' per-recording records (code ids, continuous
features, window starts/lengths) and their fixed [num_codes, feature_dim]
quantized-vector lookup tables, from the extracted populations (extract.py)
and fitted preprocessing (fit.py).
"""

from __future__ import annotations

import numpy as np
import torch

from script.action_transport import config as at_config
from script.fsq_reconstruction.experiment import DATASET_DEFAULTS
from script.quantized_input_comparison import config as cfg
from script.quantized_input_comparison import fit as fitmod


def window_starts(num_patches: int, window: int) -> np.ndarray:
    return (np.arange(num_patches, dtype=np.int32) * window)


def vq_native_lookup(vq_model) -> np.ndarray:
    """[K, W*D] float32: the VQ codebook, flattened exactly as the encoder
    patch it replaces (`_embedding` is (K, W, D))."""
    emb = vq_model.vq._embedding.detach().cpu().numpy()
    k = emb.shape[0]
    return emb.reshape(k, -1).astype(np.float32)


def fsq_native_lookup(fsq_model) -> np.ndarray:
    """[K, W*D] float32: every one of the K grid points, run through the
    frozen `out_proj` -- the same map the FSQ adapter applies to a real
    quantized patch, evaluated exhaustively rather than only at codes that
    happen to occur in the data."""
    adapter = fsq_model.vq
    device = adapter.out_proj.weight.device
    idx = torch.arange(adapter.num_embeddings, dtype=torch.long, device=device)
    zhat = adapter.fsq.indexes_to_codes(idx).to(dtype=adapter.out_proj.weight.dtype, device=device)
    with torch.no_grad():
        vecs = adapter.out_proj(zhat)
    return vecs.detach().cpu().numpy().astype(np.float32)


def kmeans_lookup(pop: fitmod.FittedPopulation) -> np.ndarray:
    """[K, pca_dim] float32: the fitted K-means cluster centers, in the
    population's PCA space (the "assigned PCA-space centroid" -- never the
    empirical per-code mean)."""
    return pop.kmeans.cluster_centers_.astype(np.float32)


def build_family_records(family: str, source: dict[str, dict], window: int,
                          pop: fitmod.FittedPopulation | None) -> dict:
    """Returns {name: dict(codes int64[P], continuous float32[P,d],
    starts int32[P], lengths int32[P])} for one family, given only the ONE
    extracted population `source` that this family is defined over
    (`cfg.FAMILY_POPULATION[family]`) -- never the other two populations,
    so the caller can hold a single population's per-window arrays resident
    at a time. `pop` (the fitted preprocessing for that population) is
    required for "kmeans" families and unused for "native" families."""
    kind = cfg.FAMILY_KIND[family]
    out = {}
    if kind == "native":
        for name, rec in source.items():
            out[name] = dict(codes=rec["ids"].astype(np.int64), continuous=rec["flat"].astype(np.float32),
                              starts=window_starts(rec["num_patches"], window), lengths=rec["lengths"].astype(np.int32))
    else:
        for name, rec in source.items():
            pca_feats, ids = fitmod.assign_kmeans(pop, rec["flat"])
            out[name] = dict(codes=ids, continuous=pca_feats, starts=window_starts(rec["num_patches"], window),
                              lengths=rec["lengths"].astype(np.int32))
    return out


def build_lookup(family: str, vq_model, fsq_model, pops: dict[str, fitmod.FittedPopulation]) -> np.ndarray:
    if family == "vq_native":
        return vq_native_lookup(vq_model)
    if family == "fsq_native":
        return fsq_native_lookup(fsq_model)
    return kmeans_lookup(pops[cfg.FAMILY_POPULATION[family]])
