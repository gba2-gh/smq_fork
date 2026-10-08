"""Temporal embedding network and prototypes. Explicit erf GELU (torch 2.1 CPU F.gelu backward bug, see memory note)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from script.self_labelled import config as C


def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x * 0.7071067811865476))


class Block(nn.Module):
    def __init__(self, dim: int, kernel: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel, padding=kernel // 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:           # x [B, L, D], mask [B, L, 1]
        h = self.norm(x) * mask
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        return x + gelu(h) * mask


class Encoder(nn.Module):
    """windows [B, L, 1536] (standardized) -> unit embeddings [B, L, 128]. Padded positions are zero and never mix into valid ones:
    zero padding at recording ends, exactly like a convolution at the edge."""

    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(C.IN_DIM, C.HID)
        self.blocks = nn.ModuleList([Block(C.HID, C.KERNEL) for _ in range(C.N_BLOCKS)])
        self.out = nn.Linear(C.HID, C.OUT_DIM)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.inp(x) * mask
        for blk in self.blocks:
            h = blk(h, mask)
        return F.normalize(self.out(h), dim=-1) * mask


def receptive_field_windows() -> int:
    return C.N_BLOCKS * (C.KERNEL // 2)


def init_prototypes(z: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    """k-means++ centres of unit embeddings z [N, D] (label-free, seeded), re-normalized."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, init="k-means++", n_init=5, random_state=seed).fit(z.cpu().numpy())
    return F.normalize(torch.as_tensor(km.cluster_centers_, dtype=torch.float32), dim=-1)
