"""MASQ losses: the mask-aware velocity loss and the per-batch loss assembly (SMQ's terms plus the velocity term).

No labels are read here.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.model.utils import distance_joints

_MSE = nn.MSELoss(reduction="none")


def velocity_loss(x: torch.Tensor, x_hat: torch.Tensor, mask: torch.Tensor, joint_keep: torch.Tensor) -> torch.Tensor:
    """First-order differences of the reconstruction vs the input, on kept joints and unpadded frame pairs only.

    x, x_hat, mask: (N, C, T, V, M); joint_keep: (N, M, V) in {0, 1}. The squared L2 norm is taken over a joint's channels
    and averaged over valid (frame pair, joint) points (MASQ eq. 5: sum over t, v divided by N_vld).
    """
    dx = x[:, :, 1:] - x[:, :, :-1]
    dxh = x_hat[:, :, 1:] - x_hat[:, :, :-1]
    pv = mask[:, 0, 1:] * mask[:, 0, :-1]  # (N, T-1, V, M): both frames unpadded
    jk = joint_keep.permute(0, 2, 1)[:, None]  # (N, 1, V, M)
    w = pv * jk
    sq = ((dxh - dx) ** 2).sum(dim=1)  # (N, T-1, V, M)
    return (sq * w).sum() / w.sum().clamp_min(1.0)


def compute_losses(model, x, mask, x_hat, mse_loss_weight: float, commit_weight: float, vel_weight: float,
                   joint_distance_recons: bool = True) -> dict:
    """rec + commit + tc (+ vel), exactly as model.Trainer.train plus the velocity term (computed only if vel_weight > 0)."""
    if joint_distance_recons:
        a, b = distance_joints(x), distance_joints(x_hat)
    else:
        a, b = x, x_hat
    rec = mse_loss_weight * torch.mean(_MSE(a, b))
    commit = commit_weight * model.commit_loss
    tc = model.vq.tc_loss
    vel = x.new_zeros(())
    if vel_weight > 0:
        vel = vel_weight * velocity_loss(x, x_hat, mask, model.joint_keep)
    return dict(rec=rec, commit=commit, tc=tc, vel=vel, total=rec + commit + tc + vel)
