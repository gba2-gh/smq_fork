"""Scale-free velocity-term variants for the HuGaDB-focused follow-up (docs/plans/week5/masq_experiments/TIER0_RESULTS_AND_HUGADB_FOCUS.md).

Why: in masq_v1 the velocity loss of MASQ's equation 5 (decoded vs true first differences in coordinate space) was
unlearnable at the patch bottleneck (velocity loss / true velocity energy = 1.00, correlation about 0, loss flat over 30
epochs on HuGaDB) and its size depended on the data scale (about 2x the reconstruction loss on HuGaDB, about 0.06x on BABEL).

Both variants add ONE term to SMQ's loss (reconstruction + commit), with JLSD at the published dataset rate, and set the
term's weight by a label-free rule at initialization: weight x term(init) = rho x reconstruction(init), where
reconstruction(init) is SMQ's weighted reconstruction loss for an all-zero decoder output on a fixed sample of training
inputs. Every quantity is dimensionless after this calibration, so the weight does not depend on the data scale.

  V1   (coord_scalefree) equation 5 as written (coordinate-space velocity matching), weight from the rule; rho = rho_pub.
  V2a/V2b (latent_smooth) a smoothness prior on the pre-quantization latents of visible joints, the other reading of the
       paper's text ("first-order smoothness prior ... in a continuous feature space"): sum ||z_t - z_{t-1}||^2 over visible
       joints and unpadded frame pairs, divided by the energy sum ||z_t||^2 (scale invariant: shrinking the latents cannot
       reduce it); rho 0.01 and 0.1.

No labels are read here.
"""
from __future__ import annotations

import torch

from script.week5.masq_experiments.losses import velocity_loss
from src.model.utils import distance_joints


def latent_smoothness(latent: torch.Tensor, mask: torch.Tensor, joint_keep: torch.Tensor) -> torch.Tensor:
    """latent (N*M*V, Z, T) as produced by the encoder; mask (N, C, T, V, M); joint_keep (N, M, V) in {0, 1}.

    Ratio of the first-difference energy to the energy of the latent trajectory, over visible joints and unpadded frame
    pairs. Row r of the packed latent is n*M*V + m*V + v, as everywhere in the model."""
    N, _, T, V, M = mask.shape
    pv = mask[:, 0, 1:] * mask[:, 0, :-1]  # (N, T-1, V, M)
    w = pv.permute(0, 3, 2, 1).reshape(N * M * V, T - 1) * joint_keep.reshape(N * M * V, 1)
    d = latent[:, :, 1:] - latent[:, :, :-1]
    num = ((d ** 2).sum(dim=1) * w).sum()
    den = ((latent[:, :, 1:] ** 2).sum(dim=1) * w).sum().clamp_min(1e-12)
    return num / den


def initial_reconstruction(x: torch.Tensor, mse_loss_weight: float) -> torch.Tensor:
    """SMQ's weighted joint-distance reconstruction loss for an all-zero decoder output."""
    return mse_loss_weight * torch.mean(distance_joints(x) ** 2)


def calibrate_weight(mode: str, rho: float, x: torch.Tensor, mask: torch.Tensor, mse_loss_weight: float, latent=None) -> dict:
    """weight such that weight x term(init) = rho x reconstruction(init). `latent` is required for 'latent_smooth'."""
    N, _, T, V, M = x.shape
    rec_init = float(initial_reconstruction(x, mse_loss_weight))
    keep = torch.ones(N, M, V, dtype=x.dtype, device=x.device)
    if mode == "coord_scalefree":
        term_init = float(velocity_loss(x, torch.zeros_like(x), mask, keep))
    elif mode == "latent_smooth":
        term_init = float(latent_smoothness(latent, mask, keep))
    else:
        raise ValueError(mode)
    return dict(mode=mode, rho=rho, rec_init=rec_init, term_init=term_init, weight=rho * rec_init / term_init)


def variant_term(mode: str, weight: float, model, x, mask, x_hat) -> torch.Tensor:
    if mode == "coord_scalefree":
        return weight * velocity_loss(x, x_hat, mask, model.joint_keep)
    if mode == "latent_smooth":
        return weight * latent_smoothness(model.latent, mask, model.joint_keep)
    raise ValueError(mode)
