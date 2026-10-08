"""MASQ model: SMQ plus Joint-Level Structured Dropout (JLSD), a training-time regularizer on the encoder latents.

JLSD acts on the per-joint encoder latents, one Bernoulli draw per (sample, person, joint), broadcast over time and
channels (MASQ eq. 1): a dropped joint's whole latent trajectory is zeroed and the kept ones are scaled by 1/(1-p). The
quantizer then sees only the visible joints; the decoder reconstructs every joint. JLSD is never applied in eval mode.
With jlsd_p = 0 no random number is drawn and the forward pass equals SMQModel's exactly (gate G1).

No labels are read here.
"""
from __future__ import annotations

import torch

from script.week5.masq_experiments import config as C
from src.model.smq import SMQModel
from src.model.utils import process_mask


class MASQModel(SMQModel):
    def __init__(self, *args, jlsd_p: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 <= jlsd_p < 1.0:
            raise ValueError("jlsd_p must be in [0, 1)")
        self.jlsd_p = float(jlsd_p)
        self.joint_keep = None  # (N, M, V) of the last forward, detached

    def forward(self, x, mask):
        """SMQModel.forward (src/model/smq.py) verbatim, with the JLSD block inserted after the encoder."""
        N, C_, T, V, M = x.size()

        # Pack person+joint dims : (N, C, T, V, M) -> (N*M*V, C, T)
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N * M * V, C_, T)
        mask = mask.permute(0, 4, 3, 1, 2).contiguous().view(N * M * V, C_, T)

        # Encode joint-disentangled latents: (N*M*V, C, T) -> (N*M*V, latent_dim, T)
        self.latent = self.encoder(x, mask)

        # --- MASQ: JLSD ---
        # Row r of the packed layout is n*M*V + m*V + v, so keep.view(N, M, V) is indexed (n, m, v).
        if self.training and self.jlsd_p > 0:
            keep = (torch.rand(N * M * V, 1, 1, device=self.latent.device) >= self.jlsd_p).to(self.latent.dtype)
            latent_src = self.latent * keep / (1.0 - self.jlsd_p)
        else:
            keep = torch.ones(N * M * V, 1, 1, device=self.latent.device, dtype=self.latent.dtype)
            latent_src = self.latent
        self.joint_keep = keep.view(N, M, V).detach()
        # --- end MASQ ---

        # Repack into per-frame skeleton : (N*M*V, latent_dim, T) -> (N, T, V, M, latent_dim)
        latent = latent_src.view(N * M, V, self.latent_dim, T).contiguous()
        latent = latent.permute(0, 3, 1, 2)
        latent = latent.reshape(N, M, T, V, self.latent_dim)
        latent = latent.permute(0, 2, 3, 1, 4)

        # Concat joints for quantization : (N, T, V, M, latent_dim) -> (N, T, V*M*latent_dim)
        latent = latent.reshape(N, T, -1)

        vq_mask = process_mask(mask, batch_size=N, num_person=M, num_joints=self.num_joints, seq_length=T,
                               num_features_per_joint=self.latent_dim)

        quantized, self.indices, self.commit_loss, _ = self.vq(latent, vq_mask)

        quantized = quantized.reshape(N, T, V, M * self.latent_dim)
        quantized = quantized.reshape(N, T, V, M, self.latent_dim).permute(0, 3, 4, 1, 2)
        quantized = quantized.reshape(N * M, self.latent_dim, T, V)
        quantized = quantized.permute(0, 3, 1, 2)
        quantized = quantized.reshape(N * M * V, self.latent_dim, T)

        decoded = self.decoder(quantized, mask)

        decoded = decoded.reshape(N * M, V, C_, T)
        decoded = decoded.reshape(N, M, V, C_, T)
        return decoded.permute(0, 3, 4, 2, 1)


def build_masq(dataset: str, num_actions: int, jlsd_p: float = 0.0, grad_checkpoint: bool = False,
               **overrides) -> MASQModel:
    """The SMQ constructor arguments of main.py for `dataset` (declared settings), plus JLSD."""
    d, s = C.DATASETS[dataset], dict(C.SHARED, **overrides)
    model = MASQModel(in_channels=d["num_features"], filters=s["num_f_maps"], num_layers=s["num_layers"],
                      latent_dim=s["latent_dim"], num_actions=num_actions, num_joints=d["num_joints"],
                      num_person=d["num_person"], patch_size=overrides.get("patch_size", d["patch_size"]),
                      kmeans=s["kmeans"], kmeans_metric=s["kmeans_metric"], sampling_quantile=s["sampling_quantile"],
                      replacement_strategy=s["replacement_strategy"], decay=s["decay"],
                      dead_code_threshold=s["dead_code_threshold"], tc_weight=s["tc_weight"], tc_beta=s["tc_beta"],
                      jlsd_p=jlsd_p)
    for m in model.modules():  # as model.Trainer: activation checkpointing in every TCN stage (memory only)
        if hasattr(m, "grad_checkpoint"):
            m.grad_checkpoint = grad_checkpoint
    return model
