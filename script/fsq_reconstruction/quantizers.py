"""FSQ math and the SMQ-compatible patch adapter
(docs/plans/FSQ_RECONSTRUCTION_PILOT_PLAN.md Revision 3, §5), plus the RNG
isolation and matched-initialization helpers §4 requires.

`PatchFSQAdapter` implements the exact diagram in §5:

    SMQ encoder -> [W,D] -> flatten [W*D]
                -> Linear(W*D,3)
                -> FSQ bound + round with straight-through gradients
                -> three quantized scalars, levels [8,8,8]
                -> Linear(3,W*D) -> reshape [W,D]
                -> existing SMQ decoder

Its forward(x, mask) has the same signature and return contract as
`SkeletonMotionQuantizer.forward` (src/model/motion_quantizer.py): it can be
substituted for `self.vq` on an `SMQModel` instance without changing
`SMQModel.forward` or `model.Trainer`.

FSQ formulas follow the construction in Mentzer et al. 2023 ("Finite Scalar
Quantization: VQ-VAE Made Simple"), independently re-derived from the paper
and its published reference algorithm rather than executed against the JAX
implementation (docs/plans/FSQ_RECONSTRUCTION_PILOT_PLAN.md §5: "formula
parity ... not ... execution parity with JAX").
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.smq import SMQModel

FSQ_EPS = 1e-3


class FSQ:
    """Per-coordinate bounded rounding onto a fixed product grid, plus a
    mixed-radix map from a grid point to a single integer index in
    [0, prod(levels)).

    `levels[i]` is the number of quantization levels on coordinate i. For
    [8,8,8] the codebook size is 512, matching the pilot's alphabet."""

    def __init__(self, levels: tuple[int, ...], eps: float = FSQ_EPS):
        self.levels = tuple(int(l) for l in levels)
        if any(l < 2 for l in self.levels):
            raise ValueError("every FSQ level count must be >= 2")
        self.eps = eps
        self.codebook_size = int(np.prod(self.levels))
        basis = np.concatenate([[1], np.cumprod(np.asarray(self.levels[:-1], dtype=np.int64))])
        self._basis_np = basis.astype(np.int64)

    def _levels_tensor(self, like: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.levels, dtype=like.dtype, device=like.device)

    def _basis_tensor(self, like: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self._basis_np, dtype=like.dtype, device=like.device)

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound z (..., d) into the open interval matched to each
        coordinate's level count, via a shifted/scaled tanh."""
        levels = self._levels_tensor(z)
        half_l = (levels - 1) * (1 - self.eps) / 2
        offset = torch.where(levels % 2 == 0, torch.full_like(levels, 0.5), torch.zeros_like(levels))
        shift = torch.tan(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Bound, round with a straight-through gradient, and renormalize
        each coordinate to [-1, 1]. Differentiable end to end (the round
        itself is the only non-differentiable op, handled by STE)."""
        levels = self._levels_tensor(z)
        bounded = self.bound(z)
        rounded = bounded.round()
        rounded_ste = bounded + (rounded - bounded).detach()
        half_width = torch.div(levels, 2, rounding_mode="floor")
        return rounded_ste / half_width

    def codes_to_indexes(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        """zhat_normalized (..., d) in [-1,1] grid coordinates (the output of
        `quantize`) -> mixed-radix integer index in [0, codebook_size)."""
        levels = self._levels_tensor(zhat_normalized)
        basis = self._basis_tensor(zhat_normalized)
        half_width = torch.div(levels, 2, rounding_mode="floor")
        zhat = zhat_normalized * half_width + half_width  # unsigned per-coordinate digit, in [0, level-1]
        index = (zhat.round() * basis).sum(dim=-1)
        return index.round().long()

    def indexes_to_codes(self, indexes: torch.Tensor) -> torch.Tensor:
        """Inverse of codes_to_indexes: integer index -> zhat_normalized
        (..., d) in [-1,1]. Used only by tests to check the round trip."""
        idx = indexes.clone()
        digits = []
        for level in self.levels:
            digits.append(idx % level)
            idx = torch.div(idx, level, rounding_mode="floor")
        digits = torch.stack(digits, dim=-1).to(torch.float64)
        half_width = torch.tensor([l // 2 for l in self.levels], dtype=torch.float64, device=indexes.device)
        return (digits - half_width) / half_width


class PatchFSQAdapter(nn.Module):
    """Drop-in replacement for `SkeletonMotionQuantizer` (same
    forward(x, mask) -> (quantize, encoding_indices, loss, distances)
    contract). No codebook, no EMA, no dead-code replacement: the quantizer
    is a fixed product grid plus two learned linear adapters."""

    def __init__(self, embedding_dim: int, window: int, levels: tuple[int, ...] = (8, 8, 8)):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.window = window
        self.levels = tuple(levels)
        self.fsq = FSQ(levels)
        self.num_embeddings = self.fsq.codebook_size
        d = len(levels)
        self.in_proj = nn.Linear(window * embedding_dim, d, bias=True)
        # Raw SMQ latents are not unit-scale (HuGaDB patches run to |z| in the
        # thousands), which saturates FSQ's tanh bound almost everywhere and
        # kills the reconstruction gradient into the encoder from step one.
        # A fixed (non-affine) LayerNorm keeps the tanh input at unit scale
        # regardless of what in_proj/the encoder learn, so it can't drift
        # back into saturation the way a learned affine norm could.
        self.pre_quant_norm = nn.LayerNorm(d, elementwise_affine=False)
        self.out_proj = nn.Linear(d, window * embedding_dim, bias=True)
        self.tc_loss = torch.zeros(())  # zero, every forward call; FSQ has no temporal-consistency term
        self.last_pre_bound = None  # [N_valid, d], post-norm pre-tanh activation, for saturation diagnostics

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """x: (B,T,D) float; mask: (B,T,D) float/bool, 1=valid, 0=pad.
        Returns (quantize [B,T,D], encoding_indices [B,T], loss (zero
        scalar), distances (None -- FSQ has no VQ-style per-code distance,
        per §5: 'make that explicit rather than fabricating distances')."""
        B, T, D = x.shape
        W = self.window
        remainder = T % W
        padding_needed = (W - remainder) if remainder != 0 else 0

        x_pad = F.pad(x, (0, 0, 0, padding_needed), mode="constant", value=0)
        mask_pad = F.pad(mask, (0, 0, 0, padding_needed), mode="constant", value=0)
        _, T_pad, _ = x_pad.shape
        P = T_pad // W

        x_patches = x_pad.reshape(B * P, W, D)
        mask_patches = mask_pad.reshape(B * P, W, D)
        valid_patch_mask = mask_patches.sum(dim=(1, 2)) > 0
        valid_patches = x_patches[valid_patch_mask]

        flat = valid_patches.reshape(valid_patches.shape[0], W * D)
        z = self.pre_quant_norm(self.in_proj(flat))
        zhat_normalized = self.fsq.quantize(z)  # differentiable (STE); feeds the decoder
        with torch.no_grad():
            encoding_indices_valid = self.fsq.codes_to_indexes(zhat_normalized.detach())
        self.last_pre_bound = z.detach()
        recon_flat = self.out_proj(zhat_normalized)
        quantize_valid = recon_flat.reshape(-1, W, D)

        quantized_patches = torch.zeros_like(x_patches)
        quantized_patches[valid_patch_mask] = quantize_valid
        quantize_pad = quantized_patches.reshape(B, T_pad, D)

        end = -padding_needed if padding_needed > 0 else None
        quantize = quantize_pad[:, :end, :]

        loss = x.new_zeros(())  # FSQ has no commitment term
        self.tc_loss = x.new_zeros(())

        patch_indices = torch.zeros(B * P, dtype=torch.int64, device=x.device)
        patch_indices[valid_patch_mask] = encoding_indices_valid
        encoding_indices = patch_indices.reshape(B, P).repeat_interleave(W, dim=1)[:, :end]

        return quantize.contiguous(), encoding_indices, loss, None


class FSQSMQModel(SMQModel):
    """SMQModel with `self.vq` replaced by a `PatchFSQAdapter`. Encoder and
    decoder are built identically to SMQModel (same constructor, same
    architecture); only the quantizer differs. `SMQModel.forward` calls
    `self.vq(latent, vq_mask)` generically, so no override is needed here."""

    def __init__(self, *, fsq_levels: tuple[int, ...] = (8, 8, 8), **smq_kwargs):
        super().__init__(**smq_kwargs)  # builds a throwaway SkeletonMotionQuantizer, discarded below
        embedding_dim = self.latent_dim * self.num_joints * self.num_person
        self.vq = PatchFSQAdapter(embedding_dim=embedding_dim, window=smq_kwargs["patch_size"],
                                  levels=fsq_levels)


# --- RNG isolation and matched initialization (§4) -----------------------------------

@contextlib.contextmanager
def isolated_rng(seed: int):
    """Saves Python/NumPy/torch CPU+CUDA RNG state, seeds all four to `seed`,
    runs the block, then restores the saved state exactly. Guarantees that
    whatever randomness the block consumes never desynchronizes the shared
    training-time RNG stream the two matched arms rely on."""
    import random
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())] \
        if torch.cuda.is_available() else []
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        for i, state in enumerate(cuda_states):
            torch.cuda.set_rng_state(state, i)


def reseed_all(seed: int) -> None:
    """Reseeds Python/NumPy/torch CPU+CUDA identically. Called once,
    immediately before training each matched arm, so their dropout masks
    (the only global-RNG consumer inside SMQModel.forward for either
    quantizer -- see README "RNG matching") draw from the same stream given
    the same fixed batch order."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def eager_init_vq(vq_module) -> None:
    """Replicates `SkeletonMotionQuantizer.init_embed_`'s uniform-init branch
    (kaiming_uniform_, embed_avg = embed, cluster_size = 1) eagerly, so the
    codebook is valid before epoch 0 rather than being built lazily on the
    first forward pass (§4.1: 'do not consume a training update merely to
    initialize them'). Caller wraps this in `isolated_rng` so the
    kaiming_uniform_ draw does not touch the shared training RNG stream."""
    with torch.no_grad():
        embed = torch.empty_like(vq_module._embedding)
        nn.init.kaiming_uniform_(embed)
        vq_module._embedding.data.copy_(embed)
        vq_module.embed_avg.data.copy_(embed)
        vq_module.cluster_size.data.fill_(1.0)
        vq_module.initted.data.copy_(torch.tensor([True]))


def reinit_fsq_adapters(fsq_vq: PatchFSQAdapter) -> None:
    """Re-draws in_proj/out_proj weights (PyTorch's default Linear init).
    Caller wraps this in `isolated_rng` (§4.1: 'initialize FSQ adapters in
    isolated CPU/CUDA RNG contexts')."""
    fsq_vq.in_proj.reset_parameters()
    fsq_vq.out_proj.reset_parameters()


@dataclass(frozen=True)
class SharedBackbone:
    encoder_state: dict
    decoder_state: dict
    init_hash: str


def build_shared_backbone(smq_kwargs: dict, seed: int) -> SharedBackbone:
    """Builds one canonical SMQModel inside an isolated RNG context and
    returns its encoder/decoder state dicts (a throwaway quantizer is built
    and discarded). Both matched arms load these exact tensors, so encoder
    and decoder start identical (§4: 'Copy exactly identical initial encoder
    and decoder tensors into the two models')."""
    import copy
    import hashlib
    import io
    with isolated_rng(seed):
        template = SMQModel(**smq_kwargs)
    encoder_state = copy.deepcopy(template.encoder.state_dict())
    decoder_state = copy.deepcopy(template.decoder.state_dict())
    buf = io.BytesIO()
    torch.save({"encoder": encoder_state, "decoder": decoder_state}, buf)
    init_hash = hashlib.sha256(buf.getvalue()).hexdigest()
    return SharedBackbone(encoder_state, decoder_state, init_hash)


def build_vq_arm(smq_kwargs: dict, backbone: SharedBackbone, init_seed: int) -> SMQModel:
    model = SMQModel(**smq_kwargs)
    model.encoder.load_state_dict(backbone.encoder_state)
    model.decoder.load_state_dict(backbone.decoder_state)
    with isolated_rng(init_seed):
        eager_init_vq(model.vq)
    return model


def build_fsq_arm(smq_kwargs: dict, backbone: SharedBackbone, init_seed: int,
                  levels: tuple[int, ...]) -> FSQSMQModel:
    model = FSQSMQModel(fsq_levels=levels, **smq_kwargs)
    model.encoder.load_state_dict(backbone.encoder_state)
    model.decoder.load_state_dict(backbone.decoder_state)
    with isolated_rng(init_seed):
        reinit_fsq_adapters(model.vq)
    return model
