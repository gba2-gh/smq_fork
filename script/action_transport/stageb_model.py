"""B-code: the masked-code Transformer (docs/plans/THREE_STAGE_MOTION_PLAN.md
§4). Input is code IDs and a learned mask token; targets are the same K=500
codes at masked positions only. Sinusoidal absolute positions. The
representation is the final encoder hidden state, before the prediction
head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from script.action_transport.stageb_data import MAX_CONTEXT


@dataclass(frozen=True)
class ModelConfig:
    num_codes: int = 500
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_context: int = MAX_CONTEXT
    norm_first: bool = False  # False = PyTorch's default post-LN (the v1.0-v1.5 model)
    tie_head: bool = False  # True = output logits use the input code embeddings (BERT-style tying)
    # "sinusoidal" (plan default) | "alibi" (symmetric relative bias) | "none" (no positional
    # information at all: no sinusoidal addition and no attention bias)
    position_mode: str = "sinusoidal"
    # --- Task 2 (quantized_input_comparison) extension: vector inputs -----------------
    # "id" (default): unchanged nn.Embedding(K+1, d_model) lookup path -- byte-identical
    # to every pre-Task-2 model. "vector": a learned nn.Linear(input_dim, d_model)
    # projection + a learned mask token replace masked positions' input entirely,
    # forward via `forward_vectors` (context_quantized/context_continuous). Note
    # context_id and context_geometry_init BOTH use input_mode="id" -- geometry_init
    # is architecturally identical to a plain trainable embedding table, differing
    # only in its initial weight values (set externally, see
    # script/quantized_input_comparison/contextual.py), so it needs no new branch here.
    input_mode: str = "id"  # "id" | "vector"
    input_dim: int | None = None  # required when input_mode == "vector"


def sinusoidal_positions(length: int, d_model: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
                    * (-math.log(10000.0) / d_model))
    pe = torch.zeros(length, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe


class CodeContextModel(nn.Module):
    """Embedding(K+1) [code ids 0..K-1, mask id K] -> + sinusoidal position
    -> TransformerEncoder -> linear head to K classes. Padded positions are
    excluded from self-attention via a key-padding mask; they are never a
    target and never replaced by the mask token."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.mask_id = cfg.num_codes  # one extra embedding row for the learned mask token
        if cfg.input_mode == "vector":
            if not cfg.input_dim:
                raise ValueError("input_mode='vector' requires input_dim")
            self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
            self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))
        elif cfg.input_mode == "id":
            self.embedding = nn.Embedding(cfg.num_codes + 1, cfg.d_model)
        else:
            raise ValueError(cfg.input_mode)
        self.dropout = nn.Dropout(cfg.dropout)
        layer = nn.TransformerEncoderLayer(d_model=cfg.d_model, nhead=cfg.n_heads,
                                           dim_feedforward=cfg.d_ff, dropout=cfg.dropout,
                                           batch_first=True, norm_first=cfg.norm_first)
        # Pre-LN needs a final LayerNorm on the encoder output; post-LN already ends in one.
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers,
                                             norm=nn.LayerNorm(cfg.d_model) if cfg.norm_first else None,
                                             enable_nested_tensor=not cfg.norm_first and cfg.position_mode == "sinusoidal")
        if cfg.position_mode == "alibi":
            # ALiBi slopes 2^(-8h/H), h=1..H; bias -slope*|i-j| on attention scores
            slopes = torch.tensor([2.0 ** (-8.0 * h / cfg.n_heads) for h in range(1, cfg.n_heads + 1)])
            self.register_buffer("alibi_slopes", slopes, persistent=False)
        elif cfg.position_mode not in ("sinusoidal", "none"):
            raise ValueError(cfg.position_mode)
        if cfg.tie_head:
            self.head_bias = nn.Parameter(torch.zeros(cfg.num_codes))
        else:
            self.head = nn.Linear(cfg.d_model, cfg.num_codes)

    def _encode_with_bias(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        """PyTorch's reference TransformerEncoderLayer computation, spelled out.

        torch 2.1's eval-mode fast path silently drops a float attention bias
        (eval outputs became position-blind and produced NaNs), so ALiBi mode
        never calls the fused kernels: attention runs with need_weights=True,
        which forces the reference path in both train and eval mode."""
        for layer in self.encoder.layers:
            def attend(y):
                return layer.self_attn(y, y, y, attn_mask=attn_bias, need_weights=True,
                                       average_attn_weights=False)[0]

            def feed_forward(y):
                return layer.linear2(layer.dropout(layer.activation(layer.linear1(y))))

            if self.cfg.norm_first:
                x = x + layer.dropout1(attend(layer.norm1(x)))
                x = x + layer.dropout2(feed_forward(layer.norm2(x)))
            else:
                x = layer.norm1(x + layer.dropout1(attend(x)))
                x = layer.norm2(x + layer.dropout2(feed_forward(x)))
        if self.encoder.norm is not None:
            x = self.encoder.norm(x)
        return x

    def _run_backbone(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Shared encoder dispatch (position mode + backbone), independent of
        how `x` [B,L,d_model] (already the pre-encoder embedding, before
        dropout) was produced -- identical code for the id and vector input
        paths, so a matched pair's backbone genuinely runs the same forward
        computation once their embeddings agree."""
        batch, length, _ = x.shape
        if self.cfg.position_mode == "sinusoidal":
            # Dropout applied to (embedding + position), matching the
            # original id-only forward()'s exact order -- NOT dropout(embed)
            # + position, which would be a different (undropped-out-position)
            # computation and break id-path backward compatibility.
            positions = sinusoidal_positions(length, self.cfg.d_model, x.device)
            x = self.dropout(x + positions.unsqueeze(0))
            hidden = self.encoder(x, src_key_padding_mask=padding_mask)
        elif self.cfg.position_mode == "none":
            # No positional signal of any kind (neither sinusoidal nor ALiBi); only padded
            # keys are excluded. Never falls back to ALiBi.
            x = self.dropout(x)
            hidden = self.encoder(x, src_key_padding_mask=padding_mask)
        else:
            x = self.dropout(x)
            idx = torch.arange(length, device=x.device)
            distance = (idx[:, None] - idx[None, :]).abs().float()
            bias = -self.alibi_slopes[:, None, None] * distance  # [H, L, L]
            bias = bias.unsqueeze(0).expand(batch, -1, -1, -1).clone()
            bias = bias.masked_fill(padding_mask[:, None, None, :], float("-inf"))  # padded keys
            hidden = self._encode_with_bias(x, bias.reshape(batch * self.cfg.n_heads, length, length))
        return hidden

    def _head(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.cfg.tie_head:
            # the mask-token row (index K) is not a prediction target
            return hidden @ self.embedding.weight[:self.cfg.num_codes].T + self.head_bias
        return self.head(hidden)

    def forward(self, input_codes: torch.Tensor, padding_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """input_codes: [B,L] int64, already with mask_id substituted at
        masked positions (padded positions hold code 0, ignored via the
        mask below). padding_mask: [B,L] bool, True where PADDED (to match
        torch's `src_key_padding_mask` convention). Requires input_mode="id"
        (also used, unmodified, by context_geometry_init).

        Returns (hidden [B,L,d_model], logits [B,L,K])."""
        if self.cfg.input_mode != "id":
            raise ValueError("forward() requires input_mode='id'; use forward_vectors() for 'vector'")
        x = self.embedding(input_codes)
        hidden = self._run_backbone(x, padding_mask)
        return hidden, self._head(hidden)

    def forward_vectors(self, input_vectors: torch.Tensor, mask_bool: torch.Tensor,
                        padding_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """input_vectors: [B,L,input_dim] float32, the RAW (already
        RMS-scaled) per-position feature -- frozen quantized lookup rows for
        context_quantized, or continuous per-window features for
        context_continuous -- BEFORE projection, at every position including
        masked ones (the caller need not zero them out). mask_bool: [B,L]
        bool, True at positions whose PROJECTED input is replaced completely
        by the learned mask token (spec: "Replace masked projected inputs
        completely with the learned mask token"). padding_mask: as in
        forward(). Requires input_mode="vector".

        Returns (hidden [B,L,d_model], logits [B,L,K])."""
        if self.cfg.input_mode != "vector":
            raise ValueError("forward_vectors() requires input_mode='vector'; use forward() for 'id'")
        projected = self.input_proj(input_vectors)
        mask_tok = self.mask_token.to(dtype=projected.dtype).expand_as(projected)
        x = torch.where(mask_bool.unsqueeze(-1), mask_tok, projected)
        hidden = self._run_backbone(x, padding_mask)
        return hidden, self._head(hidden)
