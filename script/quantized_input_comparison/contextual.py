"""Task 2: the masked-code contextualizer, generalized to accept quantized/
continuous vector inputs in addition to the historical code-id path, on top
of Task 1's artifacts (`artifacts.Loader`) and Task 1's receptive-field
masking eligibility helper (`masking.build_eligible_mask`).

Reuses `script.action_transport.stageb_model.CodeContextModel` (extended,
see that module's docstring, to keep the id-only path byte-identical) --
the existing selected architecture (width 128, 4 layers, 4 heads, ff 512,
post-LN, symmetric ALiBi, dropout 0.1, untied head), resolved by name via
`script.action_transport.stageb_configs.resolve_by_name` exactly like
`script/fsq_reconstruction/evaluate.py` does, so this is not a second copy
of that configuration.

Six contextual/direct arm identities per family (spec):
    context_id            trainable nn.Embedding, historical path
    context_quantized      frozen lookup/RMS -> learned projection
    context_continuous     continuous/RMS -> learned projection (same
                            projection architecture and initial weights as
                            context_quantized)
    context_geometry_init   trainable nn.Embedding identical in architecture
                            to context_id, initial weights set to
                            W(q_c/RMS)+b from context_quantized's INITIAL
                            (untrained) projection, then fully independent
    direct_id / direct_quantized / direct_continuous have no contextualizer
    at all (handled in ot_arms.py, not here).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from script.action_transport.budget import check_deadline
from script.action_transport.stageb_configs import resolve_by_name
from script.action_transport.stageb_data import chunk_starts, stable_seed
from script.action_transport.stageb_model import CodeContextModel, ModelConfig
from script.action_transport.stageb_train import TrainConfig
from script.fsq_reconstruction.quantizers import isolated_rng, reseed_all
from script.quantized_input_comparison import masking as qic_masking

CONTEXTUALIZER_CONFIG_NAME = "postLN_cosine_wu500_lr0.0003_alibi"
CONTEXTUAL_ARMS = ("context_id", "context_quantized", "context_continuous", "context_geometry_init")


def base_train_config(**overrides) -> TrainConfig:
    """The existing selected architecture/schedule (stageb_configs), with
    Task 2's step budget (5,000 updates, eval every 100, warmup 500 already
    part of the named config) and mask fraction/span applied on top -- the
    same resolve-by-name + override pattern `fsq_reconstruction/evaluate.py`
    uses, not a second hand-copied config."""
    cfg = resolve_by_name(CONTEXTUALIZER_CONFIG_NAME)
    fields = dict(mask_fraction=0.30, mask_span=8, max_steps=5000, eval_every=100,
                 patience_steps=10**9)  # spec: "No early stopping for the main matrix"
    fields.update(overrides)
    return replace(cfg, **fields)


def model_config(num_codes: int, input_mode: str, input_dim: int | None, cfg: TrainConfig) -> ModelConfig:
    return ModelConfig(num_codes=num_codes, d_model=128, n_layers=4, n_heads=4, d_ff=512,
                       dropout=cfg.dropout, norm_first=cfg.norm_first, tie_head=cfg.tie_head,
                       position_mode=cfg.position_mode, input_mode=input_mode, input_dim=input_dim)


# --- matched initialization -------------------------------------------------------

@dataclass
class MatchedInit:
    """One family+seed's shared initialization ingredients: a single
    backbone+head template and a single projection template, each drawn
    once inside an isolated RNG context so every arm's backbone/head/
    mask-token/projection starts from EXACTLY the same tensors (spec:
    'Copy identical backbone, output-head and mask-token initialization
    across matched arms', 'Copy identical projection initialization
    between the quantized and continuous arms')."""
    encoder_state: dict
    head_state: dict  # {"head.weight":..., "head.bias":...}
    mask_token_init: torch.Tensor  # [d_model], shared initial mask-token value
    proj_state: dict  # {"weight":..., "bias":...} for nn.Linear(input_dim, d_model)


def build_matched_init(num_codes: int, input_dim: int, cfg: TrainConfig, seed: int) -> MatchedInit:
    """Draws the template backbone/head (via an "id"-mode template model,
    whose encoder/head parts do not depend on input_mode) and the template
    projection, both inside one isolated RNG context keyed on `seed` --
    isolated so this construction never perturbs the global RNG streams
    (dropout / batch order / masks) that training consumes later, per
    spec's 'Isolate initialization RNG streams and reset training
    randomness so adapter construction does not change dropout, masks or
    batch order.'"""
    with isolated_rng(seed):
        template = CodeContextModel(model_config(num_codes, "id", None, cfg))
        mask_token_init = template.embedding.weight.data[num_codes].detach().clone()
        proj = nn.Linear(input_dim, template.cfg.d_model)
    encoder_state = {k: v.detach().clone() for k, v in template.encoder.state_dict().items()}
    if cfg.tie_head:
        head_state = {"head_bias": template.head_bias.detach().clone()}
    else:
        head_state = {k: v.detach().clone() for k, v in template.head.state_dict().items()}
    proj_state = {k: v.detach().clone() for k, v in proj.state_dict().items()}
    return MatchedInit(encoder_state, head_state, mask_token_init, proj_state)


def _apply_shared_backbone_head(model: CodeContextModel, init: MatchedInit) -> None:
    model.encoder.load_state_dict(init.encoder_state)
    if model.cfg.tie_head:
        with torch.no_grad():
            model.head_bias.copy_(init.head_state["head_bias"])
    else:
        model.head.load_state_dict(init.head_state)


def build_arm_model(arm: str, num_codes: int, input_dim: int, cfg: TrainConfig, seed: int,
                    init: MatchedInit, lookup_over_rms: np.ndarray | None = None,
                    device: torch.device = torch.device("cpu")) -> CodeContextModel:
    """Builds one arm's model with matched backbone/head/mask-token
    initialization (and, for quantized/continuous, matched projection
    initialization) from `init`. `lookup_over_rms` ([num_codes, input_dim],
    the family's frozen lookup table already divided by the shared RMS) is
    required only for arm == 'context_geometry_init' (spec's E_c =
    W(q_c/RMS)+b construction).

    Model construction itself happens inside `isolated_rng(seed)` so any
    architecture-specific random draws (context_id's regular embedding
    rows; the throwaway default nn.Linear init for quantized/continuous,
    immediately overwritten) never touch the global RNG stream training
    will use afterward."""
    if arm == "context_id":
        with isolated_rng(seed):
            model = CodeContextModel(model_config(num_codes, "id", None, cfg))
        _apply_shared_backbone_head(model, init)
        with torch.no_grad():
            model.embedding.weight.data[num_codes].copy_(init.mask_token_init)
        return model.to(device)

    if arm in ("context_quantized", "context_continuous"):
        with isolated_rng(seed):
            model = CodeContextModel(model_config(num_codes, "vector", input_dim, cfg))
        _apply_shared_backbone_head(model, init)
        model.input_proj.load_state_dict(init.proj_state)
        with torch.no_grad():
            model.mask_token.copy_(init.mask_token_init)
        return model.to(device)

    if arm == "context_geometry_init":
        if lookup_over_rms is None:
            raise ValueError("context_geometry_init requires lookup_over_rms")
        with isolated_rng(seed):
            model = CodeContextModel(model_config(num_codes, "id", None, cfg))
        _apply_shared_backbone_head(model, init)
        proj = nn.Linear(input_dim, model.cfg.d_model)
        proj.load_state_dict(init.proj_state)
        with torch.no_grad():
            q = torch.as_tensor(lookup_over_rms, dtype=torch.float32)
            e_c = proj(q).detach()  # [num_codes, d_model] = W(q_c/RMS) + b
            model.embedding.weight.data[:num_codes].copy_(e_c)
            model.embedding.weight.data[num_codes].copy_(init.mask_token_init)
        return model.to(device)

    raise ValueError(arm)


def verify_geometry_init_matches_quantized(geometry_model: CodeContextModel, quantized_model: CodeContextModel,
                                           codes: np.ndarray, rms_scaled_vectors: np.ndarray) -> float:
    """Spec: 'Verify identical initial outputs against context_quantized in
    evaluation mode using rtol=1e-5, atol=1e-6.' Both models must be freshly
    built (untrained) when this is called. Feeds the SAME unmasked chunk
    through geometry_model.forward (id path) and quantized_model.forward_vectors
    (vector path) and returns the max abs difference of the logits (also
    asserts allclose at the stated tolerances)."""
    geometry_model.eval()
    quantized_model.eval()
    geo_device = next(geometry_model.parameters()).device
    quant_device = next(quantized_model.parameters()).device
    codes_t = torch.as_tensor(codes[None], dtype=torch.int64, device=geo_device)
    vec_t = torch.as_tensor(rms_scaled_vectors[None], dtype=torch.float32, device=quant_device)
    pad_geo = torch.zeros(1, codes.shape[0], dtype=torch.bool, device=geo_device)
    pad_quant = torch.zeros(1, codes.shape[0], dtype=torch.bool, device=quant_device)
    no_mask = torch.zeros(1, codes.shape[0], dtype=torch.bool, device=quant_device)
    with torch.no_grad():
        _, logits_geo = geometry_model.forward(codes_t, pad_geo)
        _, logits_quant = quantized_model.forward_vectors(vec_t, no_mask, pad_quant)
    logits_geo_cpu, logits_quant_cpu = logits_geo.cpu(), logits_quant.cpu()
    if not torch.allclose(logits_geo_cpu, logits_quant_cpu, rtol=1e-5, atol=1e-6):
        diff = (logits_geo_cpu - logits_quant_cpu).abs().max().item()
        raise AssertionError(f"context_geometry_init does not match context_quantized's initial "
                             f"outputs (max abs diff {diff:.3e}, rtol=1e-5 atol=1e-6)")
    return float((logits_geo_cpu - logits_quant_cpu).abs().max())


# --- chunking with receptive-field-aware masking -----------------------------------

@dataclass
class VectorChunk:
    recording: str
    start: int
    codes: np.ndarray  # int64 [P] (P = max_context, zero-padded)
    continuous: np.ndarray  # float32 [P, feature_dim] (window continuous feature, zero-padded)
    starts: np.ndarray  # int32 [P] raw-frame window starts
    lengths: np.ndarray  # int32 [P] real per-window frame counts (0 where padded)
    valid_len: int
    frame_count: int


def make_vector_chunks(loader, family: str, names: list[str], chunk: int = 128, stride: int = 64) -> list[VectorChunk]:
    out = []
    for name in names:
        rec = loader.recording(family, name)
        codes, continuous, starts, lengths = rec["codes"], rec["continuous"], rec["starts"], rec["lengths"]
        n = len(codes)
        frame_count = loader.frame_count(name)
        for s in chunk_starts(n, chunk, stride):
            end = min(s + chunk, n)
            valid = end - s
            p_codes = np.zeros(chunk, dtype=np.int64)
            p_codes[:valid] = codes[s:end]
            p_cont = np.zeros((chunk, continuous.shape[1]), dtype=np.float32)
            p_cont[:valid] = continuous[s:end]
            p_starts = np.zeros(chunk, dtype=np.int32)
            p_starts[:valid] = starts[s:end]
            p_lengths = np.zeros(chunk, dtype=np.int32)
            p_lengths[:valid] = lengths[s:end]
            out.append(VectorChunk(name, s, p_codes, p_cont, p_starts, p_lengths, valid, frame_count))
    return out


def chunk_masks(chunks: list[VectorChunk], base: int, fraction: float, span: int,
                radius: int = qic_masking.RADIUS_DEFAULT) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic (selected, eligible) mask pair per chunk, via Task 1's
    `masking.build_eligible_mask` -- the SAME masker used by every arm and
    every family, never a separate/reimplemented masking path."""
    out = []
    for c in chunks:
        rng = np.random.default_rng(stable_seed(c.recording, c.start, base))
        selected, eligible = qic_masking.build_eligible_mask(c.starts, c.lengths, c.valid_len, c.frame_count,
                                                              rng, radius=radius, fraction=fraction, span=span)
        out.append((selected, eligible))
    return out


# --- training -----------------------------------------------------------------------

def _encode_id(chunks: list[VectorChunk], masks: list[tuple[np.ndarray, np.ndarray]], mask_id: int,
              device: torch.device):
    length = len(chunks[0].codes)
    input_codes = np.stack([c.codes.copy() for c in chunks])
    targets = np.stack([c.codes.copy() for c in chunks])
    weights = np.zeros((len(chunks), length), dtype=np.float32)
    elig = np.zeros((len(chunks), length), dtype=bool)
    padding_mask = np.ones((len(chunks), length), dtype=bool)
    for i, (chunk, (selected, eligible)) in enumerate(zip(chunks, masks)):
        padding_mask[i, :chunk.valid_len] = False
        weights[i, eligible] = chunk.lengths[eligible]
        elig[i] = eligible
        input_codes[i, selected] = mask_id
    to = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    return (to(input_codes, torch.int64), to(targets, torch.int64), to(weights, torch.float32),
            to(elig, torch.bool), to(padding_mask, torch.bool))


def _encode_vector(chunks: list[VectorChunk], masks: list[tuple[np.ndarray, np.ndarray]],
                   feature_of: "callable", rms: float, device: torch.device):
    """feature_of(chunk) -> float32 [P, d] raw (un-scaled) per-position
    feature (lookup[codes] for context_quantized, continuous for
    context_continuous). Targets are the codes themselves (same vocabulary
    as context_id/context_geometry_init, K=num_codes)."""
    length = len(chunks[0].codes)
    dim = feature_of(chunks[0]).shape[1]
    vectors = np.zeros((len(chunks), length, dim), dtype=np.float32)
    targets = np.stack([c.codes.copy() for c in chunks])
    weights = np.zeros((len(chunks), length), dtype=np.float32)
    elig = np.zeros((len(chunks), length), dtype=bool)
    mask_bool = np.zeros((len(chunks), length), dtype=bool)
    padding_mask = np.ones((len(chunks), length), dtype=bool)
    for i, (chunk, (selected, eligible)) in enumerate(zip(chunks, masks)):
        padding_mask[i, :chunk.valid_len] = False
        weights[i, eligible] = chunk.lengths[eligible]
        elig[i] = eligible
        vectors[i] = feature_of(chunk) / rms
        mask_bool[i] = selected
    to = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    return (to(vectors, torch.float32), to(targets, torch.int64), to(weights, torch.float32),
            to(elig, torch.bool), to(mask_bool, torch.bool), to(padding_mask, torch.bool))


def masked_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor | None:
    """Real-frame-weighted CE at eligible positions only. Returns None (skip
    this batch) when no chunk in the batch has any eligible target -- spec:
    'Skip chunks without eligible targets and save their counts.'"""
    if float(weights.sum()) <= 0.0:
        return None
    ce = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    return (ce * weights).sum() / weights.sum().clamp_min(1e-8)


@dataclass
class TrainResult:
    model: CodeContextModel
    info: dict


def train_vector_contextualizer(arm: str, train_chunks: list[VectorChunk], val_chunks: list[VectorChunk],
                                num_codes: int, input_dim: int, seed: int, cfg: TrainConfig,
                                init: MatchedInit, device: torch.device,
                                lookup: np.ndarray | None = None, rms: float = 1.0,
                                mask_base: int = 111, deadline: float | None = None) -> TrainResult:
    """Trains one contextual arm's model. `lookup`/`rms` required for
    context_quantized/context_geometry_init (rms also scales continuous
    features for context_continuous). No action label is read anywhere in
    this function."""
    model = build_arm_model(arm, num_codes, input_dim, cfg, seed, init,
                            lookup_over_rms=(lookup / rms if lookup is not None else None), device=device)
    # Reset the CPU/CUDA training-time RNG deterministically from this cell's own
    # seed, immediately before training starts. `build_arm_model` isolates
    # INITIALIZATION randomness (via `isolated_rng`) so it never touches this
    # stream, but without an explicit reset here the training-time stream (which
    # dropout masks and batch order draw from) is whatever the global RNG happens
    # to be left at by prior cells / resume history -- not reproducible. Mirrors
    # `script.fsq_reconstruction.quantizers.reseed_all`'s existing "called once,
    # immediately before training each matched arm" pattern.
    reseed_all(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                  betas=(0.9, 0.999), eps=1e-8)
    warmup, max_steps = cfg.warmup_steps, cfg.max_steps

    def lr_lambda(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        span = max(1, max_steps - warmup)
        progress = min(1.0, (step - warmup) / span)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    val_masks = chunk_masks(val_chunks, mask_base, cfg.mask_fraction, cfg.mask_span)

    def feature_of(chunk: VectorChunk) -> np.ndarray:
        if arm == "context_continuous":
            return chunk.continuous
        return lookup[chunk.codes]  # context_quantized

    def forward_batch(chunks, masks):
        if arm == "context_id" or arm == "context_geometry_init":
            inp, tgt, w, elig, pad = _encode_id(chunks, masks, model.mask_id, device)
            _, logits = model.forward(inp, pad)
        else:
            vec, tgt, w, elig, mbool, pad = _encode_vector(chunks, masks, feature_of, rms, device)
            _, logits = model.forward_vectors(vec, mbool, pad)
        return logits, tgt, w, elig

    @torch.no_grad()
    def validate():
        model.eval()
        total, weight, skipped = 0.0, 0.0, 0
        for start in range(0, len(val_chunks), 256):
            part, part_masks = val_chunks[start:start + 256], val_masks[start:start + 256]
            logits, tgt, w, elig = forward_batch(part, part_masks)
            if float(w.sum()) <= 0:
                skipped += len(part)
                continue
            ce = F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
            total += float((ce * w).sum())
            weight += float(w.sum())
        model.train()
        if weight <= 0.0:
            # Every chunk in this validation pass had zero eligible targets --
            # total/weight would silently read 0.0 (a spurious "perfect" loss
            # that would then win best-checkpoint selection). This is a broken
            # masking/data setup, not a valid validation result: fail loudly.
            raise RuntimeError(f"validation pass has zero eligible targets across all {len(val_chunks)} "
                               f"chunk(s) ({skipped} skipped) -- cannot compute a validation loss")
        return total / weight, skipped

    n_batches = max(1, math.ceil(len(train_chunks) / cfg.batch_size))
    step, best_val, best_at, best_state = 0, float("inf"), -1, None
    history, skipped_train_chunks = [], 0
    epoch = 0
    # Safety against a pathological data/masking setup where every batch has
    # zero eligible targets forever: an empty batch no longer counts toward
    # the step budget (spec: "Skip chunks without eligible targets" must not
    # silently count as an optimizer update), so without this cap such a
    # setup would spin indefinitely instead of ever reaching max_steps.
    consecutive_empty_epochs = 0
    MAX_CONSECUTIVE_EMPTY_EPOCHS = 5
    while step < max_steps:
        model.train()
        epoch_rng = np.random.default_rng(seed * 1_000_003 + epoch)
        order = epoch_rng.permutation(len(train_chunks))
        accepted_steps_this_epoch = 0
        for b in range(n_batches):
            if step >= max_steps:
                break
            check_deadline(deadline)
            idx = order[b * cfg.batch_size:(b + 1) * cfg.batch_size]
            batch = [train_chunks[i] for i in idx]
            batch_masks = []
            for c in batch:
                rng = np.random.default_rng(int(epoch_rng.integers(0, 2**31 - 1)))
                selected, eligible = qic_masking.build_eligible_mask(
                    c.starts, c.lengths, c.valid_len, c.frame_count, rng,
                    fraction=cfg.mask_fraction, span=cfg.mask_span)
                batch_masks.append((selected, eligible))
            logits, tgt, w, elig = forward_batch(batch, batch_masks)
            loss = masked_loss(logits, tgt, w)
            if loss is None:
                # Zero eligible targets in this batch: no optimizer update, so
                # this must not consume budget or advance the LR schedule --
                # only count it and try the next batch.
                skipped_train_chunks += len(batch)
                continue
            optimizer.zero_grad()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            optimizer.step()
            scheduler.step()
            step += 1
            accepted_steps_this_epoch += 1
            if step % cfg.eval_every == 0 or step == max_steps:
                val_loss, val_skipped = validate()
                history.append({"step": step, "train_loss": float(loss.detach()), "val_loss": val_loss,
                               "lr": scheduler.get_last_lr()[0], "grad_norm": grad_norm,
                               "val_skipped_chunks": val_skipped})
                if val_loss < best_val:
                    best_val, best_at = val_loss, step
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if step - best_at >= cfg.patience_steps:
                    break
        if step - best_at >= cfg.patience_steps:
            break
        if accepted_steps_this_epoch == 0:
            consecutive_empty_epochs += 1
            if consecutive_empty_epochs >= MAX_CONSECUTIVE_EMPTY_EPOCHS:
                raise RuntimeError(f"{consecutive_empty_epochs} consecutive epoch(s) with zero eligible-target "
                                   f"batches (arm={arm}); no optimizer update has ever happened -- this looks "
                                   f"like a broken masking/data setup, not transient bad luck")
        else:
            consecutive_empty_epochs = 0
        epoch += 1
    if best_state is None:  # degenerate: never validated (max_steps < eval_every)
        val_loss, val_skipped = validate()
        best_val, best_at = val_loss, step
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    last_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    info = dict(arm=arm, seed=seed, best_step=best_at, best_val_loss=best_val, steps_run=step,
               skipped_train_chunks=skipped_train_chunks, history=history,
               n_train_chunks=len(train_chunks), n_val_chunks=len(val_chunks),
               n_parameters=sum(p.numel() for p in model.parameters()))
    return TrainResult(model, dict(**info, last_state=last_state))


# --- inference: overlapping-chunk averaged, L2-normalized embeddings --------------

@torch.no_grad()
def extract_vector_embeddings(arm: str, model: CodeContextModel, loader, family: str, name: str,
                              lookup: np.ndarray | None, rms: float,
                              device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Every window retained (no masking at inference); overlapping chunks
    (length 128, stride 64, final-anchored) averaged, then L2-normalized --
    identical scheme to `script.action_transport.stageb_infer.extract_embeddings`,
    generalized to the vector input path."""
    model.eval()
    rec = loader.recording(family, name)
    codes, continuous = rec["codes"], rec["continuous"]
    length = len(codes)
    d_model = model.cfg.d_model
    accum = np.zeros((length, d_model), dtype=np.float64)
    counts = np.zeros(length, dtype=np.float64)
    for start in chunk_starts(length):
        end = min(start + 128, length)
        valid = end - start
        pad = torch.zeros(1, 128, dtype=torch.bool)
        pad[0, valid:] = True
        no_mask = torch.zeros(1, 128, dtype=torch.bool)
        if arm in ("context_id", "context_geometry_init"):
            chunk = np.zeros(128, dtype=np.int64)
            chunk[:valid] = codes[start:end]
            hidden, _ = model.forward(torch.as_tensor(chunk[None], dtype=torch.int64, device=device),
                                      pad.to(device))
        else:
            dim = lookup.shape[1] if arm == "context_quantized" else continuous.shape[1]
            chunk = np.zeros((128, dim), dtype=np.float32)
            feat = lookup[codes[start:end]] if arm == "context_quantized" else continuous[start:end]
            chunk[:valid] = feat / rms
            hidden, _ = model.forward_vectors(torch.as_tensor(chunk[None], dtype=torch.float32, device=device),
                                              no_mask.to(device), pad.to(device))
        accum[start:end] += hidden[0, :valid].cpu().numpy().astype(np.float64)
        counts[start:end] += 1.0
    mean = accum / counts[:, None]
    norms = np.linalg.norm(mean, axis=1, keepdims=True)
    zero = norms[:, 0] <= 1e-12
    unit = mean / np.where(zero[:, None], 1.0, norms)
    unit[zero] = 0.0
    return unit.astype(np.float32), zero
