"""Phase 9 trainer: a gated copy of `contextualizer_calibration.train.train_core` / `run_training_job` and of
`evaluate.extract_embeddings` / `build_model_from_state`, with exactly three additions (nothing else differs):

  1. `feature_fn(arm, lookup, chunk)`: the vector arms read `chunk.continuous` for `context_continuous` and the frozen
     lookup row for `context_quantized`, as `quantized_input_comparison.contextual` already does (default unchanged);
  2. `train_core(..., lookup_over_rms=...)`, passed to `build_arm_model` (needed by `context_geometry_init`, whose
     embedding table is initialized to W(q_c/RMS)+b);
  3. embedding extraction and model rebuilding for the `context_continuous` / `context_geometry_init` arms.

The shared files are not edited: their contents are hashed into every Phase 5-8c implementation identity. Gate G9a
retrains BABEL-2 seed 111 with the default path and requires the saved job's selected step, validation history and
checkpoint tensors. The body of `train_core` below is generated from the Phase 5 source (sha256 prefix of the original
text: 2c18e35ea531ccf9); the three changed lines carry a `PHASE9` marker.

No action label is read anywhere in this module.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from script.action_transport.budget import check_deadline
from script.action_transport.stageb_data import chunk_starts
from script.action_transport.stageb_model import CodeContextModel
from script.fsq_reconstruction.quantizers import reseed_all
from script.quantized_input_comparison import contextual as C
from script.week5.contextualizer_calibration import train as T
from script.week5.contextualizer_calibration.plan import JobSpec, Scale
from script.week5.contextualizer_calibration.train import (CoreResult, MAX_CONSECUTIVE_EMPTY_EPOCHS, MICRO_CHUNKS,
                                                           OPTIMIZER_SETTINGS, VAL_PART, VALIDATION_MASK_BASE, _build,
                                                           _weight, bounded_split, checkpoint_roles, cpu_state,
                                                           matched_batch_masks, realized_runs, save_state,
                                                           tensor_hash, train_config)

ORIGINAL_CORE_SHA = "2c18e35ea531ccf9"
VECTOR_ARMS = ("context_quantized", "context_continuous")
ID_ARMS = ("context_id", "context_geometry_init")


def feature_fn(arm: str, lookup, chunk) -> np.ndarray:
    """Per-window input vectors (before the shared RMS scaling) of a chunk, for the vector arms."""
    if arm == "context_continuous":
        return chunk.continuous
    return lookup[chunk.codes]  # context_quantized (the Phase 5 behaviour)


def train_core(*, arm: str, train_chunks, val_chunks, num_codes: int, input_dim: int, seed: int, cfg: TrainConfig,
               init: "C.MatchedInit", device: torch.device, lookup: np.ndarray | None, rms: float,
               guard_radius: int, fixed_steps: dict | None = None, reference_span: int | None = None,
               ckpt_dir: Path | None = None, deadline: float | None = None, micro_chunks: int | None = None,
               model=None, lookup_over_rms: np.ndarray | None = None) -> CoreResult:  # PHASE9 (2)
    """See module docstring. `fixed_steps`: {role: step} checkpoints to keep;
    `reference_span` switches on the matched-target-count control; `ckpt_dir`
    None disables writing (the in-memory result is identical)."""
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if model is None:
        model = C.build_arm_model(arm, num_codes, input_dim, cfg, seed, init, lookup_over_rms=lookup_over_rms, device=device)  # PHASE9 (2)
    reseed_all(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                  betas=OPTIMIZER_SETTINGS["betas"], eps=OPTIMIZER_SETTINGS["eps"])
    warmup, max_steps = cfg.warmup_steps, cfg.max_steps

    def lr_lambda(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        span = max(1, max_steps - warmup)
        progress = min(1.0, (step - warmup) / span)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    val_masks = C.chunk_masks(val_chunks, VALIDATION_MASK_BASE, cfg.mask_fraction, cfg.mask_span, radius=guard_radius)
    keep_steps = {step: role for role, step in (fixed_steps or {}).items() if step <= max_steps}
    context = len(train_chunks[0].codes)

    def feature_of(chunk):
        return feature_fn(arm, lookup, chunk)  # PHASE9 (1)

    def forward_batch(chunks, masks):
        if arm in ("context_id", "context_geometry_init"):
            inp, tgt, w, elig, pad = C._encode_id(chunks, masks, model.mask_id, device)
            _, logits = model.forward(inp, pad)
        else:
            vec, tgt, w, elig, mbool, pad = C._encode_vector(chunks, masks, feature_of, rms, device)
            _, logits = model.forward_vectors(vec, mbool, pad)
        return logits, tgt, w

    @torch.no_grad()
    def validate():
        model.eval()
        total, weight, skipped = 0.0, 0.0, 0
        for start in range(0, len(val_chunks), VAL_PART):
            part, part_masks = val_chunks[start:start + VAL_PART], val_masks[start:start + VAL_PART]
            logits, tgt, w = forward_batch(part, part_masks)
            if float(w.sum()) <= 0:
                skipped += len(part)
                continue
            ce = F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
            total += float((ce * w).sum())
            weight += float(w.sum())
        model.train()
        if weight <= 0.0:
            raise RuntimeError(f"validation pass has zero eligible targets across all {len(val_chunks)} chunk(s) "
                               f"({skipped} skipped) -- cannot compute a validation loss")
        return total / weight, skipped

    def loss_and_backward(batch, masks):
        """Returns the update's loss value, or None when the batch has no eligible target."""
        if micro_chunks is None:
            logits, tgt, w = forward_batch(batch, masks)
            loss = C.masked_loss(logits, tgt, w)
            if loss is None:
                return None
            optimizer.zero_grad()
            loss.backward()
            return float(loss.detach())
        # Weighted accumulation: sum per-part numerators, normalize by the effective batch's
        # total real-frame weight, backpropagate each part, clip and step once.
        total_w = float(sum(c.lengths[e].sum() for c, (_, e) in zip(batch, masks)))
        if total_w <= 0.0:
            return None
        optimizer.zero_grad()
        value = 0.0
        for s in range(0, len(batch), micro_chunks):
            logits, tgt, w = forward_batch(batch[s:s + micro_chunks], masks[s:s + micro_chunks])
            ce = F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
            part = (ce * w).sum() / total_w
            part.backward()
            value += float(part.detach())
        return value

    n_batches = max(1, math.ceil(len(train_chunks) / cfg.batch_size))
    step, best_val, best_at, best_state = 0, float("inf"), -1, None
    history, ledger, saved = [], [], {}
    attempted = skipped_chunks = skipped_batches = 0
    consecutive_empty_epochs = 0
    tails: Counter = Counter()
    span_hist: Counter = Counter()
    pos_selected = np.zeros(context, dtype=np.int64)
    pos_eligible = np.zeros(context, dtype=np.int64)
    pos_valid = np.zeros(context, dtype=np.int64)
    rows: dict[str, list] = {k: [] for k in ("batch", "valid", "selected", "eligible", "weight", "seconds",
                                             "n_ref", "n_before", "retries", "w_ref")}

    def checkpoint(reason_step: int, val_loss: float, improved: bool):
        state = cpu_state(model)
        reasons = (["improved"] if improved else []) + ([keep_steps[reason_step]] if reason_step in keep_steps else []) \
            + (["final"] if reason_step == max_steps else [])
        path = None
        if ckpt_dir is not None:
            path = Path(ckpt_dir) / f"step{reason_step:05d}.pt"
            save_state(path, state, dict(config=model.cfg.__dict__, step=reason_step, val_loss=val_loss))
        ledger.append(dict(step=reason_step, val_loss=val_loss, reasons=reasons, file=path.name if path else None,
                           tensor_sha256=tensor_hash(state)))
        return state

    epoch = 0
    while step < max_steps:
        model.train()
        epoch_rng = np.random.default_rng(seed * 1_000_003 + epoch)
        order = epoch_rng.permutation(len(train_chunks))
        accepted_this_epoch = 0
        for b in range(n_batches):
            if step >= max_steps:
                break
            check_deadline(deadline)
            idx = order[b * cfg.batch_size:(b + 1) * cfg.batch_size]
            batch = [train_chunks[i] for i in idx]
            attempted += 1
            seeds = [int(epoch_rng.integers(0, 2**31 - 1)) for _ in batch]
            extra = dict(n_ref=0, n_before=0, retries=0, w_ref=0.0)
            if reference_span is None:
                batch_masks = [_build(c, np.random.default_rng(s), guard_radius, cfg.mask_fraction, cfg.mask_span)
                               for c, s in zip(batch, seeds)]
            else:
                matched = matched_batch_masks(batch, seeds, radius=guard_radius, fraction=cfg.mask_fraction,
                                              span_long=cfg.mask_span, span_ref=reference_span, base_seed=seed,
                                              epoch=epoch, batch_index=b)
                if matched["skip"]:
                    skipped_chunks += len(batch)
                    skipped_batches += 1
                    continue
                batch_masks = matched["masks"]
                extra = dict(n_ref=matched["n_ref"], n_before=matched["n_before"], retries=matched["retries"],
                             w_ref=matched["w_ref"])
            tick = time.perf_counter()
            loss_value = loss_and_backward(batch, batch_masks)
            if loss_value is None:
                skipped_chunks += len(batch)
                skipped_batches += 1
                continue
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            optimizer.step()
            scheduler.step()
            step += 1
            accepted_this_epoch += 1
            seconds = time.perf_counter() - tick
            n_sel = int(sum(s.sum() for s, _ in batch_masks))
            n_elig = int(sum(e.sum() for _, e in batch_masks))
            rows["batch"].append(len(batch))
            rows["valid"].append(int(sum(c.valid_len for c in batch)))
            rows["selected"].append(n_sel)
            rows["eligible"].append(n_elig)
            rows["weight"].append(float(sum(_weight(c, e) for c, (_, e) in zip(batch, batch_masks))))
            rows["seconds"].append(seconds)
            for k, v in extra.items():
                rows[k].append(v)
            if len(batch) != cfg.batch_size:
                tails[len(batch)] += 1
            for c, (sel, elig) in zip(batch, batch_masks):
                span_hist.update(realized_runs(sel, c.valid_len))
                pos_selected += sel
                pos_eligible += elig
                pos_valid[:c.valid_len] += 1
            if step % cfg.eval_every == 0 or step == max_steps:
                val_loss, val_skipped = validate()
                history.append({"step": step, "train_loss": loss_value, "val_loss": val_loss,
                                "lr": scheduler.get_last_lr()[0], "grad_norm": grad_norm,
                                "val_skipped_chunks": val_skipped})
                improved = val_loss < best_val
                if improved:
                    best_val, best_at = val_loss, step
                if improved or step in keep_steps or step == max_steps:
                    state = checkpoint(step, val_loss, improved)
                    if improved:
                        best_state = state
                    if step == max_steps:
                        saved["last"] = state
        if accepted_this_epoch == 0:
            consecutive_empty_epochs += 1
            if consecutive_empty_epochs >= MAX_CONSECUTIVE_EMPTY_EPOCHS:
                raise RuntimeError(f"{consecutive_empty_epochs} consecutive epoch(s) with zero eligible-target "
                                   f"batches; no optimizer update has ever happened -- broken masking/data setup")
        else:
            consecutive_empty_epochs = 0
        epoch += 1
    if best_state is None:  # never validated (horizon shorter than the validation cadence)
        val_loss, _ = validate()
        best_val, best_at = val_loss, step
        best_state = checkpoint(step, val_loss, True)
    last_state = saved.get("last") or cpu_state(model)
    model.load_state_dict(best_state)
    stats = dict(attempted_batches=attempted, accepted_updates=step, skipped_batches=skipped_batches,
                 skipped_train_chunks=skipped_chunks, epochs_started=epoch + 0, context=context,
                 nominal_positions=int(sum(rows["batch"]) * context), valid_tokens=int(sum(rows["valid"])),
                 padded_tokens=int(sum(rows["batch"]) * context - sum(rows["valid"])),
                 selected_targets=int(sum(rows["selected"])), eligible_targets=int(sum(rows["eligible"])),
                 real_frame_weight=float(sum(rows["weight"])),
                 batch_tail_sizes={str(k): v for k, v in sorted(tails.items())},
                 realized_span_hist={str(k): v for k, v in sorted(span_hist.items())},
                 mean_eligible_per_update=float(np.mean(rows["eligible"])) if rows["eligible"] else 0.0)
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    arrays.update(pos_selected=pos_selected, pos_eligible=pos_eligible, pos_valid=pos_valid)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return CoreResult(model, history, best_at, best_val, step, best_state, last_state, ledger, stats, arrays,
                      time.perf_counter() - started, peak)


def run_training_job9(job: JobSpec, scale: Scale, loader, job_dir: Path, device: torch.device,
                      deadline: float | None, identity: str, impl_hash: str) -> dict:
    """`train.run_training_job`, line for line, except for the arm-dependent `lookup` / `lookup_over_rms` arguments."""
    names, train_names, val_names = bounded_split(loader, scale.recording_cap)
    family = job.family
    lookup, rms = loader.lookup(family), float(loader.rms(family))
    num_codes, input_dim = lookup.shape
    train_chunks = C.make_vector_chunks(loader, family, train_names, chunk=job.context, stride=job.stride)
    val_chunks = C.make_vector_chunks(loader, family, val_names, chunk=job.context, stride=job.stride)
    cfg = train_config(job, scale)
    horizon = cfg.max_steps
    init = C.build_matched_init(num_codes, input_dim, cfg, seed=job.seed)
    ckpt_dir = job_dir / "ckpt"
    result = train_core(arm=job.arm, train_chunks=train_chunks, val_chunks=val_chunks, num_codes=num_codes,
                        input_dim=input_dim, seed=job.seed, cfg=cfg, init=init, device=device,
                        lookup=lookup if job.arm == "context_quantized" else None, rms=rms,
                        guard_radius=job.guard_radius, fixed_steps=scale.fixed_steps,
                        reference_span=8 if job.target_count == "match" else None, ckpt_dir=ckpt_dir,
                        deadline=deadline, micro_chunks=MICRO_CHUNKS,
                        lookup_over_rms=(lookup / rms) if job.arm == "context_geometry_init" else None)
    roles = checkpoint_roles(result, scale, horizon)
    by_step = {row["step"]: row for row in result.ledger}
    info = dict(job_id=job.job_id, status="done", identity=identity, implementation_hash=impl_hash,
                horizon_steps=horizon, **{k: v for k, v in asdict(job).items() if k != "job_id"},
                selected_step=roles["roles"]["selected"], best_val_loss=result.best_val,
                final_step=result.steps_run, replay_stop_step=roles["replay"]["stop_step"],
                replay_selected_step=roles["replay"]["selected_step"], roles=roles["roles"],
                n_train_chunks=len(train_chunks), n_val_chunks=len(val_chunks), n_scored_recordings=len(names),
                n_parameters=sum(p.numel() for p in result.model.parameters()), seconds=result.seconds,
                peak_memory_bytes=result.peak_memory_bytes, device=str(device), history=result.history,
                exposure=result.stats, role_tensor_sha256={r: by_step[s]["tensor_sha256"]
                                                           for r, s in roles["roles"].items() if s in by_step})
    np.savez_compressed(job_dir / "exposure_steps.npz", **result.arrays)
    (job_dir / "checkpoints.json").write_text(json.dumps(
        dict(ledger=result.ledger, roles=roles["roles"], replay=roles["replay"]), indent=2))
    return info


def build_model_from_state9(job: JobSpec, scale: Scale, loader, state: dict, device: torch.device) -> CodeContextModel:
    lookup = loader.lookup(job.family)
    cfg = train_config(job, scale)
    mode, dim = ("vector", lookup.shape[1]) if job.arm in VECTOR_ARMS else ("id", None)
    model = CodeContextModel(C.model_config(lookup.shape[0], mode, dim, cfg))
    model.load_state_dict(state)
    return model.to(device).eval()


def layer_states(model: CodeContextModel, x: torch.Tensor, padding_mask: torch.Tensor) -> list[torch.Tensor]:
    """Output of every encoder layer (the ALiBi path of `CodeContextModel._run_backbone` / `_encode_with_bias`, spelled
    out the same way); the last entry equals the model's hidden states."""
    if model.cfg.position_mode != "alibi":
        raise ValueError("layer extraction is written for the ALiBi backbone used throughout")
    batch, length, _ = x.shape
    x = model.dropout(x)
    idx = torch.arange(length, device=x.device)
    bias = -model.alibi_slopes[:, None, None] * (idx[:, None] - idx[None, :]).abs().float()
    bias = bias.unsqueeze(0).expand(batch, -1, -1, -1).clone()
    bias = bias.masked_fill(padding_mask[:, None, None, :], float("-inf"))
    bias = bias.reshape(batch * model.cfg.n_heads, length, length)
    outs = []
    for layer in model.encoder.layers:
        def attend(y):
            return layer.self_attn(y, y, y, attn_mask=bias, need_weights=True, average_attn_weights=False)[0]

        def feed_forward(y):
            return layer.linear2(layer.dropout(layer.activation(layer.linear1(y))))

        if model.cfg.norm_first:
            x = x + layer.dropout1(attend(layer.norm1(x)))
            x = x + layer.dropout2(feed_forward(layer.norm2(x)))
        else:
            x = layer.norm1(x + layer.dropout1(attend(x)))
            x = layer.norm2(x + layer.dropout2(feed_forward(x)))
        outs.append(x)
    if model.encoder.norm is not None:
        outs[-1] = model.encoder.norm(outs[-1])
    return outs


@torch.no_grad()
def extract_embeddings9(arm: str, model: CodeContextModel, loader, family: str, name: str, lookup, rms: float,
                        device: torch.device, chunk: int, stride: int, layers: bool = False):
    """`evaluate.extract_embeddings` (every window retained, no masking, chunks of `chunk` at `stride` with a final
    anchored chunk, hidden states averaged where they overlap, then L2-normalized), for every arm. With
    `layers=True` it also returns the raw (un-normalized, chunk-averaged) states of every layer as float32."""
    model.eval()
    rec = loader.recording(family, name)
    codes, continuous_feat = rec["codes"], rec["continuous"]
    length = len(codes)
    accum = [np.zeros((length, model.cfg.d_model), dtype=np.float64) for _ in range(model.cfg.n_layers if layers else 1)]
    counts = np.zeros(length, dtype=np.float64)
    for start in chunk_starts(length, chunk, stride):
        end = min(start + chunk, length)
        valid = end - start
        pad = torch.zeros(1, chunk, dtype=torch.bool)
        pad[0, valid:] = True
        pad = pad.to(device)
        if arm in ID_ARMS:
            ids = np.zeros(chunk, dtype=np.int64)
            ids[:valid] = codes[start:end]
            x = model.embedding(torch.as_tensor(ids[None], dtype=torch.int64, device=device))
        else:
            src = continuous_feat if arm == "context_continuous" else lookup[codes]
            vecs = np.zeros((chunk, src.shape[1]), dtype=np.float32)
            vecs[:valid] = src[start:end] / rms
            x = model.input_proj(torch.as_tensor(vecs[None], dtype=torch.float32, device=device))
        states = layer_states(model, x, pad) if layers else [model._run_backbone(x, pad)]
        for k, h in enumerate(states):
            accum[k][start:end] += h[0, :valid].cpu().numpy().astype(np.float64)
        counts[start:end] += 1.0
    means = [a / counts[:, None] for a in accum]
    mean = means[-1]
    norms = np.linalg.norm(mean, axis=1, keepdims=True)
    zero = norms[:, 0] <= 1e-12
    unit = mean / np.where(zero[:, None], 1.0, norms)
    unit[zero] = 0.0
    unit = unit.astype(np.float32)
    if layers:
        return unit, zero, [m.astype(np.float32) for m in means]
    return unit, zero


@torch.no_grad()
def masked_val_accuracy(job: JobSpec, scale: Scale, loader, state: dict, device: torch.device) -> dict:
    """Weighted masked-code accuracy and loss on the validation chunks (the same validation masks the trainer used) for
    the model in `state`. Label-free; a diagnostic of how hard the prediction task is."""
    names, train_names, val_names = bounded_split(loader, scale.recording_cap)
    family = job.family
    lookup, rms = loader.lookup(family), float(loader.rms(family))
    cfg = train_config(job, scale)
    model = build_model_from_state9(job, scale, loader, state, device)
    val_chunks = C.make_vector_chunks(loader, family, val_names, chunk=job.context, stride=job.stride)
    masks = C.chunk_masks(val_chunks, VALIDATION_MASK_BASE, cfg.mask_fraction, cfg.mask_span, radius=job.guard_radius)
    hit = tot = loss = 0.0
    for s in range(0, len(val_chunks), VAL_PART):
        part, pm = val_chunks[s:s + VAL_PART], masks[s:s + VAL_PART]
        if job.arm in ID_ARMS:
            inp, tgt, w, _, pad = C._encode_id(part, pm, model.mask_id, device)
            _, logits = model.forward(inp, pad)
        else:
            vec, tgt, w, _, mbool, pad = C._encode_vector(part, pm, lambda c: feature_fn(job.arm, lookup, c), rms, device)
            _, logits = model.forward_vectors(vec, mbool, pad)
        if float(w.sum()) <= 0:
            continue
        ce = F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
        loss += float((ce * w).sum())
        hit += float(((logits.argmax(-1) == tgt).float() * w).sum())
        tot += float(w.sum())
    return dict(val_loss=loss / tot if tot else None, val_accuracy=hit / tot if tot else None, weight=tot)
