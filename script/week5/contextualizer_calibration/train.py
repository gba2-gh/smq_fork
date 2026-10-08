"""Contextualizer training for the calibration pilot.

`train_core` is a faithful, parameterized re-statement of
`quantized_input_comparison.contextual.train_vector_contextualizer`: same
initialization, RNG reset, epoch shuffling, per-chunk mask seeds, schedule,
validation cadence and strict-improvement best tracking, so the anchor jobs
(context 128, stride 64, batch 64, guard 14, 5k updates) follow Task 2's
trajectory exactly. What it adds, none of which touches the training RNG:

* explicit context / stride / batch / guard radius / horizon (no hidden constants);
* a checkpoint on every strict validation improvement plus fixed-step and final
  checkpoints, with tensor identities, and an offline patience replay;
* exposure accounting (accepted updates, valid/padded tokens, selected and
  eligible targets, frame weights, batch tails, realized spans, positions);
* optional weighted gradient accumulation (`micro_chunks`);
* the matched-target-count control (`target_count == "match"`).

No action label is read anywhere in this module.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from script.action_transport.budget import check_deadline
from script.action_transport.stageb_train import TrainConfig
from script.fsq_reconstruction.quantizers import reseed_all
from script.quantized_input_comparison import contextual as C
from script.quantized_input_comparison import masking as qic_masking
from script.week5.contextualizer_calibration.plan import FIXED_ROLES, JobSpec, Scale

VALIDATION_MASK_BASE = 111
MICRO_CHUNKS = None  # no accumulation for any planned job (8,192 tokens fit in one forward pass)
MAX_CONSECUTIVE_EMPTY_EPOCHS = 5
MAX_MATCH_ATTEMPTS = 100
VAL_PART = 256
OPTIMIZER_SETTINGS = dict(name="AdamW", betas=(0.9, 0.999), eps=1e-8, schedule="linear warmup then cosine to 0.1x",
                          schedule_floor=0.1)


class TargetMatchUnavailable(RuntimeError):
    """The long mask could not supply the reference number of eligible targets."""


# --- configuration ----------------------------------------------------------------------

def train_config(job: JobSpec, scale: Scale) -> TrainConfig:
    overrides = dict(max_steps=scale.horizon(job.horizon), eval_every=scale.eval_every, patience_steps=10**9,
                     batch_size=job.batch_chunks, mask_span=job.mask_span, mask_fraction=job.mask_fraction)
    if scale.warmup is not None:
        overrides["warmup_steps"] = scale.warmup
    return C.base_train_config(**overrides)


def model_settings(cfg: TrainConfig, width: int = 128, layers: int = 4, heads: int = 4, d_ff: int = 512) -> dict:
    return dict(d_model=width, n_layers=layers, n_heads=heads, d_ff=d_ff, dropout=cfg.dropout,
                norm_first=cfg.norm_first, tie_head=cfg.tie_head, position_mode=cfg.position_mode)


def bounded_split(loader, recording_cap: int | None) -> tuple[list[str], list[str], list[str]]:
    """(all names, train names, val names) in the loader's order. The existing
    train/val membership is kept; a cap only takes the first N of each side
    (never reassigns a recording)."""
    names = loader.recording_names()
    members = {n: loader.split_membership(n) for n in names}
    unknown = [n for n, m in members.items() if m == "unknown"]
    if unknown:
        raise RuntimeError(f"{len(unknown)} recording(s) are in neither split side, e.g. {unknown[:3]}")
    train = [n for n in names if members[n] == "train"]
    val = [n for n in names if members[n] == "val"]
    if recording_cap:
        n_val = max(2, recording_cap // 4)
        train, val = train[:max(2, recording_cap - n_val)], val[:n_val]
        keep = set(train) | set(val)
        names = [n for n in names if n in keep]
    if not train or not val:
        raise RuntimeError("train or validation side is empty")
    return names, train, val


# --- identities and checkpoint files ----------------------------------------------------

def tensor_hash(state: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        t = state[key].detach().cpu().contiguous()
        h.update(f"{key}|{t.dtype}|{tuple(t.shape)}".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def save_state(path: Path, state: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.pt")
    torch.save(dict(state_dict=state, **meta), tmp)
    os.replace(tmp, path)


def load_state(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def cpu_state(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# --- early-stopping replay --------------------------------------------------------------

def replay_patience(history: list[tuple[int, float]], patience: int) -> dict:
    """Offline replay of the historical rule over a validation history of
    (step, loss): strict improvement updates the best (earliest ties are kept);
    the first evaluation with step - best_step >= patience is the simulated
    stop. The selected checkpoint is the best available at the stop, or the
    best through the endpoint when no stop occurs. Pure function: it cannot
    perturb training."""
    best_val, best_step, stop = math.inf, -1, None
    for step, val in history:
        if val < best_val:
            best_val, best_step = val, step
        if step - best_step >= patience:
            stop = step
            break
    return dict(stop_step=stop, selected_step=best_step, selected_val=best_val if best_step >= 0 else None)


def full_history_best(history: list[tuple[int, float]]) -> tuple[int, float]:
    best_val, best_step = math.inf, -1
    for step, val in history:
        if val < best_val:
            best_val, best_step = val, step
    return best_step, best_val


# --- masks ------------------------------------------------------------------------------

def realized_runs(selected: np.ndarray, valid_len: int) -> list[int]:
    """Lengths of maximal runs of selected positions within [0, valid_len)."""
    s = np.zeros(valid_len + 2, dtype=np.int8)
    s[1:valid_len + 1] = selected[:valid_len]
    d = np.diff(s)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return (ends - starts).tolist()


def _build(chunk, rng, radius, fraction, span):
    return qic_masking.build_eligible_mask(chunk.starts, chunk.lengths, chunk.valid_len, chunk.frame_count, rng,
                                           radius=radius, fraction=fraction, span=span)


def _weight(chunk, eligible) -> float:
    return float(chunk.lengths[eligible].sum())


def matched_batch_masks(batch, seeds: list[int], *, radius: int, fraction: float, span_long: int, span_ref: int,
                        base_seed: int, epoch: int, batch_index: int, max_attempts: int = MAX_MATCH_ATTEMPTS) -> dict:
    """Masks for one effective batch under the matched-target-count control.

    The reference masks (span `span_ref`) and the long masks (span `span_long`)
    both come from the chunks' ordinary per-chunk seeds, so the reference is
    exactly what the paired span-8 job drew. Every selected long-mask position
    stays masked in the input; only the supervised (eligible) set is thinned,
    uniformly over the batch, to the reference's eligible count. Retries and
    thinning use their own deterministic streams and never touch the model,
    dropout or batch-order RNG. Returns skip=True when the reference batch has
    no eligible target (the paired reference job skips it too)."""
    ref = [_build(c, np.random.default_rng(s), radius, fraction, span_ref) for c, s in zip(batch, seeds)]
    n_ref = int(sum(e.sum() for _, e in ref))
    w_ref = float(sum(_weight(c, e) for c, (_, e) in zip(batch, ref)))
    if n_ref == 0:
        return dict(skip=True, masks=None, n_ref=0, n_before=0, n_after=0, retries=0, w_ref=0.0, w_after=0.0)
    long = [_build(c, np.random.default_rng(s), radius, fraction, span_long) for c, s in zip(batch, seeds)]
    n_long = int(sum(e.sum() for _, e in long))
    retries = 0
    while n_long < n_ref:
        retries += 1
        if retries > max_attempts:
            raise TargetMatchUnavailable(
                f"batch (epoch={epoch}, batch={batch_index}, first chunk={batch[0].recording}@{batch[0].start}) "
                f"supplies {n_long} eligible long-mask targets < reference {n_ref} after {max_attempts} retries")
        retry_rng = np.random.default_rng([base_seed, epoch, batch_index, retries, 17])
        new_seeds = retry_rng.integers(0, 2**31 - 1, size=len(batch))
        long = [_build(c, np.random.default_rng(int(s)), radius, fraction, span_long) for c, s in zip(batch, new_seeds)]
        n_long = int(sum(e.sum() for _, e in long))
    n_before = n_long
    if n_long > n_ref:
        positions = [(i, t) for i, (_, e) in enumerate(long) for t in np.flatnonzero(e)]
        pick = np.random.default_rng([base_seed, epoch, batch_index, 29]).choice(len(positions), n_ref, replace=False)
        thinned = [np.zeros_like(e) for _, e in long]
        for k in pick:
            i, t = positions[k]
            thinned[i][t] = True
        long = [(sel, th) for (sel, _), th in zip(long, thinned)]
    w_after = float(sum(_weight(c, e) for c, (_, e) in zip(batch, long)))
    return dict(skip=False, masks=long, n_ref=n_ref, n_before=n_before, n_after=n_ref, retries=retries,
                w_ref=w_ref, w_after=w_after)


# --- core training ------------------------------------------------------------------------

@dataclass
class CoreResult:
    model: torch.nn.Module
    history: list
    best_step: int
    best_val: float
    steps_run: int
    best_state: dict
    last_state: dict
    ledger: list
    stats: dict
    arrays: dict
    seconds: float
    peak_memory_bytes: int | None


def train_core(*, arm: str, train_chunks, val_chunks, num_codes: int, input_dim: int, seed: int, cfg: TrainConfig,
               init: "C.MatchedInit", device: torch.device, lookup: np.ndarray | None, rms: float,
               guard_radius: int, fixed_steps: dict | None = None, reference_span: int | None = None,
               ckpt_dir: Path | None = None, deadline: float | None = None, micro_chunks: int | None = None,
               model=None) -> CoreResult:
    """See module docstring. `fixed_steps`: {role: step} checkpoints to keep;
    `reference_span` switches on the matched-target-count control; `ckpt_dir`
    None disables writing (the in-memory result is identical)."""
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if model is None:
        model = C.build_arm_model(arm, num_codes, input_dim, cfg, seed, init, device=device)
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
        return lookup[chunk.codes]  # context_quantized

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


def checkpoint_roles(result: CoreResult, scale: Scale, horizon: int) -> dict:
    """Role -> step for every checkpoint policy of one finished fit (roles are
    aliases: several may point at the same step / file)."""
    pairs = [(h["step"], h["val_loss"]) for h in result.history]
    replay = replay_patience(pairs, scale.patience)
    roles = {r: scale.fixed_steps[r] for r in FIXED_ROLES if scale.fixed_steps[r] <= horizon}
    roles.update(selected=full_history_best(pairs)[0], final=result.steps_run, replay=replay["selected_step"])
    return dict(roles=roles, replay=replay)


# --- job-level wrapper --------------------------------------------------------------------

def run_training_job(job: JobSpec, scale: Scale, loader, job_dir: Path, device: torch.device,
                     deadline: float | None, identity: str, impl_hash: str) -> dict:
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
                        deadline=deadline, micro_chunks=MICRO_CHUNKS)
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
    (job_dir / "checkpoints.json").write_text(__import__("json").dumps(
        dict(ledger=result.ledger, roles=roles["roles"], replay=roles["replay"]), indent=2))
    return info
