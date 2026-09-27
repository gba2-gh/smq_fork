"""Training loop for B-code (docs/plans/THREE_STAGE_MOTION_PLAN.md §4). No
action label enters this module: only code IDs and real frame counts
(window-length metadata, not annotations) are used.

Two modes share one loop:
- epoch mode (max_steps=None, the v1.0-v1.5 default): evaluate once per epoch,
  patience counted in epochs, constant learning rate;
- step mode (max_steps set): evaluate every `eval_every` optimizer steps,
  patience counted in steps, optional linear warmup and cosine decay.
The defaults reproduce the v1.5 trainer exactly.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from script.action_transport.budget import check_deadline
from script.action_transport.stageb_data import Chunk, build_mask, stable_seed
from script.action_transport.stageb_model import CodeContextModel, ModelConfig

DEPTH_BUCKETS = (1, 2, 3, 4)  # distance to the nearest unmasked position; 4 means >= 4


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    grad_clip: float = 1.0
    max_epochs: int = 500  # v1.5: 50 was binding (best epoch 44-49 in all stage_b_001 cells)
    patience: int = 8  # epochs (epoch mode only)
    mask_fraction: float = 0.30
    mask_span: int = 8
    # Optimization-study factors; defaults = v1.5 behavior.
    dropout: float = 0.1
    norm_first: bool = False
    tie_head: bool = False
    position_mode: str = "sinusoidal"
    warmup_steps: int = 0
    schedule: str = "constant"  # "constant" | "cosine" (cosine needs max_steps)
    max_steps: int | None = None  # set -> step mode
    eval_every: int = 100  # step mode only
    patience_steps: int = 1000  # step mode only


def _encode(chunks: list[Chunk], masks: list[np.ndarray], mask_id: int, device: torch.device):
    """chunks + one boolean mask per chunk (over [0, valid_len)) -> tensors.
    Padded positions (>= valid_len) are never masked, never a loss target
    (weight 0), and excluded from attention via padding_mask."""
    length = len(chunks[0].codes)
    input_codes = np.stack([c.codes.copy() for c in chunks])
    targets = np.stack([c.codes.copy() for c in chunks])
    weights = np.zeros((len(chunks), length), dtype=np.float32)
    mask_indicator = np.zeros((len(chunks), length), dtype=bool)
    padding_mask = np.ones((len(chunks), length), dtype=bool)
    for i, (chunk, mask) in enumerate(zip(chunks, masks)):
        padding_mask[i, :chunk.valid_len] = False
        weights[i, :chunk.valid_len] = chunk.lengths[:chunk.valid_len]
        mask_indicator[i, :chunk.valid_len] = mask
        input_codes[i, :chunk.valid_len][mask] = mask_id
    to = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    return (to(input_codes, torch.int64), to(targets, torch.int64), to(weights, torch.float32),
            to(mask_indicator, torch.bool), to(padding_mask, torch.bool))


def masked_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
    ce = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")  # [B,L]
    w = weights * mask.float()
    return (ce * w).sum() / w.sum().clamp_min(1e-8)


def fixed_masks(chunks: list[Chunk], base: int, cfg: "TrainConfig") -> list[np.ndarray]:
    """Deterministic per-chunk masks (fixed across epochs and model seeds)."""
    return [build_mask(c.valid_len, np.random.default_rng(stable_seed(c.recording, c.start, base)),
                       cfg.mask_fraction, cfg.mask_span) for c in chunks]


def mask_depths(mask: np.ndarray) -> np.ndarray:
    """For each masked position, the distance to the nearest unmasked valid
    position (1 = at a span edge). Large if a chunk has no unmasked position."""
    unmasked = np.flatnonzero(~mask)
    masked = np.flatnonzero(mask)
    if not len(unmasked):
        return np.full(len(masked), 10**6)
    return np.abs(masked[:, None] - unmasked[None, :]).min(axis=1)


def _copy_predictions(chunk: Chunk, mask: np.ndarray) -> np.ndarray:
    """The copy baseline: the nearest unmasked code (left wins ties)."""
    valid = chunk.codes[:chunk.valid_len]
    unmasked = np.flatnonzero(~mask)
    masked = np.flatnonzero(mask)
    if not len(unmasked):
        return np.full(len(masked), -1)
    nearest = unmasked[np.abs(masked[:, None] - unmasked[None, :]).argmin(axis=1)]
    return valid[nearest]


@torch.no_grad()
def evaluate_masked(model: CodeContextModel, chunks: list[Chunk], masks: list[np.ndarray],
                    device: torch.device, batch: int = 256) -> dict:
    """Masked-position loss and accuracy for the model and the copy baseline,
    overall and by span depth. Label-free: targets are the codes themselves."""
    model.eval()
    model_hits, copy_hits, depths = [], [], []
    loss_sum = weight_sum = 0.0
    for start in range(0, len(chunks), batch):
        part, part_masks = chunks[start:start + batch], masks[start:start + batch]
        inp, tgt, w, m, pad = _encode(part, part_masks, model.mask_id, device)
        _, logits = model(inp, pad)
        ce = F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
        ww = w * m.float()
        loss_sum += float((ce * ww).sum())
        weight_sum += float(ww.sum())
        pred = logits.argmax(-1).cpu().numpy()
        for i, (chunk, mask) in enumerate(zip(part, part_masks)):
            idx = np.flatnonzero(mask)
            if not len(idx):
                continue
            truth = chunk.codes[:chunk.valid_len][idx]
            model_hits.append(pred[i, idx] == truth)
            copy_hits.append(_copy_predictions(chunk, mask) == truth)
            depths.append(np.minimum(mask_depths(mask), DEPTH_BUCKETS[-1]))
    model_hits = np.concatenate(model_hits)
    copy_hits = np.concatenate(copy_hits)
    depths = np.concatenate(depths)
    by_depth = {}
    for d in DEPTH_BUCKETS:
        sel = depths == d
        if sel.any():
            key = f"{d}+" if d == DEPTH_BUCKETS[-1] else str(d)
            by_depth[key] = {"n": int(sel.sum()), "model_acc": float(model_hits[sel].mean()),
                             "copy_acc": float(copy_hits[sel].mean())}
    return {"loss": loss_sum / max(weight_sum, 1e-8), "model_acc": float(model_hits.mean()),
            "copy_acc": float(copy_hits.mean()), "n_masked": int(len(model_hits)), "by_depth": by_depth}


def _baselines(chunks: list[Chunk], masks: list[np.ndarray], global_code_freq: np.ndarray) -> tuple[float, float]:
    """(most-frequent-code accuracy, copy-baseline accuracy) at masked positions."""
    most_frequent = int(global_code_freq.argmax())
    freq_hits, copy_hits = [], []
    for chunk, mask in zip(chunks, masks):
        idx = np.flatnonzero(mask)
        if not len(idx):
            continue
        truth = chunk.codes[:chunk.valid_len][idx]
        freq_hits.append(truth == most_frequent)
        copy_hits.append(_copy_predictions(chunk, mask) == truth)
    return float(np.concatenate(freq_hits).mean()), float(np.concatenate(copy_hits).mean())


def _lr_lambda(cfg: TrainConfig):
    def factor(step: int) -> float:
        if cfg.warmup_steps and step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        if cfg.schedule == "cosine":
            if cfg.max_steps is None:
                raise ValueError("cosine schedule needs max_steps")
            span = max(1, cfg.max_steps - cfg.warmup_steps)
            progress = min(1.0, (step - cfg.warmup_steps) / span)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))  # decays to 10%
        return 1.0
    return factor


def train(train_chunks: list[Chunk], val_chunks: list[Chunk], num_codes: int, seed: int,
          device: torch.device, cfg: TrainConfig = TrainConfig(),
          mask_base: int = 111, probe_chunks: list[Chunk] | None = None,
          deadline: float | None = None, val_eval_batch: int = 256) -> tuple[CodeContextModel, dict]:
    """Trains one B-code model; returns the lowest-validation-loss state.

    `mask_base` seeds the fixed validation mask (independent of `seed`, which
    controls weight init and minibatch order/masks). `probe_chunks`, if
    given, are training chunks evaluated at the end with fixed masks to
    separate underfitting (low training accuracy) from overfitting.
    `deadline` is a cooperative check (see budget.check_deadline), tested
    once per optimizer step -- v1.6: the previous per-cell-only deadline in
    run_stage_b.py could not interrupt a single very long training run."""
    torch.manual_seed(seed)
    model = CodeContextModel(ModelConfig(num_codes=num_codes, dropout=cfg.dropout,
                                         norm_first=cfg.norm_first, tie_head=cfg.tie_head,
                                         position_mode=cfg.position_mode)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda(cfg))

    val_masks = fixed_masks(val_chunks, mask_base, cfg)
    global_freq = np.zeros(num_codes, dtype=np.float64)
    for c in train_chunks:
        global_freq += np.bincount(c.codes[:c.valid_len], minlength=num_codes)
    freq_acc, nbr_acc = _baselines(val_chunks, val_masks, global_freq)

    step_mode = cfg.max_steps is not None
    best_val, best_at, best_state = float("inf"), -1, None
    history = []
    started = time.perf_counter()
    # v1.6: math.ceil (not floor division) so the final, smaller minibatch of
    # an epoch is trained on too, instead of being silently dropped every
    # epoch (it was still seen in other epochs via shuffling, but never in
    # its own tail position -- HuGaDB dropped 34/930 chunks per epoch, LARa
    # 13/2765).
    n_batches = max(1, math.ceil(len(train_chunks) / cfg.batch_size))
    step, running, running_n = 0, 0.0, 0
    stopped_by = "epoch_cap"

    def validate() -> float:
        """Batched (v1.6: a single forward pass over the whole validation
        set could exhaust memory on larger populations)."""
        model.eval()
        total, weight = 0.0, 0.0
        with torch.no_grad():
            for start in range(0, len(val_chunks), val_eval_batch):
                part = val_chunks[start:start + val_eval_batch]
                part_masks = val_masks[start:start + val_eval_batch]
                inp, tgt, w, m, pad = _encode(part, part_masks, model.mask_id, device)
                _, logits = model(inp, pad)
                ce = F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
                ww = w * m.float()
                total += float((ce * ww).sum())
                weight += float(ww.sum())
        model.train()
        return total / max(weight, 1e-8)

    done = False
    for epoch in range(10**9 if step_mode else cfg.max_epochs):
        model.train()
        epoch_rng = np.random.default_rng(seed * 1_000_003 + epoch)
        order = epoch_rng.permutation(len(train_chunks))
        epoch_loss = 0.0
        for b in range(n_batches):
            check_deadline(deadline)
            idx = order[b * cfg.batch_size:(b + 1) * cfg.batch_size]
            batch = [train_chunks[i] for i in idx]
            masks = [build_mask(c.valid_len, epoch_rng, cfg.mask_fraction, cfg.mask_span) for c in batch]
            inp, tgt, w, m, pad = _encode(batch, masks, model.mask_id, device)
            _, logits = model(inp, pad)
            loss = masked_loss(logits, tgt, w, m)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            optimizer.step()
            scheduler.step()
            step += 1
            value = float(loss.detach())
            epoch_loss += value / n_batches
            running += value
            running_n += 1
            if step_mode and step % cfg.eval_every == 0:
                val_loss = validate()
                history.append({"step": step, "epoch": epoch, "train_loss": running / running_n,
                                "val_loss": val_loss, "lr": scheduler.get_last_lr()[0],
                                "grad_norm": grad_norm})
                running, running_n = 0.0, 0
                if val_loss < best_val:  # strict '<': ties keep the earlier point
                    best_val, best_at = val_loss, step
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if step - best_at >= cfg.patience_steps:
                    stopped_by, done = "patience", True
                    break
            if step_mode and step >= cfg.max_steps:
                if not history or history[-1]["step"] != step:
                    val_loss = validate()
                    history.append({"step": step, "epoch": epoch, "train_loss": running / max(1, running_n),
                                    "val_loss": val_loss, "lr": scheduler.get_last_lr()[0],
                                    "grad_norm": grad_norm})
                    if val_loss < best_val:
                        best_val, best_at = val_loss, step
                        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stopped_by, done = "step_cap", True
                break
        if done:
            break
        if not step_mode:
            val_loss = validate()
            history.append({"epoch": epoch, "train_loss": epoch_loss, "val_loss": val_loss})
            if val_loss < best_val:
                best_val, best_at = val_loss, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if epoch - best_at >= cfg.patience:
                stopped_by = "patience"
                break

    model.load_state_dict(best_state)
    val_eval = evaluate_masked(model, val_chunks, val_masks, device)
    info = {
        "model_masked_acc": val_eval["model_acc"],
        "beats_neighbor_baseline": val_eval["model_acc"] > nbr_acc,
        "stopped_by": stopped_by,
        "best_epoch": (best_at if not step_mode
                       else next(h["epoch"] for h in history if h["step"] == best_at)),
        "best_step": best_at if step_mode else None,
        "best_val_loss": best_val,
        "epochs_run": len({h["epoch"] for h in history}),
        "steps_run": step,
        "history": history,
        "diagnostics": {"val_loss": best_val, "val_masked_positions": val_eval["n_masked"],
                        "frequency_baseline_acc": freq_acc, "neighbor_baseline_acc": nbr_acc},
        "val_by_depth": val_eval["by_depth"],
        "train_config": asdict(cfg),
        "runtime_seconds": time.perf_counter() - started,
        "seed": seed, "n_train_chunks": len(train_chunks), "n_val_chunks": len(val_chunks),
    }
    if probe_chunks:
        probe = evaluate_masked(model, probe_chunks, fixed_masks(probe_chunks, mask_base + 1, cfg), device)
        info["train_probe"] = {k: probe[k] for k in ("loss", "model_acc", "copy_acc")}
    return model, info


def save_checkpoint(path: Path, model: CodeContextModel, info: dict, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.pt")
    torch.save({"state_dict": model.state_dict(), "config": model.cfg.__dict__,
                "info": info, "extra": extra}, temp)
    os.replace(temp, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[CodeContextModel, dict, dict]:
    blob = torch.load(path, map_location=device)
    model = CodeContextModel(ModelConfig(**blob["config"])).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob["info"], blob["extra"]
