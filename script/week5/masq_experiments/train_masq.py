"""One training unit: a faithful copy of model.Trainer.train (no micro-batching) plus the MASQ velocity term.

Seeding follows main.py: batch_gen seeds `random` with 42 at import; then random, torch and cuda are seeded with the unit
seed; the model is built, then the batch generator. Labels are read only through `get_num_actions` (the number of classes,
K = C, the benchmark convention shared by every baseline).
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import torch
from torch import optim

import batch_gen as BG  # noqa: F401  (import seeds `random` with 42, as main.py does)
from batch_gen import BatchGenerator
from script.week5.masq_experiments import config as C
from script.week5.masq_experiments.losses import compute_losses
from script.week5.masq_experiments.model_masq import build_masq
from src.model.utils import get_num_actions


def data_paths(dataset: str) -> dict:
    root = C.ROOT / "data" / dataset
    return dict(features=root / "features", gt=root / "groundTruth", mapping=root / "mapping" / "mapping.txt")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # as main.py (documented there)


def prepare(unit: C.Unit, smoke: bool = False):
    """Seed, build the model, then the batch generator (the order of main.py). Returns (model, batch_gen, num_actions)."""
    st = C.settings(unit, smoke)
    paths = data_paths(unit.dataset)
    seed_everything(unit.seed)
    num_actions = get_num_actions(paths["gt"])
    model = build_masq(unit.dataset, num_actions, jlsd_p=unit.jlsd_p, grad_checkpoint=st["grad_checkpoint"])
    gen = BatchGenerator(features_path=paths["features"], sample_rate=st["sample_rate"], num_features=st["num_features"],
                         num_joints=st["num_joints"], num_person=st["num_person"])
    gen.read_data()
    return model, gen, num_actions


def train_unit(unit: C.Unit, out_dir: Path, device: torch.device, smoke: bool = False, identity: str | None = None) -> dict:
    st = C.settings(unit, smoke)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, gen, num_actions = prepare(unit, smoke)
    model.train()
    model.to(device)

    epochs, batch_size = st["epochs"], st["batch_size"]
    max_batches = st.get("max_batches_per_epoch")
    num_batches = gen.num_batches(batch_size)
    if max_batches:
        num_batches = min(num_batches, max_batches)
    optimizer = optim.Adam(model.parameters(), lr=st["lr"])

    # dead-code replacements, counted without changing the replacement
    counter = {"n": 0}
    original_replace = model.vq.replace

    def counted_replace(batch_samples, batch_mask, random_generator=None):
        counter["n"] += int(batch_mask.sum())
        return original_replace(batch_samples, batch_mask, random_generator=random_generator)

    model.vq.replace = counted_replace

    (out_dir / "config.json").write_text(json.dumps(dict(settings=st, identity=identity, num_actions=num_actions), indent=2))
    heartbeat = out_dir / "heartbeat.txt"
    losses_path = out_dir / "losses.jsonl"
    if losses_path.exists():
        losses_path.unlink()
    per_epoch, t0 = [], time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        sums = dict(rec=0.0, commit=0.0, tc=0.0, vel=0.0)
        before = counter["n"]
        te = time.perf_counter()
        batch_idx = 0
        while gen.has_next() and (not max_batches or batch_idx < max_batches):
            heartbeat.write_text(f"{time.strftime('%H:%M:%S')} epoch {epoch + 1}/{epochs} batch {batch_idx + 1}/{num_batches}\n")
            x, mask = gen.next_batch(batch_size)
            x, mask = x.to(device), mask.to(device)
            optimizer.zero_grad()
            x_hat = model(x, mask)
            L = compute_losses(model, x, mask, x_hat, st["mse_loss_weight"], st["commit_weight"], st["vel_weight"],
                               st["joint_distance_recons"])
            if not torch.isfinite(L["total"]):
                raise FloatingPointError(f"non-finite loss at epoch {epoch + 1} batch {batch_idx + 1}: "
                                         f"{ {k: float(v) for k, v in L.items()} }")
            L["total"].backward()
            optimizer.step()
            for k in sums:
                sums[k] += float(L[k])
            batch_idx += 1
        gen.reset()
        row = dict(epoch=epoch + 1, batches=batch_idx, seconds=time.perf_counter() - te,
                   dead_code_replacements=counter["n"] - before, **{k: v / max(batch_idx, 1) for k, v in sums.items()})
        per_epoch.append(row)
        with open(losses_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{unit.unit_id}] epoch {epoch + 1}/{epochs} rec {row['rec']:.6f} commit {row['commit']:.6f} "
              f"vel {row['vel']:.6f} dead {row['dead_code_replacements']}", flush=True)

    ckpt = out_dir / f"epoch-{epochs}.model"
    tmp = ckpt.with_suffix(".tmp")
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, ckpt)
    cluster = model.vq.cluster_size.squeeze().detach().cpu()
    info = dict(unit_id=unit.unit_id, identity=identity, smoke=smoke, epochs=epochs, seconds=time.perf_counter() - t0,
                seconds_per_batch=sum(r["seconds"] for r in per_epoch) / max(sum(r["batches"] for r in per_epoch), 1),
                batches_per_epoch=num_batches, num_actions=num_actions, jlsd_p=unit.jlsd_p, vel_weight=unit.vel_weight,
                peak_gpu_bytes=int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None,
                ema_cluster_size_nonzero=int((cluster > 0.5).sum()), dead_code_replacements_total=counter["n"],
                per_epoch=per_epoch, checkpoint=ckpt.name)
    C.write_json_atomic(out_dir / "training_info.json", info)
    return info
