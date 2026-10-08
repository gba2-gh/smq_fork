"""One training unit for the velocity-term variants (V1, V2a, V2b): train_masq.train_unit's loop with one calibrated extra term.

The calibration (a) uses a fixed, label-free sample (the first CALIB_FILES recordings by file name, cropped to
CALIB_MAX_FRAMES), (b) runs under a forked RNG, and (c) calls only the encoder (never the quantizer), so the random stream,
the codebook initialization and the batch order are identical to those of the matching M-JL unit: the variant differs from
M-JL only by the extra term. Labels are read only through get_num_actions (via train_masq.prepare).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim

from script.week5.masq_experiments import config as C
from script.week5.masq_experiments import losses_variants as LV
from script.week5.masq_experiments.losses import compute_losses
from script.week5.masq_experiments.train_masq import data_paths, prepare


def calibration_batch(dataset: str, device: torch.device):
    st = C.DATASETS[dataset]
    d = data_paths(dataset)["features"]
    files = sorted(os.listdir(d))[:C.CALIB_FILES]
    arrays = [np.load(d / f)[:, :C.CALIB_MAX_FRAMES] for f in files]
    T = max(a.shape[1] for a in arrays)
    x = torch.zeros(len(arrays), st["num_features"], T, st["num_joints"], st["num_person"])
    mask = torch.zeros_like(x)
    for i, a in enumerate(arrays):
        x[i, :, :a.shape[1]] = torch.from_numpy(a)
        mask[i, :, :a.shape[1]] = 1.0
    return x.to(device), mask.to(device), files


def encoder_latent(model, x, mask):
    """The encoder's output for a batch, in the packed layout (N*M*V, Z, T), without touching the quantizer."""
    N, Ch, T, V, M = x.shape
    xp = x.permute(0, 4, 3, 1, 2).contiguous().view(N * M * V, Ch, T)
    mp = mask.permute(0, 4, 3, 1, 2).contiguous().view(N * M * V, Ch, T)
    return model.encoder(xp, mp)


def calibrate(model, unit: C.Unit, st: dict, device: torch.device) -> dict:
    x, mask, files = calibration_batch(unit.dataset, device)
    mode, rho = st["vel_mode"], st["rho"]
    latent = None
    if mode == "latent_smooth":
        was_training = model.training
        model.eval()
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []), torch.no_grad():
            latent = encoder_latent(model, x, mask)
        model.train(was_training)
    out = LV.calibrate_weight(mode, rho, x, mask, st["mse_loss_weight"], latent=latent)
    out["files"] = files
    return out


def train_unit_variant(unit: C.Unit, out_dir: Path, device: torch.device, smoke: bool = False, identity: str | None = None) -> dict:
    st = C.settings(unit, smoke)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, gen, num_actions = prepare(unit, smoke)
    model.train()
    model.to(device)
    calib = calibrate(model, unit, st, device)
    print(f"[{unit.unit_id}] calibration: {json.dumps({k: v for k, v in calib.items() if k != 'files'})}", flush=True)

    epochs, batch_size = st["epochs"], st["batch_size"]
    max_batches = st.get("max_batches_per_epoch")
    num_batches = min(gen.num_batches(batch_size), max_batches) if max_batches else gen.num_batches(batch_size)
    optimizer = optim.Adam(model.parameters(), lr=st["lr"])

    counter = {"n": 0}
    original_replace = model.vq.replace

    def counted_replace(batch_samples, batch_mask, random_generator=None):
        counter["n"] += int(batch_mask.sum())
        return original_replace(batch_samples, batch_mask, random_generator=random_generator)

    model.vq.replace = counted_replace

    (out_dir / "config.json").write_text(json.dumps(dict(settings=st, identity=identity, num_actions=num_actions, calibration=calib), indent=2))
    heartbeat, losses_path = out_dir / "heartbeat.txt", out_dir / "losses.jsonl"
    if losses_path.exists():
        losses_path.unlink()
    per_epoch, t0 = [], time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        sums = dict(rec=0.0, commit=0.0, tc=0.0, vel=0.0)
        before, te, batch_idx = counter["n"], time.perf_counter(), 0
        while gen.has_next() and (not max_batches or batch_idx < max_batches):
            heartbeat.write_text(f"{time.strftime('%H:%M:%S')} epoch {epoch + 1}/{epochs} batch {batch_idx + 1}/{num_batches}\n")
            x, mask = gen.next_batch(batch_size)
            x, mask = x.to(device), mask.to(device)
            optimizer.zero_grad()
            x_hat = model(x, mask)
            L = compute_losses(model, x, mask, x_hat, st["mse_loss_weight"], st["commit_weight"], 0.0, st["joint_distance_recons"])
            extra = LV.variant_term(st["vel_mode"], calib["weight"], model, x, mask, x_hat)
            total = L["total"] + extra
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at epoch {epoch + 1} batch {batch_idx + 1}")
            total.backward()
            optimizer.step()
            for k, v in dict(rec=L["rec"], commit=L["commit"], tc=L["tc"], vel=extra).items():
                sums[k] += float(v)
            batch_idx += 1
        gen.reset()
        row = dict(epoch=epoch + 1, batches=batch_idx, seconds=time.perf_counter() - te, dead_code_replacements=counter["n"] - before,
                   **{k: v / max(batch_idx, 1) for k, v in sums.items()})
        per_epoch.append(row)
        with open(losses_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{unit.unit_id}] epoch {epoch + 1}/{epochs} rec {row['rec']:.2f} commit {row['commit']:.2f} extra {row['vel']:.2f} dead {row['dead_code_replacements']}", flush=True)

    ckpt = out_dir / f"epoch-{epochs}.model"
    tmp = ckpt.with_suffix(".tmp")
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, ckpt)
    cluster = model.vq.cluster_size.squeeze().detach().cpu()
    info = dict(unit_id=unit.unit_id, identity=identity, smoke=smoke, epochs=epochs, seconds=time.perf_counter() - t0,
                seconds_per_batch=sum(r["seconds"] for r in per_epoch) / max(sum(r["batches"] for r in per_epoch), 1),
                batches_per_epoch=num_batches, num_actions=num_actions, jlsd_p=unit.jlsd_p, vel_weight=calib["weight"],
                calibration=calib, peak_gpu_bytes=int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None,
                ema_cluster_size_nonzero=int((cluster > 0.5).sum()), dead_code_replacements_total=counter["n"],
                per_epoch=per_epoch, checkpoint=ckpt.name)
    C.write_json_atomic(out_dir / "training_info.json", info)
    return info
