"""Explicit configuration, paired autoencoder training, extraction and job
scheduling for the FSQ reconstruction pilot
(docs/plans/FSQ_RECONSTRUCTION_PILOT_PLAN.md Revision 3).

Implements the REQUIRED experiment (§3): HuGaDB, one paired autoencoder
seed, VQ512 vs FSQ512[8,8,8], four streams (native/K-means-512 x VQ/FSQ),
three downstream seeds. LARa, a second autoencoder seed, and the
released-encoder reference are declared but not auto-scheduled here -- see
README.md "Scope of this implementation".
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from batch_gen import BatchGenerator
from src.model.smq import SMQModel
from src.model.utils import distance_joints, process_mask

from script.action_transport import categorical, config as at_config, continuous, transport
from script.action_transport.budget import DeadlineExceeded
from script.fsq_reconstruction.quantizers import (
    FSQSMQModel, PatchFSQAdapter, build_fsq_arm, build_shared_backbone, build_vq_arm,
    isolated_rng, reseed_all,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "results" / "fsq_reconstruction"
PILOT_VERSION = "1.0"

# --- dataset defaults (main.py's get_dataset_defaults, HuGaDB/LARa only; §1 scope) -----

DATASET_DEFAULTS = {
    "hugadb": dict(num_features=6, num_joints=6, num_person=1, patch_size=15, fps=60),
    "lara": dict(num_features=6, num_joints=22, num_person=1, patch_size=12, fps=50),
}


@dataclass(frozen=True)
class PilotConfig:
    dataset: str = "hugadb"
    levels: tuple[int, ...] = (8, 8, 8)
    autoencoder_seed: int = 111
    downstream_seeds: tuple[int, ...] = (111, 222, 1538574472)
    preprocessing_seed: int = 111  # PCA/K-means sample seed; fixed across downstream seeds (§6)

    epochs: int = 30
    checkpoint_epochs: tuple[int, ...] = (0, 10, 20, 30)
    batch_size: int = 8
    micro_batch_size: int | None = None
    lr: float = 5e-4
    mse_loss_weight: float = 0.001
    commit_weight: float = 1.0
    joint_distance_recons: bool = True
    filters: int = 128
    num_layers: int = 3
    latent_dim: int = 16

    vq_dead_code_threshold: int = 1  # adapted for K=512; see README "dead-code threshold"
    vq_decay: float = 0.5
    vq_sampling_quantile: float = 0.5
    vq_replacement_strategy: str = "representative"

    kmeans_k: int = 512
    kmeans_sample_cap: int = 10_000

    replacement_warn_fraction: float = 0.05  # diagnostic flag threshold, not a validity gate (§4)
    contextualizer_max_steps: int = 5000
    contextualizer_eval_every: int = 100
    contextualizer_patience_steps: int = 1000
    final_rungs: tuple[int, ...] = (500, 1000, 2000, 4000)
    fit_outer_cap: int = 50  # §7: "50 outer fitting iterations, 25 inner steps"
    fit_inner_steps: int = 25

    @property
    def num_codes(self) -> int:
        return int(np.prod(self.levels))

    def smq_kwargs(self, num_actions_for_vq: int | None = None) -> dict:
        d = DATASET_DEFAULTS[self.dataset]
        return dict(in_channels=d["num_features"], filters=self.filters, num_layers=self.num_layers,
                   latent_dim=self.latent_dim, num_actions=num_actions_for_vq or self.num_codes,
                   num_joints=d["num_joints"], num_person=d["num_person"], patch_size=d["patch_size"],
                   kmeans=False, kmeans_metric="euclidean", sampling_quantile=self.vq_sampling_quantile,
                   replacement_strategy=self.vq_replacement_strategy, decay=self.vq_decay,
                   dead_code_threshold=self.vq_dead_code_threshold)


def config_hash(cfg: PilotConfig) -> str:
    return hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True, default=str).encode()).hexdigest()[:16]


# --- fixed batch/epoch order (§4.3: precomputed schedule shared by both arms) ----------

def precompute_epoch_orders(features_path: Path, num_epochs: int, seed: int) -> list[list[str]]:
    files = sorted(os.listdir(features_path))
    if not files:
        raise RuntimeError(f"no feature files found in {features_path}")
    rng = np.random.default_rng(seed)
    orders = []
    for _ in range(num_epochs):
        order = rng.permutation(len(files))
        orders.append([files[i] for i in order])
    return orders


class FixedOrderBatchGenerator(BatchGenerator):
    """BatchGenerator whose epoch order comes from a precomputed schedule
    instead of `random.shuffle` (batch_gen.py seeds Python's global `random`
    at import time, so relying on its shuffle would not be reproducible
    across two separately-imported training runs in the same process)."""

    def __init__(self, features_path, sample_rate, num_features, num_joints, num_person,
                epoch_orders: list[list[str]]):
        super().__init__(features_path, sample_rate, num_features, num_joints, num_person)
        self.epoch_orders = epoch_orders
        self._epoch = 0

    def read_data(self):
        self.list_of_examples = list(self.epoch_orders[0])
        self._epoch = 1

    def reset(self):
        self.index = 0
        self.list_of_examples = list(self.epoch_orders[self._epoch % len(self.epoch_orders)])
        self._epoch += 1


# --- gradient diagnostics (§4: per-loss encoder gradient, epoch 0/10/20/30) ------------

def gradient_diagnostics(model, batch_input: torch.Tensor, batch_mask: torch.Tensor,
                         mse_loss_weight: float, commit_weight: float,
                         joint_distance_recons: bool = True) -> dict:
    """Eval-mode forward with autograd enabled (no optimizer step, no
    EMA/replacement update -- eval mode makes SkeletonMotionQuantizer skip
    its training-only branch). Measures d(rec_loss)/d(pre-quantization
    encoder output) and d(commit_loss)/d(same), separately. Restores the
    model's training-mode flag and RNG state (torch.no_grad is not used
    here on purpose: autograd is required)."""
    was_training = model.training
    cpu_state = torch.get_rng_state()
    model.eval()
    reconstructed = model(batch_input, batch_mask)
    latent = model.latent
    if not latent.requires_grad:
        latent.requires_grad_(True)
    x, x_hat = (distance_joints(batch_input), distance_joints(reconstructed)) if joint_distance_recons \
        else (batch_input, reconstructed)
    rec_loss = mse_loss_weight * torch.mean(F.mse_loss(x, x_hat, reduction="none"))
    commit_loss = commit_weight * model.commit_loss

    def grad_stats(loss: torch.Tensor):
        if not loss.requires_grad:
            return None
        grads = torch.autograd.grad(loss, latent, retain_graph=True, allow_unused=True)
        grad = grads[0]
        if grad is None:
            return None
        return grad.detach()

    rec_grad = grad_stats(rec_loss)
    commit_grad = grad_stats(commit_loss)
    rec_l2 = float(rec_grad.norm()) if rec_grad is not None else 0.0
    rec_rms = float(rec_grad.pow(2).mean().sqrt()) if rec_grad is not None else 0.0
    commit_l2 = float(commit_grad.norm()) if commit_grad is not None else 0.0
    commit_rms = float(commit_grad.pow(2).mean().sqrt()) if commit_grad is not None else 0.0
    ratio = (commit_l2 / rec_l2) if rec_l2 > 0 else None
    cosine = None
    if rec_grad is not None and commit_grad is not None and rec_l2 > 0 and commit_l2 > 0:
        cosine = float(F.cosine_similarity(rec_grad.flatten(), commit_grad.flatten(), dim=0))
    model.train(was_training)
    torch.set_rng_state(cpu_state)
    return dict(rec_grad_l2=rec_l2, rec_grad_rms=rec_rms, commit_grad_l2=commit_l2,
               commit_grad_rms=commit_rms, ratio_commit_over_rec=ratio, cosine=cosine,
               rec_loss=float(rec_loss.detach()), commit_loss=float(commit_loss.detach()))


# --- matched training (§4) -------------------------------------------------------------

@dataclass
class TrainingLog:
    arm: str
    epoch_rows: list[dict] = field(default_factory=list)
    gradient_rows: list[dict] = field(default_factory=list)
    occupancy_rows: list[dict] = field(default_factory=list)


def _vq_occupancy(vq_module) -> dict:
    cluster_size = vq_module.cluster_size.detach().cpu().numpy().reshape(-1)
    return dict(active_codes=int((cluster_size >= 1).sum()),
               below_threshold=int((cluster_size < vq_module.threshold_ema_dead_code).sum()),
               cluster_size_min=float(cluster_size.min()), cluster_size_median=float(np.median(cluster_size)),
               cluster_size_max=float(cluster_size.max()))


def _fsq_occupancy(vq_module) -> dict:
    """Fraction of the (post-norm, pre-tanh) diagnostic-batch activations
    saturating the FSQ bound (|z| > 3 -> tanh(z) within 1% of +/-1). A high
    fraction here means the reconstruction gradient into the encoder is
    near zero and the codebook will collapse -- catch it at epoch 0, not
    after a multi-hour downstream job reports "only 1 occupied prototype"."""
    z = vq_module.last_pre_bound
    if z is None:
        return dict(saturation_frac=None)
    return dict(saturation_frac=float((z.abs() > 3).float().mean()))


def train_one_arm(model, save_dir: Path, features_path: Path, cfg: PilotConfig, seed: int,
                  epoch_orders: list[list[str]], arm: str, diagnostic_batch=None,
                  device: torch.device = torch.device("cpu"), deadline: float | None = None) -> TrainingLog:
    """Reimplements model.Trainer.train's loss computation (rec + commit +
    tc, joint-distance reconstruction, gradient accumulation) directly,
    rather than reusing Trainer, because Trainer.__init__ always builds a
    fresh SMQModel(VQ) internally and this pilot needs to train an
    already-constructed, externally-initialized model (VQ or FSQ) under a
    fixed batch schedule. The per-step loss arithmetic below is copied
    verbatim from model.py's Trainer.train (see there for provenance)."""
    reseed_all(seed)
    model.train()
    model.to(device)
    save_dir.mkdir(parents=True, exist_ok=True)
    mse = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    d = DATASET_DEFAULTS[cfg.dataset]
    batch_gen = FixedOrderBatchGenerator(features_path, sample_rate=1, num_features=d["num_features"],
                                        num_joints=d["num_joints"], num_person=d["num_person"],
                                        epoch_orders=epoch_orders)
    batch_gen.read_data()
    num_batches = batch_gen.num_batches(cfg.batch_size)
    chunk_size = cfg.micro_batch_size or cfg.batch_size
    log = TrainingLog(arm=arm)

    def checkpoint_and_diagnose(epoch: int):
        torch.save(model.state_dict(), save_dir / f"epoch-{epoch}.model")
        if diagnostic_batch is not None:
            batch_input, batch_mask = diagnostic_batch
            diag = gradient_diagnostics(model, batch_input.to(device), batch_mask.to(device),
                                        cfg.mse_loss_weight, cfg.commit_weight, cfg.joint_distance_recons)
            diag.update(epoch=epoch, arm=arm)
            log.gradient_rows.append(diag)
        # gradient_diagnostics' forward call (above) is what populates
        # model.vq.last_pre_bound for this checkpoint's FSQ saturation read.
        if isinstance(model.vq, PatchFSQAdapter):
            occ = dict(kind="fsq", **_fsq_occupancy(model.vq))
        else:
            occ = dict(kind="vq", **_vq_occupancy(model.vq))
        occ.update(epoch=epoch, arm=arm)
        log.occupancy_rows.append(occ)

    if 0 in cfg.checkpoint_epochs:
        checkpoint_and_diagnose(0)

    for epoch in range(cfg.epochs):
        if deadline is not None and time.perf_counter() > deadline:
            print(f"[WARN] {arm}: compute deadline reached before epoch {epoch + 1}/{cfg.epochs}; "
                 "stopping this arm's training early (checkpoints/diagnostics up to the last completed "
                 "checkpoint epoch remain valid)", flush=True)
            log.epoch_rows.append(dict(arm=arm, epoch=epoch + 1, status="interrupted"))
            break
        epoch_rec = epoch_commit = epoch_tc = 0.0
        replaced_total = valid_patches_total = 0
        batch_idx = 0
        while batch_gen.has_next():
            batch_input, mask = batch_gen.next_batch(cfg.batch_size)
            batch_input, mask = batch_input.to(device), mask.to(device)
            actual_batch_size = batch_input.shape[0]
            optimizer.zero_grad()
            batch_rec = batch_commit = batch_tc = 0.0
            for start in range(0, actual_batch_size, chunk_size):
                end = min(start + chunk_size, actual_batch_size)
                chunk_input, chunk_mask = batch_input[start:end], mask[start:end]
                chunk_weight = (end - start) / actual_batch_size
                reconstructed = model(chunk_input, chunk_mask)
                x, x_hat = (distance_joints(chunk_input), distance_joints(reconstructed)) \
                    if cfg.joint_distance_recons else (chunk_input, reconstructed)
                rec_loss = cfg.mse_loss_weight * torch.mean(mse(x, x_hat))
                commit_loss = cfg.commit_weight * model.commit_loss
                tc_loss = model.vq.tc_loss
                loss = (rec_loss + commit_loss + tc_loss) * chunk_weight
                loss.backward()
                batch_rec += rec_loss.item() * chunk_weight
                batch_commit += commit_loss.item() * chunk_weight
                batch_tc += float(tc_loss) * chunk_weight
            optimizer.step()
            epoch_rec += batch_rec
            epoch_commit += batch_commit
            epoch_tc += batch_tc
            batch_idx += 1
        batch_gen.reset()
        log.epoch_rows.append(dict(arm=arm, epoch=epoch + 1, rec_loss=epoch_rec / num_batches,
                                   commit_loss=epoch_commit / num_batches, tc_loss=epoch_tc / num_batches))
        if (epoch + 1) in cfg.checkpoint_epochs:
            checkpoint_and_diagnose(epoch + 1)
    return log


def build_diagnostic_batch(features_path: Path, num_features: int, num_joints: int, num_person: int,
                           n_recordings: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """One fixed, small padded batch used only for gradient diagnostics
    (never for training)."""
    gen = BatchGenerator(features_path, sample_rate=1, num_features=num_features,
                         num_joints=num_joints, num_person=num_person)
    gen.list_of_examples = sorted(os.listdir(features_path))[:n_recordings]
    gen.index = 0
    batch_input, mask = gen.next_batch(n_recordings)
    return batch_input, mask


# --- reconstruction evaluation (§4: eval mode, one recording per batch) ---------------

@torch.no_grad()
def evaluate_reconstruction(model, features_path: Path, num_features: int, num_joints: int,
                            num_person: int, joint_distance_recons: bool, device: torch.device) -> dict:
    """Checkpoint reconstruction error under the training objective (padded
    per-batch as at training time is NOT used here: §4 requires eval mode
    with one unpadded recording per batch for every extraction/evaluation
    path, including this one) and a valid-frame diagnostic with an explicit
    denominator (every frame in every recording is real when batched one at
    a time, so the denominator is simply total valid elements)."""
    was_training = model.training
    model.eval()
    model.to(device)
    mse = nn.MSELoss(reduction="none")
    total_rec = total_valid_rec = 0.0
    total_n = total_valid_n = 0
    for name in sorted(os.listdir(features_path)):
        arr = np.load(features_path / name)  # (C,T,V,M)
        x = torch.as_tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.ones_like(x)
        reconstructed = model(x, mask)
        xj, xhat_j = (distance_joints(x), distance_joints(reconstructed)) if joint_distance_recons \
            else (x, reconstructed)
        se = mse(xj, xhat_j)
        total_rec += float(se.sum())
        total_n += se.numel()
        total_valid_rec += float(se.sum())  # every element is valid (single, unpadded recording)
        total_valid_n += se.numel()
    model.train(was_training)
    return dict(mean_sq_error=total_rec / max(1, total_n), valid_frame_mse=total_valid_rec / max(1, total_valid_n),
               valid_denominator=total_valid_n)


# --- extraction: native codes + pre-quantization patch features (§4, §6) --------------

def _repacked_latent(model, N: int, T: int, V: int, M: int) -> torch.Tensor:
    """Mirrors SMQModel.forward's repack of `model.latent` (packed
    (N*M*V, latent_dim, T)) into (N, T, V*M*latent_dim), the tensor the
    quantizer actually receives. SMQModel does not expose this intermediate
    itself, so this duplicates that small reshape (src/model/smq.py
    forward, lines computing `latent = latent.view(...)` through
    `.reshape(N, T, -1)`)."""
    latent = model.latent.view(N * M, V, model.latent_dim, T).contiguous()
    latent = latent.permute(0, 3, 1, 2)
    latent = latent.reshape(N, M, T, V, model.latent_dim)
    latent = latent.permute(0, 2, 3, 1, 4)
    return latent.reshape(N, T, -1)


@torch.no_grad()
def extract_recording(model, feature_path: Path, window: int, num_features: int, num_joints: int,
                      num_person: int, device: torch.device) -> dict:
    """One unpadded recording per batch, eval mode. Returns per-patch:
    flattened pre-quantization encoder features [P, W*D], native code id
    [P], and real frame count per patch [P] (<=W; the terminal patch may be
    shorter)."""
    was_training = model.training
    model.eval()
    arr = np.load(feature_path)  # (C,T,V,M)
    T = arr.shape[1]
    x = torch.as_tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.ones_like(x)
    model(x, mask)  # populates model.latent and model.indices; eval mode -> no side effects
    latent_ntd = _repacked_latent(model, N=1, T=T, V=num_joints, M=num_person)[0]  # (T, D)
    D = latent_ntd.shape[1]
    num_patches = int(np.ceil(T / window))
    pad = num_patches * window - T
    if pad:
        latent_ntd = F.pad(latent_ntd, (0, 0, 0, pad))
    patches = latent_ntd.reshape(num_patches, window, D)
    flat = patches.reshape(num_patches, window * D).cpu().numpy().astype(np.float32)
    lengths = np.full(num_patches, window, dtype=np.int32)
    if pad:
        lengths[-1] = window - pad
    ids_per_frame = model.indices[0].cpu().numpy()
    patch_ids = np.array([int(ids_per_frame[min(p * window, T - 1)]) for p in range(num_patches)], dtype=np.int64)
    model.train(was_training)
    return dict(flat=flat, ids=patch_ids, lengths=lengths, num_patches=num_patches)


@torch.no_grad()
def extract_dataset(model, features_path: Path, window: int, num_features: int, num_joints: int,
                    num_person: int, device: torch.device) -> dict[str, dict]:
    out = {}
    for name in sorted(os.listdir(features_path)):
        out[name] = extract_recording(model, features_path / name, window, num_features, num_joints,
                                      num_person, device)
    return out


# --- K-means-512 tokenization on pre-quantization encoder patches (§6) ----------------

@dataclass
class FittedVocabulary:
    scaler: StandardScaler
    pca: PCA
    kmeans: KMeans
    sample_names: list[str]
    n_fit_patches: int
    n_occupied: int


def fit_pca_kmeans(extracted: dict[str, dict], cfg: PilotConfig, pca_dim: int = 64) -> FittedVocabulary:
    """Standardize + randomized PCA + full KMeans(K=512) on <=10,000
    uniformly sampled complete patches (§6). Complete = a patch whose real
    length equals the patch window (excludes the terminal partial patch from
    fitting, per §6: 'Exclude incomplete terminal patches from fitting')."""
    window_len = None
    all_flat, all_names = [], []
    for name, rec in extracted.items():
        window_len = rec["lengths"].max() if window_len is None else window_len
        complete = rec["lengths"] == rec["lengths"].max()
        all_flat.append(rec["flat"][complete])
        all_names.extend([name] * int(complete.sum()))
    flat = np.concatenate(all_flat, axis=0)
    if len(flat) < cfg.kmeans_k:
        raise RuntimeError(f"only {len(flat)} complete patches available, need >= {cfg.kmeans_k} for K-means "
                          f"(K={cfg.kmeans_k}); comparison unavailable")
    rng = np.random.default_rng(cfg.preprocessing_seed)
    sample_size = min(cfg.kmeans_sample_cap, len(flat))
    sample_idx = rng.choice(len(flat), size=sample_size, replace=False)
    sample = flat[sample_idx]

    scaler = StandardScaler().fit(sample)
    sample_std = scaler.transform(sample)
    n_components = min(pca_dim, sample_std.shape[0], sample_std.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=cfg.preprocessing_seed)
    sample_pca = pca.fit_transform(sample_std)

    kmeans = KMeans(n_clusters=cfg.kmeans_k, init="k-means++", n_init=5, max_iter=300, tol=1e-4,
                    algorithm="lloyd", random_state=cfg.preprocessing_seed)
    kmeans.fit(sample_pca)
    occupied = len(set(kmeans.labels_.tolist()))
    return FittedVocabulary(scaler, pca, kmeans, [all_names[i] for i in sample_idx], len(sample), occupied)


def transform_and_assign(vocab: FittedVocabulary, flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pca_features [N, d], kmeans_ids [N])."""
    pca_features = vocab.pca.transform(vocab.scaler.transform(flat))
    ids = vocab.kmeans.predict(pca_features).astype(np.int64)
    return pca_features.astype(np.float32), ids


def project_to_pca(vocab_or_scaler_pca: tuple, flat: np.ndarray) -> np.ndarray:
    scaler, pca = vocab_or_scaler_pca
    return pca.transform(scaler.transform(flat)).astype(np.float32)


# --- identity / manifest -----------------------------------------------------------

def sha256_of_state_dict(state_dict: dict) -> str:
    import io
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(temp, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({k for row in rows for k in row})
    temp = path.with_suffix(".tmp.csv")
    with temp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


# --- end-to-end required pilot (§3, §11 priority 2): VQ512 vs FSQ512, one HuGaDB -------
# autoencoder seed, four streams, three downstream seeds, both OT methods -------------

def check_identity_against_released_meta(cfg: PilotConfig, extracted: dict[str, dict]) -> dict:
    """Reuses data.py's frame-count/window-grid/ground-truth loader (the
    window grid there depends only on real frame counts, which is identical
    regardless of which encoder produced the pre-quantization features --
    data.py: 'independent of normalization: it only depends on frame
    counts'). Asserts our own extraction produced the same per-recording
    frame count, so `RecordingMeta.starts/lengths/frame_count` can be reused
    unmodified for frame-broadcast scoring. Fails fast (not silently) on any
    mismatch."""
    from script.action_transport import data as at_data
    loaded = at_data.load_dataset(cfg.dataset)
    meta_by_name = {m.name: m for m in loaded.metas}
    missing = sorted(set(extracted) - set(meta_by_name))
    if missing:
        raise SystemExit(f"{len(missing)} recording(s) in {cfg.dataset}'s features/ do not appear in "
                         f"data.py's loaded dataset (e.g. {missing[:3]}) -- name mismatch between the raw "
                         "feature files and the released-checkpoint latent cache")
    for name, rec in extracted.items():
        expected = int(sum(rec["lengths"]))
        actual = meta_by_name[name].frame_count
        if expected != actual:
            raise SystemExit(f"{name}: this pilot's extraction saw {expected} real frames but data.py's "
                             f"loaded dataset has frame_count {actual} -- window grids are not aligned")
    return dict(loaded=loaded, meta_by_name=meta_by_name,
               gts_by_name={m.name: loaded.core_dataset.gts[i] for i, m in enumerate(loaded.metas)
                           if m.name in extracted})


def build_streams(extracted: dict[str, dict[str, dict]], vocabs: dict, cfg: PilotConfig) -> dict[str, dict]:
    streams = {}
    for arm in ("vq", "fsq"):
        native_ids = {name: rec["ids"] for name, rec in extracted[arm].items()}
        native_pca = {name: project_to_pca((vocabs[arm].scaler, vocabs[arm].pca), rec["flat"])
                     for name, rec in extracted[arm].items()}
        kmeans_ids, kmeans_pca = {}, {}
        for name, rec in extracted[arm].items():
            pca_feats, ids = transform_and_assign(vocabs[arm], rec["flat"])
            kmeans_pca[name], kmeans_ids[name] = pca_feats, ids
        lengths_by_name = {name: rec["lengths"] for name, rec in extracted[arm].items()}
        streams[f"{arm}_native"] = dict(arm=arm, kind="native", ids=native_ids, pca=native_pca,
                                        lengths=lengths_by_name)
        streams[f"{arm}_kmeans"] = dict(arm=arm, kind="kmeans", ids=kmeans_ids, pca=kmeans_pca,
                                        lengths=lengths_by_name)
    return streams


def run_stream_downstream(stream_name: str, stream: dict, cfg: PilotConfig, seed: int,
                          identity: dict, deadline: float | None, artifacts_dir: Path,
                          contextualizer_device: torch.device | None = None) -> dict:
    """Categorical OT + contextual ASOT for one stream at one downstream
    seed. Returns segmentation rows plus saved-artifact payloads.

    `contextualizer_device` routes only the fresh Stage-B contextualizer's
    training onto this device (GPU when available); the OT solves
    (`fit_categorical_k`/`fit_continuous`/both final-solve ladders) stay on
    `evaluate.DEVICE` (CPU) unconditionally, matching the historical
    action_transport convention. Defaults to CPU if not given."""
    from script.fsq_reconstruction.evaluate import (
        DEVICE, categorical_final_ladder, contextual_embeddings, contextual_final_ladder, fit_categorical_k,
        per_code_pca_centers, rung_rows_to_dicts, train_contextualizer,
    )
    d = DATASET_DEFAULTS[cfg.dataset]
    fps = d["fps"]
    C = at_config.NUM_ACTIONS[cfg.dataset]
    names = sorted(stream["ids"])
    meta_by_name, gts_by_name = identity["meta_by_name"], identity["gts_by_name"]
    starts_by_name = {n: meta_by_name[n].starts for n in names}
    frame_counts_by_name = {n: meta_by_name[n].frame_count for n in names}

    recs = [categorical.RecordingInput(n, np.asarray(stream["ids"][n], dtype=np.int32),
                                       np.asarray(stream["lengths"][n], dtype=np.int32), fps, d["patch_size"])
           for n in names]
    log_k = math.log(cfg.num_codes)
    eta = at_config.eta(cfg.dataset)
    counts_all = categorical.frame_weighted_counts(recs, cfg.num_codes)
    centers, counts = per_code_pca_centers(recs, stream["pca"], cfg.num_codes)
    rows: list[dict] = []
    artifacts: dict = {}
    try:
        grouping = categorical.build_grouping(centers, counts, C, seed, is_kc_identity=False)
    except categorical.UnavailableError as exc:
        return dict(rows=[dict(stream=stream_name, downstream_seed=seed, method="categorical_asot",
                              rung_steps=None, status="unavailable", reason=str(exc))], artifacts={})
    theta0 = categorical.build_theta0(grouping, counts_all, C, cfg.num_codes, eta)
    coeffs_fit = transport.Coeffs(at_config.FIT_SETTING.a, at_config.FIT_SETTING.beta, at_config.FIT_SETTING.lam,
                                  at_config.FIT_SETTING.eps)
    fit = fit_categorical_k(recs, theta0, C, cfg.num_codes, eta, coeffs_fit, cfg.fit_inner_steps,
                            cfg.fit_outer_cap, at_config.FIT_OUTER_PATIENCE, at_config.FIT_OUTER_RELTOL,
                            DEVICE, log_k, deadline)
    coeffs_final = transport.Coeffs(at_config.SETTING_T.a, at_config.SETTING_T.beta, at_config.SETTING_T.lam,
                                    at_config.SETTING_T.eps)
    if fit.outer_status != "invalid":
        cat_rungs = categorical_final_ladder(fit.params, recs, C, log_k, coeffs_final, cfg.final_rungs,
                                             gts_by_name, starts_by_name, frame_counts_by_name, deadline)
        rows += rung_rows_to_dicts(stream_name, seed, "categorical_asot", cat_rungs)
        artifacts["categorical_theta"] = fit.params
        artifacts["categorical_grouping"] = grouping
    else:
        rows.append(dict(stream=stream_name, downstream_seed=seed, method="categorical_asot", rung_steps=None,
                        status="invalid", reason="fit hit a nonfinite value or backtracking failure"))

    # contextual ASOT: fresh contextualizer for this stream/seed, then continuous OT on its embeddings
    ctx = train_contextualizer(stream["ids"], stream["lengths"], frame_counts_by_name, cfg.num_codes, seed,
                               split_seed=at_config.SAMPLING_SEED, deadline=deadline,
                               max_steps=cfg.contextualizer_max_steps, eval_every=cfg.contextualizer_eval_every,
                               patience_steps=cfg.contextualizer_patience_steps,
                               device=contextualizer_device or torch.device("cpu"))
    embeddings = contextual_embeddings(ctx.model, stream["ids"], device=contextualizer_device or torch.device("cpu"))
    fps_of, window_of = fps, d["patch_size"]
    cont_recs = []
    for n in names:
        unit, zero_stored = embeddings[n]
        z, zero_renorm = continuous.l2_normalize(unit.astype(np.float64))
        cont_recs.append(continuous.ContinuousRecordingInput(n, np.asarray(stream["ids"][n], dtype=np.int32),
                                                             np.asarray(stream["lengths"][n], dtype=np.int32),
                                                             fps_of, window_of, z, zero_stored | zero_renorm))
    mu0, zero_mean = continuous.build_mu0(grouping, cont_recs, C)
    if zero_mean.any():
        rows.append(dict(stream=stream_name, downstream_seed=seed, method="contextual_asot", rung_steps=None,
                        status="unavailable", reason=f"{int(zero_mean.sum())} state(s) had a zero initial mean"))
    else:
        ctx_fit = continuous.fit_continuous(cont_recs, mu0, C, coeffs_fit, cfg.fit_inner_steps,
                                            cfg.fit_outer_cap, at_config.FIT_OUTER_PATIENCE,
                                            at_config.FIT_OUTER_RELTOL, DEVICE, deadline)
        if ctx_fit.outer_status != "invalid":
            ctx_rungs = contextual_final_ladder(ctx_fit.params, cont_recs, coeffs_final, cfg.final_rungs,
                                               gts_by_name, starts_by_name, frame_counts_by_name, deadline)
            rows += rung_rows_to_dicts(stream_name, seed, "contextual_asot", ctx_rungs)
            artifacts["contextual_mu"] = ctx_fit.params
        else:
            rows.append(dict(stream=stream_name, downstream_seed=seed, method="contextual_asot", rung_steps=None,
                            status="invalid", reason="fit hit a nonfinite value or backtracking failure"))
    return dict(rows=rows, artifacts=artifacts, contextualizer_info=ctx.info)


def run_required_pilot(cfg: PilotConfig, run_dir: Path, device: torch.device,
                       deadline: float | None = None) -> None:
    """§3's required experiment: HuGaDB, one paired autoencoder seed, VQ512
    vs FSQ512, four streams, three downstream seeds, both OT methods."""
    d = DATASET_DEFAULTS[cfg.dataset]
    features_path = ROOT / "data" / cfg.dataset / "features"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)

    manifest = dict(pilot_version=PILOT_VERSION, config=asdict(cfg), config_hash=config_hash(cfg),
                    dataset=cfg.dataset, started_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"), deadline=deadline)
    write_json_atomic(run_dir / "manifest.json", manifest)

    epoch_orders = precompute_epoch_orders(features_path, cfg.epochs, seed=cfg.autoencoder_seed)
    diagnostic_batch = build_diagnostic_batch(features_path, d["num_features"], d["num_joints"], d["num_person"])
    backbone = build_shared_backbone(cfg.smq_kwargs(), seed=cfg.autoencoder_seed)
    write_json_atomic(run_dir / "backbone_init_hash.json", dict(init_hash=backbone.init_hash))

    training_rows, gradient_rows, occupancy_rows, reconstruction_rows = [], [], [], []

    vq_model = build_vq_arm(cfg.smq_kwargs(), backbone, init_seed=cfg.autoencoder_seed + 1)
    vq_log = train_one_arm(vq_model, run_dir / "checkpoints" / "vq", features_path, cfg, cfg.autoencoder_seed,
                           epoch_orders, "vq", diagnostic_batch, device, deadline)
    training_rows += vq_log.epoch_rows
    gradient_rows += vq_log.gradient_rows
    occupancy_rows += vq_log.occupancy_rows
    reconstruction_rows.append(dict(arm="vq", epoch=cfg.epochs,
                                    **evaluate_reconstruction(vq_model, features_path, d["num_features"],
                                                              d["num_joints"], d["num_person"],
                                                              cfg.joint_distance_recons, device)))

    fsq_model = build_fsq_arm(cfg.smq_kwargs(), backbone, init_seed=cfg.autoencoder_seed + 1, levels=cfg.levels)
    fsq_log = train_one_arm(fsq_model, run_dir / "checkpoints" / "fsq", features_path, cfg, cfg.autoencoder_seed,
                            epoch_orders, "fsq", diagnostic_batch, device, deadline)
    training_rows += fsq_log.epoch_rows
    gradient_rows += fsq_log.gradient_rows
    occupancy_rows += fsq_log.occupancy_rows
    reconstruction_rows.append(dict(arm="fsq", epoch=cfg.epochs,
                                    **evaluate_reconstruction(fsq_model, features_path, d["num_features"],
                                                              d["num_joints"], d["num_person"],
                                                              cfg.joint_distance_recons, device)))

    write_csv(run_dir / "training.csv", training_rows)
    write_csv(run_dir / "gradient_diagnostics.csv", gradient_rows)
    write_csv(run_dir / "occupancy.csv", occupancy_rows)
    write_csv(run_dir / "reconstruction.csv", reconstruction_rows)

    for row in occupancy_rows:
        if row["kind"] == "vq" and row["below_threshold"] / cfg.num_codes > cfg.replacement_warn_fraction:
            print(f"[WARN] vq epoch {row['epoch']}: {row['below_threshold']}/{cfg.num_codes} codes below "
                 "the dead-code threshold", flush=True)
        if row["kind"] == "fsq" and row["saturation_frac"] is not None and row["saturation_frac"] > 0.5:
            print(f"[WARN] fsq epoch {row['epoch']}: {row['saturation_frac']:.2f} of the diagnostic batch's "
                 "pre-quantization activations are saturating the FSQ bound; the reconstruction gradient into "
                 "the encoder is likely near zero and the codebook may collapse", flush=True)

    extracted = {
        "vq": extract_dataset(vq_model, features_path, d["patch_size"], d["num_features"], d["num_joints"],
                              d["num_person"], device),
        "fsq": extract_dataset(fsq_model, features_path, d["patch_size"], d["num_features"], d["num_joints"],
                               d["num_person"], device),
    }
    identity = check_identity_against_released_meta(cfg, extracted["vq"])
    vocabs = {"vq": fit_pca_kmeans(extracted["vq"], cfg), "fsq": fit_pca_kmeans(extracted["fsq"], cfg)}
    for arm, vocab in vocabs.items():
        write_json_atomic(run_dir / f"kmeans_{arm}_meta.json",
                          dict(n_fit_patches=vocab.n_fit_patches, n_occupied=vocab.n_occupied,
                              converged=bool(vocab.kmeans.n_iter_ < vocab.kmeans.max_iter)))
    streams = build_streams(extracted, vocabs, cfg)

    expected_rows_per_job = 2 * len(cfg.final_rungs)  # categorical_asot + contextual_asot, one row per rung
    segmentation_rows: list[dict] = []
    done_job_ids: set[str] = set()
    seg_path = run_dir / "segmentation.csv"
    if seg_path.exists():
        import csv
        with seg_path.open(newline="", encoding="utf-8") as fh:
            existing_rows = list(csv.DictReader(fh))
        counts: dict[str, int] = {}
        for r in existing_rows:
            jid = f"{r['stream']}_s{r['downstream_seed']}"
            counts[jid] = counts.get(jid, 0) + 1
        done_job_ids = {jid for jid, n in counts.items() if n >= expected_rows_per_job}
        segmentation_rows = [r for r in existing_rows if f"{r['stream']}_s{r['downstream_seed']}" in done_job_ids]
        if done_job_ids:
            print(f"[resume] {len(done_job_ids)} downstream job(s) already complete in {seg_path.name}, "
                 f"skipping: {sorted(done_job_ids)}", flush=True)

    not_run: list[dict] = []
    for stream_name, stream in streams.items():
        for seed in cfg.downstream_seeds:
            job_id = f"{stream_name}_s{seed}"
            if job_id in done_job_ids:
                continue
            if deadline is not None and time.perf_counter() > deadline:
                not_run.append(dict(job_id=job_id, reason="compute deadline reached"))
                continue
            print(f"[downstream] {job_id} ...", flush=True)
            try:
                result = run_stream_downstream(stream_name, stream, cfg, seed, identity, deadline,
                                               run_dir / "artifacts", contextualizer_device=device)
            except DeadlineExceeded:
                not_run.append(dict(job_id=job_id, reason="interrupted at a cooperative deadline check "
                                    "mid-job (no partial rows saved for this job)"))
                continue
            segmentation_rows += result["rows"]
            np.savez_compressed(run_dir / "artifacts" / f"{job_id}.npz", **result.get("artifacts", {}))
            write_csv(run_dir / "segmentation.csv", segmentation_rows)  # incremental save
    write_csv(run_dir / "not_run.csv", not_run)
    print(f"Required HuGaDB pilot (seed {cfg.autoencoder_seed}): {len(segmentation_rows)} segmentation rows, "
         f"{len(not_run)} downstream jobs not run.", flush=True)


def smoke_config() -> PilotConfig:
    """-Smoke: a tiny model (filters=16, 1 layer), 2 epochs, one downstream
    seed, a 32-code alphabet, and a short contextualizer/OT budget. Runs
    against the real dataset (all recordings) but every other cost knob is
    shrunk so the whole pipeline finishes in minutes, not hours -- this
    checks that every stage runs end to end, not that any stage has
    converged."""
    return PilotConfig(epochs=2, checkpoint_epochs=(0, 1, 2), downstream_seeds=(111,),
                      kmeans_sample_cap=200, kmeans_k=32, levels=(2, 4, 4),
                      filters=16, num_layers=1, latent_dim=4,
                      contextualizer_max_steps=20, contextualizer_eval_every=5, contextualizer_patience_steps=20,
                      final_rungs=(5, 10), fit_outer_cap=2, fit_inner_steps=3)


def cmd_validate_only() -> None:
    from script.fsq_reconstruction import tests
    tests.run_all()


def cmd_run(args) -> None:
    cfg = smoke_config() if args.smoke else PilotConfig(
        dataset=args.dataset, autoencoder_seed=args.autoencoder_seed,
        downstream_seeds=tuple(args.downstream_seeds), epochs=args.epochs)
    run_dir = OUT_ROOT / args.run_id
    if (run_dir / "manifest.json").exists() and not args.resume:
        raise SystemExit(f"{run_dir}/manifest.json already exists; pass --resume or choose a new --run-id")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    deadline = time.perf_counter() + args.hours * 3600.0 - 3 * 3600.0 if args.hours else None
    run_required_pilot(cfg, run_dir, device, deadline)


def build_parser():
    import argparse
    parser = argparse.ArgumentParser(description="FSQ reconstruction pilot")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--dataset", default="hugadb", choices=list(DATASET_DEFAULTS))
    parser.add_argument("--run-id", dest="run_id", default=None)
    parser.add_argument("--hours", type=float, default=None)
    parser.add_argument("--autoencoder-seed", dest="autoencoder_seed", type=int, default=111)
    parser.add_argument("--downstream-seeds", dest="downstream_seeds", type=int, nargs="+",
                        default=[111, 222, 1538574472])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.validate_only:
        return cmd_validate_only()
    if not args.run_id:
        raise SystemExit("--run-id is required for --smoke and for a full run")
    cmd_run(args)


if __name__ == "__main__":
    main()
