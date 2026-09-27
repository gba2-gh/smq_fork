"""Stage C: contextual temporal transport on frozen B-code embeddings
(docs/plans/THREE_STAGE_MOTION_PLAN.md §5; docs/plans/STAGE_C_AGENT_INSTRUCTIONS.md
Part 2). Reuses Stage A's continuous-arm machinery (transport, cosine cost,
prototype updates) with `z` swapped from PCA-64 windows to the frozen
`stage_b_003` contextualizer's 128-d embeddings. Fits no contextualizer;
trains nothing. Never imports `evaluate`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.action_transport import categorical, config, continuous, permute, transport
from script.action_transport.run_stage_b import OUT_ROOT as STAGEB_OUT_ROOT
from script.action_transport.run_stage_b import cell_key as stageb_cell_key
from script.action_transport.run_stage_b import codes_sha256
from script.action_transport.stageb_configs import SELECTED_CONFIG_HASH
from script.action_transport.stageb_infer import extract_embeddings
from script.action_transport.stageb_train import load_checkpoint

STAGEC_VERSION = "1.0"


class UnavailableError(RuntimeError):
    """A cell is unavailable per a declared, label-free rule; never worked around."""


class IdentityError(RuntimeError):
    """The stage_b_003 / stage_a identity chain (STAGE_C_AGENT_INSTRUCTIONS.md
    §2.2) failed a check; the cell is refused rather than silently scored."""


# --- stage_b_003 identity and embedding loading --------------------------------------

def verify_stage_b_manifest(stage_b_dir: Path) -> dict:
    """§2.2.1: the run must be trained under the frozen selected hash."""
    import json
    manifest_path = stage_b_dir / "manifest.json"
    if not manifest_path.exists():
        raise IdentityError(f"{manifest_path} does not exist")
    manifest = json.loads(manifest_path.read_text())
    got = manifest.get("config_hash")
    if got != SELECTED_CONFIG_HASH:
        raise IdentityError(f"{stage_b_dir} has config_hash {got!r}, expected the frozen selected hash "
                            f"{SELECTED_CONFIG_HASH!r} -- this is not the stage_b_003 identity Stage C requires")
    return manifest


@dataclass
class StageBCell:
    dataset: str
    normalize: bool
    seed: int
    index: dict  # embeddings/<cell>/INDEX.json
    embeddings_dir: Path
    checkpoint_path: Path


def load_stage_b_cell(stage_b_run: str, dataset: str, normalize: bool, seed: int) -> StageBCell:
    """§2.2.2-2.2.4: checkpoint hash, INDEX.json consistency, embedding shape."""
    import json
    run_dir = STAGEB_OUT_ROOT / stage_b_run
    key = stageb_cell_key(dataset, normalize, seed)
    embeddings_dir = run_dir / "embeddings" / key
    checkpoint_path = run_dir / "checkpoints" / f"{key}.pt"
    index_path = embeddings_dir / "INDEX.json"
    if not index_path.exists():
        raise IdentityError(f"{index_path} does not exist (was {stage_b_run} trained for this cell?)")
    if not checkpoint_path.exists():
        raise IdentityError(f"{checkpoint_path} does not exist")
    index = json.loads(index_path.read_text())
    if index.get("config_hash") != SELECTED_CONFIG_HASH:
        raise IdentityError(f"{index_path} has config_hash {index.get('config_hash')!r}, expected "
                            f"{SELECTED_CONFIG_HASH!r}")
    import hashlib
    actual_ckpt_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if index.get("checkpoint_sha256") != actual_ckpt_hash:
        raise IdentityError(f"{checkpoint_path} sha256 does not match {index_path}'s recorded value "
                            "-- the checkpoint file was replaced after the embeddings were extracted")
    for name, length in index["lengths"].items():
        emb_path = embeddings_dir / f"{name}.npz"
        if not emb_path.exists():
            raise IdentityError(f"{emb_path} listed in INDEX.json but missing")
        blob = np.load(emb_path)
        if blob["embedding"].shape != (length, 128):
            raise IdentityError(f"{emb_path} has shape {blob['embedding'].shape}, expected ({length}, 128)")
    return StageBCell(dataset, normalize, seed, index, embeddings_dir, checkpoint_path)


def load_embeddings(cell: StageBCell) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """name -> (unit embedding [L,128] float32, zero_flags [L] bool)."""
    out = {}
    for name in cell.index["recording_order"]:
        blob = np.load(cell.embeddings_dir / f"{name}.npz")
        out[name] = (blob["embedding"], blob["zero_flags"])
    return out


def verify_tokenizer_continuity(loaded, tok, dataset: str, seed: int, cell: StageBCell) -> None:
    """§2.2.2-2.2.3: the tokenizer this Stage C process rebuilt must be the
    exact one stage_b_003's embeddings and codes_sha256 were produced from."""
    if loaded.core_dataset.cache_identity != cell.index.get("cache_identity"):
        raise IdentityError(f"{dataset} seed{seed}: current cache_identity "
                            f"{loaded.core_dataset.cache_identity!r} != stage_b_003's recorded "
                            f"{cell.index.get('cache_identity')!r}")
    all_names = [m.name for m in loaded.metas]
    codes_by_name = {loaded.metas[i].name: np.asarray(tok.codes_fine[i], dtype=np.int64)
                     for i in range(len(loaded.metas))}
    current_hash = codes_sha256(all_names, codes_by_name)
    if current_hash != cell.index.get("codes_sha256"):
        raise IdentityError(f"{dataset} seed{seed}: current codes_sha256 {current_hash} != stage_b_003's "
                            f"recorded {cell.index.get('codes_sha256')}")
    if sorted(cell.index["recording_order"]) != sorted(all_names):
        raise IdentityError(f"{dataset} seed{seed}: recording lists differ between the current dataset and "
                            "stage_b_003's INDEX.json")


def verify_stage_a_grouping(loaded, tok, dataset: str, normalize: bool, seed: int, stage_a_run: str,
                            C: int) -> np.ndarray:
    """§2.2: the grouping g used to initialize Stage C's prototypes must be
    the identical grouping Stage A's A-cont used, for paired initialization
    (docs/plans/THREE_STAGE_MOTION_PLAN.md §5). Recomputes g from the current
    tokenizer cache and checks it against the artifact stage_a_pooled_002
    actually saved, rather than trusting that rebuilt caches still match."""
    from script.action_transport import run as stagea_run  # categorical_inputs/continuous_inputs helpers
    fit_idx = tuple(range(len(loaded.metas)))
    counts = categorical.frame_weighted_counts(
        stagea_run.categorical_inputs(loaded, tok.codes_fine, fit_idx), config.K_FINE)
    try:
        g = categorical.build_grouping(tok.centers_fine, counts, C, seed, is_kc_identity=False)
    except categorical.UnavailableError as exc:
        raise UnavailableError(str(exc)) from exc
    cell_id = config.CellId(dataset, normalize, seed, "pooled", None, "continuous_asot")
    artifact_path = config.OUT_ROOT / stage_a_run / "artifacts" / f"{cell_id.key()}.npz"
    if not artifact_path.exists():
        raise IdentityError(f"{artifact_path} does not exist -- Stage A's A-cont must have run for this cell")
    artifact = np.load(artifact_path)
    if not np.array_equal(artifact["grouping"], g):
        raise IdentityError(f"{dataset}/{('norm' if normalize else 'raw')} seed{seed}: the grouping recomputed "
                            f"from the current tokenizer cache no longer matches {artifact_path}'s saved "
                            "grouping -- Stage A's paired initialization would not actually be paired")
    return g


# --- context inputs and cost ----------------------------------------------------------

def context_inputs(loaded, tok, embeddings: dict[str, tuple[np.ndarray, np.ndarray]],
                   indices) -> list[continuous.ContinuousRecordingInput]:
    """Build ContinuousRecordingInput per recording with z = the re-L2-
    normalized 128-d contextual embedding (STAGE_C_AGENT_INSTRUCTIONS.md
    §2.2: float64 re-normalization; zero flags are the logical OR of the
    stored flag and the re-normalization flag)."""
    fps_of, window_of = config.FPS[loaded.dataset], config.WINDOW[loaded.dataset]
    out = []
    for i in indices:
        meta = loaded.metas[i]
        unit, zero_stored = embeddings[meta.name]
        z, zero_renorm = continuous.l2_normalize(unit.astype(np.float64))
        zero = zero_stored | zero_renorm
        out.append(continuous.ContinuousRecordingInput(
            meta.name, np.asarray(tok.codes_fine[i], dtype=np.int32), meta.lengths, fps_of, window_of, z, zero))
    return out


def coeffs_for(setting: config.CoeffSetting, arm: str) -> transport.Coeffs:
    beta = 0.0 if arm == "context_only" else setting.beta
    return transport.Coeffs(setting.a, beta, setting.lam, setting.eps)


# --- the order-destruction diagnostic (§2.5) -------------------------------------------

def permuted_context_embeddings(stage_b_cell: StageBCell, loaded, tok, dataset: str, seed: int,
                                device: torch.device) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """For each recording: build the Stage A permutation convention, feed
    permuted codes through the FROZEN stage_b_003 checkpoint, then restore
    embedding order. Returns name -> (Z_restored [L,128] float32, zero_flags,
    perm)."""
    model, _, _ = load_checkpoint(stage_b_cell.checkpoint_path, device)
    rng = np.random.default_rng(seed + config.PERMUTATION_SEED_OFFSET)
    window = config.WINDOW[dataset]
    out = {}
    name_to_idx = {m.name: i for i, m in enumerate(loaded.metas)}
    for meta in sorted(loaded.metas, key=lambda m: m.name):
        i = name_to_idx[meta.name]
        codes = np.asarray(tok.codes_fine[i], dtype=np.int32)
        perm = permute.build_permutation(meta.lengths, window, rng)
        codes_perm = permute.apply_permutation(codes, perm)
        unit_perm, zero_perm = extract_embeddings(model, codes_perm.astype(np.int64), device)
        z_restored = permute.restore_states(unit_perm, perm)
        zero_restored = permute.restore_states(zero_perm, perm)
        out[meta.name] = (z_restored, zero_restored, perm)
    return out


def check_permutation_matches_stage_a(perms: dict[str, np.ndarray], stage_a_run: str, dataset: str,
                                      normalize: bool, seed: int) -> None:
    """The permutation used here must equal the one Stage A's `permuted_asot`
    diagnostic saved for the same (dataset, normalize, seed) -- same codes,
    same RNG convention, so a mismatch is an implementation error, not a
    data difference."""
    cell_id = config.CellId(dataset, normalize, seed, "pooled", None, "permuted_asot")
    path = config.OUT_ROOT / stage_a_run / "artifacts" / f"{cell_id.key()}.npz"
    if not path.exists():
        return  # not fatal: only a cross-check when Stage A's artifact is present
    blob = np.load(path, allow_pickle=True)
    saved = {str(n): p for n, p in zip(blob["perm_names"], blob["perm"])}
    for name, perm in perms.items():
        if name in saved and not np.array_equal(saved[name], perm):
            raise IdentityError(f"{name}: Stage C's permutation differs from Stage A's saved permuted_asot "
                                "permutation for the same seed -- the permutation convention has diverged")
