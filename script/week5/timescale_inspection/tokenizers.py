"""Phase 5 preparation: alternative tokenizers for one (dataset, tokenizer, seed, window) cell.

Mirrors `raw_kmeans_cross_dataset.prepare` step for step (same sampling, standardize -> randomized PCA ->
K-means-512, split, grouping, occupancy), changing only the per-window input (`source`), the PCA size and
whether the PCA output is whitened. The result is written in the unchanged `quantized_input_comparison`
artifact layout, under a family name of its own (`tok5_<key>`), so nothing registered in the shared config is
touched and every earlier run keeps its identity.

Sources (all label-free):
  raw            the flattened raw patch -- exactly `raw_kmeans_cross_dataset.prepare.extract_raw` (T0, T1)
  raw_positions  LARa: position channels only (3-5), orientations dropped (T3, T3b)
  relational     per-window inter-joint distances: mean, sd and mean |delta| over the window's own frames
                 (position channels only; BABEL 25 joints, LARa 22) (T2)
  smq_latent     SMQ's released encoder, pre-quantization per-frame latents (16 per joint), patched at our
                 window (T4)

With the T0 spec (`raw`, PCA-64, not whitened) the whole path reduces to `raw_kmeans_cross_dataset.prepare`;
gate G1 checks that against the saved `rkx_v2` artifacts.
"""

from __future__ import annotations

import dataclasses
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from script.fsq_reconstruction.experiment import write_json_atomic
from script.quantized_input_comparison import artifacts
from script.quantized_input_comparison import fit as fitmod
from script.quantized_input_comparison.families import kmeans_lookup, window_starts
from script.week5.raw_kmeans_cross_dataset import datasets as D
from script.week5.raw_kmeans_cross_dataset import prepare as PREP

ROOT = D.ROOT
SCHEMA = 1
PREP_THREADS = PREP.PREP_THREADS  # matched to rkx_v2 (thread count changes the 6th decimal of PCA / K-means)

LARA_POSITION_CHANNELS = (3, 4, 5)  # channels 0-2 are orientations; gate G3 asserts this per run


@dataclass(frozen=True)
class TokenizerSpec:
    key: str
    label: str
    source: str
    pca_dim: int
    whiten: bool
    datasets: tuple[str, ...]
    guard: int = 0  # contextualizer guard radius (frames) used for this tokenizer's `context` arm
    span: int = 8  # contextualizer mask span (tokens)
    selectable: bool = True  # may be chosen for the span / window stages

    @property
    def family(self) -> str:
        return f"tok5_{self.key}"


SPECS: dict[str, TokenizerSpec] = {s.key: s for s in (
    TokenizerSpec("t0_raw", "T0", "raw", 64, False, ("babel2", "babel3", "hugadb", "lara")),
    TokenizerSpec("t1_whiten", "T1", "raw", 64, True, ("babel2", "babel3", "hugadb")),
    TokenizerSpec("t2_relational", "T2", "relational", 64, False, ("babel2", "babel3", "lara")),
    TokenizerSpec("t3_lara_pos256", "T3", "raw_positions", 256, False, ("lara",), selectable=False),
    TokenizerSpec("t3b_lara_pos64", "T3b", "raw_positions", 64, False, ("lara",), selectable=False),
    # SMQ's encoder sees +-14 frames: its `context` arm needs guard 14, which in turn needs span 16 to score
    # a usable share of targets (see PHASE5_TOKENIZER.md). Never selectable: spans 2 and 4 score under 5%.
    TokenizerSpec("t4_smq", "T4", "smq_latent", 64, False, ("babel2", "babel3", "hugadb", "lara"), guard=14,
                  span=16, selectable=False),
)}
SELECTABLE = tuple(k for k, s in SPECS.items() if s.selectable)

SMQ_DEFAULTS = {  # `script.repro.dump_predictions.DEFAULTS`
    "hugadb": dict(num_features=6, num_joints=6, num_person=1),
    "lara": dict(num_features=6, num_joints=22, num_person=1),
    "babel1": dict(num_features=3, num_joints=25, num_person=1),
    "babel2": dict(num_features=3, num_joints=25, num_person=1),
    "babel3": dict(num_features=3, num_joints=25, num_person=1),
}


@dataclass(frozen=True)
class PrepKey:
    dataset: str
    tokenizer: str
    source: str
    pca_dim: int
    whiten: bool
    seed: int
    window: int
    num_states: int
    num_codes: int = 512
    sample_cap: int = 10_000
    split_seed: int = 111
    recording_cap: int | None = None
    threads: int = PREP_THREADS


def dataset_spec(dataset: str, window: int | None = None) -> D.DatasetSpec:
    spec = D.SPECS[dataset]
    return spec if window in (None, spec.window) else dataclasses.replace(spec, window=int(window))


def make_key(dataset: str, tokenizer: str, seed: int, window: int | None = None,
             recording_cap: int | None = None, num_codes: int = 512) -> PrepKey:
    tok, spec = SPECS[tokenizer], dataset_spec(dataset, window)
    if dataset not in tok.datasets:
        raise ValueError(f"tokenizer {tokenizer} is not defined for {dataset}")
    return PrepKey(dataset, tokenizer, tok.source, tok.pca_dim, tok.whiten, seed, spec.window, spec.num_states,
                   num_codes=num_codes, recording_cap=recording_cap)


# -- inputs -------------------------------------------------------------------------------------------------------

def position_channels(dataset: str) -> list[int] | None:
    """Channel indices holding joint positions (None: every channel is a position, as in BABEL)."""
    if dataset == "lara":
        return list(LARA_POSITION_CHANNELS)
    if dataset.startswith("babel"):
        return None
    raise ValueError(f"{dataset} has no skeleton position channels")


def patch_frames(frames_np: np.ndarray, window: int) -> dict:
    """Frames (T, d) -> flattened windows on the zero-padded grid, as `extract_raw`."""
    T = frames_np.shape[0]
    frames = torch.as_tensor(frames_np, dtype=torch.float32)
    num_patches = int(np.ceil(T / window))
    pad = num_patches * window - T
    if pad:
        frames = F.pad(frames, (0, 0, 0, pad))
    lengths = np.full(num_patches, window, dtype=np.int32)
    if pad:
        lengths[-1] = window - pad
    return dict(flat=frames.reshape(num_patches, -1).numpy(), lengths=lengths, num_patches=num_patches)


def extract_positions_only(spec: D.DatasetSpec, names: list[str]) -> dict[str, dict]:
    """LARa position channels only, flattened exactly like `extract_raw` but with channels 3-5."""
    ch = position_channels(spec.name)
    out = {}
    for name in names:
        arr = np.load(D.features_dir(spec) / name)  # (C, T, V, M)
        if ch is not None:
            arr = arr[ch]
        T = arr.shape[1]
        out[name] = patch_frames(arr.transpose(1, 0, 2, 3).reshape(T, -1), spec.window)
    return out


def frame_distances(pos: np.ndarray) -> np.ndarray:
    """(T, V, 3) positions -> (T, V*(V-1)/2) pairwise Euclidean distances (rotation- and translation-invariant)."""
    iu = np.triu_indices(pos.shape[1], 1)
    diff = pos[:, iu[0], :] - pos[:, iu[1], :]
    return np.sqrt((diff.astype(np.float64) ** 2).sum(-1)).astype(np.float32)


def window_stats(d: np.ndarray, window: int) -> np.ndarray:
    """(T, J) per-frame features -> (P, 3J): mean, sd and mean |frame-to-frame change| inside each window.
    Only the window's own real frames are used (never the previous window's), so a masked token's feature
    never depends on its neighbour. A terminal partial window uses its real frames."""
    T, J = d.shape
    P = int(np.ceil(T / window))
    out = np.zeros((P, 3 * J), dtype=np.float32)

    def stats(x: np.ndarray) -> np.ndarray:  # (n, w, J) -> (n, 3J)
        change = np.abs(np.diff(x, axis=1)).mean(1) if x.shape[1] > 1 else np.zeros((x.shape[0], J), x.dtype)
        return np.concatenate([x.mean(1), x.std(1), change], axis=1)

    full = T // window
    if full:
        out[:full] = stats(d[:full * window].reshape(full, window, J))
    if full < P:
        out[full] = stats(d[full * window:][None])[0]
    return out


def extract_relational(spec: D.DatasetSpec, names: list[str]) -> dict[str, dict]:
    ch = position_channels(spec.name)
    out = {}
    for name in names:
        arr = np.load(D.features_dir(spec) / name)  # (C, T, V, M)
        if arr.shape[3] != 1:
            raise ValueError("relational features assume one person")
        if ch is not None:
            arr = arr[ch]
        pos = arr[..., 0].transpose(1, 2, 0)  # (T, V, 3)
        if pos.shape[2] != 3:
            raise ValueError(f"{spec.name}: expected 3 position channels, found {pos.shape[2]}")
        T = pos.shape[0]
        num_patches = int(np.ceil(T / spec.window))
        lengths = np.full(num_patches, spec.window, dtype=np.int32)
        pad = num_patches * spec.window - T
        if pad:
            lengths[-1] = spec.window - pad
        out[name] = dict(flat=window_stats(frame_distances(pos.astype(np.float32)), spec.window), lengths=lengths,
                         num_patches=num_patches)
    return out


def load_smq(dataset: str, device: torch.device):
    """The released SMQ model, loaded verbatim (strict) from `models/pretrained/{dataset}.model`, in eval mode."""
    from script.repro.dump_predictions import build_model
    model, num_actions, patch = build_model(dataset, ROOT / "data" / dataset / "groundTruth")
    state = torch.load(ROOT / "models" / "pretrained" / f"{dataset}.model", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, state


def extract_smq(spec: D.DatasetSpec, names: list[str], device: torch.device) -> dict[str, dict]:
    from script.fsq_reconstruction.experiment import extract_recording
    model, _ = load_smq(spec.name, device)
    cfg = SMQ_DEFAULTS[spec.name]
    out = {}
    for name in names:
        rec = extract_recording(model, D.features_dir(spec) / name, spec.window, cfg["num_features"],
                                cfg["num_joints"], cfg["num_person"], device)
        out[name] = dict(flat=rec["flat"], lengths=rec["lengths"], num_patches=rec["num_patches"])
    del model
    return out


def extract(spec: D.DatasetSpec, source: str, names: list[str], device: torch.device | None = None) -> dict[str, dict]:
    if source == "raw":
        return PREP.extract_raw(spec, names, "global")
    if source == "raw_positions":
        return extract_positions_only(spec, names)
    if source == "relational":
        return extract_relational(spec, names)
    if source == "smq_latent":
        return extract_smq(spec, names, device or torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    raise ValueError(source)


# -- population fit -----------------------------------------------------------------------------------------------

def fit_population(name: str, extracted_pop: dict[str, dict], sample_ids: list[tuple[str, int]], pca_dim: int,
                   num_codes: int, seed: int, whiten: bool = False) -> fitmod.FittedPopulation:
    """A local copy of `fitmod.fit_population` with PCA whitening switchable. With `whiten=False` it is the same
    arithmetic (tested equal). With `whiten=True` K-means runs in the whitened space, and `project_pca` (which calls
    `pca.transform`) whitens automatically."""
    sample = fitmod.gather(extracted_pop, sample_ids)
    if len(sample) < num_codes:
        raise RuntimeError(f"population {name!r}: only {len(sample)} sampled complete windows, "
                           f"need >= {num_codes} for K-means (K={num_codes})")
    scaler = StandardScaler().fit(sample)
    sample_std = scaler.transform(sample)
    n_components = min(pca_dim, sample_std.shape[0], sample_std.shape[1])
    pca = PCA(n_components=n_components, whiten=whiten, svd_solver="randomized", random_state=seed)
    sample_pca = pca.fit_transform(sample_std)
    kmeans = KMeans(n_clusters=num_codes, init="k-means++", n_init=5, max_iter=300, tol=1e-4, algorithm="lloyd",
                    random_state=seed)
    kmeans.fit(sample_pca)
    occupied = len(set(kmeans.labels_.tolist()))
    return fitmod.FittedPopulation(name, scaler, pca, kmeans, sample_ids, len(sample), occupied)


# -- cell preparation ---------------------------------------------------------------------------------------------

def manifest_ok(run_dir: Path, key: PrepKey, names: list[str], counts: dict[str, int]) -> bool:
    if not (Path(run_dir) / "manifest.json").exists():
        return False
    family = SPECS[key.tokenizer].family
    try:
        m = artifacts.read_manifest(run_dir)
        return (m.get("schema") == SCHEMA and m.get("prep_key") == asdict(key) and
                sorted(m["recording_names"]) == sorted(names) and
                {k: int(v) for k, v in m["recording_frame_counts"].items()} == counts and
                artifacts.family_artifacts_complete(run_dir, family, names, key.num_codes) and
                (Path(run_dir) / "split.json").exists())
    except (OSError, KeyError, ValueError):
        return False


def is_prepared(dataset: str, tokenizer: str, seed: int, run_dir: Path, window: int | None = None,
                recording_cap: int | None = None) -> bool:
    key = make_key(dataset, tokenizer, seed, window, recording_cap)
    spec = dataset_spec(dataset, window)
    names = D.recording_names(spec, recording_cap)
    return manifest_ok(Path(run_dir), key, names, D.frame_counts(spec, names))


def prepare_cell(dataset: str, tokenizer: str, seed: int, run_dir: Path, window: int | None = None,
                 recording_cap: int | None = None, device: torch.device | None = None) -> Path:
    key = make_key(dataset, tokenizer, seed, window, recording_cap)
    spec = dataset_spec(dataset, window)
    names = D.recording_names(spec, recording_cap)
    counts = D.frame_counts(spec, names)
    run_dir = Path(run_dir)
    if manifest_ok(run_dir, key, names, counts):
        return run_dir
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    with threadpool_limits(limits=key.threads):
        _build(spec, key, run_dir, names, counts, device)
    return run_dir


def _build(spec: D.DatasetSpec, key: PrepKey, run_dir: Path, names: list[str], counts: dict[str, int],
           device: torch.device | None) -> None:
    tok = SPECS[key.tokenizer]
    family = tok.family

    # raw feature extraction
    raw = extract(spec, tok.source, names, device)


    frame_counts = {n: int(sum(raw[n]["lengths"])) for n in names}
    if frame_counts != counts:
        raise RuntimeError("windowed frame counts disagree with the feature files' own shapes")
    lengths = fitmod.window_lengths(raw)
    all_ids = fitmod.complete_window_ids(lengths, spec.window)
    sample_ids = fitmod.sample_window_ids(all_ids, key.sample_cap, key.seed)
    artifacts.save_sample_ids(run_dir, sample_ids)
    train_names, val_names = fitmod.shared_validation_split(frame_counts, key.split_seed)
    artifacts.save_split(run_dir, train_names, val_names, frame_counts)
    train_set = set(train_names)
    train_complete = [(n, p) for n, p in all_ids if n in train_set]

    #population fitting
    pop = fit_population(family, raw, sample_ids, key.pca_dim, key.num_codes, key.seed, whiten=key.whiten)


    artifacts.save_population(run_dir, pop)
    pca_family = {n: fitmod.project_pca(pop, rec["flat"]) for n, rec in raw.items()}
    records = {}
    for name, rec in raw.items():
        pca_feats, ids = fitmod.assign_kmeans(pop, rec["flat"])
        records[name] = dict(codes=ids, continuous=pca_feats, starts=window_starts(rec["num_patches"], spec.window),
                             lengths=rec["lengths"].astype(np.int32))
    lookup = kmeans_lookup(pop)
    stats = fitmod.conditioning_stats({n: r["continuous"] for n, r in records.items()}, sample_ids, train_complete)
    codes_by_name = {n: r["codes"] for n, r in records.items()}
    lengths_by_name = {n: r["lengths"] for n, r in records.items()}
    grouping, _, counts_w = fitmod.initial_grouping(codes_by_name, lengths_by_name, pca_family, key.num_codes,
                                                    spec.num_states, spec.fps, spec.window, key.seed)
    window_counts = np.zeros(key.num_codes, dtype=np.int64)
    for codes in codes_by_name.values():
        np.add.at(window_counts, codes, 1)
    artifacts.save_family(run_dir, family, records, lookup, stats["mean"], stats["rms"], grouping, counts_w,
                          window_counts)
    write_json_atomic(run_dir / "manifest.json", dict(
        schema=SCHEMA, prep_key=asdict(key), families=[family], recording_names=names,
        recording_frame_counts=frame_counts, num_windows={family: {n: int(len(r["codes"])) for n, r in records.items()}},
        feature_dims={family: int(lookup.shape[1])}, sample_ids_hash=artifacts.sample_ids_hash(sample_ids),
        n_train_recordings=len(train_names), n_val_recordings=len(val_names),
        kmeans_occupied=int(pop.n_occupied), n_fit_windows=int(pop.n_fit_windows)))
