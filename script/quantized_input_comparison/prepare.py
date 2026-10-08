"""Task 1 entrypoint: preflight tests, a bounded real-data smoke test, full
preparation, and resume.

    python -m script.quantized_input_comparison.prepare --preflight
    python -m script.quantized_input_comparison.prepare --smoke --run-id qic_smoke
    python -m script.quantized_input_comparison.prepare --full --run-id qic_v1
    python -m script.quantized_input_comparison.prepare --full --run-id qic_v1 --resume

Resume safety (see artifacts.py's module docstring for the full contract):
  - Reuse of ANY preprocessing/family artifact is gated on a verified,
    matching manifest.json already existing (schema version, config,
    checkpoint hashes, recording names AND per-recording frame counts all
    match). `can_reuse` below is that single gate; nothing downstream reuses
    a file because `--resume` was passed, only because `can_reuse` is True.
  - `--resume` against a run_dir with partial artifacts but NO manifest
    (e.g. a prior run crashed mid-way) is refused outright, never silently
    treated as reusable.
  - Even when `can_reuse` is True, a family's artifacts are only skipped
    (never regenerated) when `artifacts.family_artifacts_complete` confirms
    they are structurally complete and valid for the current recording set;
    per-population preprocessing (scaler/PCA/K-means) is reloaded via real
    sklearn objects (not a hand-rolled affine map) so a resumed run's
    transform arithmetic is identical to a fresh run's, not a ~1e-5-drifted
    approximation of it.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from script.action_transport import config as at_config
from script.fsq_reconstruction.experiment import DATASET_DEFAULTS
from script.quantized_input_comparison import artifacts, config as cfg, extract, families, fit as fitmod

POP_FAMILIES = {
    "raw": ("raw_kmeans",),
    "vq_encoder": ("vq_native", "vq_kmeans"),
    "fsq_encoder": ("fsq_native", "fsq_kmeans"),
}


def run_preparation(prep_cfg: cfg.PrepConfig, device: torch.device, resume: bool = False) -> Path:
    run_dir = prep_cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    source_cfg = extract.load_source_pilot_config()

    loaded = extract.load_models_and_names(source_cfg, device, recording_cap=prep_cfg.recording_cap)
    vq_model, fsq_model = loaded["vq_model"], loaded["fsq_model"]
    checkpoint_hashes = dict(vq=loaded["vq_hash"], fsq=loaded["fsq_hash"])
    names = loaded["names"]
    window = loaded["window"]

    # Cheap (mmap shape-only), so we can gate --resume reuse before any
    # heavy per-population extraction runs.
    frame_counts_probe = extract.recording_frame_counts(source_cfg, names)

    manifest_path = run_dir / "manifest.json"
    has_manifest = manifest_path.exists()
    has_partial_artifacts = (run_dir / "preprocessing").exists() or (run_dir / "families").exists()

    can_reuse = False
    if resume:
        if has_manifest:
            if not artifacts.manifest_matches(run_dir, prep_cfg, checkpoint_hashes, names, frame_counts_probe):
                raise SystemExit(
                    f"{manifest_path} does not match the current schema version, config, checkpoints, recording "
                    "set or per-recording frame counts; refusing to reuse mismatched preprocessing/family "
                    "artifacts under --resume. Choose a new --run-id (or fix whatever changed) instead of "
                    "resuming this one.")
            can_reuse = True
            print(f"[resume] {run_dir} manifest verified (schema/config/checkpoints/recordings/frame-counts all "
                  "match current invocation); reusing complete preprocessing/family artifacts, "
                  "(re)computing only what's missing or incomplete.", flush=True)
        elif has_partial_artifacts:
            raise SystemExit(
                f"{run_dir} contains preprocessing/family artifacts but no manifest.json -- most likely an "
                "earlier run crashed before completing (a manifest is written only once a run finishes "
                "successfully). These artifacts cannot be verified, so --resume refuses to reuse them. Remove "
                f"{run_dir} and start a fresh run (drop --resume), or choose a new --run-id.")
        # else: run_dir is empty/new; --resume with nothing to resume just
        # behaves like a fresh run (can_reuse stays False).
    elif run_dir.exists() and (has_manifest or has_partial_artifacts):
        raise SystemExit(f"{run_dir} already has {'a manifest' if has_manifest else 'partial artifacts'}; "
                          "pass --resume or choose a new run id")

    if can_reuse and all(artifacts.family_artifacts_complete(run_dir, fam, names, prep_cfg.num_codes)
                          for fam in cfg.FAMILIES):
        # Every family is already complete and verified against a matching
        # manifest -- nothing to (re)compute, not even extraction. Avoids
        # paying for a full re-extraction (the dominant cost) just to end up
        # regenerating byte-identical files; the manifest itself is left
        # untouched (it already matches, by construction of `can_reuse`).
        print(f"[resume] {run_dir}: all {len(cfg.FAMILIES)} families already complete and valid; "
              "nothing to (re)compute.", flush=True)
        return run_dir

    # --- "raw" population: also drives frame counts, the shared window
    # grid, the shared train/val split and the shared fitting sample. -------
    raw = extract.extract_population("raw", source_cfg, None, device, names)
    frame_counts = {name: int(sum(raw[name]["lengths"])) for name in names}
    mismatched = {n for n in names if frame_counts[n] != frame_counts_probe[n]}
    if mismatched:
        raise RuntimeError(f"raw windowed frame count disagrees with the raw feature file's own shape for "
                            f"{sorted(mismatched)[:5]}... -- extraction is internally inconsistent")

    lengths_by_name_raw = fitmod.window_lengths(raw)
    all_complete_ids = fitmod.complete_window_ids(lengths_by_name_raw, prep_cfg.window)

    sample_ids_path = run_dir / "preprocessing" / "sample_ids.json"
    if can_reuse and sample_ids_path.exists():
        sample_ids = artifacts.load_sample_ids(run_dir)
        sample_hash = artifacts.sample_ids_hash(sample_ids)
        recorded_hash = artifacts.read_manifest(run_dir).get("sample_ids_hash")
        if recorded_hash != sample_hash:
            raise SystemExit(
                f"{sample_ids_path} hash ({sample_hash}) does not match manifest.json's recorded "
                f"sample_ids_hash ({recorded_hash}); refusing to reuse a corrupted/mismatched "
                "fitting sample under --resume.")
    else:
        sample_ids = fitmod.sample_window_ids(all_complete_ids, prep_cfg.kmeans_sample_cap, prep_cfg.seed)
        artifacts.save_sample_ids(run_dir, sample_ids)
        sample_hash = artifacts.sample_ids_hash(sample_ids)

    train_names, val_names = fitmod.shared_validation_split(frame_counts, prep_cfg.seed)
    artifacts.save_split(run_dir, train_names, val_names, frame_counts)
    train_complete_ids = [(n, p) for n, p in all_complete_ids if n in set(train_names)]

    d = DATASET_DEFAULTS[source_cfg.dataset]
    fps = d["fps"]
    num_states = at_config.NUM_ACTIONS[source_cfg.dataset]

    pops: dict[str, fitmod.FittedPopulation] = {}
    dims: dict[str, int] = {}
    num_windows: dict[str, dict[str, int]] = {}
    prior_manifest = artifacts.read_manifest(run_dir) if can_reuse else None

    def process_population(pop_name: str, source: dict[str, dict]) -> None:
        """Fits (or reloads) `pop_name`'s preprocessing, then builds and
        saves every family defined over it -- skipping any family whose
        artifacts are already complete and valid -- all while `source` (the
        one big per-window population array) is the only population held in
        memory. Caller drops `source` immediately after this returns."""
        pop_path = run_dir / "preprocessing" / f"{pop_name}_kmeans.npz"
        family_names = POP_FAMILIES[pop_name]
        if can_reuse and pop_path.exists():
            # Reconstructed via real sklearn objects (see _load_population),
            # so its .transform()/.predict() match the fresh-fit path
            # exactly -- not a hand-rolled approximation of it.
            pop = _load_population(run_dir, pop_name, sample_ids)
        else:
            pop = fitmod.fit_population(pop_name, source, sample_ids, prep_cfg.pca_dim, prep_cfg.num_codes,
                                         prep_cfg.seed)
            artifacts.save_population(run_dir, pop)
        pops[pop_name] = pop

        pca_family = {n: fitmod.project_pca(pop, rec["flat"]) for n, rec in source.items()}
        for family in family_names:
            if can_reuse and artifacts.family_artifacts_complete(run_dir, family, names, prep_cfg.num_codes):
                lookup = np.load(run_dir / "families" / family / "lookup.npy")
                dims[family] = int(lookup.shape[1])
                num_windows[family] = (prior_manifest or {}).get("num_windows", {}).get(family) or {
                    n: int(len(np.load(run_dir / "families" / family / "recordings" / f"{n}.npz")["codes"]))
                    for n in names}
                print(f"[resume] {family}: artifacts complete and valid, skipping regeneration.", flush=True)
                continue
            records = families.build_family_records(family, source, window, pop if cfg.FAMILY_KIND[family] ==
                                                      "kmeans" else None)
            lookup = families.build_lookup(family, vq_model, fsq_model, pops)
            feature_by_name = {n: r["continuous"] for n, r in records.items()}
            stats = fitmod.conditioning_stats(feature_by_name, sample_ids, train_complete_ids)
            lengths_by_name = {n: r["lengths"] for n, r in records.items()}
            codes_by_name = {n: r["codes"] for n, r in records.items()}
            grouping, centers, counts = fitmod.initial_grouping(codes_by_name, lengths_by_name, pca_family,
                                                                  prep_cfg.num_codes, num_states, fps,
                                                                  prep_cfg.window, prep_cfg.seed)
            window_counts = np.zeros(prep_cfg.num_codes, dtype=np.int64)
            for codes in codes_by_name.values():
                np.add.at(window_counts, codes, 1)
            artifacts.save_family(run_dir, family, records, lookup, stats["mean"], stats["rms"], grouping, counts,
                                   window_counts)
            dims[family] = int(lookup.shape[1])
            num_windows[family] = {n: int(len(r["codes"])) for n, r in records.items()}

    process_population("raw", raw)
    del raw  # release before extracting the next (larger) population

    vq_encoder = extract.extract_population("vq_encoder", source_cfg, vq_model, device, names)
    process_population("vq_encoder", vq_encoder)
    del vq_encoder

    fsq_encoder = extract.extract_population("fsq_encoder", source_cfg, fsq_model, device, names)
    process_population("fsq_encoder", fsq_encoder)
    del fsq_encoder

    artifacts.write_manifest(run_dir, prep_cfg, checkpoint_hashes, sample_hash, dims, names, num_windows,
                              recording_frame_counts=frame_counts)
    print(f"[done] {run_dir}: {len(names)} recordings, families={list(cfg.FAMILIES)}, dims={dims}", flush=True)
    return run_dir


def _load_population(run_dir: Path, pop_name: str, sample_ids) -> fitmod.FittedPopulation:
    """Reconstructs a FittedPopulation from saved arrays for resume, using
    REAL sklearn StandardScaler/PCA/KMeans objects (attributes assigned
    directly, not fit) rather than hand-rolled affine maps -- so
    `.transform()`/`.predict()` on a resumed population run the exact same
    code path, with the exact same dtype-promotion behavior, as a freshly
    fit population's. (A prior version used a manual `(X - mean) @ comp.T`
    lambda, which silently computed in a different floating-point precision
    than sklearn's in-place `X -= mean_; X /= scale_` and drifted from the
    fresh-run path by ~1e-5 -- see the smoke test's resume-vs-sklearn
    transform check.)"""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    d = run_dir / "preprocessing"
    s = np.load(d / f"{pop_name}_scaler.npz")
    p = np.load(d / f"{pop_name}_pca.npz")
    k = np.load(d / f"{pop_name}_kmeans.npz")

    scaler = StandardScaler()
    scaler.mean_, scaler.scale_, scaler.var_ = s["mean_"], s["scale_"], s["var_"]
    scaler.n_features_in_ = s["mean_"].shape[0]
    scaler.n_samples_seen_ = np.int64(len(sample_ids))

    pca = PCA(n_components=int(p["n_components_"]))
    pca.components_ = p["components_"]
    pca.mean_ = p["mean_"]
    pca.explained_variance_ratio_ = p["explained_variance_ratio_"]
    pca.n_components_ = int(p["n_components_"])
    pca.n_features_in_ = p["components_"].shape[1]

    kmeans = KMeans(n_clusters=k["cluster_centers_"].shape[0])
    kmeans.cluster_centers_ = k["cluster_centers_"]
    kmeans.n_iter_ = int(k["n_iter_"])
    kmeans.inertia_ = float(k["inertia_"])
    kmeans._n_threads = 1

    return fitmod.FittedPopulation(pop_name, scaler, pca, kmeans, sample_ids,
                                    n_fit_windows=len(sample_ids), n_occupied=int(k["cluster_centers_"].shape[0]))


def build_parser():
    import argparse
    p = argparse.ArgumentParser(description="Quantized-input comparison: Task 1 data preparation")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="Run the unit test suite (tests.py), no real data.")
    mode.add_argument("--smoke", action="store_true", help="Bounded real-data extraction + full verification.")
    mode.add_argument("--full", action="store_true", help="Full preparation over all HuGaDB recordings.")
    p.add_argument("--run-id", dest="run_id", default=None)
    p.add_argument("--recording-cap", dest="recording_cap", type=int, default=12,
                    help="Recordings used by --smoke (default 12).")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.preflight:
        from script.quantized_input_comparison import tests
        tests.run_all()
        return
    if not args.run_id:
        raise SystemExit("--run-id is required for --smoke and --full")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cap = args.recording_cap if args.smoke else None
    prep_cfg = cfg.PrepConfig(run_id=args.run_id, recording_cap=cap)
    t0 = time.perf_counter()
    run_dir = run_preparation(prep_cfg, device, resume=args.resume)
    print(f"prepared in {time.perf_counter() - t0:.1f}s -> {run_dir}", flush=True)
    if args.smoke:
        from script.quantized_input_comparison import tests
        tests.run_real_data_checks(run_dir)


if __name__ == "__main__":
    main()
