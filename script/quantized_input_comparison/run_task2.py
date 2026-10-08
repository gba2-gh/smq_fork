"""Task 2 orchestrator: contextualizer training + OT decoding + segmentation
scoring on top of Task 1's artifacts (`qic_run_id`, default `qic_v3`).

Layout under results/quantized_input_comparison/task2/<run_id>/:

    manifest.json           schema_version, effective config, qic source
                             manifest hash, the predeclared cell list, seeds,
                             device, timings
    segmentation.csv         every rung-ladder row (context + direct +
                             historical), one row per (cell, rung)
    contextualizer.csv       one row per trained contextualizer (steps run,
                             skipped-chunk counts, coverage, param count,
                             losses, selected step)
    failed.csv               explicit failed/invalid/interrupted/incomplete
                             cell entries (never silently dropped)
    cells/<cell_id>/checkpoint_best.pt, checkpoint_last.pt, info.json,
        embeddings.npz, ot.npz (theta/mu + traces), geometry_init_tables.npz
        (context_geometry_init only: initial/final embedding tables)
    historical/<stream>/...  the two historical-compatibility jobs' own
                             artifacts + diff-vs-historical-CSV report

Resume safety follows Task 1's pattern exactly (see artifacts.py's and
prepare.py's module docstrings): a single verified `manifest_matches` gate
decides whether ANY cell output may be treated as reusable; a `run_dir`
with partial cell outputs but no manifest.json is refused, not trusted.
Every predeclared cell is always enumerated (never silently reduced), and
`--smoke` only ever shrinks per-cell BUDGETS (steps/rungs/recording subset/
seed count), never the arm/family coverage -- every family and every arm
type is still exercised at least once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from zipfile import BadZipFile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.action_transport.budget import DeadlineExceeded, check_deadline
from script.fsq_reconstruction.experiment import write_csv, write_json_atomic
from script.fsq_reconstruction.evaluate import FinalSolveInvalid
from script.fsq_reconstruction.quantizers import isolated_rng
from script.quantized_input_comparison import contextual as C
from script.quantized_input_comparison import diagnostics as diag
from script.quantized_input_comparison import historical_compat as hist
from script.quantized_input_comparison import ot_arms as ot
from script.quantized_input_comparison.artifacts import Loader, load_sample_ids, read_manifest

OUT_ROOT = Path(__file__).resolve().parents[2] / "results" / "quantized_input_comparison" / "task2"
# Bumped from 1 -> 2 for this fix round (predictions/diagnostics saved,
# invalid-fit status propagation, per-cell effective-config stamping): a
# schema bump means any `--resume` against a run_dir produced by the old
# code (e.g. `t2_smoke_v1`, whose manifest declared schema_version=1) is
# automatically refused by `manifest_matches` rather than silently trusted
# -- exactly the "these artifacts cannot certify the current implementation"
# gap the review found. See `TASK2_SOURCE_FILES`/`task2_source_hash` below
# for the additional per-cell config-hash gate this alone does not cover
# (two runs both at schema_version=2 with the same nominal Task2Config but
# produced by two different in-place code edits, as `t2_smoke_v1` was).
SCHEMA_VERSION = 2
NATIVE_SEEDS = (111, 222, 1538574472)
SEED_ONLY = (111,)
FAMILIES = ("vq_native", "fsq_native", "vq_kmeans", "fsq_kmeans", "raw_kmeans")
CONTEXT_ARMS = ("context_id", "context_quantized", "context_continuous")
GEOMETRY_FAMILIES = ("vq_native", "raw_kmeans")
DIRECT_ARMS = ("direct_id", "direct_quantized", "direct_continuous")
FULL_RUNGS = (500, 1000, 2000, 4000)
FULL_MAX_STEPS, FULL_EVAL_EVERY = 5000, 100

# Every module whose behavior determines a cell's actual output (training,
# masking, OT fitting/scoring, orchestration itself) -- hashed into the
# manifest and NOT just relied on via schema_version, so an in-place edit to
# any of these during a long-running/resumed invocation is caught rather
# than silently producing a mixed-implementation run (spec: "version/hash
# the effective implementation"; this is the concrete finding that
# `t2_smoke_v1` mixed rungs=(20,40) cells with rungs=(500,1000,2000,4000)
# cells from what was, in effect, two different implementations of
# `smoke_config`/the per-cell budget wiring).
TASK2_SOURCE_FILES = (
    "script/quantized_input_comparison/contextual.py",
    "script/quantized_input_comparison/ot_arms.py",
    "script/quantized_input_comparison/historical_compat.py",
    "script/quantized_input_comparison/masking.py",
    "script/quantized_input_comparison/diagnostics.py",
    "script/quantized_input_comparison/run_task2.py",
    "script/quantized_input_comparison/artifacts.py",
    "script/quantized_input_comparison/config.py",
    "script/quantized_input_comparison/fit.py",
    "script/quantized_input_comparison/families.py",
    "script/quantized_input_comparison/extract.py",
    "script/quantized_input_comparison/labels.py",
    "script/action_transport/categorical.py",
    "script/action_transport/continuous.py",
    "script/action_transport/transport.py",
    "script/action_transport/evaluate.py",
    "script/action_transport/stageb_model.py",
    "script/action_transport/stageb_data.py",
    "script/action_transport/stageb_configs.py",
    "script/action_transport/stageb_train.py",
    "script/action_transport/config.py",
    "script/action_transport/budget.py",
    "script/action_transport/stageb_infer.py",
    "script/fsq_reconstruction/evaluate.py",
    "script/fsq_reconstruction/experiment.py",
    "script/fsq_reconstruction/encoder_sweep.py",
    "script/fsq_reconstruction/quantizers.py",
    "script/exp2round/q1q2/core.py",
    "script/repro/d_error_decomposition.py",
    "src/model/smq.py",
    "src/model/motion_quantizer.py",
    "src/model/ms_tcn.py",
    "src/model/eval_utils.py",
)

_RUN_SOURCE_HASH: str | None = None


def task2_source_hash() -> str:
    root = Path(__file__).resolve().parents[2]
    h = hashlib.sha256()
    for name in TASK2_SOURCE_FILES:
        h.update(name.encode())
        h.update((root / name).read_bytes())
    return h.hexdigest()[:16]


def assert_source_unchanged() -> None:
    if _RUN_SOURCE_HASH is not None and task2_source_hash() != _RUN_SOURCE_HASH:
        raise RuntimeError("Task 2 source changed during this run; stopping before accepting another cell")


@dataclass(frozen=True)
class Task2Config:
    run_id: str
    qic_run_id: str = "qic_v3"
    max_steps: int = FULL_MAX_STEPS
    eval_every: int = FULL_EVAL_EVERY
    rungs: tuple = FULL_RUNGS
    recording_cap: int | None = None  # smoke-only: bound both training AND OT to the first N recordings
    native_seeds: tuple = NATIVE_SEEDS
    smoke: bool = False


class CellFitInvalid(Exception):
    """Raised when a cell's OT fit hit a nonfinite value or a backtracking
    failure (`outer_status == "invalid"`) -- a distinct, explicit status
    from a generic exception/"failed", so `failed.csv` records WHY the cell
    never became scoreable rather than folding it into an opaque crash."""


def cell_id(family: str, arm: str, seed: int) -> str:
    return f"{family}__{arm}__seed{seed}"


def enumerate_context_cells(cfg: Task2Config) -> list[tuple[str, str, int]]:
    cells = []
    for family in FAMILIES:
        seeds = cfg.native_seeds if family == "vq_native" else SEED_ONLY
        for arm in CONTEXT_ARMS:
            for seed in seeds:
                cells.append((family, arm, seed))
    for family in GEOMETRY_FAMILIES:
        seeds = cfg.native_seeds if family == "vq_native" else SEED_ONLY
        for seed in seeds:
            cells.append((family, "context_geometry_init", seed))
    return cells


def enumerate_direct_cells() -> list[tuple[str, str, int]]:
    return [(family, arm, 111) for family in FAMILIES for arm in DIRECT_ARMS]


def config_hash(cfg: Task2Config) -> str:
    """Hashes both the declared config AND the resolved Task 2 source
    (`task2_source_hash`) into one value -- two invocations with the
    identical `Task2Config` fields but a different in-place code edit in
    between (exactly how `t2_smoke_v1` ended up mixing rungs=(20,40) and
    rungs=(500,1000,2000,4000) cells) now hash differently, so `--resume`
    cannot silently treat the earlier cells as reusable."""
    payload = dict(config=asdict(cfg), task2_source_hash=_RUN_SOURCE_HASH or task2_source_hash())
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def manifest_matches(run_dir: Path, cfg: Task2Config, qic_manifest_hash: str) -> bool:
    path = run_dir / "manifest.json"
    if not path.exists():
        return False
    m = json.loads(path.read_text())
    return (m.get("schema_version") == SCHEMA_VERSION and m.get("config_hash") == config_hash(cfg)
            and m.get("qic_manifest_hash") == qic_manifest_hash)


def qic_manifest_hash(qic_run_dir: Path) -> str:
    m = read_manifest(qic_run_dir)
    return hashlib.sha256(json.dumps(m, sort_keys=True, default=str).encode()).hexdigest()[:16]


def write_manifest(run_dir: Path, cfg: Task2Config, qic_manifest_hash_: str, cells: dict) -> None:
    write_json_atomic(run_dir / "manifest.json", dict(
        schema_version=SCHEMA_VERSION, config=asdict(cfg), config_hash=config_hash(cfg),
        task2_source_hash=_RUN_SOURCE_HASH or task2_source_hash(), qic_manifest_hash=qic_manifest_hash_, cells=cells,
        updated_local=time.strftime("%Y-%m-%dT%H:%M:%S%z")))


def cell_complete(cell_dir: Path, cfg: Task2Config, loader: Loader | None = None) -> bool:
    """A cell counts as complete/reusable only when ALL of: its `info.json`
    and `ladder.json` exist, `info.json` records `status == "done"` (never
    "invalid"/"failed" -- an invalid fit must not be reused as if it
    succeeded), `info.json`'s own `effective_config_hash` matches the
    CURRENTLY requested run's `config_hash(cfg)` (catches a cell produced
    under a different budget/rungs/source, even if the top-level manifest
    was later overwritten with the current config -- the concrete
    `t2_smoke_v1` failure mode), and `ladder.json`'s rows actually cover
    every rung `cfg.rungs` declares (not just "some rows exist")."""
    info_path, ladder_path = cell_dir / "info.json", cell_dir / "ladder.json"
    required = [info_path, ladder_path, cell_dir / "ot.npz", cell_dir / "predictions.npz"]
    is_context = "context_" in cell_dir.name
    if is_context:
        required += [cell_dir / "checkpoint_best.pt", cell_dir / "checkpoint_last.pt",
                     cell_dir / "embeddings.npz"]
    if "context_geometry_init" in cell_dir.name:
        required.append(cell_dir / "geometry_init_tables.npz")
    if any(not path.exists() or path.stat().st_size == 0 for path in required):
        return False
    try:
        info = json.loads(info_path.read_text())
        ladder = json.loads(ladder_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if info.get("status") != "done":
        return False
    if info.get("effective_config_hash") != config_hash(cfg):
        return False
    rows = ladder.get("rows", [])
    rung_steps_present = {r.get("rung_steps") for r in rows}
    if rung_steps_present != set(cfg.rungs) or len(rows) != len(cfg.rungs):
        return False
    metric_fields = ("MoF", "Edit", "F1_10", "F1_25", "F1_50")
    try:
        if any(not math.isfinite(float(row[field])) for row in rows for field in metric_fields):
            return False
        if info.get("ot_status") == "invalid" or any(
            int(row.get("status_counts", {}).get("invalid", 0)) > 0 for row in rows
        ):
            return False
        names = info["scored_recording_names"]
        if not names or len(names) != info["n_scored_recordings"] or len(names) != len(set(names)):
            return False
        if loader is not None and set(names) != set(bounded_names(loader, cfg.recording_cap)):
            return False
        with np.load(cell_dir / "predictions.npz") as predictions:
            expected = {f"{name}__frame_pred" for name in names} | {f"{name}__window_states" for name in names}
            if set(predictions.files) != expected:
                return False
            for name in names:
                frame_pred = predictions[f"{name}__frame_pred"]
                states = predictions[f"{name}__window_states"]
                if frame_pred.ndim != 1 or states.ndim != 1 or not len(frame_pred) or not len(states):
                    return False
                if loader is not None and (len(frame_pred) != loader.frame_count(name) or
                                           len(states) != loader.manifest["num_windows"][info["family"]][name]):
                    return False
    except (KeyError, TypeError, ValueError, OSError, EOFError, BadZipFile):
        return False
    return True


def read_csv_rows(path: Path) -> list[dict]:
    import csv
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def ladder_to_rows(family: str, arm: str, seed: int, rows: list) -> list[dict]:
    out = []
    for r in rows:
        out.append(dict(family=family, arm=arm, seed=seed, rung_steps=r.rung_steps, MoF=r.MoF, Edit=r.Edit,
                       **{"F1@10": r.F1_10, "F1@25": r.F1_25, "F1@50": r.F1_50},
                       status_counts=r.status_counts, seconds=r.seconds))
    return out


def bounded_names(loader: Loader, cap: int | None) -> list[str]:
    names = loader.recording_names()
    return names[:cap] if cap else names


# --- one context cell ---------------------------------------------------------------

def run_context_cell(family: str, arm: str, seed: int, cfg: Task2Config, loader: Loader, cell_dir: Path,
                     device: torch.device, deadline: float | None) -> dict:
    names = bounded_names(loader, cfg.recording_cap)
    train_names = [n for n in names if loader.split_membership(n) == "train"] or names[:max(1, len(names) - 1)]
    val_names = [n for n in names if loader.split_membership(n) == "val"] or names[-1:]
    input_dim = loader.lookup(family).shape[1]
    num_codes = loader.lookup(family).shape[0]
    lookup = loader.lookup(family)
    rms = float(loader.rms(family))

    chunks_tr = C.make_vector_chunks(loader, family, train_names)
    chunks_val = C.make_vector_chunks(loader, family, val_names)

    train_cfg = C.base_train_config(max_steps=cfg.max_steps, eval_every=cfg.eval_every,
                                    patience_steps=10**9)
    init = C.build_matched_init(num_codes, input_dim, train_cfg, seed=seed)

    if arm == "context_geometry_init":
        geo_lookup_over_rms = lookup / rms
    else:
        geo_lookup_over_rms = None

    result = C.train_vector_contextualizer(arm, chunks_tr, chunks_val, num_codes, input_dim, seed, train_cfg,
                                           init, device, lookup=lookup, rms=rms, mask_base=111, deadline=deadline)
    model = result.model

    if arm == "context_geometry_init":
        # spec's rtol=1e-5/atol=1e-6 equivalence check, against a FRESH (untrained)
        # context_quantized model sharing this cell's exact init.
        quant_model = C.build_arm_model("context_quantized", num_codes, input_dim, train_cfg, seed, init,
                                        device=device)
        probe_codes = np.arange(min(num_codes, chunks_tr[0].codes.shape[0]), dtype=np.int64)
        probe_codes = np.resize(probe_codes, chunks_tr[0].codes.shape[0])
        probe_vecs = geo_lookup_over_rms[probe_codes]
        fresh_geo = C.build_arm_model("context_geometry_init", num_codes, input_dim, train_cfg, seed, init,
                                      lookup_over_rms=geo_lookup_over_rms, device=device)
        geo_diff = C.verify_geometry_init_matches_quantized(fresh_geo, quant_model, probe_codes, probe_vecs)
        initial_table = fresh_geo.embedding.weight.detach().cpu().numpy().copy()
    else:
        geo_diff = None
        initial_table = None

    cell_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": result.info.pop("last_state"), "config": model.cfg.__dict__},
              cell_dir / "checkpoint_last.pt")
    torch.save({"state_dict": model.state_dict(), "config": model.cfg.__dict__},
              cell_dir / "checkpoint_best.pt")
    if initial_table is not None:
        np.savez(cell_dir / "geometry_init_tables.npz", initial=initial_table,
                final=model.embedding.weight.detach().cpu().numpy())

    emb_by_name, zero_by_name, codes_by_name, len_by_name, starts_by_name = {}, {}, {}, {}, {}
    all_names = loader.recording_names() if cfg.recording_cap is None else names
    for n in all_names:
        check_deadline(deadline)
        emb, zero = C.extract_vector_embeddings(arm, model, loader, family, n, lookup, rms, device)
        rec = loader.recording(family, n)
        emb_by_name[n], zero_by_name[n] = emb, zero
        codes_by_name[n], len_by_name[n], starts_by_name[n] = rec["codes"], rec["lengths"], rec["starts"]
    np.savez_compressed(cell_dir / "embeddings.npz", **{f"{n}__emb": emb_by_name[n] for n in all_names},
                        **{f"{n}__zero": zero_by_name[n] for n in all_names})

    recs = ot.continuous_recordings_from_embeddings(codes_by_name, len_by_name, emb_by_name, zero_by_name)
    grouping = loader.grouping(family)
    fit = ot.fit_continuous_arm(recs, grouping, deadline)
    np.savez(cell_dir / "ot.npz", mu=fit.params, objective_trace=np.asarray(fit.objective_trace),
             outer_status=fit.outer_status, outer_iterations=fit.outer_iterations)
    if fit.outer_status == "invalid":
        # A continuous OT fit that hit a nonfinite value or a backtracking
        # failure produces a degenerate `mu` (final prototypes); scoring
        # against it (or, worse, silently marking this cell "done" with an
        # empty ladder) would be meaningless. Fail this cell explicitly.
        raise CellFitInvalid(f"context cell {family}/{arm}/seed{seed}: continuous OT fit invalid "
                             f"(outer_status={fit.outer_status}, outer_iterations={fit.outer_iterations})")

    frame_counts = {n: loader.frame_count(n) for n in all_names}
    predictions: dict = {}
    ladder = ot.score_continuous(fit.params, recs, starts_by_name, frame_counts, all_names, deadline,
                                 rungs=cfg.rungs, predictions_out=predictions)

    from script.quantized_input_comparison import labels as qic_labels
    gts_by_name = qic_labels.gts_for(all_names)
    window_states = predictions.pop("__window_states__", {})
    np.savez_compressed(cell_dir / "predictions.npz",
                        **{f"{n}__frame_pred": predictions[n] for n in predictions},
                        **{f"{n}__window_states": window_states[n] for n in window_states})

    run_length = diag.run_length_diagnostics(predictions, gts_by_name, ot.FPS)
    valid_len_by_name = {n: len(codes_by_name[n]) for n in all_names}
    adjacent = diag.adjacent_input_cosine_similarity(emb_by_name, valid_len_by_name)
    fitting_ids = [(n, i) for n, i in load_sample_ids(loader.run_dir) if n in emb_by_name]
    rank_sample = diag.rank_sample_vectors(emb_by_name, fitting_ids)
    vector_diag = diag.vector_norm_rank_diagnostics(
        [emb_by_name[n][:valid_len_by_name[n]] for n in all_names], rank_sample)
    emission = diag.emission_cost_contrast(ot.continuous_emission_costs(fit.params, recs))
    occupancy = loader.occupancy(family)
    diagnostics = dict(run_length=run_length, adjacent_input_similarity=adjacent,
                       vector_norm_rank=vector_diag, emission_cost_contrast=emission,
                       code_occupancy=dict(counts=occupancy["counts"].tolist(),
                                          window_counts=occupancy["window_counts"].tolist()))

    result_info = {k: v for k, v in result.info.items() if k not in ("family", "arm", "seed")}
    info = dict(family=family, arm=arm, seed=seed, status="done", effective_config_hash=config_hash(cfg),
               **result_info, geometry_init_max_abs_diff=geo_diff, ot_status=fit.outer_status,
               ot_outer_iterations=fit.outer_iterations, n_scored_recordings=len(all_names),
               scored_recording_names=all_names,
               diagnostics=diagnostics)
    write_json_atomic(cell_dir / "info.json", info)
    write_json_atomic(cell_dir / "ladder.json", dict(rows=[asdict(r) for r in ladder]))
    return dict(info=info, ladder_rows=ladder_to_rows(family, arm, seed, ladder))


def run_direct_cell(family: str, arm: str, seed: int, cfg: Task2Config, loader: Loader, cell_dir: Path,
                    deadline: float | None) -> dict:
    names = bounded_names(loader, cfg.recording_cap)
    from script.quantized_input_comparison import labels as qic_labels
    gts_by_name = qic_labels.gts_for(names)
    cell_dir.mkdir(parents=True, exist_ok=True)
    predictions: dict = {}
    if arm == "direct_id":
        result = ot.fit_and_score_categorical(loader, family, arm, names, names, deadline, rungs=cfg.rungs,
                                              predictions_out=predictions)
        fit_result, ladder = result["fit_result"], result["ladder"]
        np.savez(cell_dir / "ot.npz", theta=fit_result.params, objective_trace=np.asarray(fit_result.objective_trace),
                 outer_status=fit_result.outer_status, outer_iterations=fit_result.outer_iterations)
        if fit_result.outer_status == "invalid":
            raise CellFitInvalid(f"direct cell {family}/{arm}/seed{seed}: categorical OT fit invalid "
                                 f"(outer_status={fit_result.outer_status})")
        codes_by_name = {n: loader.recording(family, n)["codes"] for n in names}
        emission = diag.emission_cost_contrast(
            ot.categorical_emission_costs(fit_result.params, ot.categorical_recordings(loader, family, names)))
        adjacent = diag.adjacent_same_code_rate(codes_by_name)
        vector_diag = dict(mean_norm=None, std_norm=None, effective_rank=None, ambient_dim=None, n_vectors=0)
    else:
        recs = ot.continuous_recordings_from_raw(loader, family, arm, names)
        grouping = loader.grouping(family)
        fit_result = ot.fit_continuous_arm(recs, grouping, deadline)
        np.savez(cell_dir / "ot.npz", mu=fit_result.params, objective_trace=np.asarray(fit_result.objective_trace),
                 outer_status=fit_result.outer_status, outer_iterations=fit_result.outer_iterations)
        if fit_result.outer_status == "invalid":
            raise CellFitInvalid(f"direct cell {family}/{arm}/seed{seed}: continuous OT fit invalid "
                                 f"(outer_status={fit_result.outer_status})")
        starts = {n: loader.recording(family, n)["starts"] for n in names}
        frame_counts = {n: loader.frame_count(n) for n in names}
        ladder = ot.score_continuous(fit_result.params, recs, starts, frame_counts, names, deadline,
                                     rungs=cfg.rungs, predictions_out=predictions)
        vecs_by_name = {r.name: r.z for r in recs}
        len_by_name = {r.name: r.z.shape[0] for r in recs}
        adjacent = diag.adjacent_input_cosine_similarity(vecs_by_name, len_by_name)
        fitting_ids = [(n, i) for n, i in load_sample_ids(loader.run_dir) if n in vecs_by_name]
        rank_sample = diag.rank_sample_vectors(vecs_by_name, fitting_ids)
        vector_diag = diag.vector_norm_rank_diagnostics([r.z for r in recs], rank_sample)
        emission = diag.emission_cost_contrast(ot.continuous_emission_costs(fit_result.params, recs))

    window_states = predictions.pop("__window_states__", {})
    np.savez_compressed(cell_dir / "predictions.npz",
                        **{f"{n}__frame_pred": predictions[n] for n in predictions},
                        **{f"{n}__window_states": window_states[n] for n in window_states})
    run_length = diag.run_length_diagnostics(predictions, gts_by_name, ot.FPS)
    occupancy = loader.occupancy(family)
    diagnostics = dict(run_length=run_length, adjacent_input_similarity=adjacent,
                       vector_norm_rank=vector_diag, emission_cost_contrast=emission,
                       code_occupancy=dict(counts=occupancy["counts"].tolist(),
                                          window_counts=occupancy["window_counts"].tolist()))

    info = dict(family=family, arm=arm, seed=seed, status="done", effective_config_hash=config_hash(cfg),
               ot_status=fit_result.outer_status, ot_outer_iterations=fit_result.outer_iterations,
               n_scored_recordings=len(names), scored_recording_names=names, diagnostics=diagnostics)
    write_json_atomic(cell_dir / "info.json", info)
    write_json_atomic(cell_dir / "ladder.json", dict(rows=[asdict(r) for r in ladder]))
    return dict(info=info, ladder_rows=ladder_to_rows(family, arm, seed, ladder))


# --- orchestration --------------------------------------------------------------------

def run(cfg: Task2Config, resume: bool, deadline: float | None, cpu_only: bool) -> None:
    global _RUN_SOURCE_HASH
    _RUN_SOURCE_HASH = task2_source_hash()
    run_dir = OUT_ROOT / cfg.run_id
    qic_dir = Path(__file__).resolve().parents[2] / "results" / "quantized_input_comparison" / cfg.qic_run_id
    qmh = qic_manifest_hash(qic_dir)

    if run_dir.exists() and (run_dir / "manifest.json").exists():
        if not manifest_matches(run_dir, cfg, qmh):
            raise SystemExit(f"{run_dir}/manifest.json does not match the current schema version, config or "
                             f"qic source manifest; refusing to reuse mismatched artifacts under --resume. "
                             f"Choose a new --run-id or fix the mismatch.")
        if not resume:
            raise SystemExit(f"{run_dir}/manifest.json already exists; pass --resume or choose a new --run-id")
    elif run_dir.exists() and any(run_dir.rglob("info.json")):
        raise SystemExit(f"{run_dir} contains cell artifacts but no manifest.json -- most likely an earlier "
                         f"run crashed before completing. These artifacts cannot be verified, refusing --resume.")

    loader = Loader(qic_dir)
    ot.set_cpu_thread_cap()
    device = torch.device("cuda" if (torch.cuda.is_available() and not cpu_only) else "cpu")

    context_cells = enumerate_context_cells(cfg)
    direct_cells = enumerate_direct_cells()
    all_cells = {cell_id(f, a, s): dict(family=f, arm=a, seed=s, kind="context" if a in CONTEXT_ARMS or
                                       a == "context_geometry_init" else "direct")
               for f, a, s in context_cells + direct_cells}
    write_manifest(run_dir, cfg, qmh, all_cells)

    seg_rows = read_csv_rows(run_dir / "segmentation.csv")
    ctx_rows = read_csv_rows(run_dir / "contextualizer.csv")
    failed_rows = read_csv_rows(run_dir / "failed.csv")
    done_ids = {r["cell_id"] for r in read_csv_rows(run_dir / "done.csv")}
    # Defensive prune on load: a cell that failed in an earlier invocation and
    # was later completed (its cell_id already in done.csv) must not carry a
    # stale "failed" ledger entry forward, even across separate process runs.
    if any(r.get("cell_id") in done_ids for r in failed_rows):
        failed_rows = [r for r in failed_rows if r.get("cell_id") not in done_ids]
        write_csv(run_dir / "failed.csv", failed_rows)

    def mark_done(cid: str) -> None:
        done_ids.add(cid)
        write_csv(run_dir / "done.csv", [{"cell_id": c} for c in sorted(done_ids)])
        # A cell that failed on a previous attempt and now succeeds on --resume
        # (e.g. a bug fix) must not leave a stale "failed" ledger entry behind.
        before = len(failed_rows)
        failed_rows[:] = [r for r in failed_rows if r.get("cell_id") != cid]
        if len(failed_rows) != before:
            write_csv(run_dir / "failed.csv", failed_rows)

    def discard_stale_cell(cid: str, family: str, arm: str, seed: int) -> None:
        """Remove ledger rows for a done cell whose artifacts failed verification.

        A rerun may fail, so the old rows must not remain labeled as completed
        or be duplicated when a new attempt succeeds.
        """
        if cid not in done_ids:
            return
        done_ids.remove(cid)
        write_csv(run_dir / "done.csv", [{"cell_id": c} for c in sorted(done_ids)])
        seg_rows[:] = [r for r in seg_rows if not (r.get("family") == family and
                       r.get("arm") == arm and str(r.get("seed")) == str(seed))]
        ctx_rows[:] = [r for r in ctx_rows if r.get("cell_id") != cid]
        write_csv(run_dir / "segmentation.csv", seg_rows)
        write_csv(run_dir / "contextualizer.csv", ctx_rows)

    def mark_failed(cid: str, family, arm, seed, reason: str, status: str) -> None:
        failed_rows[:] = [r for r in failed_rows if r.get("cell_id") != cid]
        failed_rows.append(dict(cell_id=cid, family=family, arm=arm, seed=seed, status=status, reason=reason))
        write_csv(run_dir / "failed.csv", failed_rows)

    started = time.perf_counter()
    print(f"[task2] run_dir={run_dir} device={device} cells={len(all_cells)} "
         f"(context={len(context_cells)} direct={len(direct_cells)})", flush=True)

    for family, arm, seed in context_cells:
        assert_source_unchanged()
        cid = cell_id(family, arm, seed)
        cell_dir = run_dir / "cells" / cid
        if cid in done_ids and cell_complete(cell_dir, cfg, loader):
            continue
        discard_stale_cell(cid, family, arm, seed)
        if deadline is not None and time.perf_counter() > deadline:
            mark_failed(cid, family, arm, seed, "compute deadline reached before this cell started", "not_run")
            continue
        print(f"[context] {cid} ...", flush=True)
        try:
            out = run_context_cell(family, arm, seed, cfg, loader, cell_dir, device, deadline)
        except DeadlineExceeded:
            mark_failed(cid, family, arm, seed, "interrupted at a cooperative deadline check", "interrupted")
            continue
        except (CellFitInvalid, FinalSolveInvalid) as exc:
            mark_failed(cid, family, arm, seed, str(exc), "invalid")
            continue
        except Exception as exc:  # noqa: BLE001 -- explicit failed-cell bookkeeping, never a silent skip
            mark_failed(cid, family, arm, seed, f"{type(exc).__name__}: {exc}", "failed")
            continue
        assert_source_unchanged()
        ctx_rows.append(dict(cell_id=cid, **{k: v for k, v in out["info"].items() if k != "history"}))
        seg_rows += out["ladder_rows"]
        write_csv(run_dir / "contextualizer.csv", ctx_rows)
        write_csv(run_dir / "segmentation.csv", seg_rows)
        mark_done(cid)

    for family, arm, seed in direct_cells:
        assert_source_unchanged()
        cid = cell_id(family, arm, seed)
        cell_dir = run_dir / "cells" / cid
        if cid in done_ids and cell_complete(cell_dir, cfg, loader):
            continue
        discard_stale_cell(cid, family, arm, seed)
        if deadline is not None and time.perf_counter() > deadline:
            mark_failed(cid, family, arm, seed, "compute deadline reached before this cell started", "not_run")
            continue
        print(f"[direct] {cid} ...", flush=True)
        try:
            out = run_direct_cell(family, arm, seed, cfg, loader, cell_dir, deadline)
        except DeadlineExceeded:
            mark_failed(cid, family, arm, seed, "interrupted at a cooperative deadline check", "interrupted")
            continue
        except (CellFitInvalid, FinalSolveInvalid) as exc:
            mark_failed(cid, family, arm, seed, str(exc), "invalid")
            continue
        except Exception as exc:  # noqa: BLE001
            mark_failed(cid, family, arm, seed, f"{type(exc).__name__}: {exc}", "failed")
            continue
        assert_source_unchanged()
        seg_rows += out["ladder_rows"]
        write_csv(run_dir / "segmentation.csv", seg_rows)
        mark_done(cid)

    # --- historical compatibility (2 fits, 2 evals) ---
    hist_dir = run_dir / "historical"
    hist_done = {r["stream"] for r in read_csv_rows(hist_dir / "done.csv")} if hist_dir.exists() else set()
    hist_rows = read_csv_rows(hist_dir / "segmentation.csv") if hist_dir.exists() else []
    hist_diff_rows = read_csv_rows(hist_dir / "diff.csv") if hist_dir.exists() else []
    for stream_name in hist.STREAMS:
        assert_source_unchanged()
        if stream_name in hist_done:
            continue
        if deadline is not None and time.perf_counter() > deadline:
            mark_failed(f"historical_{stream_name}", None, stream_name, hist.SEED,
                       "compute deadline reached before this job started", "not_run")
            continue
        print(f"[historical] {stream_name} ...", flush=True)
        # NOTE: spec says the historical-compatibility jobs use "patience_steps=1000"
        # (the real historical setting) -- kept exactly for a --full run. Under --smoke
        # only, the contextualizer step budget is ALSO shrunk (not just final_rungs)
        # so the smoke matrix stays fast; a real patience_steps=1000/max_steps=5000
        # reproduction check against the historical CSV is deferred to the full run.
        overrides = dict(final_rungs=cfg.rungs, contextualizer_max_steps=cfg.max_steps,
                        contextualizer_eval_every=cfg.eval_every,
                        contextualizer_patience_steps=cfg.max_steps) if cfg.smoke else None
        try:
            result = hist.run_one_stream(stream_name, "fsq_pilot_002", device, hist_dir / "artifacts", deadline,
                                        cfg_overrides=overrides)
        except DeadlineExceeded:
            mark_failed(f"historical_{stream_name}", None, stream_name, hist.SEED,
                       "interrupted at a cooperative deadline check", "interrupted")
            continue
        except Exception as exc:  # noqa: BLE001
            mark_failed(f"historical_{stream_name}", None, stream_name, hist.SEED, f"{type(exc).__name__}: {exc}",
                        "failed")
            continue
        assert_source_unchanged()
        hist_rows += result["rows"]
        diffs = hist.diff_against_historical(result["rows"], hist.read_historical_rows(stream_name))
        hist_diff_rows += diffs
        hist_dir.mkdir(parents=True, exist_ok=True)
        write_csv(hist_dir / "segmentation.csv", hist_rows)
        write_csv(hist_dir / "diff.csv", hist_diff_rows)
        hist_done.add(stream_name)
        write_csv(hist_dir / "done.csv", [{"stream": s} for s in sorted(hist_done)])

    assert_source_unchanged()
    write_manifest(run_dir, cfg, qmh, all_cells)
    print(f"[task2] done in {time.perf_counter() - started:.1f}s. "
         f"{len(seg_rows)} segmentation rows, {len(failed_rows)} failed/incomplete cell(s).", flush=True)


def smoke_config(run_id: str, qic_run_id: str) -> Task2Config:
    return Task2Config(run_id=run_id, qic_run_id=qic_run_id, max_steps=40, eval_every=10,
                       rungs=(20, 40), recording_cap=8, native_seeds=(111,), smoke=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2: contextualizer training + OT decoding + scoring")
    parser.add_argument("--run-id", dest="run_id")
    parser.add_argument("--qic-run-id", dest="qic_run_id", default="qic_v3")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hours", type=float, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.preflight:
        from script.quantized_input_comparison import tests_task2
        ok = tests_task2.run_all()
        raise SystemExit(0 if ok else 1)

    if not args.run_id:
        raise SystemExit("--run-id is required for --smoke/--full runs")

    cfg = smoke_config(args.run_id, args.qic_run_id) if args.smoke else Task2Config(
        run_id=args.run_id, qic_run_id=args.qic_run_id)
    deadline = time.perf_counter() + args.hours * 3600.0 if args.hours else None
    run(cfg, args.resume, deadline, args.cpu)


if __name__ == "__main__":
    main()
