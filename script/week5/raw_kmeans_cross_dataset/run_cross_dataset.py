"""Raw K-means tokens across datasets (HuGaDB anchor, LARa x 2 normalizations, BABEL 1/2/3).

    python -m script.week5.raw_kmeans_cross_dataset.run_cross_dataset --preflight | --dry-run | --smoke | full

Layout under results/week5/raw_kmeans_cross_dataset/<run_id>/:

    manifest.json  planned_cells.csv  gates.json
    prep/<cell>/            per-cell raw K-means artifacts (qic-compatible layout, derived and deterministic)
    jobs/<cell>/            contextualizer checkpoints, ledger, exposure arrays, training_info.json
    evals/<identity>/       embeddings, prototypes, predictions, ladder
    eval_records/           planned decode -> identity / status
    cells.csv segmentation.csv summary.csv runtime.csv not_run.csv REPORT.md   (rebuilt from the artifacts)

One cell = (dataset, normalization, seed): prepare tokens, train the frozen reference contextualizer, decode the
validation-selected and the final checkpoint. Everything except dataset geometry (window, fps, number of classes) is
frozen from HuGaDB; nothing is tuned on LARa or BABEL. Execution is sequential; nothing is shrunk or substituted.
"""

from __future__ import annotations

import os

_CAP = str(min(8, os.cpu_count() or 1))
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_var] = _CAP

import argparse
import hashlib
import json
import shutil
import statistics
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from script.action_transport import evaluate as at_evaluate
from script.action_transport.budget import DeadlineExceeded
from script.fsq_reconstruction.evaluate import FinalSolveInvalid
from script.fsq_reconstruction.experiment import write_csv, write_json_atomic
from script.quantized_input_comparison.artifacts import Loader
from script.week5.contextualizer_calibration import evaluate as E
from script.week5.contextualizer_calibration import plan as P
from script.week5.contextualizer_calibration import run_calibration as R
from script.week5.contextualizer_calibration import train as T
from script.week5.raw_kmeans_cross_dataset import datasets as D
from script.week5.raw_kmeans_cross_dataset import decode as DEC
from script.week5.raw_kmeans_cross_dataset import prepare as PREP

OUT_ROOT = ROOT / "results" / "week5" / "raw_kmeans_cross_dataset"
SCHEMA_VERSION = 1
SEEDS = (111, 222, 1538574472)
ROLES = ("selected", "final")
REFERENCE_JOB = "raw__ctx128_span8_g0_h5000_seed111_tcnone"
OWN_FILES = tuple(f"script/week5/raw_kmeans_cross_dataset/{n}.py"
                  for n in ("datasets", "prepare", "decode", "run_cross_dataset", "report", "tests"))
SOURCE_FILES = tuple(R.SOURCE_FILES) + tuple(f for f in (
    "script/week5/contextualizer_calibration/evaluate.py",) if f not in R.SOURCE_FILES) + OWN_FILES
GATE_TOLERANCE = 0.005  # the scorer rounds to two decimals; saved SMQ numbers are unrounded


@dataclass(frozen=True)
class Cell:
    dataset: str
    normalization: str
    seed: int

    @property
    def cell_id(self) -> str:
        return f"{self.dataset}__{self.normalization}__seed{self.seed}"


def make_cells(datasets: tuple[str, ...] | None = None) -> list[Cell]:
    cells = []
    for ds in D.DATASET_ORDER:
        if datasets and ds not in datasets:
            continue
        for norm in (PREP.NORMALIZATIONS if ds == "lara" else ("global",)):
            cells += [Cell(ds, norm, s) for s in SEEDS]
    return cells


def make_job(cell: Cell) -> P.JobSpec:
    """The frozen HuGaDB reference configuration (context 128, span 8, guard 0, 5,000 updates)."""
    context, span, stride, batch = P.CONFIGS["reference"]
    return P.JobSpec(cell.cell_id, "main", "reference", "raw", PREP.FAMILY, "context_id", 0, context, stride, span,
                     0.30, batch, P.FULL_HORIZON, cell.seed, "none", "")


def scale_for(smoke: bool) -> P.Scale:
    return replace(P.SMOKE, recording_cap=None) if smoke else P.FULL  # smoke caps recordings in the preparation itself


def source_hash() -> str:
    h = hashlib.sha256()
    for name in SOURCE_FILES:
        h.update(name.encode())
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()[:16]


# --- gates ----------------------------------------------------------------------------------

def gate_label_loading_and_scoring(spec: D.DatasetSpec) -> dict:
    """Gate 3: the label loader reproduces the saved SMQ ground truth, and re-scoring the saved SMQ predictions
    reproduces the published-protocol numbers in results/e0 (the scorer rounds to two decimals)."""
    pred = np.load(ROOT / "results" / "preds" / f"{spec.name}_pretrained.npz", allow_pickle=True)
    names = [str(n) for n in pred["names"]]
    gts = D.load_gts(spec, names)
    saved = [np.asarray(g, dtype=np.int32) for g in pred["gt"]]
    labels_equal = all(np.array_equal(gts[n], g) for n, g in zip(names, saved))
    preds = [np.asarray(p, dtype=np.int32) for p in pred["pred"]]
    metrics = at_evaluate.score_pooled(saved, preds)[0]
    reference = json.loads((ROOT / "results" / "e0" / f"{spec.name}_pretrained.json").read_text())["global"]
    diffs = {k: float(metrics[k]) - float(reference[k]) for k in metrics}
    smq_f1_50 = float(reference["F1@50"])
    return dict(dataset=spec.name, passed=bool(labels_equal and all(abs(v) <= GATE_TOLERANCE + 1e-9 for v in diffs.values())),
                labels_equal_saved_gt=labels_equal, n_recordings=len(names), metric_diffs=diffs,
                smq_local_f1_50=smq_f1_50, smq_local={k: float(v) for k, v in reference.items()},
                num_classes_mapping=len(D.label_map(spec)), expected_states=spec.num_states)


LOOKUP_ABS_TOLERANCE = 1e-5  # float32 K-means center arithmetic; thread-order dependent, not an input to the ID arm


def compare_arrays(a_loader: Loader, b_loader: Loader, family_a: str, family_b: str) -> dict:
    """Gate 1: two prepared raw-K-means families. Codes, features, windows, statistics, grouping, occupancy, split and the
    fitted scaler / PCA must be identical; the K-means centers (`lookup`, unused by the raw ID arm) may differ only by
    float32 accumulation noise, and the measured difference is recorded."""
    names = a_loader.recording_names()
    same_names = names == b_loader.recording_names()
    rec_equal = same_names and all(
        all(np.array_equal(a_loader.recording(family_a, n)[k], b_loader.recording(family_b, n)[k])
            for k in ("codes", "continuous", "starts", "lengths")) for n in names)
    arrays = {k: bool(np.array_equal(getattr(a_loader, k)(family_a), getattr(b_loader, k)(family_b)))
              for k in ("mean", "rms", "grouping")}
    la, lb = a_loader.lookup(family_a), b_loader.lookup(family_b)
    lookup_diff = float(np.abs(la - lb).max()) if la.shape == lb.shape else float("inf")
    occ_a, occ_b = a_loader.occupancy(family_a), b_loader.occupancy(family_b)
    split_equal = a_loader.split["train"] == b_loader.split["train"] and a_loader.split["val"] == b_loader.split["val"]
    pre = {}
    for stem in ("scaler", "pca"):
        try:
            with np.load(a_loader.run_dir / "preprocessing" / f"raw_{stem}.npz") as x,                     np.load(b_loader.run_dir / "preprocessing" / f"raw_{stem}.npz") as y:
                pre[stem] = all(np.array_equal(x[k], y[k]) for k in x.files if k in y.files and k != "n_components_")
        except OSError:
            pre[stem] = False
    occ_equal = bool(np.array_equal(occ_a["counts"], occ_b["counts"]))
    passed = bool(rec_equal and all(arrays.values()) and split_equal and occ_equal and all(pre.values()) and
                  lookup_diff <= LOOKUP_ABS_TOLERANCE)
    return dict(same_recording_names=same_names, recordings_equal=bool(rec_equal), **arrays, occupancy_equal=occ_equal,
                split_equal=bool(split_equal), preprocessing_equal=pre, lookup_max_abs_diff=lookup_diff,
                lookup_bitwise_equal=bool(lookup_diff == 0.0), passed=passed)


def compare_to_reference(info: dict, state_sha: str, eval_dir: Path, ref_info: dict, ref_state_sha: str,
                         ref_eval_dir: Path) -> dict:
    """Gate 2: the HuGaDB seed-111 cell against the calibration run's identical job."""
    mine = {r["rung_steps"]: r for r in json.loads((eval_dir / "ladder.json").read_text())["rows"]}
    ref = {r["rung_steps"]: r for r in json.loads((ref_eval_dir / "ladder.json").read_text())["rows"]}
    diffs = {f"{m}@{rung}": float(mine[rung][m]) - float(ref[rung][m]) for rung in sorted(ref) if rung in mine
             for m in E.METRICS}
    traj = [abs(a["val_loss"] - b["val_loss"]) for a, b in zip(info["history"], ref_info["history"])]
    with np.load(eval_dir / "embeddings.npz") as a, np.load(ref_eval_dir / "embeddings.npz") as b:
        emb = max((float(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max()) for k in b.files
                   if k in a.files and a[k].shape == b[k].shape), default=float("nan"))
    metrics_exact = bool(diffs) and set(ref) == set(mine) and all(v == 0.0 for v in diffs.values())
    step_equal = info["selected_step"] == ref_info["selected_step"]
    return dict(metrics_exact=metrics_exact, selected_step_equal=step_equal, selected_step=info["selected_step"],
                reference_selected_step=ref_info["selected_step"], metric_diffs=diffs,
                validation_max_abs_diff=max(traj) if traj else None, checkpoint_tensors_equal=state_sha == ref_state_sha,
                embeddings_max_abs_diff=emb, f1_50_last_rung=float(mine[max(mine)]["F1_50"]),
                reference_f1_50_last_rung=float(ref[max(ref)]["F1_50"]), passed=bool(metrics_exact and step_equal))


# --- run ---------------------------------------------------------------------------------------

class Run:
    def __init__(self, args, device: torch.device, deadline: float | None):
        self.args, self.device, self.deadline = args, device, deadline
        self.smoke = bool(args.smoke)
        self.scale = scale_for(self.smoke)
        self.run_dir = OUT_ROOT / args.run_id
        self.specs = D.SPECS
        self.cells = make_cells(tuple(args.datasets) if args.datasets else None)
        self.cell_by_id = {c.cell_id: c for c in self.cells}
        self.impl_hash, self.env = source_hash(), R.environment_details(device)
        self.qic_dir = R.resolve_source("qic", args.qic_run_id)
        self.reference_dir = ROOT / "results" / "week5" / "contextualizer_calibration" / args.reference_run
        self.num_codes = 512
        self.cell_identity = {c.cell_id: self.identity_of(c) for c in self.cells}
        self.job_infos: dict[str, dict] = {}
        self.loaders: dict[str, Loader] = {}
        self.verified_evals: set[str] = set()
        self.not_run: dict[str, dict] = {}
        self.timings: dict[str, list] = {}
        self.gates: dict = {}
        self.blocked: dict[str, str] = {}
        self.stop = False

    # -- identity ------------------------------------------------------------------------------
    def cap_for(self, cell: Cell) -> int | None:
        return self.specs[cell.dataset].smoke_cap if self.smoke else None

    def prep_key(self, cell: Cell) -> PREP.PrepKey:
        spec = self.specs[cell.dataset]
        return PREP.PrepKey(spec.name, cell.normalization, cell.seed, spec.window, spec.num_states,
                            num_codes=self.num_codes, recording_cap=self.cap_for(cell))

    def identity_of(self, cell: Cell) -> str:
        return P.stable_hash(dict(job=P.effective_settings(make_job(cell), self.scale), prep=asdict(self.prep_key(cell)),
                                  spec=asdict(self.specs[cell.dataset]), schema=SCHEMA_VERSION))

    def resume_identity(self) -> str:
        return P.stable_hash(dict(schema=SCHEMA_VERSION, scale=asdict(self.scale), cells=self.cell_identity,
                                  environment=self.env, implementation=self.impl_hash,
                                  reference=self.args.reference_run, qic=R.manifest_hash(self.qic_dir)))

    def assert_source_unchanged(self) -> None:
        if source_hash() != self.impl_hash:
            raise RuntimeError("implementation source changed during this invocation; stopping at a unit boundary")

    def write_manifest(self, status: str, extra: dict | None = None) -> None:
        write_json_atomic(self.run_dir / "manifest.json", dict(
            schema_version=SCHEMA_VERSION, run_id=self.args.run_id, status=status, scale=asdict(self.scale),
            resume_identity=self.resume_identity(), implementation_hash=self.impl_hash, environment=self.env,
            cells=len(self.cells), datasets=sorted({c.dataset for c in self.cells}), seeds=list(SEEDS),
            cell_identities=self.cell_identity, reference_run=str(self.reference_dir), qic_run=str(self.qic_dir),
            hours=self.args.hours, final_write_allowance_s=R.FINAL_WRITE_ALLOWANCE_S,
            updated_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **(extra or {})))

    def check_resume(self) -> None:
        manifest = self.run_dir / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text()).get("resume_identity") != self.resume_identity():
                raise SystemExit(f"{manifest} does not match the current sources, configuration, environment or cell "
                                 f"set; refusing to reuse its artifacts. Choose a new --run-id.")
            if not self.args.resume:
                raise SystemExit(f"{manifest} exists; pass --resume or choose a new --run-id")
        elif any((self.run_dir / d).exists() for d in ("jobs", "evals", "prep")):
            raise SystemExit(f"{self.run_dir} has artifacts but no manifest.json; they cannot be verified")

    # -- planning ------------------------------------------------------------------------------
    def plan_rows(self) -> list[dict]:
        rows = []
        for c in self.cells:
            spec = self.specs[c.dataset]
            rows.append(dict(cell_id=c.cell_id, dataset=c.dataset, normalization=c.normalization, seed=c.seed,
                             window_frames=spec.window, token_seconds=round(spec.token_seconds, 4), fps=spec.fps,
                             num_states=spec.num_states, kernel_half_width=max(1, spec.fps // spec.window),
                             recording_cap=self.cap_for(c), planned_decodes="|".join(ROLES),
                             identity=self.cell_identity[c.cell_id]))
        return rows

    # -- bookkeeping ---------------------------------------------------------------------------
    def log_runtime(self, unit: str, kind: str, seconds: float, extra: dict | None = None) -> None:
        entry = dict(unit=unit, kind=kind, seconds=seconds, device=str(self.device),
                     started_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **(extra or {}))
        with (self.run_dir / "runtime_log.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

    def load_timings(self) -> None:
        path = self.run_dir / "runtime_log.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                e = json.loads(line)
                if e.get("projection_key"):
                    self.timings.setdefault(e["projection_key"], []).append(e["seconds"])

    def can_start(self, unit: str, key: str) -> bool:
        if self.stop:
            self.not_run[unit] = dict(unit=unit, status="gate", reason="run stopped before this unit")
            return False
        if self.deadline is None:
            return True
        now, hard = time.perf_counter(), self.deadline - R.FINAL_WRITE_ALLOWANCE_S
        history = self.timings.get(key)
        projected = statistics.median(history) * R.PROJECTION_MARGIN if history else None
        if now >= hard or (projected is not None and now + projected > hard):
            self.not_run[unit] = dict(unit=unit, status="deadline", reason=(
                f"projected {projected:.0f}s would cross the compute deadline" if projected is not None
                else "compute deadline reached before this unit started"))
            return False
        return True

    def hard_deadline(self) -> float | None:
        return None if self.deadline is None else self.deadline - R.FINAL_WRITE_ALLOWANCE_S

    def block(self, cell: Cell, reason: str) -> None:
        self.not_run[cell.cell_id] = dict(unit=cell.cell_id, status="gate", reason=reason)

    # -- preparation -----------------------------------------------------------------------------
    def prep_dir(self, cell: Cell) -> Path:
        return self.run_dir / "prep" / cell.cell_id

    def prepare(self, cell: Cell) -> Loader | None:
        if cell.cell_id in self.loaders:
            return self.loaders[cell.cell_id]
        loader = self.reload(cell)
        if loader is not None:
            return loader
        spec, key = self.specs[cell.dataset], f"prep:{cell.dataset}"
        if not self.can_start(cell.cell_id, key):
            return None
        started = time.perf_counter()
        try:
            PREP.prepare_cell(spec, cell.normalization, cell.seed, self.prep_dir(cell), self.cap_for(cell),
                              self.num_codes)
            loader = Loader(self.prep_dir(cell))
        except Exception as exc:  # noqa: BLE001
            self.not_run[cell.cell_id] = dict(unit=cell.cell_id, status="failed",
                                              reason=f"preparation failed: {type(exc).__name__}: {exc}")
            return None
        seconds = time.perf_counter() - started
        self.log_runtime(f"{cell.cell_id}:prep", "prep", seconds, dict(projection_key=key))
        self.loaders[cell.cell_id] = loader
        return loader

    def reload(self, cell: Cell) -> Loader | None:
        """Load already-verified preparation artifacts. Never builds, and ignores the deadline and stop flag: reading
        verified files is not a unit of work, and a refusal here must never be read as 'unverified'."""
        if cell.cell_id in self.loaders:
            return self.loaders[cell.cell_id]
        try:
            if PREP.is_prepared(self.specs[cell.dataset], cell.normalization, cell.seed, self.prep_dir(cell),
                                self.cap_for(cell), self.num_codes):
                self.loaders[cell.cell_id] = Loader(self.prep_dir(cell))
                return self.loaders[cell.cell_id]
        except (OSError, KeyError, ValueError):
            pass
        return None

    def prep_hash(self, cell: Cell) -> str:
        return R.manifest_hash(self.prep_dir(cell))

    # -- training ------------------------------------------------------------------------------
    def job_dir(self, cell: Cell) -> Path:
        return self.run_dir / "jobs" / cell.cell_id

    def job_state(self, cell: Cell) -> str:
        if cell.cell_id in self.job_infos:
            return "done"
        jd, path = self.job_dir(cell), self.job_dir(cell) / "training_info.json"
        if not path.exists():
            return "incomplete"
        try:
            info = json.loads(path.read_text())
            if info.get("identity") != self.cell_identity[cell.cell_id] or info.get("status") != "done" or \
                    not (jd / "exposure_steps.npz").exists():
                return "incomplete"
            ledger = json.loads((jd / "checkpoints.json").read_text())["ledger"]
            if any(step not in {r["step"] for r in ledger} for step in info["roles"].values()):
                return "incomplete"
            for row in ledger:
                if T.tensor_hash(T.load_state(jd / "ckpt" / row["file"])["state_dict"]) != row["tensor_sha256"]:
                    return "incomplete"
        except (OSError, KeyError, ValueError, json.JSONDecodeError, RuntimeError):
            return "incomplete"
        self.job_infos[cell.cell_id] = info
        return "done"

    def ensure_train(self, cell: Cell) -> bool:
        if cell.cell_id in self.not_run:
            return False
        if self.job_state(cell) == "done":
            return (self.reload(cell) or self.prepare(cell)) is not None  # reload verified artifacts for decoding
        loader = self.prepare(cell)
        if loader is None:
            return False
        if self.stop and self.gates.get("tokenizer_port", {}).get("status") == "failed":
            self.block(cell, "tokenizer port regression failed")
            return False
        if not self.can_start(cell.cell_id, f"train:{cell.dataset}"):
            return False
        jd = self.job_dir(cell)
        if jd.exists():  # interrupted earlier: restart deterministically from initialization
            marker = jd / "train_started.json"
            lost = time.time() - marker.stat().st_mtime if marker.exists() else None
            shutil.rmtree(jd)
            with (self.run_dir / "restarts.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(cell_id=cell.cell_id, lost_seconds=lost, reason="incomplete fit restarted from "
                                         "initialization (exact intra-fit resume not implemented)")) + "\n")
        jd.mkdir(parents=True)
        write_json_atomic(jd / "train_started.json", dict(identity=self.cell_identity[cell.cell_id]))
        print(f"[train] {cell.cell_id} ({self.scale.horizon(P.FULL_HORIZON)} updates) ...", flush=True)
        started = time.perf_counter()
        try:
            info = T.run_training_job(make_job(cell), self.scale, loader, jd, self.device, self.hard_deadline(),
                                      self.cell_identity[cell.cell_id], self.impl_hash)
        except DeadlineExceeded:
            self.not_run[cell.cell_id] = dict(unit=cell.cell_id, status="interrupted",
                                              reason="interrupted at a cooperative deadline check; restarted on --resume")
            self.stop = True
            return False
        except Exception as exc:  # noqa: BLE001
            self.not_run[cell.cell_id] = dict(unit=cell.cell_id, status="failed", reason=f"{type(exc).__name__}: {exc}")
            return False
        write_json_atomic(jd / "training_info.json", info)
        (jd / "train_started.json").unlink(missing_ok=True)
        seconds = time.perf_counter() - started
        self.log_runtime(f"{cell.cell_id}:train", "train", seconds,
                         dict(projection_key=f"train:{cell.dataset}", peak_memory_bytes=info["peak_memory_bytes"]))
        self.timings.setdefault(f"train:{cell.dataset}", []).append(seconds)
        self.job_infos[cell.cell_id] = info
        self.assert_source_unchanged()
        return True

    # -- decoding ------------------------------------------------------------------------------
    def record_path(self, cell: Cell, role: str) -> Path:
        return self.run_dir / "eval_records" / f"{cell.cell_id}__{role}.json"

    def identity_for(self, cell: Cell, state_sha: str) -> str:
        return DEC.eval_identity(state_sha=state_sha, job=make_job(cell), scale=self.scale,
                                 outer_cap=self.scale.outer_cap, spec=self.specs[cell.dataset],
                                 loader=self.loaders[cell.cell_id], names=self.loaders[cell.cell_id].recording_names(),
                                 prep_hash=self.prep_hash(cell))

    def verified_record(self, cell: Cell, role: str) -> dict | None:
        """A completion record only if the job, checkpoint and artifact contents verify now; stale records are removed."""
        path = self.record_path(cell, role)
        if not path.exists():
            return None
        ok, rec = False, None
        try:
            rec = json.loads(path.read_text())
            if self.job_state(cell) != "done":
                path.unlink()
                return None
            if rec.get("status") != "done":
                return None
            if self.reload(cell) is None:
                return None  # cannot verify now (preparation not reloadable): neither trusted nor deleted
            info = self.job_infos[cell.cell_id]
            step = info["roles"].get(role)
            ledger = json.loads((self.job_dir(cell) / "checkpoints.json").read_text())["ledger"]
            row = next((r for r in ledger if r["step"] == step), None)
            spec = self.specs[cell.dataset]
            ok = (row is not None and rec["step"] == step and rec["state_sha"] == row["tensor_sha256"] and
                  rec["identity"] == self.identity_for(cell, rec["state_sha"]) and
                  (rec["identity"] in self.verified_evals or E.eval_complete(
                      self.run_dir / "evals" / rec["identity"], rec["identity"], self.scale,
                      self.loaders[cell.cell_id], spec.num_states)))
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            ok = False
        if not ok:
            path.unlink(missing_ok=True)
            return None
        self.verified_evals.add(rec["identity"])
        return rec

    def ensure_eval(self, cell: Cell, role: str) -> bool:
        key = f"{cell.cell_id}::{role}"
        if not self.ensure_train(cell):
            why = self.not_run.get(cell.cell_id, {})
            self.not_run.setdefault(key, dict(unit=key, status="unavailable", reason=(
                f"cell did not train ({why.get('status', 'not run')}: {why.get('reason', '')})")))
            return False
        if self.verified_record(cell, role) is not None:
            return True
        info = self.job_infos[cell.cell_id]
        step = info["roles"].get(role)
        jd, spec, loader = self.job_dir(cell), self.specs[cell.dataset], self.loaders[cell.cell_id]
        row = next(r for r in json.loads((jd / "checkpoints.json").read_text())["ledger"] if r["step"] == step)
        state = T.load_state(jd / "ckpt" / row["file"])["state_dict"]
        state_sha = T.tensor_hash(state)
        if state_sha != row["tensor_sha256"]:
            self.not_run[key] = dict(unit=key, status="failed", reason="checkpoint hash mismatch")
            return False
        identity = self.identity_for(cell, state_sha)
        eval_dir, path = self.run_dir / "evals" / identity, self.record_path(cell, role)
        record = dict(cell_id=cell.cell_id, role=role, step=step, identity=identity, state_sha=state_sha)
        if path.exists():
            try:
                old = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                old = {}
            if old.get("status") == "invalid" and old.get("identity") == identity:
                self.not_run[key] = dict(unit=key, status="invalid", reason=old.get("reason"))
                return False  # a deterministic invalid solve is a final outcome, not retried on resume
            path.unlink(missing_ok=True)
        if E.eval_complete(eval_dir, identity, self.scale, loader, spec.num_states):
            owner = json.loads((eval_dir / "info.json").read_text())["eval_id"]
            write_json_atomic(path, dict(record, status="done", aliased_of=None if owner == key else owner, seconds=0.0))
            return True
        if not self.can_start(key, f"eval:{cell.dataset}"):
            return False
        if eval_dir.exists():
            shutil.rmtree(eval_dir)
        print(f"[eval] {key} (step {step}) ...", flush=True)
        started = time.perf_counter()
        meta = dict(identity=identity, eval_id=key, job_id=cell.cell_id, family=PREP.FAMILY, step=step, ckpt_role=role,
                    state_sha=state_sha)
        try:
            DEC.decode_state(make_job(cell), self.scale, loader, spec, state, self.scale.outer_cap, self.device,
                             self.hard_deadline(), eval_dir, meta)
        except DeadlineExceeded:
            self.not_run[key] = dict(unit=key, status="interrupted", reason="interrupted at a cooperative deadline check")
            self.stop = True
            return False
        except (E.EvalInvalid, FinalSolveInvalid) as exc:
            write_json_atomic(path, dict(record, status="invalid", reason=str(exc)))
            self.not_run[key] = dict(unit=key, status="invalid", reason=str(exc))
            return False
        except Exception as exc:  # noqa: BLE001
            self.not_run[key] = dict(unit=key, status="failed", reason=f"{type(exc).__name__}: {exc}")
            return False
        seconds = time.perf_counter() - started
        write_json_atomic(path, dict(record, status="done", aliased_of=None, seconds=seconds))
        self.timings.setdefault(f"eval:{cell.dataset}", []).append(seconds)
        self.log_runtime(key, "eval", seconds, dict(projection_key=f"eval:{cell.dataset}"))
        self.assert_source_unchanged()
        return True

    # -- gates -----------------------------------------------------------------------------------
    def gate_labels(self) -> None:
        datasets = sorted({c.dataset for c in self.cells})
        self.gates["label_loading_and_scoring"] = {}
        for ds in datasets:
            try:
                result = gate_label_loading_and_scoring(self.specs[ds])
            except Exception as exc:  # noqa: BLE001
                result = dict(dataset=ds, passed=False, reason=f"{type(exc).__name__}: {exc}")
            self.gates["label_loading_and_scoring"][ds] = result
            if not result["passed"]:
                self.blocked[ds] = "label loading / scoring gate failed"
            print(f"[gate3] {ds}: {'passed' if result['passed'] else 'FAILED'}", flush=True)
        self.save_gates()

    def gate_tokenizer(self, loader: Loader) -> None:
        if self.smoke:
            self.gates["tokenizer_port"] = dict(status="not_applicable_smoke")
            self.save_gates()
            return
        qic = Loader(self.qic_dir)
        result = compare_arrays(loader, qic, PREP.FAMILY, "raw_kmeans")
        self.gates["tokenizer_port"] = dict(status="passed" if result["passed"] else "failed", **result)
        print(f"[gate1] tokenizer port {'passed' if result['passed'] else 'FAILED'}", flush=True)
        if not result["passed"] and not self.args.continue_on_gate_failure:
            self.stop = True
        self.save_gates()

    def gate_pipeline(self) -> None:
        cell = Cell("hugadb", "global", 111)
        if self.smoke:
            self.gates["pipeline_port"] = dict(status="not_applicable_smoke")
            self.save_gates()
            return
        if cell.cell_id not in {c.cell_id for c in self.cells}:
            self.gates["pipeline_port"] = dict(status="not_applicable_dataset_not_selected")
            self.save_gates()
            return
        if self.gates.get("tokenizer_port", {}).get("status") == "failed" and self.job_state(cell) != "done":
            self.gates["pipeline_port"] = dict(status="blocked", reason="tokenizer port gate failed; the cell was not trained")
            self.save_gates()
            return
        rec = self.verified_record(cell, "selected") if cell.cell_id in self.job_infos else None
        try:
            ref_info = json.loads((self.reference_dir / "jobs" / REFERENCE_JOB / "training_info.json").read_text())
            ref_rec = json.loads((self.reference_dir / "eval_records" /
                                  f"{REFERENCE_JOB}__selected__cap50.json").read_text())
            if rec is None:
                raise RuntimeError("the HuGaDB seed-111 cell has no verified validation-selected decode")
            result = compare_to_reference(
                self.job_infos[cell.cell_id], rec["state_sha"], self.run_dir / "evals" / rec["identity"], ref_info,
                ref_info["role_tensor_sha256"]["selected"], self.reference_dir / "evals" / ref_rec["identity"])
        except Exception as exc:  # noqa: BLE001 -- a failed comparison is a failed gate, never a crash
            result = dict(passed=False, reason=f"{type(exc).__name__}: {exc}")
        self.gates["pipeline_port"] = dict(status="passed" if result["passed"] else "failed", **result)
        print(f"[gate2] pipeline port {'passed' if result['passed'] else 'FAILED'}", flush=True)
        if not result["passed"]:
            write_json_atomic(self.run_dir / "gate_mismatch_environment.json", dict(
                environment=self.env, gate=self.gates["pipeline_port"],
                note="Compare device class, batching and numerical modes by hand; a mismatch is not attributed to "
                     "GPU nondeterminism without evidence and no setting is changed to recover the score."))
            if not self.args.continue_on_gate_failure:
                self.stop = True
        self.save_gates()

    def save_gates(self) -> None:
        write_json_atomic(self.run_dir / "gates.json", self.gates)

    def load_previous_gates(self) -> None:
        """Gate outcomes of earlier invocations of this run (the resume identity guarantees identical sources)."""
        path = self.run_dir / "gates.json"
        if path.exists():
            try:
                previous = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                return
            for name in ("tokenizer_port", "pipeline_port"):
                if name in previous and name not in self.gates:
                    self.gates[name] = previous[name]

    def anchor_cell(self) -> Cell | None:
        return next((c for c in self.cells if (c.dataset, c.normalization, c.seed) == ("hugadb", "global", 111)), None)

    def ensure_tokenizer_gate(self) -> None:
        """Gate 1 runs once per run, before anything trains, whenever its outcome is not already recorded."""
        anchor = self.anchor_cell()
        if anchor is None:
            self.gates["tokenizer_port"] = dict(status="not_applicable_dataset_not_selected")
            self.save_gates()
            return
        if self.gates.get("tokenizer_port", {}).get("status") in ("passed", "failed", "not_applicable_smoke"):
            if self.gates["tokenizer_port"]["status"] == "failed" and not self.args.continue_on_gate_failure:
                self.stop = True
            return
        loader = self.prepare(anchor)
        if loader is not None:
            self.gate_tokenizer(loader)

    # -- schedule ----------------------------------------------------------------------------------
    def execute(self) -> None:
        self.load_timings()
        self.load_previous_gates()
        self.gate_labels()
        self.ensure_tokenizer_gate()
        for cell in self.cells:
            if self.stop:
                break
            if cell.dataset in self.blocked:
                self.block(cell, self.blocked[cell.dataset])
                continue
            for role in ROLES:
                if not self.stop:
                    self.ensure_eval(cell, role)
            if cell.dataset == "hugadb" and cell.normalization == "global" and cell.seed == 111:
                self.gate_pipeline()
            self.write_aggregates()
        reason = ("a port-regression gate failed; dependent cells not run" if self.stop and any(
            g.get("status") == "failed" for g in self.gates.values() if isinstance(g, dict)) else
            "not started in this invocation")
        for cell in self.cells:
            for role in ROLES:
                if self.verified_record(cell, role) is None:
                    self.not_run.setdefault(f"{cell.cell_id}::{role}", dict(
                        unit=f"{cell.cell_id}::{role}", status="not_run", reason=reason))

    # -- aggregates -------------------------------------------------------------------------------
    def write_aggregates(self) -> None:
        cell_rows, seg_rows = [], []
        for cell in self.cells:
            if self.job_state(cell) != "done":
                continue
            info = self.job_infos[cell.cell_id]
            ex, spec = info["exposure"], self.specs[cell.dataset]
            row = dict(cell_id=cell.cell_id, dataset=cell.dataset, normalization=cell.normalization, seed=cell.seed,
                       selected_step=info["selected_step"], best_val_loss=info["best_val_loss"],
                       final_step=info["final_step"], replay_stop_step=info["replay_stop_step"],
                       replay_selected_step=info["replay_selected_step"], accepted_updates=ex["accepted_updates"],
                       valid_token_fraction=ex["valid_tokens"] / max(ex["nominal_positions"], 1),
                       eligible_per_update=ex["mean_eligible_per_update"], train_chunks=info["n_train_chunks"],
                       val_chunks=info["n_val_chunks"], train_seconds=info["seconds"], n_parameters=info["n_parameters"])
            diag = self.run_dir / "prep" / cell.cell_id / "diagnostics.json"
            if diag.exists():
                row.update({f"code_{k}": v for k, v in json.loads(diag.read_text()).items()})
            cell_rows.append(row)
            for role in ROLES:
                rec = self.verified_record(cell, role)
                if rec is None:
                    continue
                edir = self.run_dir / "evals" / rec["identity"]
                einfo = json.loads((edir / "info.json").read_text())
                for r in E.ladder_rows(json.loads((edir / "ladder.json").read_text())["rows"]):
                    seg_rows.append(dict(cell_id=cell.cell_id, dataset=cell.dataset, normalization=cell.normalization,
                                         seed=cell.seed, role=role, checkpoint_step=rec["step"],
                                         identity=rec["identity"], aliased_of=rec.get("aliased_of") or "",
                                         ot_status=einfo["ot_status"], outer_iterations=einfo["ot_outer_iterations"],
                                         smoke=self.smoke, num_states=spec.num_states,
                                         **{f"seg_{k}": v for k, v in einfo.get("segment_stats", {}).items()}, **r))
        write_csv(self.run_dir / "cells.csv", cell_rows)
        write_csv(self.run_dir / "segmentation.csv", seg_rows)
        write_csv(self.run_dir / "not_run.csv", list(self.not_run.values()))
        log = self.run_dir / "runtime_log.jsonl"
        write_csv(self.run_dir / "runtime.csv", [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
                  if log.exists() else [])


def ensure_diagnostics(run: Run) -> None:
    """Label-aware code diagnostics per prepared cell (occupancy; code information about subject / action)."""
    from script.week5.raw_kmeans_cross_dataset.prepare import code_diagnostics
    for cell in run.cells:
        path = run.prep_dir(cell) / "diagnostics.json"
        if cell.cell_id in run.loaders and not path.exists():
            spec = run.specs[cell.dataset]
            loader = run.loaders[cell.cell_id]
            write_json_atomic(path, code_diagnostics(spec, loader, D.load_gts(spec, loader.recording_names())))


# --- CLI ----------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Raw K-means tokens across datasets")
    ap.add_argument("--run-id")
    ap.add_argument("--qic-run-id", default="qic_v3")
    ap.add_argument("--reference-run", default="ctx_cal_v1")
    ap.add_argument("--datasets", nargs="+", choices=list(D.SPECS))
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--continue-on-gate-failure", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(int(_CAP))
    if args.preflight:
        from script.week5.raw_kmeans_cross_dataset import tests
        raise SystemExit(0 if tests.run_all() else 1)
    if not args.run_id:
        raise SystemExit("--run-id is required")
    if not (args.smoke or args.dry_run) and not args.hours:
        raise SystemExit("a full run requires an explicit elapsed-time budget: pass --hours")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    deadline = time.perf_counter() + args.hours * 3600.0 if args.hours else None
    run = Run(args, device, deadline)
    if args.dry_run:
        run.run_dir.mkdir(parents=True, exist_ok=True)
        write_csv(run.run_dir / "planned_cells.csv", run.plan_rows())
        run.gate_labels()
        summary = dict(cells=len(run.cells), planned_ot_fits=len(run.cells) * len(ROLES),
                       gates={ds: g["passed"] for ds, g in run.gates["label_loading_and_scoring"].items()},
                       reference_run_present=(run.reference_dir / "jobs" / REFERENCE_JOB / "training_info.json").exists(),
                       qic_run=str(run.qic_dir), implementation_hash=run.impl_hash)
        write_json_atomic(run.run_dir / "dry_run.json", dict(summary, environment=run.env,
                                                             resume_identity=run.resume_identity()))
        print(json.dumps(summary, indent=2))
        return
    run.check_resume()
    run.run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run.run_dir / "planned_cells.csv", run.plan_rows())
    run.write_manifest("running")
    started = time.perf_counter()
    print(f"[cross-dataset] run_dir={run.run_dir} device={device} cells={len(run.cells)} smoke={run.smoke}", flush=True)
    run.execute()
    ensure_diagnostics(run)
    run.write_aggregates()
    run.write_manifest("finished" if not run.not_run else "finished_with_unfinished_units",
                       dict(elapsed_seconds=time.perf_counter() - started, unfinished=len(run.not_run)))
    from script.week5.raw_kmeans_cross_dataset import report
    report.write_report(run.run_dir)
    if args.smoke:
        from script.week5.raw_kmeans_cross_dataset import tests
        ok = tests.verify_run_dir(run.run_dir, run.cells, run.specs)
        print(f"[smoke] verification {'PASSED' if ok else 'FAILED'}")
        raise SystemExit(0 if ok else 1)
    print(f"[cross-dataset] done in {time.perf_counter() - started:.0f}s; unfinished units: {len(run.not_run)}")


if __name__ == "__main__":
    main()
