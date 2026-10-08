"""Contextualizer search calibration: preflight, dry-run, smoke, probe and the
full user-launched run (docs/plans/week5/CONTEXTUALIZER_SEARCH_CALIBRATION/).

Layout under results/week5/contextualizer_calibration/<run_id>/:

    manifest.json                 identities, sources, plan counts, resume identity
    planned_training.csv / planned_evaluations.csv / search_declaration.json / search_rejections.csv
    jobs/<job_id>/                ckpt/step*.pt, checkpoints.json, exposure_steps.npz, training_info.json
    evals/<identity>/             embeddings, prototypes, predictions, ladder (shared by aliased evaluations)
    eval_records/<eval_id>.json   planned evaluation -> identity / alias / status
    training.csv checkpoint_policies.csv segmentation.csv exposure.csv reproduction.csv
    runtime.csv not_run.csv       aggregates rebuilt from the artifacts above (no duplicate rows)
    REPORT.md

Execution is sequential. Order: native-VQ reference, guarded raw reference,
reproduction gate against Task 2, then the remaining jobs, then the solver
sensitivity fits. Nothing is silently shrunk or substituted.
"""

from __future__ import annotations

import os

_CAP = str(min(8, os.cpu_count() or 1))
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_var] = _CAP

import argparse
import hashlib
import json
import platform
import shutil
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from script.action_transport.budget import DeadlineExceeded
from script.fsq_reconstruction.evaluate import FinalSolveInvalid
from script.fsq_reconstruction.experiment import write_csv, write_json_atomic
from script.quantized_input_comparison import ot_arms as ot
from script.quantized_input_comparison.artifacts import Loader
from script.quantized_input_comparison.run_task2 import TASK2_SOURCE_FILES
from script.week5.contextualizer_calibration import evaluate as E
from script.week5.contextualizer_calibration import plan as P
from script.week5.contextualizer_calibration import search_space as S
from script.week5.contextualizer_calibration import train as T

OUT_ROOT = ROOT / "results" / "week5" / "contextualizer_calibration"
SCHEMA_VERSION = 1
FINAL_WRITE_ALLOWANCE_S = 600.0  # reserved for aggregates / report at the end of a budgeted run
PROJECTION_MARGIN = 1.15
OWN_FILES = tuple(f"script/week5/contextualizer_calibration/{n}.py"
                  for n in ("plan", "train", "evaluate", "search_space", "run_calibration", "report", "tests"))
SOURCE_FILES = tuple(TASK2_SOURCE_FILES) + ("script/action_transport/stageb_model.py",) + OWN_FILES
REPRODUCTION_TARGETS = {  # reference job -> (Task 2 cell, declared validation-selected rung-4000 F1@50)
    "native_vq": ("vq_native__context_quantized__seed111", 38.81),
    "raw": ("raw_kmeans__context_id__seed111", 36.08),
}


# --- identity ------------------------------------------------------------------------------

def source_hash() -> str:
    h = hashlib.sha256()
    for name in SOURCE_FILES:
        h.update(name.encode())
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()[:16]


def environment_details(device: torch.device) -> dict:
    import scipy
    import sklearn
    details = dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
                   numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                   threads=int(_CAP), device=str(device), cudnn_deterministic=torch.backends.cudnn.deterministic,
                   cudnn_benchmark=torch.backends.cudnn.benchmark,
                   deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    if device.type == "cuda":
        details.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                       gpu=torch.cuda.get_device_name(device), capability=list(torch.cuda.get_device_capability(device)))
    return details


def resolve_source(kind: str, run_id: str) -> Path:
    base = [ROOT / "results" / "week5" / "quantized_input_comparison", ROOT / "results" / "quantized_input_comparison"]
    candidates = [b / run_id for b in base] if kind == "qic" else [b / "task2" / run_id for b in base]
    for c in candidates:
        if (c / "manifest.json").exists():
            return c
    raise SystemExit(f"{kind} source run {run_id!r} not found; looked in: {[str(c) for c in candidates]}")


def manifest_hash(path: Path) -> str:
    return hashlib.sha256(json.dumps(json.loads((path / "manifest.json").read_text()), sort_keys=True,
                                     default=str).encode()).hexdigest()[:16]


def verify_checkpoints(qic_manifest: dict) -> dict:
    """Recompute the source tokenizer checkpoint hashes and compare with qic's manifest."""
    from script.quantized_input_comparison import config as qcfg
    out = {}
    for arm in ("vq", "fsq"):
        path = qcfg.checkpoint_path(arm)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        out[arm] = dict(path=str(path), sha256=actual, manifest_sha256=qic_manifest["checkpoint_sha256"][arm],
                        match=actual == qic_manifest["checkpoint_sha256"][arm])
    return out


# --- run context ----------------------------------------------------------------------------

class Run:
    def __init__(self, args, scale: P.Scale, device: torch.device, deadline: float | None):
        self.args, self.scale, self.device = args, scale, device
        self.run_dir = OUT_ROOT / args.run_id
        self.qic_dir = resolve_source("qic", args.qic_run_id)
        self.task2_dir = resolve_source("task2", args.task2_run_id)
        self.loader = Loader(self.qic_dir)
        self.jobs = P.enumerate_training_jobs(args.include_combined_target_match)
        self.evals = P.enumerate_evaluations(self.jobs, scale.outer_cap, scale.outer_cap_sensitivity)
        self.job_by_id = {j.job_id: j for j in self.jobs}
        self.impl_hash = source_hash()
        self.env = environment_details(device)
        self.qic_hash, self.task2_hash = manifest_hash(self.qic_dir), manifest_hash(self.task2_dir)
        self.names, self.train_names, self.val_names = T.bounded_split(self.loader, scale.recording_cap)
        self.deadline = deadline
        self.checkpoint_check = verify_checkpoints(self.loader.manifest)
        self.job_identity = {j.job_id: P.job_identity(j, scale) for j in self.jobs}
        self.job_infos: dict[str, dict] = {}
        self.timings: dict[str, list] = {}
        self.not_run: dict[str, dict] = {}
        self.stop = False
        self.verified_evals: set[str] = set()
        self.repro_status = "pending"

    # -- identity / manifest -------------------------------------------------------------
    def resume_identity(self) -> str:
        return P.stable_hash(dict(schema=SCHEMA_VERSION, scale=asdict(self.scale),
                                  combined=self.args.include_combined_target_match, qic=self.qic_hash,
                                  task2=self.task2_hash, split=P.stable_hash([self.train_names, self.val_names]),
                                  environment=self.env,
                                  jobs=self.job_identity, implementation=self.impl_hash))

    def assert_source_unchanged(self) -> None:
        if source_hash() != self.impl_hash:
            raise RuntimeError("implementation source changed during this invocation; stopping at a job boundary")

    def write_manifest(self, status: str, extra: dict | None = None) -> None:
        qic_manifest = self.loader.manifest
        write_json_atomic(self.run_dir / "manifest.json", dict(
            schema_version=SCHEMA_VERSION, run_id=self.args.run_id, status=status, scale=asdict(self.scale),
            resume_identity=self.resume_identity(), implementation_hash=self.impl_hash,
            source_files=list(SOURCE_FILES), environment=self.env, qic_run=str(self.qic_dir), qic_hash=self.qic_hash,
            task2_run=str(self.task2_dir), task2_hash=self.task2_hash,
            checkpoint_hashes=self.checkpoint_check, job_identities=self.job_identity,
            include_combined_target_match=self.args.include_combined_target_match,
            planned=dict(training_jobs=len(self.jobs), planned_ot_fits=len(self.evals),
                         policy_fits=sum(e.kind == "policy" for e in self.evals),
                         sensitivity_fits=sum(e.kind == "outer_sensitivity" for e in self.evals)),
            hours=self.args.hours, final_write_allowance_s=FINAL_WRITE_ALLOWANCE_S,
            updated_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **(extra or {})))

    def require_sources(self) -> None:
        """Source tokenizer checkpoints must match the qic manifest before any training starts."""
        bad = [arm for arm, c in self.checkpoint_check.items() if not c["match"]]
        if bad:
            raise SystemExit(f"source checkpoint hash mismatch for {bad}: {self.checkpoint_check}")

    def check_resume(self) -> None:
        manifest = self.run_dir / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text()).get("resume_identity") != self.resume_identity():
                raise SystemExit(f"{manifest} does not match the current sources, effective configs, split, "
                                 f"optional-job settings or implementation; refusing to reuse its artifacts. "
                                 f"Choose a new --run-id.")
            if not self.args.resume:
                raise SystemExit(f"{manifest} exists; pass --resume or choose a new --run-id")
        elif (self.run_dir / "jobs").exists() or (self.run_dir / "evals").exists():
            raise SystemExit(f"{self.run_dir} has artifacts but no manifest.json; they cannot be verified")

    # -- plan files ------------------------------------------------------------------------
    def write_plan(self) -> dict:
        write_csv(self.run_dir / "planned_training.csv", P.planned_training_rows(self.jobs, self.scale))
        write_csv(self.run_dir / "planned_evaluations.csv", P.planned_evaluation_rows(self.evals, self.jobs))
        declaration = S.declare({n: v[2] for n, v in P.PIPELINES.items()}, ot.FPS, ot.WINDOW, self.loader,
                                {n: v[0] for n, v in P.PIPELINES.items()}, self.train_names, self.val_names)
        rejections = declaration.pop("rejections")
        write_json_atomic(self.run_dir / "search_declaration.json", declaration)
        write_csv(self.run_dir / "search_rejections.csv", [dict(
            pipeline=r["pipeline"], definition_index=r["definition_index"], reason=r["reason"],
            candidate=json.dumps(r["candidate"]), effective=json.dumps(r["effective"]),
            audit=json.dumps(r["audit"])) for r in rejections])
        return dict(n_jobs=len(self.jobs), n_evals=len(self.evals), n_rejections=len(rejections),
                    search_complete={k: v["complete"] for k, v in declaration["pipelines"].items()})

    # -- bookkeeping -----------------------------------------------------------------------
    def log_runtime(self, unit: str, kind: str, seconds: float, projected: float | None, extra: dict) -> None:
        entry = dict(unit=unit, kind=kind, seconds=seconds, projected_seconds=projected, device=str(self.device),
                     started_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)
        with (self.run_dir / "runtime_log.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

    def load_timings(self) -> None:
        path = self.run_dir / "runtime_log.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                e = json.loads(line)
                if e.get("per_unit") is not None:
                    self.timings.setdefault(e["kind"], []).append(e["per_unit"])

    def project(self, kind: str, units: float) -> float | None:
        history = self.timings.get(kind)
        if not history and kind == "eval_cap150" and self.timings.get("eval_cap50"):
            return 3.0 * statistics.median(self.timings["eval_cap50"]) * units * PROJECTION_MARGIN
        return statistics.median(history) * units * PROJECTION_MARGIN if history else None

    def can_start(self, unit: str, kind: str, units: float, status: str = "deadline") -> bool:
        if self.stop:
            self.not_run[unit] = dict(unit=unit, status="gate", reason="run stopped before this unit")
            return False
        if self.deadline is None:
            return True
        now, hard = time.perf_counter(), self.deadline - FINAL_WRITE_ALLOWANCE_S
        projected = self.project(kind, units)
        if now >= hard or (projected is not None and now + projected > hard):
            self.not_run[unit] = dict(unit=unit, status=status, reason=(
                f"projected {projected:.0f}s would cross the compute deadline (remaining "
                f"{max(0.0, hard - now):.0f}s after the final-writing allowance)" if projected is not None
                else "compute deadline reached before this unit started"))
            return False
        return True

    # -- training units ----------------------------------------------------------------
    def job_dir(self, job) -> Path:
        return self.run_dir / "jobs" / job.job_id

    def job_state(self, job) -> str:
        """done | unavailable | incomplete (artifacts verified, not trusted by name)."""
        if job.job_id in self.job_infos:
            return "done"  # verified earlier in this invocation
        jd, path = self.job_dir(job), self.job_dir(job) / "training_info.json"
        if not path.exists():
            return "incomplete"
        try:
            info = json.loads(path.read_text())
            if info.get("identity") != self.job_identity[job.job_id]:
                return "incomplete"
            if info.get("status") == "unavailable":
                return "unavailable"
            if info.get("status") != "done" or not (jd / "exposure_steps.npz").exists():
                return "incomplete"
            ledger = json.loads((jd / "checkpoints.json").read_text())["ledger"]
            steps = {row["step"] for row in ledger}
            if any(step not in steps for step in info["roles"].values()):
                return "incomplete"
            for row in ledger:
                state = T.load_state(jd / "ckpt" / row["file"])["state_dict"]
                if T.tensor_hash(state) != row["tensor_sha256"]:
                    return "incomplete"
        except (OSError, KeyError, ValueError, json.JSONDecodeError, RuntimeError):
            return "incomplete"
        self.job_infos[job.job_id] = info
        return "done"

    def ensure_job(self, job) -> bool:
        if job.job_id in self.not_run:  # failed / interrupted / unavailable earlier in this invocation: never retried
            return False
        state = self.job_state(job)
        if state == "done":
            return True
        if state == "unavailable":
            self.not_run[job.job_id] = dict(unit=job.job_id, status="unavailable", reason=json.loads(
                (self.job_dir(job) / "training_info.json").read_text()).get("reason"))
            return False
        for ref in filter(None, [job.ref_job_id]):
            if self.job_state(self.job_by_id[ref]) != "done":
                self.not_run[job.job_id] = dict(unit=job.job_id, status="not_run",
                                                reason=f"paired reference job {ref} is not complete")
                return False
        steps = self.scale.horizon(job.horizon)
        if not self.can_start(job.job_id, "train", steps):
            return False
        jd = self.job_dir(job)
        if jd.exists():  # interrupted earlier: exact intra-fit resume is not implemented, restart deterministically
            marker = jd / "train_started.json"
            lost = time.time() - marker.stat().st_mtime if marker.exists() else None
            shutil.rmtree(jd)
            with (self.run_dir / "restarts.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(job_id=job.job_id, lost_seconds=lost, reason="incomplete fit restarted "
                                         "from initialization (exact intra-fit resume not implemented)")) + "\n")
        jd.mkdir(parents=True)
        write_json_atomic(jd / "train_started.json", dict(identity=self.job_identity[job.job_id]))
        print(f"[train] {job.job_id} ({steps} updates) ...", flush=True)
        started = time.perf_counter()
        try:
            info = T.run_training_job(job, self.scale, self.loader, jd, self.device, self._hard_deadline(),
                                      self.job_identity[job.job_id], self.impl_hash)
        except DeadlineExceeded:
            self.not_run[job.job_id] = dict(unit=job.job_id, status="interrupted",
                                            reason="interrupted at a cooperative deadline check; partial fit is "
                                                   "restarted from initialization on --resume")
            self.stop = True
            return False
        except T.TargetMatchUnavailable as exc:
            write_json_atomic(jd / "training_info.json", dict(job_id=job.job_id, status="unavailable",
                                                               identity=self.job_identity[job.job_id],
                                                               reason=str(exc)))
            self.not_run[job.job_id] = dict(unit=job.job_id, status="unavailable", reason=str(exc))
            return False
        except Exception as exc:  # noqa: BLE001 -- explicit bookkeeping, never a silent skip
            self.not_run[job.job_id] = dict(unit=job.job_id, status="failed", reason=f"{type(exc).__name__}: {exc}")
            return False
        write_json_atomic(jd / "training_info.json", info)
        (jd / "train_started.json").unlink(missing_ok=True)
        seconds = time.perf_counter() - started
        self.timings.setdefault("train", []).append(seconds / steps)
        self.log_runtime(job.job_id, "train", seconds, None, dict(per_unit=seconds / steps, units=steps,
                         peak_memory_bytes=info["peak_memory_bytes"]))
        self.job_infos[job.job_id] = info
        self.assert_source_unchanged()
        return True

    def _hard_deadline(self) -> float | None:
        return None if self.deadline is None else self.deadline - FINAL_WRITE_ALLOWANCE_S

    # -- evaluation units --------------------------------------------------------------
    def record_path(self, ev) -> Path:
        return self.run_dir / "eval_records" / (ev.eval_id.replace("::", "__") + ".json")

    def ensure_eval(self, ev) -> bool:
        job = self.job_by_id[ev.job_id]
        if job.job_id not in self.job_infos and not self.ensure_job(job):
            why = self.not_run.get(job.job_id, {})
            self.not_run.setdefault(ev.eval_id, dict(unit=ev.eval_id, status="unavailable", reason=(
                f"job {job.job_id} did not complete ({why.get('status', 'not run')}: {why.get('reason', '')})")))
            return False
        info = self.job_infos[job.job_id]
        step = info["roles"].get(ev.ckpt_role)
        if step is None:
            self.not_run[ev.eval_id] = dict(unit=ev.eval_id, status="not_run",
                                            reason=f"role {ev.ckpt_role} undefined for this job")
            return False
        jd = self.job_dir(job)
        row = next(r for r in json.loads((jd / "checkpoints.json").read_text())["ledger"] if r["step"] == step)
        state = T.load_state(jd / "ckpt" / row["file"])["state_dict"]
        state_sha = T.tensor_hash(state)
        if state_sha != row["tensor_sha256"]:
            self.not_run[ev.eval_id] = dict(unit=ev.eval_id, status="failed", reason="checkpoint hash mismatch")
            return False
        identity = E.eval_identity(state_sha=state_sha, job=job, scale=self.scale, outer_cap=ev.outer_cap,
                                   qic_hash=self.qic_hash, loader=self.loader, names=self.names)
        eval_dir, record_path = self.run_dir / "evals" / identity, self.record_path(ev)
        record = dict(eval_id=ev.eval_id, job_id=job.job_id, ckpt_role=ev.ckpt_role, step=step, identity=identity,
                      outer_cap=ev.outer_cap, kind=ev.kind, state_sha=state_sha)
        if record_path.exists():
            old = json.loads(record_path.read_text())
            if old.get("identity") == identity and old.get("status") == "invalid":
                self.not_run[ev.eval_id] = dict(unit=ev.eval_id, status="invalid", reason=old.get("reason"))
                return False
            if old.get("identity") == identity and old.get("status") == "done" and \
                    E.eval_complete(eval_dir, identity, self.scale, self.loader):
                return True
            record_path.unlink()
        if E.eval_complete(eval_dir, identity, self.scale, self.loader):
            owner = json.loads((eval_dir / "info.json").read_text())["eval_id"]
            write_json_atomic(record_path, dict(record, status="done", aliased_of=None if owner == ev.eval_id
                                                else owner, seconds=0.0))
            return True
        kind = "eval_cap150" if ev.outer_cap > self.scale.outer_cap else "eval_cap50"
        if not self.can_start(ev.eval_id, kind, 1.0):
            return False
        if eval_dir.exists():
            shutil.rmtree(eval_dir)
        print(f"[eval] {ev.eval_id} (step {step}, outer cap {ev.outer_cap}) ...", flush=True)
        started = time.perf_counter()
        meta = dict(identity=identity, eval_id=ev.eval_id, job_id=job.job_id, family=job.family, step=step,
                    ckpt_role=ev.ckpt_role, state_sha=state_sha)
        try:
            E_info = E.decode_state(job, self.scale, self.loader, state, ev.outer_cap, self.device,
                                    self._hard_deadline(), eval_dir, meta)
        except DeadlineExceeded:
            self.not_run[ev.eval_id] = dict(unit=ev.eval_id, status="interrupted",
                                            reason="interrupted at a cooperative deadline check")
            self.stop = True
            return False
        except (E.EvalInvalid, FinalSolveInvalid) as exc:
            write_json_atomic(record_path, dict(record, status="invalid", reason=str(exc)))
            self.not_run[ev.eval_id] = dict(unit=ev.eval_id, status="invalid", reason=str(exc))
            return False
        except Exception as exc:  # noqa: BLE001
            self.not_run[ev.eval_id] = dict(unit=ev.eval_id, status="failed", reason=f"{type(exc).__name__}: {exc}")
            return False
        seconds = time.perf_counter() - started
        write_json_atomic(record_path, dict(record, status="done", aliased_of=None, seconds=seconds))
        self.timings.setdefault(kind, []).append(seconds)
        self.log_runtime(ev.eval_id, kind, seconds, None, dict(per_unit=seconds, units=1,
                         extract_seconds=E_info["extract_seconds"]))
        self.assert_source_unchanged()
        return True

    def verified_record(self, ev) -> dict | None:
        """The completion record of a planned evaluation, only if its job, checkpoint and artifacts verify
        now. A stale completion record (job or artifacts no longer verifiable) is removed, never trusted."""
        path = self.record_path(ev)
        if not path.exists():
            return None
        job, ok, rec = self.job_by_id[ev.job_id], False, None
        try:
            rec = json.loads(path.read_text())
            job_ok = self.job_state(job) == "done"
            if rec.get("status") != "done":
                if not job_ok:
                    path.unlink()
                return None
            if job_ok:
                step = self.job_infos[job.job_id]["roles"].get(ev.ckpt_role)
                ledger = json.loads((self.job_dir(job) / "checkpoints.json").read_text())["ledger"]
                row = next((r for r in ledger if r["step"] == step), None)
                ok = (row is not None and rec["step"] == step and rec["state_sha"] == row["tensor_sha256"] and
                      rec["identity"] == E.eval_identity(state_sha=rec["state_sha"], job=job, scale=self.scale,
                                                         outer_cap=ev.outer_cap, qic_hash=self.qic_hash,
                                                         loader=self.loader, names=self.names) and
                      (rec["identity"] in self.verified_evals or E.eval_complete(
                          self.run_dir / "evals" / rec["identity"], rec["identity"], self.scale, self.loader)))
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            ok = False
        if not ok:
            path.unlink(missing_ok=True)
            return None
        self.verified_evals.add(rec["identity"])
        return rec

    # -- reproduction gate -------------------------------------------------------------
    def reproduction_gate(self) -> None:
        if self.scale.smoke:
            self.repro_status = "not_applicable_smoke"
            write_json_atomic(self.run_dir / "reproduction.json", dict(status=self.repro_status, rows=[],
                                                                       note="smoke budgets cannot reproduce Task 2"))
            return
        rows = []
        for pipeline, (cell, target) in REPRODUCTION_TARGETS.items():
            job = self.job_by_id[P.make_job_id(pipeline, 128, 8, P.PIPELINES[pipeline][2] if pipeline == "native_vq"
                                               else P.RAW_GUARDED_RADIUS, P.FULL_HORIZON, P.REFERENCE_SEED, "none")]
            ev = next(e for e in self.evals if e.job_id == job.job_id and e.ckpt_role == "selected")
            info = self.job_infos.get(job.job_id)
            rec = self.verified_record(ev)
            row = dict(pipeline=pipeline, job_id=job.job_id, task2_cell=cell, declared_target_f1_50=target)
            if info is None or rec is None:
                row.update(passed=False, reason="reference job or its verified validation-selected decode did not "
                                                "complete: " + str(self.not_run.get(ev.eval_id) or
                                                                   self.not_run.get(job.job_id) or "not run"))
            else:
                try:
                    cmp_ = E.compare_to_task2(info, info["role_tensor_sha256"]["selected"],
                                              self.run_dir / "evals" / rec["identity"], self.task2_dir / "cells" / cell)
                    cmp_["declared_target_met"] = abs(cmp_["f1_50_rung_4000"] - target) < 5e-3
                    cmp_["passed"] = cmp_["passed"] and cmp_["declared_target_met"]
                    row.update(cmp_)
                except Exception as exc:  # noqa: BLE001 -- a failed comparison is a failed gate, not a crash
                    row.update(passed=False, reason=f"comparison failed: {type(exc).__name__}: {exc}")
            rows.append(row)
        passed = all(r.get("passed") for r in rows)
        self.repro_status = "passed" if passed else "failed"
        write_json_atomic(self.run_dir / "reproduction.json", dict(status=self.repro_status, rows=rows,
                                                                   environment=self.env, task2_run=str(self.task2_dir)))
        if not passed:
            write_json_atomic(self.run_dir / "reproduction_mismatch_environment.json", dict(
                environment=self.env, note="Task 2 did not record its own device/package details; compare device "
                "class, batching and numerical modes by hand. Mismatch is not attributed to GPU nondeterminism "
                "without evidence.", rows=rows))
            if not self.args.continue_on_repro_mismatch:
                self.stop = True
                print("[gate] reproduction FAILED; stopping before dependent cells "
                      "(--continue-on-repro-mismatch overrides)", flush=True)
        print(f"[gate] reproduction {self.repro_status}", flush=True)

    # -- main schedule -----------------------------------------------------------------
    def execute(self) -> None:
        self.load_timings()
        native_ref, guard_ref = self.jobs[0], self.jobs[1]
        selected = lambda job: next(e for e in self.evals if e.job_id == job.job_id and e.ckpt_role == "selected")
        for job in (native_ref, guard_ref):
            if self.ensure_job(job):
                self.ensure_eval(selected(job))
        self.reproduction_gate()
        for job in self.jobs:
            if self.stop:
                break
            self.ensure_job(job)
            for ev in self.evals:
                if ev.job_id == job.job_id and ev.kind == "policy" and not self.stop:
                    self.ensure_eval(ev)
            self.write_aggregates()
        for ev in self.evals:
            if ev.kind == "outer_sensitivity" and not self.stop:
                self.ensure_eval(ev)
        reason = ("reproduction gate failed; dependent cells not run" if self.repro_status == "failed"
                  else "not started in this invocation")
        for ev in self.evals:  # explicit reasons for everything unfinished
            if self.verified_record(ev) is None:
                self.not_run.setdefault(ev.eval_id, dict(unit=ev.eval_id, status="not_run", reason=reason))
        for job in self.jobs:
            if self.job_state(job) != "done":
                self.not_run.setdefault(job.job_id, dict(unit=job.job_id, status="not_run", reason=reason))

    # -- aggregates ----------------------------------------------------------------------
    def write_aggregates(self) -> None:
        run_dir = self.run_dir
        train_rows, policy_rows, exposure_rows, seg_rows = [], [], [], []
        for job in self.jobs:
            if self.job_state(job) != "done":
                continue
            info = self.job_infos[job.job_id]
            flat = {k: v for k, v in info.items() if k not in ("history", "exposure", "roles", "role_tensor_sha256")}
            train_rows.append(dict(flat, group=job.group, config_name=job.config_name,
                                   accepted_updates=info["exposure"]["accepted_updates"]))
            ledger = {r["step"]: r for r in json.loads((self.job_dir(job) / "checkpoints.json").read_text())["ledger"]}
            first_role: dict[int, str] = {}
            for role, step in info["roles"].items():
                first_role.setdefault(step, role)
                policy_rows.append(dict(job_id=job.job_id, role=role, step=step, val_loss=ledger[step]["val_loss"],
                                        file=ledger[step]["file"], tensor_sha256=ledger[step]["tensor_sha256"],
                                        alias_of=first_role[step] if first_role[step] != role else "",
                                        replay_stop_step=info["replay_stop_step"],
                                        replay_selected_step=info["replay_selected_step"],
                                        selected_step=info["selected_step"], final_step=info["final_step"]))
            exposure_rows.append(dict(job_id=job.job_id, group=job.group, **{
                k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in info["exposure"].items()}))
        for ev in self.evals:
            rec = self.verified_record(ev)
            if rec is None:
                continue
            job = self.job_by_id[ev.job_id]
            eval_dir = run_dir / "evals" / rec["identity"]
            if not (eval_dir / "ladder.json").exists():
                continue
            info = json.loads((eval_dir / "info.json").read_text())
            for r in E.ladder_rows(json.loads((eval_dir / "ladder.json").read_text())["rows"]):
                seg_rows.append(dict(
                    eval_id=ev.eval_id, job_id=job.job_id, group=job.group, pipeline=job.pipeline,
                    config_name=job.config_name, context=job.context, mask_span=job.mask_span,
                    guard_radius=job.guard_radius, horizon=job.horizon, seed=job.seed,
                    target_count=job.target_count, ckpt_role=ev.ckpt_role, checkpoint_step=rec["step"],
                    outer_cap=ev.outer_cap, kind=ev.kind, identity=rec["identity"],
                    aliased_of=rec.get("aliased_of") or "", ot_status=info["ot_status"],
                    outer_iterations=info["ot_outer_iterations"], repro_gate=self.repro_status,
                    smoke=self.scale.smoke, **r))
        write_csv(run_dir / "training.csv", train_rows)
        write_csv(run_dir / "checkpoint_policies.csv", policy_rows)
        write_csv(run_dir / "exposure.csv", exposure_rows)
        write_csv(run_dir / "segmentation.csv", seg_rows)
        write_csv(run_dir / "not_run.csv", list(self.not_run.values()))
        runtime = []
        log = run_dir / "runtime_log.jsonl"
        if log.exists():
            runtime = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        write_csv(run_dir / "runtime.csv", runtime)
        repro = run_dir / "reproduction.json"
        if repro.exists():
            rows = json.loads(repro.read_text()).get("rows", [])
            write_csv(run_dir / "reproduction.csv", [dict(
                {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()}) for r in rows])


# --- probe ------------------------------------------------------------------------------------

PROBE_CONFIG = dict(width=256, layers=8, heads=4, d_ff=1024, context=256, stride=128, batch=32, updates=100)


def run_probe(args, device: torch.device, loader=None, config: dict | None = None, out_dir: Path | None = None) -> dict:
    """Width-256 / 8-layer / context-256 memory and throughput probe (native-VQ
    vector input, 100 updates, no OT). A short timing is an extrapolation, not
    a bound or a convergence certificate."""
    from script.action_transport.stageb_model import CodeContextModel, ModelConfig
    from script.fsq_reconstruction.quantizers import isolated_rng
    from script.quantized_input_comparison import contextual as C
    pc = config or PROBE_CONFIG
    loader = loader or Loader(resolve_source("qic", args.qic_run_id))
    names, train_names, val_names = T.bounded_split(loader, None)
    family, ctx, stride, batch = "vq_native", pc["context"], pc["stride"], pc["batch"]
    lookup, rms = loader.lookup(family), float(loader.rms(family))
    cfg = C.base_train_config(max_steps=pc["updates"], eval_every=pc["updates"], patience_steps=10**9,
                              batch_size=batch, warmup_steps=10)
    result = dict(config=dict(width=pc["width"], layers=pc["layers"], heads=pc["heads"], d_ff=pc["d_ff"], context=ctx,
                              stride=stride, effective_batch=batch, updates=pc["updates"], family=family,
                              input="context_quantized", guard_radius=14, micro_chunks=args.probe_micro_chunks),
                  environment=environment_details(device), status="ok",
                  note="Short timing is an extrapolation, not a bound or a convergence certificate; no OT, no "
                       "performance ranking.")
    try:
        train_chunks = C.make_vector_chunks(loader, family, train_names, chunk=ctx, stride=stride)
        val_chunks = C.make_vector_chunks(loader, family, val_names, chunk=ctx, stride=stride)
        with isolated_rng(111):
            model = CodeContextModel(ModelConfig(
                num_codes=lookup.shape[0], d_model=pc["width"], n_layers=pc["layers"], n_heads=pc["heads"],
                d_ff=pc["d_ff"], dropout=cfg.dropout,
                norm_first=cfg.norm_first, tie_head=cfg.tie_head, position_mode=cfg.position_mode,
                input_mode="vector", input_dim=lookup.shape[1])).to(device)
        res = T.train_core(arm="context_quantized", train_chunks=train_chunks, val_chunks=val_chunks,
                           num_codes=lookup.shape[0], input_dim=lookup.shape[1], seed=111, cfg=cfg, init=None,
                           device=device, lookup=lookup, rms=rms, guard_radius=14, micro_chunks=args.probe_micro_chunks,
                           model=model)
        seconds = res.arrays["seconds"]
        tokens = batch * ctx
        result.update(
            parameters=sum(p.numel() for p in model.parameters()), accepted_updates=res.stats["accepted_updates"],
            first10_seconds_per_update=float(seconds[:10].mean()),
            later_seconds_per_update=float(seconds[10:].mean()) if len(seconds) > 10 else None,
            first10_tokens_per_second=tokens / float(seconds[:10].mean()),
            later_tokens_per_second=tokens / float(seconds[10:].mean()) if len(seconds) > 10 else None,
            total_seconds_including_validation=res.seconds, peak_memory_bytes=res.peak_memory_bytes,
            validation_loss=res.history[-1]["val_loss"], eligible_per_update=res.stats["mean_eligible_per_update"])
        started = time.perf_counter()
        model.eval()
        windows = 0
        for n in names[:20]:
            emb, _ = E.extract_embeddings("context_quantized", model, loader, family, n, lookup, rms, device, ctx, stride)
            windows += len(emb)
        elapsed = time.perf_counter() - started
        result.update(extraction_windows=windows, extraction_seconds=elapsed, extraction_windows_per_second=windows / elapsed)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        result.update(status="oom", error=str(exc)[:300])  # recorded, never silently repaired
    out = out_dir or OUT_ROOT / args.run_id / "probe"
    write_json_atomic(out / "probe_result.json", result)
    if out_dir is None:
        print(json.dumps(result, indent=2, default=str))
    return result


# --- CLI --------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Contextualizer search calibration")
    ap.add_argument("--run-id")
    ap.add_argument("--qic-run-id", default="qic_v3")
    ap.add_argument("--task2-run-id", default="task2_v1")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--probe-micro-chunks", type=int, default=None)
    ap.add_argument("--include-combined-target-match", action="store_true")
    ap.add_argument("--continue-on-repro-mismatch", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(int(_CAP))

    if args.preflight:
        from script.week5.contextualizer_calibration import tests
        raise SystemExit(0 if tests.run_all() else 1)
    if not args.run_id:
        raise SystemExit("--run-id is required")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if args.probe:
        run_probe(args, device)
        return
    scale = P.SMOKE if args.smoke else P.FULL
    if not (args.smoke or args.dry_run) and not args.hours:
        raise SystemExit("a full run requires an explicit elapsed-time budget: pass --hours")
    deadline = time.perf_counter() + args.hours * 3600.0 if args.hours else None
    run = Run(args, scale, device, deadline)
    if args.dry_run:
        summary = run.write_plan()
        checks = run.checkpoint_check
        missing = [c for c, _ in REPRODUCTION_TARGETS.values() if not (run.task2_dir / "cells" / c / "info.json").exists()]
        write_json_atomic(run.run_dir / "dry_run.json", dict(
            summary=summary, checkpoint_hashes=checks, missing_task2_reproduction_cells=missing,
            implementation_hash=run.impl_hash, resume_identity=run.resume_identity(), environment=run.env,
            qic_run=str(run.qic_dir), task2_run=str(run.task2_dir)))
        print(json.dumps(dict(summary, checkpoint_hashes_match=all(c["match"] for c in checks.values()),
                              missing_task2_cells=missing, qic=str(run.qic_dir), task2=str(run.task2_dir)), indent=2))
        run.require_sources()
        return
    run.require_sources()
    run.check_resume()
    run.run_dir.mkdir(parents=True, exist_ok=True)
    plan_summary = run.write_plan()
    run.write_manifest("running", dict(plan_summary=plan_summary))
    started = time.perf_counter()
    print(f"[calibration] run_dir={run.run_dir} device={device} scale={scale.name} jobs={len(run.jobs)} "
          f"planned_ot_fits={len(run.evals)}", flush=True)
    run.execute()
    run.write_aggregates()
    run.write_manifest("finished" if not run.not_run else "finished_with_unfinished_units",
                       dict(elapsed_seconds=time.perf_counter() - started, unfinished=len(run.not_run)))
    from script.week5.contextualizer_calibration import report
    report.write_report(run.run_dir)
    if args.smoke:
        from script.week5.contextualizer_calibration import tests
        ok = tests.verify_run_dir(run.run_dir, run.jobs, run.evals)
        print(f"[smoke] verification {'PASSED' if ok else 'FAILED'}")
        raise SystemExit(0 if ok else 1)
    print(f"[calibration] done in {time.perf_counter() - started:.0f}s; unfinished units: {len(run.not_run)}")


if __name__ == "__main__":
    main()
