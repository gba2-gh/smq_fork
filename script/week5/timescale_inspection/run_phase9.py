"""Phase 9 of the time-scale inspection: iterative contextual target refinement (a diagnostic).

Arms (span 2, T2 tokens; `R` = ts5_v1's `context` unit, read):
  g    `context_geometry_init`  code identity, free table initialized to W(q_c/RMS)+b
  i1q  `context_quantized`      the code's centroid vector / RMS through a trainable projection
  i1   `context_continuous`     the window's own T2 PCA-64 features / RMS, T2-code targets
  i2   `context_continuous`     the same input, targets = K-means-512 of I1's layer-2 states (iteration 2)

Per dataset: A (g, i1q, i1 x seeds, T2-code grouping), B (I1's per-layer states -> K-means -> the I2 family, layer
diagnostics), C (I2 x seeds with both groupings; I1 with the I2-code grouping). `--stage P` is the separate prospective
BABEL-1 run (i1, i2 only). No labels enter training, layer choice, the iteration count or any stopping rule.

Design and declared rules: docs/plans/week5/TIMESCALE_INSPECTION/PHASE9_LEARNED_UNITS.md (Revision 3). No shared module
and no earlier phase file is edited; `trainer9.py` is the gated copy of the Phase 5 trainer.

Usage:
  python -m script.week5.timescale_inspection.run_phase9 --run-id ts5_v1 --hours 10
  python -m script.week5.timescale_inspection.run_phase9 --run-id ts9_dry --dry-run
  python -m script.week5.timescale_inspection.run_phase9 --run-id ts5_v1 --stage P --hours 5
  python -m script.week5.timescale_inspection.run_phase9 --run-id ts9_smoke --source-run ts5_v1 --smoke
"""

from __future__ import annotations

import os

_CAP = str(min(8, os.cpu_count() or 8))
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_var] = _CAP

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass, replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from script.action_transport import categorical, continuous  # noqa: E402
from script.action_transport.budget import DeadlineExceeded  # noqa: E402
from script.fsq_reconstruction.evaluate import FinalSolveInvalid  # noqa: E402
from script.fsq_reconstruction.experiment import write_json_atomic  # noqa: E402
from script.quantized_input_comparison import artifacts  # noqa: E402
from script.quantized_input_comparison import contextual as C  # noqa: E402
from script.quantized_input_comparison.artifacts import Loader  # noqa: E402
from script.week5.contextualizer_calibration import evaluate as E  # noqa: E402
from script.week5.contextualizer_calibration import plan as P  # noqa: E402
from script.week5.contextualizer_calibration import train as T  # noqa: E402
from script.week5.raw_kmeans_cross_dataset import datasets as D  # noqa: E402
from script.week5.timescale_inspection import radius as RAD  # noqa: E402
from script.week5.timescale_inspection import run_phase1 as R1  # noqa: E402
from script.week5.timescale_inspection import run_phase5 as R5  # noqa: E402
from script.week5.timescale_inspection import run_phase7 as R7  # noqa: E402
from script.week5.timescale_inspection import tokenizers as TOK  # noqa: E402
from script.week5.timescale_inspection import trainer9 as T9  # noqa: E402
from script.week5.timescale_inspection import units9 as U9  # noqa: E402

SCHEMA_VERSION = 1
SEEDS = (111, 222, 1538574472)
DEV_DATASETS = ("babel2", "babel3", "lara")
ARMS = ("g", "i1q", "i1", "i2")
ARM_JOB = {"g": "context_geometry_init", "i1q": "context_quantized", "i1": "context_continuous", "i2": "context_continuous"}
TOK9 = {"babel1": "t2_relational_b1", "babel2": "t2_relational", "babel3": "t2_relational", "lara": "t2_relational"}
OWN_FILES = R7.OWN_FILES + tuple(f"script/week5/timescale_inspection/{m}.py" for m in ("trainer9", "units9", "run_phase9"))
COST = {k: dict(v) for k, v in R7.COST.items()}
COST["context"]["babel1"] = 20.0  # measured in phase 7 (all BABEL-1 decodes took ~20 min)
LAYER_FIT_MIN = dict(babel1=15, babel2=10, babel3=10, lara=25)  # budget for stage B per dataset (minutes)
GATE_MIN = 15.0


@dataclass(frozen=True)
class Cell9:
    dataset: str
    arm: str
    seed: int

    @property
    def tok(self) -> str:
        return TOK9[self.dataset]

    @property
    def spec(self) -> TOK.TokenizerSpec:
        return TOK.SPECS[self.tok]

    @property
    def window(self) -> int:
        return D.SPECS[self.dataset].window

    @property
    def span(self) -> int:
        return 2

    @property
    def guard(self) -> int:
        return 0

    @property
    def cell_id(self) -> str:
        return f"{self.dataset}__{self.arm}__seed{self.seed}"

    @property
    def prep_id(self) -> str:
        return f"{self.dataset}__{self.tok}__seed{self.seed}"


@dataclass(frozen=True)
class Item9:
    cell: Cell9
    tags: tuple[str, ...]  # decodes: "tg" (T2-code grouping) / "ig" (I2-code grouping)
    train: bool = True


def source_hash() -> str:
    h = hashlib.sha256()
    for name in OWN_FILES + R5.TRAIN_FILES:
        h.update(name.encode())
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()[:16]


def stage_items(dataset: str, stage: str, seeds=SEEDS, prospective: bool = False) -> list[Item9]:
    """A: train + T2-grouping decode (prospective BABEL-1 runs I1 only). C: I2 with both groupings (BABEL-1: the I2-code
    grouping only) and I1 re-decoded with the I2-code grouping."""
    if stage == "A":
        arms = ("i1",) if prospective else ("g", "i1q", "i1")
        return [Item9(Cell9(dataset, a, s), ("tg",)) for s in seeds for a in arms]
    if stage == "C":
        out = []
        for s in seeds:
            out.append(Item9(Cell9(dataset, "i2", s), ("ig",) if prospective else ("tg", "ig")))
            out.append(Item9(Cell9(dataset, "i1", s), ("ig",), train=False))
        return out
    return []


class Phase9Runner(R7.Phase7Runner):
    """Does not call `Phase7Runner.__init__`. Reuses unchanged: `expired`, `device`, `_gate`, `unit_dir`, `unit_done` (a
    unit needs its saved predictions), `selected_state`, `inherit_phase5_gate`."""

    Cell9 = Cell9  # for gates9, which must not import this module (run with -m it would load a second copy)

    def __init__(self, args):
        self.args = args
        self.smoke = bool(args.smoke)
        self.stage = args.stage
        self.prospective = self.stage == "P"
        self.datasets = ("babel1",) if self.prospective else DEV_DATASETS
        self.scale = replace(P.SMOKE, recording_cap=None) if self.smoke else P.FULL
        base = ROOT / "results" / "week5" / "timescale_inspection"
        self.out = base / args.run_id / ("phase9p" if self.prospective else "phase9")
        for sub in ("prep", "jobs", "units", "gates", "i2", "diagnostics"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.source_root = base / args.source_run
        self.phase5_dir = self.source_root / "phase5"
        self.deadline = time.perf_counter() + args.hours * 3600 if args.hours else None
        self.impl = source_hash()
        self.radius = RAD.CURRENT
        self.seeds = (111,) if self.smoke else SEEDS
        self.job_infos, self.loaders, self.not_run, self.families = {}, {}, {}, {}
        self.gates_path = self.out / "gates.json"
        self.gates = json.loads(self.gates_path.read_text()) if self.gates_path.exists() else {}
        self.manifest_path = self.out / "manifest.json"
        self.base_cells = [it.cell for ds in self.datasets for st in ("A", "C") for it in self.items(ds, st)]
        self.base_cells = list({c.cell_id: c for c in self.base_cells}.values())
        self.check_resume()

    def items(self, dataset: str, stage: str) -> list[Item9]:
        return stage_items(dataset, stage, self.seeds, self.prospective)

    # -- identity ------------------------------------------------------------------------------------------------------
    def recording_cap(self, cell) -> int | None:
        return D.SPECS[cell.dataset].smoke_cap if self.smoke else None

    def make_job(self, cell: Cell9) -> P.JobSpec:
        context, _, stride, batch = P.CONFIGS["reference"]
        return P.JobSpec(cell.cell_id, "main", "reference", f"tok5_{cell.tok}", cell.spec.family, ARM_JOB[cell.arm], 0,
                         context, stride, cell.span, 0.30, batch, P.FULL_HORIZON, cell.seed, "none", "")

    def identity_of(self, cell: Cell9) -> str:
        key = TOK.make_key(cell.dataset, cell.tok, cell.seed, cell.window, self.recording_cap(cell))
        dep = f"i2 targets: layer {U9.LAYER} of {cell.dataset}__i1__seed{cell.seed}, K={U9.NUM_CODES}" if cell.arm == "i2" else None
        return P.stable_hash(dict(job=P.effective_settings(self.make_job(cell), self.scale), prep=asdict(key), arm=cell.arm,
                                  spec=asdict(TOK.dataset_spec(cell.dataset, cell.window)), dep=dep, schema=SCHEMA_VERSION))

    def check_resume(self) -> None:
        ident = dict(schema=SCHEMA_VERSION, implementation_hash=self.impl, scale=self.scale.name, stage=self.stage,
                     outer_cap=self.scale.outer_cap, rungs=list(self.scale.rungs), radius=self.radius.name,
                     cells={c.cell_id: self.identity_of(c) for c in self.base_cells}, source_run=self.args.source_run,
                     seeds=list(self.seeds), layer=U9.LAYER)
        if self.manifest_path.exists():
            old = json.loads(self.manifest_path.read_text())
            if old.get("identity") != ident:
                raise SystemExit(f"{self.out} was written by a different implementation or cell set; choose a new --run-id.")
        from script.week5.timescale_inspection import report9  # local: report9 imports this module
        write_json_atomic(self.manifest_path, dict(
            identity=ident, run_id=self.args.run_id, phase="9p" if self.prospective else 9, status="running", smoke=self.smoke,
            analysis_hash_at_launch=report9.analysis_hash(), python=platform.python_version(),
            started_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"), radius=self.radius.name))
        self.ident = ident

    # -- preparations ---------------------------------------------------------------------------------------------------
    def prep_candidates(self, cell: Cell9) -> list[Path]:
        if cell.dataset == "babel1":
            return [self.source_root / "phase7" / "prep" / cell.tok / cell.prep_id]
        return [self.phase5_dir / "prep" / cell.tok / cell.prep_id]

    def base_loader(self, cell: Cell9) -> Loader | None:
        """T2's preparation, read from ts5_v1 (never rebuilt); under --smoke a capped copy is built into this run."""
        if cell.prep_id in self.loaders:
            return self.loaders[cell.prep_id]
        cap = self.recording_cap(cell)
        for d in ([] if self.smoke else self.prep_candidates(cell)):
            try:
                if TOK.is_prepared(cell.dataset, cell.tok, cell.seed, d, cell.window, cap):
                    self.loaders[cell.prep_id] = Loader(d)
                    return self.loaders[cell.prep_id]
            except (OSError, KeyError, ValueError):
                pass
        if not self.smoke:
            return None
        d = self.out / "prep" / cell.tok / cell.prep_id
        if not TOK.is_prepared(cell.dataset, cell.tok, cell.seed, d, cell.window, cap):
            TOK.prepare_cell(cell.dataset, cell.tok, cell.seed, d, cell.window, cap, self.device)
        self.loaders[cell.prep_id] = Loader(d)
        return self.loaders[cell.prep_id]

    def i2_dir(self, dataset: str, seed: int) -> Path:
        return self.out / "i2" / f"{dataset}__i2__seed{seed}"

    def teacher_sha(self, dataset: str, seed: int) -> str | None:
        info = self.job_state(Cell9(dataset, "i1", seed))
        return info["role_tensor_sha256"]["selected"] if info else None

    def family(self, dataset: str, seed: int) -> U9.I2Source | None:
        """The I2 preparation family, if it was built from the current I1 checkpoint."""
        if (dataset, seed) in self.families:
            return self.families[(dataset, seed)]
        d = self.i2_dir(dataset, seed)
        cell = Cell9(dataset, "i1", seed)
        base = self.base_loader(cell)
        sha = self.teacher_sha(dataset, seed)
        if base is None or sha is None or not (d / "meta.json").exists():
            return None
        try:
            src = U9.load_family(base, cell.spec.family, d)
        except (OSError, KeyError, ValueError):
            return None
        if src.meta.get("teacher_sha") != sha:
            return None
        self.families[(dataset, seed)] = src
        return src

    def train_loader(self, cell: Cell9):
        return self.family(cell.dataset, cell.seed) if cell.arm == "i2" else self.base_loader(cell)

    # -- training ---------------------------------------------------------------------------------------------------------
    def job_dir(self, cell: Cell9) -> Path:
        return self.out / "jobs" / cell.cell_id

    def job_state(self, cell: Cell9) -> dict | None:
        if cell.cell_id in self.job_infos:
            return self.job_infos[cell.cell_id]
        jd, path = self.job_dir(cell), self.job_dir(cell) / "training_info.json"
        if not path.exists():
            return None
        try:
            info = json.loads(path.read_text())
            if info.get("identity") != self.identity_of(cell) or info.get("status") != "done":
                return None
            step = info["roles"]["selected"]
            ledger = json.loads((jd / "checkpoints.json").read_text())["ledger"]
            row = next(r for r in ledger if r["step"] == step)
            if T.tensor_hash(T.load_state(jd / "ckpt" / row["file"])["state_dict"]) != row["tensor_sha256"]:
                return None
            if cell.arm == "i2" and info.get("teacher_sha") != self.teacher_sha(cell.dataset, cell.seed):
                return None
        except (OSError, KeyError, ValueError, StopIteration, json.JSONDecodeError, RuntimeError):
            return None
        self.job_infos[cell.cell_id] = info
        return info

    def ensure_train(self, cell: Cell9) -> bool:
        if self.job_state(cell) is not None:
            return True
        if self.expired():
            return False
        loader = self.train_loader(cell)
        if loader is None:
            return False
        jd = self.job_dir(cell)
        if jd.exists():
            shutil.rmtree(jd)  # an incomplete fit restarts from initialization
        jd.mkdir(parents=True)
        job = self.make_job(cell)
        print(f"[train] {cell.cell_id} ({job.arm}, span {job.mask_span}, {self.scale.horizon(P.FULL_HORIZON)} updates) ...", flush=True)
        try:
            info = T9.run_training_job9(job, self.scale, loader, jd, self.device, self.deadline, self.identity_of(cell), self.impl)
        except DeadlineExceeded:
            return False
        state, _ = self._state_of(jd, info)
        info["masked_val"] = T9.masked_val_accuracy(job, self.scale, loader, state, self.device)
        if cell.arm == "i2":
            info["teacher_sha"] = self.teacher_sha(cell.dataset, cell.seed)
        write_json_atomic(jd / "training_info.json", info)
        self.job_infos[cell.cell_id] = info
        print(f"[train] {cell.cell_id} done (selected step {info['roles']['selected']}, val loss {info['best_val_loss']:.4f}, "
              f"masked accuracy {info['masked_val']['val_accuracy']:.3f})", flush=True)
        return True

    @staticmethod
    def _state_of(jd: Path, info: dict):
        row = next(r for r in json.loads((jd / "checkpoints.json").read_text())["ledger"] if r["step"] == info["roles"]["selected"])
        return T.load_state(jd / "ckpt" / row["file"])["state_dict"], row["tensor_sha256"]

    # -- stage B: I1's per-layer states -> K-means -> the I2 family, layer diagnostics ---------------------------------
    def build_i2(self, dataset: str, tag: str = "B") -> None:
        spec = D.SPECS[dataset]
        for seed in self.seeds:
            cell = Cell9(dataset, "i1", seed)
            sha = self.teacher_sha(dataset, seed)
            if sha is None:
                self.not_run[f"{dataset}__i2__seed{seed}"] = "I1 checkpoint missing; no iteration-2 targets"
                continue
            if self.family(dataset, seed) is not None:
                continue
            if self.expired():
                return
            t0 = time.perf_counter()
            base, fam = self.base_loader(cell), cell.spec.family
            job, info = self.make_job(cell), self.job_state(cell)
            state, _ = self.selected_state(cell, info)
            model = T9.build_model_from_state9(job, self.scale, base, state, self.device)
            lookup, rms = base.lookup(fam), float(base.rms(fam))
            names = sorted(base.recording_names())
            layers = [dict() for _ in range(model.cfg.n_layers)]
            for n in names:
                _, _, states = T9.extract_embeddings9(job.arm, model, base, fam, n, lookup, rms, self.device, job.context,
                                                       job.stride, layers=True)
                for k, s in enumerate(states):
                    layers[k][n] = s
            sample_ids = artifacts.load_sample_ids(base.run_dir)
            diag = dict(seed=seed, teacher_sha=sha)
            layer_codes = {}
            for k in range(len(layers)):
                if self.expired():
                    return
                km = U9.cluster_states(layers[k], sample_ids, seed, TOK.PREP_THREADS)
                codes = U9.assign_codes(km, layers[k], TOK.PREP_THREADS)
                layer_codes[k + 1] = codes
                nn = U9.nn_same_recording_share({n: _unit(layers[k][n]) for n in names})
                diag[f"layer{k + 1}"] = dict(**U9.occupancy(codes), recording_nmi=U9.recording_nmi(codes),
                                             subject_nmi=U9.subject_nmi(codes, spec), nn_same_recording=nn,
                                             kmeans_iterations=int(km.n_iter_))
                if k + 1 == U9.LAYER:
                    teacher = km.cluster_centers_.astype(np.float32)
                    target_codes = codes
            ref = {n: base.recording(fam, n)["codes"] for n in names}
            cont = {n: base.recording(fam, n)["continuous"] for n in names}
            diag["t2"] = dict(**U9.occupancy(ref), recording_nmi=U9.recording_nmi(ref), subject_nmi=U9.subject_nmi(ref, spec),
                              nn_same_recording=U9.nn_same_recording_share(cont))
            try:
                src = U9.build_family(base, fam, target_codes, TOK.dataset_spec(dataset), seed, teacher, meta=dict(
                    teacher_sha=sha, layer=U9.LAYER, k=U9.NUM_CODES, seed=seed, sampled_windows=len(sample_ids),
                    kmeans_iterations=diag[f"layer{U9.LAYER}"]["kmeans_iterations"]))
            except categorical.UnavailableError as exc:  # e.g. collapsed targets: recorded, never worked around
                self.not_run[f"{dataset}__i2__seed{seed}"] = f"iteration-2 family unavailable: {exc}"
                print(f"[{tag}] {dataset} seed {seed}: iteration-2 family UNAVAILABLE -- {exc}", flush=True)
                continue
            g9f = U9.check_family_dims(src, fam, model_input_dim=model.cfg.input_dim)
            self.gates[f"g9f_{dataset}_{seed}"] = dict(g9f, implementation_hash=self.impl)
            write_json_atomic(self.gates_path, self.gates)
            U9.save_family(src, self.i2_dir(dataset, seed))
            np.savez_compressed(self.out / "diagnostics" / f"{dataset}__seed{seed}__layer_codes.npz",
                                **{f"l{k}__{n}": c.astype(np.int16) for k, d in layer_codes.items() for n, c in d.items()})
            write_json_atomic(self.out / "diagnostics" / f"{dataset}__seed{seed}.json", diag)
            self.families[(dataset, seed)] = src
            print(f"[{tag}] {dataset} seed {seed}: iteration-2 family built ({time.perf_counter() - t0:.0f}s), G9f {g9f['status']}",
                  flush=True)
        self._finish_diagnostics(dataset)

    def _finish_diagnostics(self, dataset: str) -> None:
        files = sorted((self.out / "diagnostics").glob(f"{dataset}__seed*.json"))
        if len(files) < 2:
            return
        per_seed = {json.loads(f.read_text())["seed"]: json.loads(f.read_text()) for f in files}
        out = dict(dataset=dataset, seeds=sorted(per_seed), layers={})
        for k in range(1, 5):
            codes = {}
            for s in per_seed:
                with np.load(self.out / "diagnostics" / f"{dataset}__seed{s}__layer_codes.npz") as z:
                    codes[s] = {key.split("__", 1)[1]: z[key] for key in z.files if key.startswith(f"l{k}__")}
            out["layers"][k] = dict(cross_seed_nmi=U9.cross_seed_nmi(codes))
        base = self.base_loader(Cell9(dataset, "i1", next(iter(per_seed))))
        t2 = {}
        for s in per_seed:
            b = self.base_loader(Cell9(dataset, "i1", s))
            t2[s] = {n: b.recording(TOK.SPECS[TOK9[dataset]].family, n)["codes"] for n in sorted(b.recording_names())}
        out["t2_cross_seed_nmi"] = U9.cross_seed_nmi(t2)
        out["chance_nmi"] = U9.chance_nmi(t2)
        out["windows"] = int(sum(len(c) for c in next(iter(t2.values())).values()))
        write_json_atomic(self.out / "diagnostics" / f"{dataset}.json", out)

    # -- decoding ---------------------------------------------------------------------------------------------------------
    def group_sha(self, cell: Cell9, tag: str) -> str | None:
        if tag != "ig":
            return None
        fam = self.family(cell.dataset, cell.seed)
        return hashlib.sha256(fam.grouping(None).tobytes() + str(fam.meta.get("teacher_sha")).encode()).hexdigest()[:16] if fam else "missing"

    def unit_identity(self, cell: Cell9, tag: str, state_sha: str | None) -> str:
        return hashlib.sha256(json.dumps([cell.cell_id, tag, state_sha, self.radius.name, self.impl, self.scale.outer_cap,
                                          list(self.scale.rungs), self.scale.name, self.identity_of(cell),
                                          self.group_sha(cell, tag)]).encode()).hexdigest()[:16]

    def embeddings(self, job: P.JobSpec, loader, state: dict, names: list[str]):
        model = T9.build_model_from_state9(job, self.scale, loader, state, self.device)
        fam = job.family
        lookup, rms = loader.lookup(fam), float(loader.rms(fam))
        z, zero = {}, {}
        for n in names:
            z[n], zero[n] = T9.extract_embeddings9(job.arm, model, loader, fam, n, lookup, rms, self.device, job.context, job.stride)
        return z, zero

    def run_ot(self, spec, names, loader, z, zero, codes_loader, grouping):
        fam = TOK.SPECS[TOK9[spec.name]].family
        codes, lens, starts = {}, {}, {}
        for n in names:
            rec = codes_loader.recording(fam, n)
            codes[n], lens[n], starts[n] = rec["codes"], rec["lengths"], rec["starts"]
        recs = [continuous.ContinuousRecordingInput(n, codes[n].astype(np.int32), lens[n].astype(np.int32), spec.fps, spec.window,
                                                    z[n].astype(np.float64), zero[n]) for n in names]
        mu0, _ = continuous.build_mu0(grouping, recs, spec.num_states)
        gts = D.load_gts(spec, names)
        frame_counts = {n: loader.frame_count(n) for n in names}
        return R7.fit_and_score_keep(recs, mu0, spec.num_states, R1.coeffs_for(0.3), self.radius, self.scale.outer_cap,
                                     self.scale.rungs, gts, starts, frame_counts, self.deadline)

    def ensure_decode(self, cell: Cell9, tag: str) -> str:
        info = self.job_state(cell)
        if info is None:
            return "no_checkpoint"
        state, sha = self.selected_state(cell, info)
        i2 = self.family(cell.dataset, cell.seed) if tag == "ig" else None
        if tag == "ig" and i2 is None:
            return "no_i2_family"
        identity = self.unit_identity(cell, tag, sha)
        if self.unit_done(cell, tag, identity):
            return "cached"
        if self.expired():
            return "deadline"
        base = self.base_loader(cell)
        mloader = self.train_loader(cell)
        if base is None or mloader is None:
            return "no_preparation"
        spec = TOK.dataset_spec(cell.dataset, cell.window)
        names = sorted(base.recording_names())
        job = self.make_job(cell)
        z, zero = self.embeddings(job, mloader, state, names)
        codes_loader, grouping = (i2, i2.grouping(None)) if tag == "ig" else (base, base.grouping(cell.spec.family))
        udir = self.unit_dir(cell, tag)
        meta = dict(dataset=cell.dataset, tokenizer=cell.tok, seed=cell.seed, window=cell.window, span=2, guard=0, arm=cell.arm,
                    tag=tag, radius=self.radius.name, unit=f"{cell.cell_id}__{tag}")
        try:
            res = self.run_ot(spec, names, base, z, zero, codes_loader, grouping)
        except DeadlineExceeded:
            return "deadline"
        except FinalSolveInvalid as exc:
            udir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(udir / "info.json", dict(status="invalid", identity=identity, reason=str(exc), **meta))
            return "invalid"
        udir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(udir / "predictions.npz", **res.pop("predictions"))
        nn = U9.nn_same_recording_share({n: z[n] for n in names})
        write_json_atomic(udir / "info.json", dict(
            status="done", identity=identity, checkpoint_step=info["roles"]["selected"], nn_same_recording=nn,
            masked_val=info.get("masked_val"), **res, **meta))
        return "done"

    # -- items -----------------------------------------------------------------------------------------------------------
    def blocked(self, cell: Cell9) -> str | None:
        for g in ("g9a", "g9b", "g9c", "g9d", "g9e", "g9g"):
            if self.gates.get(g, {}).get("status") not in ("passed", "not_applicable_smoke"):
                return f"{g.upper()} not passed (preflight)"
        if cell.arm == "i2" and self.gates.get(f"g9f_{cell.dataset}_{cell.seed}", {}).get("status") != "passed":
            return f"G9F not passed for {cell.dataset} seed {cell.seed} (I2 family dimensions)"
        if cell.dataset == "lara" and self.gates.get("g3", {}).get("status") != "passed":
            return "G3 not passed (LARa channels 3-5 are not confirmed as positions)"
        return None

    def run_item(self, item: Item9, counts: dict, tag: str) -> None:
        cell = item.cell
        reason = self.blocked(cell)
        if reason:
            self.not_run[cell.cell_id] = reason
            counts["blocked"] = counts.get("blocked", 0) + len(item.tags)
            print(f"[{tag}] {cell.cell_id}: BLOCKED -- {reason}", flush=True)
            return
        if self.expired():
            counts["deadline"] = counts.get("deadline", 0) + len(item.tags)
            return
        if cell.arm == "i2" and self.family(cell.dataset, cell.seed) is None:
            counts["no_i2_family"] = counts.get("no_i2_family", 0) + len(item.tags)
            self.not_run[cell.cell_id] = "no iteration-2 family (stage B incomplete)"
            return
        if item.train and not self.ensure_train(cell):
            counts["no_checkpoint"] = counts.get("no_checkpoint", 0) + len(item.tags)
            return
        for t in item.tags:
            t0 = time.perf_counter()
            status = self.ensure_decode(cell, t)
            counts[status] = counts.get(status, 0) + 1
            print(f"[{tag}] {cell.cell_id} {t}: {status} ({time.perf_counter() - t0:.0f}s)", flush=True)

    def execute(self) -> None:
        from script.week5.timescale_inspection import gates9
        counts: dict[str, int] = {}
        gates9.run_gates(self)
        for ds in self.datasets:
            for st in ("A", "C"):
                if st == "C":
                    self.build_i2(ds)
                items = self.items(ds, st)
                for i, item in enumerate(items, 1):
                    self.run_item(item, counts, f"{ds} {st} {i}/{len(items)}")
        m = json.loads(self.manifest_path.read_text())
        m.update(status="finished" if not self.expired() else "budget_stop", counts=counts, not_run=self.not_run,
                 finished_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        write_json_atomic(self.manifest_path, m)
        from script.week5.timescale_inspection import report9
        print(f"report: {report9.write_report(self.out, self.source_root, prospective=self.prospective)}")

    def dry_run(self) -> None:
        total = GATE_MIN
        print(f"stage G: gates G9a-G9g (~{GATE_MIN:.0f} min)")
        for ds in self.datasets:
            ctx, trn = COST["context"][ds], COST["train"][ds]
            a = sum(trn + ctx for it in self.items(ds, "A") for _ in it.tags)
            b = LAYER_FIT_MIN[ds]
            c = sum((trn if it.train else 0) + ctx * len(it.tags) for it in self.items(ds, "C"))
            total += a + b + c
            print(f"{ds}: A {len(self.items(ds, 'A'))} trainings ~{a:.0f} min; B ~{b} min; C {len([i for i in self.items(ds, 'C') if i.train])} "
                  f"trainings + {sum(len(i.tags) for i in self.items(ds, 'C'))} decodes ~{c:.0f} min; cumulative ~{total / 60:.1f} h")
        total += 10
        print(f"projected total ~{total / 60:.1f} h (single-pass, otherwise idle machine; budget cap 10 h for the development run)")


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n > 1e-12, n, 1.0)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Time-scale inspection, phase 9 (iterative contextual target refinement)")
    ap.add_argument("--run-id", default="ts5_v1")
    ap.add_argument("--source-run", default="ts5_v1", help="run whose phase5/phase7 hold the T2 preparations and references")
    ap.add_argument("--stage", default="dev", choices=("dev", "P"), help="dev: BABEL-2/3 and LARa; P: prospective BABEL-1")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not (args.dry_run or args.smoke) and not args.hours:
        raise SystemExit("a decoding run requires an explicit elapsed-time budget: pass --hours")
    runner = Phase9Runner(args)
    if args.dry_run:
        runner.dry_run()
    else:
        runner.execute()


if __name__ == "__main__":
    main()
