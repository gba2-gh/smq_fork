"""Dimensionality-reduction ablation around the best configuration of each dataset.

Everything except the stage between the global standardizer and K-means-512 is held at the dataset's anchor (see
`prepare.ANCHORS`): source features, window, K, contextualizer (masking, guard, span, budget, checkpoint rule), OT
(setting T coefficients, radius fps // window, rung ladder 500/1000/2000/4000, outer cap 50) and the decoding arm.
Variants are in `reducers.VARIANTS`; design and reading rules: docs/plans/week5/DIMRED_ABLATION/PLAN.md.

Stages run in order, each over BABEL-2, BABEL-3, HuGaDB, then LARa (cheapest first), 3 seeds per cell, so a budget
stop always leaves every earlier stage complete:
  G  gates: the `base` preparation reproduces each anchor's saved tokens; LARa's channel gate is inherited
  M  (default) why PCA: base (PCA-64, randomized SVD), no reduction, random projection-64, slow features-64
  S  why randomized SVD: PCA-64 with the exact (full) SVD -- run only if `solver_check` finds the tokens differ
  X  declared extensions (dimension curve, whitening, recording-invariant axes), not run by default

Usage:
  python -m script.week5.dimred_ablation.solver_check                     (token level, minutes, no decoding)
  python -m script.week5.dimred_ablation.run_dimred --run-id dr_v1 --dry-run
  python -m script.week5.dimred_ablation.run_dimred --run-id dr_v1 --hours 10 [--stages M] [--datasets babel2,babel3]
  python -m script.week5.dimred_ablation.run_dimred --run-id dr_smoke --smoke
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from script.action_transport import continuous  # noqa: E402
from script.action_transport.budget import DeadlineExceeded  # noqa: E402
from script.fsq_reconstruction.evaluate import FinalSolveInvalid  # noqa: E402
from script.fsq_reconstruction.experiment import write_json_atomic  # noqa: E402
from script.quantized_input_comparison.artifacts import Loader  # noqa: E402
from script.week5.contextualizer_calibration import evaluate as E  # noqa: E402
from script.week5.contextualizer_calibration import plan as P  # noqa: E402
from script.week5.contextualizer_calibration import train as T  # noqa: E402
from script.week5.dimred_ablation import prepare as PR  # noqa: E402
from script.week5.dimred_ablation import reducers as RD  # noqa: E402
from script.week5.raw_kmeans_cross_dataset import datasets as D  # noqa: E402
from script.week5.timescale_inspection import radius as RAD  # noqa: E402
from script.week5.timescale_inspection import run_phase1 as R1  # noqa: E402
from script.week5.timescale_inspection import run_phase5 as R5  # noqa: E402

SCHEMA_VERSION = 1
SEEDS = (111, 222, 1538574472)
DATASETS = ("babel2", "babel3", "hugadb", "lara")  # execution order inside a stage: cheapest first
STAGE_ORDER = ("G",) + RD.STAGES
SMOKE_VARIANTS = {"babel2": tuple(v.key for v in RD.VARIANTS), "babel3": ("base", "sfa64", "wrpca64"),
                  "hugadb": ("base", "identity"), "lara": ("base", "rc_pca64")}
OWN_FILES = tuple(f"script/week5/dimred_ablation/{m}.py" for m in ("reducers", "prepare", "run_dimred"))
TRAIN_FILES = R5.TRAIN_FILES + ("script/week5/timescale_inspection/tokenizers.py",
                                "script/week5/timescale_inspection/run_phase1.py",
                                "script/week5/timescale_inspection/radius.py",
                                "script/quantized_input_comparison/extract.py",
                                "script/quantized_input_comparison/fit.py")
TS5 = ROOT / "results" / "week5" / "timescale_inspection" / "ts5_v1"
# the anchor's own saved tokens (seed 111), reproduced by the `base` preparation (gate G-<dataset>)
ANCHOR_PREP = {
    "babel2": (TS5 / "phase5" / "prep" / "t2_relational" / "babel2__t2_relational__seed111", "tok5_t2_relational"),
    "babel3": (TS5 / "phase5" / "prep" / "t1_whiten" / "babel3__t1_whiten__seed111", "tok5_t1_whiten"),
    "lara": (TS5 / "phase5" / "prep" / "t2_relational" / "lara__t2_relational__seed111", "tok5_t2_relational"),
    "hugadb": (ROOT / "results" / "quantized_input_comparison" / "qic_v3", "vq_kmeans"),
}
LOOKUP_ATOL = 1e-4  # gates5.LOOKUP_ATOL: thread scheduling moves the 6th decimal of PCA / K-means
CODE_AGREEMENT_MIN = 0.9999
# minutes per unit, from run_phase5.COST; HuGaDB's VQ extraction is paid once per process (cached)
COST = dict(prep=dict(babel2=0.5, babel3=0.5, lara=2.0, hugadb=0.7), train=R5.COST["train"],
            context=R5.COST["context"], direct=R5.COST["direct"])


def source_hash() -> str:
    h = hashlib.sha256()
    for name in OWN_FILES + TRAIN_FILES:
        h.update(name.encode())
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class Cell:
    dataset: str
    variant: str
    seed: int

    @property
    def anchor(self) -> PR.Anchor:
        return PR.ANCHORS[self.dataset]

    @property
    def family(self) -> str:
        return PR.family_of(self.variant)

    @property
    def cell_id(self) -> str:
        return f"{self.dataset}__{self.variant}__seed{self.seed}"

    @property
    def arm(self) -> str:
        return self.anchor.arm


def make_job(cell: Cell) -> P.JobSpec:
    """The anchor's contextualizer: reference configuration (context 128, stride 64, 64 chunks per update, 5000
    updates, best-validation checkpoint, no early stopping), code-identity input, the anchor's guard and span."""
    context, _, stride, batch = P.CONFIGS["reference"]
    a = cell.anchor
    return P.JobSpec(cell.cell_id, "main", "reference", f"dr_{cell.variant}", cell.family, "context_id", a.guard,
                     context, stride, a.span, 0.30, batch, P.FULL_HORIZON, cell.seed, "none", "")


def stage_cells(stage: str, seeds=SEEDS, datasets=DATASETS, smoke: bool = False) -> list[Cell]:
    out = []
    for ds in DATASETS:
        if ds not in datasets:
            continue
        for v in RD.VARIANTS:
            if v.stage != stage or (smoke and v.key not in SMOKE_VARIANTS[ds]):
                continue
            out.extend(Cell(ds, v.key, s) for s in seeds)
    return out


class Runner:
    def __init__(self, args):
        self.args = args
        self.smoke = bool(args.smoke)
        self.scale = replace(P.SMOKE, recording_cap=None) if self.smoke else P.FULL
        self.out = ROOT / "results" / "week5" / "dimred_ablation" / args.run_id
        self.deadline = time.perf_counter() + args.hours * 3600 if args.hours else None
        self.impl = source_hash()
        self.radius = RAD.CURRENT
        self.seeds = (111,) if self.smoke else SEEDS
        self.datasets = tuple(args.datasets.split(",")) if args.datasets else DATASETS
        self.stages = tuple(args.stages.split(",")) if args.stages else RD.DEFAULT_STAGES
        unknown = (set(self.datasets) - set(DATASETS)) | (set(self.stages) - set(RD.STAGES))
        if unknown:
            raise SystemExit(f"unknown dataset(s)/stage(s): {sorted(unknown)}")
        self.job_infos: dict[str, dict] = {}
        self.loaders: dict[str, Loader] = {}
        self.not_run: dict[str, str] = {}
        self.gates_path = self.out / "gates.json"
        self.gates: dict = json.loads(self.gates_path.read_text()) if self.gates_path.exists() else {}
        self.manifest_path = self.out / "manifest.json"
        # the identity covers every planned cell, whatever --stages / --datasets select for this invocation
        self.all_cells = [c for st in RD.STAGES for c in stage_cells(st, self.seeds, DATASETS, self.smoke)]
        if not args.dry_run:  # a dry run writes nothing
            for sub in ("prep", "jobs", "units", "diagnostics"):
                (self.out / sub).mkdir(parents=True, exist_ok=True)
            self.check_resume()

    # -- identity ----------------------------------------------------------------------------------------------------
    def recording_cap(self, dataset: str) -> int | None:
        return D.SPECS[dataset].smoke_cap if self.smoke else None

    def prep_key(self, cell: Cell) -> PR.PrepKey:
        return PR.make_key(cell.dataset, cell.variant, cell.seed, self.recording_cap(cell.dataset))

    def identity_of(self, cell: Cell) -> str:
        job = P.effective_settings(make_job(cell), self.scale) if cell.arm == "context" else None
        return P.stable_hash(dict(job=job, prep=asdict(self.prep_key(cell)), anchor=asdict(cell.anchor),
                                  variant=asdict(RD.BY_KEY[cell.variant]), spec=asdict(D.SPECS[cell.dataset]),
                                  schema=SCHEMA_VERSION))

    def check_resume(self) -> None:
        ident = dict(schema=SCHEMA_VERSION, implementation_hash=self.impl, scale=self.scale.name,
                     outer_cap=self.scale.outer_cap, rungs=list(self.scale.rungs), radius=self.radius.name,
                     cells={c.cell_id: self.identity_of(c) for c in self.all_cells})
        if self.manifest_path.exists():
            old = json.loads(self.manifest_path.read_text())
            if old.get("identity") != ident:
                raise SystemExit(f"{self.out} was written by a different implementation, scale or cell set; "
                                 f"choose a new --run-id.")
        write_json_atomic(self.manifest_path, dict(
            identity=ident, run_id=self.args.run_id, status="running", smoke=self.smoke,
            python=platform.python_version(), started_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            radius=self.radius.name, anchors={k: asdict(v) for k, v in PR.ANCHORS.items()},
            variants=[asdict(v) for v in RD.VARIANTS]))
        self.ident = ident

    def expired(self) -> bool:
        return self.deadline is not None and time.perf_counter() > self.deadline

    @property
    def device(self):
        import torch
        return torch.device("cuda" if torch.cuda.is_available() and not self.args.cpu else "cpu")

    # -- gates -------------------------------------------------------------------------------------------------------
    def _gate(self, name: str, fn) -> dict:
        cached = self.gates.get(name)
        if cached and cached.get("status") in ("passed", "differs") and cached.get("implementation_hash") == self.impl:
            return cached
        t0 = time.perf_counter()
        try:
            res = fn()
        except Exception as exc:  # a crashing gate is a failed gate, never a silent pass
            res = dict(status="failed", reason=f"{type(exc).__name__}: {exc}")
        res.update(implementation_hash=self.impl, seconds=time.perf_counter() - t0)
        self.gates[name] = res
        write_json_atomic(self.gates_path, self.gates)
        print(f"[gate] {name}: {res['status']} ({res['seconds']:.0f}s)"
              + (f" -- {res.get('reason', '')}" if res["status"] != "passed" else ""), flush=True)
        return res

    def gate_reproduce(self, dataset: str) -> dict:
        """The `base` preparation at seed 111 against the anchor's own saved tokens: codes, features, grouping, mean
        and RMS exactly, K-means centres to LOOKUP_ATOL. HuGaDB's anchor was prepared by a different module
        (qic_v3) under a different thread count, so a HuGaDB mismatch is recorded as `differs` (non-blocking: the
        ablation's own `base` is the comparison point) with the share of windows whose code agrees."""
        ref_dir, ref_fam = ANCHOR_PREP[dataset]
        if not ref_dir.exists():
            return dict(status="failed", reason=f"reference not found: {ref_dir}")
        loader = self.prepare(Cell(dataset, "base", 111))
        if loader is None:
            return dict(status="failed", reason="preparation unavailable (deadline)")
        ref, fam = Loader(ref_dir), PR.family_of("base")
        names = sorted(loader.recording_names())
        checks = dict(same_recordings=names == sorted(ref.recording_names()))
        agree = total = 0
        if checks["same_recordings"]:
            codes_eq = feats_eq = True
            for n in names:
                a, b = ref.recording(ref_fam, n), loader.recording(fam, n)
                codes_eq &= bool(np.array_equal(a["codes"], b["codes"]))
                feats_eq &= bool(np.array_equal(a["continuous"], b["continuous"]))
                agree += int((a["codes"] == b["codes"]).sum())
                total += len(a["codes"])
            checks.update(codes=codes_eq, continuous=feats_eq)
        checks["mean"] = bool(np.array_equal(ref.mean(ref_fam), loader.mean(fam)))
        checks["rms"] = float(ref.rms(ref_fam)) == float(loader.rms(fam))
        checks["grouping"] = bool(np.array_equal(ref.grouping(ref_fam), loader.grouping(fam)))
        diff = float(np.abs(ref.lookup(ref_fam).astype(np.float64) - loader.lookup(fam).astype(np.float64)).max())
        checks["lookup"] = diff <= LOOKUP_ATOL
        # centroids within LOOKUP_ATOL can flip a window sitting on a K-means tie (LARa: 1 window in ~220k), so codes
        # need not be bit-equal when every feature is: they must agree on at least CODE_AGREEMENT_MIN of windows
        codes_exact = checks.pop("codes", False)
        checks["codes"] = codes_exact or (checks.get("continuous", False) and total > 0
                                          and agree / total >= CODE_AGREEMENT_MIN)
        ok = all(checks.values())
        status = "passed" if ok else ("differs" if dataset == "hugadb" else "failed")
        return dict(status=status, checks=checks, codes_bit_equal=codes_exact, lookup_max_abs_diff=diff,
                    reference=str(ref_dir), code_agreement=agree / total if total else None,
                    windows_differing=total - agree)

    def inherit_g3(self) -> None:
        """LARa's T2 tokens need channels 3-5 to be positions; ts5_v1 phase 5 verified it (G3). Read, never re-run."""
        path = TS5 / "phase5" / "gates.json"
        g5 = json.loads(path.read_text()) if path.exists() else {}
        self.gates["g3"] = dict(g5["g3"], inherited_from=str(path)) if "g3" in g5 else dict(
            status="failed", reason=f"g3 not recorded in {path}")

    def run_gates(self) -> None:
        if "lara" in self.datasets:
            self.inherit_g3()
        if self.smoke:
            for ds in self.datasets:
                self.gates[f"g_{ds}"] = dict(status="not_applicable_smoke", implementation_hash=self.impl)
        else:
            for ds in self.datasets:
                self._gate(f"g_{ds}", lambda ds=ds: self.gate_reproduce(ds))
        write_json_atomic(self.gates_path, self.gates)

    def blocked(self, cell: Cell) -> str | None:
        g = self.gates.get(f"g_{cell.dataset}", {})
        if g.get("status") == "failed":
            return f"gate g_{cell.dataset} failed (the base preparation does not reproduce the anchor's tokens)"
        if cell.dataset == "lara" and self.gates.get("g3", {}).get("status") != "passed":
            return "G3 not passed (LARa channels 3-5 are not confirmed as positions)"
        return None

    # -- preparation -------------------------------------------------------------------------------------------------
    def prep_dir(self, cell: Cell) -> Path:
        return self.out / "prep" / cell.dataset / f"{cell.variant}__seed{cell.seed}"

    def prepare(self, cell: Cell) -> Loader | None:
        if cell.cell_id in self.loaders:
            return self.loaders[cell.cell_id]
        d, cap = self.prep_dir(cell), self.recording_cap(cell.dataset)
        try:
            if PR.is_prepared(cell.dataset, cell.variant, cell.seed, d, cap):
                self.loaders[cell.cell_id] = Loader(d)
                return self.loaders[cell.cell_id]
        except (OSError, KeyError, ValueError):
            pass
        if self.expired():
            return None
        t0 = time.perf_counter()
        PR.prepare_cell(cell.dataset, cell.variant, cell.seed, d, cap, self.device)
        info = json.loads((d / "reducer_info.json").read_text())
        print(f"[prep] {cell.cell_id} ({time.perf_counter() - t0:.0f}s; {info['input_dim']} -> {info['out_dim']} dims, "
              f"retained variance {info['retained_variance']:.3f}, {info['kmeans_occupied']} codes occupied)", flush=True)
        self.loaders[cell.cell_id] = Loader(d)
        return self.loaders[cell.cell_id]

    def ensure_diagnostics(self, cell: Cell) -> None:
        """Evaluation-only description of the tokens (written after preparation, never read by any fit): NMI of the
        window code with the window's majority action, its recording and (HuGaDB / LARa) its subject."""
        path = self.out / "diagnostics" / f"{cell.cell_id}.json"
        if path.exists():
            return
        loader = self.prepare(cell)
        if loader is None:
            return
        from sklearn.metrics import normalized_mutual_info_score as nmi
        spec = D.SPECS[cell.dataset]
        names = sorted(loader.recording_names())
        gts = D.load_gts(spec, names)
        codes, actions, recs, subjects = [], [], [], []
        for i, n in enumerate(names):
            rec = loader.recording(cell.family, n)
            g = gts[n]
            for c, s, ln in zip(rec["codes"], rec["starts"], rec["lengths"]):
                seg = g[int(s):int(s) + int(ln)]
                if len(seg) == 0:
                    continue
                codes.append(int(c))
                actions.append(int(np.bincount(seg).argmax()))
                recs.append(i)
                subjects.append(D.subject_of(spec, n) if spec.has_subject else None)
        out = dict(cell=cell.cell_id, n_windows=len(codes),
                   nmi_code_action=float(nmi(actions, codes)), nmi_code_recording=float(nmi(recs, codes)))
        if spec.has_subject:
            out["nmi_code_subject"] = float(nmi(subjects, codes))
        out["reducer"] = json.loads((self.prep_dir(cell) / "reducer_info.json").read_text())
        write_json_atomic(path, out)

    # -- training ----------------------------------------------------------------------------------------------------
    def job_dir(self, cell: Cell) -> Path:
        return self.out / "jobs" / cell.dataset / f"{cell.variant}__seed{cell.seed}"

    def job_state(self, cell: Cell) -> dict | None:
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
            row = next(r for r in json.loads((jd / "checkpoints.json").read_text())["ledger"] if r["step"] == step)
            if T.tensor_hash(T.load_state(jd / "ckpt" / row["file"])["state_dict"]) != row["tensor_sha256"]:
                return None
        except (OSError, KeyError, ValueError, StopIteration, json.JSONDecodeError, RuntimeError):
            return None
        self.job_infos[cell.cell_id] = info
        return info

    def selected_state(self, cell: Cell, info: dict) -> tuple[dict, str]:
        jd = self.job_dir(cell)
        row = next(r for r in json.loads((jd / "checkpoints.json").read_text())["ledger"]
                   if r["step"] == info["roles"]["selected"])
        return T.load_state(jd / "ckpt" / row["file"])["state_dict"], row["tensor_sha256"]

    def ensure_train(self, cell: Cell) -> bool:
        if self.job_state(cell) is not None:
            return True
        if self.expired():
            return False
        loader = self.prepare(cell)
        if loader is None:
            return False
        jd = self.job_dir(cell)
        if jd.exists():
            shutil.rmtree(jd)  # an incomplete fit restarts from initialization
        jd.mkdir(parents=True)
        job = make_job(cell)
        print(f"[train] {cell.cell_id} (span {job.mask_span}, guard {job.guard_radius}, "
              f"{self.scale.horizon(P.FULL_HORIZON)} updates) ...", flush=True)
        try:
            info = T.run_training_job(job, self.scale, loader, jd, self.device, self.deadline, self.identity_of(cell),
                                      self.impl)
        except DeadlineExceeded:
            return False
        write_json_atomic(jd / "training_info.json", info)
        self.job_infos[cell.cell_id] = info
        print(f"[train] {cell.cell_id} done (selected step {info['roles']['selected']}, "
              f"val loss {info['best_val_loss']:.4f})", flush=True)
        return True

    # -- decoding ----------------------------------------------------------------------------------------------------
    def unit_dir(self, cell: Cell) -> Path:
        return self.out / "units" / f"{cell.cell_id}__{cell.arm}"

    def unit_identity(self, cell: Cell, state_sha: str | None) -> str:
        return hashlib.sha256(json.dumps([cell.cell_id, cell.arm, state_sha, self.radius.name, self.impl,
                                          self.scale.outer_cap, list(self.scale.rungs), self.scale.name,
                                          self.identity_of(cell)]).encode()).hexdigest()[:16]

    def build_recs(self, cell: Cell, loader: Loader, names: list[str], state: dict | None):
        """`run_phase5.Runner.build_recs` for the two arms used here: contextual embeddings, or the window's own
        reduced vector, mean-centred and L2-normalized."""
        spec = D.SPECS[cell.dataset]
        z, zero, codes, lens, starts = {}, {}, {}, {}, {}
        for n in names:
            rec = loader.recording(cell.family, n)
            codes[n], lens[n], starts[n] = rec["codes"], rec["lengths"], rec["starts"]
        if cell.arm == "context":
            job = make_job(cell)
            model = E.build_model_from_state(job, self.scale, loader, state, self.device)
            lookup, rms = loader.lookup(cell.family), float(loader.rms(cell.family))
            for n in names:
                z[n], zero[n] = E.extract_embeddings(job.arm, model, loader, cell.family, n, lookup, rms, self.device,
                                                      job.context, job.stride)
        else:
            mean = loader.mean(cell.family).astype(np.float64)
            for n in names:
                raw = loader.recording(cell.family, n)["continuous"]
                z[n], zero[n] = continuous.l2_normalize(raw.astype(np.float64) - mean[None, :])
        recs = [continuous.ContinuousRecordingInput(n, codes[n].astype(np.int32), lens[n].astype(np.int32), spec.fps,
                                                    spec.window, z[n].astype(np.float64), zero[n]) for n in names]
        mu0, _ = continuous.build_mu0(loader.grouping(cell.family), recs, spec.num_states)
        return recs, mu0, starts

    def ensure_decode(self, cell: Cell) -> str:
        info = self.job_state(cell) if cell.arm == "context" else {}
        if cell.arm == "context" and info is None:
            return "no_checkpoint"
        state, state_sha = self.selected_state(cell, info) if cell.arm == "context" else (None, None)
        identity = self.unit_identity(cell, state_sha)
        path = self.unit_dir(cell) / "info.json"
        if path.exists():
            try:
                old = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                old = {}
            if old.get("identity") == identity and old.get("status") in ("done", "invalid"):
                return "cached" if old["status"] == "done" else "invalid"  # a deterministic invalid solve is final
        if self.expired():
            return "deadline"
        loader = self.prepare(cell)
        if loader is None:
            return "deadline"
        spec = D.SPECS[cell.dataset]
        names = sorted(loader.recording_names())
        recs, mu0, starts = self.build_recs(cell, loader, names, state)
        gts = D.load_gts(spec, names)
        frame_counts = {n: loader.frame_count(n) for n in names}
        udir = self.unit_dir(cell)
        meta = dict(dataset=cell.dataset, variant=cell.variant, seed=cell.seed, arm=cell.arm, radius=self.radius.name,
                    unit=f"{cell.cell_id}__{cell.arm}")
        try:
            res = R1.fit_and_score(recs, mu0, spec.num_states, R1.coeffs_for(0.3), self.radius, self.scale.outer_cap,
                                   self.scale.rungs, gts, starts, frame_counts, self.deadline)
        except DeadlineExceeded:
            return "deadline"
        except FinalSolveInvalid as exc:
            udir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(udir / "info.json", dict(status="invalid", identity=identity, reason=str(exc), **meta))
            return "invalid"
        udir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(udir / "info.json", dict(
            status="done", identity=identity, checkpoint_step=(info or {}).get("roles", {}).get("selected"),
            ot_status=res["ot_status"], outer_iterations=res["outer_iterations"], segment_stats=res["segment_stats"],
            half_widths_used=res["half_widths_used"], seconds=res["seconds"], ladder=res["ladder"], **meta))
        return "done"

    def run_cell(self, cell: Cell, counts: dict, tag: str) -> None:
        reason = self.blocked(cell)
        if reason:
            self.not_run[cell.cell_id] = reason
            counts["blocked"] = counts.get("blocked", 0) + 1
            print(f"[{tag}] {cell.cell_id}: BLOCKED -- {reason}", flush=True)
            return
        if self.expired():
            counts["deadline"] = counts.get("deadline", 0) + 1
            return
        if self.prepare(cell) is None:
            counts["deadline"] = counts.get("deadline", 0) + 1
            return
        self.ensure_diagnostics(cell)
        if cell.arm == "context" and not self.ensure_train(cell):
            counts["no_checkpoint"] = counts.get("no_checkpoint", 0) + 1
            return
        t0 = time.perf_counter()
        status = self.ensure_decode(cell)
        counts[status] = counts.get(status, 0) + 1
        print(f"[{tag}] {cell.cell_id} {cell.arm}: {status} ({time.perf_counter() - t0:.0f}s)", flush=True)

    # -- orchestration -----------------------------------------------------------------------------------------------
    def execute(self) -> None:
        counts: dict[str, int] = {}
        self.run_gates()
        if not self.args.gates_only:
            for stage in self.stages:
                cells = stage_cells(stage, self.seeds, self.datasets, self.smoke)
                for i, cell in enumerate(cells, 1):
                    self.run_cell(cell, counts, f"{stage} {i}/{len(cells)}")
        m = json.loads(self.manifest_path.read_text())
        m.update(status=("gates_only" if self.args.gates_only else
                         "finished" if not self.expired() else "budget_stop"),
                 counts=counts, not_run=self.not_run, finished_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        write_json_atomic(self.manifest_path, m)
        from script.week5.dimred_ablation import report
        print(f"report: {report.write_report(self.out)}")

    def dry_run(self) -> None:
        total = 0.15 * 60 if not self.smoke else 0.0
        print("stage G: reproduction gates, base seed 111 per dataset (~9 min; builds preparations stage A reuses)")
        for stage in self.stages:
            cells = stage_cells(stage, self.seeds, self.datasets, self.smoke)
            minutes = 0.0
            for c in cells:
                minutes += COST["prep"][c.dataset]
                minutes += (COST["train"][c.dataset] + COST["context"][c.dataset]) if c.arm == "context" \
                    else COST["direct"][c.dataset]
            per_ds = {ds: sum(1 for c in cells if c.dataset == ds) for ds in self.datasets}
            variants = sorted({c.variant for c in cells}, key=lambda k: [v.key for v in RD.VARIANTS].index(k))
            total += minutes
            print(f"stage {stage}: {len(cells)} cells ({', '.join(f'{k} {v}' for k, v in per_ds.items())}), "
                  f"variants {', '.join(variants)}; ~{minutes / 60:.1f} h, cumulative ~{total / 60:.1f} h")
        print(f"projected total ~{total / 60:.1f} h (single process, otherwise idle machine)")
        print("arms: " + ", ".join(f"{ds} {PR.ANCHORS[ds].arm}" for ds in self.datasets))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Dimensionality-reduction ablation on each dataset's best configuration")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--hours", type=float, default=None, help="elapsed-time budget; completed units are kept")
    ap.add_argument("--stages", default=None, help="comma list of M,S,X (default: M)")
    ap.add_argument("--datasets", default=None, help="comma list of babel2,babel3,hugadb,lara (default: all)")
    ap.add_argument("--gates-only", action="store_true", help="run the reproduction gates, then stop")
    ap.add_argument("--smoke", action="store_true", help="tiny budgets, capped recordings, seed 111 (plumbing check)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not (args.dry_run or args.smoke or args.gates_only) and not args.hours:
        raise SystemExit("a decoding run requires an explicit elapsed-time budget: pass --hours")
    runner = Runner(args)
    if args.dry_run:
        runner.dry_run()
    else:
        runner.execute()


if __name__ == "__main__":
    main()
