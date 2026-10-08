"""Phase 5 of the time-scale inspection: the tokenizer, then span and window (H6, H7, H2').

Phases 0-4 compared the contextualizer with no-contextualizer decoding on ONE tokenizer. This phase varies the
tokenizer (T1 whitened PCA, T2 inter-joint distances, T3/T3b LARa positions-only at PCA-256 / PCA-64, T4 SMQ's
released encoder latents), then, on the best guard-0 tokenizer chosen on BABEL-2 alone, the mask span and the token
window. Design, gates and reading rules: docs/plans/week5/TIMESCALE_INSPECTION/PHASE5_TOKENIZER.md.

Stages run in this order, so a budget stop always leaves every earlier stage complete:
  A  BABEL-2/3: T1, T2, T4 x 3 seeds x {context, centre, pca}
  B  HuGaDB control (seed 111): T1, T4 x {context, centre}
  C1 LARa: T3 (+pca), T2, T4 x 3 seeds x {context, centre}
  P1 per-token probes, seed 111, every family built so far
  C2 LARa: T3b x 3 seeds x {context, centre}
  D  span {2, 4} on the selected tokenizer, `context` only (BABEL-2, then BABEL-3)
  E  window {4, 15} on the selected tokenizer, BABEL-2/3 x {context, centre}   (only if time remains)
  P2 probes for the remaining seeds

Usage:
  python -m script.week5.timescale_inspection.run_phase5 --run-id ts5_v1 --hours 10
  python -m script.week5.timescale_inspection.run_phase5 --run-id ts5_v1 --dry-run
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
import statistics  # noqa: E402
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
from script.week5.raw_kmeans_cross_dataset import datasets as D  # noqa: E402
from script.week5.timescale_inspection import gates5 as G  # noqa: E402
from script.week5.timescale_inspection import radius as RAD  # noqa: E402
from script.week5.timescale_inspection import run_phase1 as R1  # noqa: E402
from script.week5.timescale_inspection import tokenizers as TOK  # noqa: E402

SCHEMA_VERSION = 1
SEEDS = (111, 222, 1538574472)
ARMS = ("context", "centre", "pca")
WINDOWS_E = (4, 15)
SPANS_D = (2, 4)
SMOKE_SELECTION = "t1_whiten"  # smoke only: a real run derives the selection from BABEL-2
OWN_FILES = tuple(f"script/week5/timescale_inspection/{m}.py" for m in ("tokenizers", "gates5", "run_phase5", "radius"))
TRAIN_FILES = ("script/week5/contextualizer_calibration/train.py", "script/week5/contextualizer_calibration/plan.py",
               "script/week5/contextualizer_calibration/evaluate.py", "script/action_transport/stageb_model.py",
               "script/action_transport/stageb_data.py", "script/quantized_input_comparison/contextual.py",
               "script/action_transport/transport.py", "script/action_transport/categorical.py",
               "script/action_transport/continuous.py", "script/fsq_reconstruction/evaluate.py")
# minutes, measured in rkx_v2 / ts1_v1 / ts4_v1 (the plan's unit-cost table); used only for the dry-run projection
COST = dict(prep=dict(babel2=0.5, babel3=0.5, lara=2.0, hugadb=0.5), train=dict(babel2=1.5, babel3=1.6, lara=2.2, hugadb=2.7),
            context=dict(babel2=2.5, babel3=2.1, lara=14.4, hugadb=8.0),
            direct=dict(babel2=2.1, babel3=1.5, lara=10.6, hugadb=4.6))


def source_hash() -> str:
    h = hashlib.sha256()
    for name in OWN_FILES + TRAIN_FILES:
        h.update(name.encode())
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class Cell:
    dataset: str
    tok: str
    seed: int
    window: int
    span: int

    @property
    def spec(self) -> TOK.TokenizerSpec:
        return TOK.SPECS[self.tok]

    @property
    def guard(self) -> int:
        return self.spec.guard

    @property
    def prep_id(self) -> str:
        base = f"{self.dataset}__{self.tok}__seed{self.seed}"
        return base if self.window == D.SPECS[self.dataset].window else f"{base}__w{self.window}"

    @property
    def cell_id(self) -> str:
        return self.prep_id if self.span == self.spec.span else f"{self.prep_id}__span{self.span}"

    @property
    def is_default(self) -> bool:
        return self.window == D.SPECS[self.dataset].window and self.span == self.spec.span


def mk_cell(dataset: str, tok: str, seed: int, window: int | None = None, span: int | None = None) -> Cell:
    return Cell(dataset, tok, seed, window or D.SPECS[dataset].window, span or TOK.SPECS[tok].span)


@dataclass(frozen=True)
class Item:
    cell: Cell
    arms: tuple[str, ...]


def stage_items(stage: str, seeds=SEEDS, selected: str | None = None) -> list[Item]:
    if stage == "A":
        return [Item(mk_cell(ds, tok, s), ARMS) for ds in ("babel2", "babel3") for tok in ("t1_whiten", "t2_relational", "t4_smq")
                for s in seeds]
    if stage == "B":
        return [Item(mk_cell("hugadb", tok, 111), ("context", "centre")) for tok in ("t1_whiten", "t4_smq")]
    if stage == "C1":
        return [Item(mk_cell("lara", tok, s), ARMS if tok == "t3_lara_pos256" else ("context", "centre"))
                for tok in ("t3_lara_pos256", "t2_relational", "t4_smq") for s in seeds]
    if stage == "C2":
        return [Item(mk_cell("lara", "t3b_lara_pos64", s), ("context", "centre")) for s in seeds]
    if stage == "D" and selected:
        return [Item(mk_cell(ds, selected, s, span=sp), ("context",)) for ds in ("babel2", "babel3") for sp in SPANS_D
                for s in seeds]
    if stage == "E" and selected:
        return [Item(mk_cell(ds, selected, s, window=w), ("context", "centre")) for ds in ("babel2", "babel3")
                for w in WINDOWS_E for s in seeds]
    return []


STAGE_ORDER = ("G", "A", "B", "C1", "P1", "C2", "D", "E", "P2")


def make_job(cell: Cell) -> P.JobSpec:
    """The frozen reference configuration (span 8, guard 0) for T0-T3b; T4 uses `long_mask` (span 16, guard 14)."""
    name = "long_mask" if cell.guard else "reference"
    context, _, stride, batch = P.CONFIGS[name]
    return P.JobSpec(cell.cell_id, "main", name, f"tok5_{cell.tok}", cell.spec.family, "context_id", cell.guard, context,
                     stride, cell.span, 0.30, batch, P.FULL_HORIZON, cell.seed, "none", "")


def select_tokenizer(scores: dict[str, list[float]]) -> str | None:
    """Highest mean BABEL-2 `context` F1@50 over 3 seeds among the guard-0 tokenizers. Never T4 (or T3/T3b). None until
    every candidate has all three seeds."""
    cands = {k: v for k, v in scores.items() if k in TOK.SELECTABLE}
    if not cands or any(len(v) < len(SEEDS) for v in cands.values()):
        return None
    return max(cands, key=lambda k: (statistics.mean(cands[k]), k))


class Runner:
    def __init__(self, args):
        self.args = args
        self.smoke = bool(args.smoke)
        self.scale = replace(P.SMOKE, recording_cap=None) if self.smoke else P.FULL
        self.out = ROOT / "results" / "week5" / "timescale_inspection" / args.run_id / "phase5"
        for sub in ("prep", "jobs", "units", "probes", "span_diag"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.deadline = time.perf_counter() + args.hours * 3600 if args.hours else None
        self.impl = source_hash()
        self.radius = RAD.BY_NAME[args.radius] if args.radius != "current" else RAD.CURRENT
        self.seeds = (111,) if self.smoke else SEEDS
        self.job_infos: dict[str, dict] = {}
        self.loaders: dict[str, Loader] = {}
        self.not_run: dict[str, str] = {}
        self.gates_path = self.out / "gates.json"
        self.gates: dict = json.loads(self.gates_path.read_text()) if self.gates_path.exists() else {}
        self.manifest_path = self.out / "manifest.json"
        self.base_cells = [it.cell for st in ("A", "B", "C1", "C2") for it in stage_items(st, self.seeds)]
        self.check_resume()

    # -- identity ----------------------------------------------------------------------------------------------------
    def recording_cap(self, cell: Cell) -> int | None:
        return D.SPECS[cell.dataset].smoke_cap if self.smoke else None

    def prep_key(self, cell: Cell) -> TOK.PrepKey:
        return TOK.make_key(cell.dataset, cell.tok, cell.seed, cell.window, self.recording_cap(cell))

    def identity_of(self, cell: Cell) -> str:
        return P.stable_hash(dict(job=P.effective_settings(make_job(cell), self.scale), prep=asdict(self.prep_key(cell)),
                                  spec=asdict(TOK.dataset_spec(cell.dataset, cell.window)), tok=asdict(cell.spec),
                                  schema=SCHEMA_VERSION))

    def check_resume(self) -> None:
        ident = dict(schema=SCHEMA_VERSION, implementation_hash=self.impl, scale=self.scale.name, outer_cap=self.scale.outer_cap,
                     rungs=list(self.scale.rungs), radius=self.radius.name,
                     cells={c.cell_id: self.identity_of(c) for c in self.base_cells})
        if self.manifest_path.exists():
            old = json.loads(self.manifest_path.read_text())
            if old.get("identity") != ident:
                raise SystemExit(f"{self.out} was written by a different implementation, radius, scale or cell set; "
                                 f"choose a new --run-id.")
        write_json_atomic(self.manifest_path, dict(
            identity=ident, run_id=self.args.run_id, phase=5, status="running", smoke=self.smoke,
            python=platform.python_version(), started_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            radius=self.radius.name))
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
        if cached and cached.get("status") == "passed" and cached.get("implementation_hash") == self.impl:
            return cached
        t0 = time.perf_counter()
        try:
            res = fn()
        except Exception as exc:  # a crashing gate is a failed gate, never a silent pass
            res = dict(status="failed", reason=f"{type(exc).__name__}: {exc}")
        res.update(implementation_hash=self.impl, seconds=time.perf_counter() - t0)
        self.gates[name] = res
        write_json_atomic(self.gates_path, self.gates)
        print(f"[gate] {name}: {res['status']} ({res['seconds']:.0f}s)" + (f" -- {res.get('reason')}" if res["status"] != "passed" else ""),
              flush=True)
        return res

    def run_gates(self) -> None:
        if self.smoke:
            self.gates["g1"] = dict(status="not_applicable_smoke", implementation_hash=self.impl)
        else:
            self._gate("g1", lambda: G.gate_g1(self.out / "prep" / "babel2__t0_raw__seed111"))
        datasets = sorted({c.dataset for c in self.base_cells if c.tok == "t4_smq"})
        for ds in datasets:
            self._gate(f"g2_{ds}", lambda ds=ds: G.gate_g2(ds, self.device))
        if any(c.dataset == "lara" for c in self.base_cells):
            self._gate("g3", G.gate_g3)
        write_json_atomic(self.gates_path, self.gates)

    def blocked(self, cell: Cell) -> str | None:
        if self.gates.get("g1", {}).get("status") == "failed":
            return "G1 failed (the generalized preparation does not reproduce rkx_v2)"
        if cell.tok == "t4_smq" and self.gates.get(f"g2_{cell.dataset}", {}).get("status") != "passed":
            return f"G2 not passed for {cell.dataset} (SMQ checkpoint does not reproduce its saved re-score)"
        if cell.dataset == "lara" and cell.tok in ("t2_relational", "t3_lara_pos256", "t3b_lara_pos64") \
                and self.gates.get("g3", {}).get("status") != "passed":
            return "G3 not passed (LARa channels 3-5 are not confirmed as positions)"
        return None

    # -- preparation -------------------------------------------------------------------------------------------------
    def prep_dir(self, cell: Cell) -> Path:
        return self.out / "prep" / cell.tok / cell.prep_id

    def prepare(self, cell: Cell) -> Loader | None:
        if cell.prep_id in self.loaders:
            return self.loaders[cell.prep_id]
        d = self.prep_dir(cell)
        try:
            if TOK.is_prepared(cell.dataset, cell.tok, cell.seed, d, cell.window, self.recording_cap(cell)):
                self.loaders[cell.prep_id] = Loader(d)
                return self.loaders[cell.prep_id]
        except (OSError, KeyError, ValueError):
            pass
        if self.expired():
            return None
        t0 = time.perf_counter()
        TOK.prepare_cell(cell.dataset, cell.tok, cell.seed, d, cell.window, self.recording_cap(cell), self.device)
        print(f"[prep] {cell.prep_id} ({time.perf_counter() - t0:.0f}s)", flush=True)
        self.loaders[cell.prep_id] = Loader(d)
        return self.loaders[cell.prep_id]

    # -- training ----------------------------------------------------------------------------------------------------
    def job_dir(self, cell: Cell) -> Path:
        return self.out / "jobs" / cell.tok / cell.cell_id

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
            ledger = json.loads((jd / "checkpoints.json").read_text())["ledger"]
            row = next(r for r in ledger if r["step"] == step)
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
            info = T.run_training_job(job, self.scale, loader, jd, self.device, self.deadline, self.identity_of(cell), self.impl)
        except DeadlineExceeded:
            return False
        write_json_atomic(jd / "training_info.json", info)
        self.job_infos[cell.cell_id] = info
        print(f"[train] {cell.cell_id} done (selected step {info['roles']['selected']}, val loss {info['best_val_loss']:.4f})",
              flush=True)
        return True

    # -- decoding ----------------------------------------------------------------------------------------------------
    def unit_dir(self, cell: Cell, arm: str) -> Path:
        return self.out / "units" / f"{cell.cell_id}__{arm}"

    def unit_identity(self, cell: Cell, arm: str, state_sha: str | None) -> str:
        return hashlib.sha256(json.dumps([cell.cell_id, arm, state_sha, self.radius.name, self.impl, self.scale.outer_cap,
                                          list(self.scale.rungs), self.scale.name, self.identity_of(cell)]).encode()).hexdigest()[:16]

    def unit_done(self, cell: Cell, arm: str, identity: str) -> dict | None:
        path = self.unit_dir(cell, arm) / "info.json"
        if not path.exists():
            return None
        try:
            info = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return info if info.get("status") == "done" and info.get("identity") == identity else None

    def build_recs(self, cell: Cell, arm: str, loader: Loader, names: list[str], state: dict | None):
        family = cell.spec.family
        spec = TOK.dataset_spec(cell.dataset, cell.window)
        codes, lens, starts = {}, {}, {}
        for n in names:
            rec = loader.recording(family, n)
            codes[n], lens[n], starts[n] = rec["codes"], rec["lengths"], rec["starts"]
        z, zero = {}, {}
        if arm == "context":
            job = make_job(cell)
            model = E.build_model_from_state(job, self.scale, loader, state, self.device)
            lookup, rms = loader.lookup(family), float(loader.rms(family))
            for n in names:
                z[n], zero[n] = E.extract_embeddings(job.arm, model, loader, family, n, lookup, rms, self.device,
                                                      job.context, job.stride)
        else:
            mean = loader.mean(family).astype(np.float64)
            lookup = loader.lookup(family)
            for n in names:
                rec = loader.recording(family, n)
                raw = lookup[rec["codes"]] if arm == "centre" else rec["continuous"]
                z[n], zero[n] = continuous.l2_normalize(raw.astype(np.float64) - mean[None, :])
        recs = [continuous.ContinuousRecordingInput(n, codes[n].astype(np.int32), lens[n].astype(np.int32), spec.fps,
                                                    spec.window, z[n].astype(np.float64), zero[n]) for n in names]
        mu0, _ = continuous.build_mu0(loader.grouping(family), recs, spec.num_states)
        return recs, mu0, starts

    def ensure_decode(self, cell: Cell, arm: str) -> str:
        info = self.job_state(cell) if arm == "context" else {}
        if arm == "context" and info is None:
            return "no_checkpoint"
        state, state_sha = (self.selected_state(cell, info) if arm == "context" else (None, None))
        identity = self.unit_identity(cell, arm, state_sha)
        if self.unit_done(cell, arm, identity):
            return "cached"
        prior = self.unit_dir(cell, arm) / "info.json"
        if prior.exists():
            try:
                old = json.loads(prior.read_text())
            except (OSError, json.JSONDecodeError):
                old = {}
            if old.get("status") == "invalid" and old.get("identity") == identity:
                return "invalid"  # a deterministic invalid solve is final
        if self.expired():
            return "deadline"
        loader = self.prepare(cell)
        if loader is None:
            return "deadline"
        spec = TOK.dataset_spec(cell.dataset, cell.window)
        names = sorted(loader.recording_names())
        recs, mu0, starts = self.build_recs(cell, arm, loader, names, state)
        gts = D.load_gts(spec, names)
        frame_counts = {n: loader.frame_count(n) for n in names}
        udir = self.unit_dir(cell, arm)
        meta = dict(dataset=cell.dataset, tokenizer=cell.tok, seed=cell.seed, window=cell.window, span=cell.span,
                    guard=cell.guard, arm=arm, radius=self.radius.name, unit=f"{cell.cell_id}__{arm}")
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

    # -- items and stages --------------------------------------------------------------------------------------------
    def run_item(self, item: Item, counts: dict, tag: str) -> None:
        cell = item.cell
        reason = self.blocked(cell)
        if reason:
            self.not_run[cell.cell_id] = reason
            counts["blocked"] = counts.get("blocked", 0) + len(item.arms)
            print(f"[{tag}] {cell.cell_id}: BLOCKED -- {reason}", flush=True)
            return
        if self.expired():
            counts["deadline"] = counts.get("deadline", 0) + len(item.arms)
            return
        trained = "context" not in item.arms or self.ensure_train(cell)
        for arm in item.arms:
            if arm == "context" and not trained:
                counts["no_checkpoint"] = counts.get("no_checkpoint", 0) + 1
                continue
            t0 = time.perf_counter()
            status = self.ensure_decode(cell, arm)
            counts[status] = counts.get(status, 0) + 1
            print(f"[{tag}] {cell.cell_id} {arm}: {status} ({time.perf_counter() - t0:.0f}s)", flush=True)

    # -- selection ---------------------------------------------------------------------------------------------------
    def selection_scores(self) -> dict[str, list[float]]:
        """BABEL-2 `context` F1@50, one per seed, for each selectable tokenizer: T0 from rkx_v2's saved results,
        T1/T2 from this run's finished units. BABEL-2 only."""
        from script.week5.timescale_inspection import report4, report5
        base = report4.f1s(report4.load_context_baseline(self.args.context_run).get(("babel2", "global")))
        scores = {"t0_raw": [base[s] for s in self.seeds if s in base]}
        units = report5.load_units(self.out)
        for tok in ("t1_whiten", "t2_relational"):
            rows = units.get(("babel2", tok, "context", D.SPECS["babel2"].window, TOK.SPECS[tok].span), {})
            scores[tok] = [rows[s]["F1_50"] for s in self.seeds if s in rows]
        return scores

    def ensure_selection(self) -> str | None:
        path = self.out / "selection.json"
        if path.exists():
            return json.loads(path.read_text())["selected"]  # frozen once written
        if self.smoke:
            write_json_atomic(path, dict(selected=SMOKE_SELECTION, smoke=True))
            return SMOKE_SELECTION
        scores = self.selection_scores()
        chosen = select_tokenizer(scores)
        if chosen is None:
            return None
        write_json_atomic(path, dict(selected=chosen, scores=scores, dataset="babel2", metric="context F1@50, last rung",
                                     rule="highest mean over 3 seeds among T0/T1/T2", written_local=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
        print(f"[selection] {chosen} (BABEL-2 context F1@50 means: "
              + ", ".join(f"{k} {statistics.mean(v):.2f}" for k, v in scores.items() if v) + ")", flush=True)
        return chosen

    # -- probes ------------------------------------------------------------------------------------------------------
    def probe_cells(self, seeds) -> list[Cell]:
        return [c for c in self.base_cells if c.seed in seeds and c.is_default]

    def run_probes(self, seeds, label: str) -> None:
        """Phase 0.4's per-token probes plus a 1-NN on standardized (whitened) PCA features, per family; and the same
        whitened 1-NN for the T0 baseline (rkx_v2's saved artifacts), so all three probes have a T0 reference."""
        from script.week5.timescale_inspection import probes as PR
        for cell in self.probe_cells(seeds):
            if self.expired():
                return
            if self.blocked(cell):
                continue
            path = self.out / "probes" / f"{cell.cell_id}.json"
            info = self.job_state(cell)
            if path.exists():
                try:
                    old = json.loads(path.read_text())
                    if old.get("has_context") or info is None:
                        continue
                except (OSError, json.JSONDecodeError):
                    pass
            loader = self.prepare(cell)
            if loader is None:
                continue
            spec = TOK.dataset_spec(cell.dataset, cell.window)
            names = sorted(loader.recording_names())
            gts = D.load_gts(spec, names)
            emb = None
            if info is not None:
                state, _ = self.selected_state(cell, info)
                job = make_job(cell)
                model = E.build_model_from_state(job, self.scale, loader, state, self.device)
                lookup, rms = loader.lookup(cell.spec.family), float(loader.rms(cell.spec.family))
                emb = {n: E.extract_embeddings(job.arm, model, loader, cell.spec.family, n, lookup, rms, self.device,
                                               job.context, job.stride)[0] for n in names}
            table = PR.token_table(spec, loader, gts, emb, family=cell.spec.family)
            try:
                rows = PR.run_probes(table) + PR.run_white_nn(table)
            except ValueError as exc:  # e.g. every capped smoke recording belongs to one test group; never abort the run
                print(f"[{label}] {cell.cell_id}: probes skipped ({exc})", flush=True)
                continue
            write_json_atomic(path, dict(cell=cell.cell_id, dataset=cell.dataset, tokenizer=cell.tok, seed=cell.seed,
                                         has_context=emb is not None, rows=rows))
            print(f"[{label}] {cell.cell_id}: " + ", ".join(f"{r['representation']}/{r['probe']} {r['balanced_accuracy']:.3f}"
                                                            for r in rows if r["representation"] == "pca"), flush=True)
        # T0 reference for the whitened 1-NN (the other probes come from Phase 0's probes.csv)
        for ds in sorted({c.dataset for c in self.probe_cells(seeds)}):
            for seed in seeds:
                path = self.out / "probes" / f"t0_baseline__{ds}__seed{seed}.json"
                ref = ROOT / "results" / "week5" / "raw_kmeans_cross_dataset" / self.args.context_run / "prep" / f"{ds}__global__seed{seed}"
                if path.exists() or not ref.exists() or self.smoke or self.expired():
                    continue
                spec = D.SPECS[ds]
                loader = Loader(ref)
                names = sorted(loader.recording_names())
                table = PR.token_table(spec, loader, D.load_gts(spec, names), None)
                write_json_atomic(path, dict(dataset=ds, tokenizer="t0_raw", seed=seed, rows=PR.run_white_nn(table)))

    def run_span_diagnostics(self, selected: str) -> None:
        """Phase 0.4b's masked-loss split, recomputed for every span of the selected tokenizer (span 8 from stage A)."""
        from script.week5.timescale_inspection import probes as PR
        cells = [it.cell for it in stage_items("D", self.seeds, selected)]
        cells += [mk_cell(ds, selected, s) for ds in ("babel2", "babel3") for s in self.seeds]
        for cell in cells:
            path = self.out / "span_diag" / f"{cell.cell_id}.json"
            info = self.job_state(cell)
            if info is None or path.exists() or self.expired():
                continue
            loader = self.prepare(cell)
            if loader is None:
                continue
            spec = TOK.dataset_spec(cell.dataset, cell.window)
            names = sorted(loader.recording_names())
            gts = D.load_gts(spec, names)
            state, _ = self.selected_state(cell, info)
            job = make_job(cell)
            model = E.build_model_from_state(job, self.scale, loader, state, self.device)
            val_names = T.bounded_split(loader, None)[2]
            res = PR.masked_loss_split(model, loader, val_names, gts, context=job.context, stride=job.stride,
                                       fraction=job.mask_fraction, span=job.mask_span, mask_base=T.VALIDATION_MASK_BASE,
                                       device=self.device, family=cell.spec.family, radius=job.guard_radius)
            write_json_atomic(path, dict(cell=cell.cell_id, dataset=cell.dataset, seed=cell.seed, span=cell.span, **res))
            print(f"[span_diag] {cell.cell_id}: crossing share {res['share_targets_crossing']:.2f}, within loss "
                  f"{res['within_loss']:.3f}, crossing loss {res['crossing_loss']:.3f}", flush=True)

    # -- orchestration -----------------------------------------------------------------------------------------------
    def execute(self) -> None:
        counts: dict[str, int] = {}
        self.run_gates()
        selected = None
        for stage in STAGE_ORDER[1:]:
            if stage in ("P1", "P2"):
                self.run_probes((111,) if stage == "P1" else self.seeds, stage)
                continue
            if stage in ("D", "E"):
                selected = selected or self.ensure_selection()
                if selected is None:
                    print(f"[{stage}] selection not available (stage A incomplete); skipped", flush=True)
                    continue
            items = stage_items(stage, self.seeds, selected)
            for i, item in enumerate(items, 1):
                self.run_item(item, counts, f"{stage} {i}/{len(items)}")
            if stage == "A":
                selected = self.ensure_selection()
            if stage == "D" and selected:
                self.run_span_diagnostics(selected)
        m = json.loads(self.manifest_path.read_text())
        m.update(status="finished" if not self.expired() else "budget_stop", counts=counts,
                 not_run=self.not_run, finished_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        write_json_atomic(self.manifest_path, m)
        from script.week5.timescale_inspection import report5
        print(f"report: {report5.write_report(self.out, self.args.context_run, self.args.direct_run)}")

    def dry_run(self) -> None:
        selected = SMOKE_SELECTION if self.smoke else "t1_whiten"  # stand-in: cost depends on the dataset, not the tokenizer
        total, seen = 0.1 * 60, set()  # gates ~6 min
        print(f"stage G: gates G1-G3 (~6 min)")
        for stage in STAGE_ORDER[1:]:
            if stage in ("P1", "P2"):
                n = len(self.probe_cells((111,) if stage == "P1" else self.seeds))
                print(f"stage {stage}: per-token probes (~1 min per family cell)")
                total += 1.0 * n
                continue
            items = stage_items(stage, self.seeds, selected)
            minutes = 0.0
            for it in items:
                c = it.cell
                if c.prep_id not in seen:
                    seen.add(c.prep_id)
                    minutes += COST["prep"][c.dataset]
                if "context" in it.arms:
                    minutes += COST["train"][c.dataset] + COST["context"][c.dataset]
                minutes += sum(COST["direct"][c.dataset] for a in it.arms if a != "context")
                print(f"  {c.cell_id}: arms {','.join(it.arms)}"
                      + (f"  [guard {c.guard}, span {c.span}]" if "context" in it.arms else ""))
            total += minutes
            print(f"stage {stage}: {len(items)} cells, ~{minutes:.0f} min, cumulative ~{total / 60:.1f} h"
                  + ("  (tokenizer chosen after stage A; T1 used as a cost stand-in)" if stage in ("D", "E") else ""))
        print(f"projected total ~{total / 60:.1f} h (single-pass, otherwise idle machine)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Time-scale inspection, phase 5 (tokenizer, then span and window)")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--hours", type=float, default=None, help="elapsed-time budget; completed units are kept")
    ap.add_argument("--radius", default="current", choices=list(RAD.BY_NAME))
    ap.add_argument("--context-run", default="rkx_v2", help="position-only run to read the T0 `context` baseline from")
    ap.add_argument("--direct-run", default="ts1_v1", help="Phase 1 run to read the T0 `pca`/`centre` baseline from")
    ap.add_argument("--smoke", action="store_true", help="tiny budgets, capped recordings, seed 111 only (plumbing check)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not (args.dry_run or args.smoke) and not args.hours:
        raise SystemExit("a decoding run requires an explicit elapsed-time budget: pass --hours")
    runner = Runner(args)
    if args.dry_run:
        runner.dry_run()
    else:
        runner.execute()


if __name__ == "__main__":
    main()
