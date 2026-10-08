"""Phase 4 of the time-scale inspection: input representation (H4).

Builds a second raw-patch family -- positions plus first differences
(velocities, frame-rate scaled, zero at frame 0) -- for BABEL-2, BABEL-3 and
LARa (global), 3 seeds each, trains the frozen reference contextualizer on it
exactly as `raw_kmeans_cross_dataset` does, and decodes three ways at the
current OT radius (Decision 1 is not finalized; `--radius` can override):

  context         the trained contextualizer's embedding (needs training)
  pca_velocity    the raw PCA-64 feature of the velocity patch, no contextualizer
  centre_velocity the K-means centre of its code, no contextualizer

The `pca_velocity` / `centre_velocity` arms reuse `run_phase1.fit_and_score`
directly (same decoder, same scorer), so a result is comparable point for point
with Phase 1's `pca` / `centre` arms on the position-only family. HuGaDB is not
rebuilt: it is already derivative (IMU) data and serves as the control via its
existing `rkx_v2` result. The position-only baseline for BABEL/LARa is `rkx_v2`
(context) and the Phase 1 `inputs` stage (pca/centre) -- not rebuilt here, since
rebuilding would add a second vocabulary-seed draw and make the comparison
noisier, not cleaner.

Usage:
  python -m script.week5.timescale_inspection.run_phase4 --run-id ts4_v1 --hours 6
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
from dataclasses import asdict, dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from script.action_transport import continuous  # noqa: E402
from script.action_transport.budget import DeadlineExceeded  # noqa: E402
from script.fsq_reconstruction.evaluate import FinalSolveInvalid  # noqa: E402
from script.fsq_reconstruction.experiment import write_json_atomic  # noqa: E402
from script.quantized_input_comparison import ot_arms as ot  # noqa: E402
from script.quantized_input_comparison.artifacts import Loader  # noqa: E402
from script.week5.contextualizer_calibration import evaluate as E  # noqa: E402
from script.week5.contextualizer_calibration import plan as P  # noqa: E402
from script.week5.contextualizer_calibration import train as T  # noqa: E402
from script.week5.raw_kmeans_cross_dataset import datasets as D  # noqa: E402
from script.week5.timescale_inspection import radius as RAD  # noqa: E402
from script.week5.timescale_inspection import run_phase1 as R1  # noqa: E402
from script.week5.timescale_inspection import velocity_prepare as VPREP  # noqa: E402

SCHEMA_VERSION = 1
SEEDS = (111, 222, 1538574472)
CELLS_SPEC = (("babel2", "global"), ("babel3", "global"), ("lara", "global"))  # HuGaDB excluded: already IMU data
OUTER_CAP = 50
RUNGS = (500, 1000, 2000, 4000)
ARMS = ("context", "pca_velocity", "centre_velocity")
OWN_FILES = tuple(f"script/week5/timescale_inspection/{m}.py"
                  for m in ("velocity_prepare", "run_phase4", "radius", "report4"))
TRAIN_FILES = ("script/week5/contextualizer_calibration/train.py", "script/week5/contextualizer_calibration/plan.py",
              "script/week5/contextualizer_calibration/evaluate.py", "script/action_transport/stageb_model.py",
              "script/action_transport/stageb_data.py", "script/quantized_input_comparison/contextual.py",
              "script/action_transport/transport.py", "script/action_transport/categorical.py",
              "script/action_transport/continuous.py", "script/fsq_reconstruction/evaluate.py")


def source_hash() -> str:
    h = hashlib.sha256()
    for name in OWN_FILES + TRAIN_FILES:
        h.update(name.encode())
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class Cell:
    dataset: str
    normalization: str
    seed: int

    @property
    def cell_id(self) -> str:
        return f"{self.dataset}__{self.normalization}__seed{self.seed}"


def all_cells() -> list[Cell]:
    return [Cell(d, n, s) for d, n in CELLS_SPEC for s in SEEDS]


def make_job(cell: Cell) -> P.JobSpec:
    """The same frozen reference configuration as `raw_kmeans_cross_dataset.make_job`, on the velocity family."""
    context, span, stride, batch = P.CONFIGS["reference"]
    return P.JobSpec(cell.cell_id, "main", "reference", "raw_velocity", VPREP.FAMILY, "context_id", 0, context,
                     stride, span, 0.30, batch, P.FULL_HORIZON, cell.seed, "none", "")


class Runner:
    def __init__(self, args):
        self.args = args
        self.out = ROOT / "results" / "week5" / "timescale_inspection" / args.run_id / "phase4"
        (self.out / "prep").mkdir(parents=True, exist_ok=True)
        (self.out / "jobs").mkdir(exist_ok=True)
        (self.out / "units").mkdir(exist_ok=True)
        self.deadline = time.perf_counter() + args.hours * 3600 if args.hours else None
        self.impl = source_hash()
        self.radius = RAD.BY_NAME[args.radius] if args.radius != "current" else RAD.CURRENT
        self.cells = all_cells()
        self.specs = {d: D.SPECS[d] for d, _ in CELLS_SPEC}
        self.job_infos: dict[str, dict] = {}
        self.loaders: dict[str, Loader] = {}
        self.not_run: dict[str, dict] = {}
        self.manifest_path = self.out / "manifest.json"
        self.check_resume()

    def identity_of(self, cell: Cell) -> str:
        key = VPREP.PrepKey(cell.dataset, cell.normalization, cell.seed, self.specs[cell.dataset].window,
                            self.specs[cell.dataset].num_states)
        return P.stable_hash(dict(job=P.effective_settings(make_job(cell), P.FULL), prep=asdict(key),
                                  spec=asdict(self.specs[cell.dataset]), schema=SCHEMA_VERSION))

    def check_resume(self) -> None:
        ident = dict(schema=SCHEMA_VERSION, implementation_hash=self.impl, outer_cap=OUTER_CAP, rungs=list(RUNGS),
                     radius=self.radius.name, cells={c.cell_id: self.identity_of(c) for c in self.cells})
        if self.manifest_path.exists():
            old = json.loads(self.manifest_path.read_text())
            if old.get("identity") != ident:
                raise SystemExit(f"{self.out} was written by a different implementation, radius or cell set; choose "
                                 f"a new --run-id.")
        write_json_atomic(self.manifest_path, dict(
            identity=ident, run_id=self.args.run_id, phase=4, status="running", python=platform.python_version(),
            started_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"), radius=self.radius.name))
        self.ident = ident

    def expired(self) -> bool:
        return self.deadline is not None and time.perf_counter() > self.deadline

    # -- preparation -----------------------------------------------------------------------------
    def prep_dir(self, cell: Cell) -> Path:
        return self.out / "prep" / cell.cell_id

    def prepare(self, cell: Cell) -> Loader | None:
        if cell.cell_id in self.loaders:
            return self.loaders[cell.cell_id]
        try:
            if VPREP.is_prepared(self.specs[cell.dataset], cell.normalization, cell.seed, self.prep_dir(cell)):
                self.loaders[cell.cell_id] = Loader(self.prep_dir(cell))
                return self.loaders[cell.cell_id]
        except (OSError, KeyError, ValueError):
            pass
        if self.expired():
            return None
        VPREP.prepare_cell(self.specs[cell.dataset], cell.normalization, cell.seed, self.prep_dir(cell))
        self.loaders[cell.cell_id] = Loader(self.prep_dir(cell))
        return self.loaders[cell.cell_id]

    # -- training ----------------------------------------------------------------------------------
    def job_dir(self, cell: Cell) -> Path:
        return self.out / "jobs" / cell.cell_id

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
            shutil.rmtree(jd)  # an incomplete fit is restarted from initialization, as in raw_kmeans_cross_dataset
        jd.mkdir(parents=True)
        print(f"[train] {cell.cell_id} ({P.FULL.horizon(P.FULL_HORIZON)} updates) ...", flush=True)
        try:
            info = T.run_training_job(make_job(cell), P.FULL, loader, jd, self.device, self.deadline,
                                      self.identity_of(cell), self.impl)
        except DeadlineExceeded:
            return False
        write_json_atomic(jd / "training_info.json", info)
        self.job_infos[cell.cell_id] = info
        print(f"[train] {cell.cell_id} done (selected step {info['roles']['selected']}, "
              f"val loss {info['best_val_loss']:.4f})", flush=True)
        return True

    # -- decoding ------------------------------------------------------------------------------
    def unit_dir(self, cell: Cell, arm: str) -> Path:
        return self.out / "units" / f"{cell.cell_id}__{arm}"

    def unit_identity(self, cell: Cell, arm: str, state_sha: str | None) -> str:
        return hashlib.sha256(json.dumps([cell.cell_id, arm, state_sha, self.ident["radius"], self.impl, OUTER_CAP,
                                          list(RUNGS)]).encode()).hexdigest()[:16]

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
        family = VPREP.FAMILY
        codes, lens, starts = {}, {}, {}
        for n in names:
            rec = loader.recording(family, n)
            codes[n], lens[n], starts[n] = rec["codes"], rec["lengths"], rec["starts"]
        spec = self.specs[cell.dataset]
        if arm == "context":
            job = make_job(cell)
            model = E.build_model_from_state(job, P.FULL, loader, state, self.device)
            lookup, rms = loader.lookup(family), float(loader.rms(family))
            z, zero = {}, {}
            for n in names:
                z[n], zero[n] = E.extract_embeddings(job.arm, model, loader, family, n, lookup, rms, self.device,
                                                      job.context, job.stride)
        else:
            mean = loader.mean(family).astype(np.float64)
            lookup = loader.lookup(family)
            z, zero = {}, {}
            for n in names:
                rec = loader.recording(family, n)
                raw = lookup[rec["codes"]] if arm == "centre_velocity" else rec["continuous"]
                z[n], zero[n] = continuous.l2_normalize(raw.astype(np.float64) - mean[None, :])
        recs = [continuous.ContinuousRecordingInput(n, codes[n].astype(np.int32), lens[n].astype(np.int32), spec.fps,
                                                    spec.window, z[n].astype(np.float64), zero[n]) for n in names]
        mu0, _ = continuous.build_mu0(loader.grouping(family), recs, spec.num_states)
        return recs, mu0, starts

    def ensure_decode(self, cell: Cell, arm: str) -> str:
        info = self.job_state(cell) if arm == "context" else {}
        if arm == "context" and info is None:
            return "no_checkpoint"
        state_sha = None
        state = None
        if arm == "context":
            jd = self.job_dir(cell)
            step = info["roles"]["selected"]
            row = next(r for r in json.loads((jd / "checkpoints.json").read_text())["ledger"] if r["step"] == step)
            state = T.load_state(jd / "ckpt" / row["file"])["state_dict"]
            state_sha = row["tensor_sha256"]
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
                return "invalid"  # a deterministic invalid solve is final, not retried on resume
        if self.expired():
            return "deadline"
        loader = self.prepare(cell)
        if loader is None:
            return "deadline"
        spec = self.specs[cell.dataset]
        names = sorted(loader.recording_names())
        recs, mu0, starts = self.build_recs(cell, arm, loader, names, state)
        gts = D.load_gts(spec, names)
        frame_counts = {n: loader.frame_count(n) for n in names}
        try:
            res = R1.fit_and_score(recs, mu0, spec.num_states, R1.coeffs_for(0.3), self.radius, OUTER_CAP, RUNGS,
                                   gts, starts, frame_counts, self.deadline)
        except DeadlineExceeded:
            return "deadline"
        except FinalSolveInvalid as exc:
            udir = self.unit_dir(cell, arm)
            udir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(udir / "info.json", dict(status="invalid", identity=identity, reason=str(exc),
                                                       dataset=cell.dataset, seed=cell.seed, arm=arm))
            return "invalid"
        udir = self.unit_dir(cell, arm)
        udir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(udir / "info.json", dict(
            status="done", identity=identity, unit=f"{cell.cell_id}__{arm}", dataset=cell.dataset,
            normalization=cell.normalization, seed=cell.seed, arm=arm, radius=self.radius.name,
            representation="position_velocity", checkpoint_step=(info or {}).get("roles", {}).get("selected"),
            ot_status=res["ot_status"], outer_iterations=res["outer_iterations"],
            segment_stats=res["segment_stats"], half_widths_used=res["half_widths_used"], seconds=res["seconds"],
            ladder=res["ladder"]))
        return "done"

    def run_probes(self) -> None:
        """Phase 0.4's per-token probes (same split, caps and classifiers) on the velocity family's PCA feature,
        K-means centre and code; `context` from the trained velocity checkpoint when it exists. Compared in the
        report against Phase 0's position-only `probes.csv`."""
        from script.week5.timescale_inspection import probes as PR
        rows = []
        for cell in self.cells:
            loader = self.prepare(cell)
            if loader is None:
                continue
            spec = self.specs[cell.dataset]
            names = sorted(loader.recording_names())
            gts = D.load_gts(spec, names)
            emb = None
            info = self.job_state(cell)
            if info is not None and "context" in self.args.arms:
                jd = self.job_dir(cell)
                row = next(r for r in json.loads((jd / "checkpoints.json").read_text())["ledger"]
                           if r["step"] == info["roles"]["selected"])
                state = T.load_state(jd / "ckpt" / row["file"])["state_dict"]
                job = make_job(cell)
                model = E.build_model_from_state(job, P.FULL, loader, state, self.device)
                lookup, rms = loader.lookup(VPREP.FAMILY), float(loader.rms(VPREP.FAMILY))
                emb = {n: E.extract_embeddings(job.arm, model, loader, VPREP.FAMILY, n, lookup, rms, self.device,
                                               job.context, job.stride)[0] for n in names}
            table = PR.token_table(spec, loader, gts, emb, family=VPREP.FAMILY)
            for r in PR.run_probes(table):
                rows.append(dict(dataset=cell.dataset, normalization=cell.normalization, seed=cell.seed,
                                 representation_family="position_velocity", **r))
            print(f"[probes] {cell.cell_id}: " + ", ".join(
                f"{r['representation']}/{r['probe']} {r['balanced_accuracy']:.3f}" for r in rows[-8:]), flush=True)
        from script.week5.timescale_inspection.run_phase0 import write_csv
        write_csv(self.out / "probes.csv", rows)

    # -- orchestration ---------------------------------------------------------------------------
    @property
    def device(self):
        import torch
        return torch.device("cuda" if torch.cuda.is_available() and not self.args.cpu else "cpu")

    def execute(self) -> None:
        counts: dict[str, int] = {}
        for i, cell in enumerate(self.cells, 1):
            if self.expired():
                counts["deadline"] = counts.get("deadline", 0) + 1
                continue
            trained = "context" not in self.args.arms or self.ensure_train(cell)
            if not trained:
                print(f"[{i}/{len(self.cells)}] {cell.cell_id}: training not finished (deadline); the context arm "
                      f"is skipped for now", flush=True)
            for arm in self.args.arms:
                if arm == "context" and not trained:
                    counts["no_checkpoint"] = counts.get("no_checkpoint", 0) + 1
                    continue
                t0 = time.perf_counter()
                status = self.ensure_decode(cell, arm)
                counts[status] = counts.get(status, 0) + 1
                print(f"[{i}/{len(self.cells)}] {cell.cell_id} {arm}: {status} ({time.perf_counter() - t0:.0f}s)",
                     flush=True)
        if self.args.probes and not self.expired():
            self.run_probes()
        m = json.loads(self.manifest_path.read_text())
        m.update(status="finished", counts=counts, finished_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        write_json_atomic(self.manifest_path, m)
        from script.week5.timescale_inspection import report4
        datasets = tuple(dict.fromkeys(d for d, _ in CELLS_SPEC))
        print(f"report: {report4.write_report(self.out, self.args.context_run, self.args.direct_run, datasets)}")

    def dry_run(self) -> None:
        for cell in self.cells:
            for arm in self.args.arms:
                ident = self.unit_identity(cell, arm, None if arm != "context" else "pending")
                print(f"{cell.cell_id} {arm}: would run (identity depends on the trained checkpoint for `context`)")
        print(f"{len(self.cells)} cells x {len(self.args.arms)} arms = {len(self.cells) * len(self.args.arms)} units")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Time-scale inspection, phase 4 (input representation)")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--hours", type=float, default=None, help="elapsed-time budget; completed units are kept")
    ap.add_argument("--radius", default="current", choices=list(RAD.BY_NAME),
                    help="OT kernel radius setting (Decision 1 default: current)")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--context-run", default="rkx_v2", help="position-only run to read the `context` baseline from")
    ap.add_argument("--direct-run", default="ts1_v1", help="Phase 1 run to read the `pca`/`centre` baseline from")
    ap.add_argument("--no-probes", dest="probes", action="store_false",
                    help="skip the per-token probes at the end (they take a few minutes)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not args.dry_run and not args.hours:
        raise SystemExit("a decoding run requires an explicit elapsed-time budget: pass --hours")
    runner = Runner(args)
    if args.dry_run:
        runner.dry_run()
    else:
        runner.execute()


if __name__ == "__main__":
    main()
