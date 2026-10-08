"""Phase 1 of the time-scale inspection: the OT kernel radius, re-decoding saved checkpoints.

Reads the verified selected-checkpoint embeddings of a raw K-means cross-dataset run
(`rkx_v2`), repeats the full contextual OT fit (outer EM, then the final rung ladder)
with only the kernel radius (and, in the beta stage, beta) changed, and scores with
the repository scorer. No training; never writes into the source run.

Stages (each resumable; unit results are verified by identity before reuse):
  gate     HuGaDB / global / seed 111 at the `current` radius must reproduce the source run exactly
  screen   BABEL-2 and BABEL-3, 3 seeds, the 12 radius settings (development set: BABEL-2)
  confirm  chosen settings on BABEL-1, HuGaDB, LARa(global); the settings are SELECTED FROM BABEL-2 ONLY
  beta     BABEL-2, the two chosen radii, beta in {0.1, 0.5} (0.3 comes from the screen)
  inputs         no-contextualizer inputs at the current radius: `pca` (the patch's PCA-64 feature, mean-subtracted,
                 L2-normalized) and `centre` (its K-means centre, same treatment) on every dataset, 3 seeds. These are
                 the repository's `direct_continuous` / `direct_quantized` arms; no checkpoint is involved
  inputs_chosen  the same two inputs at the radii chosen on BABEL-2 (needs `selection.json` from the screen)

Usage:
  python -m script.week5.timescale_inspection.run_phase1 --run-id ts1_v1 --stage screen --hours 4
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
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from script.action_transport import continuous, transport  # noqa: E402
from script.action_transport.budget import DeadlineExceeded, check_deadline  # noqa: E402
from script.fsq_reconstruction.evaluate import FinalSolveInvalid, contextual_final_ladder  # noqa: E402
from script.fsq_reconstruction.experiment import write_json_atomic  # noqa: E402
from script.quantized_input_comparison import ot_arms as ot  # noqa: E402
from script.quantized_input_comparison import artifacts  # noqa: E402
from script.quantized_input_comparison.artifacts import Loader  # noqa: E402
from script.week5.contextualizer_calibration import plan as P  # noqa: E402
from script.week5.raw_kmeans_cross_dataset import datasets as D  # noqa: E402
from script.week5.raw_kmeans_cross_dataset.prepare import FAMILY  # noqa: E402
from script.week5.timescale_inspection import radius as RAD  # noqa: E402
from script.week5.timescale_inspection import run_phase0 as P0  # noqa: E402

SCHEMA_VERSION = 1
SOURCE_RUN = "rkx_v2"
SEEDS = (111, 222, 1538574472)
RUNGS = (500, 1000, 2000, 4000)
OUTER_CAP = 50
BETAS = (0.1, 0.3, 0.5)
SCREEN_DATASETS = ("babel2", "babel3")
CONFIRM_DATASETS = ("babel1", "hugadb", "lara")
DEV_DATASET = "babel2"
OWN_FILES = tuple(f"script/week5/timescale_inspection/{m}.py" for m in ("radius", "run_phase1", "report1"))
OT_FILES = ("script/action_transport/transport.py", "script/action_transport/categorical.py",
            "script/action_transport/continuous.py", "script/quantized_input_comparison/ot_arms.py",
            "script/fsq_reconstruction/evaluate.py")
METRICS = ("MoF", "Edit", "F1_10", "F1_25", "F1_50")
INPUTS = ("context", "pca", "centre")
DIRECT_INPUT_DATASETS = ("babel2", "babel3", "hugadb", "babel1", "lara")  # cheap first
DIRECT_NORMALIZATIONS = {"lara": ("global", "per_recording")}
TASK2_DIR = ROOT / "results" / "quantized_input_comparison" / "task2" / "task2_v1" / "cells"
TASK2_GATE = ("raw_kmeans__direct_continuous__seed111", "pca")  # (Task 2 cell, our input) on HuGaDB / global / 111
GATE_CELLS = (("hugadb", "global", 111), ("babel2", "global", 111))


def source_hash() -> str:
    h = hashlib.sha256()
    for name in OWN_FILES + OT_FILES:
        h.update(name.encode())
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()[:16]


def coeffs_for(beta: float) -> transport.Coeffs:
    base = ot.COEFFS
    return transport.Coeffs(a=base.a, beta=beta, lam=base.lam, eps=base.eps)


def fit_and_score(recs, mu0, num_states: int, coeffs: transport.Coeffs, setting: RAD.Setting, outer_cap: int,
                  rungs: tuple, gts: dict, starts: dict, frame_counts: dict, deadline: float | None = None) -> dict:
    """The decode of `contextualizer_calibration.decode_state` with the kernel scoped to `setting`
    and an explicit beta. Returns ladder rows, fit status and the half-widths the solver used."""
    started = time.perf_counter()
    with RAD.kernel_override(setting) as used:
        fit = continuous.fit_continuous(recs, mu0, num_states, coeffs, ot.FIT_INNER_STEPS, outer_cap,
                                        ot.FIT_OUTER_PATIENCE, ot.FIT_OUTER_RELTOL, ot.DEVICE, deadline)
        if fit.outer_status == "invalid":
            raise FinalSolveInvalid(f"OT fit invalid (iterations={fit.outer_iterations})")
        predictions: dict = {}
        ladder = contextual_final_ladder(fit.params, recs, coeffs, rungs, gts, starts, frame_counts, deadline,
                                         predictions_out=predictions, fail_on_invalid=True)
    predictions.pop("__window_states__", None)
    from script.week5.raw_kmeans_cross_dataset.decode import segment_stats
    return dict(ladder=[asdict(r) for r in ladder], ot_status=fit.outer_status, outer_iterations=fit.outer_iterations,
                segment_stats=segment_stats(predictions, {n: gts[n] for n in predictions}, num_states),
                half_widths_used={str(k): v for k, v in sorted(used.items())}, seconds=time.perf_counter() - started)


class Source:
    """Read-only view of the source run's verified selected-checkpoint decodes."""

    def __init__(self, run: str, datasets: list[str]):
        self.dir = ROOT / "results" / "week5" / "raw_kmeans_cross_dataset" / run
        self.records = {(r["dataset"], r["normalization"], r["seed"]): r
                        for r in P0.done_records(self.dir, datasets) if r["role"] == "selected"}
        self._loaders: dict = {}

    def cell(self, dataset: str, normalization: str, seed: int) -> dict | None:
        return self.records.get((dataset, normalization, seed))

    def loader(self, rec: dict) -> Loader:
        key = rec["cell_id"]
        if key not in self._loaders:
            self._loaders[key] = Loader(self.dir / "prep" / key)
        return self._loaders[key]


class Unit:
    def __init__(self, dataset: str, normalization: str, seed: int, setting: RAD.Setting, beta: float, stage: str,
                 input: str = "context"):
        self.dataset, self.normalization, self.seed, self.setting, self.beta, self.stage = (
            dataset, normalization, seed, setting, beta, stage)
        self.input = input

    @property
    def uid(self) -> str:
        base = f"{self.dataset}__{self.normalization}__seed{self.seed}__{self.setting.name}__beta{self.beta:g}"
        return base if self.input == "context" else f"{base}__in-{self.input}"


class Runner:
    def __init__(self, args):
        self.args = args
        self.out = ROOT / "results" / "week5" / "timescale_inspection" / args.run_id / "phase1"
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "units").mkdir(exist_ok=True)
        self.deadline = time.perf_counter() + args.hours * 3600 if args.hours else None
        self.impl = source_hash()
        datasets = sorted(set(SCREEN_DATASETS + CONFIRM_DATASETS))
        self.source = Source(args.source_run, datasets)
        self.gates = json.loads((self.out / "gates.json").read_text()) if (self.out / "gates.json").exists() else {}
        self.manifest_path = self.out / "manifest.json"
        self.check_resume()
        self.by_identity: dict[str, str] = {}
        for path in sorted((self.out / "units").glob("*/info.json")):
            try:
                info = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if info.get("status") == "done" and not info.get("aliased_of"):
                self.by_identity.setdefault(info["identity"], info["unit"])

    # -- identity ------------------------------------------------------------------------------
    def check_resume(self) -> None:
        ident = dict(schema=SCHEMA_VERSION, implementation_hash=self.impl, source_run=self.args.source_run,
                     outer_cap=OUTER_CAP, rungs=list(RUNGS), coeffs=asdict(ot.COEFFS))
        if self.manifest_path.exists():
            old = json.loads(self.manifest_path.read_text())
            if old.get("identity") != ident:
                raise SystemExit(f"{self.out} was written by a different implementation or settings; choose a new "
                                 f"--run-id (old {old.get('identity')}, now {ident})")
        write_json_atomic(self.manifest_path, dict(
            identity=ident, run_id=self.args.run_id, phase=1, status="running", python=platform.python_version(),
            started_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"), stage=self.args.stage))

    def unit_identity(self, unit: Unit, src: dict, signature: str) -> str:
        return hashlib.sha256(json.dumps([src["identity"], signature, unit.beta, self.impl, OUTER_CAP, list(RUNGS)])
                              .encode()).hexdigest()[:16]

    def unit_dir(self, unit: Unit) -> Path:
        return self.out / "units" / unit.uid

    def unit_done(self, unit: Unit, identity: str) -> dict | None:
        path = self.unit_dir(unit) / "info.json"
        if not path.exists():
            return None
        try:
            info = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return info if info.get("status") == "done" and info.get("identity") == identity else None

    # -- one decode ----------------------------------------------------------------------------
    def direct_loader(self, unit: Unit):
        cell_id = f"{unit.dataset}__{unit.normalization}__seed{unit.seed}"
        prep = self.source.dir / "prep" / cell_id
        try:
            return cell_id, artifacts.Loader(prep) if (prep / "manifest.json").exists() else None
        except (OSError, KeyError, ValueError):
            return cell_id, None

    def load_inputs(self, unit: Unit):
        """(src, spec, loader, names, emb, zero, codes, lens, starts). For `pca` / `centre` there is no checkpoint:
        `src` is a pseudo-record carrying the preparation's identity and the features are built from the loader."""
        if unit.input != "context":
            cell_id, loader = self.direct_loader(unit)
            if loader is None:
                return None
            spec = D.SPECS[unit.dataset]
            names = sorted(loader.recording_names())
            mean = loader.mean(FAMILY).astype(np.float64)
            lookup = loader.lookup(FAMILY)
            emb, zero, codes, lens, starts = {}, {}, {}, {}, {}
            for n in names:
                rec = loader.recording(FAMILY, n)
                raw = lookup[rec["codes"]] if unit.input == "centre" else rec["continuous"]
                emb[n], zero[n] = continuous.l2_normalize(raw.astype(np.float64) - mean[None, :])
                codes[n], lens[n], starts[n] = rec["codes"], rec["lengths"], rec["starts"]
            manifest = artifacts.read_manifest(loader.run_dir)
            ident = hashlib.sha256(json.dumps([manifest["prep_key"], manifest["sample_ids_hash"], unit.input,
                                               hashlib.sha256(mean.tobytes()).hexdigest()], sort_keys=True)
                                   .encode()).hexdigest()[:16]
            return dict(cell_id=cell_id, identity=ident, step=None), spec, loader, names, emb, zero, codes, lens, starts
        src = self.source.cell(unit.dataset, unit.normalization, unit.seed)
        if src is None:
            return None
        spec, loader = D.SPECS[unit.dataset], self.source.loader(src)
        info = json.loads((src["eval_dir"] / "info.json").read_text())
        names = sorted(info["scored_recording_names"])
        with np.load(src["eval_dir"] / "embeddings.npz") as z:
            emb = {n: z[f"{n}__emb"] for n in names}
            zero = {n: z[f"{n}__zero"] for n in names}
        codes, lens, starts = {}, {}, {}
        for n in names:
            rec = loader.recording(FAMILY, n)
            codes[n], lens[n], starts[n] = rec["codes"], rec["lengths"], rec["starts"]
        return src, spec, loader, names, emb, zero, codes, lens, starts

    def run_unit(self, unit: Unit) -> str:
        loaded = self.load_inputs(unit)
        if loaded is None:
            return "missing_source"
        src, spec, loader, names, emb, zero, codes, lens, starts = loaded
        signature = RAD.signature(unit.setting, spec.fps, spec.window, [len(codes[n]) for n in names])
        identity = self.unit_identity(unit, src, signature)
        if self.unit_done(unit, identity):
            return "cached"
        udir = self.unit_dir(unit)
        udir.mkdir(parents=True, exist_ok=True)
        meta = dict(unit=unit.uid, stage=unit.stage, dataset=unit.dataset, normalization=unit.normalization,
                    seed=unit.seed, input=unit.input, setting=unit.setting.name, family=unit.setting.family, value=unit.setting.value,
                    beta=unit.beta, signature=signature, identity=identity, source_cell=src["cell_id"],
                    source_identity=src["identity"], checkpoint_step=src["step"])
        current_sig = RAD.signature(RAD.CURRENT, spec.fps, spec.window, [len(codes[n]) for n in names])
        if (unit.input == "context" and unit.beta == 0.3 and signature == current_sig and unit.stage != "gate"
                and not self.args.no_alias):
            # Same half-width on every recording as the source run => the identical solve. Reuse the source run's own
            # decode (its equality with a fresh decode is what the gates check) instead of repeating it.
            ladder = json.loads((src["eval_dir"] / "ladder.json").read_text())["rows"]
            sinfo = json.loads((src["eval_dir"] / "info.json").read_text())
            self.write_unit(udir, dict(meta, status="done", aliased_of=f"{self.args.source_run}:{src['cell_id']}"),
                            dict(ot_status=sinfo["ot_status"], outer_iterations=sinfo["ot_outer_iterations"],
                                 segment_stats=sinfo["segment_stats"], half_widths_used={}, seconds=0.0), ladder)
            return "aliased"
        twin = self.by_identity.get(identity)
        if twin is not None and twin != unit.uid and not self.args.no_alias and unit.stage != "gate":
            # another setting resolved to the same half-width on every recording: identical solve, same result
            twin_info = json.loads((self.out / "units" / twin / "info.json").read_text())
            self.write_unit(udir, dict(meta, status="done", aliased_of=twin), twin_info, twin_info["ladder"])
            return "aliased"
        recs = [continuous.ContinuousRecordingInput(n, codes[n].astype(np.int32), lens[n].astype(np.int32), spec.fps,
                                                    spec.window, emb[n].astype(np.float64), zero[n]) for n in names]
        mu0, _ = continuous.build_mu0(loader.grouping(FAMILY), recs, spec.num_states)
        gts = D.load_gts(spec, names)
        frame_counts = {n: loader.frame_count(n) for n in names}
        try:
            res = fit_and_score(recs, mu0, spec.num_states, coeffs_for(unit.beta), unit.setting, OUTER_CAP, RUNGS, gts,
                                starts, frame_counts, self.deadline)
        except DeadlineExceeded:
            return "deadline"
        except FinalSolveInvalid as exc:
            write_json_atomic(udir / "info.json", dict(meta, status="invalid", reason=str(exc)))
            return "invalid"
        self.write_unit(udir, dict(meta, status="done"), res, res["ladder"])
        self.by_identity.setdefault(identity, unit.uid)
        return "done"

    def write_unit(self, udir: Path, meta: dict, res: dict, ladder: list[dict]) -> None:
        write_json_atomic(udir / "info.json", dict(meta, ot_status=res["ot_status"],
                                                   outer_iterations=res["outer_iterations"],
                                                   segment_stats=res["segment_stats"],
                                                   half_widths_used=res["half_widths_used"],
                                                   seconds=res["seconds"], ladder=ladder))

    def unit_for(self, unit: Unit, setting: RAD.Setting) -> Unit:
        return Unit(unit.dataset, unit.normalization, unit.seed, setting, unit.beta, unit.stage)

    # -- stages --------------------------------------------------------------------------------
    def run_units(self, units: list[Unit]) -> dict:
        counts: dict[str, int] = {}
        for i, unit in enumerate(units, 1):
            check = self.deadline is not None and time.perf_counter() > self.deadline
            if check:
                counts["deadline"] = counts.get("deadline", 0) + len(units) - i + 1
                break
            t0 = time.perf_counter()
            status = self.run_unit(unit)
            counts[status] = counts.get(status, 0) + 1
            print(f"[{unit.stage} {i}/{len(units)}] {unit.uid}: {status} ({time.perf_counter() - t0:.0f}s)", flush=True)
            if status == "deadline":
                break
        return counts

    def gate(self) -> bool:
        """Fresh decodes at the `current` radius must reproduce the source run's saved ladders exactly, on one
        HuGaDB and one BABEL cell. They license aliasing every setting that resolves to the current radius."""
        ok = True
        for d, n, s in GATE_CELLS:
            key = f"current_radius_reproduces_source:{d}__{n}__seed{s}"
            if self.gates.get(key, {}).get("status") == "passed":
                continue
            unit = Unit(d, n, s, RAD.CURRENT, 0.3, "gate")
            status = self.run_unit(unit)
            if status in ("deadline", "missing_source"):
                self.gates[key] = dict(status=status)
                write_json_atomic(self.out / "gates.json", self.gates)
                ok = False
                continue
            src = self.source.cell(d, n, s)
            mine = json.loads((self.unit_dir(unit) / "info.json").read_text())["ladder"]
            ref = json.loads((src["eval_dir"] / "ladder.json").read_text())["rows"]
            diffs = {f"{a['rung_steps']}:{m}": a[m] - b[m] for a, b in zip(mine, ref) for m in METRICS}
            passed = len(mine) == len(ref) and max(abs(v) for v in diffs.values()) == 0.0
            self.gates[key] = dict(status="passed" if passed else "failed", cell=src["cell_id"],
                                   max_abs_diff=max(abs(v) for v in diffs.values()),
                                   f1_50_now=mine[-1]["F1_50"], f1_50_source=ref[-1]["F1_50"])
            write_json_atomic(self.out / "gates.json", self.gates)
            print(f"[gate] {key}: {self.gates[key]}", flush=True)
            ok = ok and passed
        return ok

    def screen_units(self) -> list[Unit]:
        return [Unit(d, "global", seed, s, 0.3, "screen") for d in SCREEN_DATASETS for seed in SEEDS for s in RAD.SCREEN]

    def read_unit(self, unit: Unit) -> dict | None:
        loaded = self.load_inputs(unit)
        if loaded is None:
            return None
        src, spec, _, names, _, _, codes, *_ = loaded
        sig = RAD.signature(unit.setting, spec.fps, spec.window, [len(codes[n]) for n in names])
        return self.unit_done(unit, self.unit_identity(unit, src, sig))

    def last_f1(self, unit: Unit) -> float | None:
        info = self.read_unit(unit)
        return None if info is None else float(max(info["ladder"], key=lambda r: r["rung_steps"])["F1_50"])

    def select(self) -> dict:
        """The two settings carried forward, from BABEL-2 ONLY: the best fixed-seconds and the best relative setting
        by mean last-rung F1@50 over the three seeds (ties go to the smaller radius, i.e. the earlier setting)."""
        path = self.out / "selection.json"
        if path.exists():
            return json.loads(path.read_text())
        chosen = {}
        table = {}
        for family, candidates in (("sec", RAD.SEC), ("rel", RAD.REL)):
            means = {}
            for s in candidates:
                vals = [self.last_f1(Unit(DEV_DATASET, "global", seed, s, 0.3, "screen")) for seed in SEEDS]
                if any(v is None for v in vals):
                    raise SystemExit(f"selection needs the finished screen on {DEV_DATASET}: {s.name} is missing seeds")
                means[s.name] = statistics.mean(vals)
            table[family] = means
            chosen[family] = max(candidates, key=lambda s: (means[s.name], -candidates.index(s))).name
        out = dict(rule=f"best fixed-seconds and best relative setting by mean F1@50 on {DEV_DATASET}, 3 seeds",
                   chosen=chosen, development_means=table)
        write_json_atomic(path, out)
        return out

    def confirm_units(self) -> list[Unit]:
        sel = self.select()["chosen"]
        settings = [RAD.CURRENT, RAD.BY_NAME[sel["sec"]], RAD.BY_NAME[sel["rel"]]]
        units = []
        for d in CONFIRM_DATASETS:
            for seed in SEEDS:
                for s in settings:
                    units.append(Unit(d, "global", seed, s, 0.3, "confirm"))
        return units

    def beta_units(self) -> list[Unit]:
        sel = self.select()["chosen"]
        settings = [RAD.BY_NAME[sel["sec"]], RAD.BY_NAME[sel["rel"]]]
        return [Unit(DEV_DATASET, "global", seed, s, b, "beta") for seed in SEEDS for s in settings
                for b in BETAS if b != 0.3]

    def direct_units(self, settings: list[RAD.Setting], stage: str) -> list[Unit]:
        """The context baseline at `current` (an alias of the source decode, free), then `pca` and `centre`."""
        units = []
        datasets = [d for d in DIRECT_INPUT_DATASETS if not self.args.datasets or d in self.args.datasets]
        for d in datasets:
            for norm in DIRECT_NORMALIZATIONS.get(d, ("global",)):
                for seed in SEEDS:
                    units.append(Unit(d, norm, seed, RAD.CURRENT, 0.3, stage, "context"))
                    for inp in ("pca", "centre"):
                        for s_ in settings:
                            units.append(Unit(d, norm, seed, s_, 0.3, stage, inp))
        return units

    def inputs_units(self) -> list[Unit]:
        return self.direct_units([RAD.CURRENT], "inputs")

    def inputs_chosen_units(self) -> list[Unit]:
        sel = self.select()["chosen"]
        return self.direct_units([RAD.BY_NAME[sel["sec"]], RAD.BY_NAME[sel["rel"]]], "inputs_chosen")

    def input_gate(self) -> bool:
        """The `pca` arm on HuGaDB / global / 111 at the current radius must reproduce Task 2's
        `direct_continuous` decode of the same family (qic_v3 preparation, bit-equal features)."""
        key = "direct_pca_reproduces_task2"
        if self.gates.get(key, {}).get("status") == "passed":
            return True
        task2_cell, inp = TASK2_GATE
        ref_path = TASK2_DIR / task2_cell / "ladder.json"
        if not ref_path.exists():
            self.gates[key] = dict(status="skipped_reference_absent")
            write_json_atomic(self.out / "gates.json", self.gates)
            return True
        unit = Unit("hugadb", "global", 111, RAD.CURRENT, 0.3, "gate", inp)
        status = self.run_unit(unit)
        if status in ("deadline", "missing_source"):
            self.gates[key] = dict(status=status)
            write_json_atomic(self.out / "gates.json", self.gates)
            return False
        mine = json.loads((self.unit_dir(unit) / "info.json").read_text())["ladder"]
        ref = json.loads(ref_path.read_text())
        ref = ref["rows"] if isinstance(ref, dict) else ref
        diffs = {f"{a['rung_steps']}:{m}": a[m] - b[m] for a, b in zip(mine, ref) for m in METRICS}
        worst = max(abs(v) for v in diffs.values())
        passed = len(mine) == len(ref) and worst <= 0.005  # the scorer rounds to two decimals
        self.gates[key] = dict(status="passed" if passed else "failed", cell=task2_cell, max_abs_diff=worst,
                               f1_50_now=mine[-1]["F1_50"], f1_50_source=ref[-1]["F1_50"])
        write_json_atomic(self.out / "gates.json", self.gates)
        print(f"[gate] {key}: {self.gates[key]}", flush=True)
        return passed

    def execute(self) -> None:
        stages = self.args.stage
        if self.args.dry_run:
            self.dry_run(stages)
            return
        if "gate" in stages or any(s in stages for s in ("screen", "confirm", "beta")):
            if not self.gate():
                print("GATE FAILED or unavailable: the radius path does not reproduce the source run; stopping.")
                self.finish("gate_failed")
                raise SystemExit(2)
        if any(s in stages for s in ("inputs", "inputs_chosen")) and not self.input_gate():
            print("GATE FAILED or unavailable: the direct-input path does not reproduce Task 2; stopping.")
            self.finish("gate_failed")
            raise SystemExit(2)
        counts = {}
        if "screen" in stages:
            counts["screen"] = self.run_units(self.screen_units())
        if "confirm" in stages:
            counts["confirm"] = self.run_units(self.confirm_units())
        if "beta" in stages:
            counts["beta"] = self.run_units(self.beta_units())
        if "inputs" in stages:
            counts["inputs"] = self.run_units(self.inputs_units())
        if "inputs_chosen" in stages:
            counts["inputs_chosen"] = self.run_units(self.inputs_chosen_units())
        self.finish("finished", counts)
        from script.week5.timescale_inspection import report1
        print(f"report: {report1.write_report(self.out)}")

    def finish(self, status: str, counts: dict | None = None) -> None:
        m = json.loads(self.manifest_path.read_text())
        m.update(status=status, counts=counts, finished_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        write_json_atomic(self.manifest_path, m)

    def dry_run(self, stages: list[str]) -> None:
        """Counts, aliasing and radius tables; no decoding."""
        plan = []
        if "screen" in stages:
            plan += self.screen_units()
        if "confirm" in stages:
            try:
                plan += self.confirm_units()
            except SystemExit as exc:
                print(f"confirm stage cannot be planned yet: {exc}")
        if "beta" in stages:
            try:
                plan += self.beta_units()
            except SystemExit as exc:
                print(f"beta stage cannot be planned yet: {exc}")
        if "inputs" in stages:
            plan += self.inputs_units()
        if "inputs_chosen" in stages:
            try:
                plan += self.inputs_chosen_units()
            except SystemExit as exc:
                print(f"inputs_chosen stage cannot be planned yet: {exc}")
        todo, alias, missing, seen = 0, 0, 0, set()
        for u in plan:
            loaded = self.load_inputs(u)
            if loaded is None:
                missing += 1
                continue
            src, spec, _, names, _, _, codes, *_ = loaded
            cur = RAD.signature(RAD.CURRENT, spec.fps, spec.window, [len(codes[n]) for n in names])
            sig = RAD.signature(u.setting, spec.fps, spec.window, [len(codes[n]) for n in names])
            ident = self.unit_identity(u, src, sig)
            if self.unit_done(u, ident) is not None:
                continue
            if (u.input == "context" and u.beta == 0.3 and sig == cur) or ident in seen or ident in self.by_identity:
                alias += 1
            else:
                todo += 1
                seen.add(ident)
        print(f"planned units {len(plan)}: to decode {todo}, aliased to an identical solve {alias}, "
              f"missing source cell {missing}")
        for ds in sorted(set(SCREEN_DATASETS + CONFIRM_DATASETS)):
            spec = D.SPECS[ds]
            lens = [int(np.ceil(c / spec.window)) for c in D.frame_counts(spec, D.recording_names(spec)).values()]
            print(f"\nhalf-width in tokens for {ds} (median over {len(lens)} recordings; share of recordings at 0)")
            for s in RAD.SCREEN:
                t = RAD.radius_table(s, spec.fps, spec.window, lens)
                print(f"  {s.name:<10} median {t['median']:>4.0f} ({t['median_seconds']:.2f} s)  "
                      f"min {t['minimum']}  max {t['maximum']}  zero {t['share_zero']:.2f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Time-scale inspection, phase 1")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--stage", nargs="+", default=["screen"], choices=["gate", "screen", "confirm", "beta", "inputs", "inputs_chosen"])
    ap.add_argument("--datasets", nargs="+", default=None, help="restrict the inputs stages to these datasets")
    ap.add_argument("--source-run", default=SOURCE_RUN)
    ap.add_argument("--hours", type=float, default=None, help="elapsed-time budget; completed units are kept")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-alias", action="store_true", help="decode settings equal to `current` instead of aliasing")
    args = ap.parse_args(argv)
    if not args.dry_run and not args.hours:
        raise SystemExit("a decoding run requires an explicit elapsed-time budget: pass --hours")
    Runner(args).execute()


if __name__ == "__main__":
    main()
