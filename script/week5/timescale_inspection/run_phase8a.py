"""Phase 8a of the time-scale inspection: gravity- and heading-aware relational tokens on BABEL-2/3, then LARa.

  G   G8a (T5 path with only the distance block reproduces ts5_v1's T2 preparation), G8b (skeleton geometry per
      dataset), G8c (yaw invariance and the heading-rate sign), G3 inherited from phase 5 (LARa position channels)
  A   BABEL-2 and BABEL-3: T5gh, T5h, T5g x seeds 111, 222 (`context`, span 2)
  B   LARa: T5gh x seeds 111, 222
  C   third seed (1538574472) only for each (dataset, tokenizer) whose two-seed mean F1@50 is >= 1.0 point above
      T2's on the same seeds; the decisions are written to third_seed.json before any third-seed job starts

T2 span-2 references are read from ts5_v1 (never rebuilt). Design and declared rules:
docs/plans/week5/TIMESCALE_INSPECTION/PHASE8A_GRAVITY_HEADING_TOKENS.md. No shared module and no earlier phase file is
edited; the T5 tokenizer keys are registered at runtime by tokenizers8.

Usage:
  python -m script.week5.timescale_inspection.run_phase8a --run-id ts5_v1 --hours 4
  python -m script.week5.timescale_inspection.run_phase8a --run-id ts5_v1 --dry-run
  python -m script.week5.timescale_inspection.run_phase8a --run-id ts8_smoke --source-run ts5_v1 --smoke
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
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from script.fsq_reconstruction.experiment import write_json_atomic  # noqa: E402
from script.quantized_input_comparison.artifacts import Loader  # noqa: E402
from script.week5.contextualizer_calibration import plan as P  # noqa: E402
from script.week5.timescale_inspection import gates5 as G  # noqa: E402
from script.week5.timescale_inspection import radius as RAD  # noqa: E402
from script.week5.timescale_inspection import report8a as R8  # noqa: E402
from script.week5.timescale_inspection import run_phase5 as R5  # noqa: E402
from script.week5.timescale_inspection import run_phase7 as R7  # noqa: E402
from script.week5.timescale_inspection import tokenizers as TOK  # noqa: E402
from script.week5.timescale_inspection import tokenizers8 as T8  # noqa: E402

SCHEMA_VERSION = 1
SCREEN_SEEDS = R8.SCREEN_SEEDS
THIRD_SEED = R8.THIRD_SEED
OWN_FILES = R7.OWN_FILES + tuple(f"script/week5/timescale_inspection/{m}.py" for m in ("tokenizers8", "run_phase8a"))
STAGE_ORDER = ("G", "A", "B", "C")
COST = {k: dict(v) for k, v in R7.COST.items()}


def source_hash() -> str:
    h = hashlib.sha256()
    for name in OWN_FILES + R5.TRAIN_FILES:
        h.update(name.encode())
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()[:16]


def stage_items(stage: str, screen=SCREEN_SEEDS, third=(THIRD_SEED,), decisions: dict | None = None) -> list[R5.Item]:
    mk = R5.mk_cell
    if stage == "A":  # tokenizer-major: T5gh first on both datasets, then the factorial arms
        return [R5.Item(mk(ds, tok, s), ("context",)) for tok in R8.TOKENS for s in screen for ds in ("babel2", "babel3")]
    if stage == "B":
        return [R5.Item(mk("lara", "t5_both", s), ("context",)) for s in screen]
    if stage == "C":
        if decisions is None:  # every cell that could be promoted (identity and dry-run upper bound)
            keys = [f"{ds}|{tok}" for ds in ("babel2", "babel3") for tok in R8.TOKENS] + ["lara|t5_both"]
        else:
            keys = [k for k, d in decisions.items() if d["run_third_seed"]]
        return [R5.Item(mk(k.split("|")[0], k.split("|")[1], s), ("context",)) for k in keys for s in third]
    return []


class Phase8aRunner(R7.Phase7Runner):
    """Does not call `Phase7Runner.__init__` (it hard-codes the phase7 directory and the BABEL-1 cell set). Reuses
    unchanged: `ensure_train`, `ensure_decode` (with frame predictions saved), `unit_done`, `run_item`, `job_dir` /
    `job_state` / `selected_state`, `unit_dir` / `unit_identity`, `build_recs`, `_gate`, `expired`, `device`."""

    def __init__(self, args):
        self.args = args
        self.smoke = bool(args.smoke)
        self.scale = replace(P.SMOKE, recording_cap=None) if self.smoke else P.FULL
        base = ROOT / "results" / "week5" / "timescale_inspection"
        self.out = base / args.run_id / "phase8a"
        for sub in ("prep", "jobs", "units", "gates"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.source_root = base / args.source_run
        self.phase5_dir = self.source_root / "phase5"
        self.deadline = time.perf_counter() + args.hours * 3600 if args.hours else None
        self.impl = source_hash()
        self.radius = RAD.CURRENT
        self.screen = (111,) if self.smoke else SCREEN_SEEDS
        self.third = (222,) if self.smoke else (THIRD_SEED,)
        self.job_infos, self.loaders, self.not_run = {}, {}, {}
        self.gates_path = self.out / "gates.json"
        self.gates = json.loads(self.gates_path.read_text()) if self.gates_path.exists() else {}
        self.manifest_path = self.out / "manifest.json"
        self.base_cells = [it.cell for st in ("A", "B", "C") for it in self.stage_items(st)]
        self.check_resume()

    def stage_items(self, stage, decisions=None):
        return stage_items(stage, self.screen, self.third, decisions)

    def check_resume(self) -> None:
        ident = dict(schema=SCHEMA_VERSION, implementation_hash=self.impl, scale=self.scale.name,
                     outer_cap=self.scale.outer_cap, rungs=list(self.scale.rungs), radius=self.radius.name,
                     cells={c.cell_id: self.identity_of(c) for c in self.base_cells}, source_run=self.args.source_run,
                     screen_seeds=list(self.screen), third_seeds=list(self.third), promise=R8.PROMISE)
        if self.manifest_path.exists():
            old = json.loads(self.manifest_path.read_text())
            if old.get("identity") != ident:
                raise SystemExit(f"{self.out} was written by a different implementation or cell set; choose a new --run-id.")
        write_json_atomic(self.manifest_path, dict(
            identity=ident, run_id=self.args.run_id, phase="8a", status="running", smoke=self.smoke,
            analysis_hash_at_launch=R8.analysis_hash(), python=platform.python_version(),
            started_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"), radius=self.radius.name))
        self.ident = ident

    # -- preparation: T5 keys are built by tokenizers8 into phase8a/prep ------------------------------------------------
    def prepare(self, cell):
        if cell.prep_id in self.loaders:
            return self.loaders[cell.prep_id]
        d = self.prep_dir(cell)
        cap = self.recording_cap(cell)
        try:
            ok = T8.is_prepared(cell.dataset, cell.tok, cell.seed, d, cell.window, cap)
        except (OSError, KeyError, ValueError):
            ok = False
        if not ok:
            if self.expired():
                return None
            t0 = time.perf_counter()
            T8.prepare_cell(cell.dataset, cell.tok, cell.seed, d, cell.window, cap)
            print(f"[prep] {cell.prep_id} ({time.perf_counter() - t0:.0f}s)", flush=True)
        self.loaders[cell.prep_id] = Loader(d)
        return self.loaders[cell.prep_id]

    # -- gates ----------------------------------------------------------------------------------------------------------
    def gate_g8a(self) -> dict:
        ref_dir = self.phase5_dir / "prep" / "t2_relational" / "babel2__t2_relational__seed111"
        if not ref_dir.exists():
            return dict(status="failed", reason=f"reference not found: {ref_dir}")
        scratch = self.out / "gates" / "g8a_scratch"
        T8.prepare_cell("babel2", "t5_dist_only", 111, scratch)
        a, b = Loader(ref_dir), Loader(scratch)
        fa, fb = TOK.SPECS["t2_relational"].family, TOK.SPECS["t5_dist_only"].family
        names = sorted(a.recording_names())
        checks = dict(same_recordings=names == sorted(b.recording_names()))
        if checks["same_recordings"]:
            for key in ("codes", "continuous", "starts", "lengths"):
                checks[key] = all(np.array_equal(a.recording(fa, n)[key], b.recording(fb, n)[key]) for n in names)
        checks["mean"] = bool(np.array_equal(a.mean(fa), b.mean(fb)))
        checks["rms"] = float(a.rms(fa)) == float(b.rms(fb))
        checks["grouping"] = bool(np.array_equal(a.grouping(fa), b.grouping(fb)))
        checks["split"] = json.loads((ref_dir / "split.json").read_text()) == json.loads((scratch / "split.json").read_text())
        diff = float(np.abs(a.lookup(fa).astype(np.float64) - b.lookup(fb).astype(np.float64)).max())
        checks["lookup"] = diff <= G.LOOKUP_ATOL
        return dict(status="passed" if all(checks.values()) else "failed", checks=checks, lookup_max_abs_diff=diff)

    def run_gates(self) -> None:
        self.inherit_phase5_gate("g3")  # LARa position channels 3-5, verified in phase 5
        cap = lambda ds: TOK.D.SPECS[ds].smoke_cap if self.smoke else None
        if self.smoke:
            self.gates["g8a"] = dict(status="not_applicable_smoke", implementation_hash=self.impl)
        else:
            self._gate("g8a", self.gate_g8a)
        for ds in ("babel2", "babel3", "lara"):
            self._gate(f"g8b_{ds}", lambda ds=ds: T8.geometry_check(ds, cap(ds)))
        self._gate("g8c", lambda: T8.invariance_check("babel2"))
        write_json_atomic(self.gates_path, self.gates)

    def blocked(self, cell) -> str | None:
        if self.gates.get("g8a", {}).get("status") == "failed":
            return "G8a failed (the T5 path with only the distance block does not reproduce T2)"
        if self.gates.get("g8c", {}).get("status") == "failed":
            return "G8c failed (T5 features are not yaw-invariant or the heading rate is wrong)"
        if self.gates.get(f"g8b_{cell.dataset}", {}).get("status") == "failed":
            return f"G8b failed for {cell.dataset} (skeleton geometry does not match the declared joint indices)"
        if cell.dataset == "lara" and self.gates.get("g3", {}).get("status") != "passed":
            return "G3 not passed (LARa channels 3-5 are not confirmed as positions)"
        return None

    # -- orchestration --------------------------------------------------------------------------------------------------
    def decide_third_seed(self) -> dict:
        dec = R8.third_seed_decisions(self.out, self.source_root, self.screen)
        write_json_atomic(self.out / "third_seed.json", dict(
            decided_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"), screening_seeds=list(self.screen), third_seeds=list(self.third),
            threshold_f1_50=R8.PROMISE, decisions=dec))
        for k, d in sorted(dec.items()):
            print(f"[C] {k}: {d['reason']}" + (f" (gain {d['gain_f1_50']:+.2f})" if d["gain_f1_50"] is not None else ""), flush=True)
        return dec

    def execute(self) -> None:
        counts: dict[str, int] = {}
        self.run_gates()
        for stage in ("A", "B"):
            items = self.stage_items(stage)
            for i, item in enumerate(items, 1):
                self.run_item(item, counts, f"{stage} {i}/{len(items)}")
        dec = self.decide_third_seed()
        items = self.stage_items("C", dec)
        for i, item in enumerate(items, 1):
            self.run_item(item, counts, f"C {i}/{len(items)}")
        m = json.loads(self.manifest_path.read_text())
        m.update(status="finished" if not self.expired() else "budget_stop", counts=counts, not_run=self.not_run,
                 finished_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        write_json_atomic(self.manifest_path, m)
        print(f"report: {R8.write_report(self.out, self.source_root)}")

    def dry_run(self) -> None:
        total, seen = 5.0, set()
        print("stage G: gates G8a-c (~5 min)")
        for stage in ("A", "B", "C"):
            minutes = 0.0
            items = self.stage_items(stage)
            for it in items:
                c = it.cell
                if c.prep_id not in seen:
                    seen.add(c.prep_id)
                    minutes += COST["prep"][c.dataset]
                minutes += COST["train"][c.dataset] + COST["context"][c.dataset]
                print(f"  {c.cell_id}: arms {','.join(it.arms)} [span {c.span}]")
            total += minutes
            print(f"stage {stage}: {len(items)} cells, ~{minutes:.0f} min, cumulative ~{total / 60:.1f} h"
                  + ("  (upper bound: every cell promoted)" if stage == "C" else ""))
        print(f"projected total ~{total / 60:.1f} h (single-pass, otherwise idle machine)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Time-scale inspection, phase 8a (gravity- and heading-aware relational tokens)")
    ap.add_argument("--run-id", default="ts5_v1")
    ap.add_argument("--source-run", default="ts5_v1", help="run whose phase5/phase6/phase7 hold the T2 references")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not (args.dry_run or args.smoke) and not args.hours:
        raise SystemExit("a decoding run requires an explicit elapsed-time budget: pass --hours")
    runner = Phase8aRunner(args)
    if args.dry_run:
        runner.dry_run()
    else:
        runner.execute()


if __name__ == "__main__":
    main()
