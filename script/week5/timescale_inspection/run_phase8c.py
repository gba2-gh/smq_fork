"""Phase 8c of the time-scale inspection: a bounded representation diagnostic of which invariance matters.

  G   G8c-1 (T6 with the rotation disabled reproduces ts5_v1's T0), G8c-2 (T6s reduces to T2's window statistics), G8c-3
      (heading invariance, within-window turning kept, chirality kept), G8c-4 (the distance block through the unweighted
      path reproduces T2); G8b geometry and G3 inherited from phases 8a and 5
  A   BABEL-2 and BABEL-3: T6, T6s, T5gh-std x seeds 111, 222, 1538574472 (`context`, span 2), plus weighted T5gh at
      seed 1538574472 (phase 8a has 111 and 222), ordered by seed
  B   LARa: T6, T6s x the three seeds

No screening and no third-seed rule: every cell runs all three fixed seeds. T0, T2 and weighted T5gh (seeds 111/222) are
read from earlier phases, never rebuilt. Design and reading rules: docs/plans/week5/TIMESCALE_INSPECTION/PHASE8C_SYMMETRY_TEST.md.
No shared module and no earlier phase file is edited.

Usage:
  python -m script.week5.timescale_inspection.run_phase8c --run-id ts5_v1 --hours 5
  python -m script.week5.timescale_inspection.run_phase8c --run-id ts8_dry --dry-run
  python -m script.week5.timescale_inspection.run_phase8c --run-id ts8c_smoke --source-run ts5_v1 --smoke
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
from script.week5.timescale_inspection import report8c as R8C  # noqa: E402
from script.week5.timescale_inspection import run_phase5 as R5  # noqa: E402
from script.week5.timescale_inspection import run_phase8a as R8A  # noqa: E402
from script.week5.timescale_inspection import tokenizers as TOK  # noqa: E402
from script.week5.timescale_inspection import tokenizers8 as T8  # noqa: E402
from script.week5.timescale_inspection import tokenizers8c as T8C  # noqa: E402

SCHEMA_VERSION = 1
SEEDS = (111, 222, 1538574472)
OWN_FILES = R8A.OWN_FILES + tuple(f"script/week5/timescale_inspection/{m}.py" for m in ("tokenizers8c", "run_phase8c"))
BABEL_TOKENS = ("t6_heading_canonical", "t6s_heading_canonical_stats", "t5_both_std")
LARA_TOKENS = ("t6_heading_canonical", "t6s_heading_canonical_stats")
COST = {k: dict(v) for k, v in R8A.COST.items()}


def source_hash() -> str:
    h = hashlib.sha256()
    for name in OWN_FILES + R5.TRAIN_FILES:
        h.update(name.encode())
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()[:16]


def stage_items(stage: str, seeds=SEEDS) -> list[R5.Item]:
    mk = R5.mk_cell
    if stage == "A":  # ordered by seed: a budget stop leaves complete seeds, not complete tokenizers
        out = []
        for s in seeds:
            for ds in ("babel2", "babel3"):
                out += [R5.Item(mk(ds, tok, s), ("context",)) for tok in BABEL_TOKENS]
                if s == seeds[-1]:  # weighted T5gh: phase 8a ran seeds 111 and 222, this adds the third
                    out.append(R5.Item(mk(ds, "t5_both", s), ("context",)))
        return out
    if stage == "B":
        return [R5.Item(mk("lara", tok, s), ("context",)) for s in seeds for tok in LARA_TOKENS]
    return []


class Phase8cRunner(R8A.Phase8aRunner):
    """Does not call `Phase8aRunner.__init__` (it hard-codes the phase8a directory, the screening seeds and the third-seed
    cell set). Reuses unchanged: `ensure_train`, `ensure_decode` (frame predictions saved), `unit_done`, `run_item`,
    `job_dir` / `job_state` / `selected_state`, `unit_dir` / `unit_identity`, `build_recs`, `_gate`, `expired`, `device`."""

    def __init__(self, args):
        self.args = args
        self.smoke = bool(args.smoke)
        self.scale = replace(P.SMOKE, recording_cap=None) if self.smoke else P.FULL
        base = ROOT / "results" / "week5" / "timescale_inspection"
        self.out = base / args.run_id / "phase8c"
        for sub in ("prep", "jobs", "units", "gates"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.source_root = base / args.source_run
        self.phase5_dir = self.source_root / "phase5"
        self.deadline = time.perf_counter() + args.hours * 3600 if args.hours else None
        self.impl = source_hash()
        self.radius = RAD.CURRENT
        self.seeds = (111, 222) if self.smoke else SEEDS
        self.job_infos, self.loaders, self.not_run = {}, {}, {}
        self.gates_path = self.out / "gates.json"
        self.gates = json.loads(self.gates_path.read_text()) if self.gates_path.exists() else {}
        self.manifest_path = self.out / "manifest.json"
        self.base_cells = [it.cell for st in ("A", "B") for it in self.stage_items(st)]
        self.check_resume()

    def stage_items(self, stage, decisions=None):
        return stage_items(stage, self.seeds)

    def check_resume(self) -> None:
        ident = dict(schema=SCHEMA_VERSION, implementation_hash=self.impl, scale=self.scale.name,
                     outer_cap=self.scale.outer_cap, rungs=list(self.scale.rungs), radius=self.radius.name,
                     cells={c.cell_id: self.identity_of(c) for c in self.base_cells}, source_run=self.args.source_run,
                     seeds=list(self.seeds))
        if self.manifest_path.exists():
            old = json.loads(self.manifest_path.read_text())
            if old.get("identity") != ident:
                raise SystemExit(f"{self.out} was written by a different implementation or cell set; choose a new --run-id.")
        write_json_atomic(self.manifest_path, dict(
            identity=ident, run_id=self.args.run_id, phase="8c", status="running", smoke=self.smoke,
            analysis_hash_at_launch=R8C.analysis_hash(), python=platform.python_version(),
            started_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"), radius=self.radius.name))
        self.ident = ident

    # -- preparation: phase-8a keys through tokenizers8, phase-8c keys through tokenizers8c ---------------------------
    def prepare(self, cell):
        if cell.prep_id in self.loaders:
            return self.loaders[cell.prep_id]
        mod = T8 if cell.tok in T8.T5_KEYS else T8C
        d = self.prep_dir(cell)
        cap = self.recording_cap(cell)
        try:
            ok = mod.is_prepared(cell.dataset, cell.tok, cell.seed, d, cell.window, cap)
        except (OSError, KeyError, ValueError):
            ok = False
        if not ok:
            if self.expired():
                return None
            t0 = time.perf_counter()
            mod.prepare_cell(cell.dataset, cell.tok, cell.seed, d, cell.window, cap)
            print(f"[prep] {cell.prep_id} ({time.perf_counter() - t0:.0f}s)", flush=True)
        self.loaders[cell.prep_id] = Loader(d)
        return self.loaders[cell.prep_id]

    # -- gates --------------------------------------------------------------------------------------------------------
    def _compare(self, ref_dir: Path, scratch: Path, fam_ref: str, fam_new: str) -> dict:
        a, b = Loader(ref_dir), Loader(scratch)
        names = sorted(a.recording_names())
        checks = dict(same_recordings=names == sorted(b.recording_names()))
        if checks["same_recordings"]:
            for key in ("codes", "continuous", "starts", "lengths"):
                checks[key] = all(np.array_equal(a.recording(fam_ref, n)[key], b.recording(fam_new, n)[key]) for n in names)
        checks["mean"] = bool(np.array_equal(a.mean(fam_ref), b.mean(fam_new)))
        checks["rms"] = float(a.rms(fam_ref)) == float(b.rms(fam_new))
        checks["grouping"] = bool(np.array_equal(a.grouping(fam_ref), b.grouping(fam_new)))
        checks["split"] = json.loads((ref_dir / "split.json").read_text()) == json.loads((scratch / "split.json").read_text())
        diff = float(np.abs(a.lookup(fam_ref).astype(np.float64) - b.lookup(fam_new).astype(np.float64)).max())
        checks["lookup"] = diff <= G.LOOKUP_ATOL
        return dict(status="passed" if all(checks.values()) else "failed", checks=checks, lookup_max_abs_diff=diff)

    def gate_g8c1(self) -> dict:
        """T6 with the rotation disabled reproduces ts5_v1's T0 (BABEL-2, seed 111)."""
        ref = self.phase5_dir / "prep" / "babel2__t0_raw__seed111"
        if not ref.exists():
            ref = self.phase5_dir / "prep" / "t0_raw" / "babel2__t0_raw__seed111"
        if not ref.exists():
            return dict(status="failed", reason=f"reference not found: {ref}")
        scratch = self.out / "gates" / "g8c1_scratch"
        T8C.prepare_cell("babel2", "t6_norot", 111, scratch)
        return self._compare(ref, scratch, TOK.SPECS["t0_raw"].family, TOK.SPECS["t6_norot"].family)

    def gate_g8c4(self) -> dict:
        """The distance block through the unweighted path reproduces ts5_v1's T2 (BABEL-2, seed 111)."""
        ref = self.phase5_dir / "prep" / "t2_relational" / "babel2__t2_relational__seed111"
        if not ref.exists():
            return dict(status="failed", reason=f"reference not found: {ref}")
        scratch = self.out / "gates" / "g8c4_scratch"
        T8C.prepare_cell("babel2", "t5_dist_only_std", 111, scratch)
        return self._compare(ref, scratch, TOK.SPECS["t2_relational"].family, TOK.SPECS["t5_dist_only_std"].family)

    def inherit_gate(self, name: str, phase: str) -> None:
        path = self.source_root / phase / "gates.json"
        g = json.loads(path.read_text()) if path.exists() else {}
        self.gates[name] = dict(g[name], inherited_from=str(path)) if name in g else dict(
            status="failed", reason=f"'{name}' not recorded in {path}")

    def run_gates(self) -> None:
        self.inherit_phase5_gate("g3")
        for ds in ("babel2", "babel3", "lara"):
            self.inherit_gate(f"g8b_{ds}", "phase8a")
        cap = lambda ds: TOK.D.SPECS[ds].smoke_cap if self.smoke else None
        if self.smoke:
            for g in ("g8c1", "g8c4"):
                self.gates[g] = dict(status="not_applicable_smoke", implementation_hash=self.impl)
        else:
            self._gate("g8c1", self.gate_g8c1)
            self._gate("g8c4", self.gate_g8c4)
        self._gate("g8c2", lambda: _both(T8C.reduction_check("babel2"), T8C.reduction_check("lara")))
        self._gate("g8c3", lambda: _both(T8C.invariance_check("babel2"), T8C.invariance_check("lara")))
        write_json_atomic(self.gates_path, self.gates)

    def blocked(self, cell) -> str | None:
        st = lambda g: self.gates.get(g, {}).get("status")
        if cell.tok in ("t6_heading_canonical", "t6s_heading_canonical_stats"):
            if st("g8c1") == "failed":
                return "G8c-1 failed (T6 without rotation does not reproduce T0)"
            if st("g8c3") != "passed":
                return "G8c-3 not passed (T6/T6s are not yaw-invariant, or turning/chirality are not kept)"
            if cell.tok == "t6s_heading_canonical_stats" and st("g8c2") != "passed":
                return "G8c-2 not passed (T6s does not reduce to T2's window statistics)"
        if cell.tok == "t5_both_std" and st("g8c4") == "failed":
            return "G8c-4 failed (the unweighted distance path does not reproduce T2)"
        if st(f"g8b_{cell.dataset}") != "passed":
            return f"G8b not passed for {cell.dataset} (skeleton geometry is not confirmed)"
        if cell.dataset == "lara" and st("g3") != "passed":
            return "G3 not passed (LARa channels 3-5 are not confirmed as positions)"
        return None

    # -- orchestration -------------------------------------------------------------------------------------------------
    def execute(self) -> None:
        counts: dict[str, int] = {}
        self.run_gates()
        for stage in ("A", "B"):
            items = self.stage_items(stage)
            for i, item in enumerate(items, 1):
                self.run_item(item, counts, f"{stage} {i}/{len(items)}")
        m = json.loads(self.manifest_path.read_text())
        m.update(status="finished" if not self.expired() else "budget_stop", counts=counts, not_run=self.not_run,
                 finished_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        write_json_atomic(self.manifest_path, m)
        print(f"report: {R8C.write_report(self.out, self.source_root)}")

    def dry_run(self) -> None:
        total, seen = 5.0, set()
        print("stage G: gates G8c-1 to G8c-4 (~5 min)")
        for stage in ("A", "B"):
            minutes, items = 0.0, self.stage_items(stage)
            for it in items:
                c = it.cell
                if c.prep_id not in seen:
                    seen.add(c.prep_id)
                    minutes += COST["prep"][c.dataset]
                minutes += COST["train"][c.dataset] + COST["context"][c.dataset]
                print(f"  {c.cell_id}: arms {','.join(it.arms)} [span {c.span}]")
            total += minutes
            print(f"stage {stage}: {len(items)} cells, ~{minutes:.0f} min, cumulative ~{total / 60:.1f} h")
        print(f"projected total ~{total / 60:.1f} h (single-pass, otherwise idle machine)")


def _both(a: dict, b: dict) -> dict:
    return dict(status="passed" if a["status"] == b["status"] == "passed" else "failed", checks=dict(babel2=a["checks"], lara=b["checks"]),
                detail=dict(babel2=a, lara=b))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Time-scale inspection, phase 8c (which invariance? T6, T6s, unweighted T5gh)")
    ap.add_argument("--run-id", default="ts5_v1")
    ap.add_argument("--source-run", default="ts5_v1", help="run whose phase5/6/7/8a hold the T0, T2 and T5gh references")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not (args.dry_run or args.smoke) and not args.hours:
        raise SystemExit("a decoding run requires an explicit elapsed-time budget: pass --hours")
    runner = Phase8cRunner(args)
    if args.dry_run:
        runner.dry_run()
    else:
        runner.execute()


if __name__ == "__main__":
    main()
