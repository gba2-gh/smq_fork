"""Phase 6 of the time-scale inspection: confirming the span rule found in Phase 5.

Phase 5 (`ts5_v1`) found that on T2 (inter-joint distances), BABEL-2's F1@50 rises steadily as the mask gets shorter
in seconds (11.2 at 4.0s -> 21.4 at 2.1s -> ~31 at <=1.1s), and that OT's predicted segment count moves towards the
ground truth as the span shortens. This phase tests that rule where it can fail, on new seeds and two datasets it was
never run on:

  S1a  BABEL-2/3, T0 span 2 at the 3 original seeds (`context`)       -- does the span effect need T2's tokens?
  S1b  BABEL-2/3, T0/T2 x span 2/8 at 2 NEW seeds (context + centre)  -- extends the headline to 5 seeds, completes
                                                                          the 2x2 (tokenizer x span) design
  S2   HuGaDB, T0, spans {2, 4} at the 3 original seeds               -- negative control (span 8 ratio is 0.58: no gain predicted)
  S3   LARa, T2, spans {2, 4} at the 3 original seeds                 -- out-of-sample test (span 8 ratio is 0.87: a
                                                                          modest gain at most is predicted)

Design, reading rules and the span-ratio table: docs/plans/week5/TIMESCALE_INSPECTION/PHASE6_SPAN_CONFIRMATION.md.

This module does not edit `run_phase5.py`, `tokenizers.py`, `gates5.py` or `radius.py` (their hashes give `ts5_v1`
its identity); it imports and reuses them. By default it writes into `ts5_v1/phase6`, reusing `ts5_v1/phase5`'s
token preparations read-only wherever they verify, so a span comparison at an original seed runs on identical
tokens on both sides.

Usage:
  python -m script.week5.timescale_inspection.run_phase6 --run-id ts5_v1 --hours 6
  python -m script.week5.timescale_inspection.run_phase6 --run-id ts5_v1 --dry-run
  python -m script.week5.timescale_inspection.run_phase6 --run-id ts6_smoke --smoke
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
from script.week5.raw_kmeans_cross_dataset import datasets as D  # noqa: E402
from script.week5.timescale_inspection import gates5 as G  # noqa: E402
from script.week5.timescale_inspection import radius as RAD  # noqa: E402
from script.week5.timescale_inspection import run_phase1 as R1  # noqa: E402
from script.week5.timescale_inspection import run_phase5 as R5  # noqa: E402
from script.week5.timescale_inspection import tokenizers as TOK  # noqa: E402

SCHEMA_VERSION = 1
ORIGINAL_SEEDS = R5.SEEDS                  # 111, 222, 1538574472 -- never replaced
NEW_SEEDS = (333, 444)                     # declared now; the headline's extension seeds
SPANS_HUGADB, SPANS_LARA = (2, 4), (2, 4)
OWN_FILES = R5.OWN_FILES + ("script/week5/timescale_inspection/run_phase6.py",)


def source_hash() -> str:
    h = hashlib.sha256()
    for name in OWN_FILES + R5.TRAIN_FILES:
        h.update(name.encode())
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()[:16]


def stage_items(stage: str) -> list[R5.Item]:
    """S1a-S3, as declared in PHASE6_SPAN_CONFIRMATION.md. All `R5.Item`s, built with `R5.mk_cell`."""
    if stage == "S1a":
        return [R5.Item(R5.mk_cell(ds, "t0_raw", s, span=2), ("context",))
                for ds in ("babel2", "babel3") for s in ORIGINAL_SEEDS]
    if stage == "S1b":
        out = []
        for ds in ("babel2", "babel3"):
            for s in NEW_SEEDS:
                out.append(R5.Item(R5.mk_cell(ds, "t2_relational", s), ("context", "centre")))
                out.append(R5.Item(R5.mk_cell(ds, "t2_relational", s, span=2), ("context",)))
                out.append(R5.Item(R5.mk_cell(ds, "t0_raw", s), ("context", "centre")))
                out.append(R5.Item(R5.mk_cell(ds, "t0_raw", s, span=2), ("context",)))
        return out
    if stage == "S2":
        return [R5.Item(R5.mk_cell("hugadb", "t0_raw", s, span=sp), ("context",))
                for sp in SPANS_HUGADB for s in ORIGINAL_SEEDS]
    if stage == "S3":
        return [R5.Item(R5.mk_cell("lara", "t2_relational", s, span=sp), ("context",))
                for sp in SPANS_LARA for s in ORIGINAL_SEEDS]
    return []


STAGE_ORDER = ("R", "S1a", "S1b", "S2", "S3")


class Phase6Runner(R5.Runner):
    """Deliberately does not call `R5.Runner.__init__`: that hard-codes the `phase5` output directory and the
    phase-5 cell set into the resume identity. Every method it inherits (`prepare` overridden below, `ensure_train`,
    `ensure_decode`, `build_recs`, `run_item`, `job_dir`/`job_state`/`selected_state`, `unit_dir`/`unit_identity`,
    `blocked`) is reused unchanged; they depend only on `self.out`, `self.scale`, `self.impl`, `self.radius` and
    `self.device`, all set here."""

    def __init__(self, args):
        self.args = args
        self.smoke = bool(args.smoke)
        self.scale = replace(P.SMOKE, recording_cap=None) if self.smoke else P.FULL
        self.out = ROOT / "results" / "week5" / "timescale_inspection" / args.run_id / "phase6"
        for sub in ("prep", "jobs", "units"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.phase5_dir = ROOT / "results" / "week5" / "timescale_inspection" / args.run_id / args.phase5_dir \
            if not Path(args.phase5_dir).is_absolute() else Path(args.phase5_dir)
        self.deadline = time.perf_counter() + args.hours * 3600 if args.hours else None
        self.impl = source_hash()
        self.radius = RAD.CURRENT
        self.seeds = (111,) if self.smoke else ORIGINAL_SEEDS
        self.new_seeds = (333,) if self.smoke else NEW_SEEDS
        self.job_infos: dict[str, dict] = {}
        self.loaders: dict[str, Loader] = {}
        self.not_run: dict[str, str] = {}
        self.gates_path = self.out / "gates.json"
        self.gates: dict = json.loads(self.gates_path.read_text()) if self.gates_path.exists() else {}
        self.manifest_path = self.out / "manifest.json"
        self.base_cells = [it.cell for st in STAGE_ORDER[1:] for it in self.stage_items(st)]
        self.check_resume()

    # `stage_items` here is an instance method (needs `self.seeds`/`self.new_seeds` for --smoke), unlike R5's
    # module-level function of the same name.
    def stage_items(self, stage: str) -> list[R5.Item]:
        if self.smoke:
            # one seed, both span values, both datasets per stage -- a plumbing check, not a result
            if stage == "S1a":
                return [R5.Item(R5.mk_cell(ds, "t0_raw", self.seeds[0], span=2), ("context",)) for ds in ("babel2", "babel3")]
            if stage == "S1b":
                out = []
                for ds in ("babel2", "babel3"):
                    s = self.new_seeds[0]
                    out.append(R5.Item(R5.mk_cell(ds, "t2_relational", s), ("context", "centre")))
                    out.append(R5.Item(R5.mk_cell(ds, "t2_relational", s, span=2), ("context",)))
                    out.append(R5.Item(R5.mk_cell(ds, "t0_raw", s), ("context", "centre")))
                    out.append(R5.Item(R5.mk_cell(ds, "t0_raw", s, span=2), ("context",)))
                return out
            if stage == "S2":
                return [R5.Item(R5.mk_cell("hugadb", "t0_raw", self.seeds[0], span=sp), ("context",)) for sp in SPANS_HUGADB]
            if stage == "S3":
                return [R5.Item(R5.mk_cell("lara", "t2_relational", self.seeds[0], span=sp), ("context",)) for sp in SPANS_LARA]
            return []
        return stage_items(stage)

    def check_resume(self) -> None:
        ident = dict(schema=SCHEMA_VERSION, implementation_hash=self.impl, scale=self.scale.name,
                     outer_cap=self.scale.outer_cap, rungs=list(self.scale.rungs), radius=self.radius.name,
                     new_seeds=list(self.new_seeds), cells={c.cell_id: self.identity_of(c) for c in self.base_cells})
        if self.manifest_path.exists():
            old = json.loads(self.manifest_path.read_text())
            if old.get("identity") != ident:
                raise SystemExit(f"{self.out} was written by a different implementation, radius, scale, seed set or "
                                 f"cell set; choose a new --run-id or a different --phase5-dir.")
        write_json_atomic(self.manifest_path, dict(
            identity=ident, run_id=self.args.run_id, phase=6, status="running", smoke=self.smoke,
            python=platform.python_version(), started_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"), radius=self.radius.name))
        self.ident = ident

    # -- preparation: reuse ts5_v1/phase5 read-only wherever it verifies ------------------------------------------
    def phase5_prep_candidates(self, cell: R5.Cell) -> list[Path]:
        """Where Phase 5 may have already built this exact cell. G1's own T0 BABEL-2 seed-111 directory sits one
        level up from the rest (`phase5/prep/<id>`, not `phase5/prep/t0_raw/<id>`), because `run_gates` passes it a
        bare path; every other tokenizer's cells sit under `phase5/prep/<tok>/<id>`."""
        out = [self.phase5_dir / "prep" / cell.tok / cell.prep_id]
        # the preparation (tokens) does not depend on span, only on (dataset, tokenizer, seed, window) -- so the
        # bare G1 directory is a candidate for babel2/t0_raw/seed111 at ANY span, not only the default span-8 cell
        default_window = cell.window == D.SPECS[cell.dataset].window
        if cell.tok == "t0_raw" and cell.dataset == "babel2" and cell.seed == 111 and default_window:
            out.append(self.phase5_dir / "prep" / cell.prep_id)
        return out

    def prepare(self, cell: R5.Cell) -> Loader | None:
        if cell.prep_id in self.loaders:
            return self.loaders[cell.prep_id]
        for d in self.phase5_prep_candidates(cell):
            try:
                if TOK.is_prepared(cell.dataset, cell.tok, cell.seed, d, cell.window, self.recording_cap(cell)):
                    self.loaders[cell.prep_id] = Loader(d)
                    return self.loaders[cell.prep_id]
            except (OSError, KeyError, ValueError):
                pass
        return super().prepare(cell)  # not found upstream: build into phase6/prep/<tok>/<id>, as Phase 5 would

    # -- gates -------------------------------------------------------------------------------------------------------
    def inherit_phase5_gate(self, name: str) -> None:
        """Copy `name`'s result from `phase5/gates.json` (read-only) rather than re-running it: Phase 5 already
        established it (e.g. G3, LARa's channel semantics), and S3 needs it to run T2 on LARa at all. A missing or
        failed phase-5 gate blocks whatever depends on it here too, exactly as it did in Phase 5."""
        path = self.phase5_dir / "gates.json"
        if not path.exists():
            self.gates[name] = dict(status="failed", reason=f"{path} not found (phase5 gates were not inherited)")
            return
        g5 = json.loads(path.read_text())
        if name not in g5:
            self.gates[name] = dict(status="failed", reason=f"'{name}' not recorded in {path}")
            return
        self.gates[name] = dict(g5[name], inherited_from=str(path))

    def run_gates(self) -> None:
        self.inherit_phase5_gate("g3")  # LARa (S3) needs this; never re-run, since Phase 5 already verified it
        if self.smoke:
            self.gates["r1"] = dict(status="not_applicable_smoke", implementation_hash=self.impl)
            self.gates["r2"] = dict(status="not_applicable_smoke", implementation_hash=self.impl)
            write_json_atomic(self.gates_path, self.gates)
            return
        self._gate("r1", self.gate_r1)
        self._gate("r2", self.gate_r2)
        write_json_atomic(self.gates_path, self.gates)

    def gate_r1(self) -> dict:
        """The generalized path, run again here, reproduces ts5_v1's T2 BABEL-3 seed-111 preparation (the
        artifacts S1b's new-seed T2 cells are built with the same code as) -- codes/features/grouping/split
        exactly, K-means centres to 1e-4 (the same tolerance as G1, for the same reason: see gates5.gate_g1)."""
        ref_dir = self.phase5_dir / "prep" / "t2_relational" / "babel3__t2_relational__seed111"
        if not ref_dir.exists():
            return dict(status="failed", reason=f"reference not found: {ref_dir}")
        scratch = self.out / "gates" / "r1_scratch"
        TOK.prepare_cell("babel3", "t2_relational", 111, scratch)
        a, b = Loader(ref_dir), Loader(scratch)
        fam = TOK.SPECS["t2_relational"].family
        names = sorted(a.recording_names())
        checks = dict(same_recordings=names == sorted(b.recording_names()))
        if checks["same_recordings"]:
            checks["codes"] = all((a.recording(fam, n)["codes"] == b.recording(fam, n)["codes"]).all() for n in names)
            checks["continuous"] = all((a.recording(fam, n)["continuous"] == b.recording(fam, n)["continuous"]).all()
                                       for n in names)
        checks["mean"] = bool(np.array_equal(a.mean(fam), b.mean(fam)))
        checks["rms"] = float(a.rms(fam)) == float(b.rms(fam))
        checks["grouping"] = bool(np.array_equal(a.grouping(fam), b.grouping(fam)))
        checks["split"] = (json.loads((ref_dir / "split.json").read_text()) == json.loads((scratch / "split.json").read_text()))
        lookup_diff = float(np.abs(a.lookup(fam).astype(np.float64) - b.lookup(fam).astype(np.float64)).max())
        checks["lookup"] = lookup_diff <= G.LOOKUP_ATOL
        ok = all(checks.values())
        return dict(status="passed" if ok else "failed", checks=checks, lookup_max_abs_diff=lookup_diff)

    def gate_r2(self) -> dict:
        """Decoding `babel2__t2_relational__seed111` `centre` from the reused (not rebuilt) preparation reproduces
        ts5_v1/phase5's own saved unit exactly, on every rung and every metric -- confirms the reuse path feeds the
        decoder byte-identical inputs."""
        ref_path = self.phase5_dir / "units" / "babel2__t2_relational__seed111__centre" / "info.json"
        if not ref_path.exists():
            return dict(status="failed", reason=f"reference unit not found: {ref_path}")
        ref = json.loads(ref_path.read_text())
        cell = R5.mk_cell("babel2", "t2_relational", 111)
        loader = self.prepare(cell)
        if loader is None:
            return dict(status="failed", reason="preparation unavailable")
        spec = TOK.dataset_spec(cell.dataset, cell.window)
        names = sorted(loader.recording_names())
        recs, mu0, starts = self.build_recs(cell, "centre", loader, names, None)
        gts = D.load_gts(spec, names)
        frame_counts = {n: loader.frame_count(n) for n in names}
        res = R1.fit_and_score(recs, mu0, spec.num_states, R1.coeffs_for(0.3), self.radius, self.scale.outer_cap,
                               self.scale.rungs, gts, starts, frame_counts, self.deadline)
        got = {r["rung_steps"]: r for r in res["ladder"]}
        want = {r["rung_steps"]: r for r in ref["ladder"]}
        diffs = {}
        for rung in sorted(set(got) & set(want)):
            for m in ("MoF", "Edit", "F1_10", "F1_25", "F1_50"):
                diffs[f"{rung}/{m}"] = abs(got[rung][m] - want[rung][m])
        ok = set(got) == set(want) and all(v == 0 for v in diffs.values())
        return dict(status="passed" if ok else "failed", diffs=diffs, reference_f1_50=ref["ladder"][-1].get("F1_50"))

    def blocked(self, cell: R5.Cell) -> str | None:
        if self.gates.get("r1", {}).get("status") == "failed":
            return "R1 failed (the generalized preparation does not reproduce ts5_v1's T2 artifacts)"
        if self.gates.get("r2", {}).get("status") == "failed":
            return "R2 failed (decoding the reused preparation does not reproduce ts5_v1's saved unit)"
        return super().blocked(cell)

    # -- orchestration -----------------------------------------------------------------------------------------------
    def execute(self) -> None:
        counts: dict[str, int] = {}
        self.run_gates()
        for stage in STAGE_ORDER[1:]:
            items = self.stage_items(stage)
            for i, item in enumerate(items, 1):
                self.run_item(item, counts, f"{stage} {i}/{len(items)}")
        m = json.loads(self.manifest_path.read_text())
        m.update(status="finished" if not self.expired() else "budget_stop", counts=counts,
                 not_run=self.not_run, finished_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        write_json_atomic(self.manifest_path, m)
        from script.week5.timescale_inspection import report6
        print(f"report: {report6.write_report(self.out, self.phase5_dir, self.args.context_run, self.args.direct_run)}")

    def dry_run(self) -> None:
        total, seen = 0.05 * 60, set()  # gates ~3 min
        print("stage R: gates R1-R2 (~3 min)")
        for stage in STAGE_ORDER[1:]:
            items = self.stage_items(stage)
            minutes = 0.0
            for it in items:
                c = it.cell
                reused = any(TOK.is_prepared(c.dataset, c.tok, c.seed, d, c.window, self.recording_cap(c))
                            for d in self.phase5_prep_candidates(c)) if not self.smoke else False
                if not reused and c.prep_id not in seen:
                    seen.add(c.prep_id)
                    minutes += R5.COST["prep"][c.dataset]
                if "context" in it.arms:
                    minutes += R5.COST["train"][c.dataset] + R5.COST["context"][c.dataset]
                minutes += sum(R5.COST["direct"][c.dataset] for a in it.arms if a != "context")
                print(f"  {c.cell_id}: arms {','.join(it.arms)}" + ("  [reused prep]" if reused else ""))
            total += minutes
            print(f"stage {stage}: {len(items)} cells, ~{minutes:.0f} min, cumulative ~{total / 60:.1f} h")
        print(f"projected total ~{total / 60:.1f} h (single-pass, otherwise idle machine)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Time-scale inspection, phase 6 (confirming the span rule)")
    ap.add_argument("--run-id", default="ts5_v1", help="written into <run-id>/phase6; defaults to the Phase 5 run")
    ap.add_argument("--hours", type=float, default=None, help="elapsed-time budget; completed units are kept")
    ap.add_argument("--context-run", default="rkx_v2", help="position-only run to read the T0 `context` baseline from")
    ap.add_argument("--direct-run", default="ts1_v1", help="Phase 1 run to read the T0 `pca`/`centre` baseline from")
    ap.add_argument("--phase5-dir", default="phase5", help="phase-5 subdirectory to reuse preparations from")
    ap.add_argument("--smoke", action="store_true", help="tiny budgets, capped recordings, one seed per side (plumbing check)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not (args.dry_run or args.smoke) and not args.hours:
        raise SystemExit("a decoding run requires an explicit elapsed-time budget: pass --hours")
    runner = Phase6Runner(args)
    if args.dry_run:
        runner.dry_run()
    else:
        runner.execute()


if __name__ == "__main__":
    main()
