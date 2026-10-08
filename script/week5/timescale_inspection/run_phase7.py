"""Phase 7 of the time-scale inspection: a prospective BABEL-1 evaluation of the frozen T2 / short-span recipe, and an
exploratory, retrospective screen of label-free span selectors.

BABEL-1 is not an untouched test set: its T0 results were examined in rkx_v2 / Phases 0-1, and 10 of its 1,960
recordings are byte-identical to BABEL-2/3 recordings. What is new is running the frozen recipe there.

  G   G7: the generalized T0 preparation on BABEL-1 reproduces rkx_v2's (gates the T0 cells of H2/H3)
  H1  BABEL-1: T2 span 2 / span 8 `context`, T2 `centre`, 3 seeds
  H2  BABEL-1 T0 span 2 `context`, 3 seeds (completes the 3-seed tokenizer x span factorial)
  S0  re-decode ts5_v1's span checkpoints (BABEL-2/3, LARa T2; HuGaDB T0) with frame predictions saved; each must
      reproduce its saved ladder exactly
  H3  BABEL-1 seeds 333/444: T2 span 2, T2 span 8, T0 span 8 (`context`)
  K   the declared selectors (K1 stability, K3 self-consistency, K4 feature time scale), exploratory; a missing declared
      cell makes the screen incomplete, never a pass. The selector/verdict code is hashed separately (analysis hash).

Design and the declared rules: docs/plans/week5/TIMESCALE_INSPECTION/PHASE7_LABEL_FREE_SCALE.md. No shared module and
no earlier phase file is edited; BABEL-1 tokenizer keys are registered at runtime.

Usage:
  python -m script.week5.timescale_inspection.run_phase7 --run-id ts5_v1 --hours 10
  python -m script.week5.timescale_inspection.run_phase7 --run-id ts5_v1 --dry-run
  python -m script.week5.timescale_inspection.run_phase7 --run-id ts7_smoke --phase5-dir <abs ts5_v1/phase5> --smoke
"""

from __future__ import annotations

import os

_CAP = str(min(8, os.cpu_count() or 8))
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_var] = _CAP

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from script.action_transport import continuous  # noqa: E402
from script.action_transport.budget import DeadlineExceeded  # noqa: E402
from script.fsq_reconstruction.evaluate import FinalSolveInvalid, contextual_final_ladder  # noqa: E402
from script.fsq_reconstruction.experiment import write_json_atomic  # noqa: E402
from script.quantized_input_comparison import ot_arms as ot  # noqa: E402
from script.quantized_input_comparison.artifacts import Loader  # noqa: E402
from script.week5.contextualizer_calibration import plan as P  # noqa: E402
from script.week5.raw_kmeans_cross_dataset import datasets as D  # noqa: E402
from script.week5.timescale_inspection import gates5 as G  # noqa: E402
from script.week5.timescale_inspection import radius as RAD  # noqa: E402
from script.week5.timescale_inspection import run_phase1 as R1  # noqa: E402
from script.week5.timescale_inspection import run_phase5 as R5  # noqa: E402
from script.week5.timescale_inspection import run_phase6 as R6  # noqa: E402
from script.week5.timescale_inspection import tokenizers as TOK  # noqa: E402

SCHEMA_VERSION = 1
ORIGINAL_SEEDS = R5.SEEDS
NEW_SEEDS = (333, 444)
OWN_FILES = R6.OWN_FILES + ("script/week5/timescale_inspection/run_phase7.py",)
HELDOUT = "babel1"
# minutes, for the dry-run projection only: BABEL-1 from rkx_v2 (decode ~5-8 min context) and ts1_v1 (~15 min centre)
COST = {k: dict(v) for k, v in R5.COST.items()}
for k, v in dict(prep=1.0, train=2.0, context=7.0, direct=15.0).items():
    COST[k][HELDOUT] = v


def register_heldout_tokenizers() -> None:
    """T0 and T2 restricted to BABEL-1, registered in-process only (tokenizers.py is never edited, so every earlier
    run's identity is unchanged). Identical to T0/T2 except for their key, family name and dataset list."""
    for base, key in (("t0_raw", "t0_raw_b1"), ("t2_relational", "t2_relational_b1")):
        if key not in TOK.SPECS:
            TOK.SPECS[key] = dataclasses.replace(TOK.SPECS[base], key=key, datasets=(HELDOUT,), selectable=False)


register_heldout_tokenizers()


def source_hash() -> str:
    h = hashlib.sha256()
    for name in OWN_FILES + R5.TRAIN_FILES:
        h.update(name.encode())
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()[:16]


def heldout_items(stage: str, seeds=ORIGINAL_SEEDS, new_seeds=NEW_SEEDS) -> list[R5.Item]:
    mk = R5.mk_cell
    if stage == "H1":
        out = []
        for s in seeds:
            out.append(R5.Item(mk(HELDOUT, "t2_relational_b1", s, span=2), ("context",)))
            out.append(R5.Item(mk(HELDOUT, "t2_relational_b1", s), ("context", "centre")))
        return out
    if stage == "H2":
        return [R5.Item(mk(HELDOUT, "t0_raw_b1", s, span=2), ("context",)) for s in seeds]
    if stage == "H3":
        out = []
        for s in new_seeds:
            out.append(R5.Item(mk(HELDOUT, "t2_relational_b1", s, span=2), ("context",)))
            out.append(R5.Item(mk(HELDOUT, "t2_relational_b1", s), ("context",)))
            out.append(R5.Item(mk(HELDOUT, "t0_raw_b1", s), ("context",)))
        return out
    return []


def s0_items(seeds=ORIGINAL_SEEDS) -> list[tuple[str, R5.Cell]]:
    """(source phase, cell) for every span checkpoint to re-decode. Only seeds present at every candidate span enter
    the criteria, so only the original seeds are re-decoded (seeds 333/444 have no span-4 checkpoints)."""
    out = []
    for ds in ("babel2", "babel3"):
        for sp in (2, 4, 8):
            out += [("phase5", R5.mk_cell(ds, "t2_relational", s, span=sp)) for s in seeds]
    out += [("phase5", R5.mk_cell("lara", "t2_relational", s)) for s in seeds]
    for sp in (2, 4):
        out += [("phase6", R5.mk_cell("lara", "t2_relational", s, span=sp)) for s in seeds]
    for sp in (2, 4):
        out += [("phase6", R5.mk_cell("hugadb", "t0_raw", s, span=sp)) for s in seeds]
    return out


STAGE_ORDER = ("G", "H1", "H2", "S0", "H3", "K")  # the 3-seed BABEL-1 comparison completes before the long re-decodes


def fit_and_score_keep(recs, mu0, num_states, coeffs, setting, outer_cap, rungs, gts, starts, frame_counts, deadline):
    """`run_phase1.fit_and_score`, line for line, except that the final rung's frame predictions are returned."""
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
                half_widths_used={str(k): v for k, v in sorted(used.items())}, seconds=time.perf_counter() - started,
                predictions={n: np.asarray(v).astype(np.int16) for n, v in predictions.items()})


def ladders_equal(a: list[dict], b: list[dict]) -> tuple[bool, float]:
    """Exact equality on every rung and every metric; also returns the largest absolute difference seen."""
    ra, rb = {r["rung_steps"]: r for r in a}, {r["rung_steps"]: r for r in b}
    if set(ra) != set(rb):
        return False, float("inf")
    diffs = [abs(ra[k][m] - rb[k][m]) for k in ra for m in ("MoF", "Edit", "F1_10", "F1_25", "F1_50")]
    return max(diffs) == 0.0, max(diffs)


class SourceView(R6.Phase6Runner):
    """Read-only view of an earlier phase's directory (ts5_v1/phase5 or phase6): verifies checkpoints with the
    inherited `job_state`, finds preparations, and never builds or writes anything."""

    def __init__(self, phase_dir: Path, phase5_dir: Path, device_cpu: bool):
        self.args = argparse.Namespace(cpu=device_cpu)
        self.smoke, self.scale, self.out, self.phase5_dir = False, P.FULL, Path(phase_dir), Path(phase5_dir)
        self.deadline, self.radius, self.job_infos, self.loaders, self.gates = None, RAD.CURRENT, {}, {}, {}

    def prepare(self, cell):
        if cell.prep_id in self.loaders:
            return self.loaders[cell.prep_id]
        for d in self.phase5_prep_candidates(cell) + [self.out / "prep" / cell.tok / cell.prep_id]:
            try:
                if TOK.is_prepared(cell.dataset, cell.tok, cell.seed, d, cell.window, None):
                    self.loaders[cell.prep_id] = Loader(d)
                    return self.loaders[cell.prep_id]
            except (OSError, KeyError, ValueError):
                pass
        return None


class Phase7Runner(R6.Phase6Runner):
    def __init__(self, args):
        self.args = args
        self.smoke = bool(args.smoke)
        self.scale = replace(P.SMOKE, recording_cap=None) if self.smoke else P.FULL
        base = ROOT / "results" / "week5" / "timescale_inspection" / args.run_id
        self.out = base / "phase7"
        for sub in ("prep", "jobs", "units", "s0"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.phase5_dir = Path(args.phase5_dir) if Path(args.phase5_dir).is_absolute() else base / args.phase5_dir
        self.source_root = self.phase5_dir.parent
        self.deadline = time.perf_counter() + args.hours * 3600 if args.hours else None
        self.impl = source_hash()
        self.radius = RAD.CURRENT
        self.seeds = (111,) if self.smoke else ORIGINAL_SEEDS
        self.new_seeds = (333,) if self.smoke else NEW_SEEDS
        self.job_infos, self.loaders, self.not_run = {}, {}, {}
        self.gates_path = self.out / "gates.json"
        self.gates = json.loads(self.gates_path.read_text()) if self.gates_path.exists() else {}
        self.manifest_path = self.out / "manifest.json"
        self.base_cells = [it.cell for st in ("H1", "H2", "H3") for it in self.stage_items(st)]
        self.s0 = s0_items(self.seeds)[:1] if self.smoke else s0_items(self.seeds)
        self.check_resume()

    def stage_items(self, stage):
        return heldout_items(stage, self.seeds, self.new_seeds)

    def check_resume(self) -> None:
        ident = dict(schema=SCHEMA_VERSION, implementation_hash=self.impl, scale=self.scale.name,
                     outer_cap=self.scale.outer_cap, rungs=list(self.scale.rungs), radius=self.radius.name,
                     cells={c.cell_id: self.identity_of(c) for c in self.base_cells},
                     s0=[f"{ph}/{c.cell_id}" for ph, c in self.s0], phase5_dir=str(self.phase5_dir))
        if self.manifest_path.exists():
            old = json.loads(self.manifest_path.read_text())
            if old.get("identity") != ident:
                raise SystemExit(f"{self.out} was written by a different implementation or cell set; choose a new --run-id.")
        from script.week5.timescale_inspection import report7  # local: report7 imports this module
        write_json_atomic(self.manifest_path, dict(
            identity=ident, run_id=self.args.run_id, phase=7, status="running", smoke=self.smoke,
            analysis_hash_at_launch=report7.analysis_hash(),
            python=platform.python_version(), started_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"), radius=self.radius.name))
        self.ident = ident

    # -- gates ---------------------------------------------------------------------------------------------------------
    def gate_g7(self) -> dict:
        """As gates5.gate_g1, for T0 on BABEL-1 (tokenizer key t0_raw_b1): codes, features, starts/lengths, mean, RMS,
        grouping and split exact; K-means centres to 1e-4."""
        ref = ROOT / "results" / "week5" / "raw_kmeans_cross_dataset" / self.args.context_run / "prep" / f"{HELDOUT}__global__seed111"
        if not ref.exists():
            return dict(status="failed", reason=f"reference not found: {ref}")
        cell = R5.mk_cell(HELDOUT, "t0_raw_b1", 111)
        mine = self.prepare(cell)
        a, fa, fb = Loader(ref), "raw_kmeans", TOK.SPECS["t0_raw_b1"].family
        names = sorted(a.recording_names())
        checks = dict(same_recordings=names == sorted(mine.recording_names()))
        if checks["same_recordings"]:
            for key in ("codes", "continuous", "starts", "lengths"):
                checks[key] = all(np.array_equal(a.recording(fa, n)[key], mine.recording(fb, n)[key]) for n in names)
        checks["mean"] = bool(np.array_equal(a.mean(fa), mine.mean(fb)))
        checks["rms"] = float(a.rms(fa)) == float(mine.rms(fb))
        checks["grouping"] = bool(np.array_equal(a.grouping(fa), mine.grouping(fb)))
        checks["split"] = json.loads((ref / "split.json").read_text()) == json.loads((self.prep_dir(cell) / "split.json").read_text())
        diff = float(np.abs(a.lookup(fa).astype(np.float64) - mine.lookup(fb).astype(np.float64)).max())
        checks["lookup"] = diff <= G.LOOKUP_ATOL
        return dict(status="passed" if all(checks.values()) else "failed", checks=checks, lookup_max_abs_diff=diff)

    def run_gates(self) -> None:
        if self.smoke:
            self.gates["g7"] = dict(status="not_applicable_smoke", implementation_hash=self.impl)
        else:
            self._gate("g7", self.gate_g7)
        write_json_atomic(self.gates_path, self.gates)

    def blocked(self, cell) -> str | None:
        if cell.tok == "t0_raw_b1" and self.gates.get("g7", {}).get("status") == "failed":
            return "G7 failed (the generalized T0 preparation does not reproduce rkx_v2 on BABEL-1)"
        return None

    def prepare(self, cell):  # BABEL-1 has nothing upstream; always build into phase7/prep
        return R5.Runner.prepare(self, cell)

    # -- decoding with predictions saved (H cells) -----------------------------------------------------------------
    def unit_done(self, cell, arm, identity):
        info = super().unit_done(cell, arm, identity)
        return info if info and (self.unit_dir(cell, arm) / "predictions.npz").exists() else None

    def ensure_decode(self, cell, arm) -> str:
        info = self.job_state(cell) if arm == "context" else {}
        if arm == "context" and info is None:
            return "no_checkpoint"
        state, state_sha = (self.selected_state(cell, info) if arm == "context" else (None, None))
        identity = self.unit_identity(cell, arm, state_sha)
        if self.unit_done(cell, arm, identity):
            return "cached"
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
            res = fit_and_score_keep(recs, mu0, spec.num_states, R1.coeffs_for(0.3), self.radius, self.scale.outer_cap,
                                     self.scale.rungs, gts, starts, frame_counts, self.deadline)
        except DeadlineExceeded:
            return "deadline"
        except FinalSolveInvalid as exc:
            udir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(udir / "info.json", dict(status="invalid", identity=identity, reason=str(exc), **meta))
            return "invalid"
        udir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(udir / "predictions.npz", **res.pop("predictions"))
        write_json_atomic(udir / "info.json", dict(
            status="done", identity=identity, checkpoint_step=(info or {}).get("roles", {}).get("selected"), **res, **meta))
        return "done"

    # -- S0: re-decode earlier checkpoints with predictions saved ------------------------------------------------------
    def s0_dir(self, phase: str, cell) -> Path:
        return self.out / "s0" / f"{phase}__{cell.cell_id}__context"

    def run_s0_item(self, phase: str, cell, counts: dict, tag: str) -> None:
        src = SourceView(self.source_root / phase, self.phase5_dir, bool(self.args.cpu))
        saved_path = src.out / "units" / f"{cell.cell_id}__context" / "info.json"
        odir = self.s0_dir(phase, cell)
        if not saved_path.exists():
            counts["s0_missing_source"] = counts.get("s0_missing_source", 0) + 1
            print(f"[{tag}] {phase}/{cell.cell_id}: no saved unit; skipped", flush=True)
            return
        saved = json.loads(saved_path.read_text())
        if (odir / "info.json").exists() and (odir / "predictions.npz").exists():
            old = json.loads((odir / "info.json").read_text())
            if old.get("source_identity") == saved.get("identity") and old.get("implementation_hash") == self.impl:
                counts["s0_cached"] = counts.get("s0_cached", 0) + 1
                return
        if self.expired():
            counts["deadline"] = counts.get("deadline", 0) + 1
            return
        info = src.job_state(cell)
        loader = src.prepare(cell)
        if info is None or loader is None:
            counts["s0_unverifiable"] = counts.get("s0_unverifiable", 0) + 1
            print(f"[{tag}] {phase}/{cell.cell_id}: checkpoint or preparation does not verify; skipped", flush=True)
            return
        state, state_sha = src.selected_state(cell, info)
        spec = TOK.dataset_spec(cell.dataset, cell.window)
        names = sorted(loader.recording_names())
        t0 = time.perf_counter()
        recs, mu0, starts = src.build_recs(cell, "context", loader, names, state)
        gts = D.load_gts(spec, names)
        frame_counts = {n: loader.frame_count(n) for n in names}
        try:
            res = fit_and_score_keep(recs, mu0, spec.num_states, R1.coeffs_for(0.3), RAD.CURRENT, P.FULL.outer_cap,
                                     P.FULL.rungs, gts, starts, frame_counts, self.deadline)
        except DeadlineExceeded:
            counts["deadline"] = counts.get("deadline", 0) + 1
            return
        same, maxdiff = ladders_equal(res["ladder"], saved["ladder"])
        odir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(odir / "predictions.npz", **res.pop("predictions"))
        write_json_atomic(odir / "info.json", dict(
            status="done", reproduced=same, max_abs_diff=maxdiff, source_unit=str(saved_path.relative_to(ROOT)),
            source_identity=saved.get("identity"), state_sha=state_sha, implementation_hash=self.impl,
            dataset=cell.dataset, tokenizer=cell.tok, seed=cell.seed, window=cell.window, span=cell.span, arm="context",
            **res))
        key = "s0_reproduced" if same else "s0_not_reproduced"
        counts[key] = counts.get(key, 0) + 1
        print(f"[{tag}] {phase}/{cell.cell_id}: {'reproduced' if same else f'NOT reproduced (max diff {maxdiff})'} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    # -- orchestration -------------------------------------------------------------------------------------------------
    def execute(self) -> None:
        counts: dict[str, int] = {}
        self.run_gates()
        for stage in STAGE_ORDER[1:]:
            if stage == "S0":
                for i, (phase, cell) in enumerate(self.s0, 1):
                    self.run_s0_item(phase, cell, counts, f"S0 {i}/{len(self.s0)}")
            elif stage == "K":
                from script.week5.timescale_inspection import report7
                try:
                    path = report7.compute_criteria(self.out, self.source_root, self.args.context_run)
                    print(f"[K] criteria: {path}", flush=True)
                except Exception as exc:  # the report must still be written; the failure is recorded, not hidden
                    counts["criteria_failed"] = 1
                    print(f"[K] criteria FAILED: {type(exc).__name__}: {exc}", flush=True)
            else:
                items = self.stage_items(stage)
                for i, item in enumerate(items, 1):
                    self.run_item(item, counts, f"{stage} {i}/{len(items)}")
        m = json.loads(self.manifest_path.read_text())
        m.update(status="finished" if not self.expired() else "budget_stop", counts=counts, not_run=self.not_run,
                 finished_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        write_json_atomic(self.manifest_path, m)
        from script.week5.timescale_inspection import report7
        print(f"report: {report7.write_report(self.out, self.source_root, self.args.context_run, self.args.direct_run)}")

    def dry_run(self) -> None:
        total, seen = 2.0, set()
        print("stage G: gate G7 (~2 min)")
        for stage in STAGE_ORDER[1:]:
            minutes = 0.0
            if stage == "S0":
                for phase, c in self.s0:
                    minutes += COST["context"][c.dataset]
                    print(f"  re-decode {phase}/{c.cell_id}")
                print(f"stage S0: {len(self.s0)} re-decodes, ~{minutes:.0f} min, cumulative ~{(total + minutes) / 60:.1f} h")
            elif stage == "K":
                minutes = 5.0
                print("stage K: criteria and report (~5 min)")
            else:
                items = self.stage_items(stage)
                for it in items:
                    c = it.cell
                    if c.prep_id not in seen:
                        seen.add(c.prep_id)
                        minutes += COST["prep"][c.dataset]
                    if "context" in it.arms:
                        minutes += COST["train"][c.dataset] + COST["context"][c.dataset]
                    minutes += sum(COST["direct"][c.dataset] for a in it.arms if a != "context")
                    print(f"  {c.cell_id}: arms {','.join(it.arms)}")
                print(f"stage {stage}: {len(items)} cells, ~{minutes:.0f} min, cumulative ~{(total + minutes) / 60:.1f} h")
            total += minutes
        print(f"projected total ~{total / 60:.1f} h (single-pass, otherwise idle machine)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Time-scale inspection, phase 7 (held-out test and label-free span criteria)")
    ap.add_argument("--run-id", default="ts5_v1")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--context-run", default="rkx_v2")
    ap.add_argument("--direct-run", default="ts1_v1")
    ap.add_argument("--phase5-dir", default="phase5", help="phase-5 directory of the source run (relative to the run id, or absolute)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not (args.dry_run or args.smoke) and not args.hours:
        raise SystemExit("a decoding run requires an explicit elapsed-time budget: pass --hours")
    runner = Phase7Runner(args)
    if args.dry_run:
        runner.dry_run()
    else:
        runner.execute()


if __name__ == "__main__":
    main()
