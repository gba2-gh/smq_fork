"""Replicate phase-5 HuGaDB T1/context without changing the historical run.

From SMQ: python -m script.week5.timescale_inspection.run_hugadb_whiten_seeds --hours 2
Completed training/decodes are reused when the same command is run again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from script.week5.timescale_inspection import run_phase5 as R


class HugadbRunner(R.Runner):
    def check_resume(self) -> None:
        # Runner.__init__ calls this before writing its manifest. Record only
        # the requested cells and include this launcher's code in its identity.
        self.base_cells = [R.mk_cell("hugadb", "t1_whiten", s) for s in self.args.seeds]
        self.impl = hashlib.sha256(
            self.impl.encode() + Path(__file__).read_bytes()
        ).hexdigest()[:16]
        super().check_resume()

    def execute(self) -> None:
        self.run_gates()
        if self.gates.get("g1", {}).get("status") != "passed":
            raise SystemExit(f"G1 failed; inspect {self.gates_path}")
        counts = {}
        for cell in self.base_cells:
            self.run_item(R.Item(cell, ("context",)), counts, "HuGaDB T1 replication")
        rows = []
        for cell in self.base_cells:
            path = self.unit_dir(cell, "context") / "info.json"
            if path.exists():
                info = json.loads(path.read_text())
                if info.get("status") == "done":
                    last = max(info["ladder"], key=lambda r: r["rung_steps"])
                    rows.append(dict(seed=cell.seed, checkpoint_step=info["checkpoint_step"],
                                     **{k: last[k] for k in ("MoF", "Edit", "F1_10", "F1_25", "F1_50")}))
        complete = len(rows) == len(self.base_cells)
        manifest = json.loads(self.manifest_path.read_text())
        manifest.update(status="finished" if complete else "incomplete", counts=counts,
                        not_run=self.not_run, finished_local=R.time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        R.write_json_atomic(self.manifest_path, manifest)
        summary = dict(rows=rows, status=manifest["status"],
                       historical_seed111="ts5_v1/phase5/units/hugadb__t1_whiten__seed111__context/info.json")
        R.write_json_atomic(self.out / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        if rows:
            scores = [r["F1_50"] for r in rows]
            print(f"New seeds only: F1@50 {statistics.mean(scores):.2f} +/- {statistics.pstdev(scores):.2f}")
        if not complete:
            raise SystemExit("Incomplete: inspect manifest and rerun this command to finish remaining units.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="ts5_hugadb_whiten_replication")
    ap.add_argument("--seeds", default="222,1538574472", help="Comma-separated seeds; optionally include 111 for a fresh replication")
    ap.add_argument("--hours", type=float)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    try:
        args.seeds = tuple(int(s.strip()) for s in args.seeds.split(","))
    except ValueError:
        ap.error("--seeds must contain comma-separated integers")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        ap.error("seeds must be unique integers in [0, 2**32)")
    if Path(args.run_id).name != args.run_id or args.run_id in (".", "..", "ts5_v1"):
        ap.error("use a new run-id consisting of a single directory name")
    if not args.dry_run and (args.hours is None or args.hours <= 0):
        ap.error("a run requires an explicit positive --hours budget")
    if args.dry_run:
        print(f"Output: {R.ROOT / 'results/week5/timescale_inspection' / args.run_id / 'phase5'}")
        print(f"Current phase-5 implementation: {R.source_hash()}")
        for seed in args.seeds:
            cell = R.mk_cell("hugadb", "t1_whiten", seed)
            print(f"{cell.cell_id} / context: {R.P.effective_settings(R.make_job(cell), R.P.FULL)}")
        print("Full budgets: PCA-64 whitened, K-means-512, 15-frame windows; OT outer cap 50, rungs 500/1000/2000/4000.")
        return
    args.smoke = False
    args.radius = "current"
    args.context_run = "rkx_v2"
    args.direct_run = "ts1_v1"
    HugadbRunner(args).execute()


if __name__ == "__main__":
    main()
