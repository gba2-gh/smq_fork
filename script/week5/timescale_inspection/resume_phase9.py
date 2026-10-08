"""Run Phase 9 in fresh processes at dataset/stage boundaries.

The numerical implementation stays in run_phase9.py.  This driver deliberately
does not enter its implementation hash: it only schedules the same runner methods
and lets each worker release its CPU/GPU allocations before the next stage.
Existing Phase 9 cells can therefore resume without changing their identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODULE = "script.week5.timescale_inspection.resume_phase9"
DATASETS = ("babel2", "babel3", "lara")
THREAD_CAP = str(min(8, os.cpu_count() or 8))
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = THREAD_CAP


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="ts5_v1")
    ap.add_argument("--source-run", default="ts5_v1")
    ap.add_argument("--stage", choices=("dev", "P"), default="dev")
    ap.add_argument("--hours", type=float)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--step", choices=("gates", "A", "B", "C", "finalize"), help=argparse.SUPPRESS)
    ap.add_argument("--dataset", help=argparse.SUPPRESS)
    ap.add_argument("--budget-stop", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--session", help=argparse.SUPPRESS)
    return ap


def output_dir(args: argparse.Namespace) -> Path:
    phase = "phase9p" if args.stage == "P" else "phase9"
    return ROOT / "results" / "week5" / "timescale_inspection" / args.run_id / phase


def run_step(args: argparse.Namespace) -> None:
    # Import the heavy numerical modules only in the worker process.
    from script.fsq_reconstruction.experiment import write_json_atomic
    from script.week5.timescale_inspection import gates9, report9
    from script.week5.timescale_inspection.run_phase9 import Phase9Runner

    manifest = output_dir(args) / "manifest.json"
    old_start = json.loads(manifest.read_text()).get("started_local") if manifest.exists() else None
    runner = Phase9Runner(args)
    if old_start:
        m = json.loads(manifest.read_text())
        m["started_local"] = old_start
        write_json_atomic(manifest, m)

    counts: dict[str, int] = {}
    if args.step == "gates":
        gates9.run_gates(runner)
    elif args.step == "B":
        runner.build_i2(args.dataset)
    elif args.step in ("A", "C"):
        items = runner.items(args.dataset, args.step)
        for i, item in enumerate(items, 1):
            runner.run_item(item, counts, f"{args.dataset} {args.step} {i}/{len(items)}")
    elif args.step == "finalize":
        steps = sorted((runner.out / "execution").glob("*.json"))
        not_run: dict[str, str] = {}
        all_counts: dict[str, int] = {}
        for path in steps:
            entry = json.loads(path.read_text())
            if entry.get("session") != args.session:
                continue
            not_run.update(entry.get("not_run", {}))
            for key, value in entry.get("counts", {}).items():
                all_counts[key] = all_counts.get(key, 0) + value
        m = json.loads(manifest.read_text())
        m.update(status="budget_stop" if args.budget_stop else "finished", counts=all_counts, not_run=not_run,
                 finished_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 execution_driver_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        write_json_atomic(manifest, m)
        print(f"report: {report9.write_report(runner.out, runner.source_root, prospective=runner.prospective)}", flush=True)
        return

    path = runner.out / "execution" / f"{args.dataset or 'preflight'}_{args.step}.json"
    write_json_atomic(path, dict(session=args.session, dataset=args.dataset, step=args.step, counts=counts, not_run=runner.not_run,
                                 expired=runner.expired(), finished_local=time.strftime("%Y-%m-%dT%H:%M:%S%z")))


MAX_ATTEMPTS = 4
RETRY_PAUSE_S = 15


def run_with_retries(cmd: list[str], label: str, deadline: float | None) -> None:
    """Run one stage worker; if it dies (nonzero exit, e.g. a native BLAS/heap crash such as 0xC0000005 or 0xC0000374,
    which leave no Python traceback) run it again. Every unit a worker finishes is cached by identity and written
    atomically, so a retry redoes only the unit that was in flight; the result is identical to an uninterrupted
    run. A stage that fails every attempt stops the run: a deterministic failure is never papered over."""
    cmd = list(cmd)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode == 0:
            return
        code = proc.returncode & 0xFFFFFFFF
        print(f"[process] {label}: worker exited with code {proc.returncode} (0x{code:08X}), attempt {attempt}/{MAX_ATTEMPTS}",
              flush=True)
        if attempt == MAX_ATTEMPTS or (deadline is not None and time.monotonic() >= deadline):
            raise SystemExit(f"stage {label} failed {attempt} time(s); last exit code 0x{code:08X}. Completed units are "
                             "cached; relaunching the same command resumes.")
        time.sleep(RETRY_PAUSE_S)
        if deadline is not None and "--hours" in cmd:  # the retried worker gets the time that is left
            cmd[cmd.index("--hours") + 1] = str(max(deadline - time.monotonic(), 0.0) / 3600)


def orchestrate(args: argparse.Namespace) -> None:
    if args.dry_run:
        from script.week5.timescale_inspection.run_phase9 import Phase9Runner
        Phase9Runner(args).dry_run()
        return
    if not (args.hours or args.smoke):
        raise SystemExit("a decoding run requires an explicit elapsed-time budget: pass --hours")
    deadline = time.monotonic() + args.hours * 3600 if args.hours else None
    session = uuid.uuid4().hex
    datasets = ("babel1",) if args.stage == "P" else DATASETS
    stages = [("gates", None)] + [(step, ds) for ds in datasets for step in ("A", "B", "C")]
    budget_stop = False
    for step, dataset in stages:
        remaining = (deadline - time.monotonic()) / 3600 if deadline is not None else None
        if remaining is not None and remaining <= 0:
            budget_stop = True
            break
        cmd = [sys.executable, "-m", MODULE, "--run-id", args.run_id, "--source-run", args.source_run,
               "--stage", args.stage, "--step", step, "--session", session]
        if dataset:
            cmd += ["--dataset", dataset]
        if remaining is not None:
            cmd += ["--hours", str(remaining)]
        if args.smoke:
            cmd.append("--smoke")
        if args.cpu:
            cmd.append("--cpu")
        print(f"[process] {dataset or 'preflight'} {step}", flush=True)
        run_with_retries(cmd, f"{dataset or 'preflight'} {step}", deadline)
        path = output_dir(args) / "execution" / f"{dataset or 'preflight'}_{step}.json"
        entry = json.loads(path.read_text())
        if entry.get("session") != session:
            raise RuntimeError(f"stage worker did not save this run's state: {path}")
        if entry["expired"]:
            budget_stop = True
            break
    cmd = [sys.executable, "-m", MODULE, "--run-id", args.run_id, "--source-run", args.source_run,
           "--stage", args.stage, "--step", "finalize", "--session", session]
    if args.smoke:
        cmd.append("--smoke")
    if args.cpu:
        cmd.append("--cpu")
    if budget_stop:
        cmd.append("--budget-stop")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    if args.step:
        if args.step in ("A", "B", "C") and args.dataset not in (("babel1",) if args.stage == "P" else DATASETS):
            raise SystemExit("--dataset must name a dataset in this stage")
        run_step(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
