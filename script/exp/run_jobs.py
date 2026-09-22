"""
Resumable, strictly sequential runner for SMQ experiment jobs.

Jobs file: JSON lines. A job is a list of steps run in order; a step is skipped
when the file it `creates` already exists, so re-running only does what is
missing.  Steps are argument lists for this interpreter:

    {"name": "T1/hugadb/K20_s111",
     "steps": [{"cmd": ["main.py", "--action", "train", ...],
                "creates": "models/exp/T1/hugadb/K20_s111/hugadb/epoch-30.model",
                "log": "models/exp/T1/hugadb/K20_s111/hugadb/train.log"},
               {"cmd": ["script/exp/dump_run.py", ...], "creates": "...npz", "log": "..."}]}

Keeps the workstation usable: every step runs at below-normal priority with 4
CPU threads, and training/eval steps pass their own hard GPU-memory cap
(`--gpu_mem_fraction`) so an oversized job raises OOM instead of Windows paging
GPU memory into system RAM.

The job files are re-read between steps, so jobs can be appended while it runs.
A failed step marks its job failed and the runner moves on to the next job.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
CPU_THREADS = "4"
PRIORITY = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)


def load_jobs(files):
    jobs = []
    for f in files:
        if f.exists():
            jobs += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return jobs


def step_done(step):
    return Path(step["creates"]).exists()


def run_step(step):
    log_path = Path(step["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, *step["cmd"]]
    env = dict(os.environ, OMP_NUM_THREADS=CPU_THREADS, MKL_NUM_THREADS=CPU_THREADS,
               OPENBLAS_NUM_THREADS=CPU_THREADS)
    with open(log_path, "w") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                             creationflags=PRIORITY)
    return rc == 0 and step_done(step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+", type=Path)
    ap.add_argument("--status", type=Path, default=Path("models/exp/runner_status.txt"))
    args = ap.parse_args()

    failed, finished = set(), 0
    t0 = time.time()
    while True:
        jobs = load_jobs(args.jobs)
        pending = [j for j in jobs if j["name"] not in failed
                   and not all(step_done(s) for s in j["steps"])]
        if not pending:
            break
        job = pending[0]
        n_left = len(pending)
        for step in job["steps"]:
            if step_done(step):
                continue
            args.status.parent.mkdir(parents=True, exist_ok=True)
            args.status.write_text(
                f"{time.strftime('%H:%M:%S')} elapsed {(time.time() - t0) / 3600:.2f} h | "
                f"jobs finished {finished} | pending {n_left} | failed {len(failed)}\n"
                f"  running: {job['name']} :: {Path(step['cmd'][0]).name}\n"
                + "".join(f"  failed: {n}\n" for n in sorted(failed)))
            ts = time.time()
            ok = run_step(step)
            mins = (time.time() - ts) / 60
            print(f"[runner] {'ok  ' if ok else 'FAIL'} {job['name']} :: "
                  f"{Path(step['cmd'][0]).name} ({mins:.1f} min)", flush=True)
            if not ok:
                failed.add(job["name"])
                break
        else:
            finished += 1

    args.status.write_text(
        f"{time.strftime('%H:%M:%S')} ALL DONE in {(time.time() - t0) / 3600:.2f} h | "
        f"jobs finished {finished} | failed {len(failed)}\n"
        + "".join(f"  failed: {n}\n" for n in sorted(failed)))
    print(f"[runner] all done: {finished} finished, {len(failed)} failed", flush=True)


if __name__ == "__main__":
    main()
