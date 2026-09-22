"""
Drains finished dumps into analysis JSON, one at a time, at below-normal
priority (launch through script/exp/lowprio.py). CPU only.

    results/exp/dumps/<exp>/<ds>/<tag>.npz  ->  results/exp/analysis/<exp>/<ds>/<tag>.json

When there is nothing to analyse it runs the next pending k-means control fit
(T1 control), and when that is also done it idles, polling every 3 minutes. It
exits once the runner reports ALL DONE and nothing is left.
"""
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from script.exp.analyze_run import analyze

DUMPS = Path("results/exp/dumps")
OUT = Path("results/exp/analysis")
STATUS = Path("models/exp/runner_status.txt")
LOG = Path("models/exp/analysis_watcher.log")


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def rate_of(path):
    m = re.match(r"R(\d+)_", path.stem)
    return int(m.group(1)) if m else 1


def pending_dumps():
    todo = []
    for f in sorted(DUMPS.rglob("*.npz")):
        rel = f.relative_to(DUMPS).with_suffix(".json")
        if not (OUT / rel).exists():
            todo.append((f, OUT / rel))
    return todo


def t3_pending():
    """T1 K = C dumps (the published configuration) still missing T3 decoding."""
    C = {"hugadb": 10, "lara": 8, "babel1": 5}
    todo = []
    for ds, k in C.items():
        for f in sorted((DUMPS / "T1" / ds).glob(f"K{k}_s*.npz")):
            if "_thr" in f.stem:
                continue
            dst = OUT / "T3" / "T1" / ds / f"{f.stem}.json"
            if not dst.exists():
                todo.append((f, dst))
    return todo


def main():
    from script.exp import kmeans_control
    while True:
        todo = pending_dumps()
        if todo:
            src, dst = todo[0]
            t0 = time.time()
            try:
                d = np.load(src, allow_pickle=True)
                res = analyze(d, rate=rate_of(src))
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(json.dumps(res, indent=2))
                log(f"analysed {src} ({time.time() - t0:.0f}s)")
            except Exception:
                log(f"FAILED {src}\n{traceback.format_exc()}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.with_suffix(".failed").write_text(traceback.format_exc())
                dst.write_text(json.dumps({"failed": True}))
            continue

        t3 = t3_pending()
        if t3:
            src, dst = t3[0]
            t0 = time.time()
            rc = subprocess.call([sys.executable, "script/exp/t3_hsmm.py", str(src), "--out", str(dst)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if rc != 0:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(json.dumps({"failed": True}))
            log(f"T3 {src} rc={rc} ({time.time() - t0:.0f}s)")
            continue

        if kmeans_control.run_next(log):
            continue

        if STATUS.exists() and "ALL DONE" in STATUS.read_text():
            log("runner finished and nothing left to analyse -- exiting")
            return
        time.sleep(180)


if __name__ == "__main__":
    main()
