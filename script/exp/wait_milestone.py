"""
Block until a milestone is reached, then exit with a one-line reason.
Exits early on any runner failure, so problems surface immediately.

    python script/exp/wait_milestone.py T1_seed <seed>
    python script/exp/wait_milestone.py exp T1|T2|T4
    python script/exp/wait_milestone.py all
"""
import json
import sys
import time
from pathlib import Path

C = {"hugadb": 10, "lara": 8, "babel1": 5}
LOG = Path("models/exp/runner.log")
STATUS = Path("models/exp/runner_status.txt")


def targets(args):
    if args[0] == "T1_seed":
        return [Path(f"results/exp/analysis/T1/{ds}/K{C[ds] * m}_s{args[1]}.json")
                for ds in C for m in (1, 2, 4, 8, 16)]
    if args[0] == "exp":
        jobs = [json.loads(l) for l in Path(f"models/exp/jobs_{args[1]}.jsonl").read_text().splitlines() if l.strip()]
        return [Path(j["steps"][-1]["creates"].replace("results/exp/dumps", "results/exp/analysis")
                     .replace(".npz", ".json")) for j in jobs if j["name"].startswith(args[1])]
    return []


def main():
    tg = targets(sys.argv[1:])
    seen_fail = LOG.read_text().count("FAIL") if LOG.exists() else 0
    while True:
        if LOG.exists() and LOG.read_text().count("FAIL") > seen_fail:
            print("[milestone] a runner step FAILED:\n" + "\n".join(
                l for l in LOG.read_text().splitlines() if "FAIL" in l))
            return
        if STATUS.exists() and "ALL DONE" in STATUS.read_text():
            print("[milestone] runner ALL DONE")
            return
        if tg and all(t.exists() for t in tg):
            print(f"[milestone] reached: {' '.join(sys.argv[1:])} ({len(tg)} analyses)")
            return
        time.sleep(120)


if __name__ == "__main__":
    main()
