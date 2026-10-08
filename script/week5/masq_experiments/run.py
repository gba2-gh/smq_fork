"""Driver for the Tier 0 MASQ replication: gates, smoke, stage 1, stage 2.

Every (unit, step) runs in its own process; a crashed step (native crashes leave no traceback) is retried. Finished steps are
cached by identity (settings + code hash) and written atomically. Never launched by the agent beyond gates, smoke and dry runs.

  python -m script.week5.masq_experiments.run --run-id RID --stage {gates,smoke,1,2} [--hours H] [--dry-run] [--cpu]
  python -m script.week5.masq_experiments.run --run-id RID --unit UNIT_ID --step {train,score,diagnose}   # worker (internal)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

for _n in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_n, "8")

from script.week5.masq_experiments import config as C  # noqa: E402  (torch-free)

MODULE = "script.week5.masq_experiments.run"
MAX_ATTEMPTS = 3
RETRY_PAUSE_S = 15
NO_RETRY_CODES = (1, 2)  # Python exceptions / argparse errors are deterministic: fail at once. Native crashes are retried.
G4_TOLERANCE = 0.005
G4_DATASETS = ("hugadb", "lara")
GATE_KEYS = ("G1", "G2", "G3", "G4", "G6", "G7")  # G6/G7: variant calibration and latent smoothness (tests 9, 10)
STEPS = ("train", "score", "diagnose")
SMOKE_TIME_LIMIT_MIN = 10.0
SMOKE_TIMINGS_FILE = C.RESULTS_ROOT / "smoke_timings.json"
DATASET_FILES = {"hugadb": 364, "lara": 439, "babel1": 1960, "babel2": 255, "babel3": 183}  # recordings (results/e0 n_sequences)


# -- caching --------------------------------------------------------------------------------------------------------------

def step_done(path: Path, identity: str) -> bool:
    """A step is done when its output exists and records the current identity."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("identity") == identity
    except (OSError, ValueError):
        return False


def _epochs(unit: C.Unit, smoke: bool) -> int:
    return C.settings(unit, smoke)["epochs"]


def unit_step_done(run_id: str, unit: C.Unit, step: str, smoke: bool = False) -> bool:
    ud, md = C.unit_dir(run_id, unit), C.model_dir(run_id, unit)
    if step == "train":
        return (md / f"epoch-{_epochs(unit, smoke)}.model").exists() and step_done(md / "training_info.json", C.unit_identity(unit, smoke))
    if step == "score":
        return (ud / "predictions.npz").exists() and step_done(ud / "scores.json", C.score_identity(unit, smoke))
    if step == "diagnose":
        return step_done(ud / "diagnostics.json", C.diagnostics_identity(unit, smoke))
    raise ValueError(step)


# -- worker ---------------------------------------------------------------------------------------------------------------

def run_worker(args) -> None:
    import torch
    unit, smoke = C.parse_unit(args.unit), args.smoke
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    ud, md = C.unit_dir(args.run_id, unit), C.model_dir(args.run_id, unit)
    if args.step == "train":
        if unit.arm in C.VARIANT_ARMS:
            from script.week5.masq_experiments.train_variants import train_unit_variant as train_unit
        else:
            from script.week5.masq_experiments.train_masq import train_unit
        train_unit(unit, md, device, smoke=smoke, identity=C.unit_identity(unit, smoke))
    elif args.step == "score":
        from script.week5.masq_experiments.score import score_unit
        score_unit(unit, md / f"epoch-{_epochs(unit, smoke)}.model", ud, device, identity=C.score_identity(unit, smoke), smoke=smoke)
    elif args.step == "diagnose":
        from script.week5.masq_experiments.diagnostics import diagnose_unit
        C.write_json_atomic(ud / "diagnostics.json", diagnose_unit(unit, ud / "predictions.npz", C.diagnostics_identity(unit, smoke)))


# -- gates ----------------------------------------------------------------------------------------------------------------

def gates_ok(run_id: str) -> bool:
    p = C.run_dir(run_id) / "gates.json"
    if not p.exists():
        return False
    g = json.loads(p.read_text())
    return g.get("code_hash") == C.code_hash() and all(g.get(k, {}).get("status") == "passed" for k in GATE_KEYS)


def run_gates(run_id: str, cpu: bool = False) -> dict:
    """G1-G3 (tests.py) in-process, then G4 (scorer parity with the released SMQ checkpoints)."""
    import torch
    from script.week5.masq_experiments import score, tests
    path = C.run_dir(run_id) / "gates.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    gates = dict(code_hash=C.code_hash())
    if old.get("code_hash") == gates["code_hash"] and "smoke_timings" in old:
        gates["smoke_timings"] = old["smoke_timings"]
    gates.update(tests.run_gate_tests())
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    g4, ok, t0 = {}, True, time.perf_counter()
    for ds in G4_DATASETS:
        saved = json.loads((C.ROOT / "results" / "e0" / f"{ds}_pretrained.json").read_text())
        got = score.released_checkpoint_scores(ds, device)
        diff = {key: {m: abs(float(got[key][m]) - float(saved["global" if key == "global_" else "local"][m])) for m in C.METRICS}
                for key in ("global_", "local")}
        worst = max(v for d in diff.values() for v in d.values())
        keys_ok = not got["missing"] and not got["unexpected"]
        passed = worst <= G4_TOLERANCE and keys_ok
        ok &= passed
        g4[ds] = dict(passed=passed, max_abs_diff=worst, diff=diff, missing=got["missing"], unexpected=got["unexpected"])
    gates["G4"] = dict(status="passed" if ok else "failed", seconds=round(time.perf_counter() - t0, 1), datasets=g4)
    C.write_json_atomic(path, gates)
    return gates


def ensure_gates(run_id: str, cpu: bool) -> None:
    if gates_ok(run_id):
        print("[gates] passed under the current code hash (cached)", flush=True)
        return
    print("[gates] running G1-G4 ...", flush=True)
    gates = run_gates(run_id, cpu)
    bad = {k: v.get("error") or v.get("status") for k, v in gates.items() if isinstance(v, dict) and v.get("status") == "failed"}
    if bad:
        raise SystemExit(f"gates failed, refusing to train: {bad}")
    print("[gates] G1-G4 passed", flush=True)


# -- manifest -------------------------------------------------------------------------------------------------------------

def check_run_id(run_id: str) -> None:
    """Refuse a run ID whose models were trained by different training code. Called before anything is written.
    A scoring-code change is allowed: the affected scores are recomputed automatically (their identity changes)."""
    path = C.run_dir(run_id) / "manifest.json"
    if path.exists():
        m = json.loads(path.read_text())
        if m.get("train_hash") != C.train_hash():
            raise SystemExit(f"{path.parent} was trained by different training code; choose a new --run-id.")
        if m.get("variant_hash") not in (None, C.variant_hash()):
            raise SystemExit(f"{path.parent} was trained by different variant code; choose a new --run-id.")


def init_manifest(run_id: str, stage: str, smoke: bool) -> dict:
    check_run_id(run_id)
    path = C.run_dir(run_id) / "manifest.json"
    if path.exists():
        m = json.loads(path.read_text())
        if m.get("score_hash") != C.score_hash():
            m.setdefault("score_hash_history", []).append(m.get("score_hash"))
            m["score_hash"] = C.score_hash()
        m["code_hash"] = C.code_hash()
        m.setdefault("variant_hash", C.variant_hash())
        m["stages"] = sorted(set(m.get("stages", [])) | {stage})
        m["status"] = "running"
    else:
        import torch
        m = dict(run_id=run_id, stages=[stage], code_hash=C.code_hash(), train_hash=C.train_hash(), score_hash=C.score_hash(),
                 variant_hash=C.variant_hash(), analysis_hash_at_launch=C.analysis_hash(), smoke=smoke,
                 python=platform.python_version(), torch=torch.__version__,
                 gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                 started_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"), not_run={}, status="running")
    C.write_json_atomic(path, m)
    return m


def save_manifest(run_id: str, m: dict) -> None:
    C.write_json_atomic(C.run_dir(run_id) / "manifest.json", m)


# -- orchestration --------------------------------------------------------------------------------------------------------

def run_child(cmd: list[str], label: str) -> bool:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        proc = subprocess.run(cmd, cwd=C.ROOT)
        if proc.returncode == 0:
            return True
        print(f"[run] {label}: exit code {proc.returncode} (0x{proc.returncode & 0xFFFFFFFF:08X}), attempt {attempt}/{MAX_ATTEMPTS}", flush=True)
        if proc.returncode in NO_RETRY_CODES:
            print(f"[run] {label}: Python error (traceback above); not retried", flush=True)
            return False
        if attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_PAUSE_S)
    return False


def worker_cmd(run_id: str, unit: C.Unit, step: str, smoke: bool, cpu: bool) -> list[str]:
    cmd = [sys.executable, "-m", MODULE, "--run-id", run_id, "--unit", unit.unit_id, "--step", step]
    return cmd + (["--smoke"] if smoke else []) + (["--cpu"] if cpu else [])


def unit_minutes_by_dataset(timings_from: str) -> tuple[dict, dict | None]:
    """Minutes per unit (train + score + diagnose): the table measured in masq_v1 (config.UNIT_MINUTES). The smoke run is no
    longer used for projections: its 3-batch timings underestimated BABEL-2 by 5x. `timings_from` is kept for the CLI."""
    return dict(C.UNIT_MINUTES), None


def dry_run(run_id: str, stage: str, timings_from: str) -> None:
    units = C.smoke_units() if stage == "smoke" else C.stage_units(stage)
    minutes, info = unit_minutes_by_dataset(timings_from)
    print(f"[dry-run] run {run_id}, stage {stage}: {len(units)} units; code hash {C.code_hash()}")
    print(f"[dry-run] per-unit minutes: {({k: round(v, 1) for k, v in minutes.items()})}; smoke-based projection for BABEL-2/3: {info or 'unavailable (static estimates)'}")
    smoke, total, pending = stage == "smoke", 0.0, 0
    for u in units:
        done = {s: unit_step_done(run_id, u, s, smoke) for s in STEPS}
        est = 0.0 if all(done.values()) else minutes[u.dataset]
        total += est
        pending += 0 if all(done.values()) else 1
        print(f"  {u.unit_id:34s} " + " ".join(f"{s}:{'done' if d else 'todo'}" for s, d in done.items()) + f"  ~{est:.1f} min")
    print(f"[dry-run] gates: {'passed (cached)' if gates_ok(run_id) else 'will run (~3 min)'}")
    print(f"[dry-run] {pending} units pending; projected total ~{total / 60:.2f} h (all-static estimate {sum(C.UNIT_MINUTES[u.dataset] for u in units) / 60:.2f} h)")
    by_ds = {}
    for u in units:
        by_ds[u.dataset] = by_ds.get(u.dataset, 0) + 1
    print("[dry-run] units per dataset: " + ", ".join(f"{k}:{v}" for k, v in by_ds.items()))


def finalize(run_id: str, m: dict, budget_skipped: bool, smoke: bool) -> None:
    from script.week5.masq_experiments import report
    failed = [k for k, v in m.get("not_run", {}).items() if not str(v).startswith("budget")]
    m["status"] = "failed" if failed else ("budget_stop" if budget_skipped else "finished")
    m["finished_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save_manifest(run_id, m)
    print(f"[run] report: {report.write_report(C.run_dir(run_id))}", flush=True)
    if "hu" in m.get("stages", []) or any(u.startswith(("V1", "V2")) for u in
                                          (p.name for p in (C.run_dir(run_id) / "units").glob("*"))):
        from script.week5.masq_experiments import report_hu
        print(f"[run] focused report: {report_hu.write_report_hu(C.run_dir(run_id))}", flush=True)


def run_units(run_id: str, units: list[C.Unit], smoke: bool, hours: float | None, cpu: bool, stage: str, timings_from: str) -> None:
    m = init_manifest(run_id, stage, smoke)
    minutes, _ = unit_minutes_by_dataset(timings_from)
    deadline = time.monotonic() + hours * 3600 if hours else None
    budget_skipped = False
    for i, u in enumerate(units, 1):
        if deadline is not None and time.monotonic() + minutes[u.dataset] * 60 > deadline:
            if not all(unit_step_done(run_id, u, s, smoke) for s in STEPS):
                m["not_run"][u.unit_id] = "budget: projected to cross the deadline"
                budget_skipped = True
            continue
        for step in STEPS:
            if unit_step_done(run_id, u, step, smoke):
                continue
            print(f"[run {i}/{len(units)}] {u.unit_id} {step}", flush=True)
            if not run_child(worker_cmd(run_id, u, step, smoke, cpu), f"{u.unit_id} {step}"):
                m["not_run"][u.unit_id] = f"{step} failed {MAX_ATTEMPTS} times"
                save_manifest(run_id, m)
                break
            m["not_run"].pop(u.unit_id, None)
        save_manifest(run_id, m)
    finalize(run_id, m, budget_skipped, smoke)


def run_smoke(run_id: str, cpu: bool) -> None:
    t0 = time.perf_counter()
    check_run_id(run_id)
    ensure_gates(run_id, cpu)
    units = C.smoke_units()
    m = init_manifest(run_id, "smoke", True)
    timings = {}
    for u in units:
        for step in STEPS:
            if unit_step_done(run_id, u, step, True):
                continue
            ts = time.perf_counter()
            print(f"[smoke] {u.unit_id} {step}", flush=True)
            if not run_child(worker_cmd(run_id, u, step, True, cpu), f"{u.unit_id} {step}"):
                m["not_run"][u.unit_id] = f"{step} failed {MAX_ATTEMPTS} times"
                break
            timings[f"{u.unit_id}:{step}"] = round(time.perf_counter() - ts, 1)
    s0 = C.Unit("S0", "babel3", 111)
    if m["not_run"]:
        save_manifest(run_id, m)
        raise SystemExit(f"smoke failed: {m['not_run']}")
    if timings:  # only when steps actually ran: a fully cached rerun has nothing to measure and must not overwrite good timings
        info = json.loads((C.model_dir(run_id, s0) / "training_info.json").read_text())
        sc = json.loads((C.unit_dir(run_id, s0) / "scores.json").read_text())
        path = C.run_dir(run_id) / "gates.json"
        g = json.loads(path.read_text())
        g["smoke_timings"] = dict(step_seconds=timings, seconds_per_batch=info["seconds_per_batch"],
                                  batches_per_epoch_full=math.ceil(DATASET_FILES["babel3"] / 32),
                                  score_seconds_per_sequence=sc["seconds_per_sequence"],
                                  diagnose_seconds=timings.get(f"{s0.unit_id}:diagnose", 60.0),
                                  total_seconds=round(time.perf_counter() - t0, 1), peak_gpu_bytes=info.get("peak_gpu_bytes"))
        C.write_json_atomic(path, g)
        C.write_json_atomic(SMOKE_TIMINGS_FILE, g["smoke_timings"])
    total_min = (time.perf_counter() - t0) / 60
    print(f"[smoke] total {total_min:.1f} min (limit {SMOKE_TIME_LIMIT_MIN:g})", flush=True)
    finalize(run_id, m, False, True)
    if total_min > SMOKE_TIME_LIMIT_MIN:
        print(f"[smoke] WARNING: exceeded {SMOKE_TIME_LIMIT_MIN:g} minutes", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Tier 0 MASQ replication driver")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--stage", choices=("gates", "smoke", "1", "2", "hu"))
    ap.add_argument("--hours", type=float)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="(worker) use the smoke settings")
    ap.add_argument("--unit", help="(worker) unit id")
    ap.add_argument("--step", choices=STEPS, help="(worker) step")
    ap.add_argument("--timings-from", default="masq_smoke", help="run id whose smoke timings adjust the budget")
    args = ap.parse_args(argv)
    if args.unit:
        return run_worker(args)
    if not args.stage:
        raise SystemExit("--stage is required")
    if args.dry_run:
        return dry_run(args.run_id, args.stage, args.timings_from)
    if args.stage == "gates":
        check_run_id(args.run_id)
        gates = run_gates(args.run_id, args.cpu)
        print(json.dumps({k: (v.get("status") if isinstance(v, dict) else v) for k, v in gates.items()}, indent=2))
        raise SystemExit(0 if gates_ok(args.run_id) else 1)
    if args.stage == "smoke":
        return run_smoke(args.run_id, args.cpu)
    if not args.hours:
        raise SystemExit("stages 1, 2 and hu require an explicit elapsed-time budget: pass --hours")
    check_run_id(args.run_id)
    ensure_gates(args.run_id, args.cpu)
    run_units(args.run_id, C.stage_units(args.stage), False, args.hours, args.cpu, args.stage, args.timings_from)


if __name__ == "__main__":
    main()
