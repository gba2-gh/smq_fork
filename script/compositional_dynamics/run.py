"""Milestone 1 driver: preflight, smoke, dry run, run (cells in fresh child processes), parity, gate report.

Modes are mutually exclusive. Nothing here launches the full experiment unless `run` is invoked by the user."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
_CAP = str(min(8, os.cpu_count() or 8))
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, _CAP)

import numpy as np  # noqa: E402

from script.compositional_dynamics import config as C  # noqa: E402

EXIT_DONE, EXIT_INTERRUPTED, EXIT_INVALID, EXIT_MISSING_DEP = 0, 3, 4, 5


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    os.replace(tmp, path)


def save_npz_atomic(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


# --- cell identity and status ----------------------------------------------------------------------------------

def cell_dir(run_dir: Path, arm: str, seed: int, split: str = "primary") -> Path:
    return run_dir / "cells" / (f"{arm}__seed{seed}" + ("" if split == "primary" else f"__{split}"))


def cell_identity(seed: int, arm: str, scale: C.Scale, source_hash: str, b0_identity: str | None, split: str = "primary") -> str:
    payload = dict(code=C.code_hash(), scale=scale.__dict__, seed=seed, arm=arm, source=source_hash, b0=b0_identity, split=split,
                   coeffs=C.COEFFS, hist=C.history_length())
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def read_info(d: Path) -> dict | None:
    p = d / "info.json"
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def completed(d: Path, identity: str) -> bool:
    info = read_info(d)
    return bool(info) and info.get("status") == "complete" and info.get("identity") == identity and (d / "scores.json").exists()


# --- one cell --------------------------------------------------------------------------------------------------

def permuted_histories(pop, seed: int = C.CALIB_SEED):
    """Each target's valid preceding codes in a random order (saved-seed permutations), target and position unchanged;
    valid entries first, padding last."""
    rng = np.random.default_rng(seed)
    keys = np.where(pop.hmask, rng.random(pop.hist.shape), 2.0)
    order = np.argsort(keys, axis=1, kind="stable")
    return np.take_along_axis(pop.hist, order, axis=1), np.take_along_axis(pop.hmask, order, axis=1)


def run_cell(run_dir: Path, seed: int, arm: str, scale: C.Scale, device: str, deadline_s: float | None, reuse: bool,
             split: str = "primary") -> int:
    import torch
    from script.action_transport.budget import DeadlineExceeded
    from script.compositional_dynamics import data as Dm, decode as Dc, evaluate as Ev, train as Tr

    d = cell_dir(run_dir, arm, seed, split)
    src = Dm.source_identity(seed)["recordings_and_all"]
    b0_identity = None
    if arm != "b0_centre":
        b0 = read_info(cell_dir(run_dir, "b0_centre", seed, split))
        if not b0 or b0.get("status") != "complete":
            print(f"[cell] {arm} seed {seed}: matched b0_centre has not completed", flush=True)
            return EXIT_MISSING_DEP
        b0_identity = b0["identity"]
    identity = cell_identity(seed, arm, scale, src, b0_identity, split)
    if reuse and completed(d, identity):
        print(f"[cell] {arm} seed {seed}: reused completed cell", flush=True)
        return EXIT_DONE
    deadline = time.perf_counter() + deadline_s if deadline_s else None
    started = time.perf_counter()
    base = dict(identity=identity, arm=arm, seed=seed, split=split, scale=scale.name, smoke=bool(scale.smoke_fit), device=device,
                started=datetime.now().isoformat(timespec="seconds"), code_hash=C.code_hash())
    write_json_atomic(d / "info.json", dict(status="running", **base))
    try:
        data = Dm.load_cell_data(seed, scale, split)
        dev = torch.device(device)
        torch.manual_seed(seed)
        info = dict(base, hist_len=C.history_length(), n_fit=len(data.fit_names), n_eval=len(data.eval_names),
                    fit_names=data.fit_names, eval_names=data.eval_names)
        art, comp = {}, {}
        if arm == "b0_centre":
            fit = Tr.fit_b0(data, scale, deadline)
            R0 = fit.plans.responsibilities()
            cost = {"fit": Tr.b0_costs(data, data.fit, fit.mu), "eval": Tr.b0_costs(data, data.eval, fit.mu)}
            torch.save(fit.plans.log_t, d / "log_t_fit.pt")
            save_npz_atomic(d / "initial.npz", R0=R0.astype(np.float32), mu=fit.mu, grouping=fit.grouping)
            info.update(b0_outer_status=fit.outer_status, b0_outer_iterations=fit.outer_iterations, fit_seconds=fit.seconds)
            theta, model, occ_c, scale_s, refreshes, losses = None, None, None, None, [], []
        else:
            init = np.load(cell_dir(run_dir, "b0_centre", seed, split) / "initial.npz")
            R0 = init["R0"]
            log_t0 = torch.load(cell_dir(run_dir, "b0_centre", seed, split) / "log_t_fit.pt")
            fitted = Tr.fit_arm(data, arm, scale, R0, log_t0, dev, deadline)
            theta, model, scale_s = fitted.theta, fitted.model, fitted.scale_s
            refreshes, losses = fitted.refreshes, fitted.losses
            comp, craw = {}, {}
            for key, pop in (("fit", data.fit), ("eval", data.eval)):
                occ, cond, h = Tr.components(model, theta, pop, dev, want_repr=model is not None)
                comp[key] = (occ, cond, h)
                craw[key] = Tr.raw_cost(occ, cond)
            cost = {k: Tr.scaled_cost(v, scale_s) for k, v in craw.items()}
            info.update(scale_s=scale_s, updates=fitted.updates, fit_seconds=fitted.seconds, timings=fitted.timings,
                        calibration_ids=int(len(fitted.calib_ids)))
            if model is not None:
                from script.compositional_dynamics.model import parameter_counts
                info["parameters"] = parameter_counts(model)
                torch.save(model.state_dict(), d / "model.pt")
            np.save(d / "calibration_ids.npy", fitted.calib_ids)
            R_final = fitted.plans.responsibilities()
            art.update(R_final=R_final.astype(np.float32), theta=theta)
        # --- frozen decode of both populations
        decoded, t_dec = {}, time.perf_counter()
        for key, pop in (("fit", data.fit), ("eval", data.eval)):
            decoded[key] = Dc.decode_ladder(cost[key], pop, scale.rungs, deadline)
        info["decode_seconds"] = time.perf_counter() - t_dec
        # --- scoring and Tier A diagnostics (labels enter only here)
        gts = Ev.load_labels(data.fit_names + data.eval_names)
        pops = {"fit": data.fit, "eval": data.eval}
        final_rung = scale.rungs[-1]
        scores = dict(primary_endpoint_steps=final_rung, rungs=list(scale.rungs))
        for key, pop in pops.items():
            scores[key] = {str(r): Ev.score_states(decoded[key]["states"][r], pop, gts, data.starts, data.frame_counts)
                           for r in scale.rungs}
        solver = {k: dict(status_counts={str(r): v for r, v in decoded[k]["status_counts"].items()},
                          steps_total=int(sum(decoded[k]["steps"].values())),
                          residual_max=float(max([x for x in decoded[k]["residual"].values() if x is not None] or [0.0])))
                  for k in pops}
        if split == "primary":
            scores["multiplier_sensitivity"] = {}
            for mult in C.MULTIPLIERS:
                dm = Dc.decode_ladder(cost["eval"], data.eval, scale.rungs, deadline, multiplier=mult)
                scores["multiplier_sensitivity"][str(mult)] = Ev.score_states(dm["states"][final_rung], data.eval, gts, data.starts, data.frame_counts)["metrics"]
        diag = dict(solver=solver)
        final_states = {k: decoded[k]["states"][final_rung] for k in pops}
        if arm == "b0_centre":
            before = final_states
            diag["initial_vs_final_agreement_fit_R0"] = Ev.agreement(
                {n: R0[data.fit.slice_of(r)].argmax(1) for r, n in enumerate(data.fit.names)}, final_states["fit"], data.fit)
        else:
            b0_states = dict(np.load(cell_dir(run_dir, "b0_centre", seed, split) / "states.npz", allow_pickle=True))
            before = {k: b0_states[f"{k}__final"].item() for k in pops}
            R0_states = {n: R0[data.fit.slice_of(r)].argmax(1) for r, n in enumerate(data.fit.names)}
            diag["agreement_fit_vs_B0_R0_argmax"] = Ev.agreement(R0_states, final_states["fit"], data.fit)
            diag["agreement_fit_vs_B0_final_decode"] = Ev.agreement(before["fit"], final_states["fit"], data.fit)
            diag["agreement_eval_vs_B0_final_decode"] = Ev.agreement(before["eval"], final_states["eval"], data.eval)
            for k in pops:
                occ, cond, _ = comp[k]
                diag[f"assigned_nll_{k}"] = Ev.assigned_nll(final_states[k], pops[k], occ, cond)
        diag["association"] = {k: {"before": Ev.association(before[k], pops[k], gts, data.starts, data.frame_counts),
                                   "after": Ev.association(final_states[k], pops[k], gts, data.starts, data.frame_counts)}
                               for k in pops}
        # --- order sensitivity (ordered model only): permute each target's history, regenerate costs, decode eval
        if arm == "m1_ordered":
            ph, pm = permuted_histories(data.eval)
            occ, cond, _ = Tr.components(model, theta, data.eval, dev, hist=ph, hmask=pm)
            dec = Dc.decode_ladder(Tr.scaled_cost(Tr.raw_cost(occ, cond), scale_s), data.eval, scale.rungs, deadline)
            diag["order_sensitivity_eval"] = dict(
                metrics=Ev.score_states(dec["states"][final_rung], data.eval, gts, data.starts, data.frame_counts)["metrics"],
                agreement_with_unpermuted=Ev.agreement(final_states["eval"], dec["states"][final_rung], data.eval))
            np.save(d / "order_permuted_hist_eval.npy", ph)
        # --- artifacts
        states_out = {}
        for k in pops:
            states_out[f"{k}__final"] = np.array(final_states[k], dtype=object)
            states_out[f"{k}__first_rung"] = np.array(decoded[k]["states"][scale.rungs[0]], dtype=object)
        np.savez_compressed(d / "states.npz", **states_out)
        for k, pop in pops.items():
            fp = Ev.frame_predictions(final_states[k], pop, data.starts, data.frame_counts)
            save_npz_atomic(d / f"frame_predictions_{k}.npz", **{n: v.astype(np.int8) for n, v in fp.items()})
        if arm != "b0_centre":
            for k in pops:
                occ, cond, h = comp[k]
                extra = dict(cost_scaled=cost[k].astype(np.float16), occ_nll=occ.astype(np.float16))
                if cond is not None:
                    extra.update(cond_nll=cond.astype(np.float16), history_repr=h.astype(np.float16))
                save_npz_atomic(d / f"costs_{k}.npz", codes=pops[k].codes, lengths=pops[k].lengths, rec_idx=pops[k].rec_idx,
                                pos=pops[k].pos, **extra)
            save_npz_atomic(d / "final_plans.npz", **art)
        else:
            for k in pops:
                save_npz_atomic(d / f"costs_{k}.npz", cost_cosine=cost[k].astype(np.float32), codes=pops[k].codes,
                                lengths=pops[k].lengths, rec_idx=pops[k].rec_idx, pos=pops[k].pos)
        write_json_atomic(d / "training.json", dict(refreshes=refreshes, losses=losses[:: max(1, len(losses) // 500)]
                                                    if losses else [], n_loss_points=len(losses)))
        write_json_atomic(d / "scores.json", scores)
        write_json_atomic(d / "diagnostics.json", diag)
        info.update(status="complete", seconds=time.perf_counter() - started, finished=datetime.now().isoformat(timespec="seconds"))
        write_json_atomic(d / "info.json", info)
        print(f"[cell] {arm} seed {seed}: complete in {info['seconds']:.0f}s", flush=True)
        return EXIT_DONE
    except DeadlineExceeded:
        write_json_atomic(d / "info.json", dict(base, status="interrupted", seconds=time.perf_counter() - started))
        print(f"[cell] {arm} seed {seed}: INTERRUPTED at the deadline (not a result)", flush=True)
        return EXIT_INTERRUPTED
    except Tr.InvalidSolve as exc:
        write_json_atomic(d / "info.json", dict(base, status="invalid", reason=str(exc), seconds=time.perf_counter() - started))
        print(f"[cell] {arm} seed {seed}: INVALID ({exc})", flush=True)
        return EXIT_INVALID


# --- preflight ---------------------------------------------------------------------------------------------------

def preflight(seeds) -> dict:
    from script.compositional_dynamics import data as Dm
    from script.compositional_dynamics import train as Tr
    report = {"seeds": {}, "ok": True, "problems": []}
    for seed in seeds:
        try:
            facts = Dm.verify_source(seed)
            data = Dm.load_cell_data(seed, C.FULL)
            pre_all = Tr.b0_grouping(data, data.fit_names + data.eval_names)
            saved = data.loader.grouping(C.FAMILY)
            facts["grouping_all_recordings_vs_saved_differing_codes"] = int((pre_all != saved).sum())
            facts["historical_b0_unit"] = str(C.historical_unit(seed)) if C.historical_unit(seed).exists() else None
            if facts["historical_b0_unit"] is None:
                report["problems"].append(f"seed {seed}: historical B0 unit missing")
            report["seeds"][str(seed)] = facts
        except Exception as exc:  # noqa: BLE001 - preflight reports every failure
            report["ok"] = False
            report["problems"].append(f"seed {seed}: {type(exc).__name__}: {exc}")
    return report


# --- estimates ----------------------------------------------------------------------------------------------------

def historical_b0_seconds() -> dict:
    info = read_info(C.historical_unit(111).parent) or json.loads(C.historical_unit(111).read_text())
    ladder = sum(r["seconds"] for r in info["ladder"])
    return dict(total=info["seconds"], ladder=ladder, fit=info["seconds"] - ladder)


def estimate(run_id: str, scale: C.Scale = C.FULL) -> dict:
    """Projected seconds per cell. B0 from the saved historical unit (439 recordings); neural/occupancy from the most
    recent smoke timings when available, otherwise marked unmeasured."""
    h = historical_b0_seconds()
    n_fit, n_eval = C.N_FIT, C.N_EVAL
    est = {"b0_centre": h["fit"] * n_fit / 439 + h["ladder"] * (n_fit + n_eval) / 439, "basis_b0": "historical unit seed 111"}
    sm = None
    for p in sorted((C.RESULTS).glob("*smoke*/cells/m1_ordered__seed*/info.json")):
        i = json.loads(p.read_text())
        if i.get("status") == "complete" and i.get("timings"):
            sm = i
    if sm is None:
        est["m1_ordered"] = est["m1_bag"] = est["occupancy"] = None
        est["basis_new"] = "unmeasured: run -Smoke first"
        return est
    t, sc = sm["timings"], C.SMOKE
    per_update = t["train_s"] / max(1, t["updates"])
    comp_per_win = t["comp_s"] / (t["comp_calls"] * t["comp_windows_per_call"])
    step_s = t["refresh_s"] / max(1.0, t["refresh_steps"])        # seconds per accepted inner step per recording
    n_win = n_fit * 500                                           # LARa recordings have 500 windows
    refreshes = scale.n_blocks - 1
    neural = (scale.total_updates * per_update + scale.n_blocks * comp_per_win * n_win
              + refreshes * n_fit * scale.inner_steps * step_s + (n_fit + n_eval) * comp_per_win * 500
              + h["ladder"] * (n_fit + n_eval) / 439)
    est["m1_ordered"] = est["m1_bag"] = neural + (h["ladder"] * n_eval / 439)     # ordered adds the order decode
    est["occupancy"] = neural - scale.total_updates * per_update
    est["basis_new"] = (f"training, cost evaluation and refresh extrapolated from smoke timings (seed {sm['seed']}, 12 fitting recordings); "
                        "final decoding borrowed from the historical B0 ladder. The new arms' costs are NLL-based and may need more "
                        "final-solver steps than cosine costs, so treat the decode share as a lower bound")
    return est


# --- driver -------------------------------------------------------------------------------------------------------

def version_info() -> dict:
    import sklearn, torch
    return dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__, sklearn=sklearn.__version__,
                platform=platform.platform(), cuda=bool(torch.cuda.is_available()),
                threads={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")})


def run_driver(run_id: str, arms, seeds, device: str, hours: float | None, reuse: bool, scale: C.Scale,
               splits=("primary",)) -> int:
    run_dir = C.RESULTS / run_id
    start = time.perf_counter()
    est = estimate(run_id, scale) if scale.name == "full" else {}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("code_hash") != C.code_hash() or old.get("scale") != scale.name:
            print(f"Refusing: run id {run_id} exists with a different code hash or scale.", file=sys.stderr)
            return 2
    write_json_atomic(manifest_path, dict(run_id=run_id, code_hash=C.code_hash(), scale=scale.name, arms=list(arms),
                                          seeds=list(seeds), splits=list(splits), device=device, hours=hours, versions=version_info(),
                                          started=datetime.now().isoformat(timespec="seconds"), declared=dict(
                                              coeffs=C.COEFFS, hist_len=C.history_length(), alpha=C.ALPHA,
                                              total_updates=scale.total_updates, inner_steps=scale.inner_steps, rungs=scale.rungs)))
    not_run, worst = [], 0
    for split, seed, arm in [(sp, sd, a) for sp in splits for sd in seeds for a in C.ARMS if a in arms]:
        if True:
            remaining = None if hours is None else hours * 3600 - (time.perf_counter() - start)
            if remaining is not None and (remaining <= 0 or (est.get(arm) and est[arm] > remaining)):
                not_run.append(dict(arm=arm, seed=seed, split=split, reason=f"deadline: {remaining:.0f}s left, projected {est.get(arm)}"))
                print(f"[run] not launching {arm} seed {seed} {split}: projected over the remaining budget", flush=True)
                continue
            cmd = [sys.executable, "-m", "script.compositional_dynamics.run", "cell", "--run-id", run_id, "--seed", str(seed),
                   "--arm", arm, "--scale", scale.name, "--device", device, "--split", split]
            if remaining is not None:
                cmd += ["--deadline-s", f"{remaining:.0f}"]
            if reuse:
                cmd.append("--reuse-completed")
            code = subprocess.run(cmd, cwd=str(C.ROOT)).returncode
            if code != EXIT_DONE:
                worst = max(worst, code)
                not_run.append(dict(arm=arm, seed=seed, split=split, reason=f"child exit code {code}"))
    from script.compositional_dynamics.report import finalize
    finalize(run_dir, arms, seeds, scale, not_run)
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="script.compositional_dynamics.run")
    sub = ap.add_subparsers(dest="mode")
    for mode in ("preflight", "dry-run", "smoke", "run", "parity"):
        p = sub.add_parser(mode)
        p.add_argument("--run-id", default=None)
        p.add_argument("--arms", default=",".join(C.ARMS))
        p.add_argument("--seeds", default=",".join(map(str, C.SEEDS)))
        p.add_argument("--device", default="cuda")
        p.add_argument("--hours", type=float, default=None)
        p.add_argument("--splits", default="primary")
        p.add_argument("--reuse-completed", action="store_true")
    c = sub.add_parser("cell")
    c.add_argument("--run-id", required=True)
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--arm", required=True)
    c.add_argument("--scale", default="full")
    c.add_argument("--device", default="cpu")
    c.add_argument("--deadline-s", type=float, default=None)
    c.add_argument("--split", default="primary")
    c.add_argument("--reuse-completed", action="store_true")
    a = ap.parse_args(argv)
    if a.mode is None:
        ap.print_help()
        return 0
    if a.mode == "cell":
        return run_cell(C.RESULTS / a.run_id, a.seed, a.arm, C.SCALES[a.scale], a.device, a.deadline_s, a.reuse_completed, a.split)
    arms = [x for x in a.arms.split(",") if x]
    seeds = [int(x) for x in a.seeds.split(",") if x]
    if not set(arms) <= set(C.ARMS) or not set(seeds) <= set(C.SEEDS):
        print("unknown arm or seed", file=sys.stderr)
        return 2
    if a.mode == "preflight":
        rep = preflight(seeds)
        print(json.dumps(rep, indent=2))
        return 0 if rep["ok"] else 1
    if a.mode == "dry-run":
        est = estimate(a.run_id or "dry")
        per_seed = sum(v for k, v in est.items() if k in C.ARMS and v)
        splits = [x for x in a.splits.split(",") if x]
        factor = sum(1.0 if sp == "primary" else 0.85 for sp in splits)     # fold fitting sets are ~330 of 395 recordings
        print(json.dumps(est, indent=2))
        print(f"projected total: {per_seed * len(seeds) * factor / 3600:.2f} h for {len(seeds)} seed(s) x splits {splits}"
              + ("" if all(est.get(x) for x in C.ARMS) else " (neural arms unmeasured)")
              + "; parity adds about 0.2 h. Lower bound: the new arms' final decode may need more solver steps than B0's.")
        return 0
    if a.mode == "parity":
        from script.compositional_dynamics.parity import run_parity
        return run_parity(C.RESULTS / (a.run_id or "parity"), seeds[0])
    if not a.run_id:
        print("--run-id is required", file=sys.stderr)
        return 2
    scale = C.SMOKE if a.mode == "smoke" else C.FULL
    if a.mode == "smoke" and "smoke" not in a.run_id:
        print("smoke run ids must contain the word smoke (throwaway, separately identified)", file=sys.stderr)
        return 2
    if a.mode == "run" and "smoke" in a.run_id:
        print("a full run id must not contain the word smoke", file=sys.stderr)
        return 2
    seeds = [111] if a.mode == "smoke" else seeds
    splits = [x for x in a.splits.split(",") if x] if a.mode in ("run", "smoke") else ["primary"]
    if not set(splits) <= set(C.SPLITS):
        print("unknown split", file=sys.stderr)
        return 2
    return run_driver(a.run_id, arms, seeds, a.device, a.hours, a.reuse_completed, scale, splits)


if __name__ == "__main__":
    sys.exit(main())
