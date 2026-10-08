"""Phase 0 of the time-scale inspection: label-only ceilings and diagnostics.

Reads ground truth and the saved artifacts of a raw K-means cross-dataset run
(`rkx_v2` by default): preparation, checkpoints, embeddings, predictions. Never
writes into that run. No OT decoding and no training happen here.

Parts (all by default):
  grid      0.1 token-grid ceilings
  smooth    0.2 smoothing ceilings
  profile   0.3 failure profile of the saved predictions
  probes    0.4 per-token action probes on four representations
  masked    0.4 masked-prediction loss split by boundary crossing

Usage:
  python -m script.week5.timescale_inspection.run_phase0 --run-id ts0_v1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from script.fsq_reconstruction.experiment import write_json_atomic  # noqa: E402
from script.quantized_input_comparison.artifacts import Loader  # noqa: E402
from script.week5.contextualizer_calibration import evaluate as E  # noqa: E402
from script.week5.contextualizer_calibration import plan as P  # noqa: E402
from script.week5.contextualizer_calibration import train as T  # noqa: E402
from script.week5.raw_kmeans_cross_dataset import datasets as D  # noqa: E402
from script.week5.raw_kmeans_cross_dataset import run_cross_dataset as X  # noqa: E402
from script.week5.timescale_inspection import oracle as O  # noqa: E402
from script.week5.timescale_inspection import probes as PR  # noqa: E402
from script.week5.timescale_inspection import profile as PF  # noqa: E402
from script.week5.timescale_inspection import report as REP  # noqa: E402

PARTS = ("grid", "smooth", "profile", "probes", "masked")
GRID_WINDOWS = {"hugadb": (15, 60), "lara": (12, 50), "babel1": (4, 8, 15, 30), "babel2": (4, 8, 15, 30),
                "babel3": (4, 8, 15, 30)}
SMOOTH_RULES = ([("fixed", r) for r in (1, 2, 3, 4)] +
                [(rule, f) for rule in ("relative", "relative_floor1") for f in (0.02, 0.04, 0.08)])
SOURCE_FILES = [f"script/week5/timescale_inspection/{m}.py" for m in ("oracle", "profile", "probes", "run_phase0", "report")]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def source_hash() -> str:
    h = hashlib.sha256()
    for name in SOURCE_FILES:
        h.update(name.encode())
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()[:16]


def all_gts(spec: D.DatasetSpec) -> dict[str, np.ndarray]:
    return D.load_gts(spec, D.recording_names(spec))


def class_names(spec: D.DatasetSpec) -> dict[int, str]:
    return {v: k for k, v in D.label_map(spec).items()}


def done_records(rkx: Path, datasets: list[str]) -> list[dict]:
    """Verified-done evaluation records of the source run (a snapshot: the run may still be going)."""
    out = []
    for path in sorted((rkx / "eval_records").glob("*.json")):
        rec = json.loads(path.read_text())
        if rec.get("status") != "done":
            continue
        dataset, normalization, seed = rec["cell_id"].split("__")
        if dataset not in datasets:
            continue
        eval_dir = rkx / "evals" / rec["identity"]
        if not ((eval_dir / "predictions.npz").exists() and (eval_dir / "info.json").exists()):
            continue
        out.append(dict(rec, dataset=dataset, normalization=normalization, seed=int(seed.removeprefix("seed")),
                        eval_dir=eval_dir))
    return out


def part_grid(datasets, out: Path) -> None:
    rows = []
    for ds in datasets:
        spec, gts = D.SPECS[ds], all_gts(D.SPECS[ds])
        for w in GRID_WINDOWS[ds]:
            rows.append(dict(dataset=ds, window_frames=w, token_seconds=round(w / spec.fps, 4),
                             current=w == spec.window, **O.grid_ceiling(gts, w)))
            print(f"[grid] {ds} w={w}: F1@50 {rows[-1]['F1@50']}", flush=True)
    write_csv(out / "grid_ceilings.csv", rows)


def part_smooth(datasets, out: Path) -> None:
    rows = []
    for ds in datasets:
        spec, gts = D.SPECS[ds], all_gts(D.SPECS[ds])
        current = max(1, spec.fps // spec.window)
        base = O.grid_ceiling(gts, spec.window)
        rows.append(dict(dataset=ds, rule="none", value=0, radius_seconds="", **base))
        for rule, value in SMOOTH_RULES:
            r = O.smoothing_ceiling(gts, spec.window, rule, value)
            secs = round(value * spec.window / spec.fps, 3) if rule == "fixed" else ""
            rows.append(dict(dataset=ds, rule=rule, value=value, radius_seconds=secs,
                             current=rule == "fixed" and value == current, **r))
            print(f"[smooth] {ds} {rule} {value}: F1@50 {r['F1@50']}", flush=True)
    write_csv(out / "smoothing_ceilings.csv", rows)


def part_profile(records, out: Path) -> None:
    tables = {k: [] for k in ("summary", "per_class", "error_types", "tertiles")}
    gts_cache: dict[str, dict] = {}
    for rec in records:
        spec = D.SPECS[rec["dataset"]]
        gts_cache.setdefault(spec.name, all_gts(spec))
        info = json.loads((rec["eval_dir"] / "info.json").read_text())
        names = sorted(info["scored_recording_names"])
        with np.load(rec["eval_dir"] / "predictions.npz") as z:
            preds = {n: z[f"{n}__frame_pred"] for n in names}
        res = PF.profile(names, gts_cache[spec.name], preds, spec.fps, class_names(spec))
        ladder = json.loads((rec["eval_dir"] / "ladder.json").read_text())["rows"]
        saved_f1 = float(max(ladder, key=lambda r: r["rung_steps"])["F1_50"])
        key = dict(dataset=rec["dataset"], normalization=rec["normalization"], seed=rec["seed"], role=rec["role"])
        tables["summary"].append(dict(key, saved_last_rung_F1_50=saved_f1, **res["summary"]))
        for t in ("per_class", "error_types", "tertiles"):
            tables[t] += [dict(key, **r) for r in res[t]]
        print(f"[profile] {rec['cell_id']} {rec['role']}: F1@50 {res['summary']['F1@50']}", flush=True)
    for t, rows in tables.items():
        write_csv(out / f"profile_{t}.csv", rows)


def selected_state(rkx: Path, cell_id: str) -> tuple[dict, dict]:
    jd = rkx / "jobs" / cell_id
    info = json.loads((jd / "training_info.json").read_text())
    step = info["roles"]["selected"]
    row = next(r for r in json.loads((jd / "checkpoints.json").read_text())["ledger"] if r["step"] == step)
    state = T.load_state(jd / "ckpt" / row["file"])["state_dict"]
    if T.tensor_hash(state) != row["tensor_sha256"]:
        raise RuntimeError(f"{cell_id}: checkpoint hash mismatch")
    return state, info


def part_probes_and_masked(records, rkx: Path, out: Path, parts: set[str], device) -> None:
    probe_rows, masked_rows = [], []
    gts_cache: dict[str, dict] = {}
    for rec in [r for r in records if r["role"] == "selected"]:
        spec = D.SPECS[rec["dataset"]]
        gts_cache.setdefault(spec.name, all_gts(spec))
        gts = gts_cache[spec.name]
        loader = Loader(rkx / "prep" / rec["cell_id"])
        key = dict(dataset=rec["dataset"], normalization=rec["normalization"], seed=rec["seed"])
        if "probes" in parts:
            with np.load(rec["eval_dir"] / "embeddings.npz") as z:
                emb = {n: z[f"{n}__emb"] for n in gts if f"{n}__emb" in z}
            table = PR.token_table(spec, loader, {n: gts[n] for n in emb}, emb)
            for r in PR.run_probes(table):
                probe_rows.append(dict(key, **r))
            print(f"[probes] {rec['cell_id']}: " + ", ".join(
                f"{r['representation']}/{r['probe']} {r['balanced_accuracy']:.3f}" for r in probe_rows[-8:]), flush=True)
        if "masked" in parts:
            state, info = selected_state(rkx, rec["cell_id"])
            job = X.make_job(X.Cell(rec["dataset"], rec["normalization"], rec["seed"]))
            model = E.build_model_from_state(job, P.FULL, loader, state, device)
            val_names = T.bounded_split(loader, None)[2]
            res = PR.masked_loss_split(model, loader, val_names, gts, context=job.context, stride=job.stride,
                                       fraction=job.mask_fraction, span=job.mask_span, mask_base=T.VALIDATION_MASK_BASE,
                                       device=device)
            masked_rows.append(dict(key, selected_step=info["roles"]["selected"], saved_best_val_loss=info["best_val_loss"], **res))
            print(f"[masked] {rec['cell_id']}: within {res['within_loss']:.3f} crossing {res['crossing_loss']:.3f} "
                  f"(share crossing {res['share_targets_crossing']:.2f}; all {res['val_loss_all']:.4f} vs saved "
                  f"{info['best_val_loss']:.4f})", flush=True)
    if "probes" in parts:
        write_csv(out / "probes.csv", probe_rows)
    if "masked" in parts:
        write_csv(out / "masked_loss_split.csv", masked_rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Time-scale inspection, phase 0")
    ap.add_argument("--run-id", default="ts0_v1")
    ap.add_argument("--source-run", default="rkx_v2", help="raw K-means cross-dataset run to read")
    ap.add_argument("--datasets", nargs="+", default=list(D.DATASET_ORDER))
    ap.add_argument("--parts", nargs="+", default=list(PARTS), choices=PARTS)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)
    rkx = ROOT / "results" / "week5" / "raw_kmeans_cross_dataset" / args.source_run
    out = ROOT / "results" / "week5" / "timescale_inspection" / args.run_id / "phase0"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    started = time.perf_counter()
    records = done_records(rkx, args.datasets)
    parts = set(args.parts)
    write_json_atomic(out / "manifest.json", dict(
        run_id=args.run_id, phase=0, status="running", source_run=args.source_run, parts=sorted(parts),
        datasets=args.datasets, implementation_hash=source_hash(), device=str(device),
        started_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"), python=platform.python_version(),
        source_records=[dict(cell_id=r["cell_id"], role=r["role"], identity=r["identity"]) for r in records]))
    if "grid" in parts:
        part_grid(args.datasets, out)
    if "smooth" in parts:
        part_smooth(args.datasets, out)
    if "profile" in parts:
        part_profile(records, out)
    if parts & {"probes", "masked"}:
        part_probes_and_masked(records, rkx, out, parts, device)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest.update(status="finished", seconds=time.perf_counter() - started)
    write_json_atomic(out / "manifest.json", manifest)
    print(f"report: {REP.write_report(out)}")


if __name__ == "__main__":
    main()
