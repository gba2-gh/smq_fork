"""Dataset-aware checkpoint decoding: contextual embeddings -> pooled continuous
cosine OT -> rung ladder. Same fit/score machinery as
`contextualizer_calibration.evaluate.decode_state`, with the frame rate, window,
number of states and ground truth supplied by the dataset instead of the HuGaDB
constants in `ot_arms`. Labels are read only to score."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from script.action_transport import continuous
from script.action_transport.budget import check_deadline
from script.fsq_reconstruction.evaluate import contextual_final_ladder
from script.fsq_reconstruction.experiment import write_json_atomic
from script.quantized_input_comparison import ot_arms as ot
from script.week5.contextualizer_calibration import evaluate as E
from script.week5.contextualizer_calibration import train as T
from script.week5.contextualizer_calibration.plan import JobSpec, Scale, stable_hash
from script.week5.raw_kmeans_cross_dataset import datasets as D
from script.week5.raw_kmeans_cross_dataset.prepare import FAMILY


def eval_identity(*, state_sha: str, job: JobSpec, scale: Scale, outer_cap: int, spec: D.DatasetSpec, loader,
                  names: list[str], prep_hash: str) -> str:
    base = E.eval_identity(state_sha=state_sha, job=job, scale=scale, outer_cap=outer_cap, qic_hash=prep_hash,
                           loader=loader, names=names)
    return stable_hash([base, spec.name, spec.fps, spec.window, spec.num_states])


def segment_stats(preds: dict[str, np.ndarray], gts: dict[str, np.ndarray], num_states: int) -> dict:
    """Per-recording predicted state usage and segment counts versus the ground truth (descriptive)."""
    def segments(a):
        return int((np.diff(a) != 0).sum()) + 1

    n_pred = [segments(p) for p in preds.values()]
    n_gt = [segments(gts[n]) for n in preds]
    used = [len(np.unique(p)) for p in preds.values()]
    gt_used = [len(np.unique(gts[n])) for n in preds]
    return dict(mean_pred_segments=float(np.mean(n_pred)), mean_gt_segments=float(np.mean(n_gt)),
                mean_pred_states_used=float(np.mean(used)), mean_gt_classes_present=float(np.mean(gt_used)),
                num_states=num_states)


def decode_state(job: JobSpec, scale: Scale, loader, spec: D.DatasetSpec, state: dict, outer_cap: int, device,
                 deadline: float | None, out_dir: Path, meta: dict) -> dict:
    started = time.perf_counter()
    names = T.bounded_split(loader, scale.recording_cap)[0]
    lookup, rms = loader.lookup(FAMILY), float(loader.rms(FAMILY))
    model = E.build_model_from_state(job, scale, loader, state, device)
    emb, zero, codes, lens, starts = {}, {}, {}, {}, {}
    t_extract = time.perf_counter()
    for n in names:
        check_deadline(deadline)
        emb[n], zero[n] = E.extract_embeddings(job.arm, model, loader, FAMILY, n, lookup, rms, device,
                                               job.context, job.stride)
        rec = loader.recording(FAMILY, n)
        codes[n], lens[n], starts[n] = rec["codes"], rec["lengths"], rec["starts"]
    extract_seconds = time.perf_counter() - t_extract
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "embeddings.npz", **{f"{n}__emb": emb[n] for n in names},
                        **{f"{n}__zero": zero[n] for n in names})
    recs = [continuous.ContinuousRecordingInput(n, codes[n].astype(np.int32), lens[n].astype(np.int32), spec.fps,
                                                spec.window, emb[n].astype(np.float64), zero[n]) for n in sorted(names)]
    mu0, _ = continuous.build_mu0(loader.grouping(FAMILY), recs, spec.num_states)
    fit = continuous.fit_continuous(recs, mu0, spec.num_states, ot.COEFFS, ot.FIT_INNER_STEPS, outer_cap,
                                    ot.FIT_OUTER_PATIENCE, ot.FIT_OUTER_RELTOL, ot.DEVICE, deadline)
    np.savez(out_dir / "ot.npz", mu0=mu0, mu=fit.params, objective_trace=np.asarray(fit.objective_trace),
             outer_status=fit.outer_status, outer_iterations=fit.outer_iterations)
    if fit.outer_status == "invalid":
        raise E.EvalInvalid(f"OT fit invalid (iterations={fit.outer_iterations})")
    gts = D.load_gts(spec, names)
    frame_counts = {n: loader.frame_count(n) for n in names}
    predictions: dict = {}
    ladder = contextual_final_ladder(fit.params, recs, ot.COEFFS, scale.rungs, gts, starts, frame_counts, deadline,
                                     predictions_out=predictions, fail_on_invalid=True)
    window_states = predictions.pop("__window_states__", {})
    np.savez_compressed(out_dir / "predictions.npz", **{f"{n}__frame_pred": predictions[n] for n in predictions},
                        **{f"{n}__window_states": window_states[n] for n in window_states})
    info = dict(status="done", outer_cap=outer_cap, ot_status=fit.outer_status, ot_outer_iterations=fit.outer_iterations,
                n_scored_recordings=len(names), scored_recording_names=names, device=str(device),
                extract_seconds=extract_seconds, seconds=time.perf_counter() - started,
                segment_stats=segment_stats(predictions, gts, spec.num_states), **meta)
    write_json_atomic(out_dir / "info.json", info)
    write_json_atomic(out_dir / "ladder.json", dict(rows=[asdict(r) for r in ladder]))
    return info


def rescore(out_dir: Path, spec: D.DatasetSpec) -> dict:
    """Recompute pooled metrics from saved final-rung frame predictions."""
    import json
    from script.action_transport import evaluate as at_evaluate
    names = sorted(json.loads((out_dir / "info.json").read_text())["scored_recording_names"])
    gts = D.load_gts(spec, names)
    with np.load(out_dir / "predictions.npz") as pred:
        return at_evaluate.score_pooled([gts[n] for n in names], [pred[f"{n}__frame_pred"] for n in names])[0]
