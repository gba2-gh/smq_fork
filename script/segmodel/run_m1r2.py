"""Execute the corrected M1 rerun in the brief's fixed priority order."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
import uuid
from pathlib import Path

import numpy as np
import sklearn
from sklearn.cluster import KMeans

from script.exp2round.q1q2.core import (
    CHECKPOINTS, D7_REFERENCE, load_dataset, majority_code_predictions,
    map_predictions, recording_zscore, score, sha256_file, prepare_windows,
)
from script.segmodel.m1r2_core import *


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) or ["cell_id", "reason"]
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temp.replace(path)


def read_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf8"))) if path.exists() else []


def upsert(path: Path, row: dict, key: str) -> None:
    rows = [old for old in read_rows(path) if old.get(key) != str(row[key])]
    rows.append(row); write_rows(path, rows)


def metric_dict(gts, raw):
    mapped = map_predictions(gts, raw, "hungarian")
    return mapped, score(gts, mapped)


def baseline_duration(codes, fps):
    return float(np.concatenate([run_durations([x], fps) for x in codes]).mean())


def fitted_artifacts(prepared, model, temperature, posteriors, assignments, path):
    atomic_npz(path,
        scaler_mean=prepared.scaler.mean_, scaler_scale=prepared.scaler.scale_,
        scaler_var=prepared.scaler.var_, pca_components=prepared.pca.components_,
        pca_mean=prepared.pca.mean_, pca_explained_variance=prepared.pca.explained_variance_,
        pca_explained_variance_ratio=prepared.pca.explained_variance_ratio_,
        prototypes=model.cluster_centers_, temperature=np.asarray(temperature),
        fit_ids=prepared.fit_ids, posteriors=np.asarray(posteriors, dtype=object),
        assignments=np.asarray(assignments, dtype=object),
        transformed=np.asarray(prepared.transformed, dtype=object))


def fit_features(data, fit_indices, standardize, k, seed, multiplier, artifact):
    arrays = recording_zscore(data.arrays) if standardize else data.arrays
    prepared = prepare_windows(arrays, WINDOW[data.name], fit_indices, seed=111)
    km = KMeans(n_clusters=k, init="k-means++", n_init=5, max_iter=300,
                tol=1e-4, algorithm="lloyd", random_state=seed).fit(prepared.fit_features)
    temperature = base_temperature(prepared.fit_features, km.cluster_centers_) * multiplier
    posterior = soft_posteriors(prepared.transformed, km.cluster_centers_, temperature)
    assignments = [q.argmax(1).astype(np.int32) for q in posterior]
    fitted_artifacts(prepared, km, temperature, posterior, assignments, artifact)
    return posterior, assignments


def lambda_rows(cell_id, run_id, ds, curve, gts, frame_counts, fps, reference):
    rows = []
    for point in curve:
        raw = expand(point.edges, point.labels, frame_counts)
        _, metrics = metric_dict(gts, raw)
        rows.append(dict(cell_id=cell_id, run_id=run_id, dataset=ds,
                         lambda_value=point.penalty, selected=False,
                         dp_blocks=point.blocks, action_runs=point.runs,
                         mean_run_duration_seconds=point.mean_duration,
                         reference_run_duration_seconds=reference,
                         objective=point.objective, **metrics))
    return rows


def run_partition(data, fit_indices, eval_indices, posterior, mode, seed, deadline,
                  cell_id, run_id, curve_path, model_path, pred_dir):
    fps, width = FPS[data.name], WINDOW[data.name]
    values = representations(posterior, mode)
    lengths = [frame_lengths(len(x), width) for x in data.arrays]
    fit_values = [values[i] for i in fit_indices]; fit_lengths = [lengths[i] for i in fit_indices]
    initial = one_second_chunks(fit_values, fit_lengths, fps, data.num_actions, seed)
    maximum = max(1, int(np.floor(MAX_SECONDS * fps / width)))
    curve = lambda_curve(fit_values, fit_lengths, initial, maximum, fps, deadline)
    reference = baseline_duration([data.codes[i] for i in fit_indices], fps)
    selected = min(curve, key=lambda point: (abs(point.mean_duration - reference), point.penalty))
    fit_gts = [data.gts[i] for i in fit_indices]
    rows = lambda_rows(cell_id, run_id, data.name, curve, fit_gts, fit_lengths, fps, reference)
    for row in rows:
        row["selected"] = bool(np.isclose(float(row["lambda_value"]), selected.penalty))
    existing = [r for r in read_rows(curve_path) if r.get("cell_id") != cell_id]
    write_rows(curve_path, existing + rows)
    centers, fit_edges, fit_labels, trace, converged, objective = alternate(
        fit_values, fit_lengths, initial, selected.penalty, maximum, deadline)
    if list(eval_indices) == list(fit_indices):
        edges, labels = fit_edges, fit_labels
    else:
        edges, labels, _ = decode_all([values[i] for i in eval_indices],
                                      [lengths[i] for i in eval_indices], centers,
                                      selected.penalty, maximum, deadline)
    eval_lengths = [lengths[i] for i in eval_indices]
    raw = expand(edges, labels, eval_lengths)
    gts = [data.gts[i] for i in eval_indices]
    mapped, metrics = metric_dict(gts, raw)
    for index, prediction, raw_prediction, edge, count in zip(eval_indices, mapped, raw, edges, eval_lengths):
        boundaries = np.r_[0, np.cumsum(count)][edge]
        atomic_npz(pred_dir / f"{data.names[i_to_int(index)]}.npz",
                   labels=prediction, raw_states=raw_prediction, boundaries=boundaries)
    reloaded = [np.load(pred_dir / f"{data.names[i_to_int(index)]}.npz")["labels"]
                for index in eval_indices]
    if score(gts, reloaded) != metrics:
        raise RuntimeError("saved predictions do not reproduce reported scores")
    durations = run_durations(raw, fps)
    gt_runs = sum(len(run_durations([gt], fps)) for gt in gts)
    blocks = sum(len(z) for z in labels); runs = len(durations)
    occupancy = np.zeros(data.num_actions, dtype=int)
    cap_blocks = 0
    for e, z in zip(edges, labels):
        occupancy += np.bincount(z, minlength=data.num_actions)
        cap_blocks += sum((b-a) == maximum for a, b in zip(e[:-1], e[1:]))
    atomic_npz(model_path, centroids=centers, initial_centroids=initial,
               lambda_value=np.asarray(selected.penalty), lambda_curve=np.asarray(
                   [[p.penalty, p.blocks, p.runs, p.mean_duration, p.objective] for p in curve]),
               trace=json_array(trace), final_objective=np.asarray(objective))
    return dict(**metrics, lambda_value=selected.penalty,
                lambda_reference_duration=reference,
                lambda_start_action_runs=selected.runs,
                lambda_final_action_runs=action_runs(labels),
                dp_blocks=blocks, action_runs=runs, gt_action_runs=gt_runs,
                dp_block_gt_ratio=blocks/gt_runs, action_run_gt_ratio=runs/gt_runs,
                mean_duration_seconds=float(durations.mean()),
                median_duration_seconds=float(np.median(durations)),
                duration_cap_blocks=int(cap_blocks),
                duration_cap_fraction=cap_blocks / max(1, blocks),
                state_occupancy=json.dumps(occupancy.tolist()),
                state_death=bool(np.any(occupancy == 0)), rounds=len(trace),
                converged=converged, cap_reached=not converged and len(trace) == 60)


def i_to_int(value):
    return int(value)


def configuration_id(ds, protocol, std, mode, k, multiplier, seed, control, fold=None):
    fields = [ds, protocol, f"std{int(std)}", mode, f"k{k}", f"t{multiplier:g}",
              f"s{seed}", control]
    if fold is not None: fields.append(f"f{fold}")
    return "_".join(map(str, fields))


def run_cell(data, protocol, standardize, mode, k, multiplier, seed, control,
             fit_indices, eval_indices, fold, run_id, deadline):
    cell_id = configuration_id(data.name, protocol, standardize, mode, k,
                               multiplier, seed, control, fold)
    started = time.monotonic()
    artifact = OUT / "models" / cell_id / "features.npz"
    source_id = configuration_id(data.name, protocol, standardize, "soft", 500,
                                 1., seed, "observed", fold)
    source = OUT / "models" / source_id / "features.npz"
    reusable = k == 500 and source.exists() and (control == "permutation" or
               mode == "hard" or multiplier == 2.)
    if reusable:
        saved = np.load(source, allow_pickle=True)
        assignments = [np.asarray(x, dtype=np.int32) for x in saved["assignments"]]
        if multiplier == 2.:
            transformed = [np.asarray(x, dtype=np.float32) for x in saved["transformed"]]
            posterior = soft_posteriors(transformed, saved["prototypes"],
                                         float(saved["temperature"]) * 2.)
        else:
            posterior = [np.asarray(x, dtype=np.float32) for x in saved["posteriors"]]
    else:
        posterior, assignments = fit_features(data, fit_indices, standardize, k, seed,
                                               multiplier, artifact)
    if control == "permutation":
        posterior = permute_rows(posterior, seed + 10_000_000)
        assignments = [q.argmax(1).astype(np.int32) for q in posterior]
    if reusable:
        atomic_npz(artifact, feature_source=np.asarray(str(source)),
                   posteriors=np.asarray(posterior, dtype=object),
                   assignments=np.asarray(assignments, dtype=object))
    pred_dir = OUT / "predictions" / cell_id
    result = run_partition(data, fit_indices, eval_indices, posterior, mode, seed,
                           deadline, cell_id, run_id, OUT / "lambda_curves.csv",
                           OUT / "models" / cell_id / "segment_model.npz", pred_dir)
    fit_used = np.unique(np.concatenate([assignments[i] for i in fit_indices]))
    if list(eval_indices) == list(fit_indices):
        unseen = 0.0
    else:
        top = np.concatenate([assignments[i] for i in eval_indices])
        unseen = float(np.mean(~np.isin(top, fit_used)))
    status = "degenerate" if not .5 <= result["action_run_gt_ratio"] <= 2 else "complete"
    return dict(cell_id=cell_id, run_id=run_id, dataset=data.name, protocol=protocol,
                fold=fold, standardize_per_recording=standardize, representation=mode,
                num_units=k, temperature_multiplier=multiplier, seed=seed, control=control,
                feature_source=str(source if reusable else artifact),
                unseen_top_unit_fraction=unseen, runtime_seconds=time.monotonic()-started,
                status=status, **result)


def validate_inputs(datasets):
    checks = []
    for ds in datasets:
        data = load_dataset(ds)
        d7 = score(data.gts, map_predictions(data.gts,
                   majority_code_predictions(data.gts, data.codes), "hungarian"))
        passed = abs(d7["MoF"] - D7_REFERENCE[ds]) <= 1.5
        folds = np.load(ROOT / "results" / "exp2round" / "q1q2" / "artifacts" / ds / "folds" / "subject.npz")
        leakage = any(set(folds["subjects"][folds["folds"] == f]) &
                      set(folds["subjects"][folds["folds"] != f]) for f in range(4))
        checks.append(dict(dataset=ds, d7_mof=d7["MoF"], d7_pass=passed,
                           frame_alignment=True, subject_fold_leakage=leakage,
                           folds_match_names=bool(np.array_equal(folds["names"], data.names))))
        if not passed or leakage or not checks[-1]["folds_match_names"]:
            raise RuntimeError(f"input acceptance gate failed for {ds}: {checks[-1]}")
        atomic_npz(OUT / "splits" / f"{ds}_subject.npz", names=folds["names"],
                   subjects=folds["subjects"], folds=folds["folds"])
    (OUT / "acceptance_inputs.json").write_text(json.dumps(checks, indent=2) + "\n")
    return checks


def plan():
    primary=[]; permutations=[]; inductive=[]; sensitivities=[]
    for ds in ("hugadb", "lara"):
        for std in (False, True):
            for seed in SEEDS:
                base=(ds,std,"soft",500,1.0,seed)
                primary.append((*base,"observed")); permutations.append((*base,"permutation"))
                for fold in range(4): inductive.append((*base,"observed",fold))
            for mode,k,temp in (("hard",500,1.),("soft",500,2.),("soft",200,1.),("soft",1000,1.)):
                for seed in SEEDS: sensitivities.append((ds,std,mode,k,temp,seed,"observed"))
    return primary, permutations, inductive, sensitivities


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--deadline-epoch",type=float,required=True)
    parser.add_argument("--run-id",default=None); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); run_id=args.run_id or uuid.uuid4().hex
    deadline=time.monotonic()+max(0,args.deadline_epoch-time.time())
    loaded={d:load_dataset(d) for d in ("hugadb","lara")}
    manifest=dict(run_id=run_id,started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
      checkpoint_sha256={d:sha256_file(CHECKPOINTS[d]) for d in ("hugadb","lara")},
      cache_identity={d:loaded[d].cache_identity for d in loaded},
      python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,
      seeds=SEEDS,window=WINDOW,fps=FPS,max_seconds=MAX_SECONDS,deadline_epoch=args.deadline_epoch,
      primary={"representation":"soft","num_units":500,"temperature_multiplier":1.0,
               "standardization":[False,True]},
      sensitivities={"representation":["hard","soft"],"num_units":[200,500,1000],
                     "temperature_multiplier":[1.0,2.0]},
      protocols=["transductive","subject_disjoint"],
      subject_splits={d:f"splits/{d}_subject.npz" for d in loaded},
      threads=min(8,os.cpu_count() or 1),status="running")
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    validate_inputs(("hugadb","lara")); cells_path=OUT/"cells.csv"
    existing={r["cell_id"] for r in read_rows(cells_path) if r.get("status") in {"complete","degenerate"}}
    primary, permutations, inductive, sensitivities=plan()
    tasks=[]
    for item in primary: tasks.append(("transductive",item,None))
    for item in permutations: tasks.append(("transductive",item,None))
    for item in inductive: tasks.append(("subject_disjoint",item[:-1],item[-1]))
    for item in sensitivities: tasks.append(("transductive",item,None))
    not_run=[]
    try:
        for protocol,item,fold in tasks:
            ds,std,mode,k,temp,seed,control=item
            cell_id=configuration_id(ds,protocol,std,mode,k,temp,seed,control,fold)
            if cell_id in existing: continue
            check_deadline(deadline); data=load_dataset(ds)
            if fold is None: fit=eval_=np.arange(len(data.names))
            else:
                saved=np.load(ROOT/"results"/"exp2round"/"q1q2"/"artifacts"/ds/"folds"/"subject.npz")
                eval_=np.flatnonzero(saved["folds"]==fold); fit=np.flatnonzero(saved["folds"]!=fold)
            try:
                row=run_cell(data,protocol,std,mode,k,temp,seed,control,fit,eval_,fold,run_id,deadline)
            except DeadlineReached: raise
            except Exception as error:
                row=dict(cell_id=cell_id,run_id=run_id,dataset=ds,protocol=protocol,fold=fold,
                         standardize_per_recording=std,representation=mode,num_units=k,
                         temperature_multiplier=temp,seed=seed,control=control,status="failed",
                         reason=repr(error))
            # Logging errors must never replace successfully computed metrics.
            upsert(cells_path,row,"cell_id")
            print(json.dumps(row),flush=True)
    except DeadlineReached as error:
        completed={r["cell_id"] for r in read_rows(cells_path)}
        for protocol,item,fold in tasks:
            ds,std,mode,k,temp,seed,control=item
            cid=configuration_id(ds,protocol,std,mode,k,temp,seed,control,fold)
            if cid not in completed: not_run.append(dict(cell_id=cid,reason=str(error)))
        write_rows(OUT/"not_run.csv",not_run)
        manifest["status"]="deadline_reached"
    else:
        write_rows(OUT/"not_run.csv",[])
        manifest["status"]="complete"
    manifest["completed_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__ == "__main__": main()
