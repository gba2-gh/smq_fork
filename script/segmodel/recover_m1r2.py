"""Recover completed M1r2 cells from artifacts, without fitting or decoding."""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import linear_sum_assignment

from script.exp2round.q1q2.core import load_dataset, score
from script.segmodel.m1r2_core import OUT, FPS, WINDOW, MAX_SECONDS, run_durations
from script.segmodel.run_m1r2 import read_rows, write_rows


def verify_mapping(raw, mapped, truth):
    """Check the stored map is consistent, injective and Hungarian-optimal."""
    states = np.unique(np.concatenate(raw))
    actions = np.unique(np.concatenate(truth))
    matrix = np.zeros((len(states), len(actions)), dtype=np.int64)
    mapping = {}
    for x, y, gt in zip(raw, mapped, truth):
        s = np.searchsorted(states, x)
        a = np.searchsorted(actions, gt)
        matrix += np.bincount(s * len(actions) + a,
                              minlength=matrix.size).reshape(matrix.shape)
        for state in np.unique(x):
            labels = np.unique(y[x == state])
            assert len(labels) == 1, "inconsistent stored mapping within recording"
            label = int(labels[0])
            assert int(state) not in mapping or mapping[int(state)] == label
            mapping[int(state)] = label
    assert len(set(mapping.values())) == len(mapping), "mapping is not injective"
    r, c = linear_sum_assignment(-matrix)
    correct = sum(int(np.count_nonzero(y == gt)) for y, gt in zip(mapped, truth))
    assert correct == int(matrix[r, c].sum()), "stored map is not Hungarian-optimal"


def recover(row, data, curves):
    cid = row['cell_id']
    folder = OUT / 'models' / cid
    with np.load(folder / 'segment_model.npz') as model:
        centers = model['centroids'].copy()
        penalty = float(model['lambda_value'])
        curve = model['lambda_curve'].copy()
        trace = [json.loads(str(t)) for t in model['trace']]
        final_objective = float(model['final_objective'])
    assert np.isfinite(centers).all() and np.isfinite(final_objective)
    assert len(trace) and np.isfinite([t['objective'] for t in trace]).all()
    monotone = all(b['objective'] <= a['objective'] + 1e-7*max(1, abs(a['objective']))
                   for a, b in zip(trace, trace[1:]))
    assert monotone and final_objective <= trace[-1]['objective'] + 1e-7*max(1, abs(trace[-1]['objective']))
    all_indices = np.arange(len(data.names))
    if row['fold'] != '':
        with np.load(OUT / 'splits' / f'{data.name}_subject.npz') as splits:
            assert np.array_equal(splits['names'], data.names)
            evaluation = all_indices[splits['folds'] == int(row['fold'])]
            fitting = all_indices[splits['folds'] != int(row['fold'])]
            assert not set(splits['subjects'][evaluation]) & set(splits['subjects'][fitting])
    else:
        fitting = evaluation = all_indices
    fps, width = FPS[data.name], WINDOW[data.name]
    maximum = int(np.floor(MAX_SECONDS*fps/width))
    raw, mapped, gts = [], [], []
    blocks = cap_blocks = 0
    occupancy = np.zeros(data.num_actions, np.int64)
    for i in evaluation:
        with np.load(OUT / 'predictions' / cid / f'{data.names[i]}.npz') as saved:
            x, y, edges = saved['raw_states'], saved['labels'], saved['boundaries']
            gt = data.gts[i]
            assert len(x) == len(y) == len(gt)
            assert edges[0] == 0 and edges[-1] == len(gt) and np.all(np.diff(edges) > 0)
            assert np.all(edges[:-1] % width == 0)
            windows = np.diff((edges + width - 1)//width)
            assert np.all((windows >= 1) & (windows <= maximum))
            assert x.min() >= 0 and x.max() < data.num_actions
            assert np.all(np.isin(np.flatnonzero(x[1:] != x[:-1])+1, edges))
            labels = x[edges[:-1]]
            occupancy += np.bincount(labels, minlength=data.num_actions)
            blocks += len(labels)
            cap_blocks += int(np.count_nonzero(windows == maximum))
            raw.append(x); mapped.append(y); gts.append(gt)
    verify_mapping(raw, mapped, gts)
    metrics = score(gts, mapped)
    durations = run_durations(raw, fps)
    gt_runs = len(run_durations(gts, fps))
    reference = float(run_durations([data.codes[i] for i in fitting], fps).mean())
    selected = min(curve, key=lambda p: (abs(p[3]-reference), p[0]))
    assert float(selected[0]) == penalty, 'lambda does not match label-free selection'
    csv_curve = curves[cid]
    assert len(csv_curve) == len(curve)
    for saved_point, csv_point in zip(curve, csv_curve):
        assert np.isclose(saved_point[0], float(csv_point['lambda_value']))
        assert np.isclose(saved_point[3], float(csv_point['mean_run_duration_seconds']))

    feature_path = folder / 'features.npz'
    with np.load(feature_path, allow_pickle=True) as feature:
        if 'feature_source' in feature.files:
            source = str(feature['feature_source'])
        else:
            source = str(feature_path)
        if row['fold'] != '':
            assignments = feature['assignments']
            used = np.unique(np.concatenate([np.asarray(assignments[i], np.int32) for i in fitting]))
            top = np.concatenate([np.asarray(assignments[i], np.int32) for i in evaluation])
            unseen = float(np.mean(~np.isin(top, used)))
            assert np.all(np.isin(feature['fit_ids'][:, 0], fitting))
        else:
            unseen = 0.0
    converged = trace[-1]['relative_improvement'] is not None and trace[-1]['relative_improvement'] <= 1e-6
    ratio = len(durations)/gt_runs
    result = {key: value for key, value in row.items() if value != ''}
    result.update(metrics)
    result.update(
        fold=row['fold'], reason='', status='complete' if .5 <= ratio <= 2 else 'degenerate',
        recovered=True, recovery_source='saved predictions and model; no refit',
        runtime_seconds='', runtime_status='unavailable: original row overwritten by logging handler',
        feature_source=source, unseen_top_unit_fraction=unseen,
        lambda_value=penalty, lambda_reference_duration=reference,
        lambda_start_action_runs=int(selected[2]), lambda_final_action_runs=len(durations),
        lambda_fit_final_action_runs=int(trace[-1]['action_runs']),
        lambda_fit_final_action_runs_note='last pre-update trace; final decode count unavailable for held-out protocol',
        dp_blocks=blocks, action_runs=len(durations), gt_action_runs=gt_runs,
        dp_block_gt_ratio=blocks/gt_runs, action_run_gt_ratio=ratio,
        mean_duration_seconds=float(durations.mean()), median_duration_seconds=float(np.median(durations)),
        duration_cap_blocks=cap_blocks, duration_cap_fraction=cap_blocks/max(1, blocks),
        state_occupancy=json.dumps(occupancy.tolist()), state_death=bool(np.any(occupancy == 0)),
        rounds=len(trace), converged=converged, cap_reached=not converged and len(trace)==60,
        final_objective=final_objective, verified_frames=sum(map(len, mapped)),
        verified_recordings=len(evaluation), mapping_verified=True,
        calibration_selection_verified=True, objective_monotone=True)
    return result


def main():
    backup = OUT / 'before_recovery'
    backup.mkdir(exist_ok=True)
    for filename in ('cells.csv', 'not_run.csv', 'REPORT.md', 'gate.json', 'manifest.json'):
        source = OUT / filename
        if source.exists() and not (backup / filename).exists():
            shutil.copy2(source, backup / filename)
    original = read_rows(backup / 'cells.csv')
    curves = defaultdict(list)
    for row in read_rows(OUT/'lambda_curves.csv'):
        curves[row['cell_id']].append(row)
    recovered, failures = [], []
    data = {name: load_dataset(name) for name in ('hugadb', 'lara')}
    for index, row in enumerate(original):
        try:
            result = recover(row, data[row['dataset']], curves)
            recovered.append(result)
            print(f"{index+1}/{len(original)} recovered {row['cell_id']}: MoF={result['MoF']}, F1@50={result['F1@50']}", flush=True)
        except Exception as error:
            failures.append({'cell_id': row['cell_id'], 'error': repr(error)})
            recovered.append(row)
            print(f"{index+1}/{len(original)} UNRECOVERED {row['cell_id']}: {error!r}", flush=True)
        write_rows(OUT/'cells_recovered.csv', recovered)
    audit = dict(completed_utc=datetime.now(timezone.utc).isoformat(), attempted=len(original),
                 recovered=len(original)-len(failures), failures=failures,
                 original_backup=str(backup), no_model_fits=True,
                 runtime_recoverable=False)
    (OUT/'recovery_audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf8')
    # Publish only after all rows have been considered; unrecoverable rows remain explicit failures.
    write_rows(OUT/'cells.csv', recovered)
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
