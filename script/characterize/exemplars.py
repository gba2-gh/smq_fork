"""Deterministic, complete-patch SMQ exemplar montages (no model fitting)."""
import argparse
import csv
import hashlib
import html
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

SEEDS = (1538574472, 111, 222)


def choose(pool, rec, subjects, names, start):
    def key(i):
        return hashlib.sha256(f'smq-characterization-exemplar/{names[rec[i]]}/{start[i]}'.encode()).hexdigest()
    order = sorted(map(int, pool), key=key)
    selected, people, recordings = [], set(), set()
    for mode in ('participant', 'recording', 'fill'):
        for i in order:
            if len(selected) == 6:
                return selected
            if i in selected:
                continue
            r = int(rec[i])
            s = str(subjects[r])
            if mode == 'participant' and s in people:
                continue
            if mode == 'recording' and r in recordings:
                continue
            selected.append(i)
            people.add(s)
            recordings.add(r)
    return selected


def render(ds, seed, code, selected, available, z, out):
    rows = max(1, len(selected))
    fig = plt.figure(figsize=(13, 2.8 * rows + 0.8), layout='constrained')
    gs = fig.add_gridspec(rows, 2, width_ratios=[1, 1.25])
    rec, start = z['rec'], z['start']
    tr = z['trajectory']
    # Shared limits across occurrences prevent apparent scale equivalence.
    if selected and ds == 'lara':
        xyz = tr[selected].reshape(len(selected), tr.shape[1], 21, 3)
        low, high = xyz.min(axis=(0, 1, 2)), xyz.max(axis=(0, 1, 2))
        centre = (low + high) / 2
        radius = max(float(np.max(high - low)) / 2, 1e-6)
        disp_limit = max(float(np.max(np.abs(xyz - xyz[:, :1]))), 1e-6)
    if selected and ds == 'hugadb':
        # Display scaling fixed to calibration patches, not per exemplar.
        mean, sd = z['_display_mean'], z['_display_sd']
    for row, i in enumerate(selected):
        r = int(rec[i])
        name = str(z['names'][r])
        person = str(z['subjects'][r])
        t = tr[i]
        heading = f'{name} | person {person} | frames [{start[i]}, {start[i] + len(t)})'
        if ds == 'lara':
            ax = fig.add_subplot(gs[row, 0], projection='3d')
            x = t.reshape(len(t), 21, 3)
            for joint in range(21):
                segments = np.stack([x[:-1, joint], x[1:, joint]], axis=1)
                line = Line3DCollection(segments, cmap='viridis', norm=Normalize(0, 1), linewidth=1.5)
                line.set_array(np.linspace(0, 1, len(segments)))
                ax.add_collection3d(line)
            ax.scatter(*x[0].T, c='#440154', s=11, marker='o', label='start')
            ax.scatter(*x[-1].T, c='#d7bd00', s=19, marker='x', label='end')
            ax.set(xlim=(centre[0]-radius, centre[0]+radius), ylim=(centre[1]-radius, centre[1]+radius), zlim=(centre[2]-radius, centre[2]+radius), xlabel='native x', ylabel='native y', zlabel='native z')
            ax.set_box_aspect((1, 1, 1))
            ax.set_title('21 root-relative joint tracks; purple → yellow', fontsize=9)
            ax.legend(fontsize=7, loc='upper left')
            hm = fig.add_subplot(gs[row, 1])
            delta = (t - t[0]).T
            im = hm.imshow(delta, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-disp_limit, vmax=disp_limit, extent=[0, len(t)/50, 0, 63])
            hm.set(xlabel='Forward time (s)', ylabel='Joint × xyz coordinate', title=heading)
            hm.title.set_fontsize(8)
            cb = fig.colorbar(im, ax=hm, shrink=.8)
            cb.set_label('Δ position (native units)', fontsize=8)
        else:
            signal = (t - mean) / sd
            for col, group in enumerate((range(18), range(18, 36))):
                ax = fig.add_subplot(gs[row, col])
                for j in group:
                    ax.plot(np.arange(len(t))/60, signal[:, j] + (j % 18)*5, lw=.6)
                ax.set_yticks(np.arange(18)*5)
                ax.set_yticklabels([f'c{j//6}/s{j%6}' for j in group], fontsize=6)
                ax.set(xlabel='Forward time (nominal s)', ylabel='Channel / sensor (5-SD trace offsets)')
                ax.set_title(heading if col == 0 else 'Calibration-standardized sensor values; native order retained', fontsize=8)
                ax.grid(axis='x', alpha=.2)
    if not selected:
        ax = fig.add_subplot(gs[:, :])
        ax.text(.5, .5, 'No eligible evaluation occurrences for this code', ha='center', va='center')
        ax.axis('off')
    fig.suptitle(f'{ds.upper()} · seed {seed} · code {code} · {len(selected)} shown / {available} eligible\nFixed sampling: participant diversity, then recording diversity', fontsize=11)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['hugadb', 'lara'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = root / 'figures/characterization/exemplars'
    output.mkdir(parents=True, exist_ok=True)
    records, cards = [], []
    for ds in args.datasets:
        with np.load(root / f'results/characterization/{ds}_cache.npz', allow_pickle=False) as archive:
            z = {k: archive[k] for k in archive.files}
        if ds == 'hugadb':
            calibration = z['trajectory'][~z['evaluation'].astype(bool)]
            z['_display_mean'] = calibration.mean(axis=(0, 1))
            z['_display_sd'] = np.maximum(calibration.std(axis=(0, 1)), 1e-6)
            del calibration
        for seed_i, seed in enumerate(SEEDS):
            for code in range(10 if ds == 'hugadb' else 8):
                pool = np.flatnonzero(z['evaluation'].astype(bool) & (z['codes'][:, seed_i] == code))
                selected = choose(pool, z['rec'], z['subjects'], z['names'], z['start'])
                fname = f'{ds}_seed{seed}_code{code:02}.png'
                render(ds, seed, code, selected, len(pool), z, output / fname)
                common = dict(dataset=ds, seed=seed, code=code, available_patches=len(pool), available_recordings=len(np.unique(z['rec'][pool])), available_participants=len(np.unique(z['subjects'][z['rec'][pool]])), selected_count=len(selected), montage=f'figures/characterization/exemplars/{fname}')
                for rank, i in enumerate(selected or [None]):
                    if i is None:
                        detail = dict(rank='', patch_index='', recording='', participant='', start_frame='', stop_frame='', majority_action='', action_purity='')
                    else:
                        r = int(z['rec'][i])
                        detail = dict(rank=rank+1, patch_index=i, recording=z['names'][r], participant=z['subjects'][r], start_frame=int(z['start'][i]), stop_frame=int(z['start'][i])+z['trajectory'].shape[1], majority_action=int(z['action'][i]), action_purity=float(z['purity'][i]))
                    records.append({**common, **detail})
                cards.append(f'<li><a href="{fname}">{html.escape(ds)} seed {seed}, code {code}</a>: {len(selected)} / {len(pool)} patches; {common["available_participants"]} eligible participants</li>')
        print(f'{ds}: finished all {len(SEEDS)} seeds', flush=True)
    with (root / 'results/characterization/exemplars.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    (output / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>SMQ complete-patch exemplars</title><style>body{font:16px system-ui;max-width:1000px;margin:40px auto;line-height:1.6}li{margin:.3em 0}</style><h1>SMQ complete-patch exemplars</h1><p>Every code of every frozen checkpoint, including rare and absent codes. Sampling uses a fixed hash order and prioritizes participant then recording diversity. All patches were held out from characterization calibration, but were seen during SMQ training. IDs are checkpoint-specific.</p><p>LARa panels show full, ordered root-relative joint trajectories and displacement heatmaps in native units. HuGaDB panels show sensor signals, standardized only for display using calibration statistics; they are not skeletal movement or posture.</p><ul>' + '\n'.join(cards) + '</ul></html>', encoding='utf-8')


if __name__ == '__main__':
    main()
