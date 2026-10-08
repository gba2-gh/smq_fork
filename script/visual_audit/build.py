"""Read-only visual audit of saved, matched LARa predictions. No fitting or decoding.

Run with the smq environment: python script/visual_audit/build.py
The dashboard embeds run-length encoded timelines; no server or network is needed.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from script.repro.d_error_decomposition import score
from script.week5.raw_kmeans_cross_dataset import datasets as D

OUT = ROOT / "results/visual_audit/lara_matched_001"
SEEDS = (111, 222, 1538574472)
M1 = ROOT / "results/compositional_dynamics/m1_lara_001/cells"
IRT = ROOT / "results/input_representation/irt_lara_001"
ARMS = {
    "T2 B0": M1 / "b0_centre__seed{seed}",
    "Body-scale B0": IRT / "b0__t2_bodyscale/cells/b0_centre__seed{seed}",
    "Centered B0": IRT / "b0__t2_centered/cells/b0_centre__seed{seed}",
    "Ordered GRU": M1 / "m1_ordered__seed{seed}",
    "Bag": M1 / "m1_bag__seed{seed}",
    "Occupancy": M1 / "occupancy__seed{seed}",
}


def runs(x):
    starts = np.r_[0, np.flatnonzero(x[1:] != x[:-1]) + 1]
    return [[int(s), int(e), int(x[s])] for s, e in zip(starts, np.r_[starts[1:], len(x)])]


def boundary_counts(gt, pred, tolerance):
    """Maximum-cardinality, one-to-one matching of sorted boundaries within tolerance.

    Class agnostic. A displaced boundary beyond tolerance produces BOTH a missed
    and an extra boundary. These counts do not identify a causal failure mode.
    """
    a = np.flatnonzero(gt[1:] != gt[:-1]) + 1
    b = np.flatnonzero(pred[1:] != pred[:-1]) + 1
    i = j = matched = 0
    while i < len(a) and j < len(b):
        if b[j] < a[i] - tolerance:
            j += 1
        elif b[j] > a[i] + tolerance:
            i += 1
        else:
            matched += 1
            i += 1
            j += 1
    return dict(matched=matched, extra=len(b) - matched, missed=len(a) - matched)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_helpers():
    a = np.array([1, 1, 2, 2, 3, 3])
    assert boundary_counts(a, a, 0) == dict(matched=2, extra=0, missed=0)
    assert boundary_counts(a, np.ones(6), 0) == dict(matched=0, extra=0, missed=2)
    b = np.array([1, 2, 2, 3, 3, 3])
    assert boundary_counts(a, b, 0) == dict(matched=0, extra=2, missed=2)
    assert boundary_counts(a, b, 1) == dict(matched=2, extra=0, missed=0)
    assert runs(a) == [[0, 2, 1], [2, 4, 2], [4, 6, 3]]


def preview(payload):
    """Export a deterministic first-recording example, not a selected success/failure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    colors = ['#517dc4', '#e49934', '#6eaf73', '#ac7bc4', '#d75b64', '#47a9b7', '#b49c49', '#999fa7']
    name = sorted(payload['recordings'])[0]
    rows = [('GT', payload['recordings'][name]['gt'])]
    rows += [(c['arm'], c['records'][name]['pred']) for c in payload['cells'] if c['seed'] == 111]
    fig, ax = plt.subplots(figsize=(15, 5))
    for i, (label, segments) in enumerate(rows):
        for start, end, c in segments:
            ax.broken_barh([(start/50, (end-start)/50)], (i-.34, .68), facecolors=colors[(c-1)%8])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, payload['recordings'][name]['frames']/50)
    ax.set_xlabel('Time (seconds)')
    ax.set_title(f'{name} | seed 111 | saved global Hungarian mapping\nFirst recording by name; no example selection')
    ax.legend(handles=[Patch(color=colors[(int(k)-1)%8], label=v) for k,v in payload['labels'].items()],
              loc='upper center', bbox_to_anchor=(.5, -.18), ncol=4, frameon=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT/'first_recording.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    verify_helpers()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = D.SPECS["lara"]
    labels = {int(v): k for k, v in D.label_map(spec).items()}
    payload = dict(dataset="LARa", fps=spec.fps, window=12, labels=labels,
                   arms=list(ARMS), seeds=list(SEEDS), recordings={}, cells=[])
    rec_rows, seg_rows, class_rows, sources = [], [], [], []
    gts = None
    names = None
    for seed in SEEDS:
        for arm, template in ARMS.items():
            folder = Path(str(template).format(seed=seed))
            pred_path, score_path = folder / "frame_predictions_eval.npz", folder / "scores.json"
            with np.load(pred_path, allow_pickle=False) as saved:
                current_names = sorted(saved.files)
                if names is None:
                    names = current_names
                    gts = D.load_gts(spec, names)
                    payload["recordings"] = {n: dict(gt=runs(gts[n]), frames=len(gts[n]),
                                                     subject=D.subject_of(spec, n)) for n in names}
                assert current_names == names, f"Unmatched population: {folder}"
                raw = {n: saved[n].copy() for n in names}
            scores = json.loads(score_path.read_text())
            endpoint = str(scores["primary_endpoint_steps"])
            reference = scores["eval"][endpoint]
            mapping = {int(k): int(v) for k, v in reference["mapping"].items()}
            confusion = {int(s): {int(c): int(v) for c, v in counts.items()}
                         for s, counts in reference["confusion"].items()}
            oracle = {s: max(sorted(counts), key=lambda c: counts[c]) for s, counts in confusion.items()}
            mapped, oracle_pred = {}, {}
            for n in names:
                assert raw[n].ndim == 1 and len(raw[n]) == len(gts[n]), f"Frame mismatch: {folder}/{n}"
                assert set(np.unique(raw[n])).issubset(mapping), f"Unmapped states: {folder}/{n}"
                mapped[n] = np.array([mapping[int(x)] for x in raw[n]], dtype=np.int32)
                oracle_pred[n] = np.array([oracle[int(x)] for x in raw[n]], dtype=np.int32)
            measured = score([gts[n] for n in names], [mapped[n] for n in names])
            for k, v in measured.items():
                assert abs(v - reference["metrics"][k]) <= .011, (folder, k, v, reference["metrics"][k])
            cell = dict(arm=arm, seed=seed, metrics=measured, mapping=mapping, confusion=confusion,
                        source=folder.relative_to(ROOT).as_posix(), records={}, recall={})
            for c, label in labels.items():
                support = sum(int((gts[n] == c).sum()) for n in names)
                hit = sum(int(((gts[n] == c) & (mapped[n] == c)).sum()) for n in names)
                recall = 100 * hit / support if support else None
                cell["recall"][c] = recall
                class_rows.append(dict(arm=arm, seed=seed, action=label, action_id=c,
                                       frames=support, recall_percent=recall))
            for n in names:
                gt, pr = gts[n], mapped[n]
                gr, rr, mr = runs(gt), runs(raw[n]), runs(pr)
                metrics = score([gt], [pr])
                exact, tolerant = boundary_counts(gt, pr, 0), boundary_counts(gt, pr, 12)
                row = dict(arm=arm, seed=seed, recording=n, subject=D.subject_of(spec, n),
                           frames=len(gt), **metrics, gt_segments=len(gr), raw_segments=len(rr),
                           predicted_segments=len(mr), segment_ratio=len(mr)/len(gr),
                           exact_extra=exact["extra"], exact_missed=exact["missed"],
                           extra_024s=tolerant["extra"], missed_024s=tolerant["missed"],
                           matched_024s=tolerant["matched"])
                rec_rows.append(row)
                pb = np.flatnonzero(pr[1:] != pr[:-1]) + 1
                for start, end, c in gr:
                    vals, cnt = np.unique(pr[start:end], return_counts=True)
                    dominant = int(vals[cnt.argmax()])
                    seg_rows.append(dict(arm=arm, seed=seed, recording=n, action=labels[c],
                        start_frame=start, end_frame=end, seconds=(end-start)/spec.fps,
                        recall_percent=100*float(np.mean(pr[start:end] == c)),
                        dominant_prediction=labels.get(dominant, str(dominant)),
                        internal_boundaries=int(np.sum((pb > start) & (pb < end))),
                        deep_internal_boundaries=int(np.sum((pb > start+12) & (pb < end-12)))))
                cell["records"][n] = dict(raw=rr, pred=mr, oracle=runs(oracle_pred[n]), metrics=row)
            payload["cells"].append(cell)
            sources.append(dict(arm=arm, seed=seed, path=folder.relative_to(ROOT).as_posix(),
                                predictions_sha256=sha(pred_path), scores_sha256=sha(score_path)))
            print(f"Verified {arm}, seed {seed}: {len(names)} recordings; MoF {measured['MoF']}", flush=True)
    write_csv(OUT / "recordings.csv", rec_rows)
    write_csv(OUT / "segments.csv", seg_rows)
    write_csv(OUT / "classes.csv", class_rows)
    payload_text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    preview(payload)
    (OUT / "audit.json").write_text(payload_text, encoding="utf-8")
    html = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    (OUT / "index.html").write_text(html.replace("__AUDIT_DATA__", payload_text.replace("<", "\\u003c")), encoding="utf-8")
    manifest = dict(sources=sources, dataset="lara", population="44 recording-held-out evaluation recordings",
        mapping="Saved global evaluation-population Hungarian mapping; never refitted per recording",
        oracle="Many-to-one mapping uses all evaluation labels, for visualization only",
        protocol="Held-out decoder evaluation with transductive tokenization; subjects overlap",
        boundary_tolerances_frames=[0, 12], fps=50, verification="All 18 cells reproduce all five saved metrics",
        ground_truth_sha256={n: sha(ROOT/'data/lara/groundTruth'/f'{Path(n).stem}.txt') for n in names},
        implementation_sha256={p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name('viewer.html'))})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = ["# Matched LARa visual audit", "", "18 cells: six arms, three seeds, the same 44 evaluation recordings.",
               "Open index.html in a browser. All five pooled metrics reproduced the saved scores in every cell.", "",
               "| Arm | Mean extra boundaries / recording | Mean missed boundaries / recording | Recordings with both (%) |",
               "|---|---:|---:|---:|"]
    for arm in ARMS:
        rows = [r for r in rec_rows if r['arm'] == arm]
        summary.append(f"| {arm} | {np.mean([r['extra_024s'] for r in rows]):.1f} | "
                       f"{np.mean([r['missed_024s'] for r in rows]):.1f} | "
                       f"{100*np.mean([r['extra_024s']>0 and r['missed_024s']>0 for r in rows]):.1f} |")
    summary += ["", "Boundary matching is class-agnostic, one-to-one, within 12 frames (0.24 s).",
                "Extra and missed boundaries can both reflect displaced transitions; these are descriptive counts, not causal labels.",
                "A high segment-count ratio alone does not establish that all errors are overfragmentation.",
                "", "The oracle row uses evaluation-label many-to-one mapping; it is not an unsupervised result.",
                "Colors identify actions consistently. Raw state IDs are local to each cell and cannot be compared across seeds.",
                "Recordings are listed without filtering out failures. Default ordering is by name, not selected examples.",
                "This first audit covers the latest matched LARa experiments, not all historical projects.",
                "No synchronized skeleton replay is included yet; label timelines cannot establish a physical explanation.",
                "The static first_recording.png export was visually inspected. Automated browser interaction verification was unavailable in this session."]
    (OUT / "README.md").write_text("\n".join(summary)+"\n", encoding="utf-8")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
