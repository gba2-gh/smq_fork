"""
Data audit for the segmentation-first programme (EXPERIMENT_MANIFEST.md).

Establishes, per dataset: recordings, lengths, classes, background handling,
GT run statistics, participant identifiers (LARa only), and overlap of BABEL
subsets by recording id and by feature content.

Writes results/seg/data_audit.json
"""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from script.seg.common import DATASETS, load_gt_labels, runs  # noqa: E402

OUT = ROOT / "results" / "seg"
OUT.mkdir(parents=True, exist_ok=True)


def content_hash(x):
    return hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()


def main():
    audit = {}
    hashes = {}
    for name, cfg in DATASETS.items():
        root = ROOT / "data" / name
        names = sorted(p.stem for p in (root / "features").glob("*.npy"))
        mapping = {}
        for line in (root / "mapping" / "mapping.txt").read_text().splitlines():
            i, a = line.strip().split(" ", 1)
            mapping[int(i)] = a
        a2i = {v: k for k, v in mapping.items()}

        lens, gt_lens, shapes, frame_count = [], [], Counter(), Counter()
        run_counts, run_durs, run_dur_by_class = [], [], {k: [] for k in mapping}
        n_single_run = 0
        hashes[name] = {}
        per_rec = {}
        for n in names:
            x = np.load(root / "features" / f"{n}.npy", mmap_mode="r")
            shapes[tuple(x.shape[:1] + x.shape[2:])] += 1
            T = x.shape[1]
            lab = load_gt_labels(root, n, a2i)
            lens.append(T)
            gt_lens.append(len(lab))
            for k, c in zip(*np.unique(lab, return_counts=True)):
                frame_count[int(k)] += int(c)
            r = runs(lab)
            run_counts.append(len(r))
            n_single_run += len(r) == 1
            for s, e, l in r:
                run_durs.append(e - s)
                run_dur_by_class[int(l)].append(e - s)
            hashes[name][n] = content_hash(np.asarray(x))
            per_rec[n] = dict(T=int(T), n_runs=len(r), classes=sorted({int(l) for _, _, l in r}))
        lens = np.array(lens)
        fps = cfg["fps"]
        tot = sum(frame_count.values())
        info = dict(
            n_recordings=len(names), fps=fps, patch=cfg["patch"], K=len(mapping),
            classes={int(k): v for k, v in mapping.items()},
            feature_shapes={str(k): v for k, v in shapes.items()},
            gt_len_mismatch=int((lens != np.array(gt_lens)).sum()),
            frames_total=int(lens.sum()),
            length_frames=dict(min=int(lens.min()), median=float(np.median(lens)), max=int(lens.max())),
            length_sec=dict(min=float(lens.min() / fps), median=float(np.median(lens) / fps),
                            max=float(lens.max() / fps)),
            class_frame_share={mapping[k]: frame_count[k] / tot for k in sorted(frame_count)},
            gt_runs_per_recording=dict(mean=float(np.mean(run_counts)), median=float(np.median(run_counts)),
                                       min=int(min(run_counts)), max=int(max(run_counts)),
                                       single_run_recordings=int(n_single_run)),
            gt_run_duration_sec=dict(mean=float(np.mean(run_durs) / fps), median=float(np.median(run_durs) / fps),
                                     p10=float(np.percentile(run_durs, 10) / fps),
                                     p90=float(np.percentile(run_durs, 90) / fps)),
            gt_runs_total=int(sum(run_counts)),
            gt_run_duration_sec_by_class={mapping[k]: (float(np.median(v) / fps) if v else None)
                                          for k, v in run_dur_by_class.items()},
            recordings_with_run_count_lt4=int(sum(c < 4 for c in run_counts)),
            recordings_shorter_than_1s_patch=int((lens < cfg["patch"]).sum()),
            per_recording=per_rec,
        )
        if name == "lara":
            subj = Counter()
            scen = Counter()
            for n in names:
                m = re.match(r"L(\d+)_S(\d+)_R(\d+)", n)
                subj[m.group(2)] += 1
                scen[m.group(1)] += 1
            info["participants"] = dict(n=len(subj), recordings_per_participant=dict(sorted(subj.items())))
            info["scenarios"] = dict(sorted(scen.items()))
        else:
            info["participants"] = "not encoded in filenames (BABEL sequence ids; no subject id available)"
        audit[name] = info

    # BABEL overlap
    ov = {}
    bn = ["babel1", "babel2", "babel3"]
    for i, a in enumerate(bn):
        for b in bn[i + 1:]:
            sa, sb = set(hashes[a]), set(hashes[b])
            same_id = sa & sb
            same_content = [n for n in same_id if hashes[a][n] == hashes[b][n]]
            ha, hb = set(hashes[a].values()), set(hashes[b].values())
            ov[f"{a}&{b}"] = dict(shared_ids=len(same_id), shared_ids_identical_features=len(same_content),
                                  shared_feature_hashes=len(ha & hb),
                                  frac_of_smaller=len(same_id) / min(len(sa), len(sb)))
    # label agreement on shared ids (same recording, differently-mapped label sets)
    audit["babel_overlap"] = ov
    for k in bn:
        for n in audit[k]["per_recording"]:
            pass
    (OUT / "data_audit.json").write_text(json.dumps(audit, indent=1))
    # concise print
    for k, v in audit.items():
        if k == "babel_overlap":
            print(json.dumps(v, indent=1))
            continue
        print(k, {kk: vv for kk, vv in v.items() if kk not in ("per_recording", "classes")})


if __name__ == "__main__":
    main()
