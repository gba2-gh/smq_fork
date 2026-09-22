"""
E4 -- analysis: discovered segment durations in seconds + JSD on a common
time base, per condition (experiment.md).

Run after a condition's training+eval has produced a checkpoint. Converts
both GT and predicted (post global-Hungarian) segment lengths from frames
to SECONDS using that condition's own fps (native_fps / sample_rate), then
bins at a fixed real-time width (native's 20-frame bin width, i.e.
20/60 s = 0.3333s) so all conditions are directly comparable regardless of
their frame rate.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.smq import SMQModel
from src.model.utils import get_num_actions
from src.model.eval_utils import evaluate_local_hungarian, global_mapping

NATIVE_FPS = 60
BIN_WIDTH_SECONDS = 20 / NATIVE_FPS  # matches E1 Part B's 20-native-frame bin

FILTERS, NUM_LAYERS, LATENT_DIM = 128, 3, 16
NUM_JOINTS, NUM_PERSON, NUM_FEATURES = 6, 1, 6

CONDITIONS = {
    "native_A1_B1": dict(data_root="data", patch_size=60,
                          ckpt="models/hugadb/epoch-30.model", fps=60, sample_rate=1),
    "A2": dict(data_root="data/e4_sweeps/hugadb_rate2", patch_size=60,
               ckpt="models/e4_sweeps/A2/hugadb/epoch-60.model", fps=30, sample_rate=2),
    "A3": dict(data_root="data/e4_sweeps/hugadb_rate4", patch_size=60,
               ckpt="models/e4_sweeps/A3/hugadb/epoch-120.model", fps=15, sample_rate=4),
    "B2": dict(data_root="data/e4_sweeps/hugadb_rate2", patch_size=30,
               ckpt="models/e4_sweeps/B2/hugadb/epoch-30.model", fps=30, sample_rate=2),
    "B3": dict(data_root="data/e4_sweeps/hugadb_rate4", patch_size=15,
               ckpt="models/e4_sweeps/B3/hugadb/epoch-30.model", fps=15, sample_rate=4),
}


def segment_lengths(seq):
    lengths = []
    run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            run += 1
        else:
            lengths.append(run)
            run = 1
    lengths.append(run)
    return lengths


def js_distance(p, q):
    return float(jensenshannon(p, q, base=2))


def hist_prob(seconds_lengths, bins):
    counts, _ = np.histogram(seconds_lengths, bins=bins)
    total = counts.sum()
    return counts / total if total > 0 else None


def run_condition(name, cfg):
    ckpt = Path(cfg["ckpt"])
    if not ckpt.exists():
        print(f"[{name}] no checkpoint at {ckpt} yet, skipping")
        return None

    data_root = Path(cfg["data_root"])
    features_path = data_root / "hugadb" / "features"
    gt_path = data_root / "hugadb" / "groundTruth"
    mapping_file = data_root / "hugadb" / "mapping" / "mapping.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_actions = get_num_actions(gt_path)

    model = SMQModel(
        in_channels=NUM_FEATURES, filters=FILTERS, num_layers=NUM_LAYERS,
        latent_dim=LATENT_DIM, num_actions=num_actions, num_joints=NUM_JOINTS,
        num_person=NUM_PERSON, patch_size=cfg["patch_size"],
    )
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()

    print(f"[{name}] running inference with {ckpt} (fps={cfg['fps']}) ...")
    with torch.no_grad():
        mof, edit_mean, f1_vec, gt_all, prediction_all = evaluate_local_hungarian(
            model=model, features_path=features_path, gt_path=gt_path,
            mapping_file=mapping_file, epoch=None, device=device, verbose=False,
        )
        pr2gt = global_mapping(gt_all, prediction_all)
    prediction_mapped = [np.vectorize(pr2gt.get)(p) for p in prediction_all]

    fps = cfg["fps"]
    gt_seconds, pred_seconds = [], []
    for gt_seq in gt_all:
        gt_seconds.extend(l / fps for l in segment_lengths(gt_seq))
    for pred_seq in prediction_mapped:
        pred_seconds.extend(l / fps for l in segment_lengths(pred_seq))

    max_s = max(max(gt_seconds), max(pred_seconds))
    bins = np.arange(0, max_s + BIN_WIDTH_SECONDS, BIN_WIDTH_SECONDS)
    p = hist_prob(gt_seconds, bins)
    q = hist_prob(pred_seconds, bins)
    jsd = js_distance(p, q) if p is not None and q is not None else float("nan")

    result = dict(
        condition=name, patch_size_frames=cfg["patch_size"], fps=fps,
        patch_duration_seconds=cfg["patch_size"] / fps,
        mof=mof, edit=edit_mean, f1_10=f1_vec[0], f1_25=f1_vec[1], f1_50=f1_vec[2],
        mean_gt_duration_s=float(np.mean(gt_seconds)),
        mean_pred_duration_s=float(np.mean(pred_seconds)),
        median_pred_duration_s=float(np.median(pred_seconds)),
        jsd_seconds_x100=jsd * 100,
    )
    print(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS.keys()))
    args = parser.parse_args()

    out_root = Path("figures/e4_sweeps")
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.conditions:
        r = run_condition(name, CONDITIONS[name])
        if r is not None:
            rows.append(r)

    if rows:
        import csv
        fieldnames = list(rows[0].keys())
        with open(out_root / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote {out_root / 'summary.csv'}")
