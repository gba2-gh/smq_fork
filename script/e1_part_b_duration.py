"""
E1 Part B -- Discovered duration structure (experiment.md, revised spec).

Metric definition (matching HiST-VQ/HVQ's convention, per experiment.md):
    - Bin segment lengths at 20 frames, with an overflow bin for the tail.
    - Normalise both (GT, predicted) histograms to sum to 1 -- per video,
      per activity (activity = dataset action class; video = sequence file;
      our datasets have no Breakfast-style video==activity structure, so
      "activity" is applied at the class-label level and "video" at the
      per-sequence-file level -- see the docstring note in run_dataset()).
    - Compute JS distance (sqrt of the JS divergence) per (video, activity).
    - Average over videos within each activity -> per-activity JSD.
    - Frame-weight across activities (weight = that activity's total GT
      frame count) -> one dataset-level number.
    - Report x100.

Bins are in RAW FRAMES (unrelated to patch size -- patch-normalised
histograms are E1 Part A's separate output).

Also computes, as required by the revised brief -- TWO nulls, which measure
different things:
    (a) Sparsity floor: bootstrap -- split each activity's pooled GT segment
        lengths into two disjoint random halves, JS-distance between them,
        repeated and averaged. This estimates the noise from building a
        histogram out of few samples.
    (b) Quantization ceiling: round the GT segmentation to patch boundaries
        (each patch gets the majority GT label within it -- the best
        segmentation any one-second-resolution method could produce), then
        JSD between that quantized GT and the raw GT. Predicted JSD is then
        interpreted against this ceiling: predicted ~= ceiling means the
        duration error is fully explained by patch size (nothing left for a
        duration-aware decoder to recover); predicted >> ceiling means
        there's real headroom.

Also: a bin-free cross-check (Wasserstein-1, pooled raw lengths, per
activity, frame-weighted to a dataset-level number), reported both raw and
normalised by mean GT segment length (Wasserstein-1 is not scale-invariant,
so raw cross-dataset comparisons are not meaningful without this).
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.smq import SMQModel
from src.model.utils import get_num_actions
from src.model.eval_utils import evaluate_local_hungarian, global_mapping, read_mapping_file

DATASET_DEFAULTS = {
    "hugadb": dict(num_features=6, num_joints=6, num_person=1, patch_size=60),
    "lara": dict(num_features=6, num_joints=22, num_person=1, patch_size=50),
    "babel1": dict(num_features=3, num_joints=25, num_person=1, patch_size=30),
    "babel2": dict(num_features=3, num_joints=25, num_person=1, patch_size=30),
    "babel3": dict(num_features=3, num_joints=25, num_person=1, patch_size=30),
}
FILTERS, NUM_LAYERS, LATENT_DIM = 128, 3, 16
BIN_WIDTH = 20
N_BOOTSTRAP = 200
RNG_SEED = 42


def labeled_segments(seq):
    """Run-length encode a 1D label sequence -> list of (label, length_in_frames)."""
    out = []
    label = seq[0]
    run = 1
    for i in range(1, len(seq)):
        if seq[i] == label:
            run += 1
        else:
            out.append((label, run))
            label = seq[i]
            run = 1
    out.append((label, run))
    return out


def make_bins(all_gt_lengths, bin_width=BIN_WIDTH):
    """20-frame bins up to the 95th percentile of pooled GT lengths, plus an
    overflow bin capturing the tail. Fixed per-dataset so every video's
    histogram uses the same bin edges."""
    cutoff = int(np.ceil(np.percentile(all_gt_lengths, 95) / bin_width) * bin_width)
    cutoff = max(cutoff, bin_width)
    edges = list(range(0, cutoff + bin_width, bin_width)) + [np.inf]
    return np.array(edges)


def hist_prob(lengths, bins):
    """Normalised (sum-to-1) histogram of lengths over the given bin edges."""
    counts, _ = np.histogram(lengths, bins=bins)
    total = counts.sum()
    if total == 0:
        return None
    return counts / total


def js_distance(p, q):
    return float(jensenshannon(p, q, base=2))


def quantize_to_patches(seq, patch_size):
    """Round a per-frame label sequence to patch boundaries: each patch
    (patch_size consecutive frames, last one possibly shorter) is assigned
    the majority label within it. This is the best segmentation any
    one-patch-resolution method could produce."""
    out = np.empty_like(seq)
    for start in range(0, len(seq), patch_size):
        end = min(start + patch_size, len(seq))
        block = seq[start:end]
        counts = np.bincount(block.astype(int))
        majority = np.argmax(counts)
        out[start:end] = majority
    return out


def run_dataset(dataset, epoch, out_root, seed=RNG_SEED):
    cfg = DATASET_DEFAULTS[dataset]
    ckpt = Path(f"models/{dataset}/epoch-{epoch}.model")
    features_path = Path(f"data/{dataset}/features")
    gt_path = Path(f"data/{dataset}/groundTruth")
    mapping_file = Path(f"data/{dataset}/mapping/mapping.txt")

    if not ckpt.exists():
        print(f"[{dataset}] no checkpoint at {ckpt}, skipping")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_actions = get_num_actions(gt_path)
    activities = read_mapping_file(mapping_file)  # {idx: name}

    model = SMQModel(
        in_channels=cfg["num_features"], filters=FILTERS, num_layers=NUM_LAYERS,
        latent_dim=LATENT_DIM, num_actions=num_actions, num_joints=cfg["num_joints"],
        num_person=cfg["num_person"], patch_size=cfg["patch_size"],
    )
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()

    print(f"[{dataset}] running inference with {ckpt} ...")
    with torch.no_grad():
        _, _, _, gt_all, prediction_all = evaluate_local_hungarian(
            model=model, features_path=features_path, gt_path=gt_path,
            mapping_file=mapping_file, epoch=epoch, device=device, verbose=False,
        )
        pr2gt = global_mapping(gt_all, prediction_all)
    prediction_mapped = [np.vectorize(pr2gt.get)(p) for p in prediction_all]

    patch_size = cfg["patch_size"]
    quantized_gt_all = [quantize_to_patches(np.asarray(seq), patch_size) for seq in gt_all]

    # --- Per-(video, activity) segment length lists ---
    # gt_by_va[activity][video_idx] = [lengths]; same for pred_by_va, quant_by_va.
    gt_by_va = defaultdict(lambda: defaultdict(list))
    pred_by_va = defaultdict(lambda: defaultdict(list))
    quant_by_va = defaultdict(lambda: defaultdict(list))
    gt_pooled_by_a = defaultdict(list)     # activity -> pooled GT lengths (all videos)
    pred_pooled_by_a = defaultdict(list)   # activity -> pooled predicted lengths (all videos)
    quant_pooled_by_a = defaultdict(list)  # activity -> pooled quantized-GT lengths (all videos)
    all_gt_lengths = []

    for v_idx, gt_seq in enumerate(gt_all):
        for label, length in labeled_segments(gt_seq):
            gt_by_va[label][v_idx].append(length)
            gt_pooled_by_a[label].append(length)
            all_gt_lengths.append(length)

    for v_idx, pred_seq in enumerate(prediction_mapped):
        for label, length in labeled_segments(pred_seq):
            pred_by_va[label][v_idx].append(length)
            pred_pooled_by_a[label].append(length)

    for v_idx, quant_seq in enumerate(quantized_gt_all):
        for label, length in labeled_segments(quant_seq):
            quant_by_va[label][v_idx].append(length)
            quant_pooled_by_a[label].append(length)

    bins = make_bins(all_gt_lengths)

    # --- Overlay histogram (patch-normalised, same axes as Part A) ---
    pred_lengths_frames_flat = [l for a in pred_pooled_by_a.values() for l in a]
    gt_patches = np.array(all_gt_lengths) / patch_size
    pred_patches = np.array(pred_lengths_frames_flat) / patch_size
    out_dir = out_root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    lo = min(gt_patches.min(), pred_patches.min())
    hi = max(gt_patches.max(), pred_patches.max())
    hbins = np.logspace(np.log10(lo), np.log10(hi), 40)
    ax.hist(gt_patches, bins=hbins, alpha=0.6, density=True, label="Ground truth")
    ax.hist(pred_patches, bins=hbins, alpha=0.6, density=True, label="SMQ prediction (post global-Hungarian)")
    ax.set_xscale("log")
    ax.set_xlabel("Segment length (patches)")
    ax.set_ylabel("Density")
    ax.set_title(f"{dataset}: GT vs. predicted segment length (patch_size={patch_size} frames)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "segment_length_overlay.svg")
    plt.close(fig)

    # --- Per-activity JS distance, averaged over videos ---
    rng = random.Random(seed)
    per_activity_rows = []

    for label in sorted(set(list(gt_by_va.keys()) + list(pred_by_va.keys()))):
        video_ids = sorted(set(gt_by_va[label].keys()) | set(pred_by_va[label].keys()))
        per_video_jsd = []
        n_skipped_no_pred = 0
        for v in video_ids:
            gt_lengths_v = gt_by_va[label].get(v, [])
            pred_lengths_v = pred_by_va[label].get(v, [])
            if not gt_lengths_v or not pred_lengths_v:
                n_skipped_no_pred += 1
                continue
            p = hist_prob(gt_lengths_v, bins)
            q = hist_prob(pred_lengths_v, bins)
            if p is None or q is None:
                continue
            per_video_jsd.append(js_distance(p, q))

        if not per_video_jsd:
            continue
        activity_jsd = float(np.mean(per_video_jsd))
        frame_weight = sum(gt_pooled_by_a[label])

        # --- Bootstrap null floor for this activity (sparsity null) ---
        pooled = gt_pooled_by_a[label]
        floor_vals = []
        if len(pooled) >= 2:
            for _ in range(N_BOOTSTRAP):
                shuffled = pooled[:]
                rng.shuffle(shuffled)
                mid = len(shuffled) // 2
                half_a, half_b = shuffled[:mid], shuffled[mid:]
                pa, pb = hist_prob(half_a, bins), hist_prob(half_b, bins)
                if pa is None or pb is None:
                    continue
                floor_vals.append(js_distance(pa, pb))
        null_floor = float(np.mean(floor_vals)) if floor_vals else float("nan")

        # --- Quantization ceiling: GT vs. patch-rounded GT, same per-video
        # averaging as the real (GT vs. predicted) JSD above ---
        ceiling_video_jsd = []
        for v in video_ids:
            gt_lengths_v = gt_by_va[label].get(v, [])
            quant_lengths_v = quant_by_va[label].get(v, [])
            if not gt_lengths_v or not quant_lengths_v:
                continue
            p = hist_prob(gt_lengths_v, bins)
            q = hist_prob(quant_lengths_v, bins)
            if p is None or q is None:
                continue
            ceiling_video_jsd.append(js_distance(p, q))
        ceiling_jsd = float(np.mean(ceiling_video_jsd)) if ceiling_video_jsd else float("nan")

        # --- Bin-free cross-check: Wasserstein-1, pooled raw lengths ---
        gt_pool = gt_pooled_by_a[label]
        pred_pool = pred_pooled_by_a.get(label, [])
        w1 = float(wasserstein_distance(gt_pool, pred_pool)) if gt_pool and pred_pool else float("nan")
        mean_gt_len = float(np.mean(gt_pool)) if gt_pool else float("nan")
        w1_norm = w1 / mean_gt_len if gt_pool and not np.isnan(w1) and mean_gt_len > 0 else float("nan")

        per_activity_rows.append({
            "activity": activities.get(label, str(label)),
            "n_videos_matched": len(per_video_jsd),
            "n_videos_no_predicted_segment": n_skipped_no_pred,
            "n_gt_segments": len(pooled),
            "frame_weight": frame_weight,
            "jsd_x100": round(activity_jsd * 100, 2),
            "null_floor_x100": round(null_floor * 100, 2) if not np.isnan(null_floor) else "",
            "jsd_minus_floor_x100": round((activity_jsd - null_floor) * 100, 2) if not np.isnan(null_floor) else "",
            "quantization_ceiling_x100": round(ceiling_jsd * 100, 2) if not np.isnan(ceiling_jsd) else "",
            "jsd_relative_to_ceiling_x100": round((activity_jsd - ceiling_jsd) * 100, 2) if not np.isnan(ceiling_jsd) else "",
            "wasserstein1_frames": round(w1, 2) if not np.isnan(w1) else "",
            "wasserstein1_normalized": round(w1_norm, 4) if not np.isnan(w1_norm) else "",
        })

    if not per_activity_rows:
        print(f"[{dataset}] no activities with both GT and predicted segments, skipping")
        return None

    total_frames = sum(r["frame_weight"] for r in per_activity_rows)

    def frame_weighted(field):
        vals = [(r["frame_weight"], r[field]) for r in per_activity_rows if r[field] != ""]
        if not vals:
            return float("nan")
        return sum(w * v for w, v in vals) / sum(w for w, v in vals)

    dataset_jsd = frame_weighted("jsd_x100")
    dataset_floor = frame_weighted("null_floor_x100")
    dataset_jsd_minus_floor = frame_weighted("jsd_minus_floor_x100")
    dataset_ceiling = frame_weighted("quantization_ceiling_x100")
    dataset_jsd_vs_ceiling = frame_weighted("jsd_relative_to_ceiling_x100")
    dataset_w1 = frame_weighted("wasserstein1_frames")
    dataset_w1_norm = frame_weighted("wasserstein1_normalized")

    # --- Interpret the raw-vs-ceiling decomposition ---
    # "Approximately equal" judged against the sparsity null floor's scale:
    # if the raw-ceiling gap is within ~1 null-floor-width, there's no
    # detectable headroom above what patch size alone explains.
    if not np.isnan(dataset_jsd_vs_ceiling) and not np.isnan(dataset_floor):
        if abs(dataset_jsd_vs_ceiling) <= dataset_floor:
            decomposition_verdict = "predicted JSD ~= quantization ceiling: duration error fully explained by patch size (no headroom for a duration-aware decoder at fixed patch size)"
        else:
            decomposition_verdict = "predicted JSD >> quantization ceiling: real headroom exists beyond what patch size alone explains"
    else:
        decomposition_verdict = "insufficient data to judge"

    # --- Ranking agreement between JSD-vs-floor and Wasserstein-1 ---
    ranked_by_jsd = sorted(per_activity_rows, key=lambda r: r["jsd_minus_floor_x100"] if r["jsd_minus_floor_x100"] != "" else 1e9)
    ranked_by_w1 = sorted(per_activity_rows, key=lambda r: r["wasserstein1_frames"] if r["wasserstein1_frames"] != "" else 1e9)
    order_jsd = [r["activity"] for r in ranked_by_jsd]
    order_w1 = [r["activity"] for r in ranked_by_w1]
    orderings_agree = order_jsd == order_w1

    # --- Write per-activity CSV ---
    fieldnames = list(per_activity_rows[0].keys())
    with open(out_dir / "jsd_per_activity.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_activity_rows:
            writer.writerow(row)

    summary = (
        f"dataset: {dataset}\n"
        f"checkpoint: {ckpt}\n"
        f"bin_edges_frames: {list(bins)}\n"
        f"n_bootstrap: {N_BOOTSTRAP}\n"
        f"--- dataset-level (frame-weighted across activities) ---\n"
        f"jsd_x100 (raw): {dataset_jsd:.2f}\n"
        f"null_floor_x100 (sparsity): {dataset_floor:.2f}\n"
        f"jsd_minus_floor_x100: {dataset_jsd_minus_floor:.2f}\n"
        f"quantization_ceiling_x100: {dataset_ceiling:.2f}\n"
        f"jsd_relative_to_ceiling_x100 (raw - ceiling): {dataset_jsd_vs_ceiling:.2f}\n"
        f"DECOMPOSITION VERDICT: {decomposition_verdict}\n"
        f"wasserstein1_frames (frame-weighted, raw): {dataset_w1:.2f}\n"
        f"wasserstein1_normalized (frame-weighted, /mean GT length): {dataset_w1_norm:.4f}\n"
        f"ranking by (jsd - floor), best to worst: {order_jsd}\n"
        f"ranking by wasserstein-1, best to worst: {order_w1}\n"
        f"orderings_agree: {orderings_agree}\n"
    )
    (out_dir / "summary.txt").write_text(summary)
    print(summary)

    return {
        "dataset": dataset,
        "checkpoint": str(ckpt),
        "jsd_x100": round(dataset_jsd, 2),
        "null_floor_x100": round(dataset_floor, 2),
        "jsd_minus_floor_x100": round(dataset_jsd_minus_floor, 2),
        "quantization_ceiling_x100": round(dataset_ceiling, 2),
        "jsd_relative_to_ceiling_x100": round(dataset_jsd_vs_ceiling, 2),
        "decomposition_verdict": decomposition_verdict,
        "wasserstein1_frames": round(dataset_w1, 2),
        "wasserstein1_normalized": round(dataset_w1_norm, 4),
        "orderings_agree_within_dataset": orderings_agree,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["hugadb", "lara", "babel1"])
    parser.add_argument("--epoch", type=int, default=30)
    args = parser.parse_args()

    out_root = Path("figures/e1_part_b")
    rows = []
    for dataset in args.datasets:
        row = run_dataset(dataset, args.epoch, out_root)
        if row is not None:
            rows.append(row)

    if rows:
        csv_path = out_root / "summary.csv"
        existing = {}
        if csv_path.exists():
            with open(csv_path, newline="") as f:
                for r in csv.DictReader(f):
                    existing[r["dataset"]] = r
        for row in rows:
            existing[row["dataset"]] = row

        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for dataset in sorted(existing):
                writer.writerow(existing[dataset])
        print(f"Wrote {csv_path}")

        # Cross-dataset ordering comparison. Raw Wasserstein-1 is NOT
        # scale-invariant (datasets with longer mean GT segments score
        # worse mechanically), so the normalized version is used here --
        # raw is still reported per-dataset for reference.
        by_jsd = sorted(rows, key=lambda r: r["jsd_minus_floor_x100"])
        by_w1_norm = sorted(rows, key=lambda r: r["wasserstein1_normalized"])
        print("Cross-dataset ranking by (jsd - floor):", [r["dataset"] for r in by_jsd])
        print("Cross-dataset ranking by normalized wasserstein-1:", [r["dataset"] for r in by_w1_norm])
        print("Cross-dataset orderings agree:", [r["dataset"] for r in by_jsd] == [r["dataset"] for r in by_w1_norm])
        print("Reminder: quantization penalizes short-segment datasets hardest (e.g. BABEL1, median 1.2 patches) -- "
              "check any apparent re-ranking against each dataset's quantization_ceiling_x100 before attributing it to sparsity.")
