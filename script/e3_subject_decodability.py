"""
E3 -- Is subject identity decodable from code usage? (experiment.md)

Uses existing single-seed checkpoints (models/<dataset>/epoch-30.model) --
NOT the formal 3-seed E0 set, which hasn't been run yet. Flagged, same
caveat as E1 Part B.

Per dataset (HuGaDB, LARa -- BABEL has no recoverable subject identity,
per Stage 0's finding):
  1. Extract per-frame RAW code indices (model.indices; no Hungarian
     remapping -- code identity is already consistent across sequences
     since they all come from the same trained codebook).
  2. Per sequence: normalised code-usage histogram (K bins, K = codebook
     size) and normalised GT action-label histogram (baseline features).
     Subject ID parsed from the filename.
  3. Logistic regression, subject ID <- code-usage histogram, 5-fold
     stratified CV over sequences (within-population classification, NOT
     leave-subject-out -- that's E2). Report accuracy vs. chance (1/n_subj).
  4. Identical classifier from the GT action-label histogram (baseline).
  5. Pairwise Jensen-Shannon divergence between per-subject (all-frames-
     aggregated) code-usage distributions -> subject x subject heatmap.

NOTE on the scale-normalisation control (brief's step 3): this is NOT run
here. It requires retraining the encoder on normalised input -- feeding
normalised skeletons through a model trained on raw (unnormalised) input
is out-of-distribution for that encoder and would not be a valid control.
Flagged explicitly rather than silently skipped or silently run invalidly.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.smq import SMQModel
from src.model.utils import get_num_actions
from src.model.eval_utils import get_framewise_predictions, read_mapping_file

DATASET_DEFAULTS = {
    "hugadb": dict(num_features=6, num_joints=6, num_person=1, patch_size=60),
    "lara": dict(num_features=6, num_joints=22, num_person=1, patch_size=50),
}
FILTERS, NUM_LAYERS, LATENT_DIM = 128, 3, 16

SUBJECT_PATTERNS = {
    "hugadb": re.compile(r"HuGaDB_v2_various_(\d+)_\d+\.npy"),
    "lara": re.compile(r"L\d+_S(\d+)_R\d+\.npy"),
}


def parse_subject(dataset, filename):
    m = SUBJECT_PATTERNS[dataset].match(filename)
    if not m:
        raise ValueError(f"Could not parse subject id from {filename}")
    return m.group(1)


def normalized_hist(indices, n_bins):
    counts = np.bincount(indices.astype(int), minlength=n_bins).astype(float)
    total = counts.sum()
    return counts / total if total > 0 else counts


def run_dataset(dataset, epoch, out_root):
    cfg = DATASET_DEFAULTS[dataset]
    ckpt = Path(f"models/{dataset}/epoch-{epoch}.model")
    features_path = Path(f"data/{dataset}/features")
    gt_path = Path(f"data/{dataset}/groundTruth")
    mapping_file = Path(f"data/{dataset}/mapping/mapping.txt")

    if not ckpt.exists():
        print(f"[{dataset}] no checkpoint at {ckpt}, skipping")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_actions = get_num_actions(gt_path)
    activities = read_mapping_file(mapping_file)
    actions_dict = {v: k for k, v in activities.items()}
    n_action_classes = len(activities)

    model = SMQModel(
        in_channels=cfg["num_features"], filters=FILTERS, num_layers=NUM_LAYERS,
        latent_dim=LATENT_DIM, num_actions=num_actions, num_joints=cfg["num_joints"],
        num_person=cfg["num_person"], patch_size=cfg["patch_size"],
    )
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()

    vids = sorted(p.name for p in features_path.glob("*.npy"))
    print(f"[{dataset}] extracting codes for {len(vids)} sequences ...")

    code_hists, action_hists, subjects = [], [], []
    subject_frame_codes = {}  # subject -> list of per-frame code arrays (for the heatmap)

    with torch.no_grad():
        for vid in vids:
            gt_array, code_array = get_framewise_predictions(
                model, vid, features_path, gt_path, actions_dict, device
            )
            subject = parse_subject(dataset, vid)
            code_hists.append(normalized_hist(code_array, num_actions))
            # mapping.txt is 1-indexed (labels 1..n_action_classes), so bin
            # count must be n_action_classes+1 (bin 0 always empty) to keep
            # a consistent histogram length regardless of which labels a
            # given sequence happens to contain.
            action_hists.append(normalized_hist(gt_array, n_action_classes + 1))
            subjects.append(subject)
            subject_frame_codes.setdefault(subject, []).append(code_array)

    code_hists = np.array(code_hists)
    action_hists = np.array(action_hists)
    subjects = np.array(subjects)
    n_subjects = len(set(subjects))
    chance = 1.0 / n_subjects

    print(f"[{dataset}] {n_subjects} subjects, chance level = {chance:.4f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=2000)

    code_scores = cross_val_score(clf, code_hists, subjects, cv=cv)
    action_scores = cross_val_score(clf, action_hists, subjects, cv=cv)

    out_dir = out_root / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Pairwise JS divergence between per-subject code-usage distributions ---
    subj_ids = sorted(set(subjects))
    subj_dist = {}
    for s in subj_ids:
        all_codes = np.concatenate(subject_frame_codes[s])
        subj_dist[s] = normalized_hist(all_codes, num_actions)

    n = len(subj_ids)
    jsd_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            jsd_matrix[i, j] = jensenshannon(subj_dist[subj_ids[i]], subj_dist[subj_ids[j]], base=2) ** 2

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(jsd_matrix, cmap="viridis")
    ax.set_xticks(range(n)); ax.set_xticklabels(subj_ids, rotation=90, fontsize=6)
    ax.set_yticks(range(n)); ax.set_yticklabels(subj_ids, fontsize=6)
    ax.set_title(f"{dataset}: pairwise JSD between per-subject code-usage distributions")
    fig.colorbar(im, ax=ax, label="JS divergence")
    fig.tight_layout()
    fig.savefig(out_dir / "subject_jsd_heatmap.svg")
    plt.close(fig)

    off_diag = jsd_matrix[~np.eye(n, dtype=bool)]

    summary = (
        f"dataset: {dataset}\n"
        f"checkpoint: {ckpt}\n"
        f"n_sequences: {len(vids)}\n"
        f"n_subjects: {n_subjects}\n"
        f"chance_level: {chance:.4f}\n"
        f"code_usage_classifier_accuracy: mean={code_scores.mean():.4f} std={code_scores.std():.4f} folds={list(np.round(code_scores, 4))}\n"
        f"action_label_baseline_accuracy: mean={action_scores.mean():.4f} std={action_scores.std():.4f} folds={list(np.round(action_scores, 4))}\n"
        f"gap_code_minus_action_baseline: {code_scores.mean() - action_scores.mean():.4f}\n"
        f"mean_pairwise_subject_jsd: {off_diag.mean():.4f}\n"
        f"max_pairwise_subject_jsd: {off_diag.max():.4f}\n"
        f"min_pairwise_subject_jsd: {off_diag.min():.4f}\n"
        f"\n"
        f"SCALE-NORMALISATION CONTROL: NOT RUN. Requires retraining the\n"
        f"encoder on normalised input (feeding normalised skeletons through\n"
        f"a model trained on raw input is out-of-distribution, not a valid\n"
        f"control). Flagging cost: 1 additional full training run for this\n"
        f"dataset (~10-15 min at current pace) plus a new preprocessing step\n"
        f"(divide joint coordinates by a body-intrinsic length).\n"
    )
    (out_dir / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["hugadb", "lara"])
    parser.add_argument("--epoch", type=int, default=30)
    args = parser.parse_args()

    out_root = Path("figures/e3_subject_decodability")
    for dataset in args.datasets:
        run_dataset(dataset, args.epoch, out_root)
