import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.smq import SMQModel
from src.model.utils import get_num_actions
from src.model.eval_utils import evaluate_local_hungarian

print("init")

# Per-dataset architecture, matching main.py's get_dataset_defaults()
DATASET_DEFAULTS = {
    "hugadb": dict(num_features=6, num_joints=6, num_person=1, patch_size=60),
    "lara": dict(num_features=6, num_joints=22, num_person=1, patch_size=50),
    "babel1": dict(num_features=3, num_joints=25, num_person=1, patch_size=30),
    "babel2": dict(num_features=3, num_joints=25, num_person=1, patch_size=30),
    "babel3": dict(num_features=3, num_joints=25, num_person=1, patch_size=30),
}

parser = argparse.ArgumentParser(description="Compare GT vs. predicted segment-length histograms")
parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS.keys()), default="hugadb")
parser.add_argument("--epoch", type=int, default=30, help="Checkpoint epoch to load.")
args = parser.parse_args()

DATASET = args.dataset
CKPT = Path(f"models/{DATASET}/epoch-{args.epoch}.model")
FEATURES_PATH = Path(f"data/{DATASET}/features")
GT_PATH = Path(f"data/{DATASET}/groundTruth")
MAPPING_FILE = Path(f"data/{DATASET}/mapping/mapping.txt")

# Architecture must match training (main.py's dataset defaults + model defaults)
cfg = DATASET_DEFAULTS[DATASET]
IN_CHANNELS, NUM_JOINTS, NUM_PERSON, PATCH_SIZE = cfg["num_features"], cfg["num_joints"], cfg["num_person"], cfg["patch_size"]
FILTERS, NUM_LAYERS, LATENT_DIM = 128, 3, 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def segment_lengths(seq):
    """Run-lengths of consecutive identical labels in a 1D sequence."""
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


# --- Ground truth segment lengths (from label text files) ---
gt_lengths = []
for file in sorted(GT_PATH.glob("*.txt")):
    with open(file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    if lines:
        gt_lengths.extend(segment_lengths(lines))

# --- Run inference with the trained model to get predicted segmentations ---
num_actions = get_num_actions(GT_PATH)

model = SMQModel(
    in_channels=IN_CHANNELS, filters=FILTERS, num_layers=NUM_LAYERS, latent_dim=LATENT_DIM,
    num_actions=num_actions, num_joints=NUM_JOINTS, num_person=NUM_PERSON, patch_size=PATCH_SIZE,
)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.to(device)
model.eval()

print(f"Running inference with {CKPT} ...")
with torch.no_grad():
    _, _, _, gt_all, prediction_all = evaluate_local_hungarian(
        model=model,
        features_path=FEATURES_PATH,
        gt_path=GT_PATH,
        mapping_file=MAPPING_FILE,
        epoch=args.epoch,
        device=device,
        verbose=False,
    )

# Segment-length distribution is invariant to the GT<->cluster-id relabeling
# (Hungarian mapping is one-to-one), so raw predicted indices are used as-is.
pred_lengths = []
for prediction in prediction_all:
    pred_lengths.extend(segment_lengths(prediction))

print(f"GT segments: {len(gt_lengths)}  |  Predicted segments: {len(pred_lengths)}")

# --- Compare histograms ---
max_len = max(max(gt_lengths), max(pred_lengths))
bins = np.logspace(0, np.log10(max_len), 40)

gt_hist, _ = np.histogram(gt_lengths, bins=bins, density=False)
pred_hist, _ = np.histogram(pred_lengths, bins=bins, density=False)

gt_prob = gt_hist / gt_hist.sum()
pred_prob = pred_hist / pred_hist.sum()

js_distance = jensenshannon(gt_prob, pred_prob, base=2)
js_divergence = js_distance ** 2
print(f"Jensen-Shannon divergence (GT vs. prediction segment-length distributions): {js_divergence:.4f}")
print(f"Jensen-Shannon distance: {js_distance:.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(gt_lengths, bins=bins, alpha=0.6, density=True, label="Ground truth")
ax.hist(pred_lengths, bins=bins, alpha=0.6, density=True, label="SMQ prediction")
ax.set_xscale("log")
ax.set_xlabel("Segment length (frames)")
ax.set_ylabel("Density")
ax.set_title(f"{DATASET}: segment-length histogram — JSD = {js_divergence:.4f}")
ax.legend()
fig.tight_layout()

out_path = Path("figures") / f"{DATASET}_segment_length_jsd.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150)
print(f"Saved plot to {out_path}")

plt.show()
