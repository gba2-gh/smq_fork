"""
E5.0 -- Feasibility gate for the segment-level code audit (experiment.md).

Must run and be reported BEFORE any E5.1/E5.2/E5.3 code is built -- this is
a hard gate, not a formality.

Part 1: distribution of codes (patch-tokens) per ground-truth segment, using
E0's checkpoint and the model's own patchification (patch_id = frame // patch_size).
Report median, quartiles, and fraction of segments with <5, <10, <20 codes.

Part 2: null floor for the audit itself. Build per-instance (= per GT
segment) code-usage histograms from the RAW code vocabulary (not
Hungarian-mapped -- the audit is about what the codes encode, independent
of any label correspondence). Form cross-recording, cross-subject D
(same label) and E (different label) pairs, compute the real D-E cosine
gap, then repeat under label-shuffling to get the null distribution of
that gap -- this tells us the smallest gap the audit could detect at this
sparsity, before we invest in the full E5.1/E5.2/E5.3 apparatus.
"""

import argparse
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

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

N_NULL_RESAMPLES = 200
MAX_PAIRS_PER_BUCKET = 8000
RNG_SEED = 42


def parse_subject(dataset, filename):
    return SUBJECT_PATTERNS[dataset].match(filename).group(1)


def labeled_segments_with_bounds(seq):
    """Run-length encode -> list of (label, start, end) frame bounds."""
    out = []
    label = seq[0]
    start = 0
    for i in range(1, len(seq)):
        if seq[i] != label:
            out.append((label, start, i))
            label = seq[i]
            start = i
    out.append((label, start, len(seq)))
    return out


def code_hist(code_array, start, end, n_codes):
    counts = np.bincount(code_array[start:end].astype(int), minlength=n_codes).astype(float)
    total = counts.sum()
    return counts / total if total > 0 else None


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def run_dataset(dataset, epoch, rng):
    cfg = DATASET_DEFAULTS[dataset]
    ckpt = Path(f"models/{dataset}/epoch-{epoch}.model")
    features_path = Path(f"data/{dataset}/features")
    gt_path = Path(f"data/{dataset}/groundTruth")
    mapping_file = Path(f"data/{dataset}/mapping/mapping.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_actions = get_num_actions(gt_path)
    activities = read_mapping_file(mapping_file)
    actions_dict = {v: k for k, v in activities.items()}
    patch_size = cfg["patch_size"]

    model = SMQModel(
        in_channels=cfg["num_features"], filters=FILTERS, num_layers=NUM_LAYERS,
        latent_dim=LATENT_DIM, num_actions=num_actions, num_joints=cfg["num_joints"],
        num_person=cfg["num_person"], patch_size=patch_size,
    )
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()

    vids = sorted(p.name for p in features_path.glob("*.npy"))
    print(f"[{dataset}] extracting per-segment code counts for {len(vids)} sequences ...")

    codes_per_segment = []
    instances = []  # dicts: recording, subject, label, hist

    with torch.no_grad():
        for vid in vids:
            gt_array, code_array = get_framewise_predictions(
                model, vid, features_path, gt_path, actions_dict, device
            )
            subject = parse_subject(dataset, vid)
            for label, start, end in labeled_segments_with_bounds(gt_array):
                patch_ids = set(range(start // patch_size, (end - 1) // patch_size + 1))
                codes_per_segment.append(len(patch_ids))

                h = code_hist(code_array, start, end, num_actions)
                if h is not None:
                    instances.append(dict(recording=vid, subject=subject, label=int(label), hist=h))

    codes_per_segment = np.array(codes_per_segment)
    q = np.percentile(codes_per_segment, [25, 50, 75])
    frac_lt5 = float((codes_per_segment < 5).mean())
    frac_lt10 = float((codes_per_segment < 10).mean())
    frac_lt20 = float((codes_per_segment < 20).mean())

    print(f"[{dataset}] codes/segment: median={q[1]:.2f} q25={q[0]:.2f} q75={q[2]:.2f} "
          f"frac<5={frac_lt5:.3f} frac<10={frac_lt10:.3f} frac<20={frac_lt20:.3f}")

    gate_pass = q[1] >= 10

    # --- Null floor for the D-vs-E gap ---
    def compute_gap(labels_for_instances):
        # Build cross-recording, cross-subject candidate pairs, bucketed by
        # whether the (possibly shuffled) labels match.
        n = len(instances)
        d_pairs, e_pairs = [], []
        idx = list(range(n))
        rng.shuffle(idx)
        # Random pairing scan (not exhaustive O(n^2)) -- sample candidate
        # pairs until buckets are full or we exhaust attempts.
        attempts = 0
        max_attempts = MAX_PAIRS_PER_BUCKET * 40
        i_ptr = 0
        while (len(d_pairs) < MAX_PAIRS_PER_BUCKET or len(e_pairs) < MAX_PAIRS_PER_BUCKET) and attempts < max_attempts:
            attempts += 1
            a = idx[rng.randrange(n)]
            b = idx[rng.randrange(n)]
            if a == b:
                continue
            ia, ib = instances[a], instances[b]
            if ia["recording"] == ib["recording"] or ia["subject"] == ib["subject"]:
                continue
            same_label = labels_for_instances[a] == labels_for_instances[b]
            if same_label and len(d_pairs) < MAX_PAIRS_PER_BUCKET:
                d_pairs.append((a, b))
            elif not same_label and len(e_pairs) < MAX_PAIRS_PER_BUCKET:
                e_pairs.append((a, b))

        if not d_pairs or not e_pairs:
            return None
        d_sims = [cosine_sim(instances[a]["hist"], instances[b]["hist"]) for a, b in d_pairs]
        e_sims = [cosine_sim(instances[a]["hist"], instances[b]["hist"]) for a, b in e_pairs]
        return float(np.mean(d_sims) - np.mean(e_sims)), len(d_pairs), len(e_pairs)

    real_labels = [inst["label"] for inst in instances]
    real_result = compute_gap(real_labels)

    null_gaps = []
    for _ in range(N_NULL_RESAMPLES):
        shuffled = real_labels[:]
        rng.shuffle(shuffled)
        r = compute_gap(shuffled)
        if r is not None:
            null_gaps.append(r[0])
    null_gaps = np.array(null_gaps)

    print(f"[{dataset}] n_instances={len(instances)} real_gap={real_result[0]:.4f} "
          f"(n_D={real_result[1]}, n_E={real_result[2]}) "
          f"null_gap: mean={null_gaps.mean():.4f} std={null_gaps.std():.4f}")

    return dict(
        dataset=dataset, n_segments=len(codes_per_segment),
        codes_per_segment_median=q[1], codes_per_segment_q25=q[0], codes_per_segment_q75=q[2],
        frac_lt5=frac_lt5, frac_lt10=frac_lt10, frac_lt20=frac_lt20,
        gate_pass=gate_pass, n_instances=len(instances),
        real_gap=real_result[0], null_gap_mean=float(null_gaps.mean()), null_gap_std=float(null_gaps.std()),
        real_gap_exceeds_null_by_std=(real_result[0] - null_gaps.mean()) / null_gaps.std() if null_gaps.std() > 0 else float("nan"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["hugadb", "lara"])
    parser.add_argument("--epoch", type=int, default=30)
    args = parser.parse_args()

    rng = random.Random(RNG_SEED)
    out_dir = Path("figures/e5_0_gate")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in args.datasets:
        r = run_dataset(dataset, args.epoch, rng)
        rows.append(r)

    import csv
    with open(out_dir / "gate_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {out_dir / 'gate_summary.csv'}")
