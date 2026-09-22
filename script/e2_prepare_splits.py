"""
E2 -- Subject holdout: split preparation (experiment.md).

Generates N reproducible held-out-subject splits per dataset and
materialises train/test data directories (mirroring the
data/<dataset>/{features,groundTruth,mapping} layout main.py expects),
under data/e2_splits/<dataset>_split{i}_{train,test}/<dataset>/...

NOTE on LARa subject count: the brief specifies 14 subjects (11 train /
3 test), but our actual downloaded data has 16 distinct subjects (01-16,
each with 14-30 sequences -- confirmed, not a partial/corrupt subset).
Adapted to 13 train / 3 test, keeping the brief's intended test-set size.
"""

import argparse
import random
import re
import shutil
from pathlib import Path

SUBJECT_PATTERNS = {
    "hugadb": re.compile(r"HuGaDB_v2_various_(\d+)_\d+\.npy"),
    "lara": re.compile(r"L\d+_S(\d+)_R\d+\.npy"),
}

# (n_test_subjects,) -- n_train is inferred as (total_subjects - n_test)
SPLIT_CONFIG = {
    "hugadb": dict(n_test=4),   # brief: 14 train / 4 test, 18 subjects -- matches our data exactly
    "lara": dict(n_test=3),     # brief: 11 train / 3 test assuming 14 subjects; our data has 16 -> 13 train / 3 test
}


def get_subjects_and_files(dataset):
    features_dir = Path(f"data/{dataset}/features")
    pattern = SUBJECT_PATTERNS[dataset]
    subject_to_files = {}
    for f in sorted(features_dir.glob("*.npy")):
        m = pattern.match(f.name)
        if not m:
            raise ValueError(f"Could not parse subject id from {f.name}")
        subject_to_files.setdefault(m.group(1), []).append(f.name)
    return subject_to_files


def materialize(dataset, files, split_name, out_root):
    src_features = Path(f"data/{dataset}/features")
    src_gt = Path(f"data/{dataset}/groundTruth")
    src_mapping = Path(f"data/{dataset}/mapping")

    dst_root = out_root / split_name / dataset
    dst_features = dst_root / "features"
    dst_gt = dst_root / "groundTruth"
    dst_mapping = dst_root / "mapping"
    for d in (dst_features, dst_gt, dst_mapping):
        d.mkdir(parents=True, exist_ok=True)

    shutil.copy(src_mapping / "mapping.txt", dst_mapping / "mapping.txt")
    for fname in files:
        shutil.copy(src_features / fname, dst_features / fname)
        txt_name = fname.replace(".npy", ".txt")
        shutil.copy(src_gt / txt_name, dst_gt / txt_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["hugadb", "lara"])
    parser.add_argument("--n_splits", type=int, default=4)
    parser.add_argument("--out_root", type=Path, default=Path("data/e2_splits"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_lines = []

    for dataset in args.datasets:
        subject_to_files = get_subjects_and_files(dataset)
        subjects = sorted(subject_to_files.keys())
        n_test = SPLIT_CONFIG[dataset]["n_test"]
        n_train = len(subjects) - n_test
        print(f"[{dataset}] {len(subjects)} subjects total -> {n_train} train / {n_test} test per split")

        rng = random.Random(args.seed)
        for split_i in range(args.n_splits):
            shuffled = subjects[:]
            rng.shuffle(shuffled)
            test_subjects = sorted(shuffled[:n_test])
            train_subjects = sorted(shuffled[n_test:])

            train_files = [f for s in train_subjects for f in subject_to_files[s]]
            test_files = [f for s in test_subjects for f in subject_to_files[s]]

            materialize(dataset, train_files, f"{dataset}_split{split_i}_train", args.out_root)
            materialize(dataset, test_files, f"{dataset}_split{split_i}_test", args.out_root)

            line = f"{dataset} split{split_i}: train_subjects={train_subjects} test_subjects={test_subjects}"
            print(line)
            manifest_lines.append(line)

    (args.out_root / "manifest.txt").write_text("\n".join(manifest_lines))
    print(f"Wrote {args.out_root / 'manifest.txt'}")


if __name__ == "__main__":
    main()
