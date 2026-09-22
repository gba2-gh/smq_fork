"""
E4 -- Prepare downsampled HuGaDB data for Sweeps A and B (experiment.md).

main.py's eval path (get_framewise_predictions) loads features directly at
native rate -- it does not apply --sample_rate at eval time. To keep
train/eval frequency consistent, we physically downsample features AND
groundTruth (same stride) into their own data directories, then train/eval
with --sample_rate=1 (the default) against those.

Native rate (sample_rate=1) needs no new directory -- data/hugadb/ already
is that condition, and models/hugadb/epoch-30.model (E0's seed-1 run) is
reused directly as the shared A1==B1 condition.

Produces, per downsample rate r in {2, 4}:
    data/e4_sweeps/hugadb_rate{r}/hugadb/{features,groundTruth,mapping}
"""

from pathlib import Path

import numpy as np

NATIVE_FPS = 60  # HuGaDB documented capture rate (Stage 0 finding)
RATES = [2, 4]


def downsample_dataset(rate, out_root):
    src_features = Path("data/hugadb/features")
    src_gt = Path("data/hugadb/groundTruth")
    src_mapping = Path("data/hugadb/mapping")

    dst_root = out_root / f"hugadb_rate{rate}" / "hugadb"
    dst_features = dst_root / "features"
    dst_gt = dst_root / "groundTruth"
    dst_mapping = dst_root / "mapping"
    for d in (dst_features, dst_gt, dst_mapping):
        d.mkdir(parents=True, exist_ok=True)

    (dst_mapping / "mapping.txt").write_text((src_mapping / "mapping.txt").read_text())

    n = 0
    for f in sorted(src_features.glob("*.npy")):
        arr = np.load(f)  # (C, T, V, M)
        arr_ds = arr[:, ::rate, :, :]
        np.save(dst_features / f.name, arr_ds)

        gt_file = src_gt / f.name.replace(".npy", ".txt")
        lines = gt_file.read_text().splitlines()
        lines_ds = lines[::rate]
        (dst_gt / gt_file.name).write_text("\n".join(lines_ds))
        n += 1

    print(f"rate={rate}: downsampled {n} sequences -> {dst_root} (fps={NATIVE_FPS/rate:.1f})")


if __name__ == "__main__":
    out_root = Path("data/e4_sweeps")
    for rate in RATES:
        downsample_dataset(rate, out_root)
