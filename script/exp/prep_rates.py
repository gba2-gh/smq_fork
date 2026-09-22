"""
T4 Sweep R data: temporally downsample features AND ground truth by the same
stride, so training and evaluation both happen at the reduced frame rate
(main.py's eval path ignores --sample_rate, so downsampling has to be physical).

Same method as the earlier script/e4_prepare_sweeps.py (plain stride, no
anti-aliasing filter), extended to all three datasets:
    data/exp_rates/<ds>_r<r>/<ds>/{features,groundTruth,mapping}
"""
from pathlib import Path

import numpy as np

RATES = {"hugadb": [2, 4], "lara": [2, 5], "babel1": [2, 3]}


def downsample(ds, r):
    src = Path("data") / ds
    dst = Path("data/exp_rates") / f"{ds}_r{r}" / ds
    for sub in ("features", "groundTruth", "mapping"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    (dst / "mapping" / "mapping.txt").write_text((src / "mapping" / "mapping.txt").read_text())
    n = 0
    for f in sorted((src / "features").glob("*.npy")):
        out_f = dst / "features" / f.name
        out_g = dst / "groundTruth" / f.name.replace(".npy", ".txt")
        if out_f.exists() and out_g.exists():
            n += 1
            continue
        arr = np.load(f)
        np.save(out_f, np.ascontiguousarray(arr[:, ::r]))
        lines = (src / "groundTruth" / f.name.replace(".npy", ".txt")).read_text().splitlines()
        out_g.write_text("\n".join(lines[::r]))
        n += 1
    print(f"{ds} r={r}: {n} sequences -> {dst}", flush=True)


if __name__ == "__main__":
    for ds, rates in RATES.items():
        for r in rates:
            downsample(ds, r)
