"""Ground-truth action labels for Task 2's segmentation SCORING only.
Intentionally separate from `artifacts.Loader` (Task 1's interface, which
does not expose labels at all) -- nothing in `contextual.py` (training) or
`run_task2.py`'s job-selection/checkpoint-selection paths imports this
module. Only `ot_arms.py`'s final scoring step does.

Loads HuGaDB ground truth the same way `script/action_transport/data.py`
does (`script.exp2round.q1q2.core.load_dataset`), keyed by recording name so
it can be looked up against Task 1's `loader.recording_names()`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.exp2round.q1q2 import core as q1q2core

_CACHE: dict | None = None


def load_hugadb_gts() -> dict[str, np.ndarray]:
    """{recording_name: int array[frame_count]} ground-truth action id per
    frame, for every HuGaDB recording. Cached per-process (read-only)."""
    global _CACHE
    if _CACHE is None:
        ds = q1q2core.load_dataset("hugadb", representation="latent")
        _CACHE = {name: np.asarray(gt) for name, gt in zip(ds.names, ds.gts)}
    return _CACHE


def gts_for(names: list[str]) -> dict[str, np.ndarray]:
    all_gts = load_hugadb_gts()
    missing = [n for n in names if n not in all_gts]
    if missing:
        raise KeyError(f"{len(missing)} recording(s) have no ground truth (e.g. {missing[:5]}); "
                       f"qic Loader and q1q2 core.load_dataset recording-name sets disagree")
    return {n: all_gts[n] for n in names}
