"""Dataset specifications and label loading for the cross-dataset raw K-means
test (docs/plans/week5/RAW_KMEANS_CROSS_DATASET/). Labels are read only for
scoring and label-aware diagnostics, never for tokenization, training or
checkpoint selection."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    fps: int
    window: int  # token window in frames (about 0.25 s)
    num_states: int  # number of classes, including background
    smoke_cap: int  # recordings used by --smoke
    has_subject: bool

    @property
    def token_seconds(self) -> float:
        return self.window / self.fps


SPECS = {
    "hugadb": DatasetSpec("hugadb", 60, 15, 10, 12, True),
    "lara": DatasetSpec("lara", 50, 12, 8, 6, True),
    "babel1": DatasetSpec("babel1", 30, 8, 5, 60, False),
    "babel2": DatasetSpec("babel2", 30, 8, 5, 60, False),
    "babel3": DatasetSpec("babel3", 30, 8, 5, 60, False),
}
DATASET_ORDER = ("hugadb", "babel1", "babel2", "babel3", "lara")  # cheap datasets first, LARa last


def features_dir(spec: DatasetSpec) -> Path:
    return ROOT / "data" / spec.name / "features"


def recording_names(spec: DatasetSpec, cap: int | None = None) -> list[str]:
    names = sorted(os.listdir(features_dir(spec)))
    return names[:cap] if cap else names


def frame_counts(spec: DatasetSpec, names: list[str]) -> dict[str, int]:
    return {n: int(np.load(features_dir(spec) / n, mmap_mode="r").shape[1]) for n in names}


def label_map(spec: DatasetSpec) -> dict[str, int]:
    path = ROOT / "data" / spec.name / "mapping" / "mapping.txt"
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            idx, label = line.strip().split(maxsplit=1)
            out[label] = int(idx)
    return out


def load_gts(spec: DatasetSpec, names: list[str]) -> dict[str, np.ndarray]:
    """Frame-level class ids from groundTruth/<stem>.txt through the dataset's mapping file."""
    mapping = label_map(spec)
    gts = {}
    for name in names:
        path = ROOT / "data" / spec.name / "groundTruth" / (Path(name).stem + ".txt")
        labels = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        unknown = sorted({lab for lab in labels if lab not in mapping})
        if unknown:
            raise KeyError(f"{spec.name}/{name}: labels {unknown[:3]} are not in the mapping file")
        gts[name] = np.asarray([mapping[lab] for lab in labels], dtype=np.int32)
    return gts


def subject_of(spec: DatasetSpec, name: str) -> str | None:
    if spec.name == "hugadb":
        m = re.search(r"_(\d{2})_\d{2}(?:\.npy)?$", name)
    elif spec.name == "lara":
        m = re.search(r"_(S\d+)_", name)
    else:
        return None
    if not m:
        raise ValueError(f"cannot parse {spec.name} subject from {name}")
    return m.group(1)
