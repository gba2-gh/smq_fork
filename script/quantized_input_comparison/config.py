"""Fixed scope for Task 1 (see the spec in the driving instructions):
HuGaDB, 15-frame windows, 512 codes, epoch-30 VQ/FSQ checkpoints from
results/fsq_reconstruction/fsq_pilot_002/, frozen (including VQ EMA
buffers).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUN = ROOT / "results" / "fsq_reconstruction" / "fsq_pilot_002"
OUT_ROOT = ROOT / "results" / "quantized_input_comparison"

sys.path.insert(0, str(ROOT))

DATASET = "hugadb"
WINDOW = 15  # patch_size, DATASET_DEFAULTS["hugadb"]
NUM_FEATURES = 6
NUM_JOINTS = 6
NUM_PERSON = 1
CHECKPOINT_EPOCH = 30

NUM_CODES = 512
PCA_DIM = 64
KMEANS_SAMPLE_CAP = 10_000

# Bumped whenever the on-disk artifact layout or semantics change in a way
# that makes an older run's files unsafe to reuse under --resume (e.g. the
# rms.npy scalar-vs-vector change that qic_v1 predates). manifest_matches()
# rejects any manifest whose recorded schema_version differs from this one
# (including manifests written before this field existed, which read back as
# None != ARTIFACT_SCHEMA_VERSION) -- old runs simply fail resume-compatibility,
# they are never silently accepted or deleted.
ARTIFACT_SCHEMA_VERSION = 2

SEED = 111  # sampling / PCA / K-means / grouping / validation-split seed (spec: seed 111 throughout)

# Encoder receptive field (verified against src/model/ms_tcn.py: two
# MultiStageModel stages, each 3 DilatedResidualLayers with dilations
# 1,2,4 and kernel 3 -> per-stage radius 1+2+4=7, two stages -> radius 14,
# full receptive field 2*14+1=29 raw frames per output frame).
RECEPTIVE_RADIUS = 14
RECEPTIVE_FIELD = 2 * RECEPTIVE_RADIUS + 1  # 29
MAX_PATCH_SUPPORT = WINDOW + 2 * RECEPTIVE_RADIUS  # 43, informational only

FAMILIES = ("vq_native", "fsq_native", "vq_kmeans", "fsq_kmeans", "raw_kmeans")
POPULATIONS = ("raw", "vq_encoder", "fsq_encoder")  # standardizer+PCA(+KMeans) fit populations
FAMILY_POPULATION = dict(vq_native="vq_encoder", fsq_native="fsq_encoder",
                          vq_kmeans="vq_encoder", fsq_kmeans="fsq_encoder", raw_kmeans="raw")
FAMILY_KIND = dict(vq_native="native", fsq_native="native", vq_kmeans="kmeans",
                    fsq_kmeans="kmeans", raw_kmeans="kmeans")

NATIVE_DIM = WINDOW * (16 * NUM_JOINTS * NUM_PERSON)  # latent_dim(16) * num_joints * num_person, expect 1440


def checkpoint_path(arm: str) -> Path:
    return SOURCE_RUN / "checkpoints" / arm / f"epoch-{CHECKPOINT_EPOCH}.model"


@dataclass(frozen=True)
class PrepConfig:
    run_id: str
    dataset: str = DATASET
    window: int = WINDOW
    num_codes: int = NUM_CODES
    pca_dim: int = PCA_DIM
    kmeans_sample_cap: int = KMEANS_SAMPLE_CAP
    seed: int = SEED
    receptive_radius: int = RECEPTIVE_RADIUS
    mask_fraction: float = 0.30
    mask_span: int = 8
    recording_cap: int | None = None  # smoke-test subset; None = all recordings

    @property
    def run_dir(self) -> Path:
        return OUT_ROOT / self.run_id
