"""Declared settings, units, identities and published references for the Tier 0 MASQ replication.

Every value here is declared in docs/plans/week5/masq_experiments/AGENT_INSTRUCTIONS.md section 3 before any scored run.
Nothing in this module may be changed because of a score. No torch import: the spreadsheet step runs without it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG = "script/week5/masq_experiments"
RESULTS_ROOT = ROOT / "results" / "week5" / "masq_experiments"
MODELS_ROOT = ROOT / "models" / "week5" / "masq_experiments"

SEEDS = (111, 222, 1538574472)

# main.py::get_dataset_defaults
DATASETS = {
    "hugadb": dict(num_features=6, num_joints=6, num_person=1, batch_size=8, patch_size=60),
    "lara": dict(num_features=6, num_joints=22, num_person=1, batch_size=8, patch_size=50),
    "babel1": dict(num_features=3, num_joints=25, num_person=1, batch_size=32, patch_size=30),
    "babel2": dict(num_features=3, num_joints=25, num_person=1, batch_size=32, patch_size=30),
    "babel3": dict(num_features=3, num_joints=25, num_person=1, batch_size=32, patch_size=30),
}
DATASET_NAMES = {"hugadb": "HuGaDB", "lara": "LARa", "babel1": "BABEL-1", "babel2": "BABEL-2", "babel3": "BABEL-3"}
# smallest first, so a partial stage still yields paired S0 / M-pub cells
DATASET_ORDER = ("babel3", "babel2", "hugadb", "babel1", "lara")

# main.py argument defaults
SHARED = dict(epochs=30, lr=5e-4, num_f_maps=128, num_layers=3, latent_dim=16, sample_rate=1, kmeans=False,
              kmeans_metric="euclidean", sampling_quantile=0.5, replacement_strategy="representative", decay=0.5,
              dead_code_threshold=10, tc_weight=0.0, tc_beta=1.0, mse_loss_weight=0.001, commit_weight=1.0,
              joint_distance_recons=True, micro_batch_size=None, grad_checkpoint=False)

JLSD_PUBLISHED = {"hugadb": 0.25, "lara": 0.20, "babel1": 0.50, "babel2": 0.50, "babel3": 0.50}
VEL_WEIGHT = 0.001
FIX_P = 0.25
FIX_DATASETS = ("lara", "babel1", "babel2", "babel3")

BASE_ARMS = ("S0", "M-pub", "M-JL", "M-fix")
# HuGaDB-focused follow-up (docs/plans/week5/masq_experiments/TIER0_RESULTS_AND_HUGADB_FOCUS.md): M-JL's dropout plus one
# calibrated term. weight x term(init) = rho x reconstruction(init); rho_pub = 0.0016 is the initial velocity/reconstruction
# ratio that MASQ's published lambda = 0.001 gives on BABEL-1/3 (measured 0.0014 and 0.0018).
VARIANT_ARMS = ("V1", "V2a", "V2b")
VARIANT_MODE = {"V1": "coord_scalefree", "V2a": "latent_smooth", "V2b": "latent_smooth"}
VARIANT_RHO = {"V1": 0.0016, "V2a": 0.01, "V2b": 0.1}
CALIB_FILES = 16  # first recordings by file name (label-free), cropped to CALIB_MAX_FRAMES
CALIB_MAX_FRAMES = 2000
ARMS = BASE_ARMS + VARIANT_ARMS
ARM_STAGE = {"S0": "1", "M-pub": "1", "M-JL": "2", "M-fix": "2", "V1": "hu", "V2a": "hu", "V2b": "hu"}

METRICS = ("MoF", "Edit", "F1@10", "F1@25", "F1@50")

# MASQ Tables 1-2 (MoF / Edit / F1@10 / F1@25 / F1@50)
MASQ_PUBLISHED = {
    "hugadb": dict(zip(METRICS, (52.3, 45.1, 52.1, 47.0, 39.5))),
    "lara": dict(zip(METRICS, (47.2, 42.4, 42.0, 35.4, 22.2))),
    "babel1": dict(zip(METRICS, (46.7, 33.1, 39.6, 34.0, 21.0))),
    "babel2": dict(zip(METRICS, (55.8, 38.1, 46.1, 38.7, 28.4))),
    "babel3": dict(zip(METRICS, (47.7, 26.9, 28.3, 27.5, 24.5))),
}
# results/REPORT_reproduction_and_diagnosis.md section 1.1 (SMQ retrained, 3 seeds): (mean, sd)
SMQ_RETRAINED_SPREAD = {"hugadb": {"MoF": (41.60, 1.13), "F1@50": (21.50, 2.54)},
                        "lara": {"MoF": (36.26, 1.61), "F1@50": (15.75, 0.75)}}

# minutes per unit (train + score + diagnose), measured in masq_v1 (max over arms and seeds, rounded up). The first estimates
# (BABEL-1 12, BABEL-2 5) were about 5x and 2x too low and made Stage 1 overrun its budget.
UNIT_MINUTES = {"hugadb": 8, "lara": 14, "babel1": 72, "babel2": 13, "babel3": 4}

BAR = 3.0

# Hashed code. config.py is not hashed: everything it contributes to a unit's computation is in settings(), so editing a
# reference constant (published numbers, time estimates) neither invalidates trained models nor blocks a resume.
TRAIN_FILES = tuple(f"{PKG}/{m}.py" for m in ("model_masq", "losses", "train_masq")) + (
    "src/model/smq.py", "src/model/ms_tcn.py", "src/model/motion_quantizer.py", "src/model/utils.py", "batch_gen.py")
SCORE_FILES = (f"{PKG}/score.py", "src/model/eval_utils.py", "script/repro/e0_metrics.py")
# Variant code is hashed separately and only into the identities of variant units, so every masq_v1 identity is unchanged.
VARIANT_FILES = (f"{PKG}/losses_variants.py", f"{PKG}/train_variants.py")
ANALYSIS_FILES = (f"{PKG}/report.py", f"{PKG}/diagnostics.py")


@dataclass(frozen=True)
class Unit:
    arm: str
    dataset: str
    seed: int

    @property
    def unit_id(self) -> str:
        return f"{self.arm}__{self.dataset}__seed{self.seed}"

    @property
    def jlsd_p(self) -> float:
        if self.arm == "S0":
            return 0.0
        if self.arm == "M-fix":
            return FIX_P
        return JLSD_PUBLISHED[self.dataset]

    @property
    def vel_weight(self) -> float:
        return 0.0 if self.arm in ("S0", "M-JL") + VARIANT_ARMS else VEL_WEIGHT  # variants: weight is calibrated, see settings


def parse_unit(unit_id: str) -> Unit:
    arm, dataset, seed = unit_id.split("__")
    return Unit(arm, dataset, int(seed.replace("seed", "")))


def stage_units(stage: str) -> list[Unit]:
    """Stage 1: S0 and M-pub. Stage 2: M-JL, then M-fix. Dataset-major (smallest first), seeds, then arms."""
    if stage == "hu":  # the HuGaDB-focused variants: 3 variant arms x 3 seeds
        return [Unit(arm, "hugadb", seed) for seed in SEEDS for arm in VARIANT_ARMS]
    if stage == "1":
        groups = [("S0", "M-pub")]
    elif stage == "2":
        groups = [("M-JL",), ("M-fix",)]
    else:
        raise ValueError(f"unknown stage {stage!r}")
    out = []
    for arms in groups:
        for ds in DATASET_ORDER:
            for seed in SEEDS:
                for arm in arms:
                    if arm == "M-fix" and ds not in FIX_DATASETS:
                        continue
                    out.append(Unit(arm, ds, seed))
    return out


def smoke_units() -> list[Unit]:
    return [Unit(arm, "babel3", 111) for arm in BASE_ARMS] + [Unit(arm, "hugadb", 111) for arm in VARIANT_ARMS]


def settings(unit: Unit, smoke: bool = False) -> dict:
    """Every value that affects the result of the unit."""
    s = dict(SHARED, **DATASETS[unit.dataset])
    s.update(arm=unit.arm, dataset=unit.dataset, seed=unit.seed, jlsd_p=unit.jlsd_p, vel_weight=unit.vel_weight)
    if unit.arm in VARIANT_ARMS:
        s.update(vel_mode=VARIANT_MODE[unit.arm], rho=VARIANT_RHO[unit.arm], calib_files=CALIB_FILES, calib_max_frames=CALIB_MAX_FRAMES)
    if smoke:
        s.update(epochs=2, max_batches_per_epoch=3, smoke=True)
    return s


def stable_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _files_hash(files) -> str:
    h = hashlib.sha256()
    for name in files:
        h.update(name.encode())
        h.update((ROOT / name).read_bytes())
    return h.hexdigest()[:16]


def train_hash() -> str:
    """Model, loss and training code, plus the SMQ files they use."""
    return _files_hash(TRAIN_FILES)


def score_hash() -> str:
    return _files_hash(SCORE_FILES)


def variant_hash() -> str:
    return _files_hash(VARIANT_FILES)


def code_hash() -> str:
    """Training and scoring code together (the gates test both)."""
    return stable_hash(dict(train=train_hash(), score=score_hash()))


def analysis_hash() -> str:
    return _files_hash(ANALYSIS_FILES)


def unit_identity(unit: Unit, smoke: bool = False) -> str:
    """Identity of a trained model: its settings and the training code."""
    d = dict(settings=settings(unit, smoke), train=train_hash())
    if unit.arm in VARIANT_ARMS:  # baseline arms keep the exact dict (and hash) they had in masq_v1
        d["variant"] = variant_hash()
    return stable_hash(d)


def score_identity(unit: Unit, smoke: bool = False) -> str:
    """Identity of a unit's scores: the trained model and the scoring code (a scoring fix re-scores, never retrains)."""
    return stable_hash(dict(unit=unit_identity(unit, smoke), score=score_hash()))


def diagnostics_identity(unit: Unit, smoke: bool = False) -> str:
    return stable_hash(dict(scores=score_identity(unit, smoke), analysis=analysis_hash()))


def model_dir(run_id: str, unit: Unit) -> Path:
    return MODELS_ROOT / run_id / unit.arm / unit.dataset / f"seed{unit.seed}"


def run_dir(run_id: str) -> Path:
    return RESULTS_ROOT / run_id


def unit_dir(run_id: str, unit: Unit) -> Path:
    return run_dir(run_id) / "units" / unit.unit_id


def write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)
