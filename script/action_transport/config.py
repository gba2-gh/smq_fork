"""Frozen configuration: seeds, coefficient settings, arms, and planned-cell
enumeration for Stage A (docs/plans/THREE_STAGE_MOTION_PLAN.md v1.3).

Nothing here is chosen from action scores. Changing any value below is a new
frozen configuration (bump PROTOCOL_VERSION) and invalidates cached artifacts
tagged with the previous version, per instructions §3: "Include protocol
version 1.3 ... in artifact identity. Do not reuse v1.2 half-scale cosine fits
or un-restored permutation metrics as current results."
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "results" / "action_transport"
THIRD_PARTY_ASOT = ROOT / "third_party" / "action_seg_ot"
LOCAL_ASOT_CLONE = Path(r"C:\Users\gzaz976\Documents\repo\Behaviour\action_seg_ot")
ASOT_PINNED_COMMIT = "0c4c86b1037eb4c3e262197aeee115b31e220c9d"

# Protocol/artifact-identity version. Bump on any numerical-contract change.
PROTOCOL_VERSION = "1.3"

DATASETS = ("hugadb", "lara")
SEEDS = (111, 222, 1538574472)
SAMPLING_SEED = 111  # sampling/PCA/fold-tiebreak seed, fixed
PERMUTATION_SEED_OFFSET = 10_000_000  # permuted_asot uses seed + this offset

WINDOW = {"hugadb": 15, "lara": 12}
FPS = {"hugadb": 60, "lara": 50}
NUM_ACTIONS = {"hugadb": 10, "lara": 8}  # C, including background
K_FINE = 500

NORMALIZATIONS = (False, True)  # per-recording z-score off / on


@dataclass(frozen=True)
class CoeffSetting:
    """One frozen ASOT coefficient setting. a=1-alpha, beta=alpha in ASOT's
    own parameterization (docs/plans/THREE_STAGE_MOTION_PLAN.md §3)."""

    name: str
    a: float
    beta: float
    lam: float  # action-marginal KL weight (lambda_actions)
    eps: float  # entropy regularization


# Pinned src/train.py CLI defaults at commit ASOT_PINNED_COMMIT. Not the
# constructor defaults and not run_bf.sh's overrides (see REVIEW_RESPONSE
# files) -- this is a declared lineage choice, not the sole valid ASOT
# protocol.
SETTING_T = CoeffSetting("T", a=0.7, beta=0.3, lam=0.05, eps=0.07)
SETTING_E = CoeffSetting("E", a=0.4, beta=0.6, lam=0.01, eps=0.04)
SETTINGS = (SETTING_T, SETTING_E)

# Fitting uses T always; final re-solve runs under both T and E.
FIT_SETTING = SETTING_T

# --- Solver budgets -------------------------------------------------------

BACKTRACK_MAX = 30
BACKTRACK_TOLERANCE = 1e-12  # relative-to-scale slack before treating a step as an increase

FIT_INNER_STEPS = 25          # accepted mirror steps per recording per outer iter
FIT_OUTER_CAP = 50
FIT_OUTER_PATIENCE = 3         # consecutive outer iters at the stopping tolerance
FIT_OUTER_RELTOL = 1e-6

FINAL_MAX_STEPS = 500
FINAL_GRAD_TOL = 1e-5
FINAL_OBJ_RELTOL = 1e-7
FINAL_PATIENCE = 5

CPU_THREAD_CAP = 8

# --- Arms ------------------------------------------------------------------

FITTED_ARMS = ("no_temporal", "categorical_asot", "categorical_asot_kc", "continuous_asot")
DERIVED_ARMS = ("no_temporal_filter",)
ALL_PREDICTION_ARMS = FITTED_ARMS + DERIVED_ARMS  # 5 arms, each x{T,E} = 10 sets
ARM_LABELS = {
    "no_temporal": "A0",
    "no_temporal_filter": "A0+filter",
    "categorical_asot": "A1",
    "categorical_asot_kc": "A1-KC",
    "continuous_asot": "A-cont",
}


def vocab_size(dataset: str, arm: str) -> int:
    return NUM_ACTIONS[dataset] if arm == "categorical_asot_kc" else K_FINE


def num_actions(dataset: str) -> int:
    return NUM_ACTIONS[dataset]


def eta(dataset: str) -> float:
    """Total pseudocount per state, eta=fps (docs plan §3)."""
    return float(FPS[dataset])


def kernel_half_width(fps: int, window: int, num_windows: int) -> int:
    """b = min(L-1, max(1, floor(fps/W))) with L=num_windows; V=0 if L<=1."""
    if num_windows <= 1:
        return 0
    return min(num_windows - 1, max(1, fps // window))


@dataclass(frozen=True)
class CellId:
    """A unique fit identity: one fitted arm under one configuration."""

    dataset: str
    normalize: bool
    seed: int
    protocol: str  # "pooled" | "subject_disjoint"
    fold: int | None  # None for pooled
    arm: str

    def key(self) -> str:
        fold_part = "pooled" if self.fold is None else f"fold{self.fold}"
        norm_part = "norm" if self.normalize else "raw"
        return f"{self.dataset}_{norm_part}_{self.protocol}_{fold_part}_seed{self.seed}_{self.arm}"


def enumerate_pooled_cells() -> list[CellId]:
    cells = []
    for dataset in DATASETS:
        for normalize in NORMALIZATIONS:
            for seed in SEEDS:
                for arm in FITTED_ARMS:
                    cells.append(CellId(dataset, normalize, seed, "pooled", None, arm))
    return cells


def enumerate_permutation_cells() -> list[CellId]:
    # Seed 111 only, both datasets and normalizations, pooled only, arm A1.
    return [CellId(dataset, normalize, SAMPLING_SEED, "pooled", None, "permuted_asot")
            for dataset in DATASETS for normalize in NORMALIZATIONS]


def enumerate_subject_disjoint_cells(n_folds: int = 4) -> list[CellId]:
    cells = []
    for dataset in DATASETS:
        for normalize in NORMALIZATIONS:
            for seed in SEEDS:
                for fold in range(n_folds):
                    for arm in FITTED_ARMS:
                        cells.append(CellId(dataset, normalize, seed, "subject_disjoint", fold, arm))
    return cells


def prediction_set_ids(cell: CellId) -> list[str]:
    """Every scored prediction set derived from one fitted arm."""
    if cell.arm == "permuted_asot":
        return [f"{cell.key()}__{setting.name}" for setting in SETTINGS]
    ids = [f"{cell.key()}__{setting.name}" for setting in SETTINGS]
    if cell.arm == "no_temporal":
        ids += [f"{cell.key()}_filter__{setting.name}" for setting in SETTINGS]
    return ids
