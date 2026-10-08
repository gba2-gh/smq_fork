"""Declared settings for Milestone 1 (docs/plans/COMPOSITIONAL_DYNAMICS_M1_AGENT_INSTRUCTIONS.md, v2.1).

These are new feasibility defaults, frozen before any scored run. Nothing here is chosen from a score."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parent
PHASE5 = ROOT / "results" / "week5" / "timescale_inspection" / "ts5_v1" / "phase5"
RESULTS = ROOT / "results" / "compositional_dynamics"

DATASET, TOKENIZER, FAMILY = "lara", "t2_relational", "tok5_t2_relational"
SEEDS = (111, 222, 1538574472)
ARMS = ("b0_centre", "m1_ordered", "m1_bag", "occupancy")
NEURAL_ARMS = ("m1_ordered", "m1_bag")

# Source facts verified by preflight (disagreement is a preflight failure).
NUM_CODES, WINDOW, FPS, NUM_STATES, LOOKUP_DIM = 512, 12, 50, 8, 64
SPLIT_SHA256 = "04f8a605d3881dea040e6e26fa0a5ab908b4268562ba3a5bdb9eec7f8d2940aa"
N_FIT, N_EVAL = 395, 44
SUBJECT_FOLDS = 4          # secondary readout: 4 folds of 4 held-out subjects (label-free, seeded)
MULTIPLIERS = (0.5, 2.0)   # secondary readout: final-decode multipliers on the frozen emission cost (primary = 1)
SPLITS = ("primary",) + tuple(f"fold{k}" for k in range(SUBJECT_FOLDS))

# Transport (script.action_transport.config SETTING_T), CPU float64.
COEFFS = dict(a=0.7, beta=0.3, lam=0.05, eps=0.07)
B0_INNER_STEPS, B0_OUTER_PATIENCE, B0_OUTER_RELTOL = 25, 3, 1e-6   # = action_transport.config FIT_*

# History model.
HIST_SECONDS = 2.0
HIDDEN, BOTTLENECK, ALPHA, PSEUDOCOUNT = 128, 16, 0.5, 1.0
BATCH, LR, WEIGHT_DECAY, CLIP = 256, 3e-4, 0.01, 1.0
CALIB_SEED = 111      # calibration sample and bag permutation and order-sensitivity permutations
MIN_SCALE = 1e-6
ADAPTER_UP_STD, BIAS_STD = 0.125, 0.02


def history_length(fps: int = FPS, window: int = WINDOW) -> int:
    return int(HIST_SECONDS * fps / window + 0.5)


@dataclass(frozen=True)
class Scale:
    name: str
    block_updates: int      # model updates per block; refresh after every block but the last
    n_blocks: int           # 10 blocks of 500 = 5,000 updates, 9 refreshes
    inner_steps: int        # accepted inner steps per refresh
    b0_outer_cap: int
    rungs: tuple            # final-solver continuation endpoints
    calib_cap: int
    smoke_fit: int = 0      # 0 = all recordings
    smoke_eval: int = 0

    @property
    def total_updates(self) -> int:
        return self.block_updates * self.n_blocks


FULL = Scale("full", 500, 10, 25, 50, (500, 4000), 10000)
SMOKE = Scale("smoke", 20, 3, 5, 3, (20, 60), 2000, smoke_fit=12, smoke_eval=4)
SCALES = {"full": FULL, "smoke": SMOKE}


def prep_dir(seed: int) -> Path:
    return PHASE5 / "prep" / TOKENIZER / f"{DATASET}__{TOKENIZER}__seed{seed}"


def historical_unit(seed: int, arm: str = "centre") -> Path:
    return PHASE5 / "units" / f"{DATASET}__{TOKENIZER}__seed{seed}__{arm}" / "info.json"


def code_hash() -> str:
    """Hash of the package's code (not tests or docs): part of every cell identity."""
    h = hashlib.sha256()
    for f in sorted(PKG.glob("*.py")):
        if f.name != "tests.py":
            h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]
