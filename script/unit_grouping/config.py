"""Declared settings for the unit-to-action grouping test (docs/plans/UNIT_GROUPING_TEST_PLAN.md, version 2)."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "unit_grouping"
REF_RUN = ROOT / "results" / "compositional_dynamics" / "m1_lara_001"      # b0_c reference cells (C-state matched B0)

SEEDS = (111, 222, 1538574472)
SPLITS = ("primary", "fold0", "fold1", "fold2", "fold3")
NUM_ACTIONS = 8
STATE_FACTOR = 3                  # M = 3C
NUM_STATES = STATE_FACTOR * NUM_ACTIONS
HORIZON = 8                       # affinity horizon in windows (Milestone 1's declared history length)
N_RANDOM = 100
SPECTRAL_N_INIT = 10
MULTIPLIERS = (0.5, 2.0)

ARMS = ("group_temporal", "group_shuffled", "group_appearance")
BAR = 3.0                         # mean difference >= +3, all seeds same direction
FLOOR = -2.0                      # the other metric's mean may not fall by more than 2
RANDOM_PERCENTILE = 95.0


def reference_cell(seed: int, split: str) -> Path:
    return REF_RUN / "cells" / (f"b0_centre__seed{seed}" + ("" if split == "primary" else f"__{split}"))


def code_hash() -> str:
    from script.compositional_dynamics import config as CD
    h = hashlib.sha256(CD.code_hash().encode())
    for f in sorted(PKG.glob("*.py")):
        if f.name != "tests.py":
            h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]
