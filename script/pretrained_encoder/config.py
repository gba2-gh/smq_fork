"""Declared settings for the pretrained-encoder test (docs/plans/PRETRAINED_ENCODER_TEST_PLAN.md). BABEL-2 only."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "pretrained_encoder"

DATASET, WINDOW, FPS, NUM_STATES = "babel2", 8, 30, 5
SEEDS = (111, 222, 1538574472)
N_FOLDS = 5
FOLD_SEED = 111
FOLDS = tuple(f"fold{k}" for k in range(N_FOLDS))
VIEW_ROTATION_DEG = 270            # input view about the vertical axis; chosen by the label-free lifting check, frozen
INPUTS = ("t2", "mb")
MB_TOKENIZER, MB_SOURCE = "mb_babel", "motionbert"

BAR = 3.0                          # mean over folds, paired by seed; all seeds positive, on MoF and F1@50
PROBE_GAIN = 5.0                   # supervised probe points (mean over seeds) that make INFORMATION_ONLY
PROBE_CAP, SMOKE_PROBE_CAP, PROBE_SEED = 40_000, 2_000, 111


def prep_dir(inp: str, seed: int) -> Path:
    if inp == "t2":
        return ROOT / "results" / "week5" / "timescale_inspection" / "ts5_v1" / "phase5" / "prep" / "t2_relational" / f"babel2__t2_relational__seed{seed}"
    return RESULTS / "prep" / f"babel2__{MB_TOKENIZER}__seed{seed}"


def family(inp: str) -> str:
    return "tok5_t2_relational" if inp == "t2" else f"tok5_{MB_TOKENIZER}"


def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(PKG.glob("*.py")):
        if f.name != "tests.py":
            h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]
