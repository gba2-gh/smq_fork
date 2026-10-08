"""Declared settings for the input-representation test (docs/plans/INPUT_REPRESENTATION_TEST_PLAN.md).

LARa only. Three fixed per-window inputs, one unsupervised decode (the matched B0 of Milestone 1) and label-using
probes used strictly as diagnostics / upper bounds. Nothing here is chosen from a score."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "input_representation"
REF_RUN = ROOT / "results" / "compositional_dynamics" / "m1_lara_001"      # T2 reference: matched B0, primary split

DATASET, WINDOW = "lara", 12
SEEDS = (111, 222, 1538574472)
# variant -> (tokenizer key, source). 't2' is the existing preparation; the other two are built here.
VARIANTS = {
    "t2": ("t2_relational", "relational"),
    "t2_bodyscale": ("t2s_bodyscale", "relational_bodyscale"),
    "t2_centered": ("t2c_centered", "centered_motion"),
}
NEW_VARIANTS = ("t2_bodyscale", "t2_centered")

# Declared decision rules.
BAR = 3.0               # mean difference vs T2 on BOTH MoF and F1@50, all three seeds positive
GAP_THRESHOLD = 15.0    # supervised-probe accuracy minus unsupervised MoF, points

# Probes (label-using diagnostics).
PROBE_CAP = 40_000      # training windows per probe fit (seeded subsample)
PROBE_SEED = 111
SMOKE_PROBE_CAP = 2_000
LR_C, LR_MAX_ITER = 1.0, 300


def prep_dir(variant: str, seed: int) -> Path:
    """Where a variant's preparation lives. 't2' is the Phase 5 preparation used by Milestone 1."""
    if variant == "t2":
        from script.compositional_dynamics import config as CD
        return CD.prep_dir(seed)
    key = VARIANTS[variant][0]
    return RESULTS / "prep" / f"{DATASET}__{key}__seed{seed}"


def family(variant: str) -> str:
    return f"tok5_{VARIANTS[variant][0]}"


def code_hash() -> str:
    h = hashlib.sha256()
    for f in sorted(PKG.glob("*.py")):
        if f.name != "tests.py":
            h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]
