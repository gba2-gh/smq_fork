"""Declared settings for the self-labelled embedding test (docs/plans/SELF_LABELLED_EMBEDDING_PLAN.md). BABEL-2 only, stage 1."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "self_labelled"
PE_RUN = ROOT / "results" / "pretrained_encoder" / "pe_babel2_001"       # F0 reference cells (seeds 111, 222, 1538574472)

DATASET, WINDOW, FPS, NUM_STATES = "babel2", 8, 30, 5
SEEDS = (111, 222, 1538574472, 333, 444)
PE_SEEDS = (111, 222, 1538574472)
FOLDS = tuple(f"fold{k}" for k in range(5))
ARMS = ("F0", "F1", "L_full", "L_noeq", "L_noaug")
LEARNED = ("L_full", "L_noeq", "L_noaug")

# augmentation bank
BANK_SEED, N_VARIANTS = 20261008, 8
SCALE_RANGE, MIRROR_P = (0.85, 1.15), 0.5
# model
IN_DIM, HID, OUT_DIM, KERNEL, N_BLOCKS = 1536, 256, 128, 5, 2          # receptive field +-4 windows
# training
BATCH_RECS, UPDATES, FREEZE_PROTOS = 16, 3000, 300
TAU, LR, WD, CLIP = 0.1, 1e-3, 1e-4, 1.0
OT_INNER_STEPS = 25
LAMBDA_TRAIN = 1.0           # relaxed column-marginal weight for TRAINING pseudo-labels (decoding keeps SETTING_T's 0.05)
FEATURE_DROPOUT = 0.1                                                    # L_noaug views
COLLAPSE_SHARE = 0.9
COLLAPSE_WINDOW = 50              # a cell is flagged from the mean over the last 50 updates
MIN_EFFECTIVE_STATES = 3.0        # usage entropy below ln(3) also flags the cell
PROTO_MAX_COS = 0.95              # two prototypes closer than this are indistinguishable states
F1_PCA_DIM, F1_SAMPLE_CAP = 64, 10_000   # F1: per-fold scaler + PCA fitted on fitting windows only
NN_SHARED_CONTEXT_RADIUS = 8      # windows; the historical radius (neighbours may still share source context)
# decision rule (development): each mean >= -1, at least one >= +2 with >= 4 of 5 seeds positive
FLOOR, GAIN, MIN_SEEDS_POS = -1.0, 2.0, 4
# decode
OUTER_CAP, RUNGS = 50, (500, 4000)
SMOKE_OUTER_CAP, SMOKE_RUNGS, SMOKE_UPDATES = 3, (20, 60), 100


def code_hash() -> str:
    """Hash of this package (not tests) and of the reused modules whose behaviour the cells depend on."""
    h = hashlib.sha256()
    reused = [ROOT / "script/action_transport/transport.py", ROOT / "script/action_transport/continuous.py",
              ROOT / "script/compositional_dynamics/decode.py", ROOT / "script/compositional_dynamics/evaluate.py",
              ROOT / "script/pretrained_encoder/encoder.py", ROOT / "script/pretrained_encoder/features.py"]
    for f in sorted(PKG.glob("*.py")):
        if f.name != "tests.py":
            h.update(f.name.encode() + f.read_bytes())
    for f in reused:
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]
