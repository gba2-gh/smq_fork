from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "exp2_fsq" / "q1q2"
VOCAB_SIZES = (10,  50,  500, 1000)
WINDOWS = {"hugadb": (60, 30, 15), "lara": (50, 25, 12)}
NUM_ACTIONS = {"hugadb": 10, "lara": 8}

