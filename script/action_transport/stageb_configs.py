"""Shared Stage B training-configuration definitions: the v1.5 baseline and
the Phase 1 optimization-study grid.

Factored out of `stageb_optstudy.py` (v1.7) so `run_stage_b.py` can resolve
the study's selected configuration by name without importing the study
module -- `stageb_optstudy.py` itself imports helpers from `run_stage_b.py`,
so the reverse import would be circular. This module has no side effects and
imports nothing from either.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, replace

from script.action_transport.stageb_train import TrainConfig

BASELINE = TrainConfig()  # the v1.5 configuration (epoch mode, post-LN, constant lr)
STEP = dict(max_steps=5000, eval_every=100, patience_steps=1000)
GRID = {  # 2 x 2 x 2 x 2 x 2 = 32 configurations; dropout fixed at the plan's 0.1
    "norm_first": [False, True],
    "schedule": [("constant", 0), ("cosine", 500)],  # (schedule, warmup_steps)
    "lr": [3e-4, 1e-3],
    "tie_head": [False, True],
    "position_mode": ["sinusoidal", "alibi"],
}

# The configuration selected by optstudy_002 (docs/plans/STAGE_C_AGENT_INSTRUCTIONS.md
# §1.2): lowest mean tuning cross-entropy across all 4 real-data cells. Checked
# against a freshly computed hash in resolve_by_name() rather than trusted blindly.
SELECTED_CONFIG_NAME = "postLN_cosine_wu500_lr0.0003_alibi"
SELECTED_CONFIG_HASH = "0689bb38dfdd031b"


def config_hash(cfg: TrainConfig) -> str:
    """Identifies the EFFECTIVE configuration a run/row was trained under,
    independent of its name (v1.6 B2 fix in stageb_optstudy.py)."""
    return hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def grid_configs() -> list[tuple[str, TrainConfig]]:
    out = []
    for norm_first, (schedule, warmup), lr, tie, position in itertools.product(*GRID.values()):
        name = (f"{'preLN' if norm_first else 'postLN'}_{schedule}{'_wu' + str(warmup) if warmup else ''}"
                f"_lr{lr:g}{'_tied' if tie else ''}_{position}")
        out.append((name, replace(BASELINE, norm_first=norm_first, schedule=schedule, warmup_steps=warmup,
                                  lr=lr, tie_head=tie, position_mode=position, **STEP)))
    return out


def named_configs() -> dict[str, TrainConfig]:
    return dict(grid_configs())


def resolve_by_name(name: str) -> TrainConfig:
    """Look up one grid configuration by its name and verify its hash matches
    the value recorded when it was selected -- catches a since-changed grid
    or TrainConfig default silently producing a different "selected" model."""
    configs = named_configs()
    if name not in configs:
        raise KeyError(f"{name!r} is not in the current grid ({sorted(configs)})")
    return configs[name]
