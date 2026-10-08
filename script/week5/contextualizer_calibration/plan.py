"""Typed effective configurations, the planned training/evaluation matrix and
their stable identities for the contextualizer search calibration
(docs/plans/week5/CONTEXTUALIZER_SEARCH_CALIBRATION/).

Nothing here trains or decodes. `enumerate_training_jobs` / `enumerate_evaluations`
return the full plan (default 14 jobs / 58 OT fits; 15 / 60 with the optional
combined target-count control) so it can be saved before any execution.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

REFERENCE_SEED, SECOND_SEED = 111, 222
FULL_HORIZON, SHORT_HORIZON = 5000, 3000
FIXED_ROLES = ("ckpt_1k", "ckpt_3k", "ckpt_5k")  # fixed-step checkpoint roles (steps come from Scale)
OUTER_CAP_DEFAULT, OUTER_CAP_SENSITIVITY = 50, 150
SPAN_REFERENCE = 8

# pipeline -> (family, input arm, guard radius in raw frames)
PIPELINES = {
    "native_vq": ("vq_native", "context_quantized", 14),
    "raw": ("raw_kmeans", "context_id", 0),
}
RAW_GUARDED_RADIUS = 14  # reproduction-only variant of the raw pipeline

# name -> (context tokens, mask span tokens, stride, effective batch in chunks)
CONFIGS = {
    "reference": (128, 8, 64, 64),
    "short_context": (64, 8, 32, 128),
    "long_mask": (128, 16, 64, 64),
    "combined": (64, 16, 32, 128),
}
NOMINAL_TOKENS_PER_UPDATE = 8192


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    group: str  # main | guard_ref | horizon3k | seed222 | target_match | target_match_combined
    config_name: str
    pipeline: str
    family: str
    arm: str
    guard_radius: int
    context: int
    stride: int
    mask_span: int
    mask_fraction: float
    batch_chunks: int
    horizon: int  # nominal horizon (5000 / 3000); Scale maps it to actual updates
    seed: int
    target_count: str  # "none" | "match"
    ref_job_id: str  # reference job paired for target counting ("" otherwise)


@dataclass(frozen=True)
class Scale:
    """Budget scale. FULL is the scientific budget; SMOKE only shrinks budgets
    (updates, rungs, outer caps, recordings), never the planned matrix."""
    name: str
    horizons: dict
    fixed_steps: dict
    eval_every: int
    patience: int
    warmup: int | None
    rungs: tuple
    outer_cap: int
    outer_cap_sensitivity: int
    recording_cap: int | None
    smoke: bool

    def horizon(self, nominal: int) -> int:
        return self.horizons[nominal]


FULL = Scale("full", {5000: 5000, 3000: 3000}, {"ckpt_1k": 1000, "ckpt_3k": 3000, "ckpt_5k": 5000}, 100, 1000,
             None, (500, 1000, 2000, 4000), OUTER_CAP_DEFAULT, OUTER_CAP_SENSITIVITY, None, False)
SMOKE = Scale("smoke", {5000: 40, 3000: 24}, {"ckpt_1k": 8, "ckpt_3k": 24, "ckpt_5k": 40}, 4, 8, 5,
              (20, 40), 3, 6, 12, True)


def make_job_id(pipeline: str, context: int, span: int, guard: int, horizon: int, seed: int,
                target_count: str) -> str:
    return f"{pipeline}__ctx{context}_span{span}_g{guard}_h{horizon}_seed{seed}_tc{target_count}"


def _job(group: str, config_name: str, pipeline: str, guard: int, horizon: int, seed: int,
         target_count: str = "none", ref_job_id: str = "") -> JobSpec:
    family, arm, _ = PIPELINES[pipeline]
    context, span, stride, batch = CONFIGS[config_name]
    return JobSpec(make_job_id(pipeline, context, span, guard, horizon, seed, target_count), group, config_name,
                   pipeline, family, arm, guard, context, stride, span, 0.30, batch, horizon, seed,
                   target_count, ref_job_id)


def primary_job(config_name: str, pipeline: str, horizon: int = FULL_HORIZON, seed: int = REFERENCE_SEED) -> JobSpec:
    return _job("main", config_name, pipeline, PIPELINES[pipeline][2], horizon, seed)


def enumerate_training_jobs(include_combined_target_match: bool = False) -> list[JobSpec]:
    """Jobs in execution order: the two reproduction references first (they
    gate everything that depends on them), then the rest of the main grid,
    then the horizon / seed / target-count controls."""
    native_ref = primary_job("reference", "native_vq")
    jobs = [native_ref, _job("guard_ref", "reference", "raw", RAW_GUARDED_RADIUS, FULL_HORIZON, REFERENCE_SEED)]
    for pipeline in PIPELINES:
        for name in CONFIGS:
            job = primary_job(name, pipeline)
            if job.job_id != native_ref.job_id:
                jobs.append(job)
    for pipeline in PIPELINES:
        jobs.append(_job("horizon3k", "reference", pipeline, PIPELINES[pipeline][2], SHORT_HORIZON, REFERENCE_SEED))
    for pipeline in PIPELINES:
        jobs.append(_job("seed222", "reference", pipeline, PIPELINES[pipeline][2], FULL_HORIZON, SECOND_SEED))
    guard = PIPELINES["native_vq"][2]
    jobs.append(_job("target_match", "long_mask", "native_vq", guard, FULL_HORIZON, REFERENCE_SEED, "match",
                     native_ref.job_id))
    if include_combined_target_match:
        jobs.append(_job("target_match_combined", "combined", "native_vq", guard, FULL_HORIZON, REFERENCE_SEED,
                         "match", primary_job("short_context", "native_vq").job_id))
    ids = [j.job_id for j in jobs]
    assert len(ids) == len(set(ids)), "job ids must be unique"
    return jobs


@dataclass(frozen=True)
class EvalSpec:
    eval_id: str
    job_id: str
    ckpt_role: str  # ckpt_1k | ckpt_3k | ckpt_5k | selected | replay | final
    outer_cap: int
    kind: str  # policy | outer_sensitivity


GROUP_ROLES = {
    "main": ("ckpt_1k", "ckpt_3k", "ckpt_5k", "selected", "replay"),
    "guard_ref": ("selected", "replay"),
    "horizon3k": ("selected", "final"),
    "seed222": ("selected", "final", "replay"),
    "target_match": ("selected", "final"),
    "target_match_combined": ("selected", "final"),
}


def make_eval_id(job_id: str, role: str, outer_cap: int) -> str:
    return f"{job_id}::{role}::cap{outer_cap}"


def enumerate_evaluations(jobs: list[JobSpec], outer_cap: int = OUTER_CAP_DEFAULT,
                          outer_cap_sensitivity: int = OUTER_CAP_SENSITIVITY) -> list[EvalSpec]:
    """Planned OT fits BEFORE checkpoint deduplication: aliased roles (for
    example selected == ckpt_5k) remain separate planned rows."""
    evals = [EvalSpec(make_eval_id(j.job_id, role, outer_cap), j.job_id, role, outer_cap, "policy")
             for j in jobs for role in GROUP_ROLES[j.group]]
    for j in jobs:
        # Outer-iteration sensitivity: Reference and Combined, final 5k checkpoint, both primary pipelines.
        if j.group == "main" and j.config_name in ("reference", "combined"):
            evals.append(EvalSpec(make_eval_id(j.job_id, "final", outer_cap_sensitivity), j.job_id, "final",
                                  outer_cap_sensitivity, "outer_sensitivity"))
    ids = [e.eval_id for e in evals]
    assert len(ids) == len(set(ids)), "evaluation ids must be unique"
    return evals


def stable_hash(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def effective_settings(job: JobSpec, scale: Scale) -> dict:
    """Every setting that determines a job's trajectory, resolved (not the
    mutable config name). Hashed into the job identity."""
    from script.week5.contextualizer_calibration import train as T
    cfg = T.train_config(job, scale)
    horizon = scale.horizon(job.horizon)
    return dict(job=asdict(job), scale=asdict(scale), train_config=asdict(cfg), model=T.model_settings(cfg),
                optimizer=T.OPTIMIZER_SETTINGS, validation_mask_base=T.VALIDATION_MASK_BASE,
                epoch_rng="default_rng(seed*1_000_003+epoch); per-chunk mask seed = epoch_rng.integers(0, 2**31-1)",
                fixed_steps={r: scale.fixed_steps[r] for r in FIXED_ROLES if scale.fixed_steps[r] <= horizon},
                replay_patience=scale.patience, micro_chunks=T.MICRO_CHUNKS)


def job_identity(job: JobSpec, scale: Scale) -> str:
    return stable_hash(effective_settings(job, scale))


def planned_training_rows(jobs: list[JobSpec], scale: Scale) -> list[dict]:
    rows = []
    for j in jobs:
        row = asdict(j)
        row.update(horizon_steps=scale.horizon(j.horizon), nominal_tokens_per_update=j.context * j.batch_chunks,
                   identity=job_identity(j, scale), optional=j.group == "target_match_combined",
                   checkpoint_roles="|".join(GROUP_ROLES[j.group]))
        rows.append(row)
    return rows


def planned_evaluation_rows(evals: list[EvalSpec], jobs: list[JobSpec]) -> list[dict]:
    by_job = {j.job_id: j for j in jobs}
    return [dict(asdict(e), group=by_job[e.job_id].group, pipeline=by_job[e.job_id].pipeline,
                 config_name=by_job[e.job_id].config_name, checkpoint_step="(resolved after training)",
                 optional=by_job[e.job_id].group == "target_match_combined") for e in evals]
