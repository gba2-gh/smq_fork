"""Task 2: OT decoding for the six arms per family (+ context_geometry_init),
built on Task 1's shared grouping and Task 2's `contextual.py`. Reuses
`script.action_transport.categorical`/`continuous`/`transport` unmodified,
and `script.fsq_reconstruction.evaluate`'s K-agnostic (log_k-parameterized)
categorical helpers -- this K=512 experiment must NOT touch
`categorical.py`'s hardcoded K=500 `LOG_500`, exactly the reason that
module's K-agnostic wrappers exist.

Arm -> OT family:
    direct_id             categorical OT directly on Task 1's raw code ids
    context_id             contextual OT on context_id's continuous embeddings
    direct_quantized/
    direct_continuous      continuous (cosine) OT directly on Task 1's raw
                            feature vectors, mean-subtracted, L2-normalized
    context_quantized/
    context_continuous/
    context_geometry_init  continuous (cosine) OT on the matching
                            contextualizer's continuous embeddings (already
                            L2-normalized by `contextual.extract_vector_embeddings`)

Labels (`labels.py`) are read ONLY inside the two `*_ladder` scoring calls
below -- never inside fitting (`fit_categorical_k`/`fit_continuous`), which
only ever sees `RecordingInput`/`ContinuousRecordingInput` (codes/vectors/
lengths, no labels).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from script.action_transport import categorical, config as at_config, continuous, transport
from script.fsq_reconstruction.evaluate import RungResult, categorical_final_ladder, contextual_final_ladder, \
    fit_categorical_k

DEVICE = torch.device("cpu")  # OT solves are CPU per the historical action_transport convention
CPU_THREAD_CAP = 8
LOG_K = math.log(512)
NUM_STATES = 10
ETA = 60.0
COEFFS = transport.Coeffs(a=0.7, beta=0.3, lam=0.05, eps=0.07)
RUNGS = (500, 1000, 2000, 4000)
FIT_INNER_STEPS, FIT_OUTER_CAP, FIT_OUTER_PATIENCE, FIT_OUTER_RELTOL = 25, 50, 3, 1e-6
FPS, WINDOW = at_config.FPS["hugadb"], at_config.WINDOW["hugadb"]


def set_cpu_thread_cap() -> None:
    torch.set_num_threads(CPU_THREAD_CAP)


# --- recording builders --------------------------------------------------------------

def categorical_recordings(loader, family: str, names: list[str]) -> list[categorical.RecordingInput]:
    out = []
    for name in names:
        rec = loader.recording(family, name)
        out.append(categorical.RecordingInput(name, rec["codes"].astype(np.int32),
                                              rec["lengths"].astype(np.int32), FPS, WINDOW))
    return out


def continuous_recordings_from_raw(loader, family: str, arm: str, names: list[str]) -> list[continuous.ContinuousRecordingInput]:
    """direct_quantized / direct_continuous: raw Task 1 feature vectors,
    minus the shared pooled mean, L2-normalized (spec: 'For direct vector
    arms, subtract Task 1's shared continuous-feature mean and L2-normalize')."""
    mean = loader.mean(family).astype(np.float64)
    out = []
    for name in names:
        rec = loader.recording(family, name)
        raw = loader.lookup(family)[rec["codes"]] if arm == "direct_quantized" else rec["continuous"]
        z, zero = continuous.l2_normalize(raw.astype(np.float64) - mean[None, :])
        out.append(continuous.ContinuousRecordingInput(name, rec["codes"].astype(np.int32),
                                                       rec["lengths"].astype(np.int32), FPS, WINDOW, z, zero))
    return out


def continuous_recordings_from_embeddings(codes_by_name: dict[str, np.ndarray], lengths_by_name: dict[str, np.ndarray],
                                          emb_by_name: dict[str, np.ndarray],
                                          zero_by_name: dict[str, np.ndarray]) -> list[continuous.ContinuousRecordingInput]:
    """context_* arms: already-unit contextual embeddings, used as-is (spec:
    'Contextual OT uses normalized contextual embeddings')."""
    out = []
    for name in sorted(codes_by_name):
        out.append(continuous.ContinuousRecordingInput(name, codes_by_name[name].astype(np.int32),
                                                       lengths_by_name[name].astype(np.int32), FPS, WINDOW,
                                                       emb_by_name[name].astype(np.float64), zero_by_name[name]))
    return out


# --- grouping / prototype initialization ----------------------------------------------

def theta0_from_grouping(loader, family: str, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    grouping = loader.grouping(family)
    recs = categorical_recordings(loader, family, names)
    counts = categorical.frame_weighted_counts(recs, num_codes=512)
    theta0 = categorical.build_theta0(grouping, counts, NUM_STATES, num_codes=512, eta=ETA)
    return theta0, grouping


def mu0_from_grouping(grouping: np.ndarray, recs: list[continuous.ContinuousRecordingInput]) -> np.ndarray:
    """spec: 'Share Task 1's grouping across arms; compute initial vector
    prototypes from each arm's own features under that grouping.'"""
    mu0, _ = continuous.build_mu0(grouping, recs, NUM_STATES)
    return mu0


# --- fit + score -----------------------------------------------------------------------

def fit_and_score_categorical(loader, family: str, arm: str, fit_names: list[str], score_names: list[str],
                              deadline: float | None = None, rungs: tuple = RUNGS,
                              predictions_out: dict | None = None) -> dict:
    theta0, grouping = theta0_from_grouping(loader, family, fit_names)
    fit_recs = categorical_recordings(loader, family, fit_names)
    fit_result = fit_categorical_k(fit_recs, theta0, NUM_STATES, num_codes=512, eta=ETA, coeffs=COEFFS,
                                   inner_steps=FIT_INNER_STEPS, outer_cap=FIT_OUTER_CAP,
                                   outer_patience=FIT_OUTER_PATIENCE, outer_reltol=FIT_OUTER_RELTOL,
                                   device=DEVICE, log_k=LOG_K, deadline=deadline)
    if fit_result.outer_status == "invalid":
        # Mirrors the continuous path (`run_task2.run_context_cell`/
        # `run_direct_cell`'s `if fit.outer_status != "invalid" else []`):
        # never score a fit that hit a nonfinite value or a backtracking
        # failure -- an invalid theta produces meaningless (or NaN/degenerate)
        # emission costs and must not silently look like a completed cell.
        return dict(arm=arm, family=family, fit_result=fit_result, ladder=[], grouping=grouping,
                   theta0=theta0, theta=fit_result.params)
    score_recs = categorical_recordings(loader, family, score_names)
    from script.quantized_input_comparison import labels as qic_labels
    gts = qic_labels.gts_for(score_names)
    starts = {n: loader.recording(family, n)["starts"] for n in score_names}
    frame_counts = {n: loader.frame_count(n) for n in score_names}
    ladder = categorical_final_ladder(fit_result.params, score_recs, NUM_STATES, LOG_K, COEFFS, rungs,
                                      gts, starts, frame_counts, deadline, predictions_out=predictions_out,
                                      fail_on_invalid=True)
    return dict(arm=arm, family=family, fit_result=fit_result, ladder=ladder, grouping=grouping,
               theta0=theta0, theta=fit_result.params)


def fit_continuous_arm(fit_recs: list[continuous.ContinuousRecordingInput], grouping: np.ndarray,
                       deadline: float | None = None) -> "continuous.FitResult":
    mu0 = mu0_from_grouping(grouping, fit_recs)
    return continuous.fit_continuous(fit_recs, mu0, NUM_STATES, COEFFS, FIT_INNER_STEPS, FIT_OUTER_CAP,
                                     FIT_OUTER_PATIENCE, FIT_OUTER_RELTOL, DEVICE, deadline)


def continuous_emission_costs(mu: np.ndarray, recs: list[continuous.ContinuousRecordingInput]) -> np.ndarray:
    """Stacked [total_windows, num_states] emission cost at the fitted `mu`,
    for diagnostics only (`diagnostics.emission_cost_contrast`) -- the SAME
    per-window cost `contextual_final_ladder`'s OT solve uses
    (`continuous.continuous_cost_fn`), just concatenated across recordings."""
    if not recs:
        return np.zeros((0, mu.shape[0]))
    return np.concatenate([continuous.continuous_cost_fn(mu, rec) for rec in recs], axis=0)


def categorical_emission_costs(theta: np.ndarray, recs: list[categorical.RecordingInput]) -> np.ndarray:
    """Categorical-arm analog of `continuous_emission_costs`, diagnostics
    only, using the SAME `emission_cost_k` cost `categorical_final_ladder`
    uses."""
    from script.fsq_reconstruction.evaluate import emission_cost_k
    if not recs:
        return np.zeros((0, theta.shape[0]))
    return np.concatenate([emission_cost_k(theta, rec.codes, LOG_K) for rec in recs], axis=0)


def score_continuous(mu: np.ndarray, score_recs: list[continuous.ContinuousRecordingInput],
                     starts_by_name: dict[str, np.ndarray], frame_counts_by_name: dict[str, int],
                     score_names: list[str], deadline: float | None = None,
                     rungs: tuple = RUNGS, predictions_out: dict | None = None) -> list[RungResult]:
    """Labels are read here (`labels.gts_for`), and ONLY here -- never inside
    `fit_continuous_arm`/`fit_and_score_categorical`'s fitting call, which
    only ever receives `RecordingInput`/`ContinuousRecordingInput`."""
    from script.quantized_input_comparison import labels as qic_labels
    gts = qic_labels.gts_for(score_names)
    return contextual_final_ladder(mu, score_recs, COEFFS, rungs, gts, starts_by_name, frame_counts_by_name,
                                   deadline, predictions_out=predictions_out, fail_on_invalid=True)
