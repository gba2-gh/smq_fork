"""Metadata-only declaration of the intended later contextualizer search.

This module enumerates and filters candidate definitions; it trains nothing and
reads no action label or segmentation score. For each tokenizer pipeline it
declares 12 unique candidates: the frozen reference plus the first 11 feasible,
unique candidates of a seed-111 permutation of the remaining definitions.
Feasibility is nominal (physical durations converted with the actual fps /
window, rounded up) and then audited on the real windows' masks, label-free.
Feasibility and adequate training are distinct questions: a feasible candidate
is not certified to train well in any particular number of updates.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from fractions import Fraction

import numpy as np

FACTORS = dict(
    position=("alibi", "sinusoidal", "none"),
    context_seconds=(8, 16, 32, 64),
    mask_fraction=(0.15, 0.30, 0.50),
    span_seconds=(0.25, 1, 2, 4),
    width=(64, 128, 256),
    layers=(2, 4, 8),
    dropout=(0.1, 0.3),
    lr=(3e-4, 1e-3),
    norm=("post", "pre"),
)
NOMINAL_COUNT = 10_368
REFERENCE = dict(position="alibi", context_seconds=32, mask_fraction=0.30, span_seconds=2, width=128, layers=4,
                 dropout=0.1, lr=3e-4, norm="post")
PERMUTATION_SEED = 111
CANDIDATES_PER_TOKENIZER = 12
HEADS = 4


@dataclass(frozen=True)
class Candidate:
    position: str
    context_seconds: float
    mask_fraction: float
    span_seconds: float
    width: int
    layers: int
    dropout: float
    lr: float
    norm: str


def all_definitions() -> list[Candidate]:
    return [Candidate(*values) for values in itertools.product(*FACTORS.values())]


def seconds_to_tokens(seconds: float, fps: int, window: int) -> int:
    """Physical duration -> tokens at the actual token duration window/fps,
    rounded UP (exact rational arithmetic, no float surprises)."""
    return math.ceil(Fraction(str(seconds)) * fps / window)


def guard_min_tokens(radius: int, window: int) -> int:
    """Tokens a masked span needs for one eligible interior target under a
    guard of `radius` raw frames; no restriction when radius is zero."""
    return 2 * math.ceil(2 * radius / window) + 1 if radius > 0 else 1


def effective_config(c: Candidate, fps: int, window: int) -> dict:
    token_s = window / fps
    context = seconds_to_tokens(c.context_seconds, fps, window)
    span = seconds_to_tokens(c.span_seconds, fps, window)
    return dict(position=c.position, context_tokens=context, span_tokens=span, mask_fraction=c.mask_fraction,
                width=c.width, layers=c.layers, heads=HEADS, d_ff=4 * c.width, dropout=c.dropout, lr=c.lr,
                norm=c.norm, realized_context_seconds=context * token_s, realized_span_seconds=span * token_s)


def effective_key(e: dict) -> tuple:
    return tuple(e[k] for k in ("position", "context_tokens", "span_tokens", "mask_fraction", "width", "layers",
                                "d_ff", "dropout", "lr", "norm"))


def nominal_rejection(e: dict, radius: int, window: int) -> str | None:
    need = guard_min_tokens(radius, window)
    if e["span_tokens"] > e["context_tokens"]:
        return "span_exceeds_context"
    target = round(e["mask_fraction"] * e["context_tokens"])
    if target < 1:
        return "no_masked_positions"
    if e["span_tokens"] < need:
        return "span_below_guard_minimum"
    if min(e["span_tokens"], target) < need:
        return "realized_span_below_guard_minimum"
    return None


def audit_masks(e: dict, loader, radius: int, train_names: list[str], val_names: list[str], family: str,
                max_recordings: int = 40) -> dict:
    """Label-free audit on actual windows: build real masks for sampled
    training chunks and all validation chunks and report eligible-target
    counts, realized span lengths and truncated chunks."""
    from script.action_transport.stageb_data import stable_seed
    from script.quantized_input_comparison import contextual as C
    from script.quantized_input_comparison import masking as qic_masking
    from script.week5.contextualizer_calibration.train import realized_runs
    ctx, stride = e["context_tokens"], max(1, e["context_tokens"] // 2)
    out = {}
    for side, names in (("train", train_names[:max_recordings]), ("val", val_names)):
        chunks = C.make_vector_chunks(loader, family, names, chunk=ctx, stride=stride)
        eligible = selected = truncated = 0
        runs: list[int] = []
        for c in chunks:
            rng = np.random.default_rng(stable_seed(c.recording, c.start, 111))
            sel, elig = qic_masking.build_eligible_mask(c.starts, c.lengths, c.valid_len, c.frame_count, rng,
                                                        radius=radius, fraction=e["mask_fraction"],
                                                        span=e["span_tokens"])
            eligible += int(elig.sum())
            selected += int(sel.sum())
            truncated += int(c.valid_len < ctx)
            runs += realized_runs(sel, c.valid_len)
        out[side] = dict(chunks=len(chunks), selected=selected, eligible=eligible, truncated_chunks=truncated,
                         mean_realized_span=float(np.mean(runs)) if runs else 0.0)
    return out


def declare(pipelines: dict, fps: int, window: int, loader=None, family_of: dict | None = None,
            train_names: list[str] | None = None, val_names: list[str] | None = None) -> dict:
    """pipelines: {name: guard_radius}. Returns the declaration (candidates,
    rejections, counts). `loader` (optional) enables the real-mask audit."""
    defs = all_definitions()
    assert len(defs) == NOMINAL_COUNT, f"expected {NOMINAL_COUNT} nominal definitions, got {len(defs)}"
    ref = Candidate(**REFERENCE)
    ref_eff = effective_config(ref, fps, window)
    rest = [d for d in defs if d != ref]
    order = np.random.default_rng(PERMUTATION_SEED).permutation(len(rest))
    result = dict(nominal_count=len(defs), factors={k: list(v) for k, v in FACTORS.items()}, reference=REFERENCE,
                  permutation_seed=PERMUTATION_SEED, fps=fps, window=window, token_seconds=window / fps,
                  pipelines={}, rejections=[])
    for name, radius in pipelines.items():
        accepted = [dict(index=-1, source="frozen_reference", **ref_eff)]
        seen = {effective_key(ref_eff)}
        scanned = 0
        for idx in order:
            if len(accepted) >= CANDIDATES_PER_TOKENIZER:
                break
            scanned += 1
            cand = rest[int(idx)]
            eff = effective_config(cand, fps, window)
            reason = nominal_rejection(eff, radius, window)
            audit = None
            if reason is None and loader is not None:
                audit = audit_masks(eff, loader, radius, train_names, val_names, family_of[name])
                if audit["train"]["eligible"] == 0 or audit["val"]["eligible"] == 0:
                    reason = "empty_objective_in_mask_audit"
            if reason is None and effective_key(eff) in seen:
                reason = "duplicate_effective_config"
            if reason is not None:
                result["rejections"].append(dict(pipeline=name, definition_index=int(idx), reason=reason,
                                                 candidate=asdict(cand), effective=eff, audit=audit))
                continue
            seen.add(effective_key(eff))
            accepted.append(dict(index=int(idx), source="permutation", audit=audit, candidate=asdict(cand), **eff))
        result["pipelines"][name] = dict(guard_radius=radius, guard_min_span_tokens=guard_min_tokens(radius, window),
                                         scanned_after_reference=scanned, candidates=accepted,
                                         complete=len(accepted) == CANDIDATES_PER_TOKENIZER)
    result["note"] = ("Declaration only; no candidate is trained here and no segmentation score admits or rejects "
                      "any candidate. Feasible spaces differ by tokenizer through the guard radius. A feasible "
                      "candidate is not certified to train adequately in any fixed number of updates.")
    return result
