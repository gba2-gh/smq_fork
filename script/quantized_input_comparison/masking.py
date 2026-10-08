"""Receptive-field-aware masking eligibility.

Reuses `script.action_transport.stageb_data.build_mask` (30% span-8 masking,
identical to the existing contextualizer pipeline) to pick which window
positions in a chunk are masked, then filters those down to the positions
whose raw-frame support does not overlap the raw-frame support of any
*visible* (unmasked, real) window in the same chunk -- so a masked target's
label can never leak into the contextualizer's input through the shared
receptive field. The same conservative radius-14 rule is applied to all
five families (they run on the same underlying window grid).
"""

from __future__ import annotations

import numpy as np

from script.action_transport.stageb_data import MASK_FRACTION, MASK_SPAN, build_mask

RADIUS_DEFAULT = 14


def window_support(starts: np.ndarray, lengths: np.ndarray, frame_count: int,
                    radius: int = RADIUS_DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """Per-window raw-frame support [lo, hi] (inclusive), clipped to the
    recording's [0, frame_count-1]. `starts`/`lengths` may include padded
    (zero-length) trailing entries; their support is degenerate but never
    used since they are never "real" (see `eligible_loss_positions`)."""
    starts = np.asarray(starts, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    lo = np.clip(starts - radius, 0, max(frame_count - 1, 0))
    hi = np.clip(starts + np.maximum(lengths, 1) - 1 + radius, 0, max(frame_count - 1, 0))
    return lo, hi


def eligible_loss_positions(starts: np.ndarray, lengths: np.ndarray, valid_len: int, frame_count: int,
                             selected_mask: np.ndarray, radius: int = RADIUS_DEFAULT) -> np.ndarray:
    """Given a chunk's window `starts`/`lengths` (length P, the padded chunk
    length), the number of real (non-padded) positions `valid_len`, the
    recording's total `frame_count`, and a boolean `selected_mask` [P]
    (True = chosen for masking by `build_mask`, including edge positions),
    returns a boolean [P] array: True only at masked, real positions whose
    raw-frame support is disjoint from the support of every visible (real,
    unmasked) window in the chunk. All-False (a legitimate, explicit
    "zero eligible targets" result) is returned when no masked position
    qualifies -- callers must skip such chunks rather than treat it as an
    error."""
    P = len(starts)
    selected_mask = np.asarray(selected_mask, dtype=bool)
    real = np.arange(P) < valid_len
    lo, hi = window_support(starts, lengths, frame_count, radius)
    visible = real & ~selected_mask
    vis_lo, vis_hi = lo[visible], hi[visible]
    eligible = np.zeros(P, dtype=bool)
    candidates = np.flatnonzero(selected_mask & real)
    for t in candidates:
        if vis_lo.size == 0:
            eligible[t] = True  # no visible windows at all -> nothing to leak into
            continue
        disjoint = (hi[t] < vis_lo) | (vis_hi < lo[t])
        eligible[t] = bool(disjoint.all())
    return eligible


def build_eligible_mask(starts: np.ndarray, lengths: np.ndarray, valid_len: int, frame_count: int,
                         rng: np.random.Generator, radius: int = RADIUS_DEFAULT,
                         fraction: float = MASK_FRACTION, span: int = MASK_SPAN) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: draws the 30%/span-8 mask over [0, valid_len)
    (`build_mask`, unmodified) and returns (selected_mask, eligible_mask),
    both boolean [P] over the full padded chunk length. `selected_mask`
    marks every position chosen for masking, including ineligible edge
    targets (per spec: "Mask all selected positions ... Return zero
    eligible targets explicitly")."""
    P = len(starts)
    selected = np.zeros(P, dtype=bool)
    selected[:valid_len] = build_mask(valid_len, rng, fraction=fraction, span=span)
    eligible = eligible_loss_positions(starts, lengths, valid_len, frame_count, selected, radius=radius)
    return selected, eligible
