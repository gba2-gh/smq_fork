"""Cooperative deadline checks used inside the fitting/inference loops
(instructions §8), plus the small set of status values that are not part of
the numerical solver's own vocabulary."""

from __future__ import annotations

import time


class DeadlineExceeded(Exception):
    """Raised by check_deadline; caught at the per-cell boundary in run.py,
    which marks the cell 'interrupted' (distinct from 'capped': an
    interrupted cell was not run to its own stopping rule at all)."""


def check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.perf_counter() > deadline:
        raise DeadlineExceeded()
