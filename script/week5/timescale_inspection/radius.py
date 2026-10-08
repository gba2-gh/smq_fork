"""Phase 1: the OT kernel radius as an explicit, recorded setting.

The shared solver computes the kernel half-width inside
`transport.kernel_half_width(fps, window, num_windows)`. Rather than editing the
shared modules (which would change the identity hashes of finished runs), this
module scopes a replacement of that one function around a decode and records
every (num_windows -> half-width) pair it actually returned. The replacement at
the `current` setting calls the original function, so it is the same code path.

Settings (half-width in tokens; 0 switches the temporal term off, as in the
solver; every value is clamped to num_windows - 1 exactly like the original):

  current        fps // window, the rule used by every run so far
  fixed_<r>      r tokens
  rel_<p>        max(1, floor(p% of the recording's token count))   (ASOT-style)
  sec_<s>        max(1, round_half_up(s seconds * fps / window))     (fixed in seconds)

The kernel weight stays L / b (`transport.build_kernel`), so changing b changes
the reach of the neighbourhood and the per-neighbour weight together; the total
stays fixed. That is the existing normalization and is kept.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
from dataclasses import dataclass

from script.action_transport import transport


@dataclass(frozen=True)
class Setting:
    name: str
    family: str  # current | fixed | rel | sec
    value: float = 0.0

    def half_width(self, fps: int, window: int, num_windows: int) -> int:
        if num_windows <= 1:
            return 0
        if self.family == "current":
            return _ORIGINAL(fps, window, num_windows)
        if self.family == "fixed":
            r = int(self.value)
        elif self.family == "rel":
            r = max(1, math.floor(self.value / 100.0 * num_windows))
        elif self.family == "sec":
            r = max(1, math.floor(self.value * fps / window + 0.5))
        else:
            raise ValueError(self.family)
        return min(num_windows - 1, max(0, r))


_ORIGINAL = transport.kernel_half_width

CURRENT = Setting("current", "current")
FIXED = tuple(Setting(f"fixed_{r}", "fixed", r) for r in (0, 1, 2, 3, 4))
REL = tuple(Setting(f"rel_{p}", "rel", p) for p in (2, 4, 8))
SEC = tuple(Setting(f"sec_{s}", "sec", s) for s in (0.25, 0.5, 1.0))
SCREEN = (CURRENT,) + FIXED + REL + SEC
BY_NAME = {s.name: s for s in SCREEN}


def signature(setting: Setting, fps: int, window: int, lengths: list[int]) -> str:
    """Hash of the half-width the setting yields for every distinct recording length in this dataset.
    Two settings with equal signatures run the identical solve on that dataset."""
    pairs = sorted({int(n): setting.half_width(fps, window, int(n)) for n in lengths}.items())
    return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()[:16]


def radius_table(setting: Setting, fps: int, window: int, lengths: list[int]) -> dict:
    r = [setting.half_width(fps, window, int(n)) for n in lengths]
    r_sorted = sorted(r)
    return dict(median=float(r_sorted[len(r) // 2]), minimum=min(r), maximum=max(r),
                share_zero=sum(x == 0 for x in r) / len(r),
                median_seconds=float(r_sorted[len(r) // 2] * window / fps))


@contextlib.contextmanager
def kernel_override(setting: Setting):
    """Scope `setting` over every kernel the solver builds; yields the dict {num_windows: half_width} it returned."""
    used: dict[int, int] = {}

    def replacement(fps: int, window: int, num_windows: int) -> int:
        r = setting.half_width(fps, window, num_windows)
        used[int(num_windows)] = r
        return r

    transport.kernel_half_width = replacement
    try:
        yield used
    finally:
        transport.kernel_half_width = _ORIGINAL
