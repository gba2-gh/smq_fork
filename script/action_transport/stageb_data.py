"""Stage B data plumbing: chunking, span masking, and the frozen train/
validation split (docs/plans/THREE_STAGE_MOTION_PLAN.md §4).

Chunking is shared by training and inference: length-`MAX_CONTEXT` windows
at stride `INFER_STRIDE`, with a final chunk anchored to end exactly at the
recording's length ("adding a final anchored chunk if needed"). The plan's
own caveat -- "overlapping training chunks do not create independent
observations" -- says training uses this same overlapping scheme, not a
non-overlapping one; this module makes both share one function so they
cannot silently diverge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_CONTEXT = 128
INFER_STRIDE = 64
MASK_FRACTION = 0.30
MASK_SPAN = 8
VAL_FRACTION = 0.10


def chunk_starts(length: int, chunk: int = MAX_CONTEXT, stride: int = INFER_STRIDE) -> list[int]:
    """Start indices covering [0, length) with the final chunk anchored at
    length-chunk (or 0, if the recording is shorter than one chunk)."""
    if length <= chunk:
        return [0]
    starts = list(range(0, length - chunk + 1, stride))
    last = length - chunk
    if starts[-1] != last:
        starts.append(last)
    return starts


@dataclass
class Chunk:
    recording: str
    start: int
    codes: np.ndarray  # int64 [MAX_CONTEXT], padded with 0 (never used as a target/key)
    valid_len: int  # real (unpadded) length <= MAX_CONTEXT
    lengths: np.ndarray  # float32 [MAX_CONTEXT], real frame count per window, 0 where padded


def make_chunks(recordings: dict[str, np.ndarray], lengths_by_name: dict[str, np.ndarray],
                chunk: int = MAX_CONTEXT, stride: int = INFER_STRIDE) -> list[Chunk]:
    """One Chunk per (recording, start) from chunk_starts. Never crosses a
    recording boundary because it is built per recording."""
    out = []
    for name, codes in recordings.items():
        length_arr = lengths_by_name[name]
        for start in chunk_starts(len(codes), chunk, stride):
            end = min(start + chunk, len(codes))
            valid = end - start
            padded_codes = np.zeros(chunk, dtype=np.int64)
            padded_codes[:valid] = codes[start:end]
            padded_lengths = np.zeros(chunk, dtype=np.float32)
            padded_lengths[:valid] = length_arr[start:end]
            out.append(Chunk(name, start, padded_codes, valid, padded_lengths))
    return out


def build_mask(valid_len: int, rng: np.random.Generator, fraction: float = MASK_FRACTION,
              span: int = MASK_SPAN) -> np.ndarray:
    """Boolean mask over [0, valid_len), True at masked positions. Contiguous
    spans of `span` tokens are placed until their union reaches the target
    count, then trimmed deterministically (drop the highest indices first,
    so the kept set is always the same set for a given union) to hit the
    target exactly. Never touches padded positions (indices >= valid_len)."""
    mask = np.zeros(valid_len, dtype=bool)
    if valid_len == 0:
        return mask
    target = round(fraction * valid_len)
    if target == 0:
        return mask
    max_start = max(0, valid_len - span)
    masked_idx: set[int] = set()
    guard = 0
    while len(masked_idx) < target and guard < 10_000:
        start = int(rng.integers(0, max_start + 1))
        masked_idx.update(range(start, min(valid_len, start + span)))
        guard += 1
    ordered = sorted(masked_idx)[:target]
    mask[ordered] = True
    return mask


def stable_seed(recording: str, start: int, base: int) -> int:
    """A deterministic per-chunk seed for the fixed validation mask (must not
    depend on epoch, batch order, model seed, or the Python process).

    v1.6 fix: v1.0-v1.5 used Python's hash() of a tuple containing a string,
    which is randomized per process (PYTHONHASHSEED), so the "fixed"
    validation mask differed between runs and stage_b_001/002 are not
    bit-reproducible."""
    import zlib
    return zlib.crc32(f"{recording}|{start}|{base}".encode("utf-8")) % (2**31 - 1)


def build_validation_split(fit_recordings: list[str], frame_counts: dict[str, int],
                           seed: int, val_fraction: float = VAL_FRACTION) -> tuple[list[str], list[str]]:
    """Whole-recording split: shuffle the fitting population with `seed`,
    take recordings from the front until val_fraction of frames is reached,
    keeping at least one fitting and one validation recording (instructions
    §4). Deterministic; callers save and reuse this split."""
    if len(fit_recordings) < 2:
        raise ValueError("need at least 2 recordings to hold one out for validation")
    rng = np.random.default_rng(seed)
    order = list(fit_recordings)
    rng.shuffle(order)
    total = sum(frame_counts[n] for n in order)
    target = total * val_fraction
    val, accumulated = [], 0
    for name in order:
        if accumulated >= target or len(val) >= len(order) - 1:
            break
        val.append(name)
        accumulated += frame_counts[name]
    if not val:
        val = [order[0]]
    val_set = set(val)
    train = [n for n in fit_recordings if n not in val_set]
    if not train:
        train = [order[-1]]
        val = [n for n in val if n != train[0]]
        if not val:
            raise ValueError("cannot hold out a validation recording from only 2 recordings "
                             "without leaving fitting empty")
    return train, val
