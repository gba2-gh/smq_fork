"""Task 2: per-cell diagnostics saved alongside aggregate MoF/Edit/F1
metrics, so a cell's result is independently re-scorable/re-inspectable
rather than being derivable only from the aggregate numbers.

Computes (per review's list, quoting the spec's "Save metrics and
information" requirements: "code occupancy, vector norm/rank diagnostics,
adjacent-input similarity and emission-cost contrast"):

- predicted-vs-ground-truth median run length (frames and seconds), from
  the final rung's frame-level predictions/labels.
- adjacent-input similarity: for continuous/cosine arms, mean cosine
  similarity between consecutive real (non-padded) per-window input
  vectors (the SAME vectors fed to the OT solve); for categorical arms
  (no vectors), the analogous adjacent-SAME-CODE rate.
- emission-cost contrast: mean(second-best-state cost - best-state cost)
  at the final fitted theta/mu, i.e. how well-separated each window's
  assigned state is from its runner-up under the fitted emission model.
- vector norm/rank diagnostics: for continuous arms, the ambient dimension,
  mean/std vector norm, and entropy effective rank of a bounded sample of
  centered per-window vectors.
- code occupancy: pass-through of Task 1's own per-family
  `loader.occupancy(family)` counts (frame-weighted `counts` and unweighted
  `window_counts`) -- already computed once by Task 1, never recomputed
  here.
"""

from __future__ import annotations

import numpy as np


def median_run_length_frames(seq: np.ndarray) -> float:
    """Median contiguous-run length (in frames) of a 1D integer label
    sequence. NaN for an empty sequence."""
    seq = np.asarray(seq)
    if seq.size == 0:
        return float("nan")
    change = np.flatnonzero(np.diff(seq) != 0) + 1
    boundaries = np.concatenate(([0], change, [seq.size]))
    run_lengths = np.diff(boundaries)
    return float(np.median(run_lengths))


def _run_lengths(seq: np.ndarray) -> np.ndarray:
    seq = np.asarray(seq)
    if not seq.size:
        return np.empty(0, dtype=np.int64)
    boundaries = np.concatenate(([0], np.flatnonzero(np.diff(seq) != 0) + 1, [seq.size]))
    return np.diff(boundaries)


def run_length_diagnostics(frame_preds_by_name: dict[str, np.ndarray], frame_gts_by_name: dict[str, np.ndarray],
                           fps: float) -> dict:
    pred_runs, gt_runs = [], []
    for name, pred in frame_preds_by_name.items():
        if name.startswith("__"):
            continue
        pred_runs.append(_run_lengths(pred))
        gt = frame_gts_by_name.get(name)
        if gt is not None:
            gt_runs.append(_run_lengths(gt))
    pred_med = float(np.median(np.concatenate(pred_runs))) if pred_runs else None
    gt_med = float(np.median(np.concatenate(gt_runs))) if gt_runs else None
    return dict(pred_median_run_length_frames=pred_med, gt_median_run_length_frames=gt_med,
               pred_median_run_length_seconds=(pred_med / fps if pred_med is not None else None),
               gt_median_run_length_seconds=(gt_med / fps if gt_med is not None else None),
               n_recordings=len(pred_runs))


def adjacent_input_cosine_similarity(vectors_by_name: dict[str, np.ndarray],
                                     valid_len_by_name: dict[str, int] | None = None) -> dict:
    """`vectors_by_name[name]`: float [L, d] per-window input vectors (real
    windows only, or padded with `valid_len_by_name[name]` giving the real
    prefix length)."""
    sims = []
    for name, v in vectors_by_name.items():
        n = valid_len_by_name[name] if valid_len_by_name is not None else v.shape[0]
        v = v[:n]
        if v.shape[0] < 2:
            continue
        a, b = v[:-1], v[1:]
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        valid = (na > 1e-12) & (nb > 1e-12)
        if not valid.any():
            continue
        cos = np.sum(a[valid] * b[valid], axis=1) / (na[valid] * nb[valid])
        sims.append(cos)
    if not sims:
        return dict(mean_adjacent_cosine_similarity=None, n_pairs=0)
    allc = np.concatenate(sims)
    return dict(mean_adjacent_cosine_similarity=float(allc.mean()), n_pairs=int(allc.size))


def adjacent_same_code_rate(codes_by_name: dict[str, np.ndarray]) -> dict:
    """Categorical-arm analog of adjacent-input similarity: fraction of
    consecutive real windows sharing the same code id."""
    matches, total = 0, 0
    for codes in codes_by_name.values():
        codes = np.asarray(codes)
        if codes.size < 2:
            continue
        matches += int((codes[:-1] == codes[1:]).sum())
        total += codes.size - 1
    return dict(adjacent_same_code_rate=(matches / total if total else None), n_pairs=total)


def emission_cost_contrast(cost: np.ndarray) -> dict:
    """`cost`: [T, num_states] emission cost at every window (lower is
    better). Contrast = mean(second-best - best) across windows: how
    clearly the fitted emission model separates each window's assigned
    state from its runner-up."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0 or cost.shape[1] < 1:
        return dict(mean_best_cost=None, mean_second_best_cost=None, mean_emission_contrast=None)
    sorted_cost = np.sort(cost, axis=1)
    best = sorted_cost[:, 0]
    second = sorted_cost[:, 1] if cost.shape[1] > 1 else best
    return dict(mean_best_cost=float(best.mean()), mean_second_best_cost=float(second.mean()),
               mean_emission_contrast=float((second - best).mean()))


def rank_sample_vectors(vectors_by_name: dict[str, np.ndarray], sample_ids: list[tuple[str, int]],
                        cap: int = 512, seed: int = 111) -> np.ndarray:
    """Fixed, bounded subsample of Task 1's saved fitting-window population."""
    if not sample_ids:
        raise ValueError("rank diagnostics require saved fitting-window identifiers")
    indices = np.random.default_rng(seed).choice(len(sample_ids), min(cap, len(sample_ids)), replace=False)
    return np.stack([vectors_by_name[sample_ids[int(i)][0]][sample_ids[int(i)][1]] for i in indices])


def vector_norm_rank_diagnostics(vectors: np.ndarray | list[np.ndarray],
                                 rank_sample: np.ndarray | None = None) -> dict:
    """Norms use every supplied window; covariance rank uses at most 512
    windows chosen from Task 1's saved fitting sample. Report both counts."""
    batches = [vectors] if isinstance(vectors, np.ndarray) else vectors
    batches = [np.asarray(batch) for batch in batches if len(batch)]
    if not batches:
        return dict(mean_norm=None, std_norm=None, effective_rank=None, ambient_dim=None, n_vectors=0)
    if rank_sample is None:
        # Small synthetic callers may pass a single array directly. Large
        # experiment cells must supply a bounded sample explicitly.
        if sum(len(batch) for batch in batches) > 512:
            raise ValueError("large vector collections require a bounded rank_sample")
        rank_sample = np.concatenate(batches)
    norms = np.concatenate([np.linalg.norm(batch.astype(np.float64), axis=1) for batch in batches])
    sample = np.asarray(rank_sample, dtype=np.float64)
    if sample.ndim != 2 or sample.shape[1] != batches[0].shape[1] or len(sample) > 512:
        raise ValueError("rank_sample must have shape [<=512, vector_dim]")
    centered = sample - sample.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    eig = np.linalg.eigvalsh(gram)
    eig = np.maximum(eig, 0.0)
    positive = eig[eig > (1e-6 * eig[-1] if eig.size else 0.0)]
    if positive.size:
        probs = positive / positive.sum()
        eff_rank = float(np.exp(-(probs * np.log(probs)).sum()))
    else:
        eff_rank = 0.0
    return dict(mean_norm=float(norms.mean()), std_norm=float(norms.std()), effective_rank=eff_rank,
                numerical_rank=int(len(positive)), rank_sample_size=int(len(sample)),
                ambient_dim=int(sample.shape[1]), n_vectors=int(sum(len(batch) for batch in batches)))
