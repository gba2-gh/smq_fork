"""Stage B inference: unmasked full-sequence embeddings for one recording,
covering it with length-128 chunks at stride 64 (plus a final anchored
chunk), averaging hidden states for windows seen in multiple chunks, then
L2-normalizing (docs/plans/THREE_STAGE_MOTION_PLAN.md §4). Offline; uses
future context.
"""

from __future__ import annotations

import numpy as np
import torch

from script.action_transport.stageb_data import MAX_CONTEXT, chunk_starts
from script.action_transport.stageb_model import CodeContextModel


@torch.no_grad()
def extract_embeddings(model: CodeContextModel, codes: np.ndarray,
                       device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Returns (unit embeddings [L, d_model] float32, zero_flags [L] bool)."""
    model.eval()
    length = len(codes)
    d_model = model.cfg.d_model
    accum = np.zeros((length, d_model), dtype=np.float64)
    counts = np.zeros(length, dtype=np.float64)
    for start in chunk_starts(length):
        end = min(start + MAX_CONTEXT, length)
        valid = end - start
        chunk = np.zeros(MAX_CONTEXT, dtype=np.int64)
        chunk[:valid] = codes[start:end]
        padding = np.ones(MAX_CONTEXT, dtype=bool)
        padding[:valid] = False
        input_t = torch.as_tensor(chunk[None], dtype=torch.int64, device=device)
        pad_t = torch.as_tensor(padding[None], dtype=torch.bool, device=device)
        hidden, _ = model(input_t, pad_t)
        accum[start:end] += hidden[0, :valid].cpu().numpy().astype(np.float64)
        counts[start:end] += 1.0
    mean = accum / counts[:, None]
    norms = np.linalg.norm(mean, axis=1, keepdims=True)
    zero = norms[:, 0] <= 1e-12
    unit = mean / np.where(zero[:, None], 1.0, norms)
    unit[zero] = 0.0
    return unit.astype(np.float32), zero
