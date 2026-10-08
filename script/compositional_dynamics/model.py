"""Finite-history action-conditioned unit predictor.

h = GRU over the (at most H) preceding codes only, reset for every target; v_a = h + Wup_a(GELU(Wdown_a(h)));
log p_a(c | history) = log_softmax(Wout v_a + b_a). No current or future code, recording id or subject id enters.
`bag=True` sorts the available history by a fixed random key over vocabulary ids before the SAME network."""
from __future__ import annotations

import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from script.compositional_dynamics import config as C


def _gen(seed: int, name: str) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed((seed * 1_000_003 + zlib.crc32(name.encode())) % (2 ** 63))
    return g


def bag_keys(num_codes: int = C.NUM_CODES, seed: int = C.CALIB_SEED) -> np.ndarray:
    """rank[code]: the saved seed-111 random permutation of vocabulary ids used as the sorting key."""
    return np.random.default_rng(seed).permutation(num_codes).astype(np.int64)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Exact GELU from elementary ops. F.gelu's float32 CPU backward in this environment's torch 2.1.0 returned gradients
    about 10^4 times too large when the incoming gradient was non-contiguous (as it is after the adapter einsum);
    float64 and this composition match finite differences."""
    return 0.5 * x * (1.0 + torch.erf(x * 0.7071067811865476))


class HistoryModel(nn.Module):
    def __init__(self, lookup: np.ndarray, num_actions: int, hist_len: int, seed: int, bag: bool = False,
                 hidden: int = C.HIDDEN, bottleneck: int = C.BOTTLENECK):
        super().__init__()
        self.hist_len, self.bag, self.num_actions, self.hidden = hist_len, bag, num_actions, hidden
        K, D = lookup.shape
        self.register_buffer("lookup", torch.as_tensor(lookup, dtype=torch.float32))
        self.register_buffer("bag_rank", torch.as_tensor(bag_keys(K)))
        self.proj = nn.Linear(D, hidden)
        self.cell = nn.GRUCell(hidden, hidden)
        self.down = nn.Parameter(torch.empty(num_actions, hidden, bottleneck))
        self.down_b = nn.Parameter(torch.zeros(num_actions, bottleneck))
        self.up = nn.Parameter(torch.empty(num_actions, bottleneck, hidden))
        self.up_b = nn.Parameter(torch.zeros(num_actions, hidden))
        self.out = nn.Linear(hidden, K)
        self.action_bias = nn.Parameter(torch.empty(num_actions, K))
        self.init_parameters(seed)

    def init_parameters(self, seed: int) -> None:
        """Each tensor is drawn from a generator keyed by (seed, name): independent of construction order, so shared
        backbone tensors are identical across arms and adapters are distinct."""
        with torch.no_grad():
            for name, p in self.named_parameters():
                g = _gen(seed, name)
                if name in ("down", "up", "action_bias"):
                    std = {"down": self.hidden ** -0.5, "up": C.ADAPTER_UP_STD, "action_bias": C.BIAS_STD}[name]
                    p.copy_(torch.randn(p.shape, generator=g) * std)
                elif name.endswith("_b") and name != "action_bias":
                    p.zero_()
                else:  # nn.Linear / GRUCell tensors: U(-1/sqrt(fan_in), 1/sqrt(fan_in))
                    fan = {"proj": self.proj.in_features, "cell": self.hidden, "out": self.hidden}[name.split(".")[0]]
                    p.copy_((torch.rand(p.shape, generator=g) * 2 - 1) * fan ** -0.5)

    def canonical(self, hist: torch.Tensor, hmask: torch.Tensor):
        """Bag control: valid codes first, sorted by the fixed random rank (ties are identical codes)."""
        if not self.bag:
            return hist, hmask
        key = torch.where(hmask, self.bag_rank[hist], torch.full_like(hist, self.bag_rank.numel()))
        order = torch.argsort(key, dim=1, stable=True)
        return torch.gather(hist, 1, order), torch.gather(hmask, 1, order)

    def history_repr(self, hist: torch.Tensor, hmask: torch.Tensor) -> torch.Tensor:
        hist, hmask = self.canonical(hist, hmask)
        h = torch.zeros(hist.shape[0], self.hidden, device=hist.device, dtype=self.proj.weight.dtype)
        x = self.proj(self.lookup[hist])
        for j in range(self.hist_len):
            h = torch.where(hmask[:, j:j + 1], self.cell(x[:, j], h), h)
        return h

    def logp_from_repr(self, h: torch.Tensor) -> torch.Tensor:
        """[B, C, K] log p_a(. | history)."""
        z = gelu(torch.einsum("bh,ahk->bak", h, self.down) + self.down_b)
        v = h[:, None, :] + torch.einsum("bak,akh->bah", z, self.up) + self.up_b
        return F.log_softmax(self.out(v) + self.action_bias, dim=-1)

    def logp(self, hist, hmask) -> torch.Tensor:
        return self.logp_from_repr(self.history_repr(hist, hmask))

    def target_nll(self, hist, hmask, target) -> torch.Tensor:
        """[B, C]: -log p_a(target | history). The target code never enters the network."""
        lp = self.logp(hist, hmask)
        return -torch.gather(lp, 2, target[:, None, None].expand(-1, self.num_actions, 1)).squeeze(2)


def build_model(lookup, num_actions, hist_len, seed, arm: str) -> HistoryModel:
    return HistoryModel(lookup, num_actions, hist_len, seed, bag=(arm == "m1_bag"))


def parameter_counts(model: HistoryModel) -> dict:
    total = sum(p.numel() for p in model.parameters())
    adapters = sum(p.numel() for n, p in model.named_parameters() if n in ("down", "down_b", "up", "up_b", "action_bias"))
    return dict(total=total, adapters_and_biases=adapters, backbone=total - adapters)
