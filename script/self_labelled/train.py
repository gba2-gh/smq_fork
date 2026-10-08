"""Self-labelled training on the FITTING recordings of one fold. Nothing here receives held-out recordings or labels."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from script.action_transport.transport import Coeffs
from script.self_labelled import batched_ot as OT
from script.self_labelled import config as C
from script.self_labelled import model as M

COEFFS = Coeffs(a=0.7, beta=0.3, lam=0.05, eps=0.07)             # script.action_transport.config SETTING_T (decoding)
TRAIN_COEFFS = Coeffs(a=0.7, beta=0.3, lam=C.LAMBDA_TRAIN, eps=0.07)   # training pseudo-labels: only lam differs (see plan Section 11)


def standardize_stats(canon: dict, names: list):
    x = np.concatenate([canon[n] for n in names])
    return x.mean(0).astype(np.float32), (x.std(0) + 1e-6).astype(np.float32)


def pad_batch(arrs: list, device) -> tuple:
    L = max(len(a) for a in arrs)
    x = torch.zeros(len(arrs), L, arrs[0].shape[1], device=device)
    mask = torch.zeros(len(arrs), L, 1, device=device)
    for i, a in enumerate(arrs):
        x[i, :len(a)] = torch.as_tensor(a, device=device)
        mask[i, :len(a)] = 1.0
    return x, mask


def flat(z: torch.Tensor, sizes: list) -> torch.Tensor:
    return torch.cat([z[i, :n] for i, n in enumerate(sizes)])


def swapped_loss(za, zb, qa, qb, protos, weights, tau: float = C.TAU):
    """CE(Qa, softmax(zb . mu / tau)) + CE(Qb, softmax(za . mu / tau)), windows weighted by real length; Q carries no gradient."""
    la = -(qa.detach() * F.log_softmax(zb @ protos.T / tau, dim=1)).sum(1)
    lb = -(qb.detach() * F.log_softmax(za @ protos.T / tau, dim=1)).sum(1)
    return ((la + lb) * weights).sum() / weights.sum()


@torch.no_grad()
def embed_all(enc, windows: dict, mean, std, device) -> dict:
    enc.eval()
    out = {}
    for n, w in windows.items():
        x, m = pad_batch([(w - mean) / std], device)
        out[n] = enc(x, m)[0].cpu().numpy().astype(np.float64)
    return out


def make_views(arm: str, names: list, canon: dict, bank: dict, mean, std, rng) -> list:
    """Two views (lists of standardized window arrays, one per recording). L_noaug: the canonical feature with independent feature
    dropout. Others: two DISTINCT bank variants (per batch, the same pair for every recording)."""
    if arm == "L_noaug":
        keep = 1.0 - C.FEATURE_DROPOUT
        return [[((canon[n] - mean) / std) * (rng.random(canon[n].shape) < keep) / keep for n in names] for _ in range(2)]
    va, vb = rng.choice(C.N_VARIANTS, 2, replace=False)
    return [[(bank[n][int(v)].astype(np.float32) - mean) / std for n in names] for v in (va, vb)]


class OTFailure(RuntimeError):
    """A pseudo-labelling solve did not reach the required number of accepted steps; the cell is invalid."""


def effective_rank(z: torch.Tensor) -> float:
    """exp(entropy of the normalized singular values) of the centred embedding matrix: 1 = all embeddings on one line."""
    zc = z - z.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(zc.double())
    if float(sv.sum()) <= 1e-12:
        return 0.0
    p = sv / sv.sum()
    return float(torch.exp(-(p * torch.log(p.clamp_min(1e-300))).sum()))


def is_collapsed(log: dict) -> dict:
    """Collapse flags from the MEAN over the last COLLAPSE_WINDOW updates (reported; a collapsed L_full cell blocks adoption):
      soft      - one state above COLLAPSE_SHARE of the soft mass, or soft usage entropy below ln(MIN_EFFECTIVE_STATES);
      hard      - fewer than MIN_EFFECTIVE_STATES effective states in the HARD (argmax) assignments;
      embedding - effective rank of the batch embeddings below MIN_EFFECTIVE_STATES (embeddings on a low-dimensional set);
      prototype - two prototypes closer than PROTO_MAX_COS (indistinguishable states).
    Balanced soft mass alone cannot show distinct states: identical embeddings and prototypes give uniform soft assignments."""
    k = C.COLLAPSE_WINDOW
    m = lambda key: float(np.mean(log[key][-k:]))
    flags = dict(soft=bool(m("max_state_share") > C.COLLAPSE_SHARE or m("usage_entropy") < np.log(C.MIN_EFFECTIVE_STATES)),
                 hard=bool(m("hard_effective_states") < C.MIN_EFFECTIVE_STATES),
                 embedding=bool(m("embedding_effective_rank") < C.MIN_EFFECTIVE_STATES),
                 prototype=bool(m("max_prototype_cos") > C.PROTO_MAX_COS))
    flags["any"] = any(flags.values())
    return flags


def train(fit_names, canon: dict, bank: dict, lengths: dict, arm: str, seed: int, device, updates: int = C.UPDATES,
          ot_steps: int = C.OT_INNER_STEPS, deadline: float | None = None):
    """Returns (encoder, prototypes, (mean, std), log). L_noeq applies the OT column marginal per recording instead of per batch.
    Raises OTFailure if any pseudo-labelling solve falls short of `ot_steps` accepted steps: no update is made from such targets."""
    torch.manual_seed(seed)
    rng = np.random.default_rng([seed, 17])
    mean, std = standardize_stats(canon, fit_names)
    enc = M.Encoder().to(device)
    init_idx = np.random.default_rng([seed, 5]).choice(len(fit_names), min(len(fit_names), 64), replace=False)
    with torch.no_grad():
        z0 = torch.cat([enc(*pad_batch([(canon[fit_names[i]] - mean) / std], device))[0] for i in init_idx])
    protos = torch.nn.Parameter(M.init_prototypes(z0, C.NUM_STATES, seed).to(device))
    opt = torch.optim.AdamW(list(enc.parameters()) + [protos], lr=C.LR, weight_decay=C.WD)
    log = dict(loss=[], usage_entropy=[], min_state_share=[], max_state_share=[], hard_effective_states=[], hard_min_share=[],
               embedding_effective_rank=[], max_prototype_cos=[], ot_accepted=[])
    t0 = time.perf_counter()
    for step in range(updates):
        if deadline is not None and time.perf_counter() > deadline:
            raise TimeoutError("deadline")
        enc.train()
        protos.requires_grad_(step >= C.FREEZE_PROTOS)
        pick = rng.choice(len(fit_names), min(C.BATCH_RECS, len(fit_names)), replace=False)
        names = [fit_names[i] for i in pick]
        per_view = make_views(arm, names, canon, bank, mean, std, rng)
        sizes = [len(a) for a in per_view[0]]
        batch = OT.Batch(sizes, [lengths[n].tolist() for n in names], C.FPS, C.WINDOW, C.NUM_STATES, device)
        zs, qs, accepted = [], [], []
        for arrs in per_view:
            x, m = pad_batch(arrs, device)
            z = flat(enc(x, m), sizes)
            with torch.no_grad():
                cost = 1.0 - z.double() @ protos.detach().double().T
                log_t, n_acc, ok = OT.solve_inner(None, cost, batch, TRAIN_COEFFS, ot_steps, per_recording=(arm == "L_noeq"))
                accepted.append(int(n_acc))
                if not ok or n_acc < ot_steps:
                    log["ot_accepted"].append(accepted)
                    raise OTFailure(f"pseudo-label solve at update {step}: {n_acc} of {ot_steps} accepted steps")
                qs.append(OT.pseudo_labels(log_t, batch).float())
            zs.append(z)
        w = batch.p.float() * batch.n
        loss = swapped_loss(zs[0], zs[1], qs[0], qs[1], protos, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + [protos], C.CLIP)
        opt.step()
        with torch.no_grad():
            protos.copy_(F.normalize(protos, dim=-1))
            pw = batch.p.unsqueeze(1)
            share = (qs[0].double() * pw).sum(0)
            share = (share / share.sum()).cpu().numpy()
            hard = torch.zeros(C.NUM_STATES, dtype=torch.float64, device=batch.p.device).index_add_(0, qs[0].argmax(1), batch.p)
            hard = (hard / hard.sum()).cpu().numpy()
            gram = protos @ protos.T
            off = gram - 2.0 * torch.eye(C.NUM_STATES, device=gram.device)
        log["loss"].append(float(loss))
        log["usage_entropy"].append(float(-(share * np.log(np.clip(share, 1e-12, None))).sum()))
        log["min_state_share"].append(float(share.min()))
        log["max_state_share"].append(float(share.max()))
        log["hard_effective_states"].append(float(np.exp(-(hard * np.log(np.clip(hard, 1e-12, None))).sum())))
        log["hard_min_share"].append(float(hard.min()))
        log["embedding_effective_rank"].append(effective_rank(zs[0].detach()))
        log["max_prototype_cos"].append(float(off.max()))
        log["ot_accepted"].append(accepted)
    log["seconds"] = time.perf_counter() - t0
    log["collapse"] = is_collapsed(log)
    log["collapsed"] = log["collapse"]["any"]
    return enc, protos.detach(), (mean, std), log
