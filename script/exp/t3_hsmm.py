"""
T3 -- duration-aware decoding of SMQ's patch codes, post-hoc (PREREG_T1-T4.md).

Emission log-prob per patch: log softmax(-beta * standardised distance to each code).
Decoders, none of which uses ground truth:
  argmax : the published decoding
  hmm    : Viterbi with a K x K transition matrix (self-transitions included)
           counted from the argmax code sequence, pooled over the dataset
  hsmm   : explicit-duration Viterbi -- per-code negative-binomial duration
           (support 1..Dmax patches) + between-code transitions, fitted by
           3 rounds of hard EM starting from the argmax runs

HMM vs HSMM separates "any temporal smoothing" from "an explicit duration model".
beta in {0.5, 1, 2, 4}; all four are reported, none selected.

Output: results/exp/analysis/T3/<source>/<ds>.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import gammaln, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from script.exp.analyze_run import score, jsd, hungarian_map, rle_lengths, NATIVE_FPS

BETAS = [0.5, 1.0, 2.0, 4.0]
DMAX = 100
EM_ROUNDS = 3


def emissions(dist, beta):
    s = (dist - dist.mean(1, keepdims=True)) / (dist.std(1, keepdims=True) + 1e-6)
    z = -beta * s
    return z - logsumexp(z, axis=1, keepdims=True)


def runs(codes):
    L = rle_lengths(codes)
    starts = np.concatenate(([0], np.cumsum(L)[:-1]))
    return L, codes[starts]


# ---------------------------------------------------------------- HMM -------
def fit_transitions(seqs, K, self_loops=True):
    A = np.ones((K, K))
    for c in seqs:
        if self_loops:
            np.add.at(A, (c[:-1], c[1:]), 1)
        else:
            _, v = runs(c)
            np.add.at(A, (v[:-1], v[1:]), 1)
    if not self_loops:
        np.fill_diagonal(A, 0)
    return np.log(A / A.sum(1, keepdims=True) + 1e-300)


def viterbi(E, logA):
    P, K = E.shape
    V = E[0] - np.log(K)
    back = np.zeros((P, K), dtype=np.int64)
    for t in range(1, P):
        M = V[:, None] + logA
        back[t] = M.argmax(0)
        V = M.max(0) + E[t]
    path = np.empty(P, dtype=np.int64)
    path[-1] = V.argmax()
    for t in range(P - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


# ---------------------------------------------------------------- HSMM ------
def nb_logpmf_table(lengths, dmax):
    """Negative binomial on d-1 (so support starts at 1), method of moments;
    Poisson when under-dispersed. Truncated to 1..dmax and renormalised."""
    x = np.arange(dmax)                                   # d - 1
    m = max(float(np.mean(lengths)) - 1, 1e-3)
    v = float(np.var(lengths))
    if v <= m:
        lp = x * np.log(m) - m - gammaln(x + 1)
    else:
        r = m * m / (v - m)
        p = r / (r + m)
        lp = gammaln(x + r) - gammaln(r) - gammaln(x + 1) + r * np.log(p) + x * np.log(1 - p)
    return lp - logsumexp(lp)


def fit_hsmm(seqs, K):
    all_L, per = [], {k: [] for k in range(K)}
    for c in seqs:
        L, v = runs(c)
        all_L.append(L)
        for l, k in zip(L, v):
            per[int(k)].append(min(int(l), DMAX))
    pooled = np.minimum(np.concatenate(all_L), DMAX)
    logD = np.stack([nb_logpmf_table(np.array(per[k]) if len(per[k]) >= 5 else pooled, DMAX)
                     for k in range(K)])                  # (K, DMAX), index d-1
    logA = fit_transitions(seqs, K, self_loops=False)     # between-segment transitions
    return logD, logA


def hsmm_viterbi(E, logD, logA):
    P, K = E.shape
    Ce = np.vstack([np.zeros((1, K)), np.cumsum(E, axis=0)])   # (P+1, K)
    V = np.full((P + 1, K), -np.inf)
    arg_d = np.zeros((P + 1, K), dtype=np.int64)
    arg_j = np.full((P + 1, K), -1, dtype=np.int64)
    Bst = np.full((P + 1, K), -np.inf)                          # best entry score at s
    Bj = np.full((P + 1, K), -1, dtype=np.int64)
    Bst[0] = -np.log(K)
    offdiag = logA.copy()
    np.fill_diagonal(offdiag, -np.inf)
    for t in range(1, P + 1):
        dmax = min(t, DMAX)
        s = t - np.arange(1, dmax + 1)                          # segment start for d = 1..dmax
        cand = Bst[s] + logD[:, :dmax].T + (Ce[t] - Ce[s])      # (dmax, K)
        best = cand.argmax(0)
        V[t] = cand[best, np.arange(K)]
        arg_d[t] = best + 1
        arg_j[t] = Bj[s[best], np.arange(K)]
        if t < P:
            M = V[t][:, None] + offdiag                         # (j, k)
            Bj[t] = M.argmax(0)
            Bst[t] = M.max(0)
    path = np.empty(P, dtype=np.int64)
    t, k = P, int(V[P].argmax())
    while t > 0:
        d = int(arg_d[t, k])
        path[t - d:t] = k
        j = int(arg_j[t, k])
        t, k = t - d, j
    return path


# ---------------------------------------------------------------- driver ----
def evaluate(d, patch_codes, rate=1):
    dataset = str(d["dataset"])
    W = int(d["W"])
    names = [str(n) for n in d["names"]]
    gts = [np.asarray(g, dtype=np.int64) for g in d["gt"]]
    frames = [np.repeat(c, W)[:len(g)] for c, g in zip(patch_codes, gts)]
    mapped = hungarian_map(gts, frames)
    res = score(gts, mapped)
    res["JSD"] = jsd(gts, mapped, names, dataset, rate)
    fps = NATIVE_FPS[dataset] / rate
    pl = np.concatenate([rle_lengths(f) for f in frames]) / fps
    gl = np.concatenate([rle_lengths(g) for g in gts]) / fps
    res["pred_mean_dur_s"] = round(float(pl.mean()), 3)
    res["gt_mean_dur_s"] = round(float(gl.mean()), 3)
    res["segments_pred_over_gt"] = round(len(pl) / len(gl), 3)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    d = np.load(args.dump, allow_pickle=True)
    K = int(d["K"])
    dists = [np.asarray(x, dtype=np.float64) for x in d["patch_dist"]]
    codes = [np.asarray(c, dtype=np.int64) for c in d["codes"]]

    res = {"dataset": str(d["dataset"]), "source": str(d["ckpt"]), "K": K, "W": int(d["W"]),
           "argmax": evaluate(d, codes)}

    logA_hmm = fit_transitions(codes, K, self_loops=True)
    res["hmm_self_transition_mean"] = round(float(np.exp(np.diag(logA_hmm)).mean()), 4)
    for beta in BETAS:
        E = [emissions(x, beta) for x in dists]
        hmm = [viterbi(e, logA_hmm) for e in E]
        res[f"hmm_beta{beta}"] = evaluate(d, hmm)

        cur = codes
        fitted = []
        for _ in range(EM_ROUNDS):
            logD, logA = fit_hsmm(cur, K)
            cur = [hsmm_viterbi(e, logD, logA) for e in E]
            fitted.append(round(float(np.mean([np.exp(logD[k]) @ np.arange(1, DMAX + 1)
                                               for k in range(K)])), 3))
        res[f"hsmm_beta{beta}"] = evaluate(d, cur)
        res[f"hsmm_beta{beta}"]["em_mean_duration_patches"] = fitted
        print(f"beta={beta}: hmm MoF {res[f'hmm_beta{beta}']['MoF']:.2f} | "
              f"hsmm MoF {res[f'hsmm_beta{beta}']['MoF']:.2f} F1@50 {res[f'hsmm_beta{beta}']['F1@50']:.2f}",
              flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"[t3] -> {args.out}")


if __name__ == "__main__":
    main()
