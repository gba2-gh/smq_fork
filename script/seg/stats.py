"""
Paired resampling statistics.

Every metric is a function of per-recording arrays and a weight vector w (one weight per
recording). A resample draws independent GROUPS (participants for LARa; recordings for BABEL,
which has no participant identifiers) with replacement and sets w[r] = multiplicity of r's group.
The same draw is applied to every condition, so comparisons stay paired. Draw 0 is the
original sample (w = 1). Clustering-seed and randomisation variability are reported separately.
"""
import numpy as np

from script.seg import metrics as M

B_DRAWS = 2000
BOOT_SEED = 12345
MIN_CLASS_QUERIES = 20


def make_weights(group, B=B_DRAWS, seed=BOOT_SEED, subset=None):
    ug, inv = np.unique(group, return_inverse=True)
    rng = np.random.default_rng(seed)
    cnt = rng.multinomial(len(ug), np.ones(len(ug)) / len(ug), size=B)          # (B, G)
    W = np.vstack([np.ones(len(group)), cnt[:, inv].astype(np.float64)])
    if subset is not None:
        W = W * subset[None, :]
    return W


def stack_condition(res, key_prefix=None):
    """res: {seed: dict} (fixed/oracle) or list of such dicts (random) or single dict (smq)
    -> dict of arrays with an extra axis M (members: seeds x randomisations)."""
    members = []
    if isinstance(res, dict) and "cont" in res:
        members = [res]
    elif isinstance(res, dict):
        members = [res[s] for s in sorted(res)]
    else:
        for r in res:
            members += [r[s] for s in sorted(r)]
    out = {}
    for k in members[0]:
        if k in ("hmap", "tmap", "fit_nseg"):
            continue
        out[k] = np.stack([np.asarray(m[k]) for m in members], axis=1)       # (R, M, ...)
    out["_M"] = len(members)
    return out


def cb_retrieval(W, arr, prefix, classes):
    """Class-balanced precision, chance and lift for every draw and member. -> dict of (B, M)"""
    S, n, ch = arr[f"{prefix}_S"], arr[f"{prefix}_n"], arr[f"{prefix}_ch"]          # (R, M, C)
    Sb = np.einsum("br,rmc->bmc", W, S)
    nb = np.einsum("br,rmc->bmc", W, n)
    cb = np.einsum("br,rmc->bmc", W, ch)
    ok = np.zeros(S.shape[-1], bool); ok[classes] = True
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(ok, Sb / nb, np.nan)
        c = np.where(ok, cb / nb, np.nan)
    return dict(p=np.nanmean(p, -1), chance=np.nanmean(c, -1), lift=np.nanmean(p - c, -1), per_class=p)


def eval_classes(arr, prefix, min_q=MIN_CLASS_QUERIES):
    n = arr[f"{prefix}_n"].sum(0).mean(0)          # (C,) mean over members
    return np.flatnonzero(n >= min_q)


def contingency_metrics(W, arr, fold_of_rec, nframes=None):
    """NMI/ARI (mean over folds), Hungarian benchmark MoF (mapping re-estimated on every draw,
    pooled over folds) and train-fitted-map MoF, for every draw and member. -> dict of (B, M)"""
    cont = arr["cont"]                                # (R, M, K, C)
    Rn, Mn, K, C = cont.shape
    Bn = W.shape[0]
    nmi_sum = np.zeros((Bn, Mn)); ari_sum = np.zeros((Bn, Mn)); n_fold = np.zeros((Bn, Mn))
    hcor = np.zeros((Bn, Mn)); tot = np.zeros((Bn, Mn))
    for f in np.unique(fold_of_rec):
        Wf = W * (fold_of_rec == f)[None, :]
        T = np.einsum("br,rmkc->bmkc", Wf, cont)      # (B, M, K, C)
        for b in range(Bn):
            for m in range(Mn):
                t = T[b, m]
                s = t.sum()
                if s == 0:
                    continue
                nmi_sum[b, m] += M.nmi_from_contingency(t)
                ari_sum[b, m] += M.ari_from_contingency(t)
                n_fold[b, m] += 1
                hm = M.hungarian_map(t)
                hcor[b, m] += sum(t[k, hm[k]] for k in range(K) if hm[k] >= 0)
                tot[b, m] += s
    with np.errstate(invalid="ignore", divide="ignore"):
        nmi = nmi_sum / n_fold; ari = ari_sum / n_fold
        mof_h = hcor / tot
        tc = np.einsum("br,rm->bm", W, arr["tcorrect"]) / np.einsum("br,rm->bm", W, arr["nframes"])
    return dict(nmi=nmi, ari=ari, mof_hungarian=mof_h, mof_transfer=tc)


def f1_metrics(W, arr, prefix="h"):
    """F1@10/25/50 (pooled tp/fp/fn), Edit (mean over recordings), MoF for a mapping kind."""
    f = arr[f"f1_{prefix}"]                            # (R, M, 3 overlaps, 3 [tp,fp,fn])
    T = np.einsum("br,rmok->bmok", W, f)
    tp, fp, fn = T[..., 0], T[..., 1], T[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        p = tp / (tp + fp); r = tp / (tp + fn)
        f1 = 100 * 2 * p * r / (p + r)
    f1 = np.nan_to_num(f1)
    edit = np.einsum("br,rm->bm", W, arr[f"edit_{prefix}"]) / W.sum(1, keepdims=True)
    return dict(F1_10=f1[..., 0], F1_25=f1[..., 1], F1_50=f1[..., 2], Edit=edit)


def boundary_metrics(W, arr, name):
    """Class-agnostic boundary P/R/F1, tolerance index j (0: 0.5 s, 1: 1.0 s)."""
    a = arr[name]                                      # (R, M, J, 3)
    T = np.einsum("br,rmjk->bmjk", W, a)
    tp, fp, fn = T[..., 0], T[..., 1], T[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        p = tp / (tp + fp); r = tp / (tp + fn)
        f = 2 * p * r / (p + r)
    return dict(P=100 * p, R=100 * r, F1=100 * np.nan_to_num(f))


def ci(x, level=0.95):
    """percentile interval of draws 1..B of a (B+1,) array; point estimate = draw 0."""
    d = x[1:]
    d = d[np.isfinite(d)]
    lo, hi = np.percentile(d, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(x[0]), float(lo), float(hi)
