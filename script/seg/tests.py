"""
Verification of the implementation components before any outcome is computed.
Writes results/seg/implementation_checks.json ; exits non-zero on failure.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from script.seg import metrics as M  # noqa: E402
from script.seg import partition as P  # noqa: E402
from script.seg import represent as R  # noqa: E402

OUT = ROOT / "results" / "seg"
OUT.mkdir(parents=True, exist_ok=True)
res = {}


def check(name, cond, detail=None):
    res[name] = dict(passed=bool(cond), detail=detail)
    print(("PASS " if cond else "FAIL ") + name, detail if detail is not None else "")


# ------------------------------------------------ 1. representation keeps temporal order
def ramp(L, D=4, rev=False):
    t = np.linspace(0, 1, L)
    if rev:
        t = t[::-1]
    z = np.zeros((L, D)); z[:, 0] = t; z[:, 1] = t ** 2
    return z


def vec(z):
    return R.bin_segments(z.astype(np.float32), np.array([0, len(z)]), 8)[0].reshape(-1)


for L in (64, 40, 13, 8, 7, 5, 3, 2):
    b = R.bin_segments(ramp(L).astype(np.float32), np.array([0, L]), 8)[0]
    mono = bool(np.all(np.diff(b[:, 0]) >= -1e-6))
    rb = R.bin_segments(ramp(L, rev=True).astype(np.float32), np.array([0, L]), 8)[0]
    rev_ok = bool(np.allclose(rb[:, 0], b[::-1, 0], atol=1e-5))
    check(f"bins_monotone_and_reverse_L{L}", mono and rev_ok, dict(monotone=mono, reverse_equals_flip=rev_ok))
b1 = R.bin_segments(np.ones((1, 4), dtype=np.float32) * 3, np.array([0, 1]), 8)[0]
check("single_frame_constant", np.allclose(b1, 3.0))
z = np.random.default_rng(0).normal(size=(64, 5)).astype(np.float32)
bb = R.bin_segments(z, np.array([0, 64]), 8)[0]
check("multiple_of_8_equals_block_mean", np.allclose(bb, z.reshape(8, 8, 5).mean(1), atol=1e-5))
# order sensitivity vs speed invariance: same shape at different durations is close; time reversal is far
a, a_slow, a_rev = vec(ramp(40)), vec(ramp(60)), vec(ramp(40, rev=True))
d_speed, d_rev = np.linalg.norm(a - a_slow), np.linalg.norm(a - a_rev)
check("time_reversal_farther_than_speed_change", d_rev > 5 * d_speed, dict(speed=float(d_speed), reversal=float(d_rev)))
# asymmetric example (same mean, opposite temporal order): fast rise then slow decay vs its reversal
asym = np.zeros((64, 2)); asym[:16, 0] = np.linspace(0, 1, 16); asym[16:, 0] = np.linspace(1, 0, 48)
check("asymmetric_shape_reversal_separated",
      np.linalg.norm(vec(asym) - vec(asym[::-1].copy())) > 0.5 and abs(asym.mean(0)[0] - asym[::-1].mean(0)[0]) < 1e-9)

# ------------------------------------------------ 2. partition invariants
rng = np.random.default_rng(1)
labs = np.repeat(rng.integers(0, 4, 12), rng.integers(1, 30, 12))
T = len(labs)
oe = P.oracle_edges(labs)
fe = P.fixed_edges(T, 10)
check("fixed_covers_exactly", P.check_partition(fe, T) and fe[-1] == T)
check("oracle_covers_exactly", P.check_partition(oe, T))
ok = True
for s in range(20):
    re = P.random_edges(P.durations(oe), np.random.default_rng(s))
    ok &= P.check_partition(re, T) and sorted(P.durations(re)) == sorted(P.durations(oe))
check("random_cover_and_duration_multiset", ok)
ids = P.frame_segment_ids(oe, T)
check("every_frame_in_exactly_one_segment", len(ids) == T and np.array_equal(np.bincount(ids), P.durations(oe)))

# ------------------------------------------------ 3. DP exactness and cost function
rng = np.random.default_rng(2)
worst = 0.0
n_ok = 0
for trial in range(12):
    T = int(rng.integers(12, 19)); D = 3
    Z = np.cumsum(rng.normal(size=(T, D)), 0) + (rng.random((T, D)) > 0.9) * rng.normal(size=(T, D)) * 3
    Lmin, Lmax = 2, 7
    cb = P.linear_fit_cost_end_indexed(Z, Lmin, Lmax, device="cuda")
    # cost function vs numpy least squares
    for (a, b) in [(0, 5), (3, 9), (T - 6, T)]:
        if b - a >= Lmin and b - a <= Lmax and b <= T:
            x = np.arange(b - a)
            r = sum(((Z[a:b, d] - np.polyval(np.polyfit(x, Z[a:b, d], 1), x)) ** 2).sum() for d in range(D)) / D
            worst = max(worst, abs(r - cb[b, (b - a) - Lmin]))
    for lam in (0.05, 0.5, 3.0):
        e, info = P.dp_segment(cb, Lmin, Lmax, lam)
        bf_val, bf_edges = P.brute_force_dp(cb, Lmin, Lmax, lam)
        val = P.partition_cost(cb, e, Lmin) + lam * (len(e) - 1)
        n_ok += int(abs(val - bf_val) < 1e-9)
check("linear_fit_cost_matches_polyfit", worst < 1e-8, dict(max_abs_err=float(worst)))
check("dp_equals_brute_force_optimum", n_ok == 36, dict(matches=n_ok, of=36))
# bounds respected and exact cover on a bigger random case
Z = np.cumsum(np.random.default_rng(3).normal(size=(400, 6)), 0)
cb = P.linear_fit_cost_end_indexed(Z, 10, 100, device="cuda")
e, _ = P.dp_segment(cb, 10, 100, 2.0)
check("dp_bounds_and_cover", P.check_partition(e, 400) and P.durations(e).min() >= 10 and P.durations(e).max() <= 100)

# ------------------------------------------------ 4. boundary matching (one-to-one, empties)
tp, fp, fn = M.match_boundaries([9, 11], [10], 1)
check("boundary_duplicate_prediction_counterexample", (tp, fp, fn) == (1, 1, 0),
      dict(tp=tp, fp=fp, fn=fn, f1_one_to_one=M.prf(tp, fp, fn)[2]))
check("boundary_empty_prediction", M.match_boundaries([], [5, 9], 2) == (0, 0, 2))
check("boundary_empty_groundtruth", M.match_boundaries([5, 9], [], 2) == (0, 2, 0))
check("boundary_both_empty", M.match_boundaries([], [], 2) == (0, 0, 0))
check("boundary_max_matching", M.match_boundaries([10, 12], [11, 13], 1) == (2, 0, 0))

# ------------------------------------------------ 5. retrieval: tie-aware P@k vs Monte-Carlo brute force
rng = np.random.default_rng(4)
Cn = 3
Q, G = 6, 9
Dm = torch.tensor(rng.random((Q, G)), dtype=torch.float32)
Gcnt = torch.tensor(rng.integers(0, 4, (G, Cn)), dtype=torch.float32)
Qw = torch.tensor(rng.integers(0, 3, (Q, Cn)), dtype=torch.float32)
qg = torch.tensor(rng.integers(0, 3, Q)); gg = torch.tensor(rng.integers(0, 3, G))
Dm[:, 1] = Dm[:, 2]   # force an exact tie between two gallery items
Pq, ch, vq = M.precision_at_k(Dm, Qw, Gcnt, qg, gg, k=5)
S = Qw.double() * Pq * vq[:, None]
worst = 0.0
for q in range(Q):
    # explode gallery items into individual anchors, excluding same group
    items = [(g, c) for g in range(G) for c in range(Cn) for _ in range(int(Gcnt[g, c])) if gg[g] != qg[q]]
    if not items:
        continue
    exp = np.zeros(Cn)
    for c in range(Cn):
        vals = []
        for _ in range(3000):
            perm = rng.permutation(len(items))
            order = sorted(perm, key=lambda i: (float(Dm[q, items[i][0]]), rng.random()))
            top = [items[i] for i in order[:5]]
            vals.append(np.mean([t[1] == c for t in top]))
        exp[c] = np.mean(vals) * float(Qw[q, c])
    worst = max(worst, float(np.abs(exp - S[q].numpy()).max()))
check("precision_at_k_matches_bruteforce_with_ties_and_group_exclusion", worst < 0.06, dict(max_abs_err=worst))

# ------------------------------------------------ 6. NMI / ARI / Hungarian vs reference implementations
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # noqa: E402
from src.model.eval_utils import create_correspondences  # noqa: E402
rng = np.random.default_rng(5)
lab = rng.integers(0, 5, 2000); pr = np.where(rng.random(2000) < 0.6, lab, rng.integers(0, 5, 2000))
Mt = np.zeros((5, 5))
np.add.at(Mt, (pr, lab), 1)
check("nmi_matches_sklearn", abs(M.nmi_from_contingency(Mt) - normalized_mutual_info_score(lab, pr)) < 1e-9)
check("ari_matches_sklearn", abs(M.ari_from_contingency(Mt) - adjusted_rand_score(lab, pr)) < 1e-9)
hm = M.hungarian_map(Mt)
_, pr2gt = create_correspondences(lab, pr, mapping=True)
check("hungarian_matches_repo_create_correspondences", all(hm[k] == v for k, v in pr2gt.items()))

(OUT / "implementation_checks.json").write_text(json.dumps(res, indent=1))
bad = [k for k, v in res.items() if not v["passed"]]
print("\nFAILED:" if bad else "\nALL PASSED", bad)
sys.exit(1 if bad else 0)
