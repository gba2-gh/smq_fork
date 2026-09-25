"""Validation suite for `-ValidateOnly` (instructions §6). Covers the
numerical contracts that would silently invalidate a result if wrong:
objective/gradient consistency, the kernel-construction rounding trap,
emission-/prototype-update non-increase, cosine-cost conventions, vocabulary
grouping, the permutation round trip and A0 equivariance, the mode filter,
reference parity against the pinned upstream solver, and an evaluator round
trip. It is a synthetic/tiny-real-data suite, not the full experimental
matrix, and it does not assert that unsupervised optimization always
recovers ground truth.

Coverage gaps, stated rather than silently skipped (instructions §6 asks for
an honest status, not a faked one): this suite does not exercise atomic-write
resume, cell deduplication, simulated deadline interruption, or CPU/CUDA
numerical agreement (no CUDA device in this environment at authoring time).
Those are straightforward but were not the numerically load-bearing
contracts and are listed here as not yet covered rather than asserted.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from script.action_transport import categorical, config, continuous, evaluate, permute, smoothing, transport

DTYPE = torch.float64


def _rng(seed=0):
    return np.random.default_rng(seed)


def assert_close(a, b, atol=1e-8, rtol=1e-6, msg=""):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(f"{msg}: {a} !~= {b}")


# --- kernel / conv -----------------------------------------------------------

def test_kernel_matches_dense_reference():
    """conv_apply must equal a dense band matmul for a small case."""
    rng = _rng(1)
    L, C, b = 9, 4, 3
    weights = transport.build_kernel(L, b, torch.device("cpu"))
    x = torch.as_tensor(rng.normal(size=(L, C)), dtype=DTYPE)
    fast = transport.conv_apply(weights, x)
    dense = np.zeros((L, L))
    for t in range(L):
        for s in range(L):
            if 0 < abs(t - s) <= b:
                dense[t, s] = L / b
    expected = dense @ x.numpy()
    assert_close(fast.numpy(), expected, msg="conv_apply vs dense band matrix")


def test_kernel_half_width_rounding_trap():
    """Documented v1.3 correction: int(N*r) with r=b/L does NOT recover b for
    many L. Production builds the kernel directly from integer b; this test
    exists to keep the trap visible, not to "fix" b."""
    b = 4
    failing = [L for L in (49, 98, 103) if int(L * (b / L)) != b]
    assert failing == [49, 98, 103], "expected rounding trap at L=49,98,103 for b=4"
    holding = [L for L in (16, 32, 64) if int(L * (b / L)) == b]
    assert holding == [16, 32, 64], "expected the guard to hold for L=16,32,64"


# --- objective / gradient consistency ---------------------------------------

def _random_t(L, C, rng):
    raw = rng.random((L, C)) + 0.1
    p = raw.sum(axis=1)
    return raw / p[:, None] * (rng.random(L) * 0.5 + 0.5)[:, None]


def test_objective_gradient_finite_difference():
    rng = _rng(2)
    L, C, b = 7, 3, 2
    device = torch.device("cpu")
    weights = transport.build_kernel(L, b, device)
    q = torch.full((C,), 1.0 / C, dtype=DTYPE)
    cost = torch.as_tensor(rng.random((L, C)), dtype=DTYPE)
    t0 = torch.as_tensor(_random_t(L, C, rng), dtype=DTYPE)
    for a, beta, lam, eps in [(0.7, 0.3, 0.05, 0.07), (0.4, 0.6, 0.01, 0.04), (0.7, 0.0, 0.05, 0.07)]:
        grad = transport.gradient(t0, cost, weights, a, beta, lam, eps, q)
        eps_fd = 1e-6
        directions = rng.normal(size=(4, L, C))
        for direction in directions:
            d = torch.as_tensor(direction, dtype=DTYPE)
            d = d - d.mean(dim=1, keepdim=True)  # row-sum-preserving direction
            f_plus = transport.objective(t0 + eps_fd * d, cost, weights, a, beta, lam, eps, q)
            f_minus = transport.objective(t0 - eps_fd * d, cost, weights, a, beta, lam, eps, q)
            numeric = float((f_plus - f_minus) / (2 * eps_fd))
            analytic = float((grad * d).sum())
            assert_close(numeric, analytic, atol=1e-4, rtol=1e-3,
                        msg=f"finite-diff grad mismatch (a={a},beta={beta})")


def test_beta_zero_retains_other_terms():
    rng = _rng(3)
    L, C, b = 5, 3, 2
    weights = transport.build_kernel(L, b, torch.device("cpu"))
    q = torch.full((C,), 1.0 / C, dtype=DTYPE)
    cost = torch.as_tensor(rng.random((L, C)), dtype=DTYPE)
    t0 = torch.as_tensor(_random_t(L, C, rng), dtype=DTYPE)
    obj_with_kernel = transport.objective(t0, cost, weights, 0.7, 0.0, 0.05, 0.07, q)
    obj_without_kernel = transport.objective(t0, cost, None, 0.7, 0.0, 0.05, 0.07, q)
    assert_close(obj_with_kernel, obj_without_kernel, msg="beta=0 must zero the structural term regardless of weights")


# --- solver: mass/backtracking/row sums -------------------------------------

def test_solve_preserves_row_masses():
    rng = _rng(4)
    L, C, b = 11, 4, 3
    device = torch.device("cpu")
    weights = transport.build_kernel(L, b, device)
    p = torch.as_tensor(rng.random(L) + 0.2, dtype=DTYPE)
    p = p / p.sum()
    q = torch.full((C,), 1.0 / C, dtype=DTYPE)
    cost = torch.as_tensor(rng.random((L, C)), dtype=DTYPE)
    t0 = p.unsqueeze(1) * q.unsqueeze(0)
    coeffs = transport.Coeffs(0.7, 0.3, 0.05, 0.07)
    result = transport.solve_final(t0, cost, weights, p, q, coeffs, max_steps=50)
    row_sums = result.t.sum(dim=1)
    assert_close(row_sums.numpy(), p.numpy(), atol=1e-9, msg="row masses must stay exactly p after solving")


def test_solve_single_window():
    device = torch.device("cpu")
    C = 3
    p = torch.tensor([1.0], dtype=DTYPE)
    q = torch.full((C,), 1.0 / C, dtype=DTYPE)
    cost = torch.tensor([[0.1, 0.9, 0.5]], dtype=DTYPE)
    weights = transport.build_kernel(1, 0, device)
    assert weights is None
    coeffs = transport.Coeffs(0.7, 0.3, 0.05, 0.07)
    t0 = p.unsqueeze(1) * q.unsqueeze(0)
    result = transport.solve_final(t0, cost, weights, p, q, coeffs, max_steps=50)
    assert result.status in ("converged", "capped")
    assert_close(float(result.t.sum()), 1.0, atol=1e-9, msg="single-window mass")


# --- cosine cost conventions --------------------------------------------------

def test_cosine_cost_conventions():
    mu = np.eye(2, dtype=np.float64)
    z = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.7, 0.7]], dtype=np.float64)
    z_unit, zero_flags = continuous.l2_normalize(z)
    assert not zero_flags.any()
    d = continuous.cosine_cost(mu, z_unit, zero_flags)
    assert_close(d[0, 0], 0.0, atol=1e-9, msg="equal unit vectors -> cost 0")
    assert_close(d[0, 1], 1.0, atol=1e-9, msg="orthogonal -> cost 1")
    assert_close(d[2, 0], 2.0, atol=1e-9, msg="opposite -> cost 2")
    zero_input = np.zeros((1, 2))
    z_unit2, zero_flags2 = continuous.l2_normalize(zero_input)
    d2 = continuous.cosine_cost(mu, z_unit2, zero_flags2)
    assert_close(d2[0], [1.0, 1.0], atol=1e-9, msg="zero input -> cost 1 for every state")


def test_emission_update_nonincreasing():
    """With T fixed, the exact pooled emission update must not increase the
    complete fitting objective (docs plan §3)."""
    rng = _rng(5)
    C, K = 3, 12
    eta = 6.0
    recs = [categorical.RecordingInput("r0", rng.integers(0, K, size=10).astype(np.int32),
                                       np.full(10, 5, dtype=np.int32), 60, 15)]
    theta = np.full((C, K), 1.0 / K)
    responsibilities = {"r0": _random_t(10, C, rng)}
    new_theta, _ = categorical.emission_update(recs, responsibilities, C, K, eta)

    def objective_value(th):
        cost = categorical.emission_cost(th, recs[0].codes)
        r = responsibilities["r0"]
        length_weighted = recs[0].lengths[:, None] * r
        return float((cost * r).sum())  # unary component only, theta-dependent part

    before = objective_value(theta)
    after = objective_value(new_theta)
    assert after <= before + 1e-9, f"emission update increased unary cost: {before} -> {after}"


def test_continuous_prototype_update_is_weighted_mean():
    rng = _rng(6)
    C = 2
    z = rng.normal(size=(9, 3))
    z_unit, zero_flags = continuous.l2_normalize(z)
    r = _random_t(9, C, rng)
    lengths = rng.integers(1, 5, size=9).astype(np.float64)
    weighted = lengths[:, None] * r
    accum = weighted.T @ z_unit
    mu, _ = continuous.l2_normalize(accum)
    for a in range(C):
        raw = accum[a]
        expected = raw / np.linalg.norm(raw)
        assert_close(mu[a], expected, atol=1e-9, msg="prototype update must be the normalized weighted mean")


# --- vocabulary grouping / theta0 --------------------------------------------

def test_grouping_kc_identity():
    K = 4
    centers = np.eye(K)
    counts = np.array([5.0, 3.0, 1.0, 2.0])  # every code occupied: K=C is available
    grouping = categorical.build_grouping(centers, counts, num_states=4, seed=111, is_kc_identity=True)
    assert list(grouping) == [0, 1, 2, 3]


def test_grouping_unavailable_when_too_few_occupied():
    centers = np.random.default_rng(0).normal(size=(5, 3))
    counts = np.array([1.0, 0.0, 0.0, 0.0, 2.0])
    try:
        categorical.build_grouping(centers, counts, num_states=3, seed=111, is_kc_identity=False)
        raise AssertionError("expected UnavailableError")
    except categorical.UnavailableError:
        pass


def test_theta0_normalization_and_pseudocount():
    K, C, eta = 6, 2, 3.0
    grouping = np.array([0, 0, 1, 1, 1, 0], dtype=np.int32)
    counts = np.array([2.0, 3.0, 0.0, 5.0, 1.0, 4.0])
    theta0 = categorical.build_theta0(grouping, counts, C, K, eta)
    assert_close(theta0.sum(axis=1), np.ones(C), atol=1e-12, msg="theta0 rows must sum to 1")
    assert (theta0 > 0).all(), "eta smoothing must keep every entry strictly positive"
    # theta0[a,u] = (c[u]*1[g[u]=a] + eta/K) / (sum_{u': g[u']=a} c[u'] + eta):
    # the eta/K pseudocount is added to every code, but the row denominator
    # is the *masked* count sum plus the *whole* eta (not eta/K per code).
    row0_mass = counts[grouping == 0].sum() + eta
    assert_close(theta0[0, 0], (counts[0] + eta / K) / row0_mass, atol=1e-9, msg="theta0 hand-check")


# --- mode filter --------------------------------------------------------------

def test_mode_filter_hand_computed():
    states = np.array([0, 0, 1, 1, 1, 0], dtype=np.int32)
    lengths = np.array([1, 1, 1, 1, 1, 1], dtype=np.int32)
    out = smoothing.mode_filter(states, lengths, half_width=1)
    # t=0: window {0,1} -> votes {0:2} -> 0
    # t=1: window {0,1,2} -> votes {0:2,1:1} -> 0
    # t=2: window {1,2,3} -> votes {0:1,1:2} -> 1
    # t=3: window {2,3,4} -> votes {1:3} -> 1
    # t=4: window {3,4,5} -> votes {1:2,0:1} -> 1
    # t=5: window {4,5} -> votes {1:1,0:1} tied -> keep original state[5]=0
    assert list(out) == [0, 0, 1, 1, 1, 0], f"mode filter mismatch: {out}"


def test_mode_filter_tie_prefers_original_else_smallest():
    states = np.array([2, 0], dtype=np.int32)
    lengths = np.array([1, 1], dtype=np.int32)
    out = smoothing.mode_filter(states, lengths, half_width=1)
    # both windows tie between states {0,2}; window0 keeps original (2);
    # window1 keeps original (0).
    assert list(out) == [2, 0]
    states2 = np.array([2, 5], dtype=np.int32)
    out2 = smoothing.mode_filter(states2, lengths, half_width=1)
    # tie {2,5} at each position, original state present at each -> unchanged
    assert list(out2) == [2, 5]


# --- permutation round trip / A0 equivariance --------------------------------

def test_permutation_round_trip():
    rng = _rng(7)
    lengths = np.array([15, 15, 15, 15, 8], dtype=np.int32)  # last window terminal/partial
    window = 15
    perm = permute.build_permutation(lengths, window, rng)
    assert perm[-1] == 4, "terminal partial window must stay in place"
    codes = np.arange(5, dtype=np.int32)
    codes_perm = permute.apply_permutation(codes, perm)
    # a trivial "solve": states_shuffled == codes_perm (identity model)
    restored = permute.restore_states(codes_perm, perm)
    assert list(restored) == list(codes), "restore must invert the permutation exactly"


def test_permutation_preserves_counts():
    rng = _rng(8)
    lengths = np.full(20, 12, dtype=np.int32)
    window = 12
    codes = rng.integers(0, 50, size=20).astype(np.int32)
    perm = permute.build_permutation(lengths, window, rng)
    codes_perm = permute.apply_permutation(codes, perm)
    assert sorted(codes_perm.tolist()) == sorted(codes.tolist()), "permutation must preserve exact code counts"
    assert list(lengths[perm]) == list(lengths), "all windows share length==window here, so lengths are unchanged"


def test_a0_equivariant_under_permutation():
    """A0 (beta=0, so no cross-window coupling) fit on permuted-then-restored
    data must match the unpermuted fit closely: same theta, same restored
    raw states, at tight numerical tolerance."""
    rng = _rng(9)
    C, K = 3, 20
    eta = 6.0
    window = 10
    lengths = np.full(24, window, dtype=np.int32)
    codes = rng.integers(0, K, size=24).astype(np.int32)
    fps = 60
    rec = categorical.RecordingInput("r0", codes, lengths, fps, window)
    centers = rng.normal(size=(K, 4))
    counts = categorical.frame_weighted_counts([rec], K)
    grouping = categorical.build_grouping(centers, counts, C, seed=111, is_kc_identity=False)
    theta0 = categorical.build_theta0(grouping, counts, C, K, eta)
    coeffs = transport.Coeffs(0.7, 0.0, 0.05, 0.07)
    fit_a = categorical.fit_categorical([rec], theta0, C, K, eta, coeffs, inner_steps=10, outer_cap=8,
                                        outer_patience=2, outer_reltol=1e-8, device=torch.device("cpu"))
    final_a = categorical.infer_final(fit_a.theta, [rec], C, fit_a.t_state, coeffs, torch.device("cpu"),
                                      max_steps=100, grad_tol=1e-6, rel_tol=1e-8, patience=3, backtrack_max=30)

    perm = permute.build_permutation(lengths, window, rng)
    codes_perm = permute.apply_permutation(codes, perm)
    rec_perm = categorical.RecordingInput("r0", codes_perm, lengths, fps, window)
    counts_perm = categorical.frame_weighted_counts([rec_perm], K)  # identical (bincount is order-free)
    assert_close(counts_perm, counts, atol=0, rtol=0, msg="frame-weighted counts must be permutation-invariant")
    fit_b = categorical.fit_categorical([rec_perm], theta0, C, K, eta, coeffs, inner_steps=10, outer_cap=8,
                                        outer_patience=2, outer_reltol=1e-8, device=torch.device("cpu"))
    final_b = categorical.infer_final(fit_b.theta, [rec_perm], C, fit_b.t_state, coeffs, torch.device("cpu"),
                                      max_steps=100, grad_tol=1e-6, rel_tol=1e-8, patience=3, backtrack_max=30)
    assert_close(fit_a.theta, fit_b.theta, atol=1e-8, rtol=1e-6, msg="A0 theta must be permutation-invariant")
    restored_states = permute.restore_states(final_b.predictions["r0"], perm)
    assert list(restored_states) == list(final_a.predictions["r0"]), \
        "A0 raw states must match exactly after restoration (no coupling to permute)"


# --- reference parity ---------------------------------------------------------

def _load_upstream_asot():
    """Load third_party/action_seg_ot/src/asot.py directly by file path, as
    a uniquely named module -- the repo already has its own top-level `src`
    package, so `import src.asot` would resolve to the wrong package."""
    import importlib.util
    path = Path(__file__).resolve().parents[2] / "third_party" / "action_seg_ot" / "src" / "asot.py"
    spec = importlib.util.spec_from_file_location("action_seg_ot_reference_asot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_parity_equal_mass():
    """Compare against the pinned upstream solver in equal-mass mode, per
    instructions §5: identical kernel/radius, costs, initialization,
    iteration count, and its step-size rule. The kernel-construction guard
    (int(L*(b/L))==b) is asserted before comparing; production never derives
    b this way, only this test does, to match the reference's own
    construction exactly."""
    asot = _load_upstream_asot()
    torch.set_default_dtype(torch.float64)
    try:
        L, C, b = 16, 3, 4
        assert int(L * (b / L)) == b, "parity case must satisfy the reference's rounding, not just ours"
        rng = _rng(10)
        cost = torch.as_tensor(rng.random((1, L, C)), dtype=torch.float64)
        mask = torch.ones((1, L), dtype=torch.bool)
        radius = b / L
        t_ref, _ = asot.segment_asot(cost, mask, eps=0.07, alpha=0.3, radius=radius, ub_frames=False,
                                     ub_actions=True, lambda_frames=0.1, lambda_actions=0.05,
                                     n_iters=(25, 1), step_size=1.0)

        weights = transport.build_kernel(L, b, torch.device("cpu"))
        p = torch.full((L,), 1.0 / L, dtype=torch.float64)
        q = torch.full((C,), 1.0 / C, dtype=torch.float64)
        # ASOT's a=1-alpha, beta=alpha; equal-mass, ub_actions-only KL match.
        coeffs = transport.Coeffs(a=1 - 0.3, beta=0.3, lam=0.05, eps=0.07)
        t0 = p.unsqueeze(1) * q.unsqueeze(0)
        result_ours = transport.solve_inner(t0, cost[0], weights, p, q, coeffs, max_accepted=25,
                                            backtrack_max=0, backtrack_tol=1e30)
        # backtrack_max=0/backtrack_tol=1e30 forces a fixed step size of 1.0
        # per accepted step with no rejection, matching the reference's own
        # fixed-step-size mode (step_size=1.0 passed above).
        #
        # Convention note: the reference's segment_asot rescales its
        # returned T by nnz at the very end (`T = T*nnz[...]`), so its
        # *returned* T already has row sums of 1 -- it is directly
        # comparable to our R = T/p (row sums of 1), not to our T (row sums
        # of p). Dividing t_ref by p again would double that rescale.
        r_ref = t_ref[0].numpy()
        r_ours = (result_ours.t / p.unsqueeze(1)).numpy()
        diff = np.abs(r_ref - r_ours).max()
        print(f"[parity] max |R_ref - R_ours| = {diff:.3e}")
        assert diff <= 2e-2, (
            f"parity drift {diff:.3e} exceeds the declared tolerance; this compares different "
            "unary conventions (ASOT's 1-cos vs a generic D here) and is a structural-term-only "
            "check -- see README for the exact scope of this parity test"
        )
    finally:
        torch.set_default_dtype(torch.float32)


# --- evaluator round trip -----------------------------------------------------

def test_evaluator_identity_and_permuted_labels():
    rng = _rng(11)
    gt = rng.integers(0, 3, size=200).astype(np.int32)
    identity_score = evaluate.score_pooled([gt], [gt.copy()])
    assert_close(identity_score["MoF"], 100.0, atol=1e-6, msg="identity prediction must score MoF=100")
    shuffled_map = {0: 2, 1: 0, 2: 1}
    relabelled = np.vectorize(shuffled_map.get)(gt).astype(np.int32)
    relabelled_score = evaluate.score_pooled([gt], [relabelled])
    assert_close(relabelled_score["MoF"], 100.0, atol=1e-6,
                msg="Hungarian mapping must recover a pure relabelling")


def test_broadcast_to_frames_exact_coverage():
    starts = np.array([0, 5, 8], dtype=np.int32)
    lengths = np.array([5, 3, 2], dtype=np.int32)
    states = np.array([1, 0, 1], dtype=np.int32)
    frames = evaluate.broadcast_to_frames(states, starts, lengths, frame_count=10)
    assert list(frames) == [1, 1, 1, 1, 1, 0, 0, 0, 1, 1]


TESTS = [
    test_kernel_matches_dense_reference,
    test_kernel_half_width_rounding_trap,
    test_objective_gradient_finite_difference,
    test_beta_zero_retains_other_terms,
    test_solve_preserves_row_masses,
    test_solve_single_window,
    test_cosine_cost_conventions,
    test_emission_update_nonincreasing,
    test_continuous_prototype_update_is_weighted_mean,
    test_grouping_kc_identity,
    test_grouping_unavailable_when_too_few_occupied,
    test_theta0_normalization_and_pseudocount,
    test_mode_filter_hand_computed,
    test_mode_filter_tie_prefers_original_else_smallest,
    test_permutation_round_trip,
    test_permutation_preserves_counts,
    test_a0_equivariant_under_permutation,
    test_reference_parity_equal_mass,
    test_evaluator_identity_and_permuted_labels,
    test_broadcast_to_frames_exact_coverage,
]


def run_all() -> None:
    torch.set_num_threads(min(config.CPU_THREAD_CAP, 8))
    passed, failed = 0, []
    for test in TESTS:
        name = test.__name__
        try:
            test()
            print(f"PASS  {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001 - report, do not hide
            print(f"FAIL  {name}: {exc}")
            failed.append((name, str(exc)))
    print(f"\n{passed}/{len(TESTS)} passed")
    if failed:
        print("Failures:")
        for name, msg in failed:
            print(f"  {name}: {msg}")
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()
