"""Tests for the unit-grouping test. python -m script.unit_grouping.tests"""
from __future__ import annotations

import json
import sys
import time
import traceback

import numpy as np
from sklearn.metrics import adjusted_rand_score

from script.unit_grouping import config as C
from script.unit_grouping import grouping as G
from script.unit_grouping import run as R

TESTS = {}


def test(fn):
    TESTS[fn.__name__] = fn
    return fn


def planted(n_groups=3, per=4, n_rec=30, L=120, seed=0, mode="adjacency"):
    """Plans over n_groups*per states. Within a segment the active group's states alternate randomly.
    mode='adjacency': every recording visits every group equally often (so occupancy carries no group information);
    mode='membership': each recording uses a single group (group information lives in recording membership only)."""
    rng = np.random.default_rng(seed)
    M = n_groups * per
    plans = []
    for r in range(n_rec):
        if mode == "membership":
            order = np.full(L // 20, r % n_groups)
        else:
            order = np.concatenate([rng.permutation(n_groups) for _ in range(L // 20 // n_groups)])
        R_ = np.zeros((L, M))
        for i, g in enumerate(order):
            for k in range(20):
                R_[i * 20 + k, g * per + rng.integers(0, per)] = 1.0
        plans.append(R_)
    return plans, np.repeat(np.arange(n_groups), per)


@test
def affinity_is_symmetric_and_reversal_invariant():
    plans, _ = planted(seed=1)
    A = G.affinity(plans, 8)
    assert np.allclose(A, A.T) and np.allclose(np.diag(A), 0)
    A_rev = G.affinity([p[::-1] for p in plans], 8)
    assert np.allclose(A, A_rev), "documented scope limit: the graph cannot see the direction of time"
    A_nocross = G.affinity([p[:60] for p in plans], 8)
    assert A_nocross.sum() < A.sum()                      # sanity: fewer windows, less affinity
    # no pairs are formed across recordings
    two = [np.eye(4)[[0, 1]], np.eye(4)[[2, 3]]]
    assert G.affinity(two, 8)[0, 3] == 0 and G.affinity(two, 8)[0, 1] > 0


@test
def shuffle_preserves_occupancy_and_destroys_adjacency():
    plans, _ = planted(seed=2)
    sh = G.shuffle_rows(plans, 111)
    for a, b in zip(plans, sh):
        assert np.allclose(a.sum(0), b.sum(0)) and a.shape == b.shape
    A, As = G.affinity(plans), G.affinity(sh)
    within = lambda X: sum(X[g * 4:(g + 1) * 4, g * 4:(g + 1) * 4].sum() for g in range(3)) / X.sum()
    assert within(A) > 0.7 and within(As) < 0.45           # planted structure is adjacency, not occupancy
    assert all(np.array_equal(x, y) for x, y in zip(G.shuffle_rows(plans, 111), sh))      # deterministic


@test
def spectral_recovers_adjacency_structure_and_shuffled_control_exposes_membership():
    plans, truth = planted(seed=3)
    g = G.spectral_groups(G.affinity(plans), 3, 111)
    assert adjusted_rand_score(truth, g) > 0.95
    gs = G.spectral_groups(G.affinity(G.shuffle_rows(plans, 111)), 3, 111)
    assert adjusted_rand_score(truth, gs) < 0.5               # adjacency destroyed, nothing left to recover
    assert np.array_equal(g, G.spectral_groups(G.affinity(plans), 3, 111))                # deterministic
    # when group identity lives only in recording membership, the shuffled control recovers it too: this is the confound
    mem, truth = planted(seed=4, mode="membership")
    assert adjusted_rand_score(truth, G.spectral_groups(G.affinity(G.shuffle_rows(mem, 111)), 3, 111)) > 0.9


@test
def appearance_groups_use_prototypes_only():
    rng = np.random.default_rng(0)
    centers = np.repeat(np.eye(3) * 5, 4, axis=0)
    protos = centers + rng.normal(size=centers.shape) * 0.1
    g = G.appearance_groups(protos, np.ones(12), 3, 111)
    assert adjusted_rand_score(np.repeat(np.arange(3), 4), g) == 1.0


@test
def size_matched_random_reproduces_sizes():
    sizes = np.array([1, 6, 2, 3, 5, 1, 4, 2])
    parts = G.size_matched_random(sizes, 100, 111)
    assert len(parts) == 100 and all(np.array_equal(np.bincount(p, minlength=8), sizes) for p in parts)
    assert not all(np.array_equal(parts[0], p) for p in parts[1:])
    assert all(np.array_equal(a, b) for a, b in zip(parts, G.size_matched_random(sizes, 100, 111)))


@test
def relabel_preserves_windows():
    states = {"a": np.array([0, 3, 3, 5]), "b": np.array([1])}
    groups = np.array([0, 1, 1, 2, 2, 3])
    out = G.relabel(states, groups)
    assert out["a"].tolist() == [0, 2, 2, 3] and out["b"].tolist() == [1] and len(out["a"]) == 4


@test
def groupings_are_label_free_and_independent_of_evaluation():
    src = (C.PKG / "grouping.py").read_text(encoding="utf-8")
    assert "evaluate" not in src and "load_gts" not in src and "load_labels" not in src
    plans, _ = planted(seed=4)
    a = G.spectral_groups(G.affinity(plans), 3, 5)                    # the function's only inputs are plans, so evaluation data cannot reach it
    assert np.array_equal(a, G.spectral_groups(G.affinity([p.copy() for p in plans]), 3, 5))
    run_src = (C.PKG / "run.py").read_text(encoding="utf-8")
    assert 'dec_fit["responsibilities"]' in run_src and "fit.plans" not in run_src.split("def run_cell")[1].split("def _diffs")[0]


@test
def decode_returns_fresh_plans_under_the_given_cost():
    from script.compositional_dynamics import config as CD
    from script.compositional_dynamics import data as Dm
    from script.compositional_dynamics import decode as Dc
    old = CD.NUM_STATES
    try:
        CD.NUM_STATES = 24
        rng = np.random.default_rng(0)
        pop = Dm.make_population(["a", "b"], [rng.integers(0, 50, 40), rng.integers(0, 50, 30)], [np.full(40, 12), np.full(30, 12)], 4)
        cost = rng.random((pop.n, 24))
        out = Dc.decode_ladder(cost, pop, (6, 12))
        for n, Rm in out["responsibilities"].items():
            assert Rm.shape[1] == 24 and np.allclose(Rm.sum(1), 1.0, atol=1e-5)
            assert (Rm.argmax(1) == out["states"][12][n]).all()
    finally:
        CD.NUM_STATES = old


@test
def oracle_and_group_summary_by_hand():
    conf = {0: {1: 10, 2: 0}, 1: {1: 2, 2: 8}, 2: {1: 0, 2: 5}, 3: {1: 3, 2: 0}}
    assign, mof = R.oracle_from_confusion(conf)
    assert assign == {0: 1, 1: 2, 2: 2, 3: 1} and abs(mof - 100 * 26 / 28) < 1e-9
    sm = R.group_summary(np.array([0, 1, 1, 0] + [0] * 0), conf, assign, np.ones(4))
    assert sm["classes_with_group"] == 2 and abs(sm["many_to_one_mof"] - 100 * 26 / 28) < 1e-9
    assert sm["ari_vs_oracle"] == 1.0


def _M(temporal, b0=(40, 20), appearance=(41, 21), shuffled=(41, 21)):
    M, P = {}, {}
    for i, s in enumerate(C.SEEDS):
        M[("b0_c", s)] = b0
        M[("group_temporal", s)] = (temporal[0] + i * 0.1, temporal[1] + i * 0.1)
        M[("group_appearance", s)] = appearance
        M[("group_shuffled", s)] = shuffled
        P[s] = (42, 22)
    return M, P


@test
def decision_rule_branches():
    v = lambda *a, **k: R.verdict(*_M(*a, **k), C.SEEDS, True)["verdict"]
    assert v((50, 30)) == "TEMPORAL_GROUPING"
    assert v((41, 21)) == "STOP"                                           # +1 on both: below the bar
    assert v((50, 30), appearance=(50, 30)) == "OVERCOMPLETE_ONLY"         # temporal does not beat appearance
    assert v((50, 30), shuffled=(49, 31)) == "OVERCOMPLETE_ONLY"
    M, P = _M((50, 30))
    P = {s: (51, 22) for s in C.SEEDS}                                      # random p95 MoF above temporal
    assert R.verdict(M, P, C.SEEDS, True)["verdict"] == "PRIOR_MASS"
    assert R.verdict(M, P, C.SEEDS, False)["verdict"] == "INCONCLUSIVE"
    # a win on one metric must not hide a large loss on the other (floor of -2)
    assert v((50, 30), appearance=(40, 36)) == "OVERCOMPLETE_ONLY"           # beats on MoF by +10 but loses F1 by -6
    assert v((50, 30), appearance=(40, 28.5), shuffled=(40, 28.5)) == "TEMPORAL_GROUPING"
    # random grouping near b0 does not void a clear learned gain
    assert R.verdict(*_M((50, 30), b0=(40, 20)), C.SEEDS, True)["verdict"] == "TEMPORAL_GROUPING"


@test
def state_count_override_reaches_the_plans():
    from script.compositional_dynamics import config as CD
    from script.compositional_dynamics import data as Dm
    from script.compositional_dynamics import train as Tr
    old = CD.NUM_STATES
    try:
        CD.NUM_STATES = C.NUM_STATES
        pop = Dm.make_population(["a"], [np.arange(40) % 50], [np.full(40, 12)], 4)
        plans = Tr.Plans(pop)
        plans.refresh(np.random.default_rng(0).random((pop.n, C.NUM_STATES)), 4)
        assert plans.responsibilities().shape == (40, 24)
    finally:
        CD.NUM_STATES = old
    assert C.NUM_STATES == 24 and C.HORIZON == 8 and C.N_RANDOM == 100


@test
def residual_expectation_matches_monte_carlo_shuffles():
    from script.unit_grouping import residual as RS
    rng = np.random.default_rng(0)
    plans = [rng.dirichlet(np.ones(6) * 0.3, size=n) for n in (40, 25, 9)]
    E = RS.expected_shuffled_affinity(plans, 4)
    mc = np.mean([G.affinity(G.shuffle_rows(plans, k), 4) for k in range(3000)], axis=0)
    assert np.allclose(E, mc, rtol=0.03, atol=1e-3), np.abs(E - mc).max()
    W = RS.residual_affinity(G.affinity(plans, 4), E)
    assert (W >= 0).all() and np.allclose(W, W.T) and np.allclose(np.diag(W), 0)


@test
def residual_keeps_adjacency_and_removes_recording_membership():
    from script.unit_grouping import residual as RS
    adj, truth = planted(seed=5)
    g = G.spectral_groups(RS.residual_affinity(G.affinity(adj), RS.expected_shuffled_affinity(adj)), 3, 111)
    assert adjusted_rand_score(truth, g) > 0.95                         # adjacency structure survives
    mem, truth = planted(seed=6, mode="membership")
    A, E = G.affinity(mem), RS.expected_shuffled_affinity(mem)
    assert adjusted_rand_score(truth, G.spectral_groups(A, 3, 111)) > 0.9          # raw affinity groups by membership
    assert RS.residual_share(A, E) < 0.1 * (A.sum() > 0) + 0.25                      # little affinity left beyond membership
    assert RS.residual_share(G.affinity(adj), RS.expected_shuffled_affinity(adj)) > RS.residual_share(A, E)


@test
def residual_verdict_mapping():
    from script.unit_grouping import residual_run as RR
    M, P = {}, {}
    for i, s in enumerate(C.SEEDS):
        M[("b0_c", s)] = (40, 20)
        M[("group_residual", s)] = (50 + i * 0.1, 30 + i * 0.1)
        M[("group_temporal", s)] = (41, 21)
        M[("group_appearance", s)] = (41, 21)
        M[("group_shuffled", s)] = (41, 21)
        P[s] = (42, 22)
    v = RR.verdict(M, P, C.SEEDS, True)
    assert v["verdict"] == "RESIDUAL_GROUPING" and v["vs_temporal_mean"][0] > 8
    M2 = dict(M)
    for s in C.SEEDS:
        M2[("group_residual", s)] = (41, 21)
    assert RR.verdict(M2, P, C.SEEDS, True)["verdict"] == "STOP"
    assert RR.verdict(M, P, C.SEEDS, False)["verdict"] == "INCONCLUSIVE"


def main() -> int:
    results, t0 = {}, time.time()
    for n, fn in TESTS.items():
        try:
            fn()
            results[n] = "pass"
            print(f"PASS {n}", flush=True)
        except Exception:  # noqa: BLE001
            results[n] = "FAIL"
            print(f"FAIL {n}\n{traceback.format_exc()}", flush=True)
    ok = all(v == "pass" for v in results.values())
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "verification.json").write_text(json.dumps(dict(passed=ok, code_hash=C.code_hash(), tests=results, seconds=time.time() - t0), indent=2))
    print(f"{sum(v == 'pass' for v in results.values())}/{len(results)} passed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
