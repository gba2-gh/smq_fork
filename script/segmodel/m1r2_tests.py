"""Acceptance tests for the corrected Milestone-1 rerun."""
from __future__ import annotations

import itertools
import time

import numpy as np

from script.segmodel.m1r2_core import (
    action_runs, alternate, assigned_objective, candidate_costs, decode_costs,
    point_cost_prefix, representations,
)


def brute(values, lengths, centers, penalty, maximum):
    best = None
    n = len(values)
    for mask in range(1 << max(0, n - 1)):
        edges = [0] + [i + 1 for i in range(n - 1) if mask & (1 << i)] + [n]
        if any(b - a > maximum for a, b in zip(edges[:-1], edges[1:])):
            continue
        for labels in itertools.product(range(len(centers)), repeat=len(edges) - 1):
            objective = assigned_objective([values], [lengths], centers,
                                           [np.asarray(edges)], [np.asarray(labels)], penalty)
            item = (objective, edges, labels)
            if best is None or objective < best[0] - 1e-12:
                best = item
    return best


def test_merging_costs() -> None:
    values = np.asarray([[1., 0.], [0., 1.]])
    lengths = np.ones(2)
    centers = values.copy()
    pooled = assigned_objective([values], [lengths], centers,
                                [np.asarray([0, 2])], [np.asarray([0])], 0)
    split = assigned_objective([values], [lengths], centers,
                               [np.asarray([0, 1, 2])], [np.asarray([0, 1])], 0)
    assert split < pooled


def test_lambda_monotonicity() -> None:
    values = np.eye(4)
    lengths = np.ones(4)
    centers = values.copy()
    costs, states = candidate_costs(values, lengths, centers, 4)
    counts = []
    for penalty in (0., .1, 1., 100.):
        edges, _, _ = decode_costs(costs, states, penalty)
        counts.append(len(edges) - 1)
    assert counts[0] == 4 and counts[-1] == 1
    assert all(a >= b for a, b in zip(counts, counts[1:]))


def test_dp_matches_brute_force() -> None:
    rng = np.random.default_rng(3)
    for n in range(2, 7):
        values = rng.normal(size=(n, 3))
        lengths = rng.integers(1, 4, size=n)
        centers = rng.normal(size=(2, 3))
        costs, states = candidate_costs(values, lengths, centers, 4)
        edges, labels, objective = decode_costs(costs, states, .37)
        reference = brute(values, lengths, centers, .37, 4)
        assert np.isclose(objective, reference[0])
        assert np.array_equal(edges, reference[1])
        assert np.array_equal(labels, reference[2])


def test_alternating_monotone() -> None:
    rng = np.random.default_rng(7)
    values = [rng.random((12, 5))]
    values = [x / x.sum(1, keepdims=True) for x in values]
    lengths = [np.full(12, 3)]
    centers = values[0][[0, 6]].copy()
    _, edges, labels, trace, _, final = alternate(
        values, lengths, centers, .2, 8, time.monotonic() + 30, max_rounds=10)
    objectives = [row["objective"] for row in trace]
    assert all(b <= a + 1e-8 * max(1, abs(a)) for a, b in zip(objectives, objectives[1:]))
    assert final <= objectives[-1] + 1e-8 * max(1, abs(objectives[-1]))
    assert action_runs(labels) <= sum(len(e) - 1 for e in edges)


def test_histogram_mass_and_hard_equivalence() -> None:
    q = [np.asarray([[.25, .75], [1., 0.]])]
    soft = representations(q, "soft")[0]
    hard = representations(q, "hard")[0]
    assert np.allclose((soft ** 2).sum(1), 1)
    assert np.allclose(hard, representations([hard], "soft")[0])
    centers = np.eye(2)
    a, _ = candidate_costs(hard, np.asarray([2, 3]), centers, 2)
    manual = point_cost_prefix(hard, np.asarray([2, 3]), centers)
    assert np.isclose(a[1, 0], (manual[1] - manual[0]).min())


def main() -> None:
    tests = [test_merging_costs, test_lambda_monotonicity, test_dp_matches_brute_force,
             test_alternating_monotone, test_histogram_mass_and_hard_equivalence]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
