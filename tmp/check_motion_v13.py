"""Tiny definition checks only; does not fit any experimental model."""
import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(111)
L, K, C = 9, 7, 3
lengths = np.array([15] * (L - 1) + [6], dtype=float)
p = lengths / lengths.sum()
codes = rng.integers(K, size=L)
perm = np.r_[rng.permutation(L - 1), L - 1]
inverse = np.argsort(perm)
assert np.array_equal(codes[perm][inverse], codes)
assert np.array_equal(lengths[perm], lengths)
assert perm[-1] == L - 1

theta = rng.random((C, K)) + .1
theta /= theta.sum(axis=1, keepdims=True)
R = rng.random((L, C)) + .1
R /= R.sum(axis=1, keepdims=True)
T = p[:, None] * R
D = -np.log(theta[:, codes].T) / np.log(500)


def f0(coupling, cost):
    m = coupling.sum(axis=0)
    q = np.full(C, 1 / C)
    entropy = -np.sum(coupling * (np.log(coupling) - 1))
    kl = np.sum(m * np.log(m / q) - m + q)
    return .7 * np.sum(cost * coupling) + .05 * kl - .07 * entropy


def update_theta(code, weights, post):
    counts = np.zeros((C, K))
    for i, unit in enumerate(code):
        counts[:, unit] += weights[i] * post[i]
    return (counts + 60 / K) / (counts.sum(axis=1, keepdims=True) + 60)


np.testing.assert_allclose(f0(T, D), f0(T[perm], D[perm]), atol=1e-14, rtol=1e-14)
np.testing.assert_allclose(update_theta(codes, lengths, R),
                           update_theta(codes[perm], lengths[perm], R[perm]),
                           atol=1e-14, rtol=1e-14)
np.testing.assert_array_equal(R[perm][inverse], R)

# Wrong-order scoring is detectable even with a best two-class label mapping.
labels = np.array([0, 0, 1, 1])
known_perm = np.array([2, 0, 3, 1])
shuffled_labels = labels[known_perm]
wrong_mof = max(np.mean(shuffled_labels == labels), np.mean((1 - shuffled_labels) == labels))
restored_labels = shuffled_labels[np.argsort(known_perm)]
assert wrong_mof == .5 and np.mean(restored_labels == labels) == 1

# A fixed update schedule checks equivariance without a stopping-threshold confound.
def tiny_alternating(code, weights, initial_theta):
    local_theta = initial_theta.copy()
    local_p = weights / weights.sum()
    coupling = local_p[:, None] * np.full((len(code), C), 1 / C)
    for _ in range(3):
        cost = -np.log(local_theta[:, code].T) / np.log(500)
        for _ in range(4):
            m = coupling.sum(axis=0)
            grad = .7 * cost + .05 * np.log(m / (1 / C)) + .07 * np.log(coupling)
            logits = np.log(coupling) - .2 * grad
            logits -= logits.max(axis=1, keepdims=True)
            cond = np.exp(logits)
            cond /= cond.sum(axis=1, keepdims=True)
            coupling = local_p[:, None] * cond
        local_theta = update_theta(code, weights, cond)
    return local_theta, cond

t1, r1 = tiny_alternating(codes, lengths, theta)
t2, r2 = tiny_alternating(codes[perm], lengths[perm], theta)
np.testing.assert_allclose(t1, t2, atol=1e-14, rtol=1e-14)
np.testing.assert_allclose(r1, r2[inverse], atol=1e-14, rtol=1e-14)

# The identity-contextualizer case catches restoring too late or twice.
Z = rng.normal(size=(L, 4))
np.testing.assert_array_equal(Z[perm][inverse], Z)
np.testing.assert_allclose(1 - np.array([[1., 0.], [0., 1.], [-1., 0.], [0., 0.]]) @ np.array([1., 0.]),
                           [0., 1., 2., 1.])
bad = [n for n in range(5, 3000) if int(n * (4 / n)) != 4]
assert len(bad) == 286 and bad[0] == 49
for n in (16, 32, 64):
    assert int(n * (4 / n)) == 4
n, b = 49, 4
kernel = np.ones(2 * b + 1) * n / b
kernel[b] = 0
assert len(kernel) == 9 and np.count_nonzero(kernel) == 8
assert np.all(kernel[kernel > 0] == 49 / 4)

out = {"definition_checks": "pass", "rounding_failures": len(bad),
       "first_failure": bad[0], "stage_a_solver_tested": False,
       "experimental_data_used": False}
Path('tmp/motion_plan_qa').mkdir(parents=True, exist_ok=True)
Path('tmp/motion_plan_qa/v13_definition_checks.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out))
