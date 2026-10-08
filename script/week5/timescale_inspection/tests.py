"""Tests for phase 0. Run: python -m script.week5.timescale_inspection.tests"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from script.week5.timescale_inspection import oracle as O  # noqa: E402
from script.week5.timescale_inspection import probes as PR  # noqa: E402
from script.week5.timescale_inspection import profile as PF  # noqa: E402


def test_token_labels_majority_and_partial_window():
    gt = np.array([1, 1, 2, 2, 2, 3, 3])
    assert O.token_labels(gt, 3).tolist() == [1, 2, 3]  # windows [1,1,2], [2,2,3], [3]
    assert O.token_labels(gt, 2).tolist() == [1, 2, 2, 3]  # tie [2,3] -> smallest
    assert O.to_frames(O.token_labels(gt, 3), 3, len(gt)).tolist() == [1, 1, 1, 2, 2, 2, 3]
    return True


def test_grid_ceiling_at_window_one_is_perfect():
    rng = np.random.default_rng(0)
    gts = {f"r{i}": np.repeat(rng.integers(0, 4, 6), rng.integers(3, 30, 6)) for i in range(5)}
    out = O.grid_ceiling(gts, 1)
    assert out["F1@50"] == 100.0 and out["MoF"] == 100.0 and out["mapping_is_identity"]
    return True


def test_mode_filter():
    t = np.array([0, 0, 0, 1, 0, 0, 2, 2, 2, 2])
    assert O.mode_filter(t, 0).tolist() == t.tolist()
    assert O.mode_filter(t, 1).tolist() == [0, 0, 0, 0, 0, 0, 2, 2, 2, 2]
    # brute-force reference with the same tie rule
    rng = np.random.default_rng(1)
    for _ in range(50):
        x = rng.integers(0, 3, 40)
        r = int(rng.integers(1, 5))
        ref = []
        for i in range(len(x)):
            win = x[max(0, i - r):i + r + 1]
            counts = np.bincount(win, minlength=3)
            ref.append(x[i] if counts[x[i]] == counts.max() else int(np.argmax(counts)))
        assert O.mode_filter(x, r).tolist() == ref
    return True


def test_relative_radius_floor_on_short_clip():
    assert O.radius_for("relative", 0.04, 16) == 0
    assert O.radius_for("relative_floor1", 0.04, 16) == 1
    assert O.radius_for("relative", 0.04, 500) == 20
    assert O.radius_for("fixed", 3, 16) == 3
    return True


def test_segment_classification():
    fps = 10
    gt = np.array([0] * 20 + [1] * 5 + [2] * 20)
    pm = np.array([0] * 20 + [0] * 5 + [2] * 20)  # the short segment absorbed by its left neighbour
    kinds = [r["kind"] for r in PF.classify_segments(gt, pm, fps)]
    assert kinds == ["hit", "merged", "hit"]
    pm = np.array([0] * 20 + [3] * 5 + [2] * 20)
    assert [r["kind"] for r in PF.classify_segments(gt, pm, fps)][1] == "wrong_label"
    pm = np.array([0] * 20 + [1, 0, 1, 0, 1] + [2] * 20)  # right label, fragmented
    assert [r["kind"] for r in PF.classify_segments(gt, pm, fps)][1] == "boundary"
    return True


def test_profile_rescore_matches_scorer():
    from script.action_transport import evaluate as at_evaluate
    rng = np.random.default_rng(2)
    names = [f"r{i}" for i in range(6)]
    gts = {n: np.repeat(rng.integers(0, 3, 5), rng.integers(10, 40, 5)) for n in names}
    preds = {n: (gts[n] + rng.integers(0, 2, len(gts[n]))) % 3 for n in names}
    res = PF.profile(names, gts, preds, 10, {0: "a", 1: "b", 2: "none"})
    ref = at_evaluate.score_pooled([gts[n] for n in names], [preds[n] for n in names])[0]
    assert res["summary"]["F1@50"] == ref["F1@50"] and res["summary"]["MoF"] == ref["MoF"]
    assert "no_none_F1@50" in res["summary"] and sum(r["recordings"] for r in res["tertiles"]) == len(names)
    return True


def test_group_split_is_deterministic_and_disjoint():
    groups = [f"s{i}" for i in range(20)]
    a, b = PR.split_groups(groups), PR.split_groups(list(reversed(groups)))
    assert a == b and len(a) == 6
    return True


ALL_TESTS = [
    test_token_labels_majority_and_partial_window,
    test_grid_ceiling_at_window_one_is_perfect,
    test_mode_filter,
    test_relative_radius_floor_on_short_clip,
    test_segment_classification,
    test_profile_rescore_matches_scorer,
    test_group_split_is_deterministic_and_disjoint,
]

if __name__ == "__main__":
    failed = 0
    for t in ALL_TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(ALL_TESTS) - failed}/{len(ALL_TESTS)} passed")
    sys.exit(1 if failed else 0)
