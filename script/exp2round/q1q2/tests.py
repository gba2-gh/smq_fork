"""Fast implementation checks required before the Q1/Q2 full run."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from script.exp2round.q1q2.core import (
    Dataset, conditional_mi, information_values, make_folds, map_predictions,
    permutation_references, segment_features,
)


def fake_data() -> Dataset:
    return Dataset(
        name="fake", names=["a", "b"],
        gts=[np.array([0, 0, 0, 1, 1]), np.array([0, 0, 1, 1])],
        codes=[np.zeros(5), np.zeros(4)],
        arrays=[np.zeros((5, 1)), np.zeros((4, 1))],
        subjects=["s1", "s2"], patch_size=3, num_actions=2,
        cache_identity="fake",
    )


def test_information() -> None:
    action = np.repeat([0, 1], 10)
    subject = np.tile(np.repeat([0, 1], 5), 2)
    units = action.copy()
    values = information_values(units, action, subject)
    assert np.isclose(values["mi_unit_action"], np.log(2))
    assert np.isclose(values["h_action_given_unit"], 0)
    assert np.isclose(conditional_mi(units, subject, action), 0)
    references = permutation_references(units, action, subject)
    assert set(references) >= {"mi_unit_action", "cmi_unit_subject_given_action"}


def test_count_preserving_reference() -> None:
    units = np.array([0] * 7 + [1] * 5 + [2] * 4)
    actions = np.tile([0, 1], 8)
    subjects = np.repeat([0, 1, 2, 3], 4)
    references = permutation_references(units, actions, subjects)
    assert np.isclose(references["h_unit"][0], information_values(units, actions, subjects)["h_unit"])
    assert np.isclose(references["h_unit"][1], 0)


def test_overlap_histograms() -> None:
    data = fake_data()
    assignments = [np.array([0, 1]), np.array([1, 0])]
    transformed = [np.array([[0.0], [3.0]]), np.array([[2.0], [4.0]])]
    hard, labels, lengths, seq = segment_features(
        data, 3, "hard", transformed, num_units=2, assignments=assignments
    )
    assert np.allclose((hard ** 2).sum(axis=1), 1)
    assert labels.tolist() == [0, 1, 0, 1]
    assert lengths.tolist() == [3, 2, 2, 2]
    assert seq.tolist() == [0, 0, 1, 1]
    continuous, _, _, _ = segment_features(data, 3, "continuous", transformed)
    assert np.isclose(continuous[1, 0], 3.0)


def test_mappings() -> None:
    gt = [np.array([0, 0, 1, 1, 0, 0])]
    fragmented = [np.array([10, 10, 11, 11, 12, 12])]
    hungarian = map_predictions(gt, fragmented, "hungarian")[0]
    many = map_predictions(gt, fragmented, "many_to_one")[0]
    assert np.mean(many == gt[0]) == 1.0
    assert np.mean(hungarian == gt[0]) < 1.0


def test_folds() -> None:
    groups = ["a", "a", "b", "c", "d", "e"]
    lengths = [5, 5, 8, 7, 6, 5]
    folds = make_folds(groups, lengths)
    assert folds[0] == folds[1]
    assert set(folds) == {0, 1, 2, 3}


def main() -> None:
    tests = [test_information, test_count_preserving_reference,
             test_overlap_histograms, test_mappings, test_folds]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
