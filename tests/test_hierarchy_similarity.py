# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Similarity construction vs the paper's own worked examples (Tables 1 and 2) and edge cases."""

import numpy as np
import pytest
from helpers import M_PAPER

from class_hierarchy_cl.hierarchy import (
    DegenerateConfusionError,
    confusion_from_predictions,
    confusion_from_probabilities,
    overlap_matrix,
    row_normalize,
    similarity_matrix,
)


def test_paper_table_1_every_intermediate_matrix():
    Mb = row_normalize(M_PAPER)  # Table 1b
    np.testing.assert_allclose(
        Mb,
        [[0.4, 0, 0.6, 0], [0, 0.4, 0.6, 0], [0.4286, 0.4286, 0.1429, 0], [0, 0, 0, 1]],
        atol=5e-5,
    )
    O = overlap_matrix(Mb)  # Table 1c
    assert O[0, 2] == pytest.approx(0.514, abs=5e-4) and O[1, 2] == pytest.approx(
        0.514, abs=5e-4
    )
    assert O[0, 1] == 0 and O[3, 0] == 0 and np.all(np.diag(O) == 1)
    S = similarity_matrix(M_PAPER)  # Table 1d
    expected = np.array(
        [[0, 1, 0.486, 1], [1, 0, 0.486, 1], [0.486, 0.486, 0, 1], [1, 1, 1, 0]]
    )
    np.testing.assert_allclose(S, expected, atol=5e-4)


def test_paper_example_1_triangle_inequality_violation():
    S = similarity_matrix(M_PAPER)
    assert S[0, 1] == 1.0
    assert S[0, 2] + S[2, 1] == pytest.approx(0.972, abs=1e-3)
    assert S[0, 1] > S[0, 2] + S[2, 1]  # semi-metric, not a metric (Remark 1)


def test_paper_table_2_probabilistic_confusion():
    P = np.array(
        [[0.7, 0.1, 0.2], [0.4, 0.6, 0], [0.8, 0, 0.2]]
    )  # e1, e2, e3 with real classes a, b, a
    M = confusion_from_probabilities(P, np.array([0, 1, 0]), 3)
    np.testing.assert_allclose(M[0], [1.5, 0.1, 0.4])
    np.testing.assert_allclose(M[1], [0.4, 0.6, 0.0])
    np.testing.assert_allclose(
        M[2], 0
    )  # the paper's example has no instance of class c
    with pytest.raises(
        DegenerateConfusionError
    ):  # ...and its normalisation is undefined: must not be silent
        similarity_matrix(M)


def test_confusion_from_predictions_counts():
    M = confusion_from_predictions(
        np.array([0, 1, 1, 2, 0]), np.array([0, 0, 1, 2, 2]), 3
    )
    assert M.tolist() == [
        [1, 1, 0],
        [0, 1, 0],
        [1, 0, 1],
    ]  # rows actual, cols predicted
    with pytest.raises(ValueError):
        confusion_from_predictions(np.array([3]), np.array([0]), 3)


def test_similarity_properties_on_random_confusions():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(2, 9))
        M = rng.integers(0, 30, size=(n, n)).astype(float) + np.eye(n)
        Mb = row_normalize(M)
        np.testing.assert_allclose(Mb.sum(1), 1.0)
        S = similarity_matrix(M)
        np.testing.assert_allclose(S, S.T)
        assert S.min() >= 0 and S.max() <= 1 and np.all(np.diag(S) == 0)


def test_scale_invariance_of_counts():
    np.testing.assert_allclose(
        similarity_matrix(M_PAPER), similarity_matrix(7 * M_PAPER)
    )


def test_degenerate_and_invalid_inputs_have_explicit_behaviour():
    M = M_PAPER.copy()
    M[1] = 0
    with pytest.raises(DegenerateConfusionError):
        similarity_matrix(M)
    S = similarity_matrix(
        M, empty_rows="isolate"
    )  # NOT in the paper: documented opt-in
    # isolated row e_1: overlap(1,j) = (0 + Mbar[j,1]) / 2  ->  only class c (row 2) ever predicts class 1
    assert S[1, 0] == 1.0 and S[1, 3] == 1.0
    assert S[1, 2] == pytest.approx(1 - (30 / 70) / 2)
    assert np.all(np.diag(S) == 0)
    for bad in (np.ones((2, 3)), -np.eye(3), np.full((3, 3), np.nan), np.ones((1, 1))):
        with pytest.raises(ValueError):
            similarity_matrix(bad)
    with pytest.raises(ValueError):
        row_normalize(M_PAPER, empty_rows="nonsense")


def test_probabilities_must_be_real_probabilities():
    y = np.array([0, 1])
    with pytest.raises(
        ValueError
    ):  # raw logits / unnormalised scores are not silently accepted
        confusion_from_probabilities(np.array([[2.0, -1.0], [0.3, 0.3]]), y, 2)
    with pytest.raises(ValueError):
        confusion_from_probabilities(np.array([[np.nan, 1.0], [0.5, 0.5]]), y, 2)
    with pytest.raises(ValueError):
        confusion_from_probabilities(np.array([[0.5, 0.5]]), y, 2)  # length mismatch
    with pytest.raises(ValueError):
        confusion_from_probabilities(
            np.array([[1.2, -0.2], [0.5, 0.5]]), y, 2
        )  # negative
