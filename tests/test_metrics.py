# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from class_hierarchy_cl.evaluation.metrics import StageRecord, evaluate_stage, summarize


def test_evaluate_stage_hand_computed():
    labels = np.array([0, 0, 1, 1, 1])
    pred = np.array([0, 1, 1, 1, 0])
    s = evaluate_stage(pred, labels, 3, known=[0, 1])
    assert s.acc_known == pytest.approx(3 / 5)
    assert (
        s.per_class_acc[0] == 0.5
        and s.per_class_acc[1] == pytest.approx(2 / 3)
        and np.isnan(s.per_class_acc[2])
    )
    assert s.confusion.tolist() == [[1, 1, 0], [1, 2, 0], [0, 0, 0]]
    assert s.pred_counts.tolist() == [2, 3, 0]


def test_evaluate_stage_rejects_unknown_predictions_and_labels():
    with pytest.raises(ValueError):
        evaluate_stage(np.array([2]), np.array([0]), 3, known=[0, 1])
    with pytest.raises(ValueError):
        evaluate_stage(np.array([0]), np.array([2]), 3, known=[0, 1])


def _stage(accs, a):
    return StageRecord(
        a, np.array(accs, float), np.zeros((3, 3), int), np.zeros(3, int), 1
    )


def test_forgetting_and_summary_from_history():
    nan = np.nan
    stages = [
        _stage([1.0, nan, nan], 1.0),
        _stage([0.6, 0.8, nan], 0.7),
        _stage([0.2, 0.9, 0.5], 0.5),
    ]
    s = summarize(stages, {0: 0, 1: 1, 2: 2})
    f = s["forgetting_per_class"]
    assert f[0] == pytest.approx(1.0 - 0.2)  # max over stages 0..1 = 1.0
    assert f[1] == pytest.approx(
        0.8 - 0.9
    )  # negative: improved after learning (kept signed)
    assert f[2] is None or np.isnan(f[2])  # introduced at the final stage: undefined
    assert s["avg_forgetting"] == pytest.approx((0.8 - 0.1) / 2)
    assert s["final_accuracy"] == 0.5 and s[
        "avg_incremental_accuracy"
    ] == pytest.approx((1 + 0.7 + 0.5) / 3)
    # curve: stage 1 uses stages 0..1 only -> class 0: 1.0-0.6
    assert s["forgetting_curve"][1] == pytest.approx(0.4) and np.isnan(
        s["forgetting_curve"][0]
    )
