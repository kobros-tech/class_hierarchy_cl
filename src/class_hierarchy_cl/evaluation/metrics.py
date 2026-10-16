# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Metrics. All accuracies are over test samples of classes KNOWN at that stage (never unseen classes).

Definitions
  acc_known[t]        top-1 accuracy at stage t over test samples whose class was introduced at or before t
  per_class_acc[t,c]  accuracy on class c at stage t (NaN if c not yet introduced)
  avg_incremental     mean_t acc_known[t]
  forgetting[c]       max_{first(c) <= t < T-1} per_class_acc[t,c] - per_class_acc[T-1,c]
                      (main's `forgetting_per_class`; classes first seen at the final stage are undefined/NaN)
  avg_forgetting      nanmean_c forgetting[c]
  forgetting_curve[t] avg forgetting computed from stages 0..t only (final stage := t)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from incremental_feature_cl.evaluation.accuracy import per_class_accuracy
from incremental_feature_cl.evaluation.forgetting import (
    average_forgetting,
    forgetting_per_class,
)


@dataclass
class StageRecord:
    acc_known: float
    per_class_acc: np.ndarray  # (C,)
    confusion: np.ndarray  # (C, C) rows true, cols predicted, counts
    pred_counts: np.ndarray  # (C,)
    n_eval: int


def evaluate_stage(
    pred: np.ndarray, labels: np.ndarray, num_classes: int, known: list[int]
) -> StageRecord:
    pred, labels = np.asarray(pred), np.asarray(labels)
    if not np.all(np.isin(labels, known)):
        raise ValueError("evaluation labels must all be known classes")
    if not np.all(np.isin(pred, known)):
        raise ValueError("prediction outside the known class set")
    correct = pred == labels
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(conf, (labels, pred), 1)
    return StageRecord(
        float(correct.mean()),
        per_class_accuracy(correct, labels, num_classes),
        conf,
        np.bincount(pred, minlength=num_classes),
        len(labels),
    )


def summarize(stages: list[StageRecord], first_exp: dict[int, int]) -> dict:
    acc = np.stack([s.per_class_acc for s in stages])  # (T, C)
    a = np.array([s.acc_known for s in stages])
    curve = [
        float(average_forgetting(acc[: t + 1], first_exp)) for t in range(len(stages))
    ]
    return {
        "acc_known_per_stage": a.tolist(),
        "final_accuracy": float(a[-1]),
        "avg_incremental_accuracy": float(a.mean()),
        "forgetting_per_class": forgetting_per_class(acc, first_exp).tolist(),
        "avg_forgetting": float(average_forgetting(acc, first_exp)),
        "forgetting_curve": curve,
    }
