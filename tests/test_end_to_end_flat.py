# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from class_hierarchy_cl.baselines import FlatConfig, FlatSoftmax
from class_hierarchy_cl.evaluation.metrics import StageRecord, summarize
from class_hierarchy_cl.experiments import run_protocol
from class_hierarchy_cl.provenance import main_revision
from class_hierarchy_cl.results import load_run, save_run


def test_end_to_end_artifact_is_auditable(make_protocol, tmp_path):
    p = make_protocol(replay_capacity=18)
    m = FlatSoftmax(p.feature_dim, FlatConfig(), 0)
    path = save_run(
        run_protocol(m, p, method_config=m.cfg.to_dict()), tmp_path / "flat.json"
    )
    r = load_run(path)

    assert r["provenance"]["incremental_feature_cl_revision"] == main_revision()
    assert r["backbone"]["frozen"] and len(r["backbone"]["fingerprint"]) == 64
    assert (
        r["method"]["name"] == "flat_softmax"
        and "argmax" in r["method"]["prediction_rule"]
    )
    assert r["class_order_original_ids"] == list(p.class_order)

    # recompute every stored metric from the stored raw predictions
    per_class = []
    for s in r["stages"]:
        pred, lab = np.array(s["test_predictions"]), np.array(s["test_labels"])
        assert s["acc_known"] == pytest.approx((pred == lab).mean())
        pc = np.array([np.nan if v is None else v for v in s["per_class_acc"]], float)
        for c in s["known_classes"]:
            assert pc[c] == pytest.approx((pred[lab == c] == c).mean())
        conf = np.array(s["confusion"])
        assert conf.sum() == len(lab) and np.trace(conf) == (pred == lab).sum()
        per_class.append(pc)
    stages = [
        StageRecord(s["acc_known"], pc, np.zeros((6, 6), int), np.zeros(6, int), 1)
        for s, pc in zip(r["stages"], per_class)
    ]
    again = summarize(stages, {c: e.index for e in p.experiences for c in e.classes})
    assert again["avg_forgetting"] == pytest.approx(r["summary"]["avg_forgetting"])
    assert again["avg_incremental_accuracy"] == pytest.approx(
        r["summary"]["avg_incremental_accuracy"]
    )

    res = r["resources"]
    assert (
        res["n_heads"] == 1
        and res["head_trainable_parameters"] > 0
        and res["head_checkpoint_bytes"] > 0
    )
    assert all(
        s["infer_ms_per_example"] > 0 and s["train_time_s"] > 0 for s in r["stages"]
    )
    assert (
        r["summary"]["final_accuracy"] > 1 / 6 + 0.2
    )  # clearly above chance, else the baseline is not learning


def test_missing_revisions_are_reported_not_silent(make_protocol, monkeypatch):
    from class_hierarchy_cl.experiments import runner

    monkeypatch.setattr(runner, "code_revision", lambda *_: None)
    monkeypatch.setattr(runner, "main_revision", lambda: None)
    p = make_protocol()
    m = FlatSoftmax(p.feature_dim, FlatConfig(train_epochs=1), 0)
    w = run_protocol(m, p)["warnings"]
    assert any("code_revision" in x for x in w) and any(
        "incremental_feature_cl" in x for x in w
    )
