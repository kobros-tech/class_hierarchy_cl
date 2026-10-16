# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

import copy

import numpy as np
import torch

from class_hierarchy_cl.baselines import FlatConfig, FlatSoftmax, decide
from class_hierarchy_cl.experiments import run_protocol
from class_hierarchy_cl.seeds import derive_seed


def test_decide_known_logits_and_ties():
    logits = torch.tensor(
        [[0.1, 2.0, -1.0], [-3.0, -2.0, -2.5], [1.0, 1.0, 0.0], [0.0, 5.0, 5.0]]
    )
    assert decide(logits).tolist() == [1, 1, 0, 1]  # ties -> lowest column
    assert decide(logits).dtype == torch.long and decide(logits).shape == (4,)
    # works when every logit is equal (all columns tie)
    assert decide(torch.zeros(2, 5)).tolist() == [0, 0]


def _run(make_protocol, epochs=5, seed=0, **proto):
    p = make_protocol(**{"seed": seed, **proto})
    m = FlatSoftmax(
        p.feature_dim, FlatConfig(train_epochs=epochs), derive_seed(seed, "head")
    )
    return p, m, run_protocol(m, p, method_config=m.cfg.to_dict())


def test_outputs_track_known_classes_and_predictions_are_known(make_protocol):
    _, m, rec = _run(make_protocol)
    assert m.model.num_outputs == 6 == len(m.known_classes)
    for s in rec["stages"]:
        assert set(s["test_predictions"]) <= set(s["known_classes"])
        assert set(s["test_labels"]) == set(s["known_classes"])
        assert s["n_eval"] == len(s["test_labels"]) == 20 * len(s["known_classes"])


def test_predict_needs_only_features_and_ignores_future_classes(make_protocol):
    p = make_protocol()
    m = FlatSoftmax(p.feature_dim, FlatConfig(), 0)
    e0 = p.experiences[0]
    m.learn_experience(0, e0.classes, e0.train_x, e0.train_y)
    pred = m.predict(
        p.test_x
    )  # includes test rows of classes 2..5 the model has never seen
    assert pred.shape == (len(p.test_x),) and set(pred.tolist()) <= {0, 1}


def test_training_changes_intended_state_and_not_the_backbone(make_protocol, extractor):
    p = make_protocol()
    m = FlatSoftmax(p.feature_dim, FlatConfig(), 0)
    fp = extractor.fingerprint()
    e0 = p.experiences[0]
    m.learn_experience(0, e0.classes, e0.train_x, e0.train_y)
    after0 = copy.deepcopy(m.model.state_dict())
    e1 = p.experiences[1]
    m.learn_experience(1, e1.classes, e1.train_x, e1.train_y)
    after1 = m.model.state_dict()
    # outputs grew 2 -> 4; compare the OLD rows only
    changed = [
        k for k in after0 if not torch.equal(after0[k], after1[k][: len(after0[k])])
    ]
    assert changed, (
        "second experience must update the old head rows too (no replay -> they drift)"
    )
    assert extractor.fingerprint() == fp == p.extractor_fingerprint


def test_different_budgets_give_different_states_same_budget_same_state(make_protocol):
    def state(epochs):
        p = make_protocol()
        m = FlatSoftmax(p.feature_dim, FlatConfig(train_epochs=epochs), 0)
        e = p.experiences[0]
        m.learn_experience(0, e.classes, e.train_x, e.train_y)
        return torch.cat([v.flatten() for v in m.model.state_dict().values()])

    assert not torch.equal(state(1), state(4))
    assert torch.equal(state(2), state(2))


def test_training_never_reads_test_data(make_protocol):
    def final_state(poison):
        p = make_protocol()
        if poison:
            p.test_x = torch.full_like(p.test_x, float("nan"))
            p.test_y = p.test_y.flip(0)
        m = FlatSoftmax(p.feature_dim, FlatConfig(), 0)
        for e in p.experiences:
            m.learn_experience(e.index, e.classes, e.train_x, e.train_y)
        return torch.cat([v.flatten() for v in m.model.state_dict().values()])

    assert torch.equal(final_state(False), final_state(True))


def test_replay_uses_only_previous_experiences_and_respects_budget(make_protocol):
    _, _, rec = _run(make_protocol, replay_capacity=12)
    s = rec["stages"]
    assert s[0]["n_train_replay"] == 0
    for t in range(1, 3):
        assert 0 < s[t]["n_train_replay"] <= 12
        assert s[t]["n_train_replay"] == s[t - 1]["memory_size"]
    assert all(r["memory_size"] <= 12 for r in s)
    assert not any("replay" in w for w in rec["warnings"])


def test_fixed_seed_is_reproducible_and_seed_matters(make_protocol):
    _, _, a = _run(make_protocol, seed=0)
    _, _, b = _run(make_protocol, seed=0)
    _, _, c = _run(make_protocol, seed=1)
    for x, y in zip(a["stages"], b["stages"]):
        assert np.array_equal(x["test_predictions"], y["test_predictions"])
    from class_hierarchy_cl.results import to_jsonable

    assert to_jsonable(a["summary"]) == to_jsonable(b["summary"])
    assert a["split_fingerprint"] != c["split_fingerprint"]
