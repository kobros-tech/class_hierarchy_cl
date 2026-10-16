# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
import torch

from class_hierarchy_cl.memory import ExemplarMemory
from class_hierarchy_cl.protocol import (
    ProtocolConfig,
    materialize,
    stratified_val_split,
)


def test_split_is_disjoint_stratified_and_deterministic():
    y = torch.tensor([0] * 10 + [1] * 10 + [2] * 3)
    a = stratified_val_split(y, 0.2, 7)
    b = stratified_val_split(y, 0.2, 7)
    assert all(np.array_equal(p, q) for p, q in zip(a, b))
    tr, va = a
    assert not set(tr) & set(va) and len(tr) + len(va) == len(y)
    for c in range(3):
        assert (y[va] == c).sum() >= 1 and (y[tr] == c).sum() >= 1


def test_class_order_and_relabelling_are_consistent(make_protocol, raw_data, extractor):
    p = make_protocol(n_experiences=3, seed=1)
    assert sorted(p.class_order) == list(range(6))
    # arrival order: experience t introduces internal ids [2t, 2t+1]
    assert [e.classes for e in p.experiences] == [[0, 1], [2, 3], [4, 5]]
    # internal label i must hold exactly the features of ORIGINAL class order[i]
    x_raw, y_raw = materialize(raw_data[1])
    f_raw = extractor.extract(x_raw)
    f_raw = (f_raw - p.feature_mean) / p.feature_scale
    for i, orig in enumerate(p.class_order):
        got = p.test_x[p.test_y == i]
        want = f_raw[y_raw == orig]
        torch.testing.assert_close(
            got.sort(dim=0).values, want.sort(dim=0).values, rtol=0, atol=1e-5
        )


def test_explicit_class_order_and_seed_dependence(make_protocol):
    p = make_protocol(class_order=(5, 4, 3, 2, 1, 0))
    assert p.class_order == (5, 4, 3, 2, 1, 0)
    assert make_protocol(seed=0).class_order == make_protocol(seed=0).class_order
    assert (
        make_protocol(seed=0).split_fingerprint
        == make_protocol(seed=0).split_fingerprint
    )
    assert (
        make_protocol(seed=0).split_fingerprint
        != make_protocol(seed=1).split_fingerprint
    )


def test_val_and_train_do_not_overlap_and_test_is_separate(make_protocol):
    p = make_protocol()
    for e in p.experiences:
        tr = {tuple(v.tolist()) for v in e.train_x}
        va = {tuple(v.tolist()) for v in e.val_x}
        te = {tuple(v.tolist()) for v in p.test_x}
        assert not tr & va and not tr & te and not va & te
        assert set(e.val_y.tolist()) <= set(e.classes) == set(e.train_y.tolist())


def test_memory_budget_balance_and_error():
    x = lambda c, n: torch.randn(n, 4) + c
    m = ExemplarMemory(capacity=6, num_classes=6, seed=0)
    for t, cls in enumerate([[0, 1], [2, 3], [4, 5]]):
        xs = torch.cat([x(c, 10) for c in cls])
        ys = torch.tensor([c for c in cls for _ in range(10)])
        m.update(xs, ys)
        assert len(m) <= 6
        assert all(v == 6 // ((t + 1) * 2) or v == 1 for v in m.class_counts().values())
    with pytest.raises(ValueError):
        ExemplarMemory(capacity=3, num_classes=6, seed=0)
    assert len(ExemplarMemory(0, 6, 0)) == 0


def test_memory_retention_depends_only_on_seed_and_data():
    xs = torch.randn(40, 4)
    ys = torch.arange(40) % 4
    a, b, c = (ExemplarMemory(8, 4, s) for s in (3, 3, 4))
    for m in (a, b, c):
        m.update(xs, ys)
    assert torch.equal(a.contents(4)[0], b.contents(4)[0])
    assert not torch.equal(a.contents(4)[0], c.contents(4)[0])


def test_protocol_config_serialises():
    d = ProtocolConfig("d", 3).to_dict()
    assert d["n_experiences"] == 3 and d["replay_capacity"] == 0


def test_standardisation_uses_only_experience_zero_train_data(make_protocol):
    p = make_protocol()
    e0 = p.experiences[0]
    assert (
        abs(float(e0.train_x.mean())) < 1e-4 and abs(float(e0.train_x.std()) - 1) < 1e-3
    )
    # changing val/test/later-train raw data must not change the fitted statistics: refit differently
    q = make_protocol(standardize_features=False)
    manual = q.experiences[0].train_x
    torch.testing.assert_close(p.feature_mean, manual.mean(0))
    assert p.feature_scale == pytest.approx(
        float((manual - manual.mean(0)).std()), rel=1e-5
    )
