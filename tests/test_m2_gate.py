# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Milestone-2 gate: forgetting vs the agreed formula, class mappings under shuffled/non-contiguous
orders, identical global replay for every method, full-artifact reproducibility."""

from pathlib import Path

import numpy as np
import pytest
import torch
from incremental_feature_cl.data.datasets import ArrayDataset

from class_hierarchy_cl.baselines import FlatConfig, FlatSoftmax
from class_hierarchy_cl.experiments import run_protocol
from class_hierarchy_cl.models import BackboneSpec, FrozenFeatureExtractor
from class_hierarchy_cl.protocol import ProtocolConfig, build_protocol, materialize
from class_hierarchy_cl.results import load_run, save_run, strip_timing

REFERENCE = (
    Path(__file__).resolve().parents[1]
    / "artifacts/reference/flat_smoke_seed0_replay18.json"
)


def forgetting_spec(acc, first):
    """F_c = max_{t in [first(c), T-2]} a[t,c] - a[T-1,c]; written independently of main's helper."""
    T = len(acc)
    out = {}
    for c, f in first.items():
        if f > T - 2 or np.isnan(acc[T - 1][c]):
            continue  # introduced at the final stage (or no eval): undefined
        out[c] = max(acc[t][c] for t in range(f, T - 1)) - acc[T - 1][c]
    return out


def _matrix(rec):
    return [
        [np.nan if v is None else v for v in s["per_class_acc"]] for s in rec["stages"]
    ]


def test_reference_artifact_forgetting_matches_the_agreed_formula():
    rec = load_run(REFERENCE)
    acc = _matrix(rec)
    first = {c: s["experience"] for s in rec["stages"] for c in s["new_classes"]}
    f = forgetting_spec(acc, first)
    stored = rec["summary"]["forgetting_per_class"]
    assert sorted(f) == [
        c for c, v in enumerate(stored) if v is not None
    ]  # same defined set
    for c, v in f.items():
        assert stored[c] == pytest.approx(v)
    assert rec["summary"]["avg_forgetting"] == pytest.approx(np.mean(list(f.values())))
    assert set(range(6)) - set(f) == {
        4,
        5,
    }  # exactly the classes first seen at the last stage
    # hand check of one class from the saved history: class 0
    assert f[0] == pytest.approx(max(acc[0][0], acc[1][0]) - acc[2][0])


def test_per_class_history_is_consistent_with_saved_predictions():
    rec = load_run(REFERENCE)
    for s in rec["stages"]:
        pred, lab = np.array(s["test_predictions"]), np.array(s["test_labels"])
        for c in s["known_classes"]:
            assert s["per_class_acc"][c] == pytest.approx((pred[lab == c] == c).mean())


# ----------------------------------------------------------------------------- class mappings
ORIGINAL_VALUES = [3, 13, 23, 33, 43, 53]  # non-contiguous original label values


def _remapped(raw):
    def remap(ds):
        y = torch.as_tensor([ORIGINAL_VALUES[int(v)] for v in ds.targets])
        return ArrayDataset(ds.x, y)

    return remap(raw[0]), remap(raw[1])


@pytest.fixture(scope="module")
def weird_raw(raw_data):
    return _remapped(raw_data)


@pytest.mark.parametrize(
    "order", [(53, 3, 33, 13, 43, 23), (23, 13, 3, 53, 43, 33), None]
)
def test_noncontiguous_shuffled_orders_keep_mapping_correct_at_every_stage(
    weird_raw, extractor, order
):
    cfg = ProtocolConfig("weird", 3, seed=2, class_order=order)
    p = build_protocol(*weird_raw, cfg, extractor)
    assert sorted(p.class_order) == ORIGINAL_VALUES
    if order is not None:
        assert p.class_order == order
    x_te, y_te_raw = materialize(weird_raw[1])
    # every test sample: internal label -> original value equals the raw label, row by row
    assert torch.equal(p.to_original(p.test_y), y_te_raw)
    f_raw = (extractor.extract(x_te) - p.feature_mean) / p.feature_scale
    torch.testing.assert_close(p.test_x, f_raw, rtol=0, atol=1e-5)
    # per experience: classes introduced are exactly the original values at those arrival positions
    for e in p.experiences:
        assert [p.original_class(c) for c in e.classes] == list(
            p.class_order[2 * e.index : 2 * e.index + 2]
        )
        assert set(p.to_original(e.train_y).tolist()) == set(
            p.class_order[2 * e.index : 2 * e.index + 2]
        )

    m = FlatSoftmax(p.feature_dim, FlatConfig(), 0)
    rec = run_protocol(m, p)
    for t, s in enumerate(rec["stages"]):
        n = 2 * (t + 1)
        assert s["output_to_internal"] == list(range(n))
        assert s["output_to_original"] == list(
            p.class_order[:n]
        )  # output index -> original class
        assert s["known_classes"] == list(range(n))
        # the classes just introduced must be learnable through this mapping (wrong mapping => ~0)
        new = s["new_classes"]
        pred, lab = np.array(s["test_predictions"]), np.array(s["test_labels"])
        assert np.mean([(pred[lab == c] == c).mean() for c in new]) > 0.5
        orig_pred = np.array(s["output_to_original"])[pred]
        assert set(orig_pred) <= set(p.class_order[:n])


def test_class_order_with_absent_label_is_rejected(weird_raw, extractor):
    with pytest.raises(ValueError):
        build_protocol(
            *weird_raw,
            ProtocolConfig("w", 3, class_order=(0, 1, 2, 3, 4, 5)),
            extractor,
        )


# ----------------------------------------------------------------------------- replay contract
class StubMethod:
    """Implements the method interface; records exactly what it is given."""

    prediction_rule = "stub"

    def __init__(self, name, n_internal_heads):
        self.name, self.heads, self._known, self.received = (
            name,
            n_internal_heads,
            [],
            [],
        )

    @property
    def known_classes(self):
        return list(self._known)

    def output_classes(self):
        return list(self._known)

    def state_fingerprint(self):
        return self.name

    def learn_experience(self, index, classes, x, y):
        self._known = sorted(set(self._known) | set(classes))
        self.received.append((x.clone(), y.clone()))

    def predict(self, x):
        return torch.full((len(x),), self._known[0], dtype=torch.long)

    def resources(self):
        return {"n_heads": self.heads}


def test_every_method_receives_the_same_global_replay_regardless_of_head_count(
    make_protocol,
):
    p = make_protocol(replay_capacity=12)
    a, b = (
        StubMethod("one_head", 1),
        StubMethod("many_heads", 5),
    )  # e.g. a hierarchy with 5 node heads
    ra, rb = run_protocol(a, p), run_protocol(b, p)
    for (xa, ya), (xb, yb) in zip(a.received, b.received):
        assert torch.equal(xa, xb) and torch.equal(ya, yb)
    for sa, sb in zip(ra["stages"], rb["stages"]):
        assert sa["train_set_fingerprint"] == sb["train_set_fingerprint"]
        assert sa["replay_fingerprint"] == sb["replay_fingerprint"]
        assert sa["memory_fingerprint_after"] == sb["memory_fingerprint_after"]
        assert sa["n_train_replay"] <= 12  # one global budget, not capacity-per-head
    # replay examples are old-class only, and come from earlier experiences' training data
    for t, (x, y) in enumerate(a.received):
        n_cur = len(p.experiences[t].train_x)
        assert set(y[n_cur:].tolist()) <= set(p.seen_classes(t - 1) if t else [])
        for row in x[n_cur:]:
            pool = torch.cat([e.train_x for e in p.experiences[:t]])
            assert (pool == row).all(1).any(), (
                "replay example not found in earlier training data"
            )


def test_flat_receives_the_same_training_set_as_a_stub(make_protocol):
    p = make_protocol(replay_capacity=12)
    flat = FlatSoftmax(p.feature_dim, FlatConfig(), 0)
    stub = StubMethod("s", 1)
    rf, rs = run_protocol(flat, p), run_protocol(stub, p)
    assert [s["train_set_fingerprint"] for s in rf["stages"]] == [
        s["train_set_fingerprint"] for s in rs["stages"]
    ]


# ----------------------------------------------------------------------------- reproducibility
def _flat_run(raw, seed, replay=12, tmp=None):
    ex = FrozenFeatureExtractor(BackboneSpec("identity", {"in_dim": 192}, 0))
    p = build_protocol(
        *raw, ProtocolConfig("synthetic6", 3, seed=seed, replay_capacity=replay), ex
    )
    m = FlatSoftmax(p.feature_dim, FlatConfig(), seed)
    rec = run_protocol(m, p)
    return rec, m


def test_repeated_runs_are_identical_in_every_field_except_timing(raw_data, tmp_path):
    (a, ma), (b, mb) = _flat_run(raw_data, 0), _flat_run(raw_data, 0)
    ja, jb = (
        load_run(save_run(a, tmp_path / "a.json")),
        load_run(save_run(b, tmp_path / "b.json")),
    )
    assert strip_timing(ja) == strip_timing(
        jb
    )  # predictions, metrics, mappings, replay, fingerprints, resources
    for k in ma.model.state_dict():
        assert torch.equal(ma.model.state_dict()[k], mb.model.state_dict()[k])
    assert any(
        s["train_time_s"] for s in ja["stages"]
    )  # timing is present, and is the only thing ignored


def test_artifact_fingerprints_react_to_seed_and_order(raw_data):
    base, _ = _flat_run(raw_data, 0)
    other, _ = _flat_run(raw_data, 1)
    assert base["split_fingerprint"] != other["split_fingerprint"]
    assert [s["model_fingerprint"] for s in base["stages"]] != [
        s["model_fingerprint"] for s in other["stages"]
    ]
    assert [s["replay_fingerprint"] for s in base["stages"][1:]] != [
        s["replay_fingerprint"] for s in other["stages"][1:]
    ]


def test_new_run_matches_archived_reference_on_all_shared_fields(tmp_path):
    from class_hierarchy_cl.experiments.smoke_flat import run

    new = load_run(run(0, 18, tmp_path))
    old = load_run(REFERENCE)

    def core(r):
        r = strip_timing(r)
        r["stages"] = [
            {
                k: v
                for k, v in s.items()
                if k
                in {
                    "acc_known",
                    "per_class_acc",
                    "confusion",
                    "test_predictions",
                    "test_labels",
                    "memory_class_counts",
                    "n_train_replay",
                }
            }
            for s in r["stages"]
        ]
        return {
            k: r[k]
            for k in (
                "class_order_original_ids",
                "split_fingerprint",
                "stages",
                "summary",
            )
        }

    assert core(new) == core(old)
