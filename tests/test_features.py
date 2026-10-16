# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

import pytest
import torch
from incremental_feature_cl import IncrementalFeatureMapModel

from class_hierarchy_cl.models import (
    BackboneSpec,
    FrozenFeatureExtractor,
    build_head,
    head_state_bytes,
    unique_parameter_count,
)

SPEC = BackboneSpec("slimresnet18", {"nf": 4}, seed=3)


@pytest.fixture(scope="module")
def x():
    return torch.randn(20, 3, 8, 8, generator=torch.Generator().manual_seed(0))


def test_init_is_reproducible_and_seed_dependent():
    a, b = FrozenFeatureExtractor(SPEC), FrozenFeatureExtractor(SPEC)
    c = FrozenFeatureExtractor(BackboneSpec("slimresnet18", {"nf": 4}, seed=4))
    assert a.fingerprint() == b.fingerprint() != c.fingerprint()


def test_building_backbone_does_not_consume_global_rng():
    torch.manual_seed(11)
    FrozenFeatureExtractor(SPEC)
    after = torch.rand(1)
    torch.manual_seed(11)
    assert torch.equal(after, torch.rand(1))


def test_backbone_is_frozen_and_state_never_changes(x):
    fe = FrozenFeatureExtractor(SPEC)
    before = fe.fingerprint()
    assert not any(p.requires_grad for p in fe.backbone.parameters())
    fe.backbone.train()  # extract() must restore eval, BN stats must not move
    fe.extract(x)
    assert not fe.backbone.training and fe.fingerprint() == before


def test_features_identical_across_calls_and_batch_sizes(x):
    fe = FrozenFeatureExtractor(SPEC)
    f1, f2, f3 = (
        fe.extract(x),
        fe.extract(x, batch_size=3),
        FrozenFeatureExtractor(SPEC).extract(x),
    )
    assert f1.shape == (20, fe.out_dim)
    torch.testing.assert_close(f1, f2, rtol=0, atol=1e-6)
    assert torch.equal(f1, f3)


def test_cached_feature_head_equals_main_model_with_shared_frozen_backbone(x):
    fe = FrozenFeatureExtractor(SPEC)
    e2e = IncrementalFeatureMapModel(
        fe.backbone, num_outputs=4, freeze_backbone=True, seed=1
    )
    e2e.expand_feature_space(6, "random")
    e2e.expand_outputs(0)
    head = build_head(fe.out_dim, 4, seed=1)
    head.expand_feature_space(6, "random")
    # same head parameters -> same logits; feature blocks psi were seeded identically
    head.load_state_dict(
        {k: v for k, v in e2e.state_dict().items() if not k.startswith("backbone.")}
    )
    e2e.eval()
    head.eval()
    torch.testing.assert_close(head(fe.extract(x)), e2e(x), rtol=0, atol=1e-5)


def test_head_training_cannot_change_shared_backbone(x):
    fe = FrozenFeatureExtractor(SPEC)
    before = fe.fingerprint()
    m = IncrementalFeatureMapModel(fe.backbone, num_outputs=1, freeze_backbone=True)
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1)
    m.train()
    for _ in range(3):
        opt.zero_grad()
        m(x).sum().backward()
        opt.step()
    assert fe.fingerprint() == before and not fe.backbone.training


def test_unique_parameter_count_counts_shared_backbone_once():
    fe = FrozenFeatureExtractor(SPEC)
    ms = [
        IncrementalFeatureMapModel(fe.backbone, num_outputs=1, freeze_backbone=True)
        for _ in range(3)
    ]
    bb = sum(p.numel() for p in fe.backbone.parameters())
    assert sum(m.parameter_count() for m in ms) > unique_parameter_count(
        ms
    )  # naive sum double counts
    assert unique_parameter_count(ms) == bb + 3 * (ms[0].parameter_count() - bb)
    assert unique_parameter_count(ms, trainable_only=True) == 3 * ms[0].parameter_count(
        trainable_only=True
    )


def test_head_state_bytes_excludes_backbone():
    fe = FrozenFeatureExtractor(SPEC)
    m = IncrementalFeatureMapModel(fe.backbone, num_outputs=1, freeze_backbone=True)
    assert head_state_bytes(m) < 0.1 * head_state_bytes(m, exclude_prefix="__none__")
