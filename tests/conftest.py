# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

import pytest
from incremental_feature_cl.data.datasets import make_synthetic_dataset

from class_hierarchy_cl.models import BackboneSpec, FrozenFeatureExtractor
from class_hierarchy_cl.protocol import ProtocolConfig, build_protocol

# Stage-1 smoke tests use main's own synthetic-test backbone (frozen flatten): the data stay learnable,
# so "above chance" assertions are meaningful. Random frozen SlimResNet18 is covered in test_features.py.
SPEC = BackboneSpec("identity", {"in_dim": 192}, seed=0)


@pytest.fixture(scope="session")
def extractor():
    return FrozenFeatureExtractor(SPEC)


@pytest.fixture(scope="session")
def raw_data():
    return make_synthetic_dataset(
        n_classes=6, n_train_per_class=40, n_test_per_class=20, seed=0
    )


@pytest.fixture
def make_protocol(raw_data, extractor):
    def _make(**kw):
        base = {
            "dataset_id": "synthetic6",
            "n_experiences": 3,
            "seed": 0,
            "replay_capacity": 0,
        }
        return build_protocol(*raw_data, ProtocolConfig(**{**base, **kw}), extractor)

    return _make
