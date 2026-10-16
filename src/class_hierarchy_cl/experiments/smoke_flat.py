# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Stage-1 reference run: synthetic stream, flat softmax baseline (Method A). Correctness, not science."""

from __future__ import annotations

import argparse
from pathlib import Path

from incremental_feature_cl.data.datasets import make_synthetic_dataset

from ..baselines import FlatConfig, FlatSoftmax
from ..models import BackboneSpec, FrozenFeatureExtractor
from ..protocol import ProtocolConfig, build_protocol
from ..results import save_run
from ..seeds import derive_seed
from .runner import run_protocol


def run(seed: int, replay: int, out: Path) -> Path:
    raw = make_synthetic_dataset(
        n_classes=6, n_train_per_class=40, n_test_per_class=20, seed=seed
    )
    ex = FrozenFeatureExtractor(
        BackboneSpec("identity", {"in_dim": 192}, derive_seed(seed, "backbone"))
    )
    cfg = ProtocolConfig(
        "synthetic:n_classes=6,n_train=40,n_test=20",
        n_experiences=3,
        seed=seed,
        replay_capacity=replay,
    )
    proto = build_protocol(*raw, cfg, ex)
    m = FlatSoftmax(proto.feature_dim, FlatConfig(), derive_seed(seed, "head"))
    rec = run_protocol(m, proto, method_config=m.cfg.to_dict())
    return save_run(rec, out / f"flat_smoke_seed{seed}_replay{replay}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay", type=int, default=18)
    ap.add_argument("--out", type=Path, default=Path("artifacts/reference"))
    a = ap.parse_args()
    print(run(a.seed, a.replay, a.out))
