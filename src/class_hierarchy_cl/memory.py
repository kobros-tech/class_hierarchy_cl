# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Replay memory: retention reuses main's ReplayBuffer; sampling is NOT used.

main's buffer draws retention and batch sampling from one RNG, so any method that samples from it
would perturb which exemplars are retained. Here retention depends only on (seed, data), hence is
identical for every method; methods train on current data ∪ the whole memory (documented protocol).
Budget: total exemplars <= capacity (requires capacity >= number of classes, else per-class >= 1
would overshoot).
"""

from __future__ import annotations

import torch
from incremental_feature_cl.data.datasets import ArrayDataset
from incremental_feature_cl.training.replay import ReplayBuffer

from .fingerprints import tensor_fingerprint


class ExemplarMemory:
    def __init__(self, capacity: int, num_classes: int, seed: int):
        if capacity < 0:
            raise ValueError("capacity must be >= 0")
        if 0 < capacity < num_classes:
            raise ValueError(
                f"capacity {capacity} < number of classes {num_classes}: budget would be exceeded"
            )
        self.capacity = capacity
        self._buf = ReplayBuffer(capacity, seed) if capacity > 0 else None

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Call once per experience with THAT experience's training data only."""
        if self._buf is not None:
            self._buf.update(ArrayDataset(x, y))

    def __len__(self) -> int:
        return len(self._buf) if self._buf is not None else 0

    def class_counts(self) -> dict[int, int]:
        return (
            {}
            if self._buf is None
            else {int(c): len(v) for c, v in sorted(self._buf.store.items())}
        )

    def contents(self, feature_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Stable order (by class id, then retention order)."""
        if not len(self):
            return torch.empty(0, feature_dim), torch.empty(0, dtype=torch.long)
        xs, ys = [], []
        for c, v in sorted(self._buf.store.items()):
            xs.append(torch.stack(v))
            ys.append(torch.full((len(v),), int(c), dtype=torch.long))
        return torch.cat(xs), torch.cat(ys)

    def fingerprint(self, feature_dim: int) -> str:
        return tensor_fingerprint(*self.contents(feature_dim))
