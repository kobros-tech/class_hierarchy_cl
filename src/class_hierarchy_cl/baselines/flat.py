# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Method A: the existing flat multiclass model from main (native cross-entropy, one head).

Unchanged from main: IncrementalFeatureMapModel, ExpansionPolicy.expand_outputs, ContinualTrainer's
optimizer and cross-entropy loop (``_train_experience``, a private method of the pinned commit).
Added here: only the decision wrapper (deterministic tie-break) and replay-as-data (see memory.py).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from incremental_feature_cl.data.datasets import ArrayDataset
from incremental_feature_cl.data.streams import Experience
from incremental_feature_cl.training.policy import ExpansionPolicy
from incremental_feature_cl.training.trainer import ContinualTrainer, TrainerConfig

from ..fingerprints import state_dict_fingerprint
from ..models.accounting import head_state_bytes, unique_parameter_count
from ..models.features import build_head


@dataclass(frozen=True)
class FlatConfig:
    lr: float = 0.05
    momentum: float = 0.9
    train_epochs: int = 5
    train_mb_size: int = 32
    new_feature_dim: int = (
        0  # 0: no feature-space expansion (primary experiment isolates the hierarchy)
    )

    def to_dict(self) -> dict:
        return asdict(self)


def decide(logits: torch.Tensor) -> torch.Tensor:
    """argmax over all output columns; ties go to the LOWEST column (explicit, not backend-defined)."""
    n = logits.shape[1]
    is_max = logits == logits.max(dim=1, keepdim=True).values
    weight = torch.arange(
        n, 0, -1, device=logits.device
    )  # n, n-1, ..., 1  -> lowest index wins
    return (is_max.long() * weight).argmax(dim=1)


class FlatSoftmax:
    name = "flat_softmax"
    prediction_rule = (
        "argmax over native logits (cross-entropy trained), ties -> lowest class id"
    )

    def __init__(self, feature_dim: int, cfg: FlatConfig, seed: int):
        self.cfg, self.feature_dim = cfg, feature_dim
        self.model = build_head(feature_dim, num_outputs=1, seed=seed)
        self._policy = ExpansionPolicy(new_feature_dim=cfg.new_feature_dim)
        self._trainer = ContinualTrainer(
            self.model,
            self._policy,
            TrainerConfig(
                mode="multiclass",
                lr=cfg.lr,
                momentum=cfg.momentum,
                train_epochs=cfg.train_epochs,
                train_mb_size=cfg.train_mb_size,
                replay_mem_size=0,
                seed=seed,
            ),
        )
        self._known: list[int] = []

    @property
    def known_classes(self) -> list[int]:
        return list(self._known)

    def output_classes(self) -> list[int]:
        """Internal class id represented by each output index (here: output j == internal class j)."""
        return list(range(self.model.num_outputs))

    def state_fingerprint(self) -> str:
        return state_dict_fingerprint(self.model)

    def learn_experience(
        self, index: int, classes: list[int], x: torch.Tensor, y: torch.Tensor
    ) -> None:
        """x, y = current-experience data ∪ replay memory (labels are internal ids)."""
        self._policy.expand_features(self.model, index)
        self._policy.expand_outputs(self.model, classes)
        self._known = sorted(set(self._known) | set(classes))
        if self.model.num_outputs != len(self._known):
            raise RuntimeError(
                "outputs must equal known classes (internal ids are arrival-ordered)"
            )
        self._trainer._train_experience(
            Experience(index, ArrayDataset(x, y), classes, classes)
        )

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Features only: no labels, no experience id."""
        was = self.model.training
        self.model.eval()
        pred = decide(self.model(x))
        self.model.train(was)
        return pred

    def resources(self) -> dict:
        return {
            "n_heads": 1,
            "head_parameters": unique_parameter_count([self.model]),
            "head_trainable_parameters": unique_parameter_count(
                [self.model], trainable_only=True
            ),
            "head_checkpoint_bytes": head_state_bytes(self.model),
            "node_decisions_per_prediction": 1,
        }
