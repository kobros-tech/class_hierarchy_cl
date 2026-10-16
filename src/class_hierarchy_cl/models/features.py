# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""One shared, frozen feature extractor + heads, all built from main's model code.

Design (primary experiment = frozen-representation CL):
  * the backbone comes from ``incremental_feature_cl.build_backbone`` and is initialised
    from an explicit seed; it is NOT pretrained and never trained, so its features are
    random-network features;
  * every method (flat, hierarchy nodes, random-hierarchy nodes) trains an unmodified
    ``IncrementalFeatureMapModel`` head whose backbone is ``IdentityBackbone`` over the
    cached feature vectors. Identical features by construction; tests also check that this
    equals main's model run end-to-end with the same shared frozen backbone.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from incremental_feature_cl import IncrementalFeatureMapModel, build_backbone
from incremental_feature_cl.models import IdentityBackbone
from torch import nn


@dataclass(frozen=True)
class BackboneSpec:
    name: str = "slimresnet18"
    kwargs: dict[str, Any] = field(default_factory=lambda: {"nf": 8})
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_frozen_backbone(spec: BackboneSpec) -> nn.Module:
    with torch.random.fork_rng(devices=[]):  # does not disturb the caller's RNG stream
        torch.manual_seed(spec.seed)
        bb = build_backbone(spec.name, **spec.kwargs)
    for p in bb.parameters():
        p.requires_grad_(False)
    return bb.eval()


class FrozenFeatureExtractor:
    """Runs the frozen backbone in eval mode under no_grad; never changes its state."""

    def __init__(self, spec: BackboneSpec, device: str = "cpu"):
        self.spec = spec
        self.backbone = build_frozen_backbone(spec).to(device)
        self.device = torch.device(device)
        self.out_dim = int(self.backbone.out_dim)

    @torch.no_grad()
    def extract(self, x: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
        self.backbone.eval()
        outs = []
        for i in range(0, len(x), batch_size):
            h = self.backbone(x[i : i + batch_size].to(self.device))
            outs.append((h.flatten(1) if h.ndim > 2 else h).cpu())
        return torch.cat(outs) if outs else torch.empty(0, self.out_dim)

    def fingerprint(self) -> str:
        """Hash of all backbone parameters and buffers (recorded in every run artifact)."""
        h = hashlib.sha256()
        for k, v in sorted(self.backbone.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()


def build_head(
    feature_dim: int, num_outputs: int, seed: int, **model_kwargs
) -> IncrementalFeatureMapModel:
    """An unmodified main-branch model over cached features (IdentityBackbone has no parameters).

    main's constructor does not seed the INITIAL classifier layer from ``seed`` (it uses the global
    RNG; ``seed`` only governs later expansions), so construction happens under a local seeded RNG
    scope: reproducible, and the caller's global RNG stream is untouched.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return IncrementalFeatureMapModel(
            IdentityBackbone(feature_dim),
            num_outputs=num_outputs,
            seed=seed,
            **model_kwargs,
        )
