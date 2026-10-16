# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Resource accounting that does not double count shared modules."""

from __future__ import annotations

import io
from collections.abc import Iterable

import torch
from torch import nn


def unique_parameter_count(
    modules: Iterable[nn.Module], trainable_only: bool = False
) -> int:
    """Parameters counted once even when several models share a module."""
    seen: dict[int, int] = {}
    for m in modules:
        for p in m.parameters():
            if p.requires_grad or not trainable_only:
                seen[id(p)] = p.numel()
    return sum(seen.values())


def head_state_bytes(model: nn.Module, exclude_prefix: str = "backbone.") -> int:
    """Serialized size of a model's state_dict without the (shared) backbone."""
    buf = io.BytesIO()
    torch.save(
        {
            k: v
            for k, v in model.state_dict().items()
            if not k.startswith(exclude_prefix)
        },
        buf,
    )
    return buf.tell()
