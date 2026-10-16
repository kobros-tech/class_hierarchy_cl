# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib

import torch


def tensor_fingerprint(*tensors: torch.Tensor) -> str:
    """sha256 over dtype, shape and bytes of each tensor (order matters)."""
    h = hashlib.sha256()
    for t in tensors:
        t = t.detach().cpu().contiguous()
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def state_dict_fingerprint(model: torch.nn.Module) -> str:
    sd = model.state_dict()
    return tensor_fingerprint(*[sd[k] for k in sorted(sd)])
