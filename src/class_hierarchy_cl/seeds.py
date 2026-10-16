# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib


def derive_seed(base: int, name: str) -> int:
    """Deterministic named sub-seed, so changing one component's seed never shifts another's."""
    h = hashlib.sha256(f"{int(base)}:{name}".encode()).digest()
    return int.from_bytes(h[:4], "little")
