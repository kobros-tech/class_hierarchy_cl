# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def to_jsonable(o: Any) -> Any:
    """numpy/torch -> plain python; NaN/inf -> None (strict JSON)."""
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, torch.Tensor):
        o = o.detach().cpu().numpy()
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def save_run(record: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(record), indent=1, allow_nan=False))
    return path


def load_run(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


_TIMING_KEYS = {
    "train_time_s",
    "infer_time_s",
    "infer_ms_per_example",
    "total_train_time_s",
}


def strip_timing(o: Any) -> Any:
    """Drop wall-clock fields so deterministic artifacts can be compared exactly."""
    if isinstance(o, dict):
        return {k: strip_timing(v) for k, v in o.items() if k not in _TIMING_KEYS}
    if isinstance(o, list):
        return [strip_timing(v) for v in o]
    return o
