# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Class-incremental protocol: splits, class order, relabelling, cached frozen features.

* train / val / test are fixed before any method runs; val is carved out of the training split
  (stratified, seeded) and is the ONLY held-out data hierarchy construction or calibration may use.
* classes are relabelled to *arrival order* (internal id = position in class_order), so a model
  with ``num_outputs == number of known classes`` has exactly one output per known class.
* methods never see test data; test arrays live in ``Protocol`` and are read only by the runner.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from incremental_feature_cl.data.datasets import ArrayDataset
from incremental_feature_cl.data.streams import build_class_incremental_stream
from torch.utils.data import DataLoader

from .models.features import FrozenFeatureExtractor
from .seeds import derive_seed


@dataclass(frozen=True)
class ProtocolConfig:
    dataset_id: str
    n_experiences: int
    seed: int = 0
    class_order: tuple[int, ...] | None = (
        None  # explicit original-id order; None -> shuffled by seed
    )
    val_fraction: float = 0.2
    replay_capacity: int = (
        0  # total exemplars kept across ALL old classes; 0 = no replay
    )
    standardize_features: bool = (
        True  # centre + global scale, fitted on experience-0 TRAIN features only
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperienceData:
    index: int
    classes: list[int]  # internal ids introduced in this experience
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor


@dataclass
class Protocol:
    config: ProtocolConfig
    class_order: tuple[
        int, ...
    ]  # ORIGINAL label values in arrival order; internal id = index
    experiences: list[ExperienceData]
    test_x: torch.Tensor
    test_y: torch.Tensor  # internal ids
    feature_dim: int
    split_fingerprint: str
    extractor_fingerprint: str
    extractor_spec: dict = field(default_factory=dict)
    feature_mean: torch.Tensor | None = (
        None  # applied to every split; fitted on experience 0 train only
    )
    feature_scale: float = 1.0

    @property
    def num_classes(self) -> int:
        return len(self.class_order)

    @property
    def first_exp(self) -> dict[int, int]:
        return {c: e.index for e in self.experiences for c in e.classes}

    def seen_classes(self, t: int) -> list[int]:
        return [c for e in self.experiences[: t + 1] for c in e.classes]

    def original_class(self, internal: int) -> int:
        return self.class_order[internal]

    def to_original(self, internal: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.class_order, dtype=torch.long)[internal]


def materialize(
    ds: ArrayDataset, batch_size: int = 512
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the dataset's own transform (uint8 scaling, mean/std) once."""
    xs, ys = [], []
    for x, y in DataLoader(ds, batch_size=batch_size, shuffle=False):
        xs.append(x)
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def stratified_val_split(
    y: torch.Tensor, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """(train_idx, val_idx); every class keeps >= 1 train and (if it has >= 2 samples) >= 1 val."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    y_np = y.cpu().numpy()
    tr, va = [], []
    for c in np.unique(y_np):
        idx = rng.permutation(np.flatnonzero(y_np == c))
        n_val = round(val_fraction * len(idx))
        n_val = min(max(n_val, 1), len(idx) - 1) if len(idx) > 1 else 0
        va.append(idx[:n_val])
        tr.append(idx[n_val:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va))


def build_protocol(
    train: ArrayDataset,
    test: ArrayDataset,
    cfg: ProtocolConfig,
    extractor: FrozenFeatureExtractor,
) -> Protocol:
    x_tr, y_tr_raw = materialize(train)
    x_te, y_te_raw = materialize(test)
    # Original labels may be arbitrary integers (non-contiguous). main's stream needs 0..C-1, so map
    # to dense ids by sorted value; Protocol.class_order reports ORIGINAL label values.
    values = torch.unique(torch.cat([y_tr_raw, y_te_raw]))
    if not torch.isin(y_te_raw, torch.unique(y_tr_raw)).all():
        raise ValueError("test split contains classes absent from the train split")
    y_tr, y_te = (
        torch.searchsorted(values, y_tr_raw),
        torch.searchsorted(values, y_te_raw),
    )
    dense_order = None
    if cfg.class_order is not None:
        lookup = {int(v): i for i, v in enumerate(values.tolist())}
        missing = [c for c in cfg.class_order if int(c) not in lookup]
        if missing:
            raise ValueError(
                f"class_order contains labels absent from the data: {missing}"
            )
        dense_order = [lookup[int(c)] for c in cfg.class_order]
    f_tr, f_te = (
        extractor.extract(x_tr),
        extractor.extract(x_te),
    )  # each sample featurised once

    tr_idx, va_idx = stratified_val_split(
        y_tr, cfg.val_fraction, derive_seed(cfg.seed, "split")
    )
    train_part = ArrayDataset(f_tr[tr_idx], y_tr[tr_idx])
    order_seed = derive_seed(cfg.seed, "class_order")
    stream = build_class_incremental_stream(
        train_part,
        ArrayDataset(f_te, y_te),
        cfg.n_experiences,
        seed=order_seed,
        class_order=dense_order,
    )
    order = tuple(
        int(values[c]) for c in stream.class_order
    )  # original label values, arrival order
    to_internal = torch.full((len(order),), -1, dtype=torch.long)
    for i, c in enumerate(stream.class_order):  # dense id -> arrival position
        to_internal[int(c)] = i

    f_va, y_va = f_tr[va_idx], y_tr[va_idx]
    exps = []
    for e in stream.train:
        orig = torch.as_tensor(e.classes, dtype=torch.long)
        vmask = torch.isin(y_va, orig)
        exps.append(
            ExperienceData(
                index=e.index,
                classes=sorted(int(to_internal[c]) for c in e.classes),
                train_x=e.dataset.x,
                train_y=to_internal[e.dataset.targets],
                val_x=f_va[vmask],
                val_y=to_internal[y_va[vmask]],
            )
        )
    mean, scale = None, 1.0
    if cfg.standardize_features:
        # Random frozen features have tiny magnitude (std ~1e-3), too small for SGD on a linear head.
        # Statistics come from experience 0's training data only (no val/test/future data), are never
        # updated afterwards, and are shared by every method.
        x0 = exps[0].train_x
        mean = x0.mean(0)
        scale = float((x0 - mean).std().clamp_min(1e-12))
        f = lambda t: (t - mean) / scale
        for e in exps:
            e.train_x, e.val_x = f(e.train_x), f(e.val_x)
        f_te = f(f_te)
    h = hashlib.sha256()
    for a in (tr_idx, va_idx, np.asarray(order)):
        h.update(np.ascontiguousarray(a, dtype=np.int64).tobytes())
    return Protocol(
        cfg,
        order,
        exps,
        f_te,
        to_internal[y_te],
        extractor.out_dim,
        h.hexdigest(),
        extractor.fingerprint(),
        extractor.spec.to_dict(),
        mean,
        scale,
    )
