# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Runs one method through a Protocol and returns an auditable record (no plotting, no I/O)."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from .. import __version__
from ..evaluation.metrics import evaluate_stage, summarize
from ..fingerprints import tensor_fingerprint
from ..memory import ExemplarMemory
from ..protocol import Protocol
from ..provenance import code_revision, main_revision
from ..seeds import derive_seed

_PKG_ROOT = Path(__file__).resolve().parents[3]


def run_protocol(
    method,
    protocol: Protocol,
    *,
    method_config: dict | None = None,
    save_predictions: bool = True,
) -> dict:
    cfg = protocol.config
    memory = ExemplarMemory(
        cfg.replay_capacity, protocol.num_classes, derive_seed(cfg.seed, "memory")
    )
    stages, rec_stages, warnings = [], [], []
    for exp in protocol.experiences:
        # ---- train on current experience ∪ memory of PREVIOUS experiences (no test, no future data)
        mx, my = memory.contents(protocol.feature_dim)
        x = torch.cat([exp.train_x, mx])
        y = torch.cat([exp.train_y, my])
        train_set_fp, replay_fp = tensor_fingerprint(x, y), tensor_fingerprint(mx, my)
        t0 = time.perf_counter()
        method.learn_experience(exp.index, exp.classes, x, y)
        train_s = time.perf_counter() - t0
        memory.update(
            exp.train_x, exp.train_y
        )  # only AFTER training, with this experience's data

        # ---- evaluate on test samples of known classes; method sees features only
        known = protocol.seen_classes(exp.index)
        if sorted(method.known_classes) != sorted(known):
            raise RuntimeError("method's known classes differ from the protocol's")
        mask = torch.isin(protocol.test_y, torch.as_tensor(known))
        tx, ty = protocol.test_x[mask], protocol.test_y[mask]
        t0 = time.perf_counter()
        pred = method.predict(tx)
        infer_s = time.perf_counter() - t0
        st = evaluate_stage(pred.numpy(), ty.numpy(), protocol.num_classes, known)
        stages.append(st)
        rec_stages.append(
            {
                "experience": exp.index,
                "new_classes": exp.classes,
                "known_classes": known,
                "n_train_current": len(exp.train_x),
                "n_train_replay": len(mx),
                "memory_size": len(memory),
                "memory_class_counts": memory.class_counts(),
                "train_set_fingerprint": train_set_fp,
                "replay_fingerprint": replay_fp,
                "memory_fingerprint_after": memory.fingerprint(protocol.feature_dim),
                "model_fingerprint": method.state_fingerprint(),
                "output_to_internal": method.output_classes(),
                "output_to_original": [
                    protocol.original_class(c) for c in method.output_classes()
                ],
                "train_time_s": train_s,
                "infer_time_s": infer_s,
                "infer_ms_per_example": 1e3 * infer_s / len(tx),
                "n_eval": st.n_eval,
                "acc_known": st.acc_known,
                "per_class_acc": st.per_class_acc,
                "confusion": st.confusion,
                "pred_counts": st.pred_counts,
                **(
                    {"test_predictions": pred.numpy(), "test_labels": ty.numpy()}
                    if save_predictions
                    else {}
                ),
            }
        )
    summary = summarize(stages, protocol.first_exp)
    if any(r["n_train_replay"] > cfg.replay_capacity for r in rec_stages):
        warnings.append("replay exceeded capacity")
    prov = {
        "package_version": __version__,
        "code_revision": code_revision(_PKG_ROOT),
        "incremental_feature_cl_revision": main_revision(),
    }
    if prov["code_revision"] is None:
        warnings.append("code_revision unavailable: package is not in a git checkout")
    if prov["incremental_feature_cl_revision"] is None:
        warnings.append(
            "incremental_feature_cl revision unavailable: not a pinned git install"
        )
    return {
        "schema": 1,
        "provenance": prov,
        "protocol": cfg.to_dict(),
        "class_order_original_ids": list(protocol.class_order),
        "split_fingerprint": protocol.split_fingerprint,
        "backbone": {
            "fingerprint": protocol.extractor_fingerprint,
            **protocol.extractor_spec,
            "frozen": True,
            "pretrained": False,
            "standardization": {
                "enabled": protocol.feature_mean is not None,
                "fit_on": "experience_0_train",
                "scale": protocol.feature_scale,
            },
        },
        "method": {
            "name": method.name,
            "prediction_rule": method.prediction_rule,
            "config": method_config or {},
        },
        "stages": rec_stages,
        "summary": summary,
        "resources": {
            **method.resources(),
            "total_train_time_s": sum(r["train_time_s"] for r in rec_stages),
            "final_memory_size": len(memory),
            "replay_capacity": cfg.replay_capacity,
        },
        "warnings": warnings,
    }
