# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Confusion matrix -> similarity matrix, exactly as Silva-Palacios et al. (2018), Section 3.

Paper (accepted manuscript JOCS 824), notation kept:
  M          n x n confusion matrix of a flat classifier F; rows = actual class, cols = predicted (p.6 l.139-145)
  M_bar      row-normalised M:  M_bar[i,j] = M[i,j] / sum_k M[i,k]                          (p.6 l.155)
  overlap    Def.1: overlap(i,j) = (M_bar[i,j] + M_bar[j,i]) / 2  for i != j, 1 for i == j   (p.6 l.157-160)
  d_S        Def.2: d_S(i,j) = 1 - overlap(i,j)   (symmetric, in [0,1], zero diagonal;
             a semi-metric: the triangle inequality can fail, Example 1)                      (p.6 l.161-p.7 l.169)
  probabilistic variant (p.8 l.187-195):  M[i,j] = sum over instances x of real class i of Pi(j | x)

Everything here takes plain arrays: no dataset, no model, no labels from any split.
"""

from __future__ import annotations

import numpy as np


class DegenerateConfusionError(ValueError):
    """The paper's definitions are undefined for this input (e.g. a class with no confusion evidence)."""


def _check_square(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"confusion matrix must be square, got shape {M.shape}")
    if M.shape[0] < 2:
        raise ValueError("need at least 2 classes")
    if not np.all(np.isfinite(M)) or np.any(M < 0):
        raise ValueError("confusion matrix must be finite and non-negative")
    return M


def confusion_from_predictions(pred: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Counts matrix (rows actual, cols predicted) from crisp class estimates (paper Sec. 3.1)."""
    pred, y = np.asarray(pred, dtype=int), np.asarray(y, dtype=int)
    if pred.shape != y.shape or pred.ndim != 1:
        raise ValueError("pred and y must be 1-d arrays of equal length")
    if len(y) and (min(pred.min(), y.min()) < 0 or max(pred.max(), y.max()) >= n):
        raise ValueError("class id out of range")
    M = np.zeros((n, n))
    np.add.at(M, (y, pred), 1.0)
    return M


def confusion_from_probabilities(
    P: np.ndarray, y: np.ndarray, n: int, atol: float = 1e-6
) -> np.ndarray:
    """Paper Sec. 3.2: M[i,j] = sum over <x,i> in D of Pi(j|x).

    P rows must be probability vectors; anything else is rejected rather than silently treated as
    calibrated probabilities (raw logits, unnormalised scores, NaN).
    """
    P, y = np.asarray(P, dtype=float), np.asarray(y, dtype=int)
    if P.ndim != 2 or P.shape[1] != n or len(P) != len(y):
        raise ValueError("P must have shape (len(y), n)")
    if not np.all(np.isfinite(P)) or np.any(P < -atol):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(P.sum(1), 1.0, atol=atol):
        raise ValueError(
            "each probability row must sum to 1 (got unnormalised scores?)"
        )
    if len(y) and (y.min() < 0 or y.max() >= n):
        raise ValueError("class id out of range")
    M = np.zeros((n, n))
    np.add.at(M, y, P)
    return M


def row_normalize(M: np.ndarray, empty_rows: str = "error") -> np.ndarray:
    """M_bar[i,j] = M[i,j] / sum_k M[i,k].

    A row with sum 0 (a class with no confusion evidence) is undefined in the paper.
      empty_rows="error"   : raise DegenerateConfusionError (default)
      empty_rows="isolate" : NOT in the paper. Row := unit vector e_i (class never confused with others,
                             i.e. d_S = 1 to every other class). Recorded as a deviation when used.
    """
    if empty_rows not in ("error", "isolate"):
        raise ValueError("empty_rows must be 'error' or 'isolate'")
    M = _check_square(M)
    sums = M.sum(1)
    empty = sums == 0
    if empty.any():
        if empty_rows == "error":
            raise DegenerateConfusionError(
                f"classes {np.flatnonzero(empty).tolist()} have no confusion evidence"
            )
        M = M.copy()
        for i in np.flatnonzero(empty):
            M[i, i] = 1.0
        sums = M.sum(1)
    return M / sums[:, None]


def overlap_matrix(M_bar: np.ndarray) -> np.ndarray:
    """Def.1."""
    O = (M_bar + M_bar.T) / 2.0
    np.fill_diagonal(O, 1.0)
    return O


def similarity_matrix(M: np.ndarray, empty_rows: str = "error") -> np.ndarray:
    """Def.2: d_S = 1 - overlap, from a (count or probability-sum) confusion matrix M."""
    return 1.0 - overlap_matrix(row_normalize(M, empty_rows))
