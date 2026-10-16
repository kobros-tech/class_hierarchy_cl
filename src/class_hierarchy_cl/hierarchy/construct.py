# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Hierarchical agglomerative clustering (HAC) of classes + the paper's hierarchy compression.

Paper (Silva-Palacios et al. 2018): Sec. 3.3 (p.8 l.201 - p.9 l.227) HAC, bottom-up from singletons; linkages
complete = max, single = min, average = mean of pairwise distances between members (p.9 l.207-211).
Sec. 4.1 (p.10 l.235 - p.12 l.267) compression, Algorithm 1 (p.11):
    theta = alpha * max_k | linkage(k) - linkage(k+1) |            (p.11 l.253-259)
    depth-first; a node absorbs an INTERNAL child c when  height(node) - height(c) <= theta.
Not specified by the paper and fixed here (design decisions, recorded in every result):
  * tie-breaking in HAC: among pairs within `tol` of the minimum distance, merge the pair with the smallest
    (cluster id, cluster id) tuple; leaves keep their class id, merged clusters get ids n, n+1, ... in order;
  * heights compared with tolerance `tol` (equal heights from float arithmetic must count as equal);
  * only internal children can be absorbed (a leaf has no linkage height, absorbing it would delete a class);
  * the paper's text ("<") and Algorithm 1 ("<=") differ; "<=" is used because alpha = 0 is described as joining
    clusters at the SAME height (p.12 l.252-254);
  * for alpha > 0 the text (contiguous clusters in creation order) and Algorithm 1 (parent/child heights) are
    not equivalent; Algorithm 1 is implemented literally and is NOT used by the primary experiments (alpha = 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .tree import ClassTree

LINKAGES = ("complete", "single", "average")


def _check_distance(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("distance matrix must be square")
    if D.shape[0] < 2:
        raise ValueError("need at least 2 classes")
    if not np.all(np.isfinite(D)) or np.any(D < 0):
        raise ValueError("distances must be finite and non-negative")
    if not np.allclose(D, D.T, atol=1e-9):
        raise ValueError("distance matrix must be symmetric")
    return D


def _cluster_distance(D, a, b, linkage):
    block = D[np.ix_(a, b)]
    return {"complete": block.max, "single": block.min, "average": block.mean}[
        linkage
    ]()


def linkage_merges(
    D: np.ndarray, linkage: str = "complete", tol: float = 1e-12
) -> list[tuple[int, int, float, int]]:
    """List of (cluster_a, cluster_b, height, new_cluster_id); ids < n are classes. Deterministic."""
    if linkage not in LINKAGES:
        raise ValueError(f"linkage must be one of {LINKAGES}")
    D = _check_distance(D)
    n = len(D)
    members = {i: [i] for i in range(n)}
    merges = []
    for step in range(n - 1):
        ids = sorted(members)
        cand = [
            (a, b, _cluster_distance(D, members[a], members[b], linkage))
            for i, a in enumerate(ids)
            for b in ids[i + 1 :]
        ]
        best = min(d for _, _, d in cand)
        a, b, d = next(
            c for c in cand if c[2] <= best + tol
        )  # cand is in lexicographic (a, b) order
        new = n + step
        members[new] = members.pop(a) + members.pop(b)
        merges.append((a, b, float(d), new))
    return merges


def _tree_from_merges(n: int, merges) -> ClassTree:
    kids = {new: (a, b) for a, b, _, new in merges}
    heights = {new: h for _, _, h, new in merges}
    return ClassTree.from_edges(n, merges[-1][3], kids, heights)


def compress(
    tree: ClassTree, alpha: float, tol: float = 1e-12
) -> tuple[ClassTree, float]:
    """Paper Algorithm 1. Returns (compressed tree, theta)."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    hs = sorted(
        tree.height.values()
    )  # monotone linkages: creation order == height order
    theta = alpha * (max(abs(b - a) for a, b in pairwise(hs)) if len(hs) > 1 else 0.0)
    kids = {n: list(cs) for n, cs in tree.children.items()}
    height = dict(tree.height)

    def verify_level(node):
        changed = True
        while changed:
            changed = False
            for child in list(kids[node]):
                if tree.is_leaf(child):
                    continue
                if (
                    height[node] - height[child] <= theta + tol
                ):  # JoinNodes: father adopts the child's children
                    kids[node].remove(child)
                    kids[node].extend(kids.pop(child))
                    height.pop(child)
                    changed = True
                    break

    def process(node):
        verify_level(node)
        for child in list(kids[node]):
            if not tree.is_leaf(child):
                process(child)

    process(tree.root)
    out = ClassTree.from_edges(
        tree.num_classes, tree.root, {k: tuple(v) for k, v in kids.items()}, height
    )
    return out, float(theta)


@dataclass(frozen=True)
class HierarchyResult:
    tree: ClassTree  # the hierarchy to use (after compression)
    uncompressed: ClassTree  # plain binary HAC tree
    merges: list
    theta: float
    params: dict

    @property
    def classifiers_saved(self) -> int:
        return len(self.uncompressed.children) - len(self.tree.children)


def build_hierarchy(
    distance: np.ndarray,
    linkage: str = "complete",
    alpha: float = 0.0,
    tol: float = 1e-12,
) -> HierarchyResult:
    """HAC on a class distance matrix (e.g. the paper's d_S), then compression with parameter alpha."""
    D = _check_distance(distance)
    merges = linkage_merges(D, linkage, tol)
    raw = _tree_from_merges(len(D), merges)
    tree, theta = compress(raw, alpha, tol)
    return HierarchyResult(
        tree,
        raw,
        merges,
        theta,
        {
            "linkage": linkage,
            "alpha": alpha,
            "tol": tol,
            "tie_break": "smallest (id, id) pair within tol",
            "compression": "paper Algorithm 1, '<=', internal children only",
        },
    )
