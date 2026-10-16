# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Class hierarchy as an explicit tree: leaves = classes, internal nodes = groups of classes.

Identifiers
  leaf id  == the class's internal id (0..C-1, arrival order); stable by construction
  internal ids C, C+1, ... assigned canonically: preorder from the root, children ordered by their smallest
  descendant class. Two equal groupings therefore always have identical ids and child orders.
A node with k children represents a k-way decision among its child groups (k == 2 for a plain HAC tree;
k > 2 appears after compression, see construct.compress).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassTree:
    num_classes: int
    root: int
    children: dict[int, tuple[int, ...]]  # internal node id -> ordered children
    height: dict[int, float]  # internal node id -> linkage height (leaves have none)

    # -------------------------------------------------------------------- structure
    @property
    def internal_nodes(self) -> list[int]:
        return sorted(self.children)

    @property
    def parent(self) -> dict[int, int]:
        return {c: p for p, cs in self.children.items() for c in cs}

    def is_leaf(self, node: int) -> bool:
        return node < self.num_classes

    def classes_under(self, node: int) -> tuple[int, ...]:
        if self.is_leaf(node):
            return (node,)
        return tuple(
            sorted(c for ch in self.children[node] for c in self.classes_under(ch))
        )

    def groups(self):
        """Nested frozenset form, ignores node ids and child order (for structure comparison)."""

        def g(n):
            return n if self.is_leaf(n) else frozenset(g(c) for c in self.children[n])

        return g(self.root)

    def path(self, cls: int) -> list[int]:
        """Node ids from the root to the leaf of `cls`."""
        par, out, n = self.parent, [cls], cls
        while n != self.root:
            n = par[n]
            out.append(n)
        return out[::-1]

    def depth(self, cls: int) -> int:
        return len(self.path(cls)) - 1

    def shape_signature(self):
        """Nested tuple of sorted child signatures: equal for trees of identical shape, whatever the classes."""

        def s(n):
            return (
                () if self.is_leaf(n) else tuple(sorted(s(c) for c in self.children[n]))
            )

        return s(self.root)

    def stats(self) -> dict:
        depths = [self.depth(c) for c in range(self.num_classes)]
        return {
            "n_internal": len(self.children),
            "max_depth": max(depths),
            "mean_depth": float(np.mean(depths)),
            "max_arity": max(len(c) for c in self.children.values()),
            "n_binary": sum(len(c) == 2 for c in self.children.values()),
        }

    # -------------------------------------------------------------------- validation
    def validate(self) -> None:
        C = self.num_classes
        if self.root in range(C) and C > 1:
            raise ValueError("root must be an internal node")
        seen_leaves, visited, stack = [], set(), [self.root]
        while stack:
            n = stack.pop()
            if n in visited:
                raise ValueError(f"node {n} reached twice (cycle or shared child)")
            visited.add(n)
            if self.is_leaf(n):
                seen_leaves.append(n)
            else:
                if n not in self.children:
                    raise ValueError(f"internal node {n} has no children entry")
                if len(self.children[n]) < 2:
                    raise ValueError(f"internal node {n} has fewer than 2 children")
                stack.extend(self.children[n])
        if sorted(seen_leaves) != list(range(C)):
            raise ValueError("every class must appear exactly once as a reachable leaf")
        if set(self.children) - visited:
            raise ValueError("unreachable internal nodes")
        if set(self.height) != set(self.children):
            raise ValueError("heights must be defined for exactly the internal nodes")
        for p, cs in self.children.items():
            for c in cs:
                if not self.is_leaf(c) and self.height[c] > self.height[p] + 1e-9:
                    raise ValueError(
                        "linkage heights must not decrease towards the root"
                    )

    # -------------------------------------------------------------------- node label mapping
    def class_to_child(self, node: int) -> dict[int, int]:
        """For an internal node: class (under that node) -> index of the child group containing it."""
        return {
            c: i
            for i, ch in enumerate(self.children[node])
            for c in self.classes_under(ch)
        }

    def node_targets(self, node: int, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Training-time label conversion. Returns (mask, child_index).

        mask[i] is True iff y[i] lies under `node`; child_index[i] (valid where mask) is the node-local label,
        i.e. the index of the child group holding y[i]. Needs original class labels only at TRAINING time;
        prediction never calls this.
        """
        y = np.asarray(y, dtype=int)
        if len(y) and (y.min() < 0 or y.max() >= self.num_classes):
            raise ValueError(
                f"labels must lie in [0, {self.num_classes}); got range [{y.min()}, {y.max()}] "
                "(negative values would silently index from the end)"
            )
        lut = np.full(self.num_classes, -1, dtype=int)
        for c, i in self.class_to_child(node).items():
            lut[c] = i
        child = lut[y]
        return child >= 0, child

    # -------------------------------------------------------------------- canonical construction
    @staticmethod
    def from_edges(num_classes: int, root_key, kids: dict, heights: dict) -> ClassTree:
        """Canonicalise an arbitrary-keyed tree: leaves are ints < num_classes, internal keys anything hashable."""

        def is_leaf(k):
            return (
                isinstance(k, (int, np.integer)) and k < num_classes and k not in kids
            )

        def smallest(k):
            return k if is_leaf(k) else min(smallest(c) for c in kids[k])

        ids, children, height = {}, {}, {}
        nxt = [num_classes]

        def visit(k):
            if is_leaf(k):
                return int(k)
            nid = nxt[0]
            nxt[0] += 1
            order = sorted(kids[k], key=smallest)
            ids[k] = nid
            children[nid] = tuple(visit(c) for c in order)
            height[nid] = float(heights[k])
            return nid

        root = visit(root_key)
        t = ClassTree(num_classes, root, children, height)
        t.validate()
        return t


def random_relabel(tree: ClassTree, rng: np.random.Generator) -> ClassTree:
    """Shape-preserving random control: identical tree shape, heights and arities; classes permuted over leaves.

    Same number of internal nodes, depth profile, balance and per-node arity as `tree`; only the grouping
    information is destroyed. Canonicalised, so ids and child orders follow the same rules as any other tree.
    """
    perm = rng.permutation(tree.num_classes)
    kids = {
        ("i", n): tuple(int(perm[c]) if tree.is_leaf(c) else ("i", c) for c in cs)
        for n, cs in tree.children.items()
    }
    return ClassTree.from_edges(
        tree.num_classes,
        ("i", tree.root),
        kids,
        {("i", n): h for n, h in tree.height.items()},
    )
