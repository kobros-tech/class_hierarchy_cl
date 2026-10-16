# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Shared test fixtures for the hierarchy tests."""

import numpy as np

from class_hierarchy_cl.hierarchy import ClassTree

# Paper Table 1a: classes a, b, c, d
M_PAPER = np.array(
    [[20, 0, 30, 0], [0, 20, 30, 0], [30, 30, 10, 0], [0, 0, 0, 100]], float
)


def tree_from(nested, n, heights=None):
    """Helper: nested tuples of class ids -> ClassTree (heights default to depth-based, monotone)."""
    kids, hts, counter = {}, {}, [0]

    def visit(x):
        if isinstance(x, int):
            return x, 0.0
        key = ("n", counter[0])
        counter[0] += 1
        parts = [visit(c) for c in x]
        kids[key] = tuple(p[0] for p in parts)
        hts[key] = (heights or {}).get(key, max(p[1] for p in parts) + 1.0)
        return key, hts[key]

    root, _ = visit(nested)
    return ClassTree.from_edges(n, root, kids, hts)


PLANTED = ((0, 1), (2, 3)), ((4, 5), (6, 7))
