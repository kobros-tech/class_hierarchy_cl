# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
from helpers import M_PAPER, PLANTED, tree_from

from class_hierarchy_cl.hierarchy import (
    ClassTree,
    build_hierarchy,
    random_relabel,
    similarity_matrix,
)


def test_valid_trees_two_and_several_classes():
    t2 = tree_from((0, 1), 2)
    assert t2.children == {2: (0, 1)} and t2.root == 2
    t = tree_from(((0, 1), (2, 3), 4), 5)
    t.validate()
    assert sorted(t.classes_under(t.root)) == [0, 1, 2, 3, 4]
    assert t.stats()["n_internal"] == 3


def test_canonical_ids_and_child_order_are_input_independent():
    a = tree_from(((3, 2), (1, 0)), 4)
    b = tree_from(((0, 1), (2, 3)), 4)
    assert (a.root, a.children, a.height) == (b.root, b.children, b.height)
    assert a.children[a.root] == (5, 6) and a.classes_under(5) == (0, 1)


@pytest.mark.parametrize(
    "children,heights,msg",
    [
        ({4: (5, 0), 5: (4, 1)}, {4: 2.0, 5: 1.0}, "twice"),  # cycle
        (
            {4: (0, 1), 5: (2, 3)},
            {4: 1.0, 5: 1.0},
            "exactly once",
        ),  # root only reaches 0,1
        ({4: (0, 1, 1, 2, 3)}, {4: 1.0}, "twice"),  # duplicate leaf
        ({4: (0,)}, {4: 1.0}, "fewer than 2"),  # unary node
        (
            {4: (0, 1, 2), 6: (3, 5)},
            {4: 1.0, 6: 1.0},
            "exactly once",
        ),  # class 3 unreachable
        (
            {4: (5, 3), 5: (0, 1, 2)},
            {4: 1.0, 5: 2.0},
            "decrease",
        ),  # heights decrease towards root
        ({4: (5, 3), 5: (0, 1, 2)}, {4: 2.0}, "heights"),  # missing height
    ],
)
def test_invalid_trees_are_rejected(children, heights, msg):
    with pytest.raises(ValueError, match=msg):
        ClassTree(4, 4, children, heights).validate()


def test_unreachable_internal_node_and_leaf_root_rejected():
    with pytest.raises(ValueError):
        ClassTree(3, 3, {3: (0, 1, 2), 4: (0, 1)}, {3: 1.0, 4: 1.0}).validate()
    with pytest.raises(ValueError):
        ClassTree(3, 0, {3: (0, 1, 2)}, {3: 1.0}).validate()


def test_node_label_mapping_partitions_the_subtree_classes():
    h = build_hierarchy(
        similarity_matrix(M_PAPER), "complete", 0.0
    ).tree  # root( {a,c}, b, d )
    assert h.children[h.root] == (5, 1, 3) and h.classes_under(5) == (0, 2)
    assert h.class_to_child(h.root) == {0: 0, 2: 0, 1: 1, 3: 2}
    for (
        node
    ) in h.internal_nodes:  # disjoint child groups whose union is the node's class set
        groups = [h.classes_under(c) for c in h.children[node]]
        flat = [c for g in groups for c in g]
        assert sorted(flat) == sorted(set(flat)) == list(h.classes_under(node))
        assert set(h.class_to_child(node)) == set(h.classes_under(node))
    y = np.array([0, 1, 2, 3, 0])
    mask, child = h.node_targets(h.root, y)
    assert mask.all() and child.tolist() == [0, 1, 0, 2, 0]
    mask, child = h.node_targets(
        5, y
    )  # node {a,c}: only a/c samples belong; a -> 0, c -> 1
    assert mask.tolist() == [True, False, True, False, True]
    assert child[mask].tolist() == [0, 1, 0]
    # mapping is stable across calls
    assert h.node_targets(5, y)[1].tolist() == child.tolist()


def test_binary_tree_node_targets_are_two_way():
    t = tree_from(((0, 1), (2, 3)), 4)
    mask, child = t.node_targets(t.root, np.array([0, 1, 2, 3]))
    assert mask.all() and child.tolist() == [0, 0, 1, 1]


def test_path_depth_and_reachability():
    t = tree_from((((0, 1), 2), 3), 4)
    for c in range(4):
        p = t.path(c)
        assert (
            p[0] == t.root and p[-1] == c and all(t.is_leaf(x) is False for x in p[:-1])
        )
    assert [t.depth(c) for c in range(4)] == [3, 3, 2, 1]


def test_random_relabel_matches_structure_but_not_grouping():
    t = tree_from(PLANTED, 8)
    seen_groupings = set()
    for seed in range(30):
        r = random_relabel(t, np.random.default_rng(seed))
        r.validate()
        assert r.shape_signature() == t.shape_signature()
        assert r.stats() == t.stats()
        assert sorted(r.height.values()) == sorted(t.height.values())
        assert sorted(c for c in r.classes_under(r.root)) == list(range(8))
        seen_groupings.add(r.groups())
    assert len(seen_groupings) > 20  # genuinely different groupings
    a = random_relabel(t, np.random.default_rng(5))
    b = random_relabel(t, np.random.default_rng(5))
    assert a == b  # deterministic given the seed


def test_random_relabel_rarely_recovers_the_planted_pairs():
    t = tree_from(PLANTED, 8)
    pairs = {frozenset(p) for p in [(0, 1), (2, 3), (4, 5), (6, 7)]}
    hits = 0
    for seed in range(400):
        r = random_relabel(t, np.random.default_rng(seed))
        sib = {
            frozenset(r.classes_under(n))
            for n in r.internal_nodes
            if len(r.classes_under(n)) == 2
        }
        hits += sib == pairs
    assert hits / 400 < 0.05  # chance level is 384/40320 ~ 0.01


def test_node_targets_rejects_out_of_range_labels_instead_of_wrapping():
    t = build_hierarchy(similarity_matrix(M_PAPER), "complete").tree
    for bad in ([-1], [4], [0, 1, -3]):
        with pytest.raises(ValueError, match="labels must lie"):
            t.node_targets(t.root, np.array(bad))
    assert t.node_targets(t.root, np.array([], dtype=int))[0].shape == (0,)
