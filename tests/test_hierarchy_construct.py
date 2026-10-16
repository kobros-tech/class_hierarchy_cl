# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
from helpers import M_PAPER, PLANTED, tree_from
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial.distance import squareform

from class_hierarchy_cl.hierarchy import (
    ClassTree,
    build_hierarchy,
    compress,
    confusion_from_predictions,
    linkage_merges,
    similarity_matrix,
)


def random_distance(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.random((n, n))
    D = (A + A.T) / 2
    np.fill_diagonal(D, 0)
    return D


def planted_confusion(noise_seed=0, jitter=0.01):
    """8 classes; planted tree (((0,1),(2,3)),((4,5),(6,7))). Partner 0.20, same-quad 0.05 each, far 0.0125 each."""
    n, rng = 8, np.random.default_rng(noise_seed)
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                P[i, j] = 0.65
            elif i // 2 == j // 2:
                P[i, j] = 0.20
            elif i // 4 == j // 4:
                P[i, j] = 0.05
            else:
                P[i, j] = 0.0125
    return 1000 * P * (1 + jitter * rng.random((n, n)))


# ------------------------------------------------------------------ HAC vs an independent implementation
@pytest.mark.parametrize("linkage", ["complete", "single", "average"])
@pytest.mark.parametrize("seed", range(6))
def test_hac_matches_scipy_on_tie_free_inputs(linkage, seed):
    D = random_distance(11, seed)
    ours = linkage_merges(D, linkage)
    Z = scipy_linkage(squareform(D, checks=False), method=linkage)
    members = {i: frozenset([i]) for i in range(11)}
    ours_sets, scipy_sets = [], []
    for k, (a, b, h, new) in enumerate(ours):
        members[new] = members[a] | members[b]
        ours_sets.append((members[new], h))
    sm = {i: frozenset([i]) for i in range(11)}
    for k, (a, b, h, _) in enumerate(Z):
        sm[11 + k] = sm[int(a)] | sm[int(b)]
        scipy_sets.append((sm[11 + k], h))
    assert [s for s, _ in ours_sets] == [s for s, _ in scipy_sets]
    np.testing.assert_allclose([h for _, h in ours_sets], [h for _, h in scipy_sets])


def test_hac_hand_computed_complete_linkage_on_paper_example():
    D = similarity_matrix(M_PAPER)  # a=0 b=1 c=2 d=3
    m = linkage_merges(D, "complete")
    assert m[0][:2] == (0, 2) and m[0][2] == pytest.approx(
        0.4857142857
    )  # tie (a,c)/(b,c): lowest ids win
    assert (
        m[1][:2] == (1, 3) and m[1][2] == 1.0 and m[2][:2] == (4, 5) and m[2][2] == 1.0
    )
    avg = linkage_merges(
        D, "average"
    )  # b joins {a,c} at mean(d(a,b), d(c,b)) = (1 + 0.4857) / 2; d joins at 1
    assert avg[1][:2] == (1, 4) and avg[1][2] == pytest.approx((1 + 0.4857142857) / 2)
    assert avg[2][:2] == (3, 5) and avg[2][2] == pytest.approx(1.0)
    single = linkage_merges(
        D, "single"
    )  # single linkage chains b onto {a,c} at min(1, 0.4857)
    assert single[1][:2] == (1, 4) and single[1][2] == pytest.approx(0.4857142857)


def test_invalid_hac_inputs_rejected():
    with pytest.raises(ValueError):
        linkage_merges(random_distance(4, 0), "ward")
    D = random_distance(4, 0)
    D[0, 1] += 0.5
    with pytest.raises(ValueError):
        build_hierarchy(D)  # asymmetric
    for bad in (
        np.ones((3, 2)),
        np.zeros((1, 1)),
        -np.ones((3, 3)),
        np.full((3, 3), np.inf),
    ):
        with pytest.raises(ValueError):
            build_hierarchy(bad)


# ------------------------------------------------------------------ structure on planted confusions
@pytest.mark.parametrize("linkage", ["complete", "single", "average"])
@pytest.mark.parametrize("seed", range(5))
def test_similar_classes_are_grouped_as_planted(linkage, seed):
    res = build_hierarchy(
        similarity_matrix(planted_confusion(seed)), linkage, alpha=0.0
    )
    planted = tree_from(PLANTED, 8)
    assert res.tree.groups() == planted.groups()
    assert res.tree.children[res.tree.root] == planted.children[planted.root]


def test_two_blocks_split_at_the_root():
    n = 6
    P = np.full((n, n), 0.002)
    np.fill_diagonal(P, 0.0)
    for blk in ((0, 1, 2), (3, 4, 5)):
        for i in blk:
            for j in blk:
                if i != j:
                    P[i, j] = 0.15
    P += np.diag(1 - P.sum(1))
    t = build_hierarchy(similarity_matrix(P), "average").tree
    assert {t.classes_under(c) for c in t.children[t.root]} == {(0, 1, 2), (3, 4, 5)}


def test_two_classes_give_one_node():
    r = build_hierarchy(similarity_matrix(np.array([[8.0, 2.0], [3.0, 7.0]])))
    assert r.tree.children == {2: (0, 1)} and r.tree.root == 2


@pytest.mark.parametrize("linkage", ["complete", "single", "average"])
def test_every_class_once_no_cycles_binary_when_tie_free(linkage):
    for seed in range(10):
        n = 3 + seed
        r = build_hierarchy(random_distance(n, seed), linkage, 0.0)
        r.tree.validate()
        assert len(r.tree.children) == n - 1 and all(
            len(c) == 2 for c in r.tree.children.values()
        )
        assert sorted(c for c in r.tree.classes_under(r.tree.root)) == list(range(n))
        assert [x for x in range(n)] == [
            x for x in range(n) if r.tree.path(x)[-1] == x
        ]  # leaf id == class id


def test_heights_monotone_and_stable_mapping_across_rebuilds():
    D = random_distance(9, 3)
    a, b = build_hierarchy(D, "average"), build_hierarchy(D.copy(), "average")
    assert a.tree == b.tree and a.merges == b.merges  # deterministic
    a.tree.validate()


def test_equivariance_under_class_relabelling_when_tie_free():
    D = random_distance(9, 4)
    perm = np.random.default_rng(1).permutation(9)  # new class perm[i] is old class i
    Dp = np.zeros_like(D)
    Dp[np.ix_(perm, perm)] = D
    g = build_hierarchy(D, "complete").tree.groups()
    gp = build_hierarchy(Dp, "complete").tree.groups()

    def mapped(x):
        return int(perm[x]) if isinstance(x, int) else frozenset(mapped(c) for c in x)

    assert mapped(g) == gp


# ------------------------------------------------------------------ ties and degenerate confusion
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("linkage", ["complete", "single", "average"])
def test_perfect_classifier_gives_flat_star(n, linkage):
    """Identity confusion -> d_S = 1 for all pairs -> every merge at one height -> alpha=0 compression -> star."""
    r = build_hierarchy(
        similarity_matrix(np.diag(np.arange(1, n + 1) * 10.0)), linkage, 0.0
    )
    assert r.tree.children == {n: tuple(range(n))} and r.classifiers_saved == (n - 2)
    assert (
        len(r.uncompressed.children) == n - 1
    )  # the uncompressed HAC tree is still binary


@pytest.mark.parametrize("linkage", ["complete", "average"])
def test_uniform_confusion_gives_flat_star(linkage):
    r = build_hierarchy(similarity_matrix(np.ones((4, 4))), linkage, 0.0)
    assert r.tree.children == {4: (0, 1, 2, 3)}


def test_single_confused_pair_is_grouped_rest_flat():
    M = np.diag([50.0] * 5)
    M[0, 1], M[0, 0] = 10, 40
    M[1, 0], M[1, 1] = 10, 40
    t = build_hierarchy(similarity_matrix(M), "complete").tree
    assert t.groups() == frozenset([frozenset([0, 1]), 2, 3, 4])


def test_ties_resolve_deterministically_to_lowest_ids_and_this_is_label_dependent():
    D = similarity_matrix(M_PAPER)
    first = build_hierarchy(D, "complete").tree
    assert all(build_hierarchy(D, "complete").tree == first for _ in range(5))
    assert first.classes_under(first.children[first.root][0]) == (
        0,
        2,
    )  # tie (a,c)|(b,c) -> {a,c}
    sw = [1, 0, 2, 3]  # relabel: new class 0 = old b, new class 1 = old a
    swapped = build_hierarchy(D[np.ix_(sw, sw)], "complete").tree
    new_pair = swapped.classes_under(swapped.children[swapped.root][0])
    assert new_pair == (0, 2)  # the lowest ids win again...
    assert sorted(sw[c] for c in new_pair) == [
        1,
        2,
    ]  # ...which is the ORIGINAL pair {b, c}: tie outcome is label-dependent


def test_alpha_zero_result_is_independent_of_how_equal_top_level_merges_are_ordered():
    """One genuinely close pair; every other pair equally far. Whatever the labels, tie-breaking only reorders
    equal-height merges, and alpha=0 compression must collapse them to the same grouping."""
    import itertools

    D = np.ones((5, 5))
    np.fill_diagonal(D, 0)
    D[0, 1] = D[1, 0] = 0.2
    expected = frozenset([frozenset([0, 1]), 2, 3, 4])
    raw_groupings = set()
    for order in itertools.permutations(range(5)):  # new class i is old class order[i]
        order = list(order)
        res = build_hierarchy(D[np.ix_(order, order)], "complete")

        def back(x, order=order):
            if isinstance(x, int):
                return order[x]
            return frozenset(back(c, order) for c in x)

        assert back(res.tree.groups()) == expected
        raw_groupings.add(back(res.uncompressed.groups()))
    assert (
        len(raw_groupings) > 1
    )  # the uncompressed binary trees really did differ with the tie order


# ------------------------------------------------------------------ compression (paper Algorithm 1)
def chain_tree():
    # ((((0,1)@.1, 2)@.1, 3)@.5, 4)@.9 : heights sorted .1 .1 .5 .9 -> consecutive gaps 0, .4, .4
    kids = {"A": (0, 1), "B": ("A", 2), "C": ("B", 3), "R": ("C", 4)}
    return ClassTree.from_edges(5, "R", kids, {"A": 0.1, "B": 0.1, "C": 0.5, "R": 0.9})


def test_compression_theta_matches_the_paper_formula():
    t = chain_tree()
    for alpha in (0.0, 0.25, 0.5, 1.0):
        _, theta = compress(t, alpha)
        assert theta == pytest.approx(alpha * 0.4)
    _, theta = compress(tree_from((0, 1), 2), 0.7)
    assert theta == 0.0  # a single merge has no consecutive gap
    with pytest.raises(ValueError):
        compress(t, 1.5)


def test_alpha_zero_merges_only_equal_height_clusters():
    c, _ = compress(chain_tree(), 0.0)
    assert c.groups() == frozenset(
        [frozenset([frozenset([0, 1, 2]), 3]), 4]
    )  # A (h=.1) absorbed into B (h=.1)
    assert [len(c.children[n]) for n in c.internal_nodes] == [2, 2, 3]
    assert compress(chain_tree(), 0.0)[0] == c  # deterministic


def test_algorithm_1_literal_semantics_at_larger_alpha():
    c, theta = compress(
        chain_tree(), 1.0
    )  # theta = 0.4: root (0.9) absorbs C (gap 0.4) but not B (gap 0.8)
    assert theta == pytest.approx(0.4)
    assert c.classes_under(c.root) == (0, 1, 2, 3, 4)
    assert sorted(len(v) for v in c.children.values()) == [
        3,
        3,
    ]  # root(4, 3, B), B(0, 1, 2)
    # NOTE: the paper text says alpha = 1 collapses everything; literal Algorithm 1 does not (documented in the note)
    assert len(c.children) == 2


def test_compression_never_loses_or_duplicates_classes():
    for seed in range(15):
        D = random_distance(10, seed)
        D = np.round(D, 1)  # many ties
        np.fill_diagonal(D, 0)
        for alpha in (0.0, 0.3, 1.0):
            t = build_hierarchy(D, "complete", alpha).tree
            t.validate()
            assert sorted(t.classes_under(t.root)) == list(range(10))
            assert all(len(c) >= 2 for c in t.children.values())


def test_compression_reduces_internal_nodes_monotonically_in_alpha():
    D = np.round(random_distance(12, 2), 1)
    np.fill_diagonal(D, 0)
    sizes = [
        len(build_hierarchy(D, "average", a).tree.children)
        for a in (0.0, 0.2, 0.5, 1.0)
    ]
    assert sizes == sorted(sizes, reverse=True) and sizes[0] <= 11


# ------------------------------------------------------------------ end to end from predictions
def test_hierarchy_from_flat_predictions_on_synthetic_labels():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(6), 200)
    pred = y.copy()
    flip = rng.random(len(y)) < 0.25
    partner = y ^ 1  # classes (0,1) (2,3) (4,5) are the confusable pairs
    pred[flip] = partner[flip]
    t = build_hierarchy(
        similarity_matrix(confusion_from_predictions(pred, y, 6)), "complete"
    ).tree
    assert {
        frozenset(t.classes_under(n))
        for n in t.internal_nodes
        if len(t.classes_under(n)) == 2
    } == {frozenset(p) for p in [(0, 1), (2, 3), (4, 5)]}
