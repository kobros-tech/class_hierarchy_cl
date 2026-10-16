# Technical note: Silva-Palacios, Ferri, Ramírez-Quintana (2018), "Probabilistic Class Hierarchies for Multiclass Classification"

Source read in full: the uploaded accepted manuscript (JOCS 824, 24 pp., preprint of 12 Feb 2018). Locations are "p. = manuscript page of 24, l. = margin line number". The manuscript's equations are unnumbered.
The second upload (Hihn & Braun, 2023) is the excluded second paper; nothing from it was used.

## 1. Items, equations, locations, implementation

| Item | Paper definition (their notation) | Location | Implementation | Status |
|---|---|---|---|---|
| Confusion matrix | M, n×n, from a flat classifier F trained on D; rows actual class, columns predicted. M_ij = count of class-i instances predicted j | p.6 l.138-145 | `confusion_from_predictions` | done, tested |
| Row normalisation | M̄_ij = M_ij / Σ_k M_ik | p.6 l.155 | `row_normalize` | done; zero-sum rows: explicit error (paper silent) |
| Overlap (Def. 1) | overlap(i,j) = (M̄_ij + M̄_ji)/2 for i≠j; 1 for i=j | p.6 l.157-160 | `overlap_matrix` | done |
| Similarity (Def. 2) | d_S(i,j) = 1 − overlap(i,j); symmetric, in [0,1], zero diagonal | p.6 l.161-p.7 l.165 | `similarity_matrix` | done; reproduces Table 1b, 1c, 1d |
| Semi-metric | satisfies non-negativity, symmetry, identity; triangle inequality can fail (d_S(a,b)=1 > d_S(a,c)+d_S(c,b)=0.972) | p.7 l.166-177 | tested on the paper's Example 1 | done |
| Probability variant | M_ij = Σ_{⟨x,i⟩∈D} Π(j\|x); then same procedure as Sec. 3.1 (called d_SP in experiments) | p.8 l.186-200, p.15 l.322-324 | `confusion_from_probabilities` (rejects non-probability rows) | done; reproduces Table 2 |
| HAC | agglomerative from singletons; linkages complete (max), single (min), average (mean of pairwise distances); dendrogram → hierarchy: leaves = classes, merges = internal nodes | p.8 l.201-p.9 l.227 | `linkage_merges`, `build_hierarchy` | done; matches scipy on tie-free inputs |
| Node classifiers | one classifier per internal node to distinguish its child nodes (LCPN); binary tree needs n−1 | p.9 l.228-p.10 l.239; p.12 l.269-271 ("binary or multiclass") | not yet: label mapping only (`class_to_child`, `node_targets`) | mapping done, training not started |
| Compression | θ = α · max_k \|linkage(k) − linkage(k+1)\|, α∈[0,1]; Algorithm 1: depth-first, node absorbs a child when height(node) − height(child) ≤ θ; α=0 joins only clusters at the same height | p.10 l.235-p.12 l.267 (Alg. 1 on p.11) | `compress` | done; α=0 is the primary setting |
| Top-down rule | from the root apply the node classifiers down to a leaf | p.12 l.268-276 | not yet | next |
| Probabilistic rule | leaf c_i: vector with Π(c_i\|x)=1, others 0. Non-leaf ν: p(x,ν) = Σ_{μ∈Chl(ν)} Π(μ\|x) × p(x,μ). Prediction = highest component of p(x,root). Applies ALL node classifiers | p.12 l.277-p.13 l.292 | not yet; implement the recursion literally and test it equals the closed form P(c\|x) = product of the Π values along the root-to-c path | next, after sign-off |

## 2. What the probabilistic equation requires
- Each node classifier must be a probability estimator: for x, a non-negative vector over that node's children summing to 1. Nothing else is stated.
- I found no requirement for calibration in Sec. 4.3. Calibration appears only in the introduction as a general technique (p.3 l.36-38, refs 11-13), and the experiments (p.15) describe no calibration step.
- So raw sigmoid or softmax outputs are legitimate under the paper's equation. Calibration would be a separately labelled variant, not the default.
- Caveat for `main`: its default `pos_weight="balanced"` reweights class priors, so a sigmoid output is then a probability under rebalanced priors. This must be decided (and reported) when node models are trained.
- The probabilistic method evaluates every node for every input; top-down evaluates one root-to-leaf path.

## 3. Unspecified or inconsistent in the paper (decision taken, recorded in every result)
1. **Def. 2 prose vs formula.** p.7 l.163-165 says d_S = 1 means "completely overlapping"; the formula and Table 1d give d_S = 1 for classes never confused. Implemented the formula (verified against Table 1d).
2. **"<" vs "≤".** Text uses "<", Algorithm 1 uses "<=". α = 0 is described as joining equal-height clusters, which needs "≤". Used "≤" with tolerance 1e-12.
3. **Compression for α>0.** Text: contiguous clusters in creation order; Algorithm 1: parent/child heights. They differ, and Algorithm 1 literally does not always collapse everything at α=1, contradicting p.12 (shown in a test). Algorithm 1 is implemented literally; α>0 is not used in primary experiments.
4. **θ indexing** ("n clusters", k up to n−1) is ambiguous. Used consecutive gaps of the sorted merge heights.
5. **Fig. 2** node labels do not match Fig. 1 or the text, so it is not used as a golden test.
6. **Top-down cost** (p.12 l.274-276, "n/2 for balanced") does not match binary-tree depth. Resource reporting counts actual node decisions per prediction.
7. **Tie-breaking in HAC, child order, empty confusion rows:** not specified. Ties go to the smallest (id, id) pair within 1e-12. Consequence, stated precisely:
   - *Ties among merges that do not compete for the same members* (e.g. several equally distant classes above one close pair) are collapsed by α=0 compression; the result is the same for all 120 label permutations in the test.
   - *Ties between competing partners* (Table 1: c is equally close to a and to b) are NOT resolved by compression: which pair merges first depends on class ids, so the grouping ({a,c} vs {b,c}) is label-dependent. This is tested and documented, and it is a real limitation of applying the paper to confusions with exact ties.
8. **Tolerance scale:** heights are compared with an absolute tolerance of 1e-12, appropriate for d_S in [0,1]; other distance scales would need a relative tolerance.
9. **Linkage:** complete in Sec. 5.1 (p.14), average in Sec. 5.2 (p.17). Default complete; linkage is a recorded configuration.
10. **Which instances train each node classifier** (only classes below the node, or all) is not stated in what I read. LCPN (their cited basis) trains on the node's own classes; to be fixed at implementation and recorded.
11. **Which data produce M.** The conclusion says "obtained from training data" (p.19 l.419-420). Resubstitution confusion on a well-fitted flat model can be near-diagonal, which gives a degenerate star. For continual learning my default is the validation split (no test data, no future data); this is a deviation from the paper, to be labelled.

## 4. Consequences for this project
- **Nodes are n-ary in general.** With α = 0, equal-height merges collapse, and sparse confusions create many equal heights. The paper's own Table 1 example compresses to root({a,c}, b, d). A perfect classifier gives a star, i.e. the flat classifier (tested for n = 2..7, all three linkages).
- **Realistic sparse confusions give very flat hierarchies.** In a quick simulation (20 classes, diagonal-dominant confusion with 25 random off-diagonal errors, complete linkage, α=0, 40 draws) the tree had 8.5 internal nodes on average (a binary tree would have 19) and maximum arity ≈ 12.5. Expect the confusion-derived hierarchy to be a shallow grouping of a few clusters, much closer to the flat classifier than a binary tree; the shape-preserving random control inherits this shape, so the comparison stays fair, but the effect size to detect may be small. (Simulation, not part of the test suite.)
- So a node is a k-way decision, not always a group-vs-group binary one. `main`'s multi-output multiclass mode (num_outputs = k, softmax, cross-entropy) fits this directly and yields exactly the normalised vector the probabilistic equation needs, with no sigmoid reinterpretation. A single-logit BCE node is only needed for binary nodes. Decision requested (see message).
- The paper is a batch, offline method. Everything about continual updates is a new design decision, not paper-established.
- Reported results (15 small UCI/LIBSVM datasets, 10-fold CV, R classifiers): hierarchies mostly beat the flat classifier except for NB and SVM; d_S and d_SP similar; α = 0 mostly better than 0.1 except NB (in Table 6, J48 averages 0.806 at α=0 vs 0.808 at α=0.1, so the text's "almost all" is a slight overstatement). These are not evidence about continual learning or frozen random features.

## 5. Test results (CPU, exact)
`pytest`: **137 passed** (130 before the review below). By file: hierarchy_construct 71, hierarchy_tree 16, hierarchy_similarity 8, protocol_memory 8, flat_baseline 8, features 8, m2_gate 11, metrics 3, end_to_end_flat 2, provenance 2.
Hierarchy tests cover: the paper's Tables 1 and 2 reproduced value by value; Example 1's triangle violation; properties on 50 random confusions; HAC equal to scipy for 3 linkages × 6 seeds; hand-computed linkage heights on the paper example; planted 8-class tree recovered for 3 linkages × 5 noisy seeds; two-block split; every class exactly once, leaf id = class id, no cycles, binary and n−1 internal nodes when tie-free; invalid trees rejected (cycle, duplicate leaf, missing class, unary node, unreachable node, decreasing heights, missing heights); determinism; equivariance under relabelling; flat star for perfect (n = 2..7, 3 linkages) and uniform confusion; α=0 grouping independent of equal-height ties over all 120 label permutations; compression keeps every class, θ formula, Algorithm 1 literal behaviour; node-label mapping partitions and stability; shape-preserving random control (same shape, heights, arities; 400 seeds recover the planted pairs < 5%, chance ≈ 1%).
Mutation checks (each deliberately breaks one paper definition; all caught): asymmetric overlap 23 failures; complete→single linkage 9; strict "<" in compression 14; last-instead-of-first tie-break 2.

## 6. Not done
Node training, top-down and probabilistic prediction, continual update policy, random-control training runs, real-benchmark runs (data unavailable in the sandbox). No accuracy or forgetting claim is made.

## 7. Review log (after first delivery)
- **Defect fixed:** `ClassTree.node_targets` silently mapped a label of -1 to the last class (negative index) and raised an uninformative IndexError for too-large labels. It now raises a ValueError for any label outside [0, C); test added.
- **Overclaims in this note corrected:** (a) tie-independence claim (item 7 above, now scoped); (b) "n = 2..7" was tested only for {2, 3, 5, 7}; tests now cover all of 2..7.
- **Tests made stronger:** five assertions with `or` clauses that could never fail were replaced by exact expected values; one test premise (raw tree shapes) was wrong and now compares groupings.
- **Checked, no problem found:** 2,700 fuzz configurations (300 tie-heavy random distance matrices, n up to 40, 3 linkages × α∈{0, 0.2, 1}): every tree valid, every class once, deterministic. HAC time: 0.85 s (complete) / 1.15 s (average) at n = 100 classes, so rebuilding per experience is affordable.
- **Known limits, not fixed:** Algorithm 1 for α>0 is implemented literally and disagrees with the paper's own α=1 description; ties between competing partners are label-dependent; HAC tolerance is absolute; the tie-break rule is a design decision, not the paper's.
