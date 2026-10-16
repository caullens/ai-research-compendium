# Boolean Universe — Architecture BFS Checkpoint 5

This checkpoint begins the four-node shell while correcting one bookkeeping issue from the prior affine architecture-equivalence analysis.

## 0. Correction: true minimum affine wiring cost

For a labeled affine Boolean universe

`x' = A x + b` over GF(2),

the true minimum directed dependency graph is simply the support of `A`: edge `j -> i` is required exactly when `A[i,j]=1`. The bias vector `b` requires no edges.

The earlier proxy used one representative per unlabeled topology while holding state labels fixed, which can overstate the minimum edge count for a particular labeled global law.

Corrected 3-node minimum-edge distribution over all 4,096 affine laws:

- 0 edges: 8
- 1: 72
- 2: 288
- 3: 672
- 4: 1008
- 5: 1008
- 6: 672
- 7: 288
- 8: 72
- 9: 8

Corrected reversible 3-node affine laws by minimum edge count:

- 3 edges: 48
- 4: 288
- 5: 576
- 6: 288
- 7: 144

The general implementation-degeneracy formula is simple: an affine law with `k` required edges can be hosted by exactly `2^(n^2-k)` labeled graph superstructures if nodes are allowed to ignore unused incoming edges.

Thus architecture remains highly non-identifiable even after correcting the cost measure.

## 1. Four-node affine law space

For four Boolean state bits,

`A` has 16 binary coefficients and `b` has 4, giving

`2^(16+4) = 1,048,576`

affine global laws.

Exactly `20,160` four-by-four matrices are invertible over GF(2), so including all 16 translations gives

`322,560`

reversible affine universes.

The true minimum wiring cost is binomial over the 16 possible dependencies:

- 0 edges: 16 laws
- 1: 256
- 2: 1,920
- 3: 8,960
- 4: 29,120
- 5: 69,888
- 6: 128,128
- 7: 183,040
- 8: 205,920
- 9: 183,040
- 10: 128,128
- 11: 69,888
- 12: 29,120
- 13: 8,960
- 14: 1,920
- 15: 256
- 16: 16

Mean number of labeled graph skeletons capable of hosting one uniformly chosen affine law grows rapidly:

- n=1: 1.5
- n=2: 5.0625
- n=3: ~38.44
- n=4: ~656.84

So implementation degeneracy grows extremely quickly with state dimension.

## 2. New monotone phenomenon: exact period 6

For monotone maps on the n-dimensional Boolean lattice, every nontrivial periodic orbit must be an antichain. Therefore the maximum possible period is bounded by the width of the Boolean lattice.

At n=3 the width is 3, and the previous exhaustive census found no period above 3.

At n=4 the middle layer has six mutually incomparable states:

`0011, 0101, 0110, 1001, 1010, 1100`.

A monotone four-node network was explicitly constructed whose global dynamics cycles through all six in order. Each coordinate rule was obtained by monotone upward extension from the prescribed labels on this antichain.

The resulting global cycle structure includes a genuine period-6 orbit.

Therefore the antichain bound is not merely theoretical here:

> **Four monotone Boolean neurons can realize an exact six-step clock, while three monotone neurons are capped at three.**

This is the first clear four-node dynamical capability that is impossible in the three-node monotone shell.

## 3. Reversible affine clocks jump from period 7 to period 15

All reversible affine maps were exhaustively censused for n=3 and n=4.

### n=3

- 168 invertible matrices
- 1,344 reversible affine maps including translations
- maximum period: 7

Longest-period distribution:

- 1: 1
- 2: 91
- 3: 224
- 4: 420
- 6: 224
- 7: 384

### n=4

- 20,160 invertible matrices
- 322,560 reversible affine maps including translations
- maximum period: 15

Longest-period distribution:

- 1: 1
- 2: 1,695
- 3: 6,272
- 4: 43,680
- 5: 21,504
- 6: 67,200
- 7: 46,080
- 8: 20,160
- 12: 26,880
- 14: 46,080
- 15: 43,008

A period-15 example has cycle partition `(1,15)`: one fixed state and one orbit visiting every other state.

Thus a fourth Boolean state bit changes the exact clock spectrum qualitatively, not just quantitatively.

## 4. Four-node reciprocal/self-loop topology shell

For four nodes:

- 6 possible reciprocal pair-links;
- 4 possible self-loops;
- 10 binary architecture decisions;
- `2^10 = 1024` labeled reciprocal/self-loop skeletons.

After arbitrary node relabeling these reduce to exactly

> **90 unlabeled reciprocal/self-loop topologies.**

For each topology, the sparsity mask of the affine matrix `A` was exhaustively tested for maximum achievable GF(2) rank.

Rank-capacity census:

- max rank 0: 1 topology
- max rank 1: 1
- max rank 2: 8
- max rank 3: 16
- max rank 4: 64

Therefore:

- 64 / 90 topologies can host at least one reversible affine law;
- 26 / 90 are structurally incapable of reversibility regardless of coefficient choices.

This is a topology-only law: before any local Boolean coefficients are selected, the wiring pattern can already forbid information-preserving dynamics.

Reciprocal connectivity strongly raises reversible capacity. Grouped by number of reciprocal pair-links, the fraction of unlabeled topologies capable of full rank is approximately:

- 0 links: 0.20
- 1: 0.33
- 2: 0.56
- 3: 0.69
- 4: 1.00
- 5: 1.00
- 6: 1.00

Four or more reciprocal pair-links are sufficient for every topology in this restricted reciprocal/self-loop shell to admit some reversible affine rule assignment.

Self-loops and reciprocal links trade off: four self-loops alone guarantee full-rank capability; without self-loops, sufficiently rich reciprocal matching structure is required.

## 5. What changed at four nodes

The first four-node shell reveals three kinds of scaling:

1. **Law-space scaling** — affine global laws jump from 4,096 to 1,048,576.
2. **Implementation degeneracy scaling** — the mean number of graph superstructures capable of hosting one affine law rises from ~38 to ~657.
3. **Capability thresholds** — new exact dynamical phenomena become possible:
   - monotone period 6;
   - reversible affine period 15;
   - far richer reciprocal topology rank structure.

The important distinction is therefore not simply `more nodes = more complexity`.

A new node can enlarge:

- state-space cardinality;
- possible dependency support;
- antichain width;
- algebraic group structure;
- maximum exact clock period;
- architectural degeneracy.

These are separable effects and should remain separate BFS axes.

## Next four-node breadth shell

Carry the same architecture dimensions forward rather than drilling into one 4-node family:

1. event-driven/asynchronous four-node dynamics;
2. observer port multiplicity and role flexibility;
3. monotone, threshold, affine, NAND-only and unrestricted local primitives;
4. fixed versus endogenous/stateful scheduler architectures;
5. layered DAGs versus recurrent graphs with the same node budget;
6. causal identifiability of internal wiring from interventions;
7. architecture equivalence under observer-limited access;
8. first search for properties that appear only at four nodes but survive multiple rule families.
