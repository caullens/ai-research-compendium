# Boolean Universe — Architecture BFS Checkpoint 8

This checkpoint closes two remaining breadth axes before scaling beyond four nodes: heterogeneous intrinsic node clocks and exact architecture-space growth with node count.

## 1. Heterogeneous local clock rates

Four-node affine local law held fixed in form; only update-rate profile changes. A node with rate `r` updates every `r` ticks. One macro-step is the least-common-multiple of all four local periods.

6,000 random affine laws sampled per rate profile.

| Rates | Macro ticks | Reversible | Period >1 | Period >=4 | Max observed |
|---|---:|---:|---:|---:|---:|
| (1,1,1,1) | 1 | ~0.3098 | ~0.8713 | ~0.5855 | 15 |
| (1,1,1,2) | 2 | ~0.1668 | ~0.6983 | ~0.2725 | 15 |
| (1,1,2,2) | 2 | ~0.1387 | ~0.7233 | ~0.2850 | 15 |
| (1,2,2,2) | 2 | ~0.1518 | ~0.7790 | ~0.3703 | 7 |
| (1,1,2,3) | 6 | ~0.0610 | ~0.4787 | ~0.1148 | 15 |
| (1,2,3,4) | 12 | ~0.0352 | ~0.3338 | ~0.0365 | 15 |
| (1,2,3,5) | 30 | ~0.0257 | ~0.3040 | ~0.0327 | 15 |

Clock heterogeneity strongly suppresses reversibility and often suppresses oscillation, but profiles with the same macro-period can behave differently. For example all three LCM-2 profiles have different oscillatory and reversible fractions.

**Conclusion:** local update-rate heterogeneity is an architecture axis distinct from global scheduler period or update ordering.

## 2. Exact unlabeled topology scaling by Burnside's lemma

Exact counts of directed Boolean graph topologies with self-loops, quotienting simultaneous node relabeling:

| n | Unlabeled directed+loops | Unlabeled reciprocal+loops |
|---:|---:|---:|
| 1 | 2 | 2 |
| 2 | 10 | 6 |
| 3 | 104 | 20 |
| 4 | 3,044 | 90 |
| 5 | 291,968 | 544 |
| 6 | 96,928,992 | 5,096 |
| 7 | 112,282,908,928 | 79,264 |
| 8 | 458,297,100,061,728 | 2,208,612 |

The corresponding labeled directed mask count is `2^(n^2)` and the affine global-law count is `2^(n^2+n)`.

At only five Boolean nodes there are already 291,968 distinct unlabeled directed topologies with self-loops. At six nodes the number is nearly 97 million.

**Conclusion:** exhaustive breadth-first enumeration is practical through four nodes but becomes the wrong instrument immediately afterward for unrestricted directed architecture. Beyond n=4, the program should use theorem-guided family sweeps, symmetry classes, constructive extremal examples, and controlled sampling rather than pretending to enumerate the full architecture lattice.

## 3. Methodological boundary

The Boolean program now has a natural phase transition in research method:

- `n <= 4`: broad exact census is feasible for many architecture axes;
- `n >= 5`: system-size scaling itself becomes a combinatorial phenomenon, and structured subfamilies must replace naive enumeration.

This is not merely a computational inconvenience. It becomes part of the research question: which architecture coordinates remain predictive under compression of an enormous design space?
