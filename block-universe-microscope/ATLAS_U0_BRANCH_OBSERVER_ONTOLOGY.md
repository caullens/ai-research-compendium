# Atlas U₀ — Branch: observer-dependent ontology

This branch keeps U₀ completely fixed and varies only what an observer is permitted to sense. The observer receives a linear projection of the full 8D local state into `k` channels.

The goal is not to claim linear sensing is uniquely natural, but to quantify how much of U₀'s apparent ontology can depend on observation capacity alone.

## 1. Local observable dimension

For a random orthogonal observer projection `P_k : R^8 -> R^k`, the observed Jacobian is

`J_obs(x) = P_k J(x)`.

Across 7000 sampled points the median and mean observed ranks are exactly:

- k=1 -> rank 1
- k=2 -> rank 2
- k=3 -> rank 3
- k>=4 -> rank 4

Thus observers with fewer than four channels lose continuous local degrees of freedom:

- k=1 hides 3 local dimensions
- k=2 hides 2
- k=3 hides 1
- k>=4 is locally four-dimensional almost everywhere in the sampled region

This is an observer effect, not a change to the universe.

## 2. Global ambiguity remains after local identifiability

For k=4,5,6,7, exact nontrivial observational collisions were found numerically:

`P_k F(a) = P_k F(b)` with `a != b`.

The search was deliberately capped after 12 distinct roots at each k and hit that cap for every k=4..7, so these collisions are abundant rather than exceptional.

Representative source separations were approximately:

- k=4: 2.89
- k=5: 1.40
- k=6: 1.34
- k=7: 1.96

with projected equality residuals around 1e-11 or better.

So an observer can be locally fully informed (rank 4) and still have a globally many-to-one world.

## 3. Generic dimension of the observer's self-identification locus

For two source points `(a,b)` in `R^4 x R^4`, observational equality imposes k scalar equations:

`P_k F(a) = P_k F(b)`.

When these constraints are transverse, the generic nontrivial pair-equality locus has dimension

`8 - k`.

This yields the observer phase diagram:

| k | local observed rank | hidden local dimension | generic distant pair-equality locus |
|---:|---:|---:|---:|
| 1 | 1 | 3 | 7D |
| 2 | 2 | 2 | 6D |
| 3 | 3 | 1 | 5D |
| 4 | 4 | 0 | 4D |
| 5 | 4 | 0 | 3D |
| 6 | 4 | 0 | 2D |
| 7 | 4 | 0 | 1D |
| 8 | 4 | 0 | 0D |

The full 8D observer therefore sees isolated double-point defects, while a 7D observer generically sees one-dimensional families of observational self-identifications, a 6D observer two-dimensional families, and so on.

For k<4, the stronger phenomenon is local: each observed state already corresponds to a continuous `(4-k)`-dimensional family of nearby source points.

## 4. Nearest distant-state ambiguity

On a common 7000-point source sample, the median distance in observed state space to the nearest source-distant state grows sharply with k:

- k=1: ~6.8e-5
- k=2: ~6.4e-3
- k=3: ~4.1e-2
- k=4: ~9.4e-2
- k=5: ~0.159
- k=6: ~0.190
- k=7: ~0.201
- k=8: ~0.215

The numerical distances are projection/metric dependent; the robust point is the qualitative loss of distinguishability as channels are removed.

## Interpretation

The same frozen U₀ does not induce one unique effective ontology for every observer.

A low-channel observer can inhabit a world with continuous hidden fibers; a k=4 observer can recover four local degrees of freedom but still experience extended global self-overlap; a k=7 observer can see defect strings/curves; the full k=8 observer sees isolated double-point defects.

This is not merely epistemic uncertainty in the everyday sense. The quotient space available to the observer is genuinely different because the observation map changes which source events are operationally identical.

The key distinction is:

- **ontic U₀:** the fixed full map `F : R^4 -> R^8`
- **observer-effective world:** the quotient induced by `P_k ∘ F`

## Next targets

1. Parameterize an actual k=7 observational double locus and visualize how an isolated full-state defect expands into a curve under one lost state dimension.
2. Track the same underlying source region across k=8 -> 7 -> 6 -> 5 -> 4 to build a literal ontology-bifurcation movie.
3. Replace random linear projections with learned/optimized sensory maps and ask what observation channels best preserve topology or distinguish defects.
4. Test whether two observers with equal channel count but different projections infer inequivalent effective topology.
5. Investigate observer fusion: how many independent partial observers are required before their joint observations reconstruct the full-state quotient?
6. Later connect this to the ontology learner: can it infer hidden dimensions or recognize that apparent 'objects' are projection-induced self-identifications?
