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

## 4. The same exact D0 defect inflates under reduced observation

The first full-state double point D0 was used as a fixed ontic landmark. Around the exact source pair `(a,b)`, the local observational equality manifold was solved numerically for the same nested observer family.

The result matches the generic dimension prediction exactly:

- k=8: one isolated exact pair
- k=7: exact 1D curve
- k=6: exact 2D surface
- k=5: exact 3D family
- k=4: exact 4D family

Hundreds of exact projected-equality samples were constructed at residuals around 1e-13 or better. The sample participation dimension follows `8-k` essentially exactly.

This is stronger than a generic counting argument. It shows the **same underlying ontic defect** continuously changes effective dimensionality solely because the observer loses state channels.

## 5. Observer-induced critical / caustic sets

Projection can create local singularities even though full U₀ remains a regular rank-4 immersion.

A projected critical state satisfies

`rank(P_k J(x)) < 4`.

Equivalently there is a nonzero source direction `v` such that

`P_k J(x) v = 0`.

For a generic `k x 4` projected Jacobian, the rank<=3 locus has codimension `k-3` in matrix space. Since the source itself is 4D, the expected dimension of observer-induced critical states is

`4 - (k-3) = 7-k`.

Exact numerical solves recover this full ladder:

- k=4: 3D caustic hypersurfaces
- k=5: 2D critical surfaces
- k=6: 1D critical curves
- k=7: isolated critical points
- k=8: generically absent

For the tested nested observer family, exact projected rank-drop points were found abundantly for k=4..7, with projected weakest singular values around machine precision while the full 8D U₀ Jacobian remained regular.

A 30k-point k=8 survey found no source-rank collapse in the sampled region.

Thus lower-dimensional observers acquire **new singular physics created by observation itself**.

## 6. Two independent observer-relative effects

The same fixed U₀ therefore supports two different observer-induced structures:

### Global ambiguity

`dim pair-equality locus = 8-k`

### Local caustics

`dim projected rank-drop locus = 7-k`

These differ by one dimension and arise from different mechanisms.

A k=7 observer is particularly informative:

- locally sees all four dimensions almost everywhere,
- has isolated caustic points where one local source direction becomes invisible,
- simultaneously sees 1D curves of distant pair-identifications.

So 'having enough channels to recover local dimension' does **not** imply recovering the global ontology.

## 7. Nearest distant-state ambiguity

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

A low-channel observer can inhabit a world with continuous hidden fibers; a k=4 observer can recover four local degrees of freedom but still experience extended global self-overlap; a k=7 observer sees defect curves plus isolated local caustics; the full k=8 observer sees isolated double-point defects and no generic projected caustics.

This is not merely epistemic uncertainty in the everyday sense. The quotient space available to the observer is genuinely different because the observation map changes which source events are operationally identical.

The key distinction is:

- **ontic U₀:** the fixed full map `F : R^4 -> R^8`
- **observer-effective world:** the quotient induced by `P_k o F`

The current branch therefore gives a precise computational example of

`ontic universe != observer-effective universe`.

Different observers of the exact same underlying reality can have different:

- apparent local dimension,
- fiber multiplicity,
- defect dimensionality,
- critical/caustic sets,
- global quotient topology.

## Next targets

1. Classify the local catastrophe type of the isolated k=7 caustics and generic folds at k=4..6.
2. Track one exact caustic continuously as the observer subspace is rotated or as k changes.
3. Compute observer-effective quotient homology/topology on bounded regions instead of only local dimension counts.
4. Ask whether an observer can infer hidden state dimensions from self-identifications and caustics alone.
5. Compare many random observers with equal channel count to separate dimension-count universality from projection-specific topology.
6. Test observer fusion: how many partial observers are required before their joint observation reconstructs the full-state quotient?
7. Build an interactive observer microscope where the observation subspace can be continuously rotated while apparent defects and caustics are shown changing in real time.
