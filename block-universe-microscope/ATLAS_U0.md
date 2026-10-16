# Atlas U₀ — Research Ledger

Universe: deterministic frozen neural block used by `index.html`, mapping **R⁴ → R⁸**, no training and no update rule.

This file distinguishes **microscope observations** from **ontology candidates**. Anything representation-dependent stays provisional.

## Equivalence standard

The strict ontology audit treats smooth invertible re-descriptions of both source and target as gauge. In singularity-theory language this is right-left / A-equivalence:

`g = ψ ∘ f ∘ φ^{-1}`

where `φ` and `ψ` are diffeomorphisms of source and target.

A candidate intrinsic feature should therefore survive this equivalence. Metric quantities, PCA axes, coordinate directions, output components, Euclidean distances, and polynomial simplicity remain microscope diagnostics only unless extra structure is explicitly promoted to the universe.

## Checkpoint 1 — broad atlas pass

### Local differential structure

Sampled 12,000 points uniformly from `[-3,3]^4`.

- Every sampled Jacobian had numerical rank 4 (`σ₄ > 1e-4`).
- Median singular-spectrum participation ratio: **2.853**.
- Median singular-value ratios relative to `σ₁` were approximately:
  - `σ₂/σ₁ = 0.618`
  - `σ₃/σ₁ = 0.352`
  - `σ₄/σ₁ = 0.117`
- Median condition number: **8.56**.

Interpretation: U₀ is locally four-dimensional throughout the sampled region, but strongly anisotropic. One local direction is usually much less influential than the strongest direction.

**Invariance status:**
- Jacobian **rank** is invariant under smooth invertible input/output reparameterizations → strong ontology candidate.
- Singular-value magnitudes, condition number, and participation ratio are invariant under orthogonal changes of chart/basis but not arbitrary diffeomorphisms → microscope geometry, not yet ontology.

### Global state geometry

PCA of the 8D state cloud from the same 12,000 points:

- PC1: 37.86%
- PCs 1–2: 68.60%
- PCs 1–3: 88.15%
- PCs 1–4: **97.70%**

Interpretation: the 8D output states lie close to a curved four-dimensional structure globally, independently echoing the source dimension.

**Invariance status:** PCA variance is output-coordinate dependent under general nonlinear reparameterization. Treat as a useful external diagnostic only.

### Effective-dynamics direction search

For a direction `v` through the 4D block, measured the directional state derivative `J(x)v`. Asked how compactly that derivative can be predicted from the local state alone.

A linear state law performed poorly: best held-out `R² ≈ 0.046`.

A quadratic state law performed substantially better:

- best direction ≈ `[0.787, 0.460, -0.412, -0.008]`
- best held-out `R² ≈ 0.786`
- median random-direction `R² ≈ 0.683`

So there is a modestly privileged direction for a low-order quadratic description, but it is not overwhelmingly separated from other directions.

**Invariance status:** polynomial simplicity depends strongly on the chosen state coordinates, so this is **not yet an intrinsic time/dynamics discovery**. More fundamentally, on any regular injective patch, a chosen source vector field can be pushed forward into state space; the mere existence of a local evolution law cannot by itself single out physical time.

## Checkpoint 2 — intrinsic fiber / topology pass

### Exact transverse double point discovered

The earlier near-collision was followed beyond the original `[-3,3]^4` search box with a deflated pair-root solve that prevents the trivial `x=y` solution. This converged to two distinct coordinates with the same state to floating-point precision:

- preimage A ≈ `[2.70476851, 0.12269473, -1.90900979, -3.01139004]`
- preimage B ≈ `[2.23181775, -0.39195596, -0.59623724, 0.35221822]`
- external-coordinate separation ≈ **3.67774**
- refined state residual `||F(A)-F(B)|| ≈ 2.5e-15`

Both local Jacobians have rank 4. The 8×8 combined derivative

`[ J(A)  -J(B) ]`

has rank 8; its smallest singular value is about **0.01624**. Therefore the two local 4D image sheets meet **transversely** at the shared state.

This is qualitatively stronger than an approximate motif. The statement

`A != B but F(A) = F(B)`

survives arbitrary diffeomorphic re-description of both source and target, as does the transversality type of the intersection.

**Status: strong intrinsic ontology candidate.**

### Multiple-point locus appears isolated

Aggressive follow-up searches were run using:

- output-space nearest-neighbor seeds across 50,000 source samples,
- differential-evolution pair searches,
- hundreds of random pair initializations,
- deflated least-squares root finding designed to avoid collapse onto the diagonal `x=y`.

So far, nontrivial converged roots either return to this same double point or fail to produce another exact fiber. No extended family of double fibers has appeared.

This is consistent with the generic dimension count for a self-transverse immersion `R^4 → R^8`: the double-point locus is expected to be zero-dimensional.

**Status: provisional isolated defect; not a global exhaustive proof.**

### Topological consequence: an intrinsic closed loop

Because A and B are distinct source points with the same image state, every source path from A to B maps to a closed path in state space. A straight source path already produces a visibly nontrivial closed state-space loop.

If one can isolate a contractible source region containing A and B with no additional fiber identifications, then the image quotient is obtained by identifying two points of a contractible region. Under that condition the quotient acquires a circle-like homotopy generator. Thus this single double fiber can create genuine nontrivial global topology in the state image, rather than merely a visually interesting crossing.

**Status: topological consequence is conditional on controlling other identifications in the chosen region.**

### Regular/singular strata

Large random surveys still find rank 4 everywhere sampled. Numerical minimization of the smallest Jacobian singular value finds highly compressed but nonzero values, including very small values farther into saturation regions; no finite exact rank-drop point has yet been established.

Under A-equivalence, differential rank and rank-drop strata are intrinsic. Singular-value magnitudes themselves are not.

**Status: evidence for a regular immersion with isolated multiple fibers; finite singular strata not yet found.**

## Updated structural picture

The prior hypothesis that U₀ might simply be an embedding is now false.

Current best description:

> **U₀ appears to be a smooth, locally rank-4 immersion of a 4D source into 8D state space with at least one isolated transverse double point.**

That double fiber is the first genuinely coordinate-resistant “feature” found in U₀. It is not a ridge, a low-norm region, or a visual motif; it is a structural identification that cannot be removed by smooth relabeling of source or state coordinates.

## Working ontology split

### Microscope only

- named coordinates `t/x/y/z`
- Euclidean input/output distances
- output components
- state norm
- PCA
- singular-value magnitudes
- polynomial-law simplicity
- visual ridges/valleys

### Strong candidates under A-equivalence

- source/target dimensions
- differential rank and rank-drop strata
- fiber cardinality
- multiple-point loci
- transversality type of multiple points
- topology of the image/quotient
- incidence relations among singular/multiple-point strata

## Prior / redescription ledger

| Observation | Orthogonal redescription | Arbitrary smooth invertible redescription | Status |
|---|---:|---:|---|
| Individual output component | No | No | representation artifact |
| State norm | Yes (output rotations) | No | microscope quantity |
| PCA spectrum | Yes (output rotations) | No | microscope quantity |
| Singular-value spectrum | Yes (orthogonal input/output) | No | microscope quantity |
| Jacobian rank | Yes | **Yes** | ontology candidate |
| Exact fiber multiplicity / self-intersection | Yes | **Yes** | **confirmed ontology candidate** |
| Transverse double-point type | Yes | **Yes** | **confirmed ontology candidate** |
| Connectedness/topology of state image | Yes | **Yes** | ontology candidate |
| Approximate state-nearness | Yes (orthogonal output) | No | provisional only |
| Polynomial dynamical simplicity | No in general | No | provisional only |

## Next atlas questions

1. **Double-point census:** broaden the exact fiber search and determine whether the discovered transverse double point is unique in increasingly large bounded source regions.
2. **Triple fibers:** search explicitly for states with three distinct preimages; generic dimension counting suggests these should be absent, which is itself testable.
3. **Rank-drop census:** continue searching for true finite singular strata rather than near-singular saturation.
4. **Quotient topology:** isolate a bounded region around the double fiber, certify its other fibers as well as possible, and compute the induced quotient topology.
5. **Intrinsic defect graph:** treat multiple fibers / singular strata as landmarks and ask what purely incidence-based relational structure can be built without metric coordinates.
6. **Symmetry audit:** distinguish genuine automorphisms of the map from arbitrary coordinate changes. A symmetry must preserve the map structure, not merely re-express it.
7. **Bare-ontology limit:** determine how much structure remains under full A-equivalence. If fiber topology is essentially all that survives, record that as the correct negative result rather than rescuing U₀ with extra geometry.
8. **Only afterward:** decide whether U₁ should add one explicit intrinsic law (for example action/energy) to support richer dynamics while keeping the ontology learner unchanged.

## Research rule

Do not make U₀ more interesting by tuning it after seeing these results. U₀ stays frozen. If its intrinsic ontology is sparse, that is the result.
