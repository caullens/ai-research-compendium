# Atlas U₀ — Research Ledger

Universe: deterministic frozen neural block used by `index.html`, mapping **R⁴ → R⁸**, no training and no update rule.

This file distinguishes **microscope observations** from **ontology candidates**. Anything representation-dependent stays provisional.

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

### Distant repeated-state candidates / folds

A nearest-neighbor search in state space found pairs that are widely separated in the external input chart but unusually close in state.

Strong candidate found during the broad scan:

- external chart separation: **3.775**
- state-vector separation: **0.159**
- local singular-spectrum mismatch: **0.138**
- coordinate A ≈ `[2.671, 0.265, -2.148, -2.706]`
- coordinate B ≈ `[2.477, -0.685, -0.628, 0.611]`

For random far-separated pairs (`input distance > 3.5`), even the 0.01% quantile of state separation was about **0.377**, so these near-collisions are highly unusual in the chosen Euclidean state metric.

However, nonlinear invertible output reparameterizations can arbitrarily distort *near*-ness. Therefore approximate repeated-state motifs are **not ontology yet**.

### Exact self-intersection search

To seek a coordinate-independent version of the fold hypothesis, optimized pairs `(x,y)` with large chart separation to minimize `||F(x)-F(y)||`.

Best minima found so far:

- state residual ≈ **3.17e-4** at chart separation **3.66** (one coordinate at the sampled-domain boundary)
- interior residual ≈ **7.37e-4** at chart separation **2.45**

Follow-up least-squares optimization did **not** reach exact equality.

Additional inverse tests: 12 randomly selected target states were reconstructed from 12 random initial guesses each. Every successful inversion converged to one unique coordinate; no second preimage was found.

Current interpretation: no evidence yet for an exact self-intersection in the tested region. U₀ may be an embedding rather than a folded quotient.

**Invariance status:** exact equality / fiber multiplicity *would* survive arbitrary bijective output redescription and input diffeomorphism. This remains a high-priority intrinsic test.

### Effective-dynamics direction search

For a direction `v` through the 4D block, measured the directional state derivative `J(x)v`. Asked how compactly that derivative can be predicted from the local state alone.

A linear state law performed poorly: best held-out `R² ≈ 0.046`.

A quadratic state law performed substantially better:

- best direction ≈ `[0.787, 0.460, -0.412, -0.008]`
- best held-out `R² ≈ 0.786`
- median random-direction `R² ≈ 0.683`

So there is a modestly privileged direction for a low-order quadratic description, but it is not overwhelmingly separated from other directions.

**Invariance status:** polynomial simplicity depends strongly on the chosen state coordinates, so this is **not yet an intrinsic time/dynamics discovery**. It is evidence that some chart directions admit simpler descriptions in the current representation.

## Provisional structural picture

The strongest current hypothesis is:

> **Within the inspected region, U₀ behaves primarily like a smooth, locally full-rank, anisotropic 4D manifold embedded in an 8D state space.**

The visually rich slice patterns should therefore not be confused with intrinsic entities. Under a sufficiently permissive redescription, much of that apparent texture may disappear.

This is a useful negative result. A generic frozen MLP may provide an alien-looking field without automatically providing rich coordinate-independent ontology.

## Prior / redescription ledger

| Observation | Orthogonal redescription | Arbitrary smooth invertible redescription | Status |
|---|---:|---:|---|
| Individual output component | No | No | representation artifact |
| State norm | Yes (output rotations) | No | microscope quantity |
| PCA spectrum | Yes (output rotations) | No | microscope quantity |
| Singular-value spectrum | Yes (orthogonal input/output) | No | microscope quantity |
| Jacobian rank | Yes | **Yes** | ontology candidate |
| Exact fiber multiplicity / self-intersection | Yes | **Yes** | ontology candidate |
| Connectedness/topology of state image | Yes | **Yes** | ontology candidate |
| Approximate state-nearness | Yes (orthogonal output) | No | provisional only |
| Polynomial dynamical simplicity | No in general | No | provisional only |

## Next atlas questions

1. **Embedding test:** broaden exact self-intersection search, including continuation / root-finding and larger coordinate domains.
2. **Topology of the image:** look for coordinate-independent holes, nontrivial fibers, boundaries, or singular sets rather than Euclidean visual motifs.
3. **Intrinsic recurrence:** define recurrence via structural equivalence rather than Euclidean state distance.
4. **Dynamics without a chosen basis:** seek foliations or one-parameter transformation families characterized relationally, then test whether any supports a closed law.
5. **Symmetry census:** search for transformations of the block that induce consistent transformations of states, with the transformation itself quotientable across representations.
6. **If U₀ remains intrinsically trivial:** freeze the negative result and construct U₁ with a richer generative constraint rather than changing the ontology inspector. Candidate later branches include action/energy constrained or learned/self-organizing universes.

## Research rule

Do not make U₀ more interesting by tuning it after seeing these results. U₀ stays frozen. If it is intrinsically simple, that is the result.
