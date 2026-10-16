# Atlas U₁ — Checkpoint 1

U₁ is the first out-of-sample universe after freezing U₀ as the method-development specimen.

## Frozen construction

- substrate: compact boundaryless `T^10`
- anonymous periodic coordinates `theta_1 ... theta_10`
- periodic input encoding `[cos(theta_i), sin(theta_i)]`, dimension 20
- frozen network: `20 -> 48 -> 48 -> 24`, tanh hidden activations
- state dimension: 24
- deterministic seed: `20261016`
- no substrate coordinate is designated space, time, momentum, scale, or anything else

The `10 -> 24` mismatch was chosen to avoid generically forcing large self-intersection families by dimension counting.

## 1. Baseline dimensionality

Across 10,000 sampled substrate points:

- Jacobian rank is 10/10 everywhere sampled
- median local tangent participation dimension is ~5.06
- 10–90% local participation interval is ~4.19–5.80
- global active-spectrum participation dimension is ~9.62
- output PCA requires ~12 components for 95% variance and ~17 for 99%

Therefore U₁ genuinely uses all ten substrate dimensions; there is no trivial emergent 4D collapse.

Same-architecture controls show the local ~5-dimensional sensitivity participation is architecture-generic rather than U₁-specific.

## 2. Strict global symplectic / conjugate-pair test

A constant nondegenerate 2-form `Omega` on the torus is automatically closed in the global periodic coordinate atlas. We searched for a constant `Omega` such that

`(g^{-1} Omega)^2 ~= -I`

across the universe, which would make the induced metric approximately compatible with one global almost-complex/symplectic pairing.

U₁ held-out RMS residual:

`~0.924`

Same-architecture control residuals are approximately `0.908–0.954`; U₁ sits in the middle of the control distribution.

The optimized `Omega` is well-conditioned and nondegenerate, but compatibility is poor. Conclusion:

> U₁ does not naturally select one simple global constant conjugate-pair / phase-space structure.

This does not rule out local or state-dependent symplectic structures.

## 3. Embedded observers recover hidden dimension by intervention

Independent random sensory projections of the 24D state were used. The observer is given local intervention buttons but no substrate coordinates or hidden state.

At one regular point, the exact local sensory fiber dimension is `10-k` for a k-channel observer. Exact local fibers were numerically continued while holding sensory state fixed.

Active intervention-response variation along those exact sensory-identical fibers gives:

- k=7 sensory channels: true hidden fiber dimension 3; response participation ~2.86; 3 PCs explain 95% and 99%
- k=8: true hidden dimension 2; response participation ~1.88; 2 PCs explain 95% and 99%
- k=9: true hidden dimension 1; response participation ~1.00; 1 PC explains 95% and 99%

So the U₀ embedded-scientist principle transfers cleanly to a 10D universe:

> same observation + controlled interventions reveal the dimension of predictively missing state.

This is an out-of-sample success of the ontology-discovery method rather than a U₀-specific fact.

## 4. Translational symmetry search

Because the substrate is a torus, constant angular directions are global translation vector fields. We searched for directions minimizing induced-metric Lie-derivative energy.

U₁ best energy:

`~0.0913`

Same-architecture controls span roughly `0.078–0.123`. U₁ is unexceptional.

Therefore there is no evidence yet for a distinguished approximate homogeneous / translation direction analogous to the unusually strong U₀ direction.

## Current verdict

U₁ begins in a deliberately hostile state for familiar physics:

- all 10 substrate degrees are genuinely active
- no special 4D active subspace
- no exceptional global conjugate-pair structure
- no exceptional translation symmetry

But the embedded-observer science survives and correctly recovers hidden dimensionality from action-response contradictions.

This is a good baseline: if familiar or lower-dimensional physics later emerges, it will have to come from richer local, relational, multiscale, or observer-effective structure rather than an obvious global decomposition already present in the substrate.

## Next targets

1. Search for **local/state-dependent symplectic forms** and test whether they patch into a bundle even though one global constant `Omega` fails.
2. Let embedded observers autonomously learn minimal latent predictive charts on U₁ and test whether the recovered local dimensions match `10-k` without exact-fiber supervision.
3. Search for multiscale dimensional reduction or geometry phases under coarse-graining.
4. Search general Killing vector fields rather than only constant translations.
5. Test whether any lower-dimensional effective geometry emerges for observer quotients even though the ontic substrate stays 10D.
6. Run description-length competition between one global model and an atlas of local models.
7. Only after native U₁ structure is mapped, compare deliberately physics-shaped universes such as 6D phase-space or 26D string-inspired variants.
