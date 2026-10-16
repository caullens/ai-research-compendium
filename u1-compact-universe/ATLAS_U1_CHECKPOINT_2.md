# Atlas U₁ — Checkpoint 2: observer science transfers; phase-space pairing does not

## 1. Embedded observers recover hidden local dimensionality

Using the same frozen U₁, random sensory projections with k=7,8,9 channels were tested. At a regular point the exact local sensory fiber has dimension `10-k`.

Exact fibers were numerically continued while holding the observer's sensory state fixed. Along each fiber, the observer's responses to ten unlabeled local intervention buttons were recorded.

Results:

- k=7: true hidden fiber dimension 3; response participation dimension ~2.86; 3 PCs explain 95% and 99% of response variation
- k=8: true hidden dimension 2; response participation ~1.88; 2 PCs explain 95% and 99%
- k=9: true hidden dimension 1; response participation ~1.00; 1 PC explains 95% and 99%

Thus the active ontology principle discovered in U₀ transfers cleanly to U₁:

> controlled interventions reveal the dimensionality of predictively missing state.

This is an out-of-sample methodological success.

## 2. No simple global symplectic / phase-space structure

A constant nondegenerate closed 2-form Omega was optimized for compatibility with the induced metric through

`(g^{-1} Omega)^2 ~= -I`.

Held-out RMS residual is ~0.924, and U₁ is unexceptional among same-architecture controls.

Therefore no privileged global constant conjugate pairing is detected.

## 3. Low-complexity state-dependent closed 2-form also fails robustly

To test whether the pairing could vary over the torus, a closed field was parameterized as

`omega = omega0 + d alpha`,

where `alpha` contains only first Fourier harmonics in the ten anonymous torus variables. Closure `d omega = 0` is therefore exact by construction.

A validation-controlled optimization was used to prevent merely fitting the training region.

Final residuals:

- train median ~0.897, RMS ~0.902
- validation median ~0.908, RMS ~0.930
- held-out test median ~0.909, RMS ~0.957

The previous constant-Omega held-out RMS benchmark is ~0.924.

So the nonconstant closed field slightly improves many individual points but produces a bad tail and is worse overall on held-out RMS. It also approaches near-degeneracy in some regions (10th-percentile smallest singular value ~0.0013).

Conclusion:

> U₁ currently shows no robust low-complexity symplectic / conjugate-pair field selected by its induced geometry.

This is a useful negative result. The ten anonymous substrate degrees do not spontaneously organize into a simple five-pair phase space under the tested global or first-harmonic closed structures.

## 4. No exceptional approximate translation direction

Constant torus translation vector fields were searched for approximate Killing behavior under the induced metric.

U₁ best metric Lie-derivative energy: ~0.0913.

Same-architecture controls span roughly ~0.078–0.123. U₁ is control-scale and has no exceptional homogeneous direction analogous to the U₀ preferred direction.

## Current interpretation

U₁ is doing what an out-of-sample universe should do: it is rejecting familiar structures that were not actually earned.

What transfers from U₀ is not a particular geometry or symmetry. What transfers is the **observer-science methodology**: action-response contradictions expose missing state dimension.

That distinction is important.

## Next targets

1. Remove exact-fiber supervision and let observers autonomously mine predictive contradictions in U₁.
2. Let local latent charts be born automatically and ask whether their dimension converges to `10-k`.
3. Search for multiscale dimensional reduction or geometry phases in observer-effective quotients rather than in the ontic 10D substrate alone.
4. Run description-length competition between a single global latent model and an atlas of local predictive models.
5. Search general nonconstant Killing fields and local symmetry algebras.
6. Keep symplectic geometry open only if a new intrinsically motivated candidate arises; do not continue adding flexibility merely to force a phase-space interpretation.
