# U₁ — Frozen Compact Neural Universe

U₁ is the first out-of-sample universe built after freezing the U₀ Atlas.

The purpose is to remove U₀'s privileged `R^4` substrate assumption and ask whether any lower-dimensional, spacetime-like, phase-space-like, gauge, geometric, or multiscale ontology emerges without being built in.

## Frozen substrate

The hidden substrate is the 10-torus

`T^10 = (S^1)^10`.

Each latent coordinate is periodic and anonymous. No coordinate is designated as space, time, momentum, scale, internal degree, or anything else.

The torus is used because it is compact and boundaryless, avoiding U₀'s privileged center/infinity structure. Its product topology is a deliberate remaining prior and must be treated as such.

## Frozen state map

Seed: `20261016`

Periodic feature map:

`theta in T^10 -> [cos(theta_1..theta_10), sin(theta_1..theta_10)] in R^20`

Frozen MLP:

- input: 20 periodic features
- hidden 1: 48 tanh units
- hidden 2: 48 tanh units
- output/state: 24 anonymous real channels

Weight scales:

- W1: `1/sqrt(20)`
- W2: `1/sqrt(48)`
- W3: `1/sqrt(48)`
- all bias scales: `0.12`

Weights and biases are generated deterministically by the same Mulberry32 + Box–Muller construction used for U₀, with seed 20261016.

U₁ is frozen from this checkpoint onward. No architecture, seed, weights, or substrate parameters should be tuned to make the universe more interesting.

## Why 10 -> 24?

The numbers are intentionally not spacetime numbers.

The output dimension 24 is chosen to avoid forcing generic global self-intersections merely by dimension count: a generic 10-manifold can embed in dimension 21 or greater by the Whitney embedding theorem. Thus any strong self-identification structure in U₁ should not be assumed from `m < 2n` dimensional pressure alone.

The output should still be treated as an anonymous state representation, not as physical Euclidean space unless an Atlas branch explicitly adopts that gauge.

## Initial Atlas baseline

10,000 uniform substrate samples were audited.

### Local dimension

Every sampled Jacobian had rank 10.

Therefore no exact lower-dimensional local manifold emerged immediately.

The singular-value participation dimension was nevertheless anisotropic:

- median local effective sensitivity dimension: ~5.06
- 10–90% interval: ~4.19–5.80

However, same-architecture random controls show this is ordinary for the architecture/random-matrix geometry rather than evidence for emergent five-dimensional physics.

### Global active dimension

The global mean pullback metric has participation dimension ~9.62.

Its normalized eigen-spectrum is broad rather than sharply truncated. There is no obvious 4D active subspace.

Same-architecture controls again put U₁ in the ordinary range.

### Linear state-cloud dimension

Output PCA requires approximately:

- 12 components for 95% variance
- 17 components for 99% variance

So the 24D state cloud is not trivially linearly compressible to four dimensions.

### Full-state ambiguity audit

A 10,000-sample nearest-neighbor search found no suspicious near-collisions between widely separated substrate points. This is only a sampling result, not a proof of injectivity.

## Methodological result already established

The first baseline is intentionally negative:

> U₁ does not spontaneously present a 4D world at the surface level.

This is desirable. Any later emergence of a 4D, 3+1, 3+3, symplectic, Lorentzian, gauge, or other familiar structure will have to be earned by deeper Atlas analysis rather than inherited from the substrate dimension.

## Immediate next branches

1. **Observer science** — create observers with fewer channels than the 24D full state and test local/global ontology reconstruction.
2. **Intrinsic geometry** — characterize the pullback geometry on T^10 without assuming that all 10 latent directions are physically fundamental.
3. **Dimension discovery** — test whether local predictive/state atlases compress to fewer dimensions even though the raw Jacobian rank is 10.
4. **Symplectic search** — look for emergent closed nondegenerate 2-forms or conjugate pair structure without labeling variables `q` or `p`.
5. **Symmetry algebra** — search for internally selected Killing/vector-field structure and commuting/noncommuting generators.
6. **Multiscale ontology** — test whether coarse variables or local geometric phases reduce the effective dimension or simplify laws.
7. **Description-length decryption** — compare one global law against patchwise/local-atlas descriptions plus transition rules.
8. **Topology recovery** — eventually ask whether embedded observers can infer compactness and recurrent cycles without being told the hidden substrate is T^10.

U₀ remains frozen as the method-development/reference universe. U₁ is the first genuine out-of-sample test universe.