# Atlas U₀ — Branch: physical-law analogues

This branch asks whether the frozen universe U₀ contains structures that play roles analogous to physical law without fitting named equations indiscriminately.

## Discipline

Candidate physics must survive at least one explicitly stated gauge and outperform same-architecture random control universes. Familiar names (Einstein, Klein–Gordon, Noether, gauge field, etc.) are used only for structural analogies when the corresponding mathematical test is actually performed.

## 1. Induced geometry

Using the Euclidean metric of the 8D state representation, U₀ inherits a pullback Riemannian metric

`g = J^T J`.

This is intrinsic under source reparameterization and output Euclidean isometry, but not under arbitrary nonlinear target redescription.

### Minimal-submanifold test

For the second fundamental form B, a minimal immersion would satisfy `trace B = 0`.

U₀ median normalized mean-curvature ratio:

`||trace B|| / ||B|| ≈ 0.841`.

Across 12 same-architecture random controls U₀ is near the middle of the distribution. Therefore U₀ is not unusually close to a Nambu–Goto/minimal-surface extremum.

### Einstein-metric test

The Ricci tensor was computed from the second fundamental form via the Gauss equation. An Einstein metric would satisfy

`Ric = (R/4) g`.

U₀ median normalized residual:

`||Ric - (R/4)g|| / ||Ric|| ≈ 0.709`.

Again U₀ is unexceptional among controls. The induced geometry is not approximately Einstein in a strong global sense.

### Curvature

U₀ has genuine mixed intrinsic curvature. On the sampled region:

- median normalized scalar curvature `R / ||B||² ≈ -0.292`
- 10–90% interval approximately `[-0.688, 0.061]`
- about 76% of sampled sectional curvatures are negative

The geometry is therefore predominantly negatively curved but highly nonconstant. This is geometry, not a discovered gravitational field equation.

## 2. Intrinsic field-equation search

For the immersion, the intrinsic Laplace–Beltrami operator of the embedding satisfies `Delta_g F = trace_g B`.

Several low-complexity candidate laws were fit on one region and evaluated held-out:

- isotropic Helmholtz / Euclidean Klein–Gordon analogue `Delta_g F ≈ -lambda(F-F0)`: R² ≈ 0.035
- affine law `Delta_g F ≈ A F + b`: R² ≈ 0.099
- quadratic state law `Delta_g F ≈ Q(F)`: R² ≈ 0.077

These scores are control-universe scale. No convincing universal mass scale or simple field equation was found.

## 3. Approximate Killing direction

Search constant source directions v minimizing normalized metric variation `||L_v g||`.

U₀ has a clear preferred direction

`v ≈ (-0.06, 0.996, -0.01, 0.06)`

with metric-change energy around `0.145`, while the remaining principal directions are around `0.63, 0.74, 1.71` in the same normalization.

This independently recovers nearly the same direction previously found by finite translation-equivariance search.

Relative to 12 same-architecture controls, only one control had a lower best-direction energy. Thus U₀ is unusually, though not uniquely, homogeneous along one direction.

### Noether-style consequence

Geodesics of the induced metric were numerically integrated and the would-be conserved momentum

`p_v = g(v, xdot)`

was monitored. The preferred direction gives the best/near-best momentum stability, but only weakly: median normalized drift is about 0.50 versus roughly 0.49–0.62 for the other principal directions.

Therefore the approximate Killing signal is real enough to track, but the induced momentum is **not** conserved accurately enough to call this a physical conservation law.

## 4. Gauge-style latent atlas

The embedded-scientist branch independently learns one-dimensional hidden predictive coordinates in overlapping regions. Their numerical values are arbitrary local gauges.

Three independent overlapping D0 charts A, B, C were trained separately. Pairwise affine transition quality:

- A↔B R² ≈ 0.992
- B↔C R² ≈ 0.999
- A↔C R² ≈ 0.987

The composed transition A→B→C agrees with the directly learned A→C transition to about 1.7% of the chart-C overlap scale. A full A→B→C→A round trip returns to chart A with about 2.1% relative mismatch.

This is the first strong gauge-theoretic analogue in U₀:

- local state coordinates are not unique,
- overlap transition functions carry the invariant content,
- transitions approximately satisfy a cocycle consistency condition.

At present this is an approximate coordinate bundle, **not** yet a gauge field: no nontrivial curvature/holonomy or dynamical gauge principle has been established.

## Current verdict

Familiar fundamental equations tested so far mostly fail:

- no minimal-action embedding,
- no Einstein-like metric,
- no simple Klein–Gordon/Helmholtz-type field equation,
- no strong Noether conservation from the approximate symmetry.

But U₀ does contain physically recognizable mathematical scaffolding:

1. a curved 4D intrinsic geometry,
2. one unusually homogeneous geometric direction,
3. observer-induced caustics and cross-caps analogous to geometric-optics singularities,
4. locally learned hidden-state gauges with consistent transition functions.

The gauge-atlas structure is currently the most promising path toward something genuinely physics-like.

## Next targets

1. Build many overlapping local hidden-state charts and test cocycle consistency on larger covers.
2. Transport latent gauge orientation around closed loops and measure holonomy. Nontrivial loop-dependent holonomy would be the first candidate gauge curvature.
3. Distinguish true bundle holonomy from neural-chart training error using repeated independently trained covers and control universes.
4. Search for Killing vector fields more generally than constant directions and test whether an approximate Lie algebra of isometries exists.
5. If a gauge connection emerges, test whether its curvature satisfies any simple differential identities beyond universal Bianchi-type identities.
6. Continue to keep the 26D/string-theory experiment parked until U₀'s native physics branch is better mapped.
