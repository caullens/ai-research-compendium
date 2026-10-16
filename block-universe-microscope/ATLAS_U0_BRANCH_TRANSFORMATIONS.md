# Atlas U₀ — Branch: preferred transformations and foliations

This branch begins after freezing the Whitney/self-intersection work as one completed Atlas chapter. The goal here is to ask whether U₀ itself selects preferred transformations, foliations, homogeneous directions, or an internal time-like decomposition before adding any new law.

## Gauge discipline

Results are classified by what survives:

1. arbitrary smooth re-description,
2. source diffeomorphism + output Euclidean isometry,
3. source/output orthogonal change of basis,
4. the exact neural implementation.

This branch intentionally studies weaker gauges than the strict-topology chapter because richer geometry is otherwise quotiented away. The weaker gauge is always stated explicitly.

## 1. Principal line fields and approximate 1+3 structure

Using the Euclidean state metric in R^8, pull back the metric through U₀:

`g(x) = J(x)^T J(x)`.

Its four eigen-directions define local principal sensitivity line fields. Across random samples, the weakest singular direction is usually well separated from the next one:

- median `sigma_weak / sigma_next ≈ 0.339`
- about 79% of samples have ratio below 0.5

For each principal line field n(x), test whether the orthogonal 3-plane distribution `ker(n · dx)` satisfies the Frobenius condition. A normalized obstruction based on `alpha wedge d alpha` gives median values:

- weakest: ~0.484
- middle-low: ~0.521
- middle-high: ~0.479
- strongest: **~0.172**

Thus the strongest-sensitivity line field has a substantially more nearly hypersurface-orthogonal complement than the other three. This is an approximate candidate 1+3 split.

However, control experiments on 20 random networks of the same architecture produce a similar strongest-line effect. U₀'s value (~0.164 on the control sample) is near the ensemble median (~0.171), not exceptional.

Interpretation: an approximate 1+3 tendency exists in U₀ under the induced Euclidean geometry, but it appears to be generic to this architecture rather than a distinctive emergent time direction.

## 2. Approximate central inversion symmetry

Search all 383 nonidentity signed coordinate permutations. For each source transformation, fit the best output-space Euclidean isometry + translation on training points and measure held-out error.

The transformation

`x -> -x`

is a dramatic outlier:

- held-out relative RMSE ~0.149
- second-best signed permutation ~0.359
- best of 200 random orthogonal source rotations ~0.375
- median signed permutation ~0.551

The fitted output transformation is almost exactly a point reflection:

`y -> c - y`

with the fitted orthogonal matrix differing from `-I` by only about 0.56% in normalized Frobenius norm on the tested domain.

This is best understood as a strong approximate Z₂ symmetry inherited from the near-odd tanh architecture with small biases. It is real for frozen U₀ but substantially architecture-prior-driven.

## 3. Preferred approximately homogeneous direction

Rather than test coordinate axes, search over unit directions v and ask whether a finite source translation

`x -> x + delta v`

can be represented by one global output-space isometry + translation.

For `delta=0.6`, optimization finds

`v_* ≈ (0.054, -0.998, -0.012, -0.033)`.

Held-out relative error:

- optimized direction: ~0.0994
- nearby second coordinate axis: ~0.1002
- median random direction: ~0.182

So U₀ selects a direction very close to the second external coordinate as substantially more homogeneous than typical directions. This direction was found by search; the axis label was not privileged in the objective.

The fitted state transformations approximately compose as a one-parameter group:

- shift 0.2 equivariance error ~0.033
- shift 0.4 ~0.067
- shift 0.6 ~0.101
- shift 0.8 ~0.136

and

- `T_0.2 o T_0.2` vs `T_0.4`: rotation mismatch ~0.0029, translation mismatch ~0.0016
- `T_0.4 o T_0.4` vs `T_0.8`: rotation mismatch ~0.0120, translation mismatch ~0.0090

This is the strongest current candidate for an emergent continuous transformation family in U₀.

The direction also corresponds to relatively weak coupling in the first layer / average pullback metric, so it may ultimately reduce to a random anisotropy of the frozen network rather than a deep symmetry. The next tests should distinguish those interpretations.

## Current interpretation

Three distinct preferred-structure signals now exist under the Euclidean-state gauge:

1. approximate Z₂ central inversion,
2. an architecture-generic near-integrable 1+3 principal distribution,
3. a U₀-specific approximately translationally homogeneous direction.

None is yet justified as physical time.

The third is the most interesting because it is operational: finite translations along one discovered source direction induce approximately composable state transformations.

## Next tests

1. Re-express the homogeneous direction without using the source Euclidean coordinate metric; search for the same one-parameter family directly from state correspondences.
2. Test whether the approximate translation has an infinitesimal generator acting on state and whether that generator closes under composition more accurately than finite Procrustes fits.
3. Search for additional independent approximately homogeneous directions and determine the dimension of the approximate transformation group.
4. Test commutators between discovered transformation generators. Commuting generators would suggest an emergent Abelian translation sector; noncommuting generators could indicate richer geometry.
5. Search for invariants of the preferred transformation family. A nontrivial conserved scalar/state relation would materially strengthen the case for emergent dynamics.
6. Compare with same-architecture random universes to separate architecture-generic transformation structure from U₀-specific structure.
