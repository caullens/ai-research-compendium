# Atlas U₀ — Complex / twistor geometry checkpoint

This checkpoint asks whether the 4D Riemannian geometry induced by `g = J^T J` contains a naturally selected complex structure rather than merely permitting arbitrary coordinate pairing.

## 1. Curvature-selected almost-complex structure

In oriented 4D Riemannian geometry, unit self-dual or anti-self-dual 2-forms define orthogonal almost-complex structures. U₀'s Weyl curvature was decomposed into `W+` and `W-` blocks, and the most Kähler-like eigenline in either block was used to construct a candidate `J`.

Algebraically this works exactly: `J^2 = -I` to machine precision.

However, the geometry does not look Kähler:

- median Kähler Weyl-pattern residual is ~0.30 in both `W+` and `W-` sectors;
- neither self-dual sector dominates globally;
- the curvature-selected `J` has large nonzero Nijenhuis tensor;
- `||nabla J||` and `||d omega||` are also large on sampled points.

Thus the curvature-selected almost-complex structure is strongly non-integrable and non-Kähler.

## 2. Curvature-selected 2+2 splitting

The dominant curvature-operator eigenbivector is strikingly close to a genuine simple 2-plane: median non-simplicity residual is approximately `4e-4` on stable sampled points.

This motivated a search for a local 2+2 foliation. The dominant 2-plane field and its orthogonal complement were tested with a corrected Frobenius obstruction normalized by the local plane-twisting scale.

Results:

- dominant plane median obstruction ~0.84;
- complement median obstruction ~0.61;
- <1% of points have dominant-plane obstruction below 0.1;
- <5% have complement obstruction below 0.1;
- essentially no sampled points have both distributions simultaneously close to integrable.

So U₀ has a remarkably plane-like dominant curvature mode, but that plane field twists non-integrably rather than assembling into clean 2D leaves.

## 3. Search for a single global integrable Hermitian complex structure

Constant orthogonal complex structures on the source chart form two `S^2` families. Because their coefficients are constant, they are integrable. The full families were searched for a `J` minimizing metric incompatibility

`||J^T g J - g|| / ||g||`.

U₀'s best median residual is ~1.60, and same-architecture control universes are in essentially the same range. Therefore the induced metric does not support a distinguished global Hermitian structure of this simple integrable type.

## 4. Twistor / half-flat audit

In 4D the canonical almost-complex structure on twistor space is integrable when one Weyl half vanishes (orientation convention aside). U₀ was therefore tested for self-duality / anti-self-duality.

Globally U₀ is far from half-flat:

- median `min(||W+||,||W-||) / ||W|| ~ 0.662`;
- random sampling found no points below 0.1.

However, bounded optimization found a rare localized near-half-flat point near

`x ≈ (-0.140, 1.309, -0.416, 0.588)`

with

- `||W+|| ≈ 0.2115`
- `||W-|| ≈ 0.00962`
- half-flatness ratio ≈ `0.0454`.

This is a genuine local self-duality-like patch, but it does not simultaneously satisfy the Kähler Weyl eigenvalue pattern: residuals remain ~0.40–0.47. The nearby region quickly becomes less half-flat, so this currently looks like a localized twistor-friendly curvature event, not a global complex law.

## Current verdict

U₀ does not presently look like a complex, Hermitian, Kähler, or globally self-dual 4-manifold.

What it *does* possess is more unusual:

1. every point admits the usual 4D twistor sphere of orthogonal almost-complex structures;
2. curvature often selects a nearly simple 2-plane very strongly;
3. that preferred plane twists non-integrably;
4. rare localized regions become strongly approximately self-dual/anti-self-dual.

A reasonable description is therefore closer to a twisted almost-complex / twistor-rich geometry than to an ordinary complex surface.

## Next targets

1. Determine whether near-half-flat points organize into curves/surfaces or are isolated minima.
2. Compare rarity/depth of half-flat patches against same-architecture random universes.
3. Track the dominant curvature 2-plane by parallel transport and measure its holonomy.
4. Search for coherent local almost-complex patches by minimizing Nijenhuis energy directly rather than deriving `J` from Weyl eigenvectors.
5. Test whether a locally optimized compatible `J` can be glued across overlaps into a nontrivial almost-complex atlas.
6. Relate the local twistor geometry to the learned latent-gauge atlas from the embedded-scientist branch.
