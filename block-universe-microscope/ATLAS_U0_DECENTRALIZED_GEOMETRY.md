# Atlas U₀ — Decentralized / multiscale geometry checkpoint

This checkpoint tests the hypothesis that U₀'s mathematical structure is not globally simple at the point level, but is distributed across loops, local geometric regimes, scales, and representation choices.

## 1. Loop information and holonomy

Using the induced Riemannian metric `g = J^T J`, the six local curvature endomorphisms `R(e_i,e_j)` were assembled in orthonormal tangent frames.

Across 320 sampled points:

- all 320 had full six-dimensional curvature-generator rank,
- so the generic local holonomy algebra is compatible with full `so(4)` rather than a reduced special-holonomy algebra,
- but the median effective generator dimension is only ~2.52,
- median singular spectrum is approximately `[4.84, 1.87, 0.91, 0.38, 0.16, 0.052]`.

Thus all six rotational degrees exist, but curvature is strongly concentrated into roughly 2–3 effective modes. This independently recovers the earlier low-effective-rank curvature finding.

Explicit Levi-Civita parallel transport around small closed loops gives nontrivial holonomy. Holonomies around different local 2-planes generally fail to commute, demonstrating genuinely non-Abelian loop information. This means part of U₀'s geometric content is intrinsically relational/loop-based rather than point-local.

## 2. Simple coarse-graining does not reveal a textbook geometry

A microscope coarse-graining was defined by Gaussian-local averaging of `F` in source coordinates, with radii from 0 to ~0.96. This is explicitly a microscope choice rather than an intrinsic RG prescription.

As scale increases:

- median curvature magnitude decreases strongly,
- but constant-curvature tensor residual only falls from ~0.922 to ~0.901,
- Weyl/Riemann fraction stays near ~0.60,
- effective holonomy-generator dimension remains around 2.6–2.8.

Thus smoothing removes curvature strength without converting U₀ into Euclidean, spherical, hyperbolic, or conformally-flat geometry. There is no obvious scale at which familiar constant-curvature physics simply appears.

## 3. Linear representation “decryption” helps, but does not solve the geometry

A global positive-definite metric on the 8D state representation was optimized, equivalent to an invertible linear re-description of state, to make the pullback metric as homogeneous as possible.

Held-out pullback-metric variation improved by about 1.40x:

- original test variation ~0.501
- optimized test variation ~0.358

So the raw state coordinates do hide some geometric simplicity.

However:

- the optimized state metric is highly anisotropic (condition number ~210),
- constant-curvature residual remains ~0.90,
- Weyl/Riemann remains ~0.59,
- absolute curvature becomes much larger under the optimized metric.

Therefore U₀ is not merely a simple geometry encrypted by a tame global linear transformation. Any deeper decryption would have to be nonlinear, local, multiscale, or relational.

## 4. Unsupervised local geometry atlas

Each sampled point was represented only by coordinate-invariant local curvature fingerprints:

- normalized curvature-operator spectrum,
- normalized Ricci spectrum,
- Weyl fraction,
- constant-curvature residual,
- effective holonomy dimension,
- normalized scalar curvature.

Unsupervised clustering strongly prefers two local geometry regimes (best silhouette ~0.46):

### Regime 1

- larger population (~78%)
- predominantly negative scalar curvature
- lower Weyl fraction (~0.58)
- lower effective holonomy dimension (~2.46)
- curvature spectrum dominated by a large negative mode

### Regime 2

- smaller population (~22%)
- much more positive scalar-curvature character
- higher Weyl fraction (~0.66)
- higher effective holonomy dimension (~2.79)
- curvature spectrum dominated by a large positive mode

The regimes are spatially coherent beyond shuffled-label expectations in both source-neighbor and state-neighbor graphs.

A source-radius audit rejects the simplest saturation-shell explanation:

- source radius alone has AUC ~0.51 for regime identity,
- the two regimes have nearly identical median source radius,
- a quadratic function of source coordinates predicts regime only moderately (CV AUC ~0.68), indicating nontrivial spatial organization rather than concentric shells.

## 5. Regimes persist across coarse-graining scales

The same points were re-fingerprinted after coarse-graining at radii 0, 0.22, 0.48, and 0.82.

Two-regime cluster quality remains stable (silhouette roughly 0.43–0.45 at all tested scales).

Adjusted-Rand similarity of the regime assignments:

- scale 0 vs 0.22: ~0.84
- 0 vs 0.48: ~0.73
- 0 vs 0.82: ~0.52
- 0.48 vs 0.82: ~0.63

After optimal label matching, microscopic vs radius-0.82 assignments still agree for ~88% of points.

Interpretation: the exact regime boundaries flow with scale, but the two-regime organization survives. This is the first concrete evidence for nested/multiscale geometric domains in U₀ rather than a single pointwise geometry classification.

## Current working model

The evidence now favors the following picture:

> U₀ is not a single hidden textbook geometry waiting to be named. Its geometric information is distributed across local invariant regimes, loop holonomy, and scale-dependent-but-persistent domain structure. Representation changes can simplify parts of the description but do not globally decrypt it into a standard model space.

This makes “decentralized geometry” a concrete research hypothesis rather than a metaphor.

## Next targets

1. Learn a nonlinear state re-description explicitly optimized for geometric description length, while guarding against degenerate metric collapse.
2. Build an intrinsic neighborhood/RG scheme using geodesic or diffusion neighborhoods rather than source-coordinate Gaussian smoothing.
3. Track local geometry-regime boundaries as codimension-1 objects and test whether defects, caustics, near-self-dual patches, or double fibers preferentially sit on those boundaries.
4. Compute Wilson-loop-like summaries of Levi-Civita / learned gauge holonomy across loops of different scales and test whether loop statistics themselves exhibit phase structure.
5. Search for a compact grammar of local models + transition rules: whether the global universe can be compressed more effectively as a patchwork atlas than as one global law.
6. Test whether an embedded scientist can discover the regime structure and loop invariants without access to source coordinates or the full state representation.
