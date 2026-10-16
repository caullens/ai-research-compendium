# Atlas U₀ — Geometry classification checkpoint

This checkpoint asks what kind of Riemannian geometry U₀ actually carries under the induced state-space metric

`g = J^T J`.

This geometry is intrinsic under source reparameterization and output Euclidean isometry, but not under arbitrary nonlinear target redescription.

## 1. Signature

Because `g = J^T J` and the sampled Jacobian has rank 4, the induced metric is positive definite. Thus the current geometry is Riemannian, not Lorentzian.

A spacetime-like Lorentzian metric would have to emerge from additional structure; it is not the metric currently induced by the Euclidean 8D state representation.

## 2. Not Euclidean / spherical / hyperbolic

The full Riemann tensor was computed in orthonormal tangent frames using the Gauss equation.

A constant-curvature 4-manifold satisfies

`R_abcd = K (delta_ac delta_bd - delta_ad delta_bc)`.

For U₀, the median normalized residual to the best local constant-curvature tensor is approximately

`0.922`.

This decisively rules out local resemblance to a single Euclidean (`K=0`), spherical (`K>0`), or hyperbolic (`K<0`) model over most sampled points.

The six curvature-operator eigenvalues are highly unequal and sign-mixed. Median ordered eigenvalues are approximately

`[-6.14, -1.50, -0.41, -0.057, 0.292, 1.89]`.

Across all curvature-operator modes, roughly 62% are negative and 38% positive.

Thus U₀ has simultaneous positive and negative sectional-curvature directions rather than one global curvature sign.

## 3. Not conformally flat

In four dimensions the Weyl tensor measures the part of curvature not determined by Ricci curvature.

For U₀,

`median ||W|| / ||Riemann|| ≈ 0.602`.

So U₀ is far from conformally flat. In particular it is not merely Euclidean/hyperbolic/spherical geometry seen through a varying local scale factor.

This is one of the strongest current indicators that the geometry is genuinely nontrivial rather than a warped visualization of a standard model space.

## 4. Not Einstein

The previously measured Einstein residual remains large:

`median ||Ric - (R/4)g|| / ||Ric|| ≈ 0.72`.

The Ricci tensor has strongly unequal eigenvalues; median eigenvalues are roughly

`[-7.63, -4.17, -0.67, 0.87]`.

So curvature is highly anisotropic and includes a positive Ricci direction at typical points.

## 5. Apparent low-rank curvature

The curvature operator on 2-forms is often approximately rank-2: retaining only the two largest-magnitude curvature eigenmodes gives a median relative residual around

`0.204`.

This initially suggested a possible local `2+2` product geometry, because a product of two surfaces has only two nonzero curvature-operator modes.

However, stricter tests reject a clean product interpretation:

- median complementarity of the two dominant curvature bivectors is only ~0.089, whereas complementary factor planes would be near 1;
- the dominant bivectors are only approximately simple 2-planes, especially the second mode;
- a Ricci spectrum constrained to repeated pairs `[a,a,b,b]` has median residual ~0.261;
- only ~43% of points have opposite signs for the two dominant curvature modes.

Therefore the low effective rank of curvature appears to be real, but it is not explained by a simple product such as `H² x H²`, `H² x S²`, `H³ x R`, or `H² x R²`.

## 6. Current geometry verdict

The best current description is:

> U₀ is a noncompact, inhomogeneous, anisotropic, sign-mixed curved 4D Riemannian manifold/immersion with substantial Weyl curvature and a surprisingly low-effective-rank curvature operator.

It is not well classified by the standard homogeneous model geometries tested so far.

This makes the geometry more interesting, not less: the next problem is to discover whether it belongs to a less obvious structured class or is simply a generic neural-induced Riemannian geometry.

## Next targets

1. Map curvature invariants over the compactified universe rather than one source box.
2. Test local symmetry by measuring `nabla R`; locally symmetric spaces satisfy `nabla R = 0`.
3. Search for general Killing vector fields, not only constant source directions, and estimate the isometry-group dimension.
4. Measure geodesic-ball volume growth and geodesic divergence; compare operationally against Euclidean, hyperbolic, and mixed-curvature models.
5. Measure conjugate points / Jacobi fields and sectional-curvature-dependent focusing.
6. Classify the curvature operator using its self-dual / anti-self-dual decomposition in 4D, including Weyl `W+` and `W-` spectra.
7. Test whether the approximate rank-2 curvature has a coherent spatial eigenbundle even though it is not a direct product geometry.
8. Compare all of these diagnostics against same-architecture random universes to separate U₀-specific geometry from generic neural-map geometry.
