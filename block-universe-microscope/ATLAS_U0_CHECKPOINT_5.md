# Atlas U₀ — Checkpoint 5: explicit compact-support regular-homotopy cancellations

This checkpoint moves beyond invoking the Whitney trick abstractly. It constructs concrete smooth compact-support deformations of U₀ and numerically continues opposite-sign double points through their annihilation events while auditing source rank.

## Why compact support matters

U₀ is non-proper. A deformation that merely decays at infinity can still create or import a defect from arbitrarily large source radius. An early trial exhibited exactly that loophole.

The corrected homotopies therefore use a C-infinity compact-support tube bump around a chosen source arc. Outside the tube the map is mathematically identical to U₀.

The generic form is

`H_epsilon(x) = F(x) + epsilon * b(x) * v`

where `b(x)=exp(-z/(1-z))` for `z<1` and `b(x)=0` for `z>=1`.

## Cancellation A: D0(+) with D3(-)

A localized tube around one source arc connecting the D0 and D3 preimages produced a critical event at

`epsilon_c ~= 0.10074338094485015`.

Immediately below the critical value, there are two nearby double roots with opposite signs. At the critical value they merge into one nontransverse double point. Immediately above it, local root searches find no nontrivial double root.

Critical diagnostics:

- output equality residual: ~2.8e-16
- smallest singular value of combined two-sheet matrix `[J_H(x), -J_H(y)]`: ~8.5e-18
- smallest singular values of the individual sheets: ~0.0689 and ~0.0486
- critical preimages remain distinct in source space, separated by ~5.22

Thus the failure of transversality happens between the two sheets, while each source sheet remains an immersion.

A dense random audit plus global optimization over the compact support and homotopy interval found no source-Jacobian rank collapse. The smallest source singular value located globally in the support was ~1.35e-3; at the cancellation event itself the sheets are much better conditioned.

Interpretation: this is strong numerical evidence for an internal Whitney-style cancellation inside the immersion class.

## Cancellation B: D5(+) with D4(-)

A second, spatially separate compact-support tube produced an independent cancellation at

`epsilon_c ~= 0.46453950475988964`.

Critical diagnostics:

- output equality residual: ~1.3e-15
- combined two-sheet smallest singular value: ~4.1e-18
- individual sheet weakest singular values: ~0.0361 and ~0.00741
- critical preimages remain distinct, separated by ~8.66

Again, local root searches show two nearby roots of opposite sign just below the event and no nearby double roots just above it.

The compact-support rank audit found no introduced source singularity. The smallest source singular value found during the support search was ~9.8e-5, occurring near the edge of the support in an already strongly saturated region where the deformation is effectively zero and the value matches U₀'s baseline conditioning.

## Independence of the two moves

The support of Cancellation A is exactly zero on the four source preimages involved in Cancellation B, and the support of Cancellation B is exactly zero on the D0/D3 preimages.

Therefore the two moves can be composed without directly changing the other pair's defining preimages. Numerically they realize removal of four of the six catalogued finite double points by two compact-support regular-homotopy events.

## Remaining pair

The remaining catalogued pair is D1(+) and D2(-).

A simple single-tube / single-target-direction ansatz did not bring this pair together. Their natural source arcs are much farther apart, and the same low-expressivity deformation family is inadequate.

This is not evidence that cancellation is impossible; it is evidence that the numerical ansatz is too restrictive. The next construction should approximate a genuine Whitney disk using multiple coordinated compact-support bumps or a 2D localized deformation field.

## Ontology consequence

The experiments now directly support the gauge hierarchy:

1. Frozen-map ontology: all six double fibers are intrinsic facts of exact U₀.
2. Regular-homotopy ontology: at least two opposite-sign pairs are explicitly removable while staying numerically in the immersion class.
3. Arbitrary map deformation: the signed count can change if source rank is allowed to collapse.

So the individual defects are real but not protected species. U₀ increasingly looks like a baseline universe with intrinsic topology but no protected physics.

## Next work

1. Build a higher-expressivity compact-support Whitney-disk deformation for D1(+)/D2(-).
2. Explicitly compose the two verified local cancellations and re-run the full defect census.
3. Verify the asymptotic boundary is exactly unchanged by the compact-support moves.
4. If the third pair is removable too, close U₀ as a 'topology without protected dynamics' baseline and move to U₁ with one intrinsic variational principle while keeping the ontology inspector fixed.
