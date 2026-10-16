# Atlas U₀ — Checkpoint 4: defect ecology under strict topology

This checkpoint asks whether the six known transverse double fibers have intrinsic relationships to one another inside fixed U₀, and what survives under regular homotopy.

## Global search stabilization

A new heavy-tailed / compactification-inspired source sampling pass (30k points, strongly biased toward large radius and near-infinity regions) produced many near-collisions in output state but no seventh exact transverse double fiber after root refinement.

This further strengthens the working hypothesis that the finite exact double-fiber census has stabilized at six, though a proof is still absent.

## Boundary-incidence audit

Each generic asymptotic facet of the compactified boundary can be parameterized by three unsaturated first-layer coordinates while the remaining 21 first-layer units saturate to ±1.

All 4048 top-dimensional boundary facets were enumerated. Coarse grid sampling over every facet was followed by nonlinear least-squares refinement of the best candidates for each of the six finite defect states.

No defect state was matched by any optimized boundary facet.

Best residuals in the current output chart:

- D0: ~0.2630
- D1: ~0.1614
- D2: ~0.1540
- D3: ~0.0510
- D4: ~0.0130
- D5: ~0.0153

The residual magnitudes themselves are representation-dependent. The useful result is the failure to reach equality: there is currently no evidence that any finite double fiber is also an asymptotic/boundary fiber.

Interpretation: the six known defects behave as finite internal self-identifications rather than endpoints of branches extending to infinity.

## Strict fixed-U₀ topology looks sparse

Assuming the finite fiber census really is complete and no extra boundary identifications appear, the finite image is homotopy equivalent to a contractible 4D source with six disjoint point-pair identifications, hence to a wedge of six circles.

Under that model:

- each double fiber contributes one free H₁ / π₁ generator,
- there are no intrinsic relations between those generators at the level of homotopy type,
- the six defects form a signed set rather than an interacting graph.

This is important negative evidence: visual proximity or apparent pairing between defects is not preserved by arbitrary diffeomorphic source/target redescription.

## Whitney self-intersection interpretation

For oriented half-dimensional immersions `N^n -> M^(2n)` with even `n`, each transverse double point has a sign and the algebraic signed sum is the classical Whitney self-intersection invariant.

The six known U₀ crossings have signs:

`+ + - - - +`

so the current algebraic sum is zero.

Classical Whitney-disjunction results in simply connected ambient manifolds (for n >= 3, under the standard compact/boundary hypotheses) allow opposite-sign double points to be removed by regular homotopy. Therefore, if a sufficiently large compact restriction of U₀ satisfies the required boundary conditions and the six-point census is complete, there is no topological obstruction from the signed double-point invariant to regularly homotoping that restriction to an embedding.

Consequences:

1. The individual six defects are genuine intrinsic features of the *exact frozen map* U₀.
2. They are probably **not protected species** under regular homotopy.
3. Their most robust collective invariant is the signed algebraic count, currently zero.
4. There is no canonical topological pairing between the 3 positive and 3 negative defects; Whitney cancellation can pair compatible opposite signs after choosing Whitney disks.

## Nearby-universe behavior

Generic unconstrained neural-weight perturbations need not stay inside the immersion class. In earlier continuation experiments a defect could terminate when the source Jacobian itself lost rank at an isolated codimension-5 event. Thus a single signed defect may disappear along a generic network family because the family crosses out of the immersion space.

Within **regular homotopy** (source derivative remains injective), the signed total is the relevant conserved quantity; creation/cancellation occurs through opposite-sign pairs at self-tangency/Whitney events rather than arbitrary single-defect disappearance.

## Revised ontology interpretation

Under the strictest smooth-redescription gauge, fixed U₀ currently supports:

- ordinary regular 4D states,
- six isolated binary self-identification defects,
- a structured non-proper boundary at infinity,
- free quotient loops induced by the six point-pair identifications (conditional on census completeness),
- a zero signed double-point sum.

What it does **not** yet support intrinsically:

- a canonical defect distance,
- a canonical nearest neighbor,
- a canonical defect interaction graph,
- a preferred pairing between opposite signs,
- protected defect species under regular homotopy,
- a preferred dynamics/time law.

## Research implication

This is evidence that a frozen random smooth map can have real intrinsic topology while still lacking a rich protected ontology. Bare random geometry gives defects, but not yet a natural dynamics or interaction law that would make those defects behave like persistent physical entities.

That strengthens the case for the later U₁ experiment: keep the ontology inspector unchanged, but give the universe one explicit intrinsic variational principle (least action / energy) and ask whether stable defect dynamics, conservation laws, and richer emergent ontology then arise.

## Next work

1. Attempt a numerical Whitney cancellation of one +/− pair while explicitly monitoring rank 4 everywhere on a large sampled region.
2. Test whether the six-defect quotient really yields rank-6 H₁ numerically via a simplicial/graph approximation.
3. Improve completeness of the global double-fiber census in compactifying coordinates.
4. Probe lower-dimensional boundary strata for exact finite/boundary identifications.
5. If the Whitney cancellation works cleanly, freeze U₀ as the baseline “topology without physics” universe before constructing U₁.
