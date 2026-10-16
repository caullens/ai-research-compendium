# Atlas U₀ — Embedded observer science

This checkpoint removes god's-eye ontology from the observer as much as possible while keeping U₀ frozen.

The embedded observer is given only:

- a 7-channel sensory vector,
- four unlabeled local intervention buttons,
- the sensory consequence of each intervention.

It is **not** given source coordinates, the map F, the eighth state channel, or knowledge that its sensory state is a projection.

## 1. Active certificate that sensory state is not sufficient

Along an exact 7-channel observational equality curve through the true D0 double fiber,

`P7 F(a(u)) = P7 F(b(u))`

holds to numerical residual around `1e-13`.

The observer's four-button infinitesimal response signature is

`R(x) = P7 J(x) A`

for an unknown-to-the-observer action basis A.

Despite exact sensory equality, the response signatures on the two branches differ strongly everywhere along the curve. Relative branch mismatch is about 1.03–1.11 across the tested neighborhood, including about 1.064 at the true full-state D0 defect.

Thus the observer can perform an entirely internal sufficiency test:

> same sensory state + same intervention does not imply same sensory consequence.

That directly falsifies the hypothesis that the current sensory vector is a complete Markov/state description.

## 2. Discoverability from unlabeled exploration data

A 40,000-state black-box exploration dataset was generated. The observer sees only sensory vectors and local intervention-response signatures.

For ordinary nearby sensory states, response-signature difference follows a smooth local scaling law. Outliers exist where sensory separation is small but causal-response separation is extreme.

The strongest outliers have response differences roughly an order of magnitude larger than the ordinary local baseline. Secret god's-eye evaluation confirms that these outliers are typically far-separated source states (median hidden separation about 3.28 for the top 25) rather than ordinary local neighbors (about 0.41 for low-surprise pairs).

Therefore a limited observer can discover state aliasing from sensory/action data alone, without being told where the aliases are.

## 3. Estimating how much hidden structure is missing

For nested observers with k=4,5,6,7 sensory channels, exact equality manifolds through the same D0 neighborhood have dimensions

`8-k = 4,3,2,1`.

The embedded observer does not know those pair-space parameters. It can, however, compare intervention-response signatures among equal-sensation branches.

The dimensionality of response-signature variation tracks the hidden equality-family dimension:

- k=7: participation dimension ≈ 1.001; 1 component explains 95%
- k=6: participation dimension ≈ 1.215; 2 components explain 95%
- k=5: participation dimension ≈ 2.146; 3 components explain 95%
- k=4: participation dimension ≈ 2.940; 3 components explain 95% in the tested local patch (under-resolving the true 4D family, likely because one response direction is weak/nonlinear over the finite patch)

So active causal variation can recover the missing-dimensionality ladder very cleanly at k=5..7 and partially at k=4.

## Interpretation

This gives a concrete route for an embedded observer to discover hidden ontology:

1. detect near-identical sensory states,
2. replay the same interventions,
3. find inconsistent consequences,
4. conclude the sensory state is not sufficient,
5. use the dimension of causal-response variation to estimate the number of unresolved hidden degrees of freedom.

This is stronger than passive anomaly detection. It is an experimentally falsifiable internal statement about the observer's own ontology.

## Important complication: true U₀ defects remain causally branched

At the genuine full-state D0 defect, the full 8D state is exactly equal on two source branches but the local derivatives differ. Thus even full pointwise state F(x) is not sufficient to determine responses to source-local interventions at a transverse self-intersection.

Operationally, the local *germ* of the universe (state plus neighborhood/derivative structure) distinguishes branches that the point value alone cannot.

This suggests a new ontology candidate:

- point state is not always fundamental enough;
- branch/germ structure may be operationally necessary.

## Next targets

1. Formalize an embedded learner that estimates hidden-state dimension from sensory/action data with no access to source coordinates.
2. Compare point-state vs first-jet/germ representations as sufficient predictive states.
3. Test whether adding history can substitute for active probes in recovering branch identity.
4. Ask whether the observer can distinguish projection-induced aliases from genuine ontic double fibers without acquiring another sensory channel.
5. Build a minimal self-supervised 'scientist' that proposes a latent augmentation only when repeated intervention contradictions require it.
