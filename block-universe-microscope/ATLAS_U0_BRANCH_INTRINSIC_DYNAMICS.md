# Atlas U₀ — Branch: intrinsic dynamics without a chosen source time

This branch asks what dynamics can be defined directly on the observed state image of U₀, without privileging one of the four source coordinates.

## 1. Transverse double fibers are universal fixed points

At each exact double fiber `F(a)=F(b)`, the two sheet tangent spaces are `im J(a)` and `im J(b)`. For all six catalogued defects, `[J(a), -J(b)]` has rank 8, so

`im J(a) ∩ im J(b) = {0}`.

Therefore any globally single-valued state-space vector field that remains tangent to U₀ on both branches must satisfy

`G(F(a)) = G(F(b)) = 0`.

This is independent of a particular time law: every intrinsic tangent dynamics must fix all six transverse defect states.

A related audit shows that no nonzero constant source direction can descend globally through the quotient. For each defect, `J(a)-J(b)` has rank 4, and the stacked six-defect constraint also has rank 4. Even the least-inconsistent constant source direction gives substantial velocity disagreement at every defect.

## 2. Finite rigid/affine source translations do not define a true infinitesimal symmetry

Earlier finite Procrustes fits suggested one approximately homogeneous source direction. Infinitesimal audits are much harsher:

- best skew-affine state generator explains only ~3.2% of held-out directional derivative variance;
- best general-affine generator explains only ~7.7%.

So the finite translation effect should not be promoted to a genuine continuous Euclidean/affine state symmetry.

## 3. Quadratic coordinate-pushforward laws are much better but do not close

Allowing quadratic state vector fields to approximate pushforwards of source directions gives held-out R² approximately:

- 0.831
- 0.723
- 0.718
- 0.689

However, the fitted quadratic vector fields fail Lie-bracket closure almost completely: projecting pairwise brackets back into the four-field span leaves ~99.7–99.9% residual.

Interpretation: the fields are good local approximations to branchwise pushforwards but do not constitute a globally closed transformation algebra. The self-identifications are a natural reason: source translations commute on the source, but a constant source direction cannot descend consistently through transverse double fibers.

## 4. Source-free tangent-flow search

Instead of selecting a source direction, search directly for state vector fields `G(s)` that:

1. are tangent to the U₀ state image on ordinary states,
2. vanish exactly at the six intrinsic double-fiber states.

### Affine fields

For `G(s)=As+b`, imposing exact zeros at all six defects leaves a 24-dimensional coefficient nullspace. The best held-out field has normal-energy fraction ~0.073, i.e. roughly 93% of its energy is tangent to U₀ on unseen states.

### Quadratic fields

For general degree-2 vector fields, exact defect-zero constraints leave 312 free coefficients. On held-out states:

- best normal-energy fraction ≈ 0.0159 (98.4% tangent),
- second ≈ 0.0166,
- first ten modes are all below ~0.0324 normal energy.

A shuffle control destroys the state↔tangent correspondence. The same search then gives independent-shuffle test normal fractions around 0.11–0.13, and the true best U₀ field evaluated on shuffled tangent planes gives ~0.15–0.20. Thus the low leakage exploits real U₀ state–tangent structure rather than arbitrary polynomial flexibility.

## 5. U₀ permits many dynamics but does not select one

The quadratic admissible-flow spectrum has no isolated lowest mode:

- mode 1 training normal fraction ≈ 0.0112,
- mode 2 ≈ 0.0133,
- mode-2/mode-1 ratio ≈ 1.18,
- 32 modes lie below 5% normal energy,
- 70 modes lie below 10% normal energy.

This is the current central result of the branch:

> Bare U₀ strongly constrains admissible global dynamics—most notably by forcing every transverse defect to be a fixed point—but it does not canonically choose a unique dynamics.

The geometry supplies a large family of low-complexity approximately tangent autonomous state flows.

## Interpretation

This cleanly separates **kinematics** from **law**:

- the immersed quotient structure determines what a globally consistent state dynamics is allowed to do;
- it does not determine which allowed vector field is physically realized.

This is direct motivation for, but not yet permission to add, an intrinsic variational principle in a later universe. A least-action/energy principle would have a sharply defined role: select trajectories from an already nontrivial admissible-flow space rather than manufacture ontology from scratch.

## Next Atlas branch

Observer-dependent ontology: restrict access to lower-dimensional projections of the 8D local state and measure how fiber multiplicity, local identifiability, and apparent dimension change. This tests whether different observers of the exact same U₀ inhabit meaningfully different effective worlds.
