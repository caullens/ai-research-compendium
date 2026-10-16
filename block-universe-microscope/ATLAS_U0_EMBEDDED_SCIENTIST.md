# Atlas U₀ — Embedded Scientist Checkpoint: discovering missing predictive state

This branch removes the god's-eye ontology from the learner. The embedded observer receives only:

- a 7-channel sensory state `y`,
- four unlabeled local intervention buttons,
- the sensory change produced by an intervention.

It is not shown source coordinates, the full map `F`, the omitted eighth channel, branch labels, or the fact that projection is occurring.

## 1. Active certificate that observation is not sufficient state

Along an exact 7-channel observational equality curve,

`P7 F(a) = P7 F(b)`,

the same sensory state can produce different intervention-response signatures under the same four actions.

Thus the observer can internally falsify the hypothesis

`observation = complete state`

by finding

`same sensation + same action -> different consequence`.

A large unlabeled exploration dataset also permits discovery of these aliases without pre-supplying exact equality pairs: sensory-neighbor pairs with anomalously different intervention responses stand out sharply from the ordinary local sensory/action relation.

## 2. Minimal latent completion on one alias family

A predictive-state learner was given one chosen probe response as context, compressed through a bottleneck of dimension `d`, and asked to predict the other three action responses.

On the D0 exact 7-channel equality family:

- d=0: held-out predictive R² ~ 0.004
- d=1: R² ~ 1.000
- d=2: no meaningful gain
- d=3: no meaningful gain

All four possible probe buttons show the same qualitative result.

The learned one-dimensional branch-separation magnitude correlates ~0.96 with the actual omitted eighth-channel separation, although the omitted channel is used only for evaluation and never for training.

Interpretation: locally, one additional predictive degree of freedom is both necessary and sufficient.

## 3. Global-coordinate hypothesis fails under hard holdout

A deliberately harsher test trained one latent model on five defect neighborhoods and held the sixth defect neighborhood out.

Transfer was mixed to poor, especially for remote/saturated defect regions, and the learned scalar did not align globally with the omitted eighth state coordinate.

This rejects the premature claim that the learner had already discovered one universal global eighth coordinate.

The failure is best interpreted as an out-of-distribution / gauge problem: local predictive completions exist, but a single latent parameterization learned in one set of regions does not automatically extrapolate to disconnected regions.

## 4. Local latent-atlas result

The same minimal-latent experiment was repeated independently in all six exact defect neighborhoods.

Held-out within-chart predictive R²:

| defect | d=0 | d=1 | d=2 |
|---|---:|---:|---:|
| D0 | ~0.001 | ~0.9999 | ~0.9999 |
| D1 | ~0.006 | ~0.9990 | ~0.9992 |
| D2 | ~0.002 | ~0.9999 | ~0.9999 |
| D3 | ~0.000 | ~1.0000 | ~0.9999 |
| D4 | ~0.000 | ~0.9999 | ~0.9998 |
| D5 | ~0.000 | ~0.9999 | ~1.0000 |

Across all six neighborhoods:

- mean d=0 R² ~ 0.0015
- mean d=1 R² ~ 0.9998
- mean d=2 R² ~ 0.9998
- mean gain d=0 -> d=1 ~ 0.9983
- mean gain d=1 -> d=2 ~ 0.00003

So six independent local experiments all conclude:

> seven-channel observation is missing exactly one local predictive degree of freedom.

This is a stronger and more defensible result than claiming one globally learned hidden coordinate.

## 5. Chart gluing on overlap

Two independent one-dimensional latent charts were trained on overlapping portions of the D0 alias neighborhood.

They selected different gauges (different signs/scales), but on the overlap:

- Spearman correlation between their branch-contrast coordinates ~ 0.91
- a simple smooth cubic transition function explains ~84% of the chart-to-chart relation

Thus independently invented latent coordinates can be related by an overlap transition rather than forced into one privileged gauge.

This is the first concrete evidence for a learned latent **atlas**:

- local predictive coordinates,
- arbitrary local gauge,
- overlap transition maps.

## Current interpretation

The embedded observer can now do more than notice prediction errors.

It can empirically infer:

1. its sensory state is not predictively sufficient,
2. one additional local degree of freedom repairs that insufficiency,
3. the same local codimension appears repeatedly in independent regions,
4. independently learned hidden coordinates need not agree numerically, but can be glued on overlap.

This is much closer to ontology discovery than direct latent reconstruction.

The emerging principle is:

`discover the smallest local state augmentation required by experiment, then glue compatible local augmentations into a global atlas if possible.`

## Next targets

1. Replace god's-eye-selected equality curves with fully autonomous contradiction mining and local chart birth.
2. Let the learner decide when to spawn a new chart versus expand an existing one.
3. Learn transition functions between overlapping charts and test cocycle consistency on triple overlaps.
4. Determine whether the local 1D hidden charts can be assembled into a global line bundle or whether nontrivial holonomy/topology obstructs a single global coordinate.
5. Use active experiment selection: choose the intervention that maximally reduces ambiguity in the current chart.
6. Ask whether the learner can distinguish projection-induced aliases from the six genuine full-state double fibers, where even the full point value is not interventionally sufficient and local germ/branch structure is required.
