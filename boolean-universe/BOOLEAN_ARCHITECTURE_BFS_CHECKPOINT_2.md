# Boolean Universe — Architecture BFS Checkpoint 2

This checkpoint continues the breadth-first search over architecture axes rather than extending one network family deeply.

## 1. Self-loops as an independent topology axis

For three Boolean nodes, allowing all directed edges including self-loops gives `2^9 = 512` labeled directed graphs. After quotienting arbitrary node relabeling, these collapse to exactly **104 unlabeled topologies**.

To keep every topology exactly enumerable, this sweep uses affine GF(2) local rules:

`x_i' = c_i XOR XOR_j a_ij x_j`

restricted to the incoming neighbors supplied by the topology.

Across the 104 topologies:

- local-rule configuration counts range from 8 to 4096;
- synchronous reversible-law fraction ranges from 0 to 0.328125;
- synchronous nontrivial-cycle fraction ranges from 0 to ~0.7517;
- possible periods reach 7.

The fully connected graph including all three self-loops is the synchronous extreme:

- 4096 affine rule configurations;
- 32.8125% reversible;
- ~75.17% contain a nontrivial periodic orbit;
- periods up to 7.

Averaging over unlabeled topologies, reversibility rises strongly with self-loop count. In this affine family, self-feedback is therefore a major enabler of reversible state evolution.

## 2. Update semantics is part of the architecture

The same graph and same local rules were evaluated under two time semantics:

1. synchronous update — all nodes update from the old global state;
2. deterministic sequential update — nodes update in order `0 -> 1 -> 2`, with later nodes seeing earlier updates during the same macro-step.

The resulting global maps frequently differ.

Across topologies, the fraction of rule assignments giving identical synchronous and sequential macro-dynamics has median only ~0.344. Dense graphs are especially sensitive: at high edge count, typically only ~15–20% of rule configurations retain the same global map.

Sequential update generally reduces the prevalence of nontrivial cycles relative to synchronous update.

Most strikingly, for the affine family used here, sequential reversibility is zero for every topology with fewer than three self-loops; with all three self-loops available, mean sequential reversibility becomes 12.5%.

Thus:

> **The rule `who updates when` is not merely an implementation detail. It is part of the physical law of a recurrent Boolean universe.**

## 3. Local computational primitive versus wiring

The maximally connected three-node graph with all self-loops was held fixed while only the local Boolean function family changed.

Because every node sees all three current bits, each local truth table uniquely specifies one coordinate of the global 8-state map.

Exact census:

| Local rule family | Functions per node | Global universes | Reversible universes | Reversible fraction |
|---|---:|---:|---:|---:|
| affine GF(2) | 16 | 4,096 | 1,344 | 0.328125 |
| threshold | 104 | 1,124,864 | 240 | ~0.000213 |
| unrestricted Boolean | 256 | 16,777,216 | 40,320 | ~0.002403 |

The unrestricted result follows because the complete self-loop graph can realize every map from eight global states to eight global states; exactly `8!` of these maps are bijections.

The affine result follows from global affine dynamics `x' = A x + b`; reversibility is exactly `A in GL(3,2)`, yielding `168 * 8 = 1344` reversible laws.

The threshold result was exhaustively enumerated.

This gives an important non-monotonic result:

> **Increasing local expressivity does not monotonically increase structural properties such as reversibility.**

Affine rules are dramatically more reversible than either threshold or unrestricted rules on the same graph, because affine algebra strongly constrains the global law ensemble toward permutations when `A` is invertible.

So `topology` and `local computation primitive` must remain separate BFS dimensions.

## 4. Role-free nodes become operational control/sensing ports

The complete three-node graph with self-loops and affine rules was then treated as fully role-free.

For an experiment:

- choose any node as an **actuator**;
- before each step overwrite that node with action bit `a_t`;
- the remaining two bits constitute persistent internal state;
- synchronously compute all three next-node values;
- choose any node as a **sensor** and read its computed next value.

Thus input/output/hidden are not permanent neuron species. They are properties of how an external observer couples to the same network.

All 4,096 affine complete-graph universes and all 9 ordered actuator/sensor choices were exhaustively tested for:

- controllability of the two-bit internal state under the binary action;
- observability/distinguishability of the four internal states through the chosen sensor under action sequences.

Results:

- 890 / 4096 networks allow **every node** to serve as a fully controlling actuator;
- no network supports all 9 ordered actuator/sensor pairs as simultaneously fully controllable and observable;
- the maximum is 8 / 9 good port pairs, achieved by 60 networks;
- same-node actuator/sensor experiments are observable in 37.5% of network-port cases and fully control+observe in ~22.27%;
- different-node pairs are observable in 25% and fully control+observe in 18.75%.

The distribution of good ordered port choices per network spans 0 through 8.

Therefore:

> **`input neuron`, `output neuron`, and `hidden neuron` are unnecessary primitive node types. Role flexibility is itself an emergent property of a network plus a coupling protocol.**

Some universes are much more role-flexible than others even though their graph topology is identical.

## 5. Updated architectural ontology

At this point, several independent architecture axes demonstrably change the possible universe ensemble:

- graph topology;
- presence of self-feedback;
- synchronous versus sequential time semantics;
- local Boolean rule family;
- observer coupling / port choice;
- layered width/depth and bottlenecks from Checkpoint 1.

This supports treating the primitive object as:

`interaction graph + local rule family + update semantics + intervention/observation protocol`

rather than a conventional typed neural network.

## Next breadth-first shell

Remain at three nodes before moving upward. High-priority untested axes are:

1. **Asynchronous/event-driven time** — update one node chosen externally, randomly, or by a local trigger rather than a fixed sequence.
2. **Symmetric/undirected constraints** — require reciprocal edges and compare to arbitrary directed graphs.
3. **Monotone/local gate restrictions** — AND/OR-style monotone rules, NAND-only, XOR-only, threshold, affine, unrestricted.
4. **Self-loop × rule-family interaction** — determine whether the strong affine reversibility effect survives other primitive families.
5. **Skip/residual architecture** in tiny layered networks.
6. **Port multiplicity** — multiple actuators, multiple sensors, and changing ports through time.
7. **Architecture equivalence** — identify when different graph/rule/update descriptions induce the exact same global transition law or the same observer-effective behavior.

Only after these three-node shells are mapped should four-node architecture space become the main frontier.