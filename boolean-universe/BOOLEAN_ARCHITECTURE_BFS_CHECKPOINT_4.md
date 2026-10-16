# Boolean Universe — Architecture BFS Checkpoint 4

This checkpoint closes two more breadth-first architecture axes at three nodes: observer coupling dimensionality and single-primitive NAND construction.

## 1. Observer bandwidth / port multiplicity

Architecture held fixed: complete 3-node recurrent graph with all self-loops and affine GF(2) local rules. All 4,096 networks were exhausted.

Node roles remain operational rather than permanent.

For each experiment:

- choose 1 or 2 nodes as actuators and clamp them before each update;
- the remaining nodes are persistent internal state;
- choose 1, 2, or 3 nodes as sensors, possibly overlapping the actuator set;
- test controllability of the hidden internal state and observability of that state from action/output histories.

Aggregate fractions of all network-port experiments:

| Actuators | Sensors | Controllable | Observable | Both |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0.4922 | 0.2917 | 0.1992 |
| 1 | 2 | 0.4922 | 0.5000 | 0.3047 |
| 1 | 3 | 0.4922 | 0.6563 | 0.3750 |
| 2 | 1 | 0.8125 | 0.5000 | 0.4167 |
| 2 | 2 | 0.8125 | 0.7500 | 0.6198 |
| 2 | 3 | 0.8125 | 0.8750 | 0.7188 |

The underlying network is unchanged; only observer coupling changes.

At the network level, the fraction admitting at least one fully controllable+observable port choice is:

- 1 actuator / 1 sensor: ~54.69%
- 1 actuator / 2 sensors: ~59.67%
- 1 actuator / 3 sensors: ~59.67%
- any 2-actuator configuration with at least one suitable sensor choice: ~94.51%

Thus:

> **Knowability and controllability are joint properties of universe plus observer bandwidth, not intrinsic properties of the network alone.**

Adding sensors cannot improve controllability because it does not change intervention authority; adding actuators can. Conversely, increasing sensor count improves observability without modifying the internal law.

This cleanly separates three notions:

- ontic state dimension;
- intervention/control dimension;
- observation/sensing dimension.

## 2. NAND-only constructive breadth-first search

To remove local rule-family freedom entirely, every computational node was constrained to the same primitive:

`NAND(a,b)`.

The only external primitives are the two Boolean signals `x` and `y`. Every added computational node is an identical 2-input NAND neuron. Its inputs may come from `x`, `y`, or any earlier NAND output; repeated inputs are allowed, so `NAND(x,x)` acts as NOT.

Acyclic circuits were searched breadth-first by number of NAND neurons, quotienting away gate names and duplicate identical intermediate signals.

All 16 binary Boolean functions are constructible with at most **five NAND neurons**.

Exact minimal NAND-node counts:

- 0 gates: 2 laws (`x`, `y`)
- 1 gate: 3 new laws (`NOT x`, `NOT y`, `NAND`)
- 2 gates: 4 new laws
- 3 gates: 4 new laws
- 4 gates: 2 new laws (`XOR`, `NOR`)
- 5 gates: 1 final law (`XNOR`)

Cumulative external law capacity by NAND budget:

- 0 gates: 2 / 16
- 1: 5 / 16
- 2: 9 / 16
- 3: 13 / 16
- 4: 15 / 16
- 5: 16 / 16

Familiar examples by minimal DAG gate count:

- NAND: 1
- AND: 2
- OR: 3
- XOR: 4
- NOR: 4
- XNOR: 5

This gives the clean opposite limit from unrestricted Boolean neurons:

> **When every neuron is identical, expressivity migrates into architecture.**

The network does not gain capabilities because individual neurons become more sophisticated. It gains them through composition, reuse, depth, and wiring.

This is especially useful for the architecture program because it disentangles `what a node can do` from `what a graph of such nodes can do`.

## 3. Emerging breadth-first picture

The three-node program now spans several largely independent axes:

- topology / edge direction;
- reciprocal versus one-way coupling;
- self-loops;
- feed-forward versus recurrent structure;
- local rule family;
- homogeneous single-gate networks;
- synchronous, sequential, and asynchronous semantics;
- fixed versus role-free ports;
- number of actuator and sensor channels;
- architecture versus effective-law equivalence;
- minimal wiring / gate complexity.

The strongest current principles are:

1. **Implementation is not law.** Many architectures realize the same exact global dynamics.
2. **Minimal implementation complexity is still meaningful.** An effective law has an intrinsic cost relative to a chosen primitive family.
3. **Node roles need not be ontological.** Input/output/hidden are experimental couplings.
4. **Observer dimensionality matters.** Control and observability can change dramatically with port count while the universe stays fixed.
5. **Scheduler semantics is physics.** Synchronous and asynchronous worlds built from identical rules can have different attractors.
6. **Primitive restrictions induce global laws.** Monotonicity bounds periods; affine structure strongly supports reversibility.
7. **Architecture can replace node sophistication.** NAND-only circuits recover all binary Boolean laws using at most five identical computational nodes.

## Next transition

The three-node shell is now broad enough that the next stage can finally begin moving outward to four nodes **without abandoning breadth-first discipline**.

Recommended first four-node shell:

1. enumerate unlabeled directed topologies for restricted families rather than immediately all arbitrary Boolean rules;
2. compare complete graph, cycle, ring/reciprocal, chain, star, and role-free architectures;
3. repeat affine, monotone, NAND-constructive, and threshold families;
4. track which three-node principles survive at four nodes;
5. specifically watch for genuinely new phenomena unavailable with three bits: longer antichains, larger reversible groups, richer scheduler dependence, and observer-role flexibility.

Three nodes should remain the exact reference shell for controls and proofs.