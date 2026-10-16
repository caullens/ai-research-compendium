# Boolean Universe — Breadth-First Architecture Sweep

This checkpoint changes the research strategy from depth-first progression through one architecture to a breadth-first sweep over **architecture dimensions**.

## Architecture axes

The search space is organized explicitly by independent choices:

1. **Node count** — 1, 2, 3, ... Boolean neurons.
2. **Graph topology** — isolated nodes, chains, forks, cycles, complete graphs, arbitrary directed graphs.
3. **Directionality** — feed-forward DAGs, directed recurrent graphs, symmetric/bidirectional graphs.
4. **Self-connections** — absent/present.
5. **Update semantics** — feed-forward evaluation, synchronous recurrence, later asynchronous/event-driven variants.
6. **Role assignment** — permanently typed input/hidden/output neurons versus role-free nodes that are clamped or observed through an interaction protocol.
7. **Local rule family** — unrestricted Boolean function, threshold neuron, affine/XOR neuron, monotone rule, restricted primitive gates, etc.
8. **Layering** — no layers, one hidden layer, multiple hidden layers, or graph architectures with no layer concept.

The central methodological rule is now:

> Sweep the smallest values across all architecture axes before increasing any one axis deeply.

## Sweep A — all directed graphs with 2 and 3 neurons

First semantics:

- Boolean nodes;
- no self-loops;
- synchronous autonomous update;
- an edge `j -> i` means node `i` may use the current bit of node `j`;
- each node may implement any Boolean function of its incoming neighbors;
- a zero-indegree node is a Boolean constant.

### Graph counts

For 2 nodes:

- 4 labeled directed graphs;
- 3 unlabeled topologies.

For 3 nodes:

- 64 labeled directed graphs;
- **16 unlabeled directed topologies**.

Every topology and every local-rule configuration is exhaustively enumerable at this size.

### Architecture matters immediately for dynamics

Some representative results:

- two isolated nodes: 4 rule configurations, no nontrivial periodic behavior, no reversible configurations;
- single directed edge: 8 configurations, again no nontrivial cycles;
- complete directed 2-node graph: 16 configurations, 25% reversible, periods up to 4;
- directed 3-cycle: 64 configurations, 12.5% reversible, periods up to 6;
- complete directed 3-node graph: 4096 configurations, ~2.73% reversible, periods up to 7, and ~56.2% of rule configurations have a nontrivial periodic orbit.

Across the sixteen 3-node graph topologies, adding connectivity and feedback changes both the available cycle lengths and the frequency of recurrent behavior.

This is the first clean breadth-first result:

> **Topology itself changes dynamical law space even when node count and local rule family are fixed.**

## Complete graph architecture

A complete directed graph on `n` Boolean neurons, without self-loops, has an edge from every neuron to every other neuron.

For `n=3`, each neuron sees the other two bits and may implement one of 16 binary Boolean functions. Therefore the complete local-law space has

`16^3 = 4096`

configurations.

It is not a layered network. Every neuron simultaneously acts as part of the current state and as an input to every other neuron's next-state computation.

This is naturally a recurrent Boolean network rather than a feed-forward neural network.

## Role-free neurons

A second sweep removes permanent `input`, `hidden`, and `output` types.

The graph is the system. A role is assigned only by the **interaction protocol**:

- to use a neuron as an input, clamp or intervene on its current state;
- to use a neuron as an output, observe its current or next state;
- an unobserved, unclamped neuron is hidden relative to that experiment;
- the same neuron may be both manipulated and observed.

On the complete directed 3-node graph, every node can therefore serve as an input port, output port, hidden state, or combinations of these depending on the experiment.

For one-step experiments with one clamped input node and one observed output node:

- when input and output are different nodes, all four unary laws ZERO/ID/NOT/ONE are realizable robustly;
- when the same node is both the manipulated and observed port, the next-state rule has no self-edge and therefore cannot directly depend on that clamped bit, leaving only constant robust laws under hidden-context variation;
- most rule configurations are context-dependent, meaning unobserved neurons alter the apparent input/output law.

This demonstrates that `input`, `output`, and `hidden` need not be neuron species. They can be **observer-relative roles**.

That formulation should be retained as a major architecture branch.

## Local-rule family is an independent architecture axis

Using unrestricted Boolean neurons makes depth deceptively unimportant: a single 2-input neuron can already compute any of the 16 binary Boolean laws, including XOR.

To expose architectural effects cleanly, a second sweep uses Boolean threshold neurons:

`y = 1[sum_i w_i x_i >= theta]`.

For two Boolean inputs, threshold neurons realize exactly 14 of the 16 laws. The missing functions are:

- XOR
- XNOR

### Width and depth under threshold neurons

Exact external-law capacity:

| Architecture | Laws of 16 realizable | XOR/XNOR? |
|---|---:|---|
| `2 -> 1` | 14 | no |
| `2 -> 1 -> 1` | 14 | no |
| `2 -> 1 -> 1 -> 1` | 14 | no |
| `2 -> 2 -> 1` | 16 | yes |
| `2 -> 1 -> 2 -> 1` | 14 | no |
| `2 -> 2 -> 1 -> 1` | 16 | yes |
| `2 -> 2 -> 2 -> 1` | 16 | yes |

This gives two strong architectural principles.

### Principle 1 — width can create distinctions unavailable to one neuron

A two-unit threshold hidden layer can represent the nonlinear partition necessary for XOR/XNOR, after which a threshold output can combine those distinctions.

So unlike the unrestricted-Boolean rule family, architecture now genuinely changes expressivity.

### Principle 2 — downstream width cannot recover an early bottleneck

The architecture

`2 -> 1 -> 2 -> 1`

still realizes only 14 laws.

Once the first one-bit threshold bottleneck has collapsed a distinction required by XOR, expanding back to two neurons later cannot reconstruct that missing information.

This directly generalizes the earlier partition result:

> **A downstream network cannot recover distinctions erased by an upstream representation.**

## Two hidden layers

Two hidden layers are therefore not one phenomenon. Their effect depends on where width occurs.

- `2 -> 1 -> 1 -> 1`: more depth, no extra expressivity.
- `2 -> 1 -> 2 -> 1`: later width, but the early bottleneck remains fatal to XOR.
- `2 -> 2 -> 1 -> 1`: the first layer preserves the needed partition; later depth retains full 16-law capacity.
- `2 -> 2 -> 2 -> 1`: also full capacity.

This makes `layer count` a poor single measure of architecture. Information-preserving topology and local rule restrictions matter more.

## Revised research program

The Boolean project should now proceed by **architecture shells** rather than a single Brick 8, Brick 9, ... path.

### Shell A: <=3 nodes, exhaust everything

For each topology and rule family:

- feed-forward DAGs;
- directed recurrent graphs;
- bidirectional/undirected connectivity;
- complete graphs;
- optional self-loops;
- fixed roles versus role-free ports;
- unrestricted Boolean vs threshold vs affine/XOR vs monotone local rules;
- synchronous versus asynchronous updates where recurrence exists.

### Shell B: layered architectures with <=4 total computational nodes

Enumerate width/depth arrangements such as:

- one hidden layer;
- two hidden layers;
- bottlenecks;
- skip connections;
- residual/recurrent shortcuts;
- all-to-all versus sparse layers.

### Shell C: first architecture families too large for full law enumeration

Only after the smaller shells are complete should node count increase enough that sampling/search replaces exact exhaustion.

## Current conceptual shift

The object of study is no longer primarily a neural network architecture of the usual `input -> hidden -> output` type.

A more general primitive is:

> **a Boolean interaction graph plus a local update rule family plus an observation/intervention protocol.**

Traditional neural networks, recurrent networks, cellular automata, Boolean networks, complete graphs, and role-free systems then become different regions of the same architecture space.

That is the breadth-first search space going forward.