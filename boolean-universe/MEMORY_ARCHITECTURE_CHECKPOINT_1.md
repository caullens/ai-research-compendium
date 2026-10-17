# Boolean Universe — Memory Architecture Checkpoint 1

This branch asks a new question suggested by the earlier delay experiments:

> If memory is as important as neuron count, what changes when the same number of Boolean state bits is placed **inside neurons** rather than distributed across more public neurons?

The key design principle is to separate three resources that ordinary neural-network language tends to conflate:

1. **state capacity** — how many Boolean degrees of freedom physically exist;
2. **communication visibility** — which state bits other neurons may read;
3. **observer visibility** — which state bits an external scientist can see or manipulate.

A memory-bearing Boolean neuron therefore has one public Boolean output plus private Boolean state.

---

## 1. Two total state bits: 2 public neurons vs 1 public + 1 private

Compare:

- Architecture A: two public memoryless Boolean neurons;
- Architecture B: one Boolean neuron with one public bit `y` and one private memory bit `m`.

With unrestricted Boolean update logic, both architectures can realize **every deterministic map on four states**:

`F:{0,1}^2 -> {0,1}^2`.

Thus both have exactly

`4^4 = 256`

autonomous dynamical laws and exactly the same cycle/attractor census.

This gives the first exact equivalence:

> **At fixed total state, a private memory bit is dynamically equivalent to an additional public neuron when all state is locally available to the computation updating it.**

The difference appears only to the observer.

For the `1 public + 1 private` architecture, exhaustive enumeration of all 256 laws gives:

- 64 laws whose public output distinguishes only 2 behavioral states;
- 96 with 3 behaviorally distinguishable states;
- 96 with all 4 internal states eventually distinguishable from output history.

Therefore hidden state is not synonymous with permanently unknowable state: in **37.5%** of all laws, future public output history eventually reveals the private bit.

However, only **25%** of laws have a public process that is first-order deterministic in the visible bit alone. In the other 75%, two identical visible states can have different visible futures because private memory differs.

So memory introduces genuine observer-level non-Markovianity without changing autonomous state capacity.

---

## 2. Four total state bits: placement begins changing the law space

Matched total state budget: 4 Boolean bits / 16 physical global states.

Architectures:

1. **4 public** — four fully public state neurons, complete access;
2. **2 public + 2 private local** — two neurons, each with one private bit; each neuron reads both public bits and only its own private bit;
3. **2 public + 2 private shared-readable** — same state placement, but each neuron may also read the other neuron's private bit;
4. **1 public + 3 private** — one neuron owns all four state bits.

Exact autonomous law-space sizes:

- 4 public: `2^64`
- 2+2 private local: `2^32`
- 2+2 shared-readable: `2^64`
- 1+3 private: `2^64`

This is a major result:

> **Memory amount alone does not determine computational capacity. Memory connectivity does.**

The two-neuron local-private architecture loses half the law-space exponent because each node's update cannot condition on the other node's hidden state.

By contrast, both concentrating all memory in one neuron and making distributed memory mutually readable recover the full `2^64` arbitrary 16-state law class.

### Observer-effective dynamics

Monte Carlo over 30,000 uniformly sampled laws from each architecture gave mean numbers of behaviorally distinguishable internal states from the public output stream:

- 4 public: 16.0 / 16
- 2+2 private local: ~10.49
- 2+2 shared-readable: ~14.02
- 1+3 private: ~10.44

Fractions where all 16 hidden states are eventually behaviorally distinguishable:

- 4 public: 1.0
- 2+2 private local: ~0.046
- 2+2 shared-readable: ~0.156
- 1+3 private: ~0.0075

Exact probabilities that the **public** state alone remains first-order deterministic:

- 4 public: 1
- 2+2 private local: `1/256`
- 2+2 shared-readable: `2^-24`
- 1+3 private: `1/16384`

Thus richer hidden-state connectivity simultaneously expands autonomous law capacity and makes the public world more strongly history-dependent.

---

## 3. Memory-read graph: a second network topology

For two public neurons with one private bit each, ordinary public signaling was held fully connected while cross-memory readability was varied.

Each neuron always reads its own memory. Two additional possible **memory-read edges** exist:

- node 1 may read node 2's private memory;
- node 2 may read node 1's private memory.

Exact law-space exponents:

- no cross-memory reads: `2^32`
- one-way cross-memory read: `2^48`
- opposite one-way read: `2^48`
- both cross-memory reads: `2^64`

Each cross-memory edge therefore adds a large new set of conditional contexts without adding a single state bit.

30k-law samples show the mean observer-effective quotient rises correspondingly:

- private: ~10.53 states
- one-way: ~11.95–11.97
- fully shared-readable: ~14.01

Exact public first-order-deterministic fractions fall sharply:

- private: `1/256`
- one-way: `1/65536`
- fully shared-readable: `2^-24`

This suggests the correct generalized primitive is not one graph but a **multiplex architecture**:

- signal graph over public outputs;
- memory-read graph over private states;
- potentially a separate memory-write graph later.

---

## 4. Literal one-step history rather than arbitrary writable memory

Arbitrary private memory can hide a full finite-state machine inside each neuron, so a stricter architecture was tested:

`h_i(t+1) = y_i(t)`.

The memory bit has one fixed semantic: the neuron's previous public output.

Two public neurons + one history bit each give four physical state bits. Only the visible output rules are programmable.

Four access architectures were compared:

1. no history access: `f_i(y1,y2)`
2. own history: `f_i(y1,y2,h_i)`
3. neighbor history: each neuron reads the other neuron's history
4. both histories: `f_i(y1,y2,h1,h2)`

Exact configuration counts:

- no history: `2^8 = 256`
- own history: `2^16 = 65,536`
- neighbor history: `2^16 = 65,536`
- both histories: `2^32` configurations (sampled)

### Exact dynamical results

No history:

- reversible laws: 0 / 256
- maximum period: 4
- oscillatory fraction: ~0.512

Own history:

- reversible: 256 / 65,536 = `1/256`
- maximum period: 12
- oscillatory fraction: ~0.833

Neighbor history:

- reversible: 256 / 65,536 = `1/256`
- maximum period: **15**
- oscillatory fraction: ~0.723

Both histories, 80k uniform sample:

- maximum period observed: 15
- oscillatory fraction: ~0.858

The surprising result is that **where the same history bits are read matters**. Remembering the neighbor's past reaches the full nonzero 4-bit cycle ceiling of 15, whereas remembering only one's own past maxes at 12.

This is obtained without arbitrary memory writing; the history mechanics are fixed.

> **Memory placement is an independent architecture axis even when state amount, rule-table size, and memory update semantics are held fixed.**

---

## 5. Revised architecture primitive

A useful memory-bearing Boolean network should now be described by at least:

1. public neuron state bits;
2. private memory bits per neuron;
3. public signal graph;
4. memory-read graph;
5. memory-write/update law;
6. external observation/intervention ports;
7. update scheduler.

This is richer than simply attaching `k` hidden bits to every neuron.

### Recommended next shell

Before trying `k=N` at large N, scale carefully through:

- `N=1, k=1` — already equivalent to a generic 2-bit autonomous dynamical system;
- `N=2, k=1` — four-bit matched-budget shell above;
- `N=2, k=2` — the first point on the proposed memory-heavy diagonal with six total state bits;
- `N=3, k=3` only after the two-neuron case is understood.

For `N` neurons with `k` private bits each, all public outputs mutually visible but private memory strictly local, the exact unrestricted local-law exponent is

`N (k+1) 2^(N+k)`.

If all private memory becomes mutually readable, the full arbitrary-map exponent is

`N (k+1) 2^(N(k+1))`.

On the proposed diagonal `k=N`, this gap grows extremely quickly. Therefore the central question is not merely how much memory each neuron receives, but **how private memory is allowed to participate in network computation**.
