# Boolean Universe — Memory Architecture Checkpoint 2

This checkpoint advances to the first direct point on the proposed diagonal `memory bits per neuron = neuron count`:

- `N=2` public neurons
- `k=2` private memory bits per neuron
- total physical Boolean state = `2*(1+2)=6` bits = 64 global states.

The purpose is to compare equal six-bit state budgets while changing where the state lives and who may read it.

## Matched six-bit architectures

1. **6 public bits** — six public state neurons, unrestricted complete update.
2. **2 neurons × 2 local private bits** — each neuron sees both public outputs plus only its own two private bits.
3. **2 neurons × 2 shared-readable private bits** — same state allocation, but both neurons may read all six bits; private bits remain hidden only from the external observer.
4. **1 public + 5 private** — all six state bits live inside one computational unit.

## Exact autonomous law-space sizes

For an unrestricted deterministic `B`-bit state machine the arbitrary-map exponent is `B * 2^B`.

Thus the fully state-readable six-bit architectures have

`64^64 = 2^384`

autonomous laws.

The local-private `N=2,k=2` architecture is far smaller. Each node:

- sees 4 input bits: `y1,y2` plus its own two private memories;
- outputs 3 next-state bits: its public output plus its two private memories;
- therefore has `2^(3*16) = 2^48` local laws.

Two independent nodes give

`2^96`

global laws.

So equal six-bit physical state budgets give:

- 6 public: `2^384`
- 2 × (1 public + 2 local private): `2^96`
- 2 × (1 public + 2 shared-readable private): `2^384`
- 1 public + 5 private: `2^384`

This is a fourfold reduction in the **law-space exponent** caused solely by private-memory communication constraints.

> **Six Boolean state bits are not six equivalent computational resources.**

## Observer-effective state

6,000 uniformly sampled laws per architecture were analyzed by Moore-style behavioral partition refinement: two internal states are considered equivalent if they produce the same entire future public-output sequence.

Mean number of behaviorally distinguishable states out of 64:

- 6 public: `64.0`
- `N=2,k=2` local private: `~34.21`
- same memory but shared-readable: `~54.89`
- 1 public + 5 private: `~41.11`

Thus making hidden memory internally readable substantially increases how much internal state leaks into public behavior, even though observer bandwidth is unchanged.

Full distinguishability of all 64 states was very rare under hidden-state architectures in this sample:

- local-private: ~0.00017
- shared-readable: ~0.00017
- one-public/five-private: none in 6,000 samples.

## Public Markov property

Exact probabilities that the current public state alone determines the next public state:

- 6 public: `1`
- `N=2,k=2` local private: `2^-24`
- shared-readable private memory: `2^-120`
- 1 public + 5 private: `2^-62`

Thus on the proposed memory-heavy diagonal, the observer almost always sees a strongly non-Markovian process unless hidden state is exposed.

## Typical autonomous dynamics

Mean longest cycle in 6,000 sampled laws:

- 6 public full-map ensemble: ~6.38
- `N=2,k=2` local private: ~4.71
- shared-readable private: ~6.38
- 1 public + 5 private: ~6.34

The shared-readable and concentrated-memory architectures recover the same typical cycle scale as generic six-bit maps. Local-private memory produces shorter typical clocks because the global transition map factorizes through locality constraints.

## Emerging principle

The memory architecture now has three separable quantities:

1. **state amount** — total bits physically stored;
2. **state ownership** — which neuron owns which bits;
3. **state readability** — which other neurons can condition their updates on those bits.

Increasing private memory without increasing memory connectivity can dramatically enlarge raw state space while leaving the realizable law class highly factorized.

Conversely, allowing hidden memory to be read across neurons can recover full global dynamical expressivity without making that memory externally visible.

This suggests a useful generalized architecture:

`public-signal graph + memory-read graph + memory-update law + observer-port graph`.

The next step should not jump immediately to `N=3,k=3`. First compare several constrained memory-update mechanisms at `N=2,k=2`:

- literal two-step self-history shift registers;
- neighbor-history registers;
- per-edge delay memory;
- arbitrary private FSM memory;
- shared scratchpad memory;
- selectively readable memory bits.

That will isolate which benefits come from **having more past**, which come from **arbitrary writable state**, and which come from **communication access to memory**.