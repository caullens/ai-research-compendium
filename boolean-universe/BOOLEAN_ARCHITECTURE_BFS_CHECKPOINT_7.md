# Boolean Universe — Architecture BFS Checkpoint 7

This checkpoint deliberately broadens the four-node shell across architecture axes rather than deepening one family. It covers update semantics, scheduler state, stochastic scheduling, identical-gate recurrent networks, complete-graph rule families, same-budget layered networks, partial observers, rule sharing/locality, the full directed topology census, higher-order Boolean interactions, delayed state, stateful edges, output noise, and active scientific query complexity.

## 1. Time/update semantics

Four-node complete affine local laws were sampled under four time semantics:

- synchronous update;
- fixed sequential sweep `0->1->2->3`;
- endogenous deterministic scheduler, where two affine selector bits of current state choose which one node updates;
- event-driven asynchronous relation, where any unstable node may update.

30,000-law sample:

- synchronous nontrivial-attractor fraction: ~0.8712
- fixed sequential: ~0.7415
- endogenous scheduler: ~0.3308
- event-driven asynchronous terminal cycle: ~0.5142

Reversible fractions among deterministic macro-laws:

- synchronous: ~0.3088
- sequential: ~0.0617
- endogenous single-site scheduler: ~0.0158

Event-driven systems are confluent from every initial state only ~72.5% of the time and have multiple terminal attractors ~41.2% of the time.

~57.6% of sampled laws oscillate synchronously but become fixed-only under the endogenous one-node scheduler.

**Conclusion:** scheduler architecture is part of the law. It changes attractor ontology, reversibility and clock spectrum while topology and local Boolean functions remain fixed.

## 2. Stochastic schedulers

Four-node affine laws were also sampled under:

1. random single-site scheduler — choose one node uniformly each microstep;
2. random full-sweep scheduler — choose one of 24 node orders uniformly, then update all nodes sequentially.

4,000-network sample:

- random single-site: one recurrent Markov class ~0.571; ergodic ~0.571
- random full sweep: one recurrent class ~0.5635; ergodic ~0.4145

The recurrent-class counts are similar, but periodicity/ergodicity differs strongly.

**Conclusion:** where stochasticity enters the scheduler matters independently of whether scheduling is stochastic at all.

## 3. Identical primitive gates

For n=3 and n=4, every computational node was forced to use the same binary primitive. Each node chooses an unordered source pair with repetition; only wiring varies.

At n=4 there are exactly `C(5,2)^4 = 10,000` wiring architectures per primitive.

Exact n=4 results:

| Primitive | Reversible fraction | Period >1 | Max period |
|---|---:|---:|---:|
| AND | 0.0024 | 0.1642 | 4 |
| OR | 0.0024 | 0.1642 | 4 |
| NAND | 0.0024 | 1.0000 | 8 |
| NOR | 0.0024 | 1.0000 | 8 |
| XOR | 0 | 0.4312 | 7 |
| XNOR | 0 | 0.4312 | 7 |

Every NAND-only and NOR-only recurrent wiring oscillates. The fourth node raises NAND/NOR max period from 6 to 8 and XOR/XNOR from 3 to 7.

**Conclusion:** zero node-type diversity does not imply trivial dynamics; topology plus primitive algebra is enough to define distinct universe families.

## 4. Complete-graph local rule families

Architecture fixed: complete four-node directed graph with self-loops, synchronous update.

Local rule sets:

- affine GF(2): 32 functions/node, all `32^4 = 1,048,576` global networks exhausted;
- monotone: all 168 local functions, 120k networks sampled uniformly over local-function tuples;
- threshold: exact local enumeration stabilized at 1,882 distinct 4-input threshold functions, 120k networks sampled;
- unrestricted: 65,536 functions/node, 120k networks sampled.

Results:

| Family | Reversible | Period >1 | Period >=4 | Period >=8 | Max observed |
|---|---:|---:|---:|---:|---:|
| affine | 0.307617 exact | 0.870131 exact | 0.586578 | 0.129822 | 15 |
| monotone | 0 in sample | 0.111475 | 0.000483 | 0 | 5 sampled; explicit period-6 construction exists |
| threshold | 0 in sample | 0.756692 | 0.175167 | 0.004258 | 11 |
| unrestricted | ~8e-6 in sample | 0.843725 | 0.370867 | 0.025717 | 14 |

Affine/additive physics is an exceptional reversible regime, not merely the simplest point on a smooth complexity spectrum.

## 5. Higher-order interactions / Boolean hyperedges

Local rules were grouped by algebraic normal form degree:

- degree 0: constants, 2 local functions
- degree <=1: affine, 32
- degree <=2: 2,048
- degree <=3: 32,768
- degree <=4: unrestricted, 65,536

Complete four-node synchronous graph, 100k samples per nontrivial family.

Reversible fractions:

- degree <=1: ~0.30765
- degree <=2: ~6e-5
- degree <=3: ~1e-5
- degree <=4: 0 in 100k sample

Oscillatory fractions remain near ~0.84 for degrees >=2, but the affine reversible regime collapses almost immediately.

Heterogeneous profiles show the transition sharply: one quadratic node among three affine nodes drops reversibility to ~0.0387; two quadratic nodes to ~0.0051.

**Conclusion:** interaction order is an architecture axis distinct from fan-in. Higher-order local terms act like Boolean hyperedges and radically change global conservation/reversibility statistics.

## 6. Exact same-budget layered threshold architectures

External task: 3 Boolean inputs -> 1 output. Four threshold computational neurons total. Exactly 256 possible external laws.

Exact capacities:

- `3->3->1`: 256 / 256
- `3->2->1->1`: 254 / 256
- `3->1->2->1`: 104 / 256
- `3->1->1->1->1`: 104 / 256
- skip-final (`3->2->1->1`, output also sees one raw input): 254 / 256
- residual-middle (middle sees hidden pair plus one raw input): 256 / 256

The only two laws missing from `3->2->1` and skip-final are 3-bit XOR and XNOR.

A one-bit early bottleneck collapses the class exactly to the 104 linearly separable 3-input threshold laws. Re-expanding afterward cannot recover lost distinctions.

A residual raw-input path restores all 256 laws under the same four-neuron budget.

**Conclusion:** depth, width, bottleneck placement and skip/residual paths are separate resources. Depth by itself does not restore information erased by an early bottleneck.

## 7. Rule sharing and locality

Four-node periodic ring.

Exact architectures:

- shared radius-1 rule `f(left,self,right)`, all 256 rules;
- shared directed rule `f(self,right)`, all 16 rules;
- shared affine radius-1 subset, all 16 rules;
- independent affine radius-1 rule per site, all `16^4=65,536` networks.

Results:

| Architecture | Reversible | Period >1 | Max period |
|---|---:|---:|---:|
| shared radius-1 | 0.03125 | 0.761719 | 8 |
| shared directed | 0.25 | 0.4375 | 8 |
| shared affine | 0.50 | 0.4375 | 4 |
| independent affine ring | 0.240234 | 0.848495 | 15 |

Eight shared radius-1 rules are reversible on ring size 4: rules 15, 51, 85, 105, 150, 170, 204 and 240.

**Conclusion:** homogeneous shared physics is a genuine architecture prior distinct from locality itself.

## 8. Full four-node directed topology census

All `2^16 = 65,536` labeled directed graph masks with self-loops were quotiented under node relabeling.

Result:

> **3,044 unlabeled directed topologies.**

Maximum structural GF(2) rank distribution:

- rank 0: 1
- rank 1: 12
- rank 2: 163
- rank 3: 1,132
- rank 4: 1,736

Exactly 1,736 can support at least one reversible affine law.

The criterion is exact:

> A topology supports some invertible affine law iff its directed support contains a cycle cover, equivalently a permutation-matrix support.

Strong connectedness is different:

- strongly connected: 1,168
- strongly connected but no cycle cover: 68
- cycle cover but not strongly connected: 636

So global communication connectivity and information-preserving capacity are distinct topological resources.

## 9. Observer-limited effective-law equivalence

All 1,048,576 affine four-bit laws were materialized exactly.

Protocol:

- reset initial state to any of all 16 states;
- evolve autonomously;
- observe only first m labeled state bits for T future steps.

Exact uniquely identified fractions:

### m=1 visible bit

- T=1: 0
- T=2: 0
- T=3: 0
- T=4: 0.328125

### m=2

- T=1: 0
- T=2: 0.375
- T=3: 0.65625
- T=4: 0.65625

### m=3

- T=1: 0
- T=2: 0.875
- T=3: 0.875
- T=4: 0.875

Thus complete control of initial conditions does not imply complete causal identification when channels are hidden.

## 10. Sensor continuity versus mobility

With exactly one observed bit per time for four future steps, all 15 sensor-schedule equality patterns were tested exactly across all affine laws.

Counterintuitively, fixed observation is best:

- fixed same node `0000`: uniquely identifies 0.328125 of laws
- switch once late `0111`: ~0.2058
- rotate all four `0123`: ~0.1104

Temporal continuity of one channel carries more system-identification information than sampling several channels only once.

**Conclusion:** sensor mobility is not the same resource as bandwidth; continuity can be more valuable than coverage.

## 11. Delay as hidden state

Four Boolean neurons with one-step delayed affine coupling:

`x(t+1) = A x(t) XOR D x(t-1) XOR b`.

The Markov state is eight bits `(x(t),x(t-1))` even though there are still four neurons.

Extended matrix:

`C = [[A,D],[I,0]]`.

Algebraically, C is invertible iff D is invertible. Therefore the exact reversible fraction for uniformly random A,D,b is

`20160/65536 = 0.3076171875`.

A reversible delayed system was found with cycle partition `(1,255)`: one fixed state and one orbit through every other extended state.

No-delay four-bit affine dynamics maxed out at period 15.

**Conclusion:** where state lives matters as much as neuron count. Delay can create exponentially richer hidden state and clocks without adding nodes.

## 12. Stateful edge / plastic architecture

Smallest explicit architecture-as-state model:

state `(x,y,g)` where g records whether edge `x->y` is active.

Node law:

- `x'=x`
- `y'=y XOR (g AND x)`

Edge law:

- `g'=h(x,y,g)`
- all 256 Boolean h exhausted.

Results:

- reversible: 16/256 = 0.0625
- longest-period distribution: 1: 63 laws, 2: 113, 3: 64, 4: 16
- 68.36% have transients
- 80 plastic edge laws create periods longer than all static-edge baselines

**Conclusion:** allowing architecture itself to carry state is already a genuinely new computational/dynamical family at the smallest scale.

## 13. Output noise / stochastic nodes

Deterministic four-bit affine core followed by independent output-bit flips with probability epsilon.

For every `0<epsilon<1`, all state-to-state transition probabilities are positive, so the 16-state Markov chain is irreducible and aperiodic with a unique stationary distribution.

If the deterministic core is reversible/permutational, the noisy transition matrix is doubly stochastic, hence stationary distribution is exactly uniform: entropy = 4 bits for every epsilon.

Sample of 160 reversible and 160 irreversible affine cores:

- epsilon .001: irreversible mean stationary entropy ~2.07 bits; reversible exactly 4
- epsilon .01: irreversible ~2.35; reversible exactly 4
- epsilon .10: irreversible ~3.44; reversible exactly 4
- epsilon .25: irreversible ~3.88; reversible exactly 4
- epsilon .50: both exactly 4

**Conclusion:** deterministic reversibility leaves an equilibrium/statistical signature even after noise erases hard attractor separation.

## 14. Universal active-query complexity by rule family

Complete four-node network. One intervention query selects a 4-bit input state and observes all four outputs, evaluating all four local rules simultaneously.

Question: what is the smallest fixed set Q of input states such that every pair of local functions in a family differs on at least one q in Q?

Exact results:

| Local family | Number of local functions | Information lower bound | Minimum universal query set |
|---|---:|---:|---:|
| affine | 32 | 5 | 5 |
| monotone | 168 | 8 | 16 |
| threshold | 1,882 | 11 | 16 |
| unrestricted | 65,536 | 16 | 16 |

Affine is exceptional: `{0000,0001,0010,0100,1000}` identifies every affine local rule and therefore every complete affine network.

For monotone and threshold families, **every one of the 16 input states is necessary in a single universal worst-case query set**, despite their hypothesis spaces being much smaller than unrestricted Boolean functions.

This is because for every omitted Boolean input state there exist two functions in these classes that agree everywhere else and differ only there.

**Conclusion:** hypothesis-space cardinality alone does not determine scientific sample complexity. The geometric/combinatorial arrangement of hypotheses matters.

## Architecture ontology after the four-node breadth sweep

At least 27 independent resources have now been separated:

1. node count / state-space cardinality
2. directed topology
3. reciprocity
4. self-feedback
5. fan-in / sparsity
6. local rule family
7. local interaction order
8. node heterogeneity
9. rule/weight sharing
10. locality / neighborhood geometry
11. feed-forward depth
12. hidden width
13. bottleneck placement
14. skip/residual paths
15. recurrence
16. update semantics
17. scheduler state/endogeneity
18. stochastic scheduler placement
19. fixed versus role-free ports
20. observer bandwidth
21. sensor continuity versus mobility
22. passive versus active intervention
23. implementation equivalence
24. minimal wiring complexity
25. delayed state
26. stateful edges / plastic topology
27. stochastic/noisy outputs

The central conclusion of the breadth-first program is now stronger:

> **Network architecture is not one variable. It is a product space describing where state lives, which dependencies exist, what local computations are legal, how time is defined, how rules are shared, whether architecture itself changes, and how an observer couples to the system.**

The four-node shell is broad enough that moving to five nodes should now be selective: only phenomena that fundamentally require larger state count or interaction order should justify the combinatorial cost.