# Boolean Universe — Architecture BFS Checkpoint 3

This checkpoint continues the three-node breadth-first shell across four additional architecture axes: event-driven time semantics, reciprocal coupling, architecture/effective-law equivalence, and local gate family.

## 1. Event-driven asynchronous time

Architecture held fixed: complete 3-node directed graph with all self-loops. Local rules held fixed to affine GF(2). All `16^3 = 4096` rule configurations were exhausted.

Three update semantics were compared:

1. synchronous — all nodes update together from the old state;
2. fixed sequential sweep — node order `0 -> 1 -> 2`;
3. event-driven asynchronous — at each microstep, any node whose local rule disagrees with its current bit may update. This produces a transition relation rather than a deterministic global map.

For the asynchronous relation, terminal strongly connected components are the attractors.

Results:

- synchronous nontrivial-attractor fraction: ~0.7517;
- fixed-sequential nontrivial-attractor fraction: ~0.614;
- event-driven asynchronous cyclic-attractor fraction: ~0.5295;
- 43.63% of universes have more than one terminal asynchronous attractor;
- only 56.37% have one globally unique asynchronous attractor;
- 79.25% are confluent from each initial state, meaning each initial condition reaches only one terminal SCC despite scheduler freedom;
- 28.47% oscillate synchronously but have only fixed-point asynchronous attractors;
- 6.25% have no synchronous oscillatory attractor but do have an asynchronous cyclic attractor.

Thus update scheduling changes not only trajectories but the attractor ontology itself.

> **Scheduler sensitivity is a distinct dynamical property.**

In asynchronous universes, the question `what happens next?` is underdetermined by the Boolean state and local rules alone unless the update scheduler is also part of the state/law.

## 2. Reciprocal/undirected coupling skeletons

A reciprocal pair `{i,j}` was represented by both directed edges `i->j` and `j->i`, while self-loops remained independently optional.

For three nodes there are:

- 3 possible reciprocal pair-links;
- 3 possible self-loops;
- `2^6 = 64` labeled reciprocal/self-loop skeletons;
- exactly **20 unlabeled reciprocal topologies** after node relabeling.

All affine GF(2) local-rule configurations were exhausted for each topology under synchronous update.

Increasing reciprocal connectivity systematically increases recurrent richness. Mean behavior grouped by number of reciprocal pair-links shows rising oscillatory and reversible fractions.

The fully reciprocal complete graph with all self-loops has:

- 4096 rule configurations;
- 32.8125% reversible;
- ~75.17% with nontrivial cycles;
- maximum period 7.

However reciprocity alone does not guarantee reversibility. Self-memory via self-loops remains a strong independent factor.

So reciprocal coupling and self-feedback are separate architectural resources.

## 3. Architecture equivalence: implementation versus effective law

All **104 unlabeled directed 3-node topologies with self-loops** were swept using all compatible affine local rules under synchronous update.

Across topology representatives there are **35,688 topology+rule implementations**.

Yet they collapse to exactly:

> **4,096 distinct exact global transition laws.**

This is precisely the complete set of affine maps `x' = A x + b` on three Boolean bits, with `2^(9+3) = 4096` possibilities.

The complete 9-edge graph can realize all 4096, but most laws also have much sparser implementations.

### Architectural degeneracy

- median number of distinct unlabeled topologies realizing the same exact global law: 5;
- maximum: all 104 topologies;
- 75% of global laws are realizable by more than one topology;
- ~50.98% are realizable by at least five distinct topologies.

Thus wiring is generally **not identifiable from the effective global law**.

### Minimal wiring cost

Each affine global law has a well-defined minimum number of directed edges needed by any implementation in this architecture family.

Exact distribution:

- 0 edges: 8 laws
- 1 edge: 16
- 2 edges: 96
- 3 edges: 256
- 4 edges: 424
- 5 edges: 632
- 6 edges: 864
- 7 edges: 584
- 8 edges: 192
- 9 edges: 1024

So 25% of affine global laws genuinely require all nine possible dependencies in this three-node affine representation, whereas the simplest laws can survive enormous architectural redundancy.

Median possible edge redundancy — difference between densest and sparsest known implementation of a law — is 3 edges; maximum is 9.

Reversible laws require at least 3 edges. Their minimum-edge distribution begins:

- 3 edges: 24 reversible laws
- 4: 48
- 5: 192
- 6: 352
- 7: 264
- 8: 80
- 9: 384

This yields a clean distinction:

> **Architecture is implementation-level ontology; the global transition map is effective-law ontology.**

Different implementations may be physically distinguishable only if the observer can access internal nodes, interventions, timing, or other structure beyond the global state transition law.

## 4. Local gate family as an independent physics axis

The architecture was fixed again to the complete 3-node directed graph with all self-loops and synchronous update. Only the local Boolean function family changed.

Families:

- affine GF(2): 16 local functions, all 4,096 universes exhausted;
- monotone Boolean: 20 local functions, all 8,000 universes exhausted;
- threshold: 104 local functions, 160,000 uniformly sampled universes;
- unrestricted Boolean: 256 local functions, 160,000 uniformly sampled universes.

Observed dynamical statistics:

| Family | reversible | period >1 | period >=3 | period >=4 | max observed period | transient |
|---|---:|---:|---:|---:|---:|---:|
| affine GF(2) | 0.328125 | ~0.7517 | ~0.4971 | ~0.3330 | 7 | ~0.5315 |
| monotone | 0.00075 | ~0.0799 | 0.00375 | 0 | 3 | ~0.7788 |
| threshold | ~0.000194 | ~0.6485 | ~0.2100 | ~0.0778 | 8 | ~0.7597 |
| unrestricted | ~0.002619 | ~0.7155 | ~0.3717 | ~0.1603 | 8 | ~0.7153 |

The monotone result is exact. Of 8,000 monotone universes:

- 7,361 have longest period 1;
- 609 have longest period 2;
- 30 have longest period 3;
- none have period >=4;
- only 6 are reversible.

This is not an accident: monotone global maps preserve the partial order of the 3-cube, so points on a nontrivial periodic orbit must form an antichain. The Boolean 3-cube has maximum antichain size 3, yielding a structural upper bound of period 3.

Thus a restriction on the **local computational primitive** induces a theorem-level global dynamical law.

> **Architecture can constrain physics before any particular weights/rules are selected.**

## 5. Updated ontology of tiny network universes

At three nodes, the following are now empirically distinct axes:

1. directed graph topology;
2. reciprocal versus one-way coupling;
3. self-feedback;
4. local rule family;
5. synchronous, fixed sequential, or event-driven time semantics;
6. permanent versus operational/role-free ports;
7. implementation architecture versus effective global law;
8. minimal wiring complexity of an effective law.

Several strong principles have emerged:

- **Time semantics is physics:** changing only the scheduler can change the attractor set.
- **Roles need not be node types:** input/output/hidden can emerge from observer coupling.
- **Architecture is non-identifiable from behavior in general:** many topologies implement the same global law.
- **Minimal architecture can still be meaningful:** each effective law has a minimum dependency cost.
- **Local algebra constrains global dynamics:** monotonicity forbids long cycles; affine structure strongly favors reversible maps.
- **More local expressivity is not monotonically more structured:** unrestricted and threshold ensembles can be much less reversible than affine ones.

## Next breadth-first shell

Remain at three nodes for one more checkpoint before making four nodes the main frontier.

Highest-value axes now are:

1. **time-varying and stochastic schedulers** — scheduler as a stateful subsystem rather than an external convention;
2. **multiple actuators/sensors** — observer coupling dimension as an architecture axis;
3. **time-varying ports** — can the observer reassign node roles during a trajectory;
4. **NAND-only / NOR-only constructive bases** — instead of allowing an entire rule family per node, constrain all nodes to one universal primitive and let topology/depth do the work;
5. **residual/skip connections** in tiny layered threshold networks;
6. **causal identifiability** — determine when internal architecture can be recovered from interventions even though passive global behavior cannot identify it;
7. **effective-law quotient across update semantics** — identify when different graph+rule+scheduler triples are observationally equivalent;
8. **first four-node shell** only after the above axes are summarized.
