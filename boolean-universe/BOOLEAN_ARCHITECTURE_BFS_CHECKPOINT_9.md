# Boolean Universe — Architecture BFS Checkpoint 9

This checkpoint begins the five-node shell using theorem-guided breadth rather than mechanically repeating every four-node experiment. The emphasis is on capabilities that four nodes provably cannot realize and on representative architectural cross-sections that expose independent scaling mechanisms.

## 1. Exact monotone period-10 clock

For monotone Boolean maps on the n-cube, every periodic orbit is an antichain, so the maximum possible period is bounded by the Sperner width `C(n, floor(n/2))`.

At n=4 that ceiling is 6 and an explicit period-6 construction was already verified.

At n=5 the weight-2 layer contains 10 mutually incomparable states. A cyclic ordering of those ten states was prescribed:

`00011 -> 00101 -> 00110 -> 01001 -> 01010 -> 01100 -> 10001 -> 10010 -> 10100 -> 11000 -> 00011`.

Each coordinate rule was extended monotonically to the whole Boolean lattice by upward closure of the antichain states assigned output 1.

All five coordinate functions were verified monotone and the orbit is exactly period 10.

Therefore the monotone ceiling jump `6 -> 10` is realizable, not merely a combinatorial upper bound.

## 2. Exact affine period-31 clock

A five-bit affine/XOR shift-register world was searched for a maximal nonzero cycle. An exact period-31 LFSR was found using feedback taps `(q4,q2)` in the indexing `q4..q0 = 0..4`:

`(q4,q3,q2,q1,q0)' = (q3,q2,q1,q0,q4 XOR q2)`.

Starting from `00001`, all 31 nonzero five-bit states are visited exactly once before return.

Thus the affine reversible clock threshold grows:

- n=3: 7
- n=4: 15
- n=5: 31

The exact reversible affine ensemble size at n=5 is also enormous:

`|GL(5,2)| = 9,999,360`

and including all 32 translations yields

`319,979,520`

reversible affine global maps.

## 3. Homogeneous local ring physics

A periodic nearest-neighbor ring with one shared local rule

`x_i' = f(x_{i-1},x_i,x_{i+1})`

was exhaustively tested for all 256 elementary 3-input Boolean rules at ring sizes n=3,4,5.

Exact summary:

| Ring size | Reversible shared rules | Period>1 fraction | Max period |
|---:|---:|---:|---:|
| 3 | 36 | 0.683594 | 6 |
| 4 | 8 | 0.761719 | 8 |
| 5 | 16 | 0.773438 | 30 |

At n=5, rules 45, 75, 89 and 101 achieve period 30.

This is especially important because nothing about the local physics changed: the same rule is copied at every site. Merely increasing universe size from 4 to 5 changes the available global clock spectrum dramatically.

Some shared rules are highly size-sensitive. For example, rules 45, 75, 89 and 101 have longest period 3 on a 3-ring, 2 on a 4-ring, and 30 on a 5-ring.

So system size is not merely extra storage; it interacts nontrivially with local translationally shared law.

## 4. Exact identical-gate recurrent census

Every neuron was forced to use the same binary primitive. Each node chooses an unordered pair of source nodes with repetition. At n=5 there are `C(6,2)=15` possible source pairs per node and therefore

`15^5 = 759,375`

wiring architectures per primitive.

All wirings were exhaustively evaluated for six primitives.

| Primitive | Reversible fraction | Period>1 fraction | Max period |
|---|---:|---:|---:|
| AND | ~0.000158 | ~0.1676 | 6 |
| OR | ~0.000158 | ~0.1676 | 6 |
| NAND | ~0.000158 | 1.0000 | 10 |
| NOR | ~0.000158 | 1.0000 | 10 |
| XOR | 0 | ~0.5817 | 15 |
| XNOR | 0 | ~0.5817 | 15 |

Every NAND-only and NOR-only five-node recurrent architecture oscillates. XOR/XNOR topology reaches period 15 even though every node uses the same fixed local primitive and no custom Boolean rule is allowed.

This confirms that the fifth-node gains are not confined to complete graphs with node-specific laws.

## 5. One-step delay produces an exact 1023-cycle

Five physical Boolean neurons with delayed affine coupling:

`x(t+1) = A x(t) XOR D x(t-1)`

have a ten-bit Markov state `z=(x_t,x_{t-1})` with block matrix

`C = [[A,D],[I,0]]`.

As before, `C` is invertible iff `D` is invertible.

A random search over invertible D and arbitrary A found a block-companion matrix of exact order

`2^10 - 1 = 1023`.

Explicit matrices:

A =
```
1 0 1 1 0
0 1 0 1 1
0 1 1 1 0
1 0 0 0 1
1 0 1 0 0
```

D =
```
0 0 1 0 1
0 1 1 1 1
1 1 1 0 1
1 0 1 1 0
1 0 1 0 0
```

`rank(D)=5` and the 10x10 extended matrix was verified to have exact multiplicative order 1023. Starting from a nonzero seed produces one orbit through all 1023 nonzero extended states.

Thus:

- five-node no-delay affine ceiling: 31
- five-node + one-step delay: 1023

Memory placement outside the instantaneous node state remains one of the strongest architectural resources found in the entire Boolean program.

## 6. Five-node observer coupling: time versus port bandwidth

For random five-bit linear worlds `x'=Ax`, state observability was tested using persistent physical sensor subsets. Full observability is the rank condition on

`[C; CA; CA^2; ...]`.

By finite-field duality, the same distribution applies to controllability with the same number of persistent actuator nodes.

Monte Carlo over 120,000 random matrices per port count gave:

### One sensor/actuator

Full observability/controllability occurs only at the full five-step horizon and for ~30.7% of random worlds.

### Two ports

- 3 steps: ~49.2%
- 4+ steps: ~61.5%

### Three ports

- 2 steps: ~65.6%
- 3+ steps: ~82.0%

### Four ports

- 2+ steps: ~93.8%

A one-port scaling audit gives approximate full observability/controllability probabilities by n steps:

- n=3: ~0.378
- n=4: ~0.326
- n=5: ~0.308

So a single persistent physical port remains surprisingly useful as Boolean state dimension grows, while time can compensate strongly for limited bandwidth.

## 7. What the fifth node actually adds

The five-node shell exposes multiple qualitatively different scaling mechanisms:

1. **Order-theoretic scaling:** monotone period ceiling follows antichain width, `6 -> 10`.
2. **Finite-field/algebraic scaling:** maximal affine nonzero cycle follows `2^n-1`, `15 -> 31`.
3. **Finite-size/locality scaling:** a homogeneous shared-rule ring jumps from max period 8 at n=4 to 30 at n=5.
4. **Topology-only computation:** identical fixed primitives gain new clock capacities solely from additional wiring freedom.
5. **Memory-location scaling:** one-step delay changes effective state from n to 2n bits and permits `31 -> 1023`.
6. **Observer scaling:** sensor/actuator count and observation time trade against one another rather than simply tracking node count.

This reinforces the program's central thesis:

> A new Boolean neuron is not one scalar unit of complexity. Depending on architecture, it can enlarge state cardinality, antichain width, algebraic group order, local finite-size resonances, wiring combinatorics, hidden temporal state and scientific accessibility in different ways.

## Next breadth frontier

The next pass should continue theorem-guided breadth rather than brute-force the 291,968 unlabeled five-node directed topologies.

High-value cross-sections:

1. five-node stochastic/asynchronous scheduler scaling;
2. five-node stateful/plastic edges;
3. five-node layered threshold networks under equal neuron budget;
4. observer-effective equivalence under hidden ports and changing ports;
5. topology cycle-cover statistics by structured graph families rather than all graphs;
6. delay depth >1 as an explicit memory-order axis;
7. identify architectural properties that have predictable n-scaling laws and separate them from finite-size accidents.
