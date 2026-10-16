# Boolean Universe — Ground-Up Baseline

This branch intentionally resets the universe program to the smallest exhaustively enumerable systems.

## Rule of the program

Increase complexity by exactly one explicit brick at a time. Do not import geometry, continuity, spacetime, metrics, dynamics, hidden dimensions, or physics unless a later system forces them to appear.

A Boolean neuron is initially defined at the most assumption-light level as an arbitrary Boolean function of its Boolean inputs. We do **not** begin with real weights, thresholds, sigmoids, or training.

## Brick 0: one Boolean input -> one Boolean output

The complete function class `F:{0,1}->{0,1}` contains exactly four universes:

| Law | F(0) | F(1) | Preserves input distinction? |
|---|---:|---:|---|
| ZERO | 0 | 0 | no |
| ID | 0 | 1 | yes |
| NOT | 1 | 0 | yes |
| ONE | 1 | 1 | no |

There is no hidden structure to infer.

The two constant universes collapse both input locations into one observed state. ID and NOT preserve the distinction between the two inputs. At this level, ID and NOT differ only by relabeling the two output symbols.

Thus, if Boolean labels themselves are treated as non-ontological, there are only **two structural classes**:

1. distinction-preserving bijection;
2. distinction-collapsing constant map.

This is the absolute reference floor.

## Brick 1: one hidden Boolean node

Architecture:

`x -> h(x)=z -> o(z)=y`

Both `h` and `o` are arbitrary unary Boolean functions, so there are `4 x 4 = 16` complete internal implementations.

Exhaustive enumeration shows that the external composite law remains one of the same four unary Boolean functions:

- ZERO: 6 implementations
- ID: 2 implementations
- NOT: 2 implementations
- ONE: 6 implementations

Therefore:

> One hidden unary Boolean node creates internal implementation multiplicity but **no new externally observable laws**.

This is the first clean separation between external law and hidden mechanism.

For the constant external laws, some implementations preserve the input distinction in the hidden node and erase it only at the output, while others erase it immediately. Those mechanisms are observationally identical from the final output alone.

So hidden ontology can exist before new external behavior exists.

## Brick 2: one input -> two hidden bits -> one output

Architecture:

`x -> (z1,z2) -> y`

Each hidden bit is an arbitrary unary Boolean map and the output is an arbitrary binary Boolean map. There are exactly

`4 x 4 x 16 = 256`

complete implementations.

The external universe is **still only one of the same four unary laws**:

- ZERO: 80 implementations
- ID: 48
- NOT: 48
- ONE: 80

But the hidden representation now has three exact structural types:

1. **collapsed** — both inputs map to the same 2-bit code;
2. **one varying hidden bit** — the two codes differ in one coordinate;
3. **both hidden bits vary** — the two codes differ in both coordinates.

Among the 16 possible hidden encoders there are:

- 4 collapsed encoders;
- 8 one-varying-bit encoders;
- 4 both-varying-bit encoders.

For a uniform input bit, any distinction-preserving hidden code contains exactly one bit of joint information. When both hidden bits vary, each marginal bit carries one bit but the joint code still carries only one bit: there is exactly one bit of redundancy.

Thus width can create **redundant internal representation without creating new external behavior**.

A second important fact appears: if the hidden representation visits only two of the four possible 2-bit hidden states, the output neuron's values on the other two states are counterfactual degrees of freedom. Multiple internal laws therefore agree on every actually reached hidden state but disagree off the realized hidden manifold.

## Brick 3: two Boolean inputs -> one Boolean output

Now the external domain contains four situations:

`00, 01, 10, 11`.

There are exactly `2^4 = 16` Boolean laws.

They divide by dependence as:

- 2 constant laws;
- 4 laws that effectively depend on only one input;
- 10 genuinely two-input laws.

This is the first brick where **relations between distinct inputs matter**.

Eight of the 16 laws are affine over GF(2), of the form

`a*x XOR b*y XOR c`.

The other eight are genuinely nonlinear over GF(2), including AND, OR, NAND, NOR and their input-relabeled variants.

After quotienting away input swap, input negations and output negation, the 16 laws fall into only four structural families:

1. constants;
2. one-input projections/negations;
3. parity (`XOR/XNOR`);
4. nonlinear conjunction/disjunction family (`AND/OR/NAND/NOR` plus relabelings).

So the first nontrivial Boolean relational ontology is already tiny after removing arbitrary symbol labels.

## Brick 4: two inputs -> one hidden bit -> one output

The hidden neuron can be any of the 16 binary laws and the output any of the four unary laws, giving 64 implementations.

All 16 external binary laws are realizable. A single hidden bit does **not** enlarge the external function class, because the hidden neuron can already compute any binary Boolean function and the final unary neuron can preserve or negate it.

Every nonconstant external law has exactly two such implementations: compute the law and preserve it, or compute its complement and negate it at the output. The two constant laws have many more degenerate implementations.

Again, hidden depth increases mechanism multiplicity rather than the set of externally possible laws.

## Brick 5: two inputs -> two hidden bits -> one output

There are

`16 x 16 = 256`

possible two-bit hidden encoders and

`256 x 16 = 4096`

complete networks after the binary output neuron is chosen.

The crucial interpretation is that a hidden encoder induces a **partition of the four input situations** according to which situations receive the same hidden code.

The 256 encoders split by number of distinct hidden states reached:

- 1 hidden state: 4 encoders;
- 2 hidden states: 84;
- 3 hidden states: 144;
- 4 hidden states: 24.

The exact partition-shape counts are:

- `4`: 4
- `3+1`: 48
- `2+2`: 36
- `2+1+1`: 144
- `1+1+1+1`: 24

Under a uniform distribution over the four inputs, these partition types carry hidden-state entropies of respectively:

- 0 bits;
- ~0.811 bits;
- 1 bit;
- 1.5 bits;
- 2 bits.

Most importantly, if the hidden representation reaches `k` distinct code states, the arbitrary downstream Boolean output can realize **exactly `2^k` different external labelings** of the input situations, because all inputs collapsed into the same hidden equivalence class must receive the same downstream label.

So:

| Distinct hidden states k | External laws realizable |
|---:|---:|
| 1 | 2 |
| 2 | 4 |
| 3 | 8 |
| 4 | 16 |

This gives the first clean constructive principle of the Boolean ladder:

> **Each new hidden distinction doubles downstream labeling freedom.**

The hidden layer is therefore naturally understood not primarily as “neurons computing features,” but as an equivalence relation / partition over situations.

## Brick 6: one recurrent Boolean state — first time

Only now do we add recurrence:

`s_(t+1) = f(s_t)`

with one Boolean state and no external input.

There are again exactly four unary update laws, but interpreting composition through time produces four dynamical worlds:

1. ZERO — both initial states flow to fixed attractor 0;
2. ONE — both flow to fixed attractor 1;
3. ID — two stable fixed points; the system retains one bit of memory forever;
4. NOT — a period-2 oscillator.

This is the first appearance of genuine temporal concepts:

- attractor;
- basin;
- persistent memory;
- periodic orbit / clock.

They were impossible in the purely feed-forward bricks because repeated application had not yet been made part of the universe.

## Current ground-up picture

The ladder now separates several ideas that were entangled in the large neural universes:

1. **distinction** — whether states remain observationally different;
2. **hidden mechanism** — different implementations of one external law;
3. **redundancy** — multiple hidden bits carrying the same distinction;
4. **counterfactual law** — rules on hidden states never reached by actual inputs;
5. **relation** — laws that genuinely depend on more than one external variable;
6. **partition / equivalence class** — hidden representation as grouping of situations;
7. **expressive freedom** — each retained hidden distinction doubles possible downstream binary labelings;
8. **time** — recurrence creates attractors, memory and oscillation from the same tiny Boolean function class.

Nothing geometric or physical has been assumed or claimed yet.

## Natural next brick

The smallest next temporal system is **two recurrent Boolean state bits with no input**. It has four global states and `4^4 = 256` deterministic update maps. This is still completely exhaustible, but it is the first universe capable of:

- fixed points plus multiple basins;
- 2-, 3-, and 4-cycles;
- transient trees feeding cycles;
- permutations/reversible dynamics;
- genuinely nontrivial finite state-space topology.

That is the recommended Brick 7.