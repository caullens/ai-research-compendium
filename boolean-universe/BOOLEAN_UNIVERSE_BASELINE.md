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

## Current lesson

Depth by itself is not yet expressive power in this unary Boolean setting. Composition stays inside the four-element monoid of unary Boolean maps.

The next brick should be chosen deliberately rather than automatically. Natural candidates include:

- add a second hidden Boolean node while retaining one input and one output;
- add a second Boolean input, which is the first change that enlarges the Boolean function class dramatically;
- add recurrence/memory to one hidden bit, which is the first route to genuine temporal dynamics;
- constrain neurons to a specific primitive such as threshold/AND/NOT and study what composition can construct.

The program should advance only one of these assumptions at a time.
