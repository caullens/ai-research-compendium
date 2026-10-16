# Boolean Universe — Architecture BFS Checkpoint 6

This checkpoint adds causal identifiability to the four-node shell.

## 1. Setup

Consider a labeled four-bit affine Boolean universe

`x' = A x + b` over GF(2).

The matrix `A` is the exact directed dependency/wiring support together with its affine coefficients; `b` is the bias vector.

The question is whether the law can be inferred from passive observation alone, and how this changes under targeted intervention.

## 2. Five active one-step queries recover the exact law

Query the one-step transition at:

- baseline `0 = 0000`;
- each single-bit basis perturbation `e_1, e_2, e_3, e_4`.

Then

`b = F(0)`

and for every input coordinate `j`,

`A[:,j] = F(e_j) XOR F(0)`.

Therefore exactly five targeted one-step experiments reconstruct the entire labeled affine law and its wiring support.

A 10,000-law simulation verified 100% reconstruction.

This is also worst-case minimal for this protocol: each output coordinate has five affine coefficients `[A_i1,A_i2,A_i3,A_i4,b_i]`, so the augmented queried input vectors `[x;1]` must span a five-dimensional GF(2) design space. Fewer than five one-step full-state queries cannot do this for an arbitrary affine law.

Thus:

> **For a four-bit affine universe, five carefully chosen interventions are necessary and sufficient for exact causal identification.**

More generally, `n+1` affinely independent one-step queries identify an arbitrary labeled `n`-bit affine law.

## 3. A passive trajectory usually cannot identify the architecture

A passive trajectory provides transition equations using only the states the universe happens to visit.

The affine law is exactly identifiable iff the augmented observed states `[x_t;1]` span rank 5 over GF(2).

Across 30,000 random four-bit affine laws and random initial states:

- fewer than five observed transitions can never identify the exact law;
- after five transitions, only ~30.6% are identifiable;
- observing 6, 8, 10, 12 or even 16 transitions does not improve the overall probability beyond ~30.6%.

The plateau occurs because once the passive trajectory enters its eventual orbit, later observations repeat the same affine design span and add no new independent causal information.

Conditioned on reversibility, passive identification is easier but still incomplete:

- reversible affine universes after 16 transitions: ~50.3% identifiable;
- irreversible affine universes: ~21.8% identifiable.

A typical passive trajectory therefore leaves many alternative affine laws exactly consistent with every observation.

The mean remaining ambiguity after long trajectories is still around five bits in `log2(number of compatible affine laws)`, i.e. dozens of distinct causal laws can remain indistinguishable.

## 4. Architectural conclusion

This sharpens the implementation/effective-law distinction:

- passive trajectory behavior may underdetermine internal wiring indefinitely;
- targeted interventions recover the exact dependency structure with a finite optimal experiment set.

So architecture can be **causally identifiable while remaining observationally non-identifiable**.

The distinction is not philosophical; it is exact in this tiny universe.

> **Observation and intervention induce different ontology quotients.**

Under passive observation, all affine laws agreeing on the visited state-span belong to one equivalence class. Under a complete active basis experiment, that equivalence class collapses to a single labeled law.

## 5. Emerging scaling law

For `n` affine Boolean state bits:

- unknown law per output coordinate: `n+1` GF(2) coefficients;
- `n+1` affinely independent full-state one-step queries are sufficient;
- the canonical query set is `{0,e_1,...,e_n}`;
- passive identifiability depends on whether natural trajectories happen to generate full augmented rank.

This gives a simple experimental-science theorem for the Boolean universe program:

> **The intervention complexity of exact affine-law discovery grows only linearly with state dimension, even though the number of possible affine laws grows as `2^(n^2+n)`.**

That is a surprisingly strong separation between hypothesis-space size and active scientific sample complexity.

## Next breadth targets

1. Repeat causal-identifiability analysis for threshold and monotone networks, where interventions no longer admit a linear closed-form decoder.
2. Restrict sensors so the scientist cannot observe full state, and measure active sample complexity under hidden nodes.
3. Compare fixed ports versus time-varying ports during identification.
4. Treat the scheduler itself as an endogenous Boolean subsystem and ask whether it can also be identified experimentally.
5. Continue the four-node shell across asynchronous, role-free, layered, NAND-only, reciprocal and observer-limited architectures without privileging any one family.