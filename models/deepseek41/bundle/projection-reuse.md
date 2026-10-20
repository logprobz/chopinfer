# Measured decode projection reuse

On 2026-10-20, Flash TP4 passed 32,000 bitwise comparisons over real AR
and DSpark inputs, including eager and repeated graph execution, input-domain
fallback, stale rows, output guards and 4,000 detected GPU controls. Two
SM120 devices at 250 W recorded cold and warm timings. Each selected bucket
beat its current entries by more than 3 percent in every input set and cache
mode on both devices. Existing numerical reductions and lock families remain
unchanged.

| Projection | Selected rows | Tile columns | Folds per CTA |
|---|---|---:|---:|
| q-a | 17-21, 23-24 | 32 | 4 |
| q-b | 9-12 | 64 | 4 |
| o-a | 17-32 | 32 | 4 |
| o-b | 17-23, 25-28 | 32 | 8 |
| o-b | 29-32 | 32 | 2 |
| LM head | 9-32 | 16 | Four fixed 128-K chains |

C1, C8 and six/eight-row verification retain their previous entries. Other
geometries and rank counts retain their routes. The native tasks share a
weight stage between two independent M16 groups; the head shares its weight
stage across four M8 groups. Every output keeps its original arithmetic and
ascending reduction order. The head's row classifier and reducer keep their
symbols, arguments and workspaces.

At 32 rows, o-a, o-b and the head project 0.420-0.612 ms saved per AR step;
32-row DSpark verification projects 0.424-0.612 ms at C8 and 0.419-0.612 ms
at C16. These add forty layer calls and one head per rank, without multiplying
by the number of parallel ranks. They are kernel-event projections using a
representative captured layer, not end-to-end throughput measurements. The
full-block draft projections exceed 32 rows at these DSpark concurrencies
and retain their routes.

Lifecycle journals add two modules, seventeen symbols and nineteen releases
per rank. The C1/C2 lifecycle pins change only through those handles; all
allocations and launches are retained. Paired graph checks preserve every
launch at C1/C8 and six/eight-row verification, and every non-projection
launch, head classifier and reducer in larger row buckets. Scratch remains
private to each prepared operation, with no new stream fork or prefetch.
