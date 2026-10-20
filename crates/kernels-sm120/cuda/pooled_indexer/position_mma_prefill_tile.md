# Compact prefill score tile

This candidate retains the guarded native scorer's arithmetic. A CTA holds
one packed query and visits 128 compact slots per iteration. Four ordinary
32-slot MMA tiles feed a shared array; all 128 threads then compute one
score each. The shared-memory column permutation is a bijection, so it
changes only which bank stores each dot product.

Each slot uses the original retained-ID binary search, tag and visibility
checks, page admission and key decoding. Its two native FP4 MMA instructions
per head group, fine exactness predicate and ordered 32-head FMA chain are
unchanged. Coarse admission is the existing sufficient bound. A rejected
slot runs the same SIMT repair. Query packing failure dispatches to the
existing SIMT scorer before the candidate body. Invalid score bits and
unconsumed padding retain the existing semantics.

The x grid can shrink because both paths use grid-stride loops. For tile
width 128 and grid width G, the position assignment is
`x * 128 + iteration * G * 128 + local_slot`. It partitions the active
compact range without overlap. Tail clearing separately partitions the
remaining score plane. Visible counts and candidate records stay on the
device; changing a row's count does not change its prepared grid.

The candidate does not share arithmetic between query rows, change score
strides or allocate another workspace. Diagnostic counter entries are
separate from the timed entries. Qualification compares complete score
planes before selection cleanup and after the unchanged selectors, ordered
IDs, status and candidate records. It also covers partitioned rows, poisoned
inactive rows, stale tags, malformed retained IDs, nonfinite values, ties
and guard repair.

This is an unqualified candidate until its pair passes. The model-owned
serving option is off by default.
