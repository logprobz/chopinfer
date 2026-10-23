# Strip chains for the indexer query and KDA low-rank up projections

The decode and verify buckets (1 through 32 rows) of two BF16 shapes move
from the one-split tiled entry to strip chains at one split. The indexer
query (4096 x 1536) runs two strips at 1 and 2 rows, one strip at 8 and 16
and four strips at 4 and 32. The KDA gate and forget low-rank up
projections (2048 x 128) run two strips at every bucket. These are the
fastest candidates of the kernel-lab strip sweep, measured on 2026-10-21 at
1, 4, 8, 16 and 32 rows: 2.0x to 3.8x and 2.2x to 3.4x faster than the
tiled entry in isolated graphs, with today's entry against the one-split
entry within 0.9% as the A/A. That run also checked every candidate equal to
the one-split entry bit for bit at every live count, under memcheck with no
errors, and rejected all six strip mutants.

Strip chains keep each 64-K chain of four k16 MMAs from zero and add the
chains in K order, so the outputs keep their bits. One split needs no chain
workspace. Prefill and the other shapes keep their entries.

The projection cubin replaces 12 entries (6 buckets of each shape) and adds
none elsewhere: its other 95 entries keep their lock records, and their
SASS is identical function by function. The other cubins keep their bytes.

The fake-host journals were audited line by line against the previous
bodies: 43 of 67 change, 24 are identical. Every changed line belongs to one
of three kinds:

- A launch of a replaced entry: 12,940 lines. The symbol, the grid, the
  block and the dynamic shared memory change, and in 12,192 of them the
  weight tensor map's box rows change from 64 to 16 rows per strip. The
  stream, the flags, the other grid and block dimensions, the cluster,
  every buffer, offset, length and scalar argument stay the same. The old
  and new symbols map one to one.
- Module creation: the 192 resolve lines of the removed entries become 192
  of the added ones, and the shared-memory configure lines follow the
  entries (80 added, 32 removed). This adds 12 lines to each create journal.
- The invariance declarations: equal after the renaming, in a new sort
  order.

The changed pins are in the create, decode-projection, invariance, NVFP4
package and context-parallel served and cache tests.
Served TP4 token and logprob equality and the decode profile are separate
GPU requirements.
