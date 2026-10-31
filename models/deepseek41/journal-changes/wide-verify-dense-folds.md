# Wide verify dense folds

With the `wide_verify` option, a Flash TP4 target verify step of 65 to 192
rows runs each dense projection through the wide ticket fold of the
narrowest wide verify bucket that holds it (`dense_decode::Entry::wide`,
from the `dense_decode_wide` module) instead of the 8192-row fold, and
Engram's projection through its matrix tile at the step's rows. Only the
two allocation fixtures with the option on (the C16 odd-width fixture at
65 rows and the C32 maximum fixture at 192 rows) change; without it every
journal is identical.

The [change specification](wide-verify-dense-folds.json) is the identity
specification of every other fixture. In each of the two journals:

- every 8192-row fold launch of the step expands to its wide fold's
  admission (one CTA per row) and compute launches, with the base launch's
  activation, weight, output, live-row and status spans, the rank's main
  or shared workspace and null second-bank pointers; Engram's expands to
  its matrix tile's admission and compute launches over the base spans.
  One rule per symbol and rank, 2640 launches in each journal:

  | Projection | Shape (N, K, groups) | Wide fold tile | Workspace |
  |---|---|---|---|
  | query down | 1280, 5120, 1 | 32 | main |
  | query up | 8192, 1280, 1 | 64 | main |
  | indexer query | 4096, 1280, 1 | 64 | main |
  | window KV | 512, 5120, 1 | 32 | main |
  | output A | 1024, 4096, 2 | 32 | main |
  | output B | 5120, 2048, 1 | 64 | main |
  | shared gate and up | 576, 5120, 1 | 32 | shared |
  | shared down | 5120, 576, 1 | 64 | shared |
  | Engram | 25600, 6144, 1 | matrix | none |

- each rank's shared workspace grows from 11797504 to 19665664 bytes (the
  wide shared folds' partials at the step's bucket); the main workspace
  keeps its size;
- the 16 resolves of each rank's wide module load are added, and destroy
  adds one release per added load and resolve (68).

Every other launch, allocation, module, resolve, launch count and
operation order is preserved. A wide fold sums each output's K128 fold
partials from zero in ascending order, as the 8192-row fold does, and fails
the rows it fails, so the outputs are unchanged; fixed-batch GPU replays
confirmed it bitwise at verify widths 64 to 192.

On 2026-10-31 the journal proof of this change against the switch commit
was PROVEN with full history coverage: 23 journals identical, 2 proven,
25 of 25 fixture histories compared.
