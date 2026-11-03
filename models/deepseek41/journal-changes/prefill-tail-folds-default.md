# Dense prefill tail-fold default

The `prefill_tail_folds` contract and shape defaults become true. The
partition remains restricted to Flash TP4. Both served TP4 profiles inherit
it; other topologies retain their split or padded tails. Complete 128-row
TMA tiles and separately qualified matrix routes keep their existing
lowering. Untabled remainders use exact-row folds of at most 64 rows,
including two folds for widths 65 through 127.

Two linked CPU proofs audit the default change. A capture-only intermediate
revision reserves both fold workspaces while retaining the previous routes.
The named workspace-reservation stage and its committed source patch are
listed in the [reproduction recipe](prefill-adoption-stages.md). They
recreate the final sealed source tree directly from the committed base.
The [workspace specification](prefill-tail-folds-workspace.json) compares
that revision with the pre-adoption baseline. It permits only these
256-byte-aligned allocation changes, one per affected rank:

| Resource | Before bytes | After bytes | Allocations |
|---|---:|---:|---:|
| Shared workspace | 11797504 | 13110272 | 140 |
| Main workspace | 546848768 | 629158912 | 32 |
| Main workspace | 537919488 | 629158912 | 12 |

Attention launches that take a whole workspace grow their views at 9216
exact positions. The specification binds each argument to its allocation,
rank and stream. Required-size subviews keep their original lengths and
offsets. Every other argument, operation and issued identity is unchanged.
The workspace proof is PROVEN: 15 journals identical, 29 changed only as
declared, and all 44 fixture histories compared. The pinned bodies contain
160 allocation records; the create comparisons cover all 184, including
records also present in pinned bodies.

The [route specification](prefill-tail-folds-default.json) then compares the
intermediate with the adopted default, holding every allocation fixed. It
removes 12976 split-tail launches and 10528 general-fold launches, replacing
them with 31488 ticket-fold pieces and 2544 merge-fold pieces. This is 23504
removed operations and 70608 added operations. Every piece retains its
weights, slices activations, scales and output at the original row origin,
keeps the step's live-row and status pointers, and advances the status row
offset. Its scratch is bounded and disjoint from the other piece and live
operands. Complete TMA tiles, streams and every retained operation stay
exact. The 4064 expansion rules select exact original launch positions, reuse
unchanged arguments, and bind sliced buffers to authenticated allocations.
The route proof is PROVEN: 30 journals identical, 14 changed only as
declared, and all 44 histories compared.

Together the stages audit 29 changed journal bodies and 25 distinct source
pins. Captures use `tools/journal_proof/capture.py`, nextest with four
workers, and the four existing export-only ignored tests listed explicitly
through `--allow-ignored`. `tools/journal_proof/proof.py` checks both stages;
the final routing proof uses the specification exported unchanged from the
adopted head, with its SHA-256 recorded in the report. Captures
and reports remain outside the repository. No kernel lock changes. These
proofs check recorded structure, not kernel numerics or performance.

The corrected tail-fold allowances move onto `tp4-ar` and `tp4-dspark*`,
including the latter's conformance variants. The old TP4 split-tail and
general middle-fold gaps disappear, so their 21 obsolete entries are
removed. Every adopted entry has a full-tier witness in single mode or at
mandatory probe rows: 122 AR entries and 139 DSpark entries, with no entry
depending on the dense-probe seed.
