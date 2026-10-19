# Prefill communication

The optional attention schedule splits the final output projection into two
row ranges for chunks of at least 1024 rows. The first range is aligned to
16 rows. The full-chunk quantizer runs first. Each range keeps the prepared
prefill entry, including its TMA admission launch. A communication stream
reduces A while the compute stream projects B. Both reductions finish
before the original full-row hyper-connection expansion.

The reduction retains the composed transport: F32 scatter, F32 gather,
then one BF16 rounding of the rank-ordered F32 sum. Its complete launch
sequence runs on one stream, including the final local rounding. The next
reduction starts only after the previous reduction on that rank finishes.
No transport registration or kernel changes accompany the schedule.

Admission requires independent producer rows, no workspace, no persistent
grid, no collective producer and no multi-CTA cluster. Resolved pointer
arguments, including tensor-map bases and peer pointer arrays, must have
disjoint byte ranges between A's reduction and B's producer. The original
scratch-lifetime checker resolves these arguments. An independent CPU
journal audit checks the submitted spans too.

Two live-count cells are uploaded once per prefill. The producer records
each completion before its reduction waits on it. Main joins both reduction
completions before scratch reuse, status readback, commit or prefix capture.
Event and partial-rank submission failures poison the instance.

Compressor, KV, ring and index writes keep their full-chunk launch order
and arguments. Decode and verify keep their existing schedule. Short chunks
and diagnostic text taps use the sequential prefill path.

`CHOPINFER_DEEPSEEK41_PREFILL_OVERLAP=1` enables this candidate;
`CHOPINFER_DEEPSEEK41_PREFILL_OVERLAP=0` selects the sequential control.
The default remains sequential pending GPU bitwise checks and paired
per-chunk timings. CPU ordering and scratch checks are not GPU correctness
or performance evidence. Mode 1 covers attention reductions only.

## Two parts across sublayers

Experimental mode `CHOPINFER_DEEPSEEK41_PREFILL_OVERLAP=2` also overlaps
embedding and MoE reductions. It admits one TP4 AR text request, starting
on the 64-token granule, with 1024 to 8192 rows. A ends at the midpoint
rounded up to that granule; B holds the remainder. Both parts fit within
4096 rows, including an odd final chunk. Other cases use mode 1's schedule.

Embedding lookup, RoPE tables and window-key lists are prepared for the
original chunk. Main computes each sublayer for A then B. The communication
stream reduces A then B, in the same sublayer order. Main waits for A's
reduction before A's next sublayer and for B's before B's next sublayer.
Thus A's reduction can overlap B's compute, and B's can overlap A's next
compute. The last hyper-connection expansion and the logits epilogue remain
full-row operations after both parts rejoin main.

`Lowering::prefill_row_origin` distinguishes `None`, the original full
chunk, from `Some(first)`, including A's `Some(0)`. Residual streams, mixer
coefficients, partials, reduced outputs, status, selections, candidates and
precomputed per-row metadata keep global row coordinates. Other compute
scratch keeps local row zero and its prepared capacity. It is reused only
on main. Token IDs, positions and item IDs are input subspans; each part has
its own immutable live count. Decode and verify ignore the origin.

Each layer gathers its old window prefix once, before A. The staged window
retains A's writes while B reads it. Compressor, compressed KV, index and
ring operations execute A then B on main, so B observes A's completed writes.
The original chunk's window bounds remain in use. The aligned split never
opens a ratio-2 compression group between A and B. Selections and candidate
metadata retain both global row ranges for later reuse layers; reindex
reads the part's count, while dense index blocks retain their original
64-row boundaries.

The resolved-span audit checks A's reduction against B's compute in every
block, and B's against A's compute in the next block. No shared byte range
is allowed across either boundary, including backend workspace. CPU tests
also inspect the submitted journal, retained selection and window views,
odd endpoints after offset 24576, prefix snapshot lifecycle and poisoning.
Prefix capture remains a committed-chunk operation; no internal part
endpoint becomes a new capture point. All communication joins precede the
existing status, commit and capture paths.

This mode remains experimental until full-chunk GPU bitwise comparisons,
prefix-hit comparisons and paired timings pass. Its CPU ordering checks
do not establish numerical equality or a speedup.
