# Exact grouped MoE prefill candidates

These frozen scalar entries keep the locked BF16 k16 arithmetic. Their
first gate/up promotion selected decoded128 for E384/top-6 TP4 at exactly
8192 prefill rows. Current serving uses the qualified vector variant and
exact buckets described in [Vector staging](vector_prefill.md). Down keeps
its previous entries. Native block-scale MMA remains deferred to the end-of-project
within-noise batch: changing prefill bits independently of decode would affect
prefix-cache determinism. Its measured final f64-relative RMSE was
0.884943-0.932470 times the column path's error, but that does not qualify it
for the current bitwise-only work.

The frozen comparison prepares the pre-promotion serving registration at each exact row bucket.
Its baseline includes the measured grouped winners at 512, 2048 and 4096
rows, except E128 down at 2048, and keeps the column entry at 8192. The prior
native experiment used the locked column entry at every bucket. CPU tests
assert the actual baseline symbols, so they cannot silently fall back to that
older comparison. The shared expert is a fixed zero addend in both arms;
the routed sum still uses the serving ordered top-k F32 sum and add operation.

## Arithmetic and storage

Each output executes eight ascending k16 BF16 MMAs per 128-K chain, followed
by an F32 addition into its running sum. Down's stored K=576 is padded to
640 in shared memory, as in the column entry. Conversion, admission checks,
clamp/SwiGLU, route-weight multiplication, BF16 rounding, the intermediate
ceil quantizer, F32 down output and ordered top-k sum retain their existing
implementations. Predecode reads the quantized codes and scales, not the
original BF16 input.

A CTA uses aligned 16-byte asynchronous copies into one, two or three raw
64-K strips. It decodes each strip once into shared BF16 operands, then reuses
those operands across its row and column fragments. Every strip waits for its
own copy group; a barrier protects reuse of both raw and decoded storage.
Empty committed groups drain the pipeline without changing the final chain.

Each projection owns its queue, optional BF16 activation buffer and clock
buffer. Gate/up and down scratch do not alias. The task entry takes a virtual
CTA index and caller-owned shared arena; grid-dependency instructions remain
in the global wrapper. Inputs must use the aligned packed allocations used
by the lab. This is not a generic unaligned-view registration.

## Ablations

| Variant | Change relative to the preceding relevant control |
| --- | --- |
| serial64 | One raw stage, 64-row/64-column tiles, scalar scale loads |
| pipe64 | Two raw stages |
| pipe128 | Up to 128 rows, sharing weight decode between row groups |
| scale128 | Asynchronous aligned scale-word loads |
| stage3 | Three raw stages |
| wide128 | 64 rows and 128 columns, sharing activation decode across columns |
| decoded128 | Decode quantized activations once per projection; include that launch in timing |
| dynamic128 | Atomic work queue instead of static persistent grid stride |

Small queues retain 8/32/64-row tails. The CPU planner and generated CUDA
static assertions agree on every shared allocation. The lock rejects all
local memory and stack frames. Register counts and shared bytes are retained
in the staged resource table; these are occupancy constraints, not measured
occupancy or proof of a bottleneck.

Separate clock-stamped entries record setup/prologue, load wait and barrier,
decode and barrier, MMA, fold and barrier, enqueue, and epilogue and barrier
cycles from lane zero of every CTA. The normal entries contain no clock
instrumentation. Phase-cycle totals include overlapping CTAs and must not
be summed as wall time. Compare ablations and maximum CTA durations alongside
whole-component event times before attributing a limit.

## Qualification

The pair matrix covers E384/top-6 and E128/top-3 at 512/2048/4096/8192 input
rows with balanced and concentrated expert routing. Each variant compares
all route IDs/weights, BF16 intermediates, ceil codes/scales, F32 down values
and final sums byte-for-byte. Full, partial, empty and full graph replays poison
outputs and scratch. Stamped entries must also match. An omitted first down
chain must be detected.

Timing uses five interleaved rounds, three samples per arm and one graph
replay per sample. Gate/up, down, both projections, and the routed chain are
measured separately. One stream and event spans avoid summing overlapping
kernel durations. Diagnostic stamps are excluded. A candidate is listed as a
win only if both devices improve the projection pair by more than 3% and
neither full chain regresses by more than 3%. The 1.5x target is reported
separately. Nothing is routed automatically. The scalar gate/up-only win
provides the baseline for the subsequent vector-store qualification.

Build the new family with the standalone `sm120` lab example (`cargo run -p model-deepseek41-lab --example sm120 -- OUT`).
Its `--initialize` option refuses to overwrite a lock. Subsequent builds
verify that family. Existing locks are never rewritten. Run the ignored GPU
test only with a staged directory named by
`CHOPINFER_DEEPSEEK41_EXACT_PREFILL_CUBINS`. The job's CPU dry run verifies
symbols, embedded baseline modules, the single-ordinal worker requirement,
and snapshot digests before any GPU session starts.
