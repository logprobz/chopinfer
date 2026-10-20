The selected-KV cache experiment targets repeated native widening in decode
attention. At Flash TP4, C1 and eight-row decode each issue 128 source CTAs
on 188 SMs. Six-row verification issues 96; 32-row verification issues 512.
The current 83,200-byte source arena admits one CTA per SM. Earlier phase
samples attributed 76-81% of mixed-source CTA cycles to KV loading,
widening, shared stores and barriers. Those samples do not isolate hardware
stall reasons. A round-five profile on 2026-10-19 measured about 1.0 ms per
C1 step in the 40 source layers, with little growth from 1K to 128K context.
The mixed source has ten 64-key tiles: six numerical splits process one
tile and two process two tiles. At C1 the longer tail therefore has only
32 CTAs. Window-only sources have two nonempty splits. The selected 8.31 MB
per-rank payload excludes repeated head staging, about 132.91 MB per C1
step, and about 10.53 MB of partial stores before merge reads. The merge's
measured exclusive cost is about 0.083 ms per C1 step.

The candidate decodes each selected key once per query row into a temporary
BF16 array. One warp owns a key; a 128-thread producer CTA owns four keys.
The mixed-source producer therefore launches 160 CTAs per row. Each source
CTA copies those operands asynchronously into its existing shared tile.
Its arena is 73,984 bytes, which still requires the dynamic shared-memory
opt-in. This is a change to staging and its distribution over CTAs, not an
increase in numerical split count.

The temporary cache holds only the selected key budget, 128 window keys
and up to 512 compressed keys. A mixed row uses 655,360 operand bytes and
2,560 validity bytes. It trades a producer launch and BF16 cache traffic
for fewer native loads and repeated conversions. The complete source path,
including the producer and unchanged merge, must win before adoption.

Every operand keeps the ordinary F32 decode, scale multiplication and BF16
rounding used by the reference. All 64-key tiles, 128-channel QK chains,
softmax updates, PV key order, empty-tile updates and eight numerical
splits remain in the same order. The existing merge entry is reused without
changes. Prefill retains one split and is outside this experiment. Merely
raising the numerical split count would change these rounding points even
if the new partials were merged in a fixed order.

Both producer and source have task-callable bodies with virtual CTA indices
and caller-owned shared arenas. Grid dependency calls occur only in entry
wrappers. The cache, validity and optional clock buffers must be distinct
from all source operands and from each other. Independent streams require
independent cache and partial buffers; this experiment adds no stream fork.

The pair covers C1, Decode-8, verify-6 from one sequence, and verify-32 from
eight sequences with four rows each. The reference includes the qualified
64-key ordinary-widening route at Decode-8 and verify-32. Contexts are 1K,
8K, 32K and 128K, with window-only and both compressed page formats. Matrix
and boundary cases are sharded across two GPUs. Boundary tests include
window and compressed-page edges, holes, misaligned native planes, poisoned
stale KV, inactive rows and entirely inactive batches. Each comparison
checks every F32 partial and output byte over two poisoned scratch replays,
plus an instrumented replay. A producer that suppresses valid keys must
fail in each of the four row buckets on each GPU.
Every timed candidate graph is also replayed after a fresh poison reset;
its partials and output must match independently of the timer's last arm.

Cold source timings include cache construction. Cold and repeated graph
attention timings include the producer, source and merge. Separate producer
timings and source phase clocks expose the staging cost. CPU validation
checks required function shared-memory opt-ins, launch shapes, scratch
bounds, alias refusals, the snapshot's complete module/symbol inventory,
and strict receipt coverage with negative parser controls.

The new artifact family is `latent_decode_cache`. Its serving module keeps
the qualified lab cubin byte-for-byte. Existing kernel locks stay unchanged.
On 2026-10-20, the pair passed 48 matrix and 168 boundary cases,
432 poison replays, 216 instrumented comparisons, 144 captured-graph
comparisons and eight negative controls. All 144 source or attention timing
cases improved. Across the twelve context/format cases in each bucket,
median cold full-attention ratios were 0.454 at C1, 0.500 at decode-8,
0.426 at verify-6 and 0.511 at verify-32. Repeated-graph ratios were 0.466,
0.557, 0.486 and 0.562 respectively. Each full-attention timing includes
cache construction and the original merge. The bitwise selection is limited
to these four measured phase/row buckets; other rows retain their current
source path. No prefill split count or operation order changes.

Serving allocates a separate 21,053,440-byte cache per rank at native Flash
TP4. Only decode rows 1 and 8 and verify rows 6 and 32 select it. The cache
is reused on the attention stream after its previous consumer, with no new
fork or cross-stream reuse. Each selected layer adds one producer launch;
the ordered merge and all other launches retain their bindings. CPU tests
check the measured buckets, unmeasured fallbacks, shared-memory opt-ins,
allocation lifetimes and exact native validation. Serving throughput still
needs an end-to-end measurement.

The journal audit checks all eight text/media topology-layout lifecycles
against their pinned baseline and a paired receipt covering the measured
and fallback row buckets. It preserves all retained launch arguments,
spans, events, waits and replay order under a resource-handle bijection.
The existing proportional AR graph splitter can move a cut when producers
add execution units; the audit checks its exact rule, node counts and whole
fork/join regions. DSpark keeps its cut immediately before the first mapped
row consumer. Fourteen journal mutations must fail before pins are updated.

A layer-weighted projection from repeated-graph full-attention medians saves
0.535 ms at C1, 0.460 ms at decode-8, 0.547 ms at verify-6 and 0.988 ms at
verify-32 per rank and step. This sums two window, twenty 64-key-page mixed
and eighteen 32-key-page mixed layers. It includes cache construction and
the ordered merge; it is a microbenchmark projection, not served throughput.
