The TP4 prefill attention entries already use BF16 tensor-core MMA for QK
and PV. The main transformer uses two window layers, 20 mixed layers with
64-position compressed pages, and 18 mixed layers with 32-position pages.
At 8192 rows, 16 local heads and width 512, a dense 128-key window costs
34.36 GFLOP for QK and PV together; 128 window keys plus 512 compressed keys
cost 171.80 GFLOP. These are operation counts, not measured execution times.

The original source engine stages a complete 64-key decoded BF16 tile, a
shared query tile, four 128-channel score blocks, validity and addresses.
Its dynamic shared-memory allocation is 83,200 bytes at width 512. The
separate `stage32_v1` candidate retains only 32 decoded keys at a time and
loads query B fragments directly from global memory. It stages the two
halves for QK, then stages them again for PV. Dynamic shared memory becomes
42,240 bytes. Compilation with nvcc 13.2.86 reports 180-182 registers per
thread, a 1024-byte shared section, no local memory and no stack. Two CTAs
can fit the resource limits, compared with one for the original entry.
The extra source loads may offset that occupancy benefit; GPU timing must
decide whether the candidate is useful.

The candidate keeps the original arithmetic:

- A warp's QK partial starts at zero and runs eight ordered m16n8k16 MMAs
  over its 128 channels. Score blocks are added in increasing block order.
- Softmax still sees 64 keys at once. Each lane visits key tiles and its
  four values in the original order, followed by XOR reductions at 1 and 2.
  Exponentials, denominator FMAs and final lane folds remain unchanged.
- Each probability rounds to one BF16 term. A PV chain starts at zero,
  executes key tiles 0, 1, 2 and 3 in that order, and enters the running
  output through the same rescale FMA. Retaining chains between staged
  halves does not repartition softmax or change a chain's accumulation.
- The original merge handles the single split, sink, F32 division, BF16
  boundary and inverse 64-channel tail rotation. It is shared by both arms.

The reference `sources.cuh` and its builder remain unchanged. The candidate
has a separate header and builder, and refuses non-prefill or multi-split
entries. It is not registered in serving. A separate decode scheduling
entry can use the original source builder without changing this candidate.

The snapshot lab covers all three served TP4 variants on each device of an
allocated GPU pair. It compares every F32 workspace byte and BF16 output
byte, including poisoned inactive rows, invalid batches and slots, holes,
missing and out-of-range pages, and fresh-input graph replays at live counts
8192, 0, 1, 4096 and 8191. A four-row companion uses the existing native
f64 oracle without relaxing its numerical gate. All correctness cases run
before timing.

Timing uses synthetic dense visibility over an 8192-row staged window and
32K compressed positions, with the served 128+512 key budgets. It reports
attention, merge after attention, and the combined graph for each of the
40 main layers, separately on both devices. The pair runs TP4's local
16-head geometry; it is not a TP2 model run. Attention and combined graphs
alternate old/new timing order with equal cache eviction. Merge timing
excludes its attention predecessor. These measurements isolate kernels;
they do not measure end-to-end prefill or actual checkpoint activations.

The 2026-10-19 pair measurement passed every workspace and output comparison
on both devices. Across the 40 main layers, mixed attention itself took
about 28.91 ms per layer, with another 0.256 ms for merge. Half-tile staging
reduced attention to about 27.72 ms, with merge unchanged. Window attention
fell from about 6.12 to 5.62 ms. These are medians over the synthetic
8192-row cases, at the 250 W cap. The mixed-layer improvement is about 4%,
so the candidate remains outside serving.

The software ablation measured the bottleneck on 2026-10-19. Both devices
passed every original and sampled-full bitwise/replay gate. The following
milliseconds are the mean of each device's median at 8192 rows and 250 W:

| Sources | Reference attention | Sampled body, stamps off | Gather/decode/stage only | Resident-data, one gather |
| --- | ---: | ---: | ---: | ---: |
| 128 window + 512 selected | 28.913 | 28.876 | 27.594 | 4.255 |
| 128 window only | 6.117 | 6.097 | 5.531 | 3.382 |
| 512 selected only | 23.255 | 23.237 | 22.147 | 4.109 |

The full mixed probe's per-CTA stage medians were about 722,940 cycles for
gather/decode/staging, 12,572 for QK, 9,711 for softmax and 6,943 for PV,
against 761,976 total cycles. Gather/staging accounts for about 94.9% of
that total; the three math stages together account for about 3.8%. Window
and selected-only gather shares are about 90.3% and 94.6%. These ratios of
stage medians describe the sampled CTA, not additive wall-time estimates.
The earlier all-layer measurement put merge at about 0.256 ms per mixed
layer. The round 3 serving trace attributes 572 + 506 ms per 8192-row chunk
to the two mixed attention families, making them the largest prefill cost.

The driver reports one resident 128-thread CTA per SM for the reference,
full, loads-only and resident-data probes, or 8.33% theoretical warp
occupancy. Half-tile staging permits two CTAs, or 16.67%. Merge permits
12 CTAs, or 100%. None spills. Residency alone did not solve the bottleneck:
the earlier half-tile candidate improved mixed attention by only about 4%.

The snapshot isolates the bottleneck before another serving design.
Its `ablation` command measures the unmodified reference and three software
probes at the mixed-layer geometry, each with mixed, window-only and
selected-only visibility. Loads-only gathers, decodes and stages all keys
without MMA or softmax. Volatile shared stores retain that work without
adding a checksum. Resident-data loads the first 64-key tile once, then
runs the original number of QK, softmax and PV iterations on that tile.
Resident-data includes its initial gather and is a diagnostic workload
with different outputs. Neither diagnostic is an attention candidate.

Event timings pass a null stamp pointer. Separate launches record warp
zero's `clock64` intervals for query setup, gather/decode/staging, QK,
softmax, PV and the epilogue in each CTA. A stage includes any barrier at
its boundary; these intervals are elapsed cycles, not hardware stall
counters. Per-SM clocks are not combined into a wall time. Every record
must partition its own start-to-end interval. The sampled full probe must
pass the original F32 workspace, BF16 output and replay comparisons before
any timing. The unmodified reference timing controls for the probes'
extra registers: compilation reports 196-198 for full, 56-61 for loads and
198-202 for resident-data, with no local memory or stack allocation.

The job queries function attributes and theoretical occupancy after
setting each entry's dynamic shared-memory limit. Its offline SASS audit
checks that loads-only contains no MMA instructions. Static instruction
counts are not executed instruction counts. Hardware counters are not
available to the job, so it does not claim measured DRAM/L2 bytes or
hardware stall reasons.

The resource probe records compiler shared-section bytes separately from
the driver's static shared allocation. For the window half-tile entry,
the compiler reports 1024 bytes while `CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES`
reports zero. The driver accepts 42,240 dynamic bytes and reports two
resident CTAs per SM. Equating the two shared-memory fields caused the
first software profiling job to stop before launching attention. The
corrected probe uses `cuOccupancyMaxActiveBlocksPerMultiprocessor` for
residency and keeps strict checks on registers, spills, launch limits and
positive occupancy. It retains both shared-memory observations in its
report without inferring why their accounting differs.

Every subprocess writes stdout, stderr, command and exit status into its
run directory before status validation. This includes snapshot checks,
resource queries, power checks, attention and the kernel journal. The
resource trace identifies the logical rank, symbol and API at each step;
timeout cleanup reaps the subprocess before reporting failure. CPU tests
reproduce the shared-memory mismatch and check failure logging, timeouts,
missing executables and both pair-device masks.

The `traffic` command counts unique encoded payload in the exact dense
fixture, excluding metadata, transaction granularity and repeated scale
loads. Without reuse across query rows or the two head groups, gathers
request 3.523 GB per mixed layer. Reuse within runs of 64 query rows lowers
that payload to 1.815 GB; complete reuse across the chunk and both head
groups lowers it to 13.64 MB. These are traffic models, not cache misses.
For any assumed payload bandwidth B in TB/s, the corresponding byte floors
are 3.523/B ms, 1.815/B ms and 0.01364/B ms. Query reads and partial writes
add 0.404 GB in all three models.

The original tile alone requires at least 10.737 GB of shared stores and
21.475 GB of QK/PV shared reads per mixed layer; half-tile staging doubles
the stores. These counts exclude query staging, score exchange, addresses,
validity and bank conflicts. A 64-row window union contains 191 keys,
occupying 195,584 decoded BF16 bytes or 100,848 native bytes. The decoded
union exceeds the shared budget, and the native union leaves insufficient
compute scratch. Any neighboring-row reuse design needs a smaller run,
another staging layout or a separate decode pass.

These byte floors are far below the measured 28.9 ms at ordinary device
memory bandwidths. The fixture's 13.64 MB unique native payload also fits
inside the measured 128 MiB L2 capacity. Neither observation establishes
cache hit rates. The loads-only and resident-data measurements instead
point to the repeated scalar gather/decode loop and low residency. Each
reference thread performs 128 two-value gather iterations per full tile,
with dependent address, code and scale reads before its shared store.

The next candidate loads eight decoded dimensions per thread in packed
transactions and writes one 16-byte shared sector. Two register packets
place the next gather before the current decode/store. It retains the
reference's F32 scale multiplication and BF16 rounding, with an unaligned
load fallback. Each key is staged once for both QK and PV.

A 256-thread CTA owns two consecutive query rows. It compares every source
address, scale address and validity tag before sharing a 64-key tile. Equal
lists share 64 decoded keys; lists shifted by one share their 65-key union.
Different lists use two cooperative staging passes through the same tile.
Only the corresponding row's warps execute math in each such pass. Thus
each row still performs exactly one original 64-key QK, softmax and PV unit,
in original unit order. QK chains, score folds, softmax, PV chains, workspace
bits and the reference merge are unchanged. Inactive row partners join all
barriers but do not read queries or write partials.

The paired layout requests 85,504 dynamic shared bytes at width 512. Its
256 threads provide eight loader warps per CTA, versus four in the original.
A separate one-row packed-load control requests 75,008 bytes. Both retain
the original argument and workspace ABI and have explicit lab shared-memory
attributes. All temporary reuse stays inside one CTA; no new global scratch
or cross-stream ownership is introduced. The pair job compares both
candidates against the original, keeps the f64 and poisoned replay gates,
adds identical/shifted selected-list cases, and times all 40 main layers.
Compilation reports 218-222 registers per paired-row thread and 184-188 for
the one-row control, with no local memory or stack in either. The reference
cubin remains identical to the measured baseline. These resource audits
are CPU evidence.

The paired-row measurement on 2026-10-19 passed all workspace, output,
f64, reuse and graph-replay gates on both devices before timing. It reached
157 of 160 layer/candidate/device records before the job timeout. Every
variant was timed on both devices; the missing records are the final three
mixed32 layers for the paired candidate on the second device. Medians of
the available per-layer records at 250 W are:

| Variant | Reference attention | Packed one-row attention | Paired attention | Reference total | Paired total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Window | 6.11-6.12 ms | 2.09 ms | 0.87-0.89 ms | 6.37 ms | 1.13-1.14 ms |
| Mixed64 | 28.89-28.93 ms | 9.71-9.72 ms | 5.88-5.89 ms | 29.14-29.18 ms | 6.14 ms |
| Mixed32 | 28.88-28.92 ms | 9.71 ms | 5.89 ms | 29.13-29.18 ms | 6.14 ms |

These are partial timing results, not a completed job verdict. The
restaged job retains every bitwise gate and all 160 timing records, with
one warmup and three alternating A/B samples for each of attention, merge
and combined execution. Merge uses an untimed attention producer for each
arm. Its prior 15 rounds per arm were a substantial part of the timeout.
The job allows 700 seconds for GPU work and 890 seconds including setup
and diagnostics; a complete GPU rerun remains required.

Serving selects the paired schedule for native Flash TP4 prefill launches
with an 8192-row host grid, including graph replays with fewer live rows.
Window, mixed64 and mixed32 all use the measured schedule. Host row counts
outside the measured admissions below, BF16 KV, other topologies, decode, verify and draft attention retain
their existing route. The model adapter validates through the original
prepared operation and changes only the attention symbol and launch plan.
Its metadata retains reference symbols for fallback and adds the paired
entry's 85,504-byte dynamic shared-memory limit, which the serving loader
sets before launch. The original merge, arguments, workspace and numerical
contract remain intact. The new lock family embeds the same paired cubin
bytes qualified by the lab; existing locks are unchanged.

CPU fake-host checks cover all 40 layers, shared-memory configuration,
8191-row fallback, BF16 and TP2 fallback, subsequent decode and resource
release. Journal comparisons check unchanged argument spans, scalar
values and merge order. GPU serving integration and completed all-layer
rerun timing remain unproven. Applying the measured attention ratio to
round 3's exclusive mixed-attention time suggests about 0.86 seconds saved
per 8192-row chunk, roughly 3.4 seconds per 32K prompt. This is a kernel-only
projection, not an end-to-end serving measurement.

## Decoding keys once

On 2026-10-19, the decoded BF16 cache passed all 24 full-chunk comparisons:
window, mixed32 and mixed64 at offsets 0, 8192, 16384 and 24576 on two GPUs
at 250 W. The selected and window lists came from a served 32K prompt. Q/KV
values and physical page placement were synthetic native fixtures. F32
partials and BF16 output were bitwise equal to paired attention, including
changed-input and live-count replay, missing and aliased pages, f64 checks,
private allocation checks and canary controls.

Combined cache construction, attention and merge took these milliseconds per
mixed layer per 8192 rows (three interleaved samples):

| Variant | Offset | GPU 0 paired | GPU 0 cached | GPU 1 paired | GPU 1 cached |
| --- | ---: | ---: | ---: | ---: | ---: |
| mixed32 | 0 | 5.689 | 5.714 | 5.611 | 5.627 |
| mixed32 | 8192 | 6.140 | 5.598 | 6.029 | 5.521 |
| mixed32 | 16384 | 6.148 | 5.432 | 6.065 | 5.350 |
| mixed32 | 24576 | 6.151 | 5.375 | 6.071 | 5.295 |
| mixed64 | 0 | 5.898 | 6.152 | 5.820 | 6.054 |
| mixed64 | 8192 | 6.143 | 6.065 | 6.056 | 5.962 |
| mixed64 | 16384 | 6.158 | 5.931 | 6.057 | 5.833 |
| mixed64 | 24576 | 6.172 | 5.813 | 6.075 | 5.729 |

Window total time fell from 1.130-1.159 ms to 1.047-1.073 ms. The binding
uses the cache for window layers at all four measured offsets and for mixed
layers after offset zero. Applying those differences to two window, eighteen
mixed32 and twenty mixed64 layers predicts about 11-21 ms saved per later
8K chunk and 50 ms per 32K prompt. These are kernel projections; the serving
binding and its table upload still require an integrated GPU measurement.

The route requires native Flash TP4, a one-item instance, and an eager
8192-row prefill with consecutive positions at one of the measured offsets.
Other row counts, offsets, topologies, BF16 KV, multiple-item instances,
decode, verify and draft attention keep their existing routes. Every decoded
consumer uses the measured paired body and 85,504 dynamic shared bytes. Its
64-key units, QK chains, softmax/PV order and original merge are unchanged.

A dense decoded copy of the complete serving KV pool would be too large.
The host instead bounds the active physical-page span at 1024 pages. It
subtracts the same page origin from the table and advances the native pool
view by that origin times the original page stride. This preserves every
selected native byte address, including repeated page IDs and holes. A wider
physical span keeps the paired mixed path. The window route is independent
of that fallback.

Each admitted rank owns a separate 75,632,640-byte data allocation and one
full page-table row. Both are reserved before the host sizes its KV pool.
The data allocation holds guarded, aligned window, compressed and mark
regions. Each layer uses its own bounded layout inside that allocation;
32-position pages fit inside the allocation sized for 64-position pages.
The cache is disjoint from native KV, activations, F32 partials, output,
collective buffers and side-stream workspace. A lowering-time check rejects
aliasing with any live attention view.

The rebased table is uploaded once before the eager chunk. Every layer
clears marks, marks selected physical pages, decodes the window and marked
pages, runs attention and merges on the same stream. Producers retain ordinary
stream ordering. A model adapter admits only the six qualified producer
symbols with that ordering and resolves the whole segment before submission;
PDL-only lists and graph dispatch retain their original helper. No cache
region is shared across a fork or used by another stream.

Upper-layer row skipping retains cached attention only in the full-width
prefix before the final compressed source. Its retained key and output
intervals keep native attention, original row origins, selected planes and
window planes. The decoded route rejects any lowering with a row origin,
even if its row count could otherwise qualify.

Output-projection overlap and standalone DMA preserve full-width attention
on the main stream, so they retain the qualified cache route. Their side
streams use separate reduction inputs, outputs and inboxes. Two-part
sublayer overlap rebuilds attention at each part's smaller row count and
uses native attention; it uploads no unused rebased cache table. The
shared-expert fork's allowed inputs and private workspace remain unchanged.
No cache region is used by a peer transfer or a side-stream launch. BF16
gather admission, reduction order, events and original merges are preserved.

Matched fake-host comparisons retain the same allocation identities with
and without the cache. They check every unmodified launch, merge argument,
event and peer transfer, and require native attention arguments to match
exactly for every smaller part. Real serving composition still requires
GPU qualification.

With upper-layer skipping, only the twenty full-width prefix layers benefit
from the cache. The full forty-layer timing projection above does not describe
that route; its total saving needs a serving measurement.

## Retained prefill rows

The native Flash TP4 mixed64 prefill entry also admits the measured row
counts 64, 192, ..., 2496. These are the upper-layer dependency cone's
query intervals for an 8192-row chunk. The existing paired body uses the
same arguments, partial workspace and merge, with half as many CTAs,
256 threads and 85,504 dynamic shared bytes. Other unqualified row counts,
window/mixed32 entries, BF16 storage and other topologies retain their
existing routes. Full 8192-row routing remains unchanged.

On two SM120 GPUs at 250 W, measured on 2026-10-19, all 320 layer,
candidate, device and offset comparisons passed before timing. Gates
compare F32 partials and BF16 output exactly, with changed-query/live-count
graph replay, poison guards, f64 controls, invalid pages and shared or
shifted selections. Selected and window lists were captured from serving;
Q and native KV values were synthetic. Timings use one warmup and three
interleaved samples, including the original merge.

| Offset | Device | Original twenty-layer total | Paired total |
| ---: | --- | ---: | ---: |
| 0 | A | 94.055 ms | 19.809 ms |
| 0 | B | 93.946 ms | 19.805 ms |
| 24576 | A | 94.544 ms | 20.030 ms |
| 24576 | B | 94.413 ms | 20.039 ms |

The totals sum the per-layer medians for 25,600 query rows across the
twenty retained layers. They save 74.1-74.5 ms per chunk in the lab,
approximately 4.7x for this attention work. They are not served chunk
latencies. One-row vector staging is 8-16 microseconds faster than paired
at 64 rows, while paired still improves that point by about 3x over the
original. Serving keeps the already-loaded paired module at that point.

This routing adds no scratch or cache producer. It leaves carried query
metadata, selected/window row origins, main/side streams, events and peer
transfers unchanged. Retained rows remain on native key storage. The
decode-once cache still applies only to its full-width prefix. The native
part admission below extends the smaller row routes. The fake-host audit
permits only the attention symbol, grid, block and dynamic shared-memory
changes; the original merge and all other launch arguments remain exact.

## Two-part row qualification

Native Flash TP4 prefill also admits 4096 rows for window, mixed32 and
mixed64, plus mixed64 counts 64, 128, ..., 1280. The latter are the aligned
halves of the retained intervals. Existing 8192-row and retained-whole
routes remain enabled. This changes only attention dispatch, and helps
an independent 4096-row request without splitting it or enabling overlap.
The original prepared operation still validates all buffers and retains
the original merge. No allocation, argument, stream or event is added.

The pair qualification on 2026-10-19 completed in 239 seconds on two
SM120 GPUs at 250 W. All 554 bitwise/replay records passed before the 364
timings. Each case compares F32 partials and BF16 output against its exact
row slice of the full-chunk paired result. Gates include changed queries,
odd live endpoints in either part, zero-live replay, distinct per-row RoPE,
cache and inactive-row guards, exact selected-page marks and f64 controls.
The exact timed A+B chain is checked in addition to the observed chain
that preserves A's output. Selected/window lists are captured from
serving; native Q/KV values are synthetic.

Measured per-part times include cache construction when present,
attention and merge, with one warmup and three interleaved samples:

| Offset | Variant | Lab winner | Device A, part A/B | Device B, part A/B |
| ---: | --- | --- | ---: | ---: |
| 0 | window | cached | 0.514 / 0.514 ms | 0.504 / 0.506 ms |
| 0 | mixed32 | native paired | 2.672 / 3.049 ms | 2.641 / 3.006 ms |
| 0 | mixed64 | native paired | 2.868 / 3.075 ms | 2.814 / 3.031 ms |
| 24576 | window | cached | 0.516 / 0.514 ms | 0.506 / 0.502 ms |
| 24576 | mixed32 | cached | 2.704 / 2.687 ms | 2.660 / 2.650 ms |
| 24576 | mixed64 | cached | 2.934 / 2.916 ms | 2.892 / 2.876 ms |

At offset 24576, native mixed parts cost 3.043-3.106 ms. Sequential mixed
A+B costs remain within 0.5% of their corresponding full-chunk route at
both offsets. The old 4096-row fallback was not timed in this job; these
are measured part costs, not a direct speedup measurement over that
fallback or an integrated serving latency result.

For upper layer L in 20..39, define k=39-L, N=64+128k and F=8192-N.
The whole retained interval is (F,N); aligned parts are (F,64(k+1)) and
(F+64(k+1),64k), omitting the empty second part. All origins refer to the
original chunk. Native paired is bitwise qualified for all these shapes
at offsets 0 and 24576; cached is qualified at 24576. Splitting the twenty
retained intervals adds 1.60-1.66 ms per chunk, so sequential serving
keeps them whole. Whole cached retained attention saves 0.970/1.004 ms
at 24576 across the two devices. Its admission remains with the future
part-cache binding instead of adding a separate cached-retained route.

Cached parts are qualified in the lab but are not selected by this serving
change. Their window decode interval for part (F,N) is
[max(H+F-127,0),H+F+N), where H=min(chunk_offset,127). Cold part B needs
[3969,8192); at offset 24576 it needs [4096,8319). Keep consumer key
coordinates unchanged, local Q/workspace/output at zero, and carried
metadata at the original row origin. Clear page marks and rebuild decoded
pages after each part publishes its native sources. Cache storage and
tables must remain disjoint from all concurrent reduction spans until
join. The overlap binding, publication ordering and served-state evidence
remain separate integration work.
