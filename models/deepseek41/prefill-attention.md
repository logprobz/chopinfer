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

The next snapshot isolates the bottleneck before another serving design.
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

The software profiling result is pending. The measured stage breakdown
will determine the next exact candidate. Serving routing and its dynamic
shared-memory binding remain contingent on bitwise and timing evidence.
