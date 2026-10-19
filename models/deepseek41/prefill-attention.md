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
thread, 1024 static shared bytes, no local memory and no stack. Two CTAs
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

CPU preparation, resource audits and planner tests are complete. GPU
bitwise and timing results are pending. There is no prefill attention
speedup claim or serving change until those results are available.
