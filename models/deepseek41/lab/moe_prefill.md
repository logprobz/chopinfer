# Grouped MoE prefill measurement

The prefill lab compares the locked path with separately locked grouped
8-, 32- and 64-row expert tiles. It keeps each eight-MMA 128-K chain, its
ascending-K F32 fold, clamped SwiGLU, route weighting, BF16 boundary,
ceil quantization and final routed/shared addition order. The candidate is
not registered for serving. GPU parity and timing are pending.

The table gives computed per-rank work for balanced, all-active Flash TP4
routes on 2026-10-19. Multiply FLOPs and bytes by four for the complete TP4
layer. GB and GFLOP are decimal. Weights include E2M1 codes and one UE8M0
scale per 32 elements. These are operation counts, not device measurements.

| experts/top-k | rows | gate/up GFLOP | down GFLOP | shared GFLOP | router GFLOP | unique routed weight GB | old row utilization |
|---|---:|---:|---:|---:|---:|---:|---:|
| 384/6 | 512 | 36.239 | 18.119 | 9.060 | 2.013 | 1.805 | 12.50% |
| 384/6 | 2048 | 144.955 | 72.478 | 36.239 | 8.053 | 1.805 | 50.00% |
| 384/6 | 4096 | 289.910 | 144.955 | 72.478 | 16.106 | 1.805 | 100.00% |
| 128/3 | 512 | 18.119 | 9.060 | 9.060 | 0.671 | 0.602 | 18.75% |
| 128/3 | 2048 | 72.478 | 36.239 | 36.239 | 2.684 | 0.602 | 75.00% |
| 128/3 | 4096 | 144.955 | 72.478 | 72.478 | 5.369 | 0.602 | 75.00% |

The stored routed formats are MXFP4 weights and MXFP8 activations; their
exact MMA operands are decoded BF16. The job measures register-only dense
BF16 m16n8k16 throughput at 250 W on each assigned GPU. Native FP4 or sparse
marketing rates are not used. It also measures a 512 MiB streaming read
and a warm 32 MiB read. Router F32 and shared MXFP8 FLOPs are counted
separately and do not enter the calibrated BF16 compute ceiling.

For each bank, the GPU suite covers 512, 2048 and 4096 rows with balanced,
seeded and concentrated routing. Synthetic BF16 router weights use an
identity prefix and dense common suffix to control routes through the
actual F32 router projection while keeping most weights nonzero. The
complete graph also includes input quantization, route selection and
permutation, gate/up, intermediate quantization, down, the shared expert
and final combine. The old path is timed before candidate comparison.

Each case compares fifteen outputs byte for byte after full, partial,
zero and full live-row replays, with poisoned outputs and scratch. A
separate down entry deliberately omits the first chain; its final MoE
result must differ. Restoring the candidate must restore equality.

Five interleaved rounds each contain five samples per arm and two graph
replays per sample. The reported value is the median of the five round
medians. Raw samples, expert counts, each component's time and the full
graph's time are retained. The report separates logical traffic, scheduled
weight reuse and extra weight-load instructions in the 64-row variant.
The latter uses two warp groups to avoid register spills.

The cold-DRAM roof assumes unique operands are fetched from DRAM; it is a
reference rather than a bound for cache-resident warm replays. The
optimistic L2 roof assumes all logical traffic hits L2. Both also include
the routed BF16 compute floor. Neither includes unpacking, quantizer,
router, shared-MMA or synchronization instruction costs. No prefill bucket
changes selection until GPU evidence establishes its parity and crossover.
