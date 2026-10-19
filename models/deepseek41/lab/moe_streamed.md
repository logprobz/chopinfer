# Streamed MoE decode candidate

The TP4 candidate streams each active expert's MXFP4 weight strip once for
all its routed rows. It has separate queues for experts with 1..8 and 9..64
rows. A CTA computes 64 output columns and one 128-K chain. Four adjacent
lanes load four 32-K groups with 128-bit loads, decode UE8M0 scales once per
group, and reuse the decoded weight operands across the expert's rows.
The last down chain at K=576 pads exactly 64 values, as the column kernel does.

Each chain starts at +0 and executes the same eight BF16 k16 MMAs. Scratch
stores each route's chain by ascending K block. A merge folds those chains
with `__fadd_rn`, then calls the existing clamp, SiLU, route weighting and
BF16 rounding epilogue. The existing ceil quantizer follows gate/up. Down
retains F32 output. The separate shared expert and final routed sum keep
their existing arithmetic and graph dependencies.

The candidate is compiled and locked under `routed_moe_streamed`. Serving
neither registers its entries nor loads its modules. The measured C1
active gate/up crossover and every fallback stay unchanged. Qualification
requires full graph parity and a paired speed win in each proposed bucket.
The table marks a candidate eligible only when its complete graph median
beats its paired old graph by more than 3%. This is evidence for review,
not an automatic routing change. Prefill and rows above 64 use the old
path in the streamed comparison arm.

## CPU audit, 2026-10-19

CUDA 13.2.86, sm_120f, Flash TP4, E384/top-6:

| Kernel | Registers/thread | Static shared bytes | Local/stack bytes |
| --- | ---: | ---: | ---: |
| Old gate/up, r16 | 80 | 21,828 | 0/0 |
| Active gate/up, r16 | 84 | 28,740 | 0/0 |
| Streamed gate/up, small | 80 | 3,232 | 0/0 |
| Streamed gate/up, wide | 164 | 18,688 | 0/0 |
| Old down, r16 | 63 | 12,612 | 0/0 |
| Active down, r16 | 67 | 19,524 | 0/0 |
| Streamed down, small | 64 | 3,232 | 0/0 |
| Streamed down, wide | 96 | 18,688 | 0/0 |

SASS contains 128-bit global loads in the streamed compute entries. The
old entries use scalar global loads. Active entries also contain vector
loads, but retain shared staging and four-warp chain handovers. At C1,
old gate/up has only 54 useful CTAs and down has 480. The active grid caps
at 376 CTAs; gate/up uses 216. Streamed gate/up has 2,160 independent chain
items and down has 2,400, with a small-queue grid capped at 1,504 CTAs.
These counts and compiler resources suggest more load concurrency; they
do not measure achieved occupancy or bandwidth. Register pressure in the
wide gate/up entry and partial-buffer traffic still need device measurement.

## Qualification and profiling

`moe_job.py` runs 50 complete serial-versus-overlapped graph comparisons
across the 384-expert/top-6 backbone and 128-expert/top-3 DSpark banks. Cases
include rows 1, 6, 8, 32 and 48, the 192-row verify boundary, prefill,
seeded routing, every expert active at 64 and 192 rows, and all 64 rows
routed to the same top-k experts. Replays poison shared and routed outputs
and streamed scratch, exercise full/one/zero/full live counts, and require
a dropped-join negative control to fail at the final MoE output.

Paired event timings run separately from profiling. Projection timings
include pack, both compute queues when present, and merge. The table gives
old, selected, block-warps and streamed timings, plus their ratio to a
weight-only bandwidth floor. That floor counts unique active expert codes
and scales at 1.6 TB/s. It excludes activation, scratch and output traffic;
warm-cache timings can beat the DRAM-based floor. The serving policy fails
qualification if any bucket is over 3% slower than its paired old graph.

After parity and timing checks, `moe_profile.py` collects Nsight Compute
LaunchStats, Occupancy, MemoryWorkloadAnalysis, SchedulerStats and
WarpStateStats for old, active and streamed kernels at rows 1, 6, 8, 32
and 48 on each bank. The profiles expose achieved occupancy, load traffic,
issue efficiency and dependent-load stalls. Profiling uses eager launches,
with cache flushing and clock control disabled. Its perturbed timings do
not enter the speed gate. Reports and CSV exports stay outside the tree.
The CPU `--static` mode saves resource tables and SASS without using a GPU.

The staged job runs two idle GPUs at a checked 250 W cap, records Xids via
`journalctl -k`, and terminates child process groups on SIGTERM. Its dry run
checks the pinned binary, test coverage, driver linkage and profiler
sections without launching CUDA work. No GPU result is claimed by this
CPU audit.
