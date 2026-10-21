# Router tile pair: wider expert tiles

The served router projection at 8192 rows is the certified projection
(router_norm.md, `f32_projection::certified`): tiles of 32 experts by 32
rows, 128 threads, a grid of 12 by 256 CTAs. Each CTA stages its 32 weight
rows and 32 BF16 activation rows per 64-K half, so every activation row
block is read 12 times and the 3.9 MB of router weights once per row block.
At 8192 rows that is about 1.0 GB of activation reads and 1.0 GB of weight
reads per call, about 2.0 GB from L2 in the 0.405 ms the zero path's
projection measured (round 8), about 5 TB/s. The FLOPs (32 GFLOP) take a
small part of that time, so the claim under test is that the projection
is limited by L2 traffic and that wider expert tiles shorten it.

## Arms

`gemm/exact_f32_certified_wide.cuh` (lab) stages E weight rows per tile
(`certified::wide_entry_spec`). Each warp runs the served warp's 16
features and rows for every 32-feature group of the tile, so each output
takes the same eight-instruction chain per 128 K, the same two merges and
the same fold; the SIMT path is the served one per 32-feature group.

| Arm | Experts per tile | Grid | Shared bytes | Activation reads | L2 reads per call |
|---|---|---|---|---|---|
| served | 32 | 12 x 256 | 32768 | 12 | about 2.0 GB |
| f32 | 32 | 12 x 256 | 17664 | 12 | about 2.0 GB |
| f64 | 64 | 6 x 256 | 24576 | 6 | about 1.5 GB |
| f128 | 128 | 3 x 256 | 40960 | 3 | about 1.26 GB |

The served plan reserves shared memory for the zero path's three component
planes; the certified tiles stage one. f32 is the served arithmetic at the
shared memory it uses, so it separates occupancy from traffic. Registers
(lock, no spills): served and f32 80, f64 118, f128 226.

If the projection were bound by L2 traffic alone, f64 would take about
0.75 and f128 about 0.63 of the served time: about 3.4 and 5.1 ms of the
13.75 ms per chunk the served profile charges to the projection. Those are
the expectations being tested, not results.

## Cases and criteria, fixed before measurement

Worker 0 takes captured layer 14, worker 1 captured layer 39: the 8192-row
chunk of the published MoE capture (prompt input-0, rank 0), whose MoE
input codes times their group scales are widened exactly to BF16, with
that layer's router weights from the checkpoint (`--prepare` of
router_wide_job.py). The router's own input rows were not captured; these
rows are the same tokens' MoE input, which differs from the router's rows
only by the input quantizer. Each worker also takes two synthetic cases
from router_norm.md's patterns with its synthetic router weights (experts
80 to 95 below the tensor domain): outliers at 8192 and 65 live rows
(worker 0), outliers at 2049 and clean at 8192 (worker 1). Their
uncertified rows and experts reach the SIMT path.

1. **Bitwise.** Over the same rows, flags (the certifying norm's: every
   widened element a tensor operand), weights and weight domain, every arm
   writes the served projection's logits byte for byte over the whole
   8192-row output, which is poisoned (0xa5) first.
2. **Timing**, report only: paired graphs of the served projection and
   each arm (5 rounds of 3 samples) on every 8192-row case.

The job fails on any difference, any missing or extra case or arm, zero
executed tests, no synthetic case reaching the SIMT path, a power cap
other than 250 W or an Xid on its GPUs.
