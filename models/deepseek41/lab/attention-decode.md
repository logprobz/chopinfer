The decode staging pair compares additive candidates against the qualified
Flash TP4 source schedule. It keeps eight numerical splits for decode and
verify, all 64-key softmax tiles, increasing 128-channel QK chains, PV key
order and the unchanged ordered merge. Prefill retains one split. Decode
and prefill therefore keep their existing numerical difference.

The matrix covers 64, 128, 256, 512, 1K, 8K, 32K and 128K visible contexts,
window-only and both compressed page formats. Short contexts bracket the
profiled decode requests; longer contexts fill the sparse selection. Dispatches are one-row decode, eight-row decode,
and exactly 32 verify rows from eight sequences with four rows each. The
previous six-row-per-sequence fixtures remain available. Selected prefix
pages are shuffled across the full cache allocation; stale cells stay
poisoned. Boundary cases include holes, unaligned payload strides, inactive
rows and both window/page boundaries. Every candidate must reproduce every
source-partial and final-output byte over two poisoned replays.

A selected window cell occupies 528 bytes and a compressed cell 288 bytes.
At full selection, one sequence reads 128 window cells and 512 compressed
cells: 215,040 unique payload bytes per mixed layer. Eight decode sequences
need 1,720,320 bytes. The 32-row verify fixture reuses its sparse prefix and
needs 1,732,992 unique bytes, while its per-row traffic sums to 6,881,280
bytes before head duplication. These are selected-payload lower bounds,
not hardware DRAM counters. Cache-line, query, list, partial and output
traffic are additional. The lab reports unique bytes and repeated staged
payload separately, normalized to 1 TB/s.

The reference reserves 83,200 shared bytes per CTA and stages KV separately
for each head at small batches. The candidates load query fragments directly,
use 75,008 shared bytes with a complete tile or 42,240 with half-tile staging,
and compare adaptive, two-head, four-head and eight-head groups. Half tiles
retain the complete softmax tile and every per-output MMA chain; they decode
KV again for PV. They trade additional widening for potential CTA residency.
No candidate changes the numerical split count to increase occupancy.

Packed widening multiplies BF16 pairs only where the product is exactly
representable. FP8 power-of-two scales use a conservative normal-product
range; extremes and NaNs retain the reference F32 conversion path. The GPU
probe checks every code/scale combination in both operand positions, including
signed zero paired with a finite opposite-sign value and non-finite inputs, and requires a mutated probe to disagree.
A separate sink-multiplicity control must also disagree with attention.

Task bodies accept a virtual CTA index and caller-owned shared arena. Only
entry wrappers contain grid dependency calls. Each CTA owns its partial;
independent streams need disjoint partial/output/stamp buffers. These lab
candidates introduce no serving state or cross-stream sharing. New artifacts
belong to `latent_decode_staging`; existing locks and serving selections stay
unchanged until GPU qualification identifies a winner.

Cold timings report sources alone and source plus merge. Merge timing follows
a source preparation outside the event interval. Repeated timings capture
32 complete attention calls in each graph to reduce event quantization; the
first call follows eviction and subsequent calls reuse the same data. Report
these separately from cold latency. Sampled entries retain their own resource
reports and compare bits again before recording per-CTA cycle totals for
query setup, addresses, widening, QK, softmax and PV. Clock intervals measure
one observer's progress and include synchronization, not hardware stall
counters. Resource queries report driver attributes and theoretical active
CTAs per SM for the exact block/shared-memory request, plus estimated waves.

Compilation and CPU checks outside the GPU queue use `tools/dev/heavy.sh`.
The GPU runner controls CPU affinity; its pair job invokes the frozen
`attention_pair` binary directly with `gpu_decode --ignored --test-threads=1`.
Its dry run verifies every module and symbol, launch shared-memory opt-ins,
coverage parser controls and the kernel-lab timing probe. Actual byte-floor
ratios, bottleneck attribution and speedups require the allocated GPU run.

On 2026-10-19, the pair passed all nine variants at 250 W: 72 matrix cases,
126 boundary cases, 1782 instrumented comparisons, 262144 widening pairs
and four negative controls. Partial buffers and final outputs matched over
poisoned and graph replays. The eight decode/verify splits and one prefill
split retain their existing relation.

The full 64-key tile with ordinary F32 widening and adaptive head grouping
was the leading candidate. Median paired source-plus-merge ratios were
1.000 at one row, 0.9365 at eight rows and 0.9320 at 32 verify rows. The
latter two won all 24 tested context/format combinations each. Half-tile
staging doubled theoretical resident CTAs but repeated widening and shared
staging; its adaptive variant's corresponding ratios were 1.356, 1.250 and
1.0663. For the leading full-tile candidate at 1K context, staging consumed
76-81 percent of sampled mixed-source CTA cycles, including synchronization.
These are instrumented intervals, not hardware stall counters.

The host was compiling during the run, and the job has no host-quiet
acceptance gate. Its timings remain informational. Keep serving unchanged
until a quiet repeat confirms the eight-row and 32-row candidates. The
shared preparation adapter can declare exact row/entry pairs, preserving
reference validation, unlisted rows, shared-memory opt-ins and the ordered
merge. Adding that adapter does not select a serving route.
