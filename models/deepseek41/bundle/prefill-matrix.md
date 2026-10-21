# Prefill matrix-load selections

On 2026-10-20, four native MXFP8 tile variants passed paired SM120 checks at
250 W. Each output retains four k32 MMAs from zero for every 128-K block,
then the ordered F32 fold. Matrix fragment loads replace scalar loads;
32, 64 and 128-row tiles vary register pressure, shared memory and reuse.

Flash TP4 selects 28 measured combinations of WQ_a, WQ_b, WO_a and WO_b at
192, 448, 960, 1472, 2496, 4096 and 8192 rows. Every selected combination
beats the served entry by at least 3% on both devices with cold and warm
caches. Unlisted rows and other topologies retain their previous entry.
The complete qualified cubin, including unused variants and controls,
retains its byte identity.

Qualification includes synthetic operands at every selected row count,
captured verify operands for all four families, and whole verify-row
repetitions at 2496 and 8192 rows. Eager and graph execution check status,
invalid and inactive rows, output guards and omitted-fold controls.
The synthetic 64-row prefill timing used the wrong serving dispatch and
is excluded. The separately qualified exact-row route through 64 rows is
described in [prefill-short.md](prefill-short.md).

Weighting paired event medians by the first chunk's selected calls gives
14.6-15.3 ms saved per 8192-row chunk. This is an additive kernel estimate,
not measured serving latency. Other cone row counts retain their fallback.
Engram and shared-expert synthetic wins lack complete real-site operands
and expected outputs, so they are not selected. Three decode candidates
passed correctness but gave no captured C1, C8 or verify32 crossover;
decode bindings remain unchanged.

The row adapter retains the fallback's capacity, workspace, weight layouts
and guarantees. Fake-host audits compare operand spans, status offsets,
row counts, stream dependencies and all other launches in native AR,
BF16 AR and native DSpark prefill. Corrupted operands and missing admission
calls fail the audit. Lifecycle, media and verify pins change only through
one added module, 108 resolves, nine shared-memory configurations and
109 releases per TP4 rank. Allocations remain unchanged.


On 2026-10-21, producer-staged scales passed paired bitwise checks against
these served entries. Flash TP4 selects the staged-scale kernel for all
four Q/O shapes at 4096 and 8192 rows. The 128-row, two-stage TMA tile keeps
four k32 MMAs from zero per 128-K block and the ordered F32 fold. Its
producer stages the original scale bytes with the matrix tiles.

The 8192-row warm sum for twenty calls of each shape improves by 4.6-4.8 ms
per chunk in the pair lab. This is an additive kernel estimate. Smaller
Q/O rows regress and keep their previous choices. The unchanged cubin also
contains an Engram candidate, which is excluded from default selection
until real-operand and served token/logprob-bit checks both pass.
