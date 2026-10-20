# Retained prefill output fusion

The native mixed64 prefill route now uses the existing fused head kernel
for all twenty retained counts, 64 through 2496 in steps of 128, at chunk
offsets 0, 8192, 16384 and 24576. Each tail must end at row 8192 of an
admitted consecutive single-item chunk. Full native and 4096-row part
admission keep their previous rules. The binding changes no CUDA source,
kernel lock, per-head arithmetic, 64-key softmax unit or merge order.

The fused kernel shares each staged key tile across sixteen heads and
applies the single-split merge, sink, division, BF16 boundary and inverse
RoPE before storing output. Its optional F32 partial pointer is null. The
full mixed32 decoded-cache route already has this behavior; adding another
output fusion there would remove no partial traffic. An unfused full
8192-row partial is 269,484,032 bytes, or 538,968,064 bytes for its logical
write/read round trip. Counting BF16 Q and output adds about 0.27 GB.

On 2026-10-20, four pair jobs at the 250 W cap passed 220 captured-selection
cases and 220 timing records. Checks compare F32 partials and BF16 outputs
bitwise, poison inactive storage, change queries and live counts, exercise
invalid inputs and holes, and compare partitioned execution. The inputs
use captured selections with synthetic local Q/KV. Eight small sanitizer
cases passed with zero errors. No claim about served token equality or
TTFT follows from these kernel probes.

The table takes the median of the two assigned ranks. Mixed32 extrapolates
one full layer across eighteen layers, including decoded-cache production.
Retained savings sum layers 20 through 38 in nonterminal chunks and through
39 in the terminal chunk, matching the recorded serving profile. Individual
timings use three samples after one warmup; pair spread is not a confidence
interval.

| Chunk offset | Arena mixed32, ms | Prior retained, ms | Fused retained, ms | Retained saving, ms |
|---|---:|---:|---:|---:|
| 0 | 61.895 | 12.137 | 12.029 | 0.109 |
| 8192 | 45.592 | 19.716 | 11.862 | 7.854 |
| 16384 | 45.730 | 20.083 | 12.058 | 8.025 |
| 24576 | 44.412 | 12.185 | 12.019 | 0.166 |

The full mixed32 route has no separate merge or F32 partial round trip at
any of these offsets. At positive offsets, independent body ablations take
2.244-2.343 ms per layer and cache producers take 0.225-0.311 ms. Ablations
need not sum to the complete graph time because their cache states differ.

A separate decoded bulk-copy experiment was 27-35% slower than the existing
fused decoded body, including cache production. It retained one CTA per SM,
used 99,600 shared bytes instead of 83,200, and added 128 KiB of shared
traffic and twelve internal CTA barriers per 64-key unit. The added work is
consistent with the regression but is not a hardware-counter attribution.
It is excluded from this serving change. Its small decoded sanitizer cases
were deliberately unaligned and therefore exercised scalar fallback and
barriers, not the aligned bulk-copy instruction. Full pair checks exercised
aligned copies, but aligned sanitizer coverage remains missing.

`audit_retained_prefill.py` compares 99 complete phase journals with the
prior arena routes. It permits only 1408 paired-body-plus-merge replacements
by the existing fused body, with the same arguments and final output, sink
and RoPE views. Six mutation controls must fail. All other operations and
handles remain exact, and the eight lifecycle journal pins are unchanged.
The fused launch runs on the same stream as both replaced launches. No
allocation, workspace alias, fork, join or cross-stream arena use changes.

Serving acceptance still requires cold 32K token IDs and logprob bits for
both TP4 AR and DSpark, plus per-profile ABBA TTFT. The integration must
observe the fused native retained route at all four chunk offsets, preserve
native fallbacks outside the admitted counts and offsets, and keep the
capacity-32 profile's memory fraction unchanged.
