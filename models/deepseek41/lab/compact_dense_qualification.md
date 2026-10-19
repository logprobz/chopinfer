# Compact native FP8 prefill criteria

The serving baseline already uses native block-scaled FP8 MMA. This candidate
changes only tile geometry, operand and scale staging, and prepared host
plans. It is unselected. It is not a decoded-BF16-to-native comparison.

Keep the exact trained E4M3/32 ceil-UE8M0 activation bytes, admitted row-major
FP8 weights and per-row 1x32 scales. Every output uses four ordered k32 native
MMAs from zero per 128-K chain and ascending-K F32 outer additions. There is
no activation requantization, sub-F32 partial sum or TF32.

Before any measurement, require bitwise equality against the current serving
route for all declared projection shapes and topologies, full/partial/zero
live replays, poisoned eager and graph repeats, and 2048/8192/tail chunk
partitions. Status bits, invalid-row refusal and inactive output bytes are
part of equality. A changed bit rejects this entry; no tolerance is added
after a mismatch.

This is a bitwise qualification on seeded admitted grid values, not a
within-bound replacement of decoded BF16 arithmetic. No real-tap capture is
needed while the unchanged chain passes strict equality. A shape that changes
bits is unqualified here. Any later within-bound proposal for that shape needs
separate real-input f64 evidence and the serving accuracy gates before routing.
The independent f64 oracle entries remain available for that investigation;
this bitwise job does not claim to execute them.

Each topology's unique shape runs at rows 1, 16, 17, 32, 33, 64, 65, 127,
128, 129, 257, 512, 2048, 4096 and 8192. Each has eager and graph poison
repeats. Rows 16, 65 and 129 additionally exercise the PDL live/input producer
and an omitted-fold negative control for every tile regime. Rows 257 exercise invalid codes, invalid
scales, sticky status and full/129/zero/full live replays. Every 8192-row
result is compared with partitions of 2048 and partitions ending in 65 and
257 rows. Timing uses 128, 512, 2048, 4096 and 8192 rows.

Retain current down/native-MoE queued jobs unchanged. The compact dense job
uses a separate sealed snapshot. GPU timing uses interleaved rounds and
exclusive kernel attribution at 250 W; a proposed routing win must beat the
current path by at least 3% on both devices. No serving route is selected
without those measurements and the required served integration checks.
