# Short prefill projections

Flash TP4 target prefill selects the existing exact-row MXFP8 fold operations
for 1 through 64 rows when the caller owns backend workspace. Other
geometries, topologies and row counts retain their previous route. Calls
that overlap an output reduction without workspace also retain their route.
No new CUDA kernel or quantization point is introduced.

The phase remains prefill. Its query quantizer and the carried status and
output offsets remain in place. The short operations request both prefill
chunk invariance and one-row verify invariance. Packed stream state remains
restricted to decode and verify; prefill uses the ordinary decode vector
with the already sized backend workspace. The compute stream owns that
workspace through every operation's final launch.

The paired 250 W checks on 2026-10-21 cover nine shapes at 1 through 8,
16, 32, 48 and 64 rows. Each device runs 108 synthetic and 96 captured Q/O
cases, eight eager/graph comparisons per case, plus 18 memcheck cases.
Status, invalid inputs, inactive rows and output guards agree. All measured
rows win with both cold and warm caches; the weakest ratio is 0.727.
CPU checks cover every prepared row count through 64, including topology
and workspace-free fallback boundaries.

This affects short prompts, newly evaluated rows in multi-turn follow-ups,
and short upper-skip cone tails. In a recorded 32K prompt, seven 64-row
projection calls occur in the final chunk. Their paired lab sum falls from
1.471-1.491 ms to 0.218-0.221 ms. General traffic frequency and serving TTFT
are not measured. AR decode and verify selection remain unchanged.

The next serving exactness gate must include short rows, a continuation
from an existing prefix, and a long prefill whose cone reaches 64 rows.
The isolated comparison does not replace that serving check.
