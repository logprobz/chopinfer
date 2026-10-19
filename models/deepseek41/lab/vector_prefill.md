# Vector staging experiment

The separate vector family preserves the scalar family's row and column
tiles, conversion instructions, barriers and eight k16 MMAs per chain.
It packs each thread's four adjacent BF16 pairs into one aligned 16-byte
shared store. The predecoded activation path also uses a 16-byte load.
The 144-byte shared row stride and every fragment offset are 16-byte
aligned. This replaces four scalar stores whose warp addresses use only
eight shared banks per instruction. It does not change the MMA layout.

The controls are pipe64, wide128 and decoded128. The qualifying comparison
used both the frozen scalar counterpart and serving, including scalar
decoded128 gate/up at E384/8192 with the previous down.
The CPU matrix asserts these baseline symbols against the bundle's actual
prepared operations. The same poisoned boundary replays, stamped parity,
and omitted-chain control apply. Gate/up and down wins are evaluated
independently, requiring more than 3% improvement on both devices and both
routing distributions.

Build with the standalone example's `OUT --initialize --vector` arguments
for the first lock and `OUT --verify --vector` thereafter. The ignored
`vector_gpu` test reads
`CHOPINFER_DEEPSEEK41_VECTOR_PREFILL_CUBINS`. Its job includes current
serving timings, scalar-control timings, phase clocks, and a staged symbol
inventory. Future comparisons follow the bundle's current serving selector.

The 2026-10-19 pair at 250 W passed all 96 normal, scalar-control and
stamped comparisons, all 384 boundary replays per comparison arm, and 32
omitted-chain controls. All outputs are bitwise. Vector decoded128 gate/up
qualifies at these exact Flash TP4 prefill buckets:

| Expert bank | Rows | Worst gate/up speedup across both devices and routings |
| --- | ---: | ---: |
| E384/top-6 | 8192 | 1.133x |
| E128/top-3 | 8192 | 1.110x |
| E128/top-3 | 4096 | 1.164x |

At 8192 rows E384 gate/up wins 1.133-1.178x, and E128 wins 1.110-1.468x.
E384 down wins on balanced routing, but only 1.007-1.012x on skewed routing.
E128 down loses about 3% on balanced routing. Down retains its existing
entry in every bucket. Other buckets and geometries retain their entries.

The bundle uses the same frozen vector declaration and cubin as the lab.
Its wrapper replaces only gate/up calls, retains the current output
quantizer, and owns disjoint conversion and queue regions. E128 at 4096
replaces four grouped gate launches; both 8192 buckets replace the column
gate. Preparation for broader phases stays on the existing path, so an
unmeasured tail cannot inherit an exact bucket's choice.

The binding audit compares 14 complete lifecycle and prefill journals and
rejects 28 deliberate geometry, operand and mapped-gather lifecycle defects.
Allocations and unrelated operations match after the checked gate replacement
and module-handle normalization. AR is covered in both KV layouts; DSpark
uses native KV. Current DSpark catch-up does not execute its E128 MoE at
prefill sizes, so E128 operand, scratch and quantizer preservation are checked
through typed launch comparisons at both prepared buckets. Its decode path
retains the existing entries.
