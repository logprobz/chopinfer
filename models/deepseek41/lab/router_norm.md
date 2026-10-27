# Served router-in-norm

At native Flash TP4's 8192-row prefill steps, the MoE-input norm writes the
router certificate alongside BF16 rows, codes, scales and row status.
The certified router reads those BF16 rows directly and uses the exact
SIMT fallback for uncertified rows or weight columns outside its tensor
domain. It replaces the separate zero-component prepare. Other row counts
follow the F32 table's admitted routes.

The served projection uses [64-expert tiles](router_wide.md) and
[omits the widened activation write](router_narrow.md). Both keep the
same logits and row-status behavior. The norm certificate is true only
when every widened BF16 element is an exact F32 tensor operand.

On 2026-10-21, Flash TP4 pair qualification at 250 W compared full-capacity
outputs, flags and poisoned tails for clean, nonfinite, tiny and split rows,
including prior status, eager repeats and graph replays. Memcheck passed.
The certify-all mutation is required to change split-row logits while
preserving certified rows; nonfinite or single-term outliers alone cannot
witness a missing certificate. The certifying norm/router pair measured
0.880 of the previous norm, prepare and projection at 8192 rows. This is
kernel-pair timing, not whole-serving latency.
