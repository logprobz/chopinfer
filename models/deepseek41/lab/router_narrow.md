# Served router without widened rows

At native Flash TP4's 8192-row prefill steps, the certifying MoE-input norm
writes BF16 rows, codes, scales, status and certificate flags without a
widened F32 activation buffer. The 64-expert certified router widens BF16
rows itself on its SIMT fallback. Those values are exactly the omitted
widened rows, so both tensor and fallback logits retain their bits.

The 2026-10-21 pair evaluation at 250 W covered clean and outlier rows,
including nonfinite, tiny and split values, at live counts 1, 65, 2049 and
8192. Bitwise outputs and poisoned unwritten regions passed, as did
memcheck. The norm/router pair measured 0.89 of the pair with the widened
write at 8192 rows. This removes about 168 MB of writes per full call;
the pair timing is not a whole-request speed claim.
