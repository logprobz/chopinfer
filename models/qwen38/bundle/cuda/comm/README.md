# Prefill hyper-connection communication

Above 32 prepared rows, prefill runs the phased pair
`chop_comm_qwen38_reduce_scatter_v2` then `chop_comm_qwen38_allgather_hc_v2`;
up to 32 rows it runs the one-shot entry. Both give the same bits.

The pair was qualified on 2026-10-04 at TP2 and TP4. Lab bitwise suites
covered poisoned delivery, cancellation inputs, skew and graph replay.
Fixed-batch logits and sampling were bitwise identical at C16 and C32 for
both rank counts, and serving A/B C1 outputs were bit-identical against
the earlier gather. Engine prefill throughput improved by 1.3 to 3.3% at
2K, 8K and 32K tokens with C1 and C8 at TP2 and TP4. Decode C1 time to
first token fell by 9%; time per output token was unchanged.

The scatter sums the ranks in rank order in F32. The gather publishes a
64-bit call-generation flag for each source and block after the sending
cluster has drained its stores. Workers visit their own rows first, then
sources in arrival order, and finish one (token, stream) group at a time
with the one-shot entry's inject and norm expressions.

The state (the contract's `buffers`) holds the one-shot banks, the scatter
slots and the F32 sums; its flags hold the scatter's flags and arrival
counter, then 32 gather flag slots per rank. It supports changing live row
counts and the short-row one-shot path. Scatter and gather in one pair must
use the same launch geometry. The bundle checks the spinning entries'
maximum grids for residency before launching.

The pair uses one worker per prepared row, rounded to eight CTAs, with 8
through 128 workers beside the eight-CTA sender.

The lab's prefill and skew cubins use the model-owned `lab/kernels.lock`.
The build enforces that lock separately and embeds it for receipt identities.
Moving these records does not change the cubins or the serving lock.
