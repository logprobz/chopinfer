# Narrow pair: the MoE-input norm without its widened write

At Flash TP4's 8192-row prefill steps the MoE-input norm runs certifying
(router_norm.md) and still writes its rows widened to F32 (`normed_wide`,
168 MB per call). The router's 64-expert certified projection
(router_wide.md) is their only reader in those steps: the TP4 journals of
the AR and drafter lifecycles show, after each certifying norm, only the
router reading `normed_wide` before the next norm overwrites it. The
router reads them only on its SIMT path (uncertified rows or experts
outside the tensor domain), which real banks and captured rows have not
reached.

The candidate drops the write:

- the bound certifying norm with no widened rows (`widened` null, which
  the served norm entry already supports); its other outputs are
  unchanged;
- the bound 64-expert router whose SIMT path widens the norm's BF16 rows
  itself (`certified::normed_entry_spec`, `FromNormed` in
  `gemm/exact_f32_certified_wide.cuh`) and never reads `activation`. The
  widened rows are the BF16 rows widened exactly, so the SIMT path stages
  the same F32 values and every output keeps its bits.

## Criteria, fixed before measurement

Worker 0 takes router_norm.md's clean pattern, worker 1 its outliers
(1e15, NaN and split rows, so uncertified rows reach the SIMT path), with
its synthetic router weights (experts 80 to 95 outside the tensor domain),
at 1, 65, 2049 and 8192 live rows over the 8192-row capacity.

1. **Bitwise.** From poisoned outputs and a clear status, the candidate
   writes the bound step's normed rows, codes, scales, status, flags and
   logits byte for byte; its widened rows stay poisoned; the bound step's
   widened rows equal its BF16 rows widened on every live row.
2. **Memory safety.** compute-sanitizer memcheck reports zero errors on
   both workers at 1, 65 and 2049 live rows (first phase).
3. **Timing**, report only: paired graphs (5 rounds of 3 samples) of the
   bound step (norm and router) and the candidate at 8192 live rows.

The job fails on any difference, a written widened row, any missing or
extra case, zero executed tests, no outlier case with an uncertified row,
a sanitizer error, a power cap other than 250 W or an Xid on its GPUs.

## Expected time

The write is 168 MB per call, about 0.1 ms at the memory rate, so about
2 to 4 ms per 8192-row chunk over the 40 MoE layers per rank, before
measurement.
