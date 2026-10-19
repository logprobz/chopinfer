# Coarse admission for guarded position scoring

This candidate retains the certificate and exact repair in
`position_mma_guard.md`. It changes how often the certificate is evaluated.

Each CTA summarizes the packed query before its key loop. Let `qlo` and
`qhi` be the smallest and largest scale bytes across every head and group.
Let `qspread` be the largest within-head range of the four scale bytes.
Each key slot supplies `klo` and `khi`, its four scale-byte extrema.

The coarse certificate requires:

- Query and key scale bytes all lie in [2, 252].
- `qlo + klo >= 130` and `qhi + khi <= 369`.
- `qspread + khi - klo <= 9`.

For any head, every combined scale lies between the two combined extrema.
For any two groups, their combined-scale difference is at most that head's
query range plus the key range, hence at most nine. These inequalities imply
every condition of the original per-head certificate. No assumption about
codes, cancellation or observed activations enters the proof.

A warp handles eight keys. If all its admitted keys pass, it stores the MMA
dots directly. Otherwise it evaluates the original certificate for each
head/key pair. Both branches use the same ordered head-weighted sum and the
original SIMT repair for any rejected dot. A CTA vote skips the repair loop
and its final barrier when no slot needs repair. The vote itself publishes
repair flags and completes all shared reads before the next tile.

Packing, row fallback, page admission, compact mapping, score layout,
nonfinite status, tail writes and ordered selectors retain their original
semantics. The certificate uses the currently loaded packed scales and
needs no persistent key metadata or new allocation. All workspace remains
on the same stream as the original scorer; no new cross-stream sharing is
introduced.

Separate diagnostic entries count admitted slots, SIMT repairs, slots
passing the coarse certificate, and slots whose warp took the fine check.
Diagnostic atomics are outside timings. On 2026-10-19, the SM120 pair passed 312 score/status/ordered-selection
comparisons, 300 poisoned comparisons and three arithmetic negative controls.
The coarse certificate accepted all examined slots in 30 sealed captures.
Across captures, median time at one decode row fell by 6% at 32K and 8% at
128K; at 32 rows it fell by 31% and 41%. At one row and 8K it was unchanged
within measurement resolution. Serving uses the qualified cubin without
changing its bytes.
