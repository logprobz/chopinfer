# Exact F32 row tile fallback

The exact F32 row tiles (`gemm/exact_f32_rows.cuh`) now run the SIMT
fallback of a 16-feature weight tile outside the tensor domain with every
thread of the CTA, staging its weight and activation lines through the
CTA's ring. Only the payloads of the row tile modules change:
`f32_verify_rows_deepseek41_sm120` and `f32_decode_rows_deepseek41_sm120`
keep their entries, parameters, shared memory and launch shapes, and every
output keeps its bits.

The [change specification](f32-row-tile-fallback.json) is the identity
specification of every fixture with one authenticated module replacement:
the verify row tile module, which the C16 odd-width and C32 maximum
allocation fixtures load under the `wide_verify` option, changes its
payload from the previous lock's SHA-256 to the new lock's, with no renamed
or removed symbol and the locked resolve order. No fixture loads the decode
row tile module. Every journal, create range and device history is
otherwise identical: the proof from the lab commit before the change to the
staged fallback compared all 25 journals and their full histories.
