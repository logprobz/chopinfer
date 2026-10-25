# Packed fold coverage

The packed-weight migration is complete. Bundle comparisons use
`Sm120::with_packed_fold_weights` and packed dense weight banks. Historical
row-major and plain-layout controls retain their original symbols and locks.

The fold conformance, decode, stream, compact and fused suites keep their
oracle, poison, PDL, status and invariance checks. Stream permutation checks
still compare the load-time transform with the historical prepare kernel.
Shared-expert MoE comparisons pack all three dense fold banks.

## Production-shape parity

`packed_serving` owns one cubin with row-major and packed twins from the
bundle catalogs. It covers 367 distinct packed compute symbols in 506
comparisons across Flash TP1/TP2/TP4, deduplicating symbols shared by
multiple topologies. Decode entries exercise one row, a partial tile and
capacity. Large v2/prefill entries use 129 rows, which exercises full and
tail paths in split prefill. Exact-row entries use their smallest registered
bucket; retained unused tile/stage variants use a valid smallest bucket
for that symbol. Wide folds run both 16 and 32 rows.

Deliberately incorrect omit-first controls have no packed twin and are not
parity candidates. They remain in their existing lab families.

The CPU test checks every twin's full entry declaration and verifies that
the prepared launches cover every packed compute symbol. GPU parity compares
all data views, including status, workspace and fused quantized outputs.
Each of the twelve families also has a moved-strip negative control.
The parity run has a 600-second elapsed budget and fails on an incomplete
run instead of skipping remaining symbols. Its nominal upload estimate is
about 36 GB before paired-bank and negative-control overhead; allocation,
host packing and readback can dominate. No runtime estimate is measured
without an assigned GPU.
