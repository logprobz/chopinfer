# Packed fold coverage

The packed-weight migration is complete. Bundle comparisons use
`Sm120::with_packed_fold_weights` and packed dense weight banks. Historical
row-major and plain-layout controls retain their original symbols and locks.

The package's former row-local, cluster and TMA-matrix reference modules
now belong to the model lab. Build them with the `build_family` example's
`lab/fold_local_rows_reference`, `lab/mxfp8_fold_cluster_reference` and
`lab/mxfp8_prefill_tma_matrix_reference` recipes. The reference locks keep
the former package cubin bytes and all 62 entry symbols. Fold-core fixtures
load these staged modules by their original names and verify their locked
hashes; served modules still come from the bundle.

Dense decode now serves one policy-based template. Its retired ticket,
prefetch, reuse and split-fold modules are lab-only references, built with
`lab/mxfp8_fold_ticket_reference`, `lab/mxfp8_fold_prefetch_reference`,
`lab/mxfp8_fold_reuse_reference` and `lab/mxfp8_fold_decode_reference`.
Their lock pairs move unchanged from the bundle, and staged module reads
check the original hashes. The old CUDA bodies and builders live only in
kernel-lab's reference sources.

The `dense_decode` twins parse the selected table rows, including the
separate runtime fold partition, and compare every selected row count.
Synthetic reuse cases cover half folds, padded group strides, paired banks
and partial live rows at partitions 2, 4 and 8. Relaunch, graph and PDL
phases poison outputs and partials and compare the complete workspace.

The fold conformance, decode, stream, compact and fused suites keep their
oracle, poison, PDL, status and invariance checks. Stream permutation checks
still compare the load-time transform with the historical prepare kernel.
Shared-expert MoE comparisons pack all three dense fold banks.

## Production-shape parity

`packed_serving` owns one cubin with row-major and packed twins from the
bundle catalogs and the retained decode reference catalogs. The 2026-10-23 parity run covered 367 distinct packed compute symbols in
506 comparisons across Flash TP1/TP2/TP4, deduplicating shared symbols.
Current coverage follows the served catalogs and template rows. Decode entries exercise one row, a partial tile and
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
