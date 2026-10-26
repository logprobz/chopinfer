# Retired fold reference modules

[The declaration](retired-fold-references.json) permits only removal of
three lab-only modules and their resolves. It changes no served kernel,
allocation, launch, argument, stream or dependency.

The row-local, cluster and TMA-matrix references remain reproducible through
`model-deepseek41-lab`'s `build_family` example, under
`lab/fold_local_rows_reference`, `lab/mxfp8_fold_cluster_reference` and
`lab/mxfp8_prefill_tma_matrix_reference`. Their locks moved from the bundle
to the model lab without changing any byte. The rebuilt cubins match the
former package hashes, covering 16, 16 and 30 entry symbols respectively.

The kernel consolidations retain the original bundle cubin bytes and entry
ABIs. The catalog tests preserve reference symbol coverage and launch plans;
external reference loading checks the moved lock's cubin hash. The journal
proof binds this declaration at the package-retirement revision and checks
all other captured operations exactly.
