# Exact HC serving replay

`served_kernels_bitwise_invariances_and_controls_on_gpu` dispatches the
bundle's prepared HC op, including its selected symbol, grid and argument
list. It loads the embedded locked cubins and requires their bytes to match
the frozen qualification artifacts. The CPU test compares both selected
plans and every argument against the separately measured recipes.

Set the corpus manifest and its digest in `CHOPINFER_HC_GRAM_CORPUS` and
`CHOPINFER_HC_GRAM_CORPUS_SHA256`. The legacy variable names identify the
shared authenticated corpus; this test uses exact comparisons only.
Set `CHOPINFER_HC_ROWS_ARTIFACTS`, `CHOPINFER_HC_ROWS_CUBIN` and
`CHOPINFER_HC_ROWS_PROBE` to the frozen artifact directory, row-producer
cubin and kernel-lab probe. Set `CHOPINFER_HC_ROWS_REPORT` outside the
repository. The artifact CPU test must pass before the GPU test.

On an idle GPU selected through `CHOPINFER_KERNEL_GPU`, run:

```sh
cargo test --locked --offline -p model-deepseek41-lab --test hc_exact_rows \
  replay::served_kernels_bitwise_invariances_and_controls_on_gpu \
  -- --exact --ignored --nocapture
```

The test must execute once and report 308 sites, 616 byte comparisons, 808
replay/control checks and 308 captured-output comparisons. Both widening
policies run. Controls cover chunking, reversed positions, replicated ranks,
concurrent private scratch, inactive rows, sticky status, special values,
random values and a changed-byte negative. It retains every numerical
mismatch and fails at the end. No GPU selected is a failure, not evidence
of success.

The two-row producer is selected only at 32 rows. Eight rows retain the
four-warp chain producer; both use the 32-warp fused tail. C1 and other row
buckets retain their existing serving paths. Larger row tiles remain lab
candidates.
