# Compact prefill row-range candidate

The `compact_dense` lab can qualify the existing compact template across
the proposed TP4 bands 65..127 and 129..511. It changes no serving route,
default, kernel source or lock. Stages 2 and 3 are existing template knobs.

Set `CHOPINFER_DEEPSEEK41_COMPACT_RANGE` to a closed JSON case:

```json
{"n":1280,"k":5120,"groups":1,"stages":2,"ranks":4,"first":65,"last":127,"neighbors":true}
```

`first` and `last` must fit inside one proposed band. `neighbors` adds
the immediately adjacent host widths. A full lower-band case has 65
widths including 64 and 128; the upper-band case has 385 including 128
and 512. Subranges support bounded qualification jobs. A smoke can select
only row 65 with `neighbors: false`.

The three arms are the explicitly selected TP4 baseline, row-major compact,
and packed compact. Baseline receipts retain their launch symbols, including
the existing matrix choices at 192 and 448. Every width compares all F32
output and sticky-status bytes under eager execution and graph replay,
full/partial/zero/restored live counts, poisoned inactive rows and invalid
activation codes/scales. Larger host widths also exercise live 127/128/129.
The existing compact lab retains its PDL rewrite, omitted-fold negative
controls and full-chunk partition checks.

Run the CPU configuration test and configured inventory through the build
wrapper. The configured tests are ignored by default because they require
the case environment:

```sh
cargo test --locked --offline -p model-deepseek41-lab --test compact_dense \
  ranges::range_configuration_covers_bands_and_neighbours -- --exact
cargo test --locked --offline -p model-deepseek41-lab --test compact_dense \
  ranges::range_inventory -- --exact --include-ignored --nocapture
```

Build the unchanged `lab/compact_dense_prefill` family with the existing
`build_family` example. An admitted GPU job sets
`CHOPINFER_DEEPSEEK41_COMPACT_DENSE_CUBINS` to that staged family and runs
`ranges::range_bitwise_gpu` with `--exact --include-ignored --nocapture`.
Require a nonzero `compact_range_complete` count matching the inventory.

This is the first qualification layer. It does not establish a range
admission or a speedup. Before production admission, cover every claimed
shape and row, chunk partitions, the current short-route neighbor controls,
and PDL producer/consumer behavior. Then add a default-off candidate or a
qualified row-range registration, with one route flip per commit and its
journal-proof specification. Run candidate-enabled route sweeps and served
full-logits/state exactness before served step-time acceptance. Pair timing
needs an in-job A/A control; no timing verdict comes from this bitwise test.

MiMo and Qwen can reuse the model-neutral MXFP8 template work only where
their arithmetic policies match. MiMo's served block-scale fragment
projections and Qwen's lab-only MXFP8 experts do not inherit a DeepSeek
dense-route admission. Their served paths require separate qualification.
