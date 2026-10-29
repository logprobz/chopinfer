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

The DSpark main-projection case N5120/K15360/G1 uses the serving generic
fold as its baseline. `Prepared::main` bypasses the prefill wrapper above
64 rows. The measured mixed steps retain three 192-row projections, so
comparing this case with the ordinary dense prefill wrapper would test the
wrong baseline. This case currently accepts only the upper candidate band.

`ranges::range_timing_gpu` compares the baseline with the packed compact
arm after full-output/status equality. Each width brackets its A/B CUDA
event rounds with A/A rounds on the identical baseline graph and operands.
The job must run on two owned idle GPUs, retain every sample, report A/A
spread and clock/power observations, and finish within five minutes. A
resolved lab gain is only a reason to continue qualification. Boundary
pilots do not admit the intervening row range or replace the PDL, chunk,
route-sweep, journal and served acceptance gates above.

Reserve all GPUs exclusively for paired timing. Running a second pair lab
on the other devices can disturb the host, PCIe traffic and power behavior;
an A/A inside either lab does not make that overlap a clean comparison.
Keep the idle check for every reserved device. A resident keeper without
a compute context does not require an idle-check exception.

On 2026-10-29, stage 2 of the TP4 N5120/K15360/G1 case passed all
385 widths 128..512 on both test devices, including the two neighbors.
The exclusive full-band jobs made 43,744 bitwise output/status comparisons
and 782 timings, including shard overlaps. Every width beat its bracketed
A/A spread. Recomputed gains were 86.81..94.07%, with A/A spread
0.25..13.84%. At 192 rows, stage 2 saved 2.750..2.763 ms per projection.

TP4 DSpark main projections use this existing stage-2 template at retained
inner rows 129..511. The tuning table owns the contiguous range. Other
rows and topologies keep the original launches and route tags. The
candidate option and its test overrides are removed by the adoption series.

On 2026-10-29, served exactness exercised all ten intended inner widths
128/129, 255/256/257, 383/384/385 and 511/512. The 20 greedy/sampled A/B
cases and two fresh A/A controls compared 38 retirement states and 1,007
full-vocabulary logit views, with zero masking. Each case asserted the
actual projection-row total and route use. That GPU run used the frozen
candidate install. The later integration fix changes only route reporting
and adds an operand-guarantees refusal; it preserves the measured launches
and arithmetic, so the exactness evidence still applies.

On 2026-10-29, a served DSpark C1 ABBA passed the declared table-row rule.
The 192-row target had median step times 186.270, 177.523, 177.741 and
185.439 ms in ABBA order. Its paired rate gains were 4.927% and 4.331%,
averaging 4.629% against 0.596 percentage points of spread, above the 1.5%
minimum. The paired step-time saving averaged 8.223 ms. Each visit provided
15 or 16 target steps and at least 17.7% time share, above the declared 5%
support threshold. The 256-row and six-row verify controls had resolved
gains of 0.696% and 0.099%; aggregate request rate gained 1.142%. There
were no resolved control regressions or coverage gaps. Both seed pairs
passed timed output equality. Energy was comparable at unchanged limits,
with a provisional mean J/token change of -0.613% and no veto.

The ABBA used the same frozen candidate install as served exactness. It
isolates this route's effect; it does not measure the combined speed of
later integration changes. Adoption still requires the audited journal
pins, full-history proof and CPU gates. No AR benefit is claimed.
