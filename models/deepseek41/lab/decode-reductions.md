# Decode reduction timing

The `bf16_partials` lab target's opt-in bench
`timing::decode_reduction_timing` times decode and verify's served F32
reduction against the BF16-partial transport that prefill serves
([communication-partials.md](../communication-partials.md)) on four idle
GPUs at 1, 8, 16, 32, 64 and 128 rows of 5120 elements. It changes no
serving route, default, kernel or lock.

## Served forms

On native Flash TP4, serving's hidden-row all-reduces take these forms:

- Decode, target verification and draft blocks: `Kernels::new_for_kv`
  registers the rounding rows (`reduce::register`) and neither the phased
  F32 nor the low-latency round entries, so the rounded all-reduce takes
  its composed form at every row count: the F32 reduce-scatter and
  all-gather, then the round row. Where the next sublayer's HC mix takes
  the round (`ReduceOps::defer_round`), a step issues only the pair. That
  needs a next sublayer (`successor` in `bundle/src/lower/hc_expand_mix.rs`:
  within a layer, and into the next layer unless that is an Engram layer)
  and an admitted phase (`Lowering::has_expand_mix`):
  - Decode runs at graph buckets of 1, 2, 4, 8, 16 and 32 rows, and every
    decode phase is admitted.
  - Verify graphs keep their exact width, up to 192 rows. A verify phase
    is admitted only where `HcOps::has_finish_norm` admits it: the Flash HC
    table's `hc.finish_admit.verify` rows at the finish norm's capacity,
    the instance's verify rows capped at 64 (`Shape::verify_rows_within`,
    `verify_bounds::HC_FINISH`). The table admits 4, 6, 8, 12 and 16 rows and
    32 rows up to that capacity, so verify defers at those widths and at
    every width from 33 to 64. Every other width, 128 rows included,
    issues the complete sequence.

  Before an Engram layer and before the head every step issues the
  complete sequence.
- Prefill, at every row count: `Kernels::new_for_kv` also registers the
  phased BF16 entries (`with_phased_bf16_all_reduce`), and
  `ReduceOps::prepare_prefill` prepares the BF16-partial all-reduce on
  them: the one-shot entry at 1 to 4 rows, the two-shot at 5 to 32, the
  two-shot with workers at 33 to 1023 and the turns entry from 1024.
  Where admitted, prefill DMA carries the same BF16 partials instead
  ([prefill-communication.md](../prefill-communication.md)).
- `Kernels::prefill_reduction_backend` registers the phased F32 rounding
  entries (`with_phased_f32_all_reduce`) only for Flash TP4 instances
  without BF16 partials (BF16 KV storage), whose prefill steps of at least
  512 rows use them. No decode or verify step serves them.

At the timed row counts, the form each served step issues at a boundary
with a next sublayer, and the primary baseline, are:

| Rows | Decode | Verify | Primary baseline |
| ---: | --- | --- | --- |
| 1 | deferred | complete | decode, `f32_deferred` |
| 8 | deferred | deferred | decode, `f32_deferred` |
| 16 | deferred | deferred | decode, `f32_deferred` |
| 32 | deferred | deferred | decode, `f32_deferred` |
| 64 | not served | deferred | verify, `f32_deferred` |
| 128 | not served | complete | verify, `f32_complete` |

The bundle's admission code is private to it, so the bench applies the
same rule through the contract's public tuning table and decode buckets.
Its CPU test `timing::timed_arms_launch_what_serving_issues` pins this
table and checks every verify width, so a change to the admission fails
it.

## Arms

- `f32_deferred`: the reduce-scatter and all-gather, a served step's
  reduction where the next HC mix takes the round.
- `f32_complete`: the reduce-scatter, all-gather and round, a served
  step's reduction where no HC mix takes it.
- `f32_control`: a second graph of the row count's primary baseline on
  the same operands, the in-job A/A control.
- `bf16_partials`, the candidate: the phased BF16-partial all-reduce as
  prefill serves it.
- `f32_phased`, a same-transport precision reference: the phased F32
  rounding entries, which share the candidate's transport.
- `f32_round`: the round alone, for reference.

For each served step at a row count, `bf16_vs_served_decode` or
`bf16_vs_served_verify` compares `bf16_partials` with the form that step
issues. The row's `primary_comparison` names decode's where decode has the
row count as a bucket, else verify's. `bf16_vs_f32_phased` isolates
precision on one transport. Every arm's complete step first matches its
exact oracle.

## Method and receipt

Each arm is a CUDA graph of 32 calls per rank, timed with CUDA events over
25 replays per visit after a shared gate releases the ranks. The arms are
visited in one order and then reversed, so each pair runs A1 B1 B2 A2. For
each row count the receipt gives the served forms, every visit's
slowest-rank median per call, the critical path and the launched symbols.
Each comparison reports d1, d2, their mean and spread as rates (positive
when BF16 is faster), the saved microseconds per call and the A/A bound,
the larger of the control's absolute mean and spread. A comparison is
resolved when d1 and d2 share a sign and the mean exceeds both its spread
and that bound.

One nvidia-smi query reads every GPU's enforced power limit in watts and
SM clock in MHz at the start, before every visit and at the end. A GPU
without both readings stops the job. Every sample's limit must equal that
GPU's first; any difference voids the job: the receipt lists the changes
and marks itself void, and the test fails. The harness's per-replay
telemetry stays in the receipt for reference; it reports the management
limit, not the enforced one, so it is not the power check.

Rows above the 40-row captures, and all rows when
`CHOPINFER_BF16_PARTIAL_INPUTS` is unset, use a synthetic finite pattern,
for timing only. Set `CHOPINFER_KERNEL_BENCH=1`, `CHOPINFER_KERNEL_GPUS` to
four idle GPUs and `CHOPINFER_BF16_PARTIAL_PROBE` to the `probe_sm120`
cubin pinned in kernel-lab's lock, which `kernel-lab stage-cubins` writes:

```sh
cargo test --locked --offline -p model-deepseek41-lab --test bf16_partials \
  timing::decode_reduction_timing -- --nocapture
```

Without the first two it prints a skip notice. Receipts go to the
`deepseek41-decode-reductions` subdirectory of `CHOPINFER_KERNEL_RECEIPTS`,
else of `$CHOPINFER_DATA/kernel-lab`, else of the system temporary
directory. Before opening a GPU, and again just before writing, the bench
resolves that final directory, a relative one against the working
directory and through any symlinks, the subdirectory included. It refuses
one inside the checkout or one that passes through a dangling symlink.
These are pair-lab timings: they locate work and never decide adoption.
