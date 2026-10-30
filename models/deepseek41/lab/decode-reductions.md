# Decode reduction timing

The `bf16_partials` lab target's opt-in bench
`timing::decode_reduction_timing` times decode's served F32 reduction
against the BF16-partial transport that prefill serves
([communication-partials.md](../communication-partials.md)) on four idle
GPUs at 1, 8, 16, 32, 64 and 128 rows of 5120 elements. It changes no
serving route, default, kernel or lock.

## Served forms

On native Flash TP4, serving's hidden-row all-reduces take these forms:

- Decode, target verification and draft blocks, at every row count:
  `Kernels::new_for_kv` registers the rounding rows (`reduce::register`)
  and neither the phased F32 nor the low-latency round entries, so the
  rounded all-reduce takes its composed form: the F32 reduce-scatter and
  all-gather, then the round row. In decode steps, and in verify steps
  whose HC finish-norm fusion is selected (`Lowering::has_expand_mix`),
  where the next sublayer's HC mix takes the round (`successor` in
  `bundle/src/lower/hc_expand_mix.rs`: within a layer, and into the next
  layer unless that is an Engram layer), the step issues only the pair
  (`ReduceOps::defer_round`). Elsewhere, including before an Engram layer
  and before the head, the round row runs as its own launch.
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
  512 rows use them. No decode step serves them.

## Arms

- `f32_served`, the served baseline: the composed pair decode issues with
  its round deferred.
- `bf16_partials`, the candidate: the phased BF16-partial all-reduce as
  prefill serves it.
- `f32_served_control`: a second graph of the served baseline on the same
  operands, the in-job A/A control.
- `f32_phased`, a same-transport precision reference: the phased F32
  rounding entries, which share the candidate's transport.
- `f32_round`: the deferred round alone. It is reported, never counted as
  work BF16 partials remove.

The primary comparison, `bf16_vs_f32_served`, is `bf16_partials` against
`f32_served`; `bf16_vs_f32_phased` isolates precision on one transport.
Every arm's complete step first matches its exact oracle.

## Method and receipt

Each arm is a CUDA graph of 32 calls per rank, timed with CUDA events over
25 replays per visit after a shared gate releases the ranks. The arms are
visited in one order and then reversed, so each pair runs A1 B1 B2 A2. For
each row count the receipt gives every visit's slowest-rank median per
call, the critical path and the launched symbols. Each comparison reports
d1, d2, their mean and spread as rates (positive when BF16 is faster), the
saved microseconds per call and the A/A bound, the larger of the control's
absolute mean and spread. A comparison is resolved when d1 and d2 share a
sign and the mean exceeds both its spread and that bound. Each GPU's
enforced power limit and SM clock are recorded at the start and end, and
its limit at every replay; a limit change voids the job.

Rows above the 40-row captures, and all rows when
`CHOPINFER_BF16_PARTIAL_INPUTS` is unset, use a synthetic finite pattern,
for timing only. Set `CHOPINFER_KERNEL_BENCH=1`, `CHOPINFER_KERNEL_GPUS` to
four idle GPUs and `CHOPINFER_BF16_PARTIAL_PROBE` to the `probe_sm120`
cubin pinned in kernel-lab's lock, which `kernel-lab stage-cubins` writes:

```sh
cargo test --locked --offline -p model-deepseek41-lab --test bf16_partials \
  timing::decode_reduction_timing -- --nocapture
```

Without the first two it prints a skip notice. Receipts go to
`CHOPINFER_KERNEL_RECEIPTS`, else `$CHOPINFER_DATA/kernel-lab`, else the
system temporary directory, never the tree. These are pair-lab timings:
they locate work and never decide adoption.
