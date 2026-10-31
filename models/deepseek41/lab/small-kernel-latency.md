# Small-kernel latency at C1 to C8

Where a DSpark step's GPU time goes at one to eight concurrent sequences
on native Flash TP4, what earlier fusion work did about it, and which
fusions that keep every value and every add are left. The first of them
is built as the `verify_row_fusions` candidate option (below).

## Method

Two nsys exports of the live service with the default `tp4-dspark`
profile, CUDA graphs traced per node, read on device 0 on 2026-10-31:

- C1: 703 steps, each six verify rows plus the drafter.
- C8: 209 steps of 32 verify rows (four per sequence). The same capture
  holds 551 plain decode steps at C32, 32 rows each, which run the decode
  fusions and give their served times.

Steps are split at the acceptance kernel. Each kernel is charged its
advance: how far it moves the end of the device's busy time
(`max(0, end - max(frontier, start))`). Idle is the gap before a kernel
when nothing runs. A kernel that runs beside another on a parallel graph
branch, or waits under programmatic dependent launch, advances little
however long it is resident, so advance is the critical-path cost, and
a kernel's raw duration overstates it.

## Inventory

Per step, device 0. Busy is the sum of advances.

| Category | C1 launches | C1 ms | C8 launches | C8 ms |
| --- | ---: | ---: | ---: | ---: |
| Routed and shared MoE | 302 | 4.17 | 323 | 13.39 |
| Dense projections | 500 | 2.87 | 407 | 4.11 |
| Attention (sources, query and window RoPE) | 274 | 1.66 | 273 | 2.73 |
| HC | 177 | 1.65 | 177 | 1.40 |
| Reduce-scatter and all-gather | 188 | 1.30 | 188 | 3.39 |
| Exact F32 (router, verify rows, head) | 112 | 1.00 | 111 | 2.55 |
| Quantize, norm and elementwise | 285 | 0.74 | 293 | 1.19 |
| Other | 30 | 0.23 | 43 | 1.33 |
| Sampling and acceptance | 3 | 0.05 | 3 | 0.21 |
| Engram | 3 | 0.02 | 3 | 0.02 |
| Busy | 1,874 | 13.68 | 1,822 | 30.33 |
| Idle | | 0.51 | | 0.61 |

The dense projections include the `_admit_v1` decode folds (141
launches and 0.48 ms per C1 step): they are the projections themselves,
admitting rows by status, not separate admission launches.

By stage at C1:

| Stage | Launches | Advance ms | Idle ms |
| --- | ---: | ---: | ---: |
| Drafter catch-up (ten small graphs) | 25 | 0.20 | 0.20 |
| Drafter forward (three layers) | 189 | 1.54 | 0.02 |
| Engram hash and sampling (not graphed) | 8 | 0.16 | 0.14 |
| Embedding and layer 0 | 49 | 0.31 | 0.01 |
| Layers 1 to 13 | 535 | 3.76 | 0.11 |
| Layers 14 to 39 and the head | 1,070 | 7.72 | 0.03 |

At C8 the drafter catch-up grows to about 176 launches and 1.98 ms, the
drafter forward takes 3.00 ms and the two target graphs 8.01 and
16.45 ms.

The latency-bound families, advance per step in us (launches per step):

| Family | C1 | C8 | Per launch C1 / C8 |
| --- | ---: | ---: | ---: |
| HC producer (round chain at C1, round columns at C8) | 718 (77) | 340 (77) | 9.3 / 4.4 |
| HC finisher (twin at C1, split at C8) | 651 (77) | 761 (77) | 8.5 / 9.9 |
| Drafter HC lagged mix (forward) | 154 (6) | 150 (5) | 25.6 / 31.4 |
| Standalone HC mixers at graph starts | 80 (3) | 88 (3) | 26.6 / 29.5 |
| Reduce-scatter | 590 (88) | 1,353 (88) | 6.7 / 15.4 |
| All-gather | 621 (93) | 1,706 (93) | 6.7 / 18.3 |
| Query norm | 8 (43) | 145 (43) | 0.2 / 3.4 |
| Query ceil | 162 (51) | 216 (51) | 3.2 / 4.2 |
| Window RoPE | 94 (46) | 119 (46) | 2.0 / 2.6 |
| Window ceil32 | 155 (46) | 184 (46) | 3.4 / 4.0 |
| of which the drafter's serial ceil32 | 76 (6) | 81 (6) | 12.3 / 13.6 |
| Window store | 111 (46) | 128 (46) | 2.4 / 2.8 |
| Attention sources: cache decode | 155 (40) | 421 (39) | 3.9 / 10.7 |
| Attention sources: cached profile | 333 (40) | 790 (39) | 8.3 / 20.1 |
| Attention sources: heads merge | 101 (40) | 187 (40) | 2.5 / 4.7 |
| Attention output ceil | 72 (43) | 190 (43) | 1.7 / 4.4 |
| Query RoPE heads | 174 (43) | 213 (43) | 4.0 / 4.9 |
| Router finish prepare | 78 (40) | 122 (40) | 1.9 / 3.1 |
| Router finish | 253 (40) | 234 (40) | 6.3 / 5.8 |
| Router F32 projection | 106 (40) | 170 (40) | 2.6 / 4.3 |
| Shared SwiGLU | 115 (43) | 133 (43) | 2.7 / 3.1 |
| Routed gate finisher | 192 (43) | 601 (40) | 4.5 / 15.0 |
| Routed down finisher | 143 (43) | 460 (40) | 3.3 / 11.5 |

At C1 the query norm runs beside the window chain and is almost never
on the critical path. Of the 0.51 ms idle per C1 step, 0.20 ms falls
between the drafter's catch-up graphs, 0.14 ms around the ungraphed
Engram hash and sampling, and 0.11 ms before the layers 1 to 13 graph.

## Earlier fusion work

- The decode row fusions (served since 2026-10-19) fuse the query norm
  and ceil, the window norm, RoPE, ceil and store, and the HC finish,
  norm and ceil tails: 248 fewer launches per decode step and GPU (2,332
  to 2,084), bitwise, keeping the old reduction order. Only decode
  selects them; verify and prefill keep their chains, so DSpark steps,
  which are verify steps, never ran them.
- The decode HC branch (`l13-fuse-deepseek-decode-quantiz`, 2026-10-20)
  landed the finish, norm and ceil tail, verify HC, the HC tails at 8
  and 32 rows and row-tile planning; the column-block producer and split
  finisher served later take 1.05 against 1.20 ms per step at C1 and
  1.11 against 2.06 ms at C8 over 77 boundaries. Every mixer redesign it
  measured was bitwise and slower, and was rejected: a 160-CTA mixer
  (30.1 to 30.5 against 18.0 to 18.5 us at C1), tail mixers (12 us
  slower), a cluster mixer (estimated 24 to 26 us against a 13.5 us
  target) and a separate chain producer and finisher (31.7 to 33.0
  against 27.0 to 27.6 us, its finisher folds serially through global
  memory). A Gram-matrix norm fold was not bitwise.
- The prefill producer fusion branch (`l13-prefill-producer-fusion`)
  served the prefill query norm and ceil; its one-kernel prefill HC
  expand, mix, norm and ceil was rejected and retired on 2026-10-25.
- The fused candidates branch (`op-l4-fused-candidates`) fused the
  prefill indexer's block maxima, gather and merge into one publisher;
  served, prefill only.
- One launch for the all-reduce and the next HC expand, mix and norm is
  refused for now: it needs a 5120-wide transport and peer-residency
  admission.
- Left open there: decode buckets 2 and 4 do not take the fused HC tail
  (autoregressive decode at two to four sequences, not DSpark), a
  next-sublayer HC weight prefetch on the served pair (7.2 to 7.6 us per
  call between a cold and a hot call of the old mixer at C1), and the
  boundaries attention merge to ceil, SwiGLU to ceil, Engram to ceil and
  query merge to RoPE.

## Proposals

Fusions that keep each value, each add and its order, by expected gain
per step against risk. Estimates come from the served advances above.

| # | Fusion | Launches per step | C1 ms | C8 ms | Work |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Decode row fusions at verify rows of at most 32 | -120 to -123 | 0 to 0.05 | 0.15 | built (below) |
| 2 | Drafter window ceil: the parallel ceil32 or the fused window | 0 to -12 | 0.06 | 0.06 | admission |
| 3 | Attention heads merge and output ceil in one kernel | -40 to -43 | 0.05 to 0.07 | 0.12 to 0.17 | new kernel |
| 4 | Router finish prepare into the finish | -40 | up to 0.08 | up to 0.12 | new kernel, unread |
| 5 | Reduce-scatter and all-gather in one launch | -88 | 0.13 to 0.26 | 0.26 to 0.5 | new kernel, high risk |
| 6 | Drafter HC on the target's producer and finisher | 0 | 0.05 | 0.08 | unchecked |

1. The decode steps' fused query norm and ceil (5.30 us at 32 rows in
   the C32 graph) and fused window write (6.71 us) replace the C8
   chains (7.78 and 8.04 us per target layer): 0.10 and 0.05 ms per
   step. At C1 the six-row window chain advances 6.46 us per layer and
   the query norm is off the critical path, so the gain there is small
   and needs served timing to settle.
2. The drafter's window chain runs the serial `latent_ceil32_v1` (12.3
   to 13.6 us a launch) where the target runs the parallel entry (2.0 to
   2.6 us). The two compute each group with the same arithmetic, so the
   codes, scales and row status match
   ([wide-verify-ceil.md](../journal-changes/wide-verify-ceil.md), which
   moved wide verify steps to the parallel entry and left drafter
   forwards on theirs by scope).
3. The output ceil needs each row's amax over the merged heads. A merge
   that quantizes each row once its last head arrives (a per-row arrival
   count) keeps the merge's adds and the ceil's amax, scale and
   rounding, removing a launch and a round trip through global memory.
4. Prepare costs 1.9 to 3.1 us per layer before the finish. Its
   arithmetic is selection and integer work, likely exact, but has not
   been read.
5. The largest gain and the largest risk. Each rank would still sum its
   slice of the four partials in rank order and then gather the slices,
   so the bytes match; one kernel then waits on peer flags between the
   phases, which needs every CTA resident on every rank and care with
   programmatic dependent launch. A one-shot all-reduce stays ruled out
   here: reads from several peers at once collapse to about 2.3 GB/s on
   this host.
6. The drafter's lagged mix takes 25.6 to 31.4 us a launch against 17.7
   us for a target boundary's producer and finisher. Whether its
   arithmetic matches the target's pair has not been checked.

Two items that are not fusions may be worth as much: the HC weight
prefetch for the next sublayer (up to 0.3 to 0.5 ms at C1 if the old mixer's cold-to-hot gap
carries over; unmeasured), and the idle between graphs (0.1 to 0.2 ms
at C1 from fewer drafter catch-up graphs and a graphed Engram hash). At
C8 the routed finishers are the larger latency cost; the direct MoE
cache-policy candidates (the `moe_direct_*` options) address them and
await served acceptance.

## Verify row fusions candidate

The `verify_row_fusions` model option, off by default, lets verify
steps of at most 32 rows (`serving::DECODE_BATCH`) select the decode
steps' fused query norm and ceil and fused window norm, RoPE, ceil and
store (`Shape::row_fusions`). The entries are the decode ones every
instance already prepares at 32 rows, so no kernel, cubin or lock
changes, and off the package launches what it did. The fake-host test
`row_fusion_option_fuses_the_query_and_window_chains_through_32_rows`
pairs instances with the option off and on at 6, 12, 32, 33 and 64 verify
rows: through 32 rows each target layer's query pair and window chain
become one launch each on every rank, the drafter's queries fuse when
its rows fit and its window keeps its chain, and above 32 rows the
launch sequences are identical.

The norm and window lab pairs (`norm_quantize`, `window_fused`) now run
6, 12 and 24 rows besides the decode buckets, and the window pairs a
verify pattern: six-row items whose last positions are indexed by
sequence. On 2026-10-31, on four idle GPUs (each pair on two GPUs), the
query and input norm entries passed 336 bitwise cases with 120 byte
mutations and 24 epsilon controls detected, and the window entries 612
bitwise cases with 216 byte mutations and 36 arithmetic controls
detected, with no Xid.

Their single-launch timings resolve only to about 2.05 us, the step of
the event timer on these GPUs, and the samples sit on that step: the
fused window took 10.24 us against 12.29 us for the chain at every row
count from 1 to 32 with a read flush, and the fused query matched the
chain's 10.24 us except one step slower at 1, 2 and 12 rows on one GPU.
The estimates in the proposals therefore come from the served graphs. The option goes to served step-time acceptance at C1
to C8 before any adoption.

## Rerunning

Stage the frozen lab artifacts outside the tree with
`CHOPINFER_NORM_EXPORT=1`, `CHOPINFER_NORM_ARTIFACTS` naming the
directory and `CHOPINFER_NORM_TIMING_PROBE` naming the kernel lab's
`probe_sm120.cubin`, through the ignored
`norm_quantize::frozen_modules_are_complete`. Then run
`norm_quantize::bitwise_graph_live_rows_and_timing_on_gpu` and
`window_fused::window_bitwise_graph_live_rows_and_timing_on_gpu` with
`CHOPINFER_NORM_ARTIFACTS` and an idle GPU. The inventory needs an nsys
capture with `--cuda-graph-trace=node` exported to SQLite; the kernel
table gives each launch's device, start, end and graph.
