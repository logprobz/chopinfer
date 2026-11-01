# Small-kernel latency at C1 to C16

Where a DSpark step's GPU time goes at one to sixteen concurrent
sequences on native Flash TP4, what earlier fusion work did about it, and
which fusions that keep every value and every add are left. Proposals 1
and 2 were built as the `verify_row_fusions` candidate option and adopted
at six rows or fewer, proposals 3 and 4 were built, measured and retired
(one neutral, one slower), and
proposal 5 was prototyped in the lab;
[estimates](#estimates-at-c1-c8-and-c16) puts their expected gains side
by side.

## Method

Two nsys exports of the live service with the default `tp4-dspark`
profile, CUDA graphs traced per node, read on device 0 on 2026-10-31:

- C1: 703 steps, each six verify rows plus the drafter.
- C8: 209 steps of 32 verify rows (four per sequence). The same capture
  holds 551 plain decode steps at C32, 32 rows each, which run the decode
  fusions and give their served times.
- C16: an export of a C16 run from the same day, read the same way: 198
  steps of 64 verify rows.

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

A C16 step launches 2,087 kernels and is busy for 50.2 ms, with 1.41 ms
idle; its 95 reduce-scatters and 95 all-gathers advance 27.1 and 29.6 us
a launch (C1: 6.7 and 6.3 over 88 each; C8: 15.4 and 18.3).

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
per step against risk, as first estimated from the served advances above.
The built candidates' measured estimates replace these
([estimates](#estimates-at-c1-c8-and-c16)).

| # | Fusion | Launches per step | C1 ms | C8 ms | Work |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Decode row fusions at verify rows of at most 32 | -120 to -123 | 0 to 0.05 | 0.15 | adopted at 6 rows or fewer (below) |
| 2 | Drafter window ceil: the parallel ceil32 or the fused window | 0 to -12 | 0.06 | 0.06 | adopted at writes of 6 rows or fewer (below) |
| 3 | Attention heads merge and output ceil in one kernel | -40 to -43 | 0.05 to 0.07 | 0.12 to 0.17 | built, neutral, retired (below) |
| 4 | Router finish prepare into the router producer | -40 | up to 0.08 | up to 0.12 | built, slower, retired (below) |
| 5 | Reduce-scatter and all-gather in one launch | -88 | 0.13 to 0.26 | 0.26 to 0.5 | prototype, slower above 4 rows (below) |
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
3. WO_a's input quantizer, where WO_a folds row status, is the sliced
   ceil: one E4M3 scale per 32-column group, and each group lies within
   one head. So the merge CTA that writes a row's head can quantize that
   head's groups itself after a block barrier and OR their flags into the
   row status, as the sliced quantizer does. No per-row amax or arrival
   count is needed (the first draft of this note assumed one).
4. The prepare runs before the router F32 producer, not the finish: it
   classifies which rows the producer's tensor path may take (every K
   value a tensor operand) and resets the finish's arrival ticket, 1.9 to
   3.1 us a layer. The producer can classify its own rows if one CTA
   cluster covers every K of a token block.
5. The largest gain and the largest risk. Each rank would still sum its
   slice of the four partials in rank order and then gather the slices,
   so the bytes match; one kernel then waits on peer flags between the
   phases, which needs every CTA resident on every rank and care with
   programmatic dependent launch. A one-shot all-reduce stays ruled out
   here: reads from several peers at once collapse to about 2.3 GB/s on
   this host. Such a kernel already exists (below).
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

## Verify row fusions

The `verify_row_fusions` candidate option, off by default, let verify
steps of at most 32 rows (`serving::DECODE_BATCH`) select the decode
steps' fused query norm and ceil and fused window norm, RoPE, ceil and
store (`Shape::row_fusions`). The entries are the decode ones every
instance already prepares at 32 rows, so no kernel, cubin or lock
changed, and off the package launched what it did. The fake-host test
`row_fusion_option_fuses_the_query_and_window_chains_through_32_rows`
paired instances with the option off and on at 6, 12, 32, 33 and 64
verify rows: through 32 rows each target layer's query pair and window
chain became one launch each on every rank, the drafter's queries fused
when its rows fit and its window kept its chain, and above 32 rows the
launch sequences were identical. The fusions are now the default route
at verify steps of at most six rows (adoption, below), and
`verify_steps_fuse_the_query_and_window_chains_through_6_rows` checks the
default's launches at 6, 12, 32, 33 and 64 rows.

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
The estimates in the proposals therefore come from the served graphs.

On 2026-11-01, Flash TP4 packages of the candidate commit with the
option on replayed 13-step fixed-batch traces on four GPUs bit for bit
against the same package's recording without options (package-comparison
mode): verify traces at one request with 5 and 6 rows a step (67 logits
rows), at 16 requests with 16 and 32 rows (304) and at 16 requests with
48 and 64 rows (688), the last covering the drafter's window ceil alone.
Allocated replays beside `wide_verify` at 72 to 184 computed rows matched
too, in logits, proposals, acceptance, committed state and drafter
confidence words, and served exactness passed at C1, C8 and C16.

Adoption. In the served step-time A/B of the option against the profile,
with the first-visit control clear, C1 (one item, six verify rows a
step) gained 1.85%, resolved, with paired deltas of 1.74% and 1.95%, and
C8 (eight items, 32 rows) gained 1.45%, neutral, with paired deltas of
0.54% and 2.35% and a spread of 1.81 points; C32, which does not draft,
was untouched. Timing acceptance needs a resolved gain at every admitted
width, so on 2026-11-01 the fusions became the default route at the six
rows C1 resolved: the Flash TP4 latent tuning row
`latent.row_fusions.verify` admits rows 1 to 6, verify steps and the
drafter's forwards by their rows and the drafter's window writes by each
write's own rows, and verify steps of 7 to 32 rows keep the chains. The
drafter's forward of one drafting item is five rows, so it fuses its
queries at every verify step with one drafting item. The drafter's
forwards and writes are bitwise in the lab pairs and replays, and the C8
cell already ran a superset, every drafter forward of up to 32 rows
fused and every write of up to 60 rows on the parallel entry, without a
regression in any cell. The option, its test overrides and its sweep
profiles were removed
([adoption journals](../journal-changes/verify-row-fusions-adoption.md)).

## Drafter window ceil

`verify_row_fusions` also carried proposal 2. The drafter's catch-up and
block window writes were prepared at prefill capacity, so they launched
the serial `latent_ceil32_v1`. The option also prepared writes of at
most the draft rows, which select the parallel
`latent_ceil32_parallel_v1` the target's window takes; its groups
compute the same codes, scales and status. Wider writes, such as the
85-row catch-up writes at C8, kept the serial entry. In the C4 and C8
allocated journals the option's 36 serial launches became parallel ones.
The drafter's catch-up picks its writer by row count wherever it runs,
so the option also changed the catch-up of DSpark prefill steps and of a
prefix capture, which runs outside any target step; their journals were
pinned with the option ([verify row fusions
journals](../journal-changes/verify-row-fusions.md)). The adopted route
keeps that count rule at writes of at most six rows.

The serial and parallel ceil lab pairs (`ceil_parallel`) now also run 3,
5 and 12 rows. On 2026-10-31, on two idle GPUs, each GPU passed 4,695
bitwise cases over every serial and parallel ceil pair at 14 row counts
from 1 to 64 and detected all 939 negative controls. The ceil32 pair took 20.5
against 8.2 us for one launch. In the served graphs the drafter's serial
ceil32 advances 12.35, 13.56 and 13.60 us a launch at C1, C8 and C16
against the target's parallel 1.99, 2.56 and 2.74 us. At the 6.1, 6 and
12 launches per step the option moved, that saves about 0.063, 0.066 and
0.13 ms a step.

## Quantizing attention merge (retired)

Proposal 3 was built as the `attention_merge_ceil` candidate option,
measured and retired on 2026-11-01 because it was neutral within the
lab's resolution at C1 to C16 and added code without a gain: its option,
kernels, lock, journals, sweep profile and lab pair were removed, and
this section keeps the measurements. Each merge entry of the latent
sources had a quantizing twin: the merge stored every BF16 output
exactly as before, then after a block barrier each warp quantized its
head's 32-column groups with the parallel ceil's group function, stored
codes and scale codes and ORed its flags into the row status, as the
sliced quantizer does. Where WO_a ran as a status fold, every decode and
verify attention of at most 64 rows, the target's and the drafter's,
launched the twin in place of the merge and the sliced quantizer.

The lab pair compared each twin with the served merge and sliced
quantizer. On 2026-10-31, on two idle GPUs, each GPU passed
2,492 bitwise cases (14 patterns, every decode twin at 11 row counts up
to its 32 rows and every verify twin at 14 up to 64) and detected all 56
negative controls; compute-sanitizer
memcheck and synccheck found no errors and racecheck no hazards, with no
Xid. Single launches sit on the timer's 2.05 us step. In chains of 16
launches the twin took 3.84 against 3.58 us a layer at 6 rows with the
32-group quantizer C1 serves, 6.40 against 5.89 us at 32 rows and 9.60
against 10.11 us at 64 rows with the 16-group quantizer C8 and C16
serve. At 43, 43 and 46 merges a step that is 0.011 and 0.022 ms slower
at C1 and C8 and 0.024 ms faster at C16, within what the chain timing
resolves.

On 2026-11-01 the same fixed-batch traces as `verify_row_fusions`' (5 to
64 rows a step) replayed bit for bit with the option on, and a
one-request AR decode trace recorded with it equalled its recording
without it in every step frame and logits row (13 rows). The host
refuses to destroy an instance after an AR trace, so the two AR
recordings were compared directly.

## Classifying router producer (retired)

Proposal 4 was built as the `router_prepare_fusion` candidate option,
measured and retired on 2026-11-01: its option, kernel, lock, journals,
sweep profile and lab pair were removed, and this section keeps the
measurements. The classifying producer was the cached exact F32 decode
regrouped into eight CTAs of five warps (the 40 chains of K 5120), one
cluster per token block and feature tile, since the runtime's portable
cluster limit of eight cannot hold the served ten CTAs of four warps.
Each warp stages and checks its chain, the CTA and then the cluster AND
the row flags through distributed shared memory, rows outside the domain
are zeroed in shared memory (the zeros decode stages for them) and take
the SIMT path, and block (0,0,0) reset the finish's ticket. Each target
layer of a step of at most 32 rows launched it in place of the prepare
and the producer; the drafter's router and wider steps kept both.

The lab pair passed 260 bitwise cases (13 row
counts, 10 patterns, tensor and tile-out weights) and detected both
negative controls on each of two GPUs on 2026-10-31, with clean
sanitizers. It was slower: in chains of 16, 7.42 against 6.91 us a
layer at 6 rows and 22.4 against 14.5 us at 32 rows, about 0.020 ms a
step at C1 and 0.32 ms at C8; C16's 64-row steps did not take it. The
regrouped grid and its cluster exchange cost more than the launch saved,
and the lab did not separate the two. No fixed-batch replays were run
for it. Computing the row domain where the router's input rows are
produced, upstream of the producer, would avoid the cluster.

## One-launch deferred reduction

Proposal 5's kernel already exists and is model-neutral. The phased F32
all-reduce (`kernels-sm120`'s `comm/allreduce_phased_f32.cuh` and
`comm::phased_f32`) runs the reduce-scatter and the all-gather in one
launch: each element's owner sums the ranks' F32 partials in rank order
from zero and the F32 sums are pushed to every rank, one destination at a
time. Its F32-output entries (`chop_comm_allreduce_phased_f32_v1`, and
`chop_comm_allreduce_phased_oneshot_f32_v1` for calls of up to 262,144
elements) give the bits of the composed pair
(`chop_comm_reduce_scatter_f32_v1` then
`chop_comm_allgather_scattered_f32_v1`); kernel-lab's `comm_phased_f32`
tests check that at 1 to 8,192 rows of 4,096 and catch its mutants,
generations, replays and epoch-tag reuse. Its rounding entries are what
the `phased_decode_reduce` candidate serves where a step issues the
complete sequence ([decode reductions](decode-reductions.md)). Where the
next HC boundary takes the round, which is 77 of each rank's 81 target
reductions at the 6-, 32- and 64-row steps of C1, C8 and C16, the
composed pair stays: the rounding entries write BF16 rows, and this
bundle's comm cubin holds no F32-output entry (glm5next's package holds
all four).

Design. A deferred phased form launches the F32-output entry of the row
count's phased protocol and geometry (16 sending CTAs, the one-shot at
1 to 4 rows and the two-shot above, with 32 summing workers above 32
rows)
into every rank's F32 sum region, which the composed form already keeps
on every rank for its all-gather, and leaves the round to the next HC
mix exactly as now. `PreparedOp::defer_bf16_round` promises that the
deferred calls are the step's calls before its round, so the form's
complete step would be that entry followed by the round row, and
`phased_decode_reduce`'s rounding entries would keep serving complete
steps from their own op, as its alternate op does today. In
`kernels-sm120` that is one more form of the rounded all-reduce
(`backend/rounded_all_reduce.rs`) behind a registration, on the phased
state and the composed form's sum region, with no CUDA change and
nothing model-specific, so it stays model-neutral in
`crates/kernels-sm120/cuda/comm/` and the backend. The bundle would add
the two F32 entries to its phased cubin (new lock files only), admit
their grids at create as it admits the rounding entries' (every CTA must
be resident on every rank) and register the form behind a candidate
option or as part of `phased_decode_reduce`.

Bitwise risk. Low, and of the kind the lab can settle. The entry sums
every element in the same order from zero as the pair, so the F32 sums
match in every bit; the round stays the HC mix's single
`cvt.rn.bf16.f32` of the same sum, so precision and the number of
roundings do not change. The prototype passes no live count, so it sums
every row as the pair does. The HC producers and finishers that read
the sum return at rows past the live count, so passing it would likely
be safe and skip dead rows, but would leave their sums stale; the
prototype does not rely on that. The protocol's own risks (residency,
programmatic dependent launch, epoch tags) are those of the rounding
entries `phased_decode_reduce` already serves on the same state.

Prototype. The lab's `bf16_partials` test holds it (`phased_deferred`):
a wrapper of the candidate registration's rounded all-reduce whose
deferred round follows the F32-output entry of the same protocol,
senders and workers, writing every rank's sum region, loaded from
kernel-lab's locked `comm_phased_f32_sm120` cubin
(`CHOPINFER_PHASED_F32_CUBIN`).
`phased_deferred::phased_deferred_matches_the_deferred_pair_on_gpu`
compares every rank's F32 sum of every row with the deferred pair's and
with the host's rank-ordered F32 sum, at the 11 parity row counts and
at 6, 10 and 13 rows, on the 11 adversarial patterns of
`phased_parity`, with every row live and with the last row dead, and in
graph replay. `timing::phased_deferred_timing`
(`CHOPINFER_KERNEL_BENCH=1`) times the pair, a second graph of the pair
as the A/A control, the prototype and the phased rounding entry at the
bench's 16 row counts, after checking the prototype's sums against the
pair's at each.

On 2026-11-01, on four exclusive GPUs at an enforced 250 W limit that
no sample changed, the parity test passed all 297 cases (14 row counts,
11 patterns, every row live and the last row dead) and every graph
replay: each rank's F32 sum of every row equalled the deferred pair's and
the host's rank-ordered sum, and nothing past the rows was written. The
cases held 1.2 million NaN sums (the GPU's canonical NaN in both forms,
whose payload the host's sum does not share), 5.4 million infinite and
3.0 million subnormal sums. The design keeps the exact rank order and
the single rounding.

It does not pay where it would serve. Per call (slowest-rank median)
against the pair, the prototype was faster at 1, 2 and 4 rows (6.9
against 11.6 us, 7.9 against 11.8 and 11.0 against 12.2), slower at 5
to 16 rows by 1.2 to 2.1 us, unresolved at 24 and 31, and slower at 32
rows (31.9 against 29.3), 48, 64 (54.6 against 51.0), 96, 128 and 192
(156.8 against 135.6); every other comparison resolved past the A/A
bound. The rounding entry of the same protocol and geometry, slower
than the pair at 5 to 8 rows, beat it from 12 rows on (26.5 against 29.3
us at 32 rows, 44.0 against 51.0 at 64 and 121.8 against 135.6 at 192).
The two entries differ only in gathering F32 rather than BF16 rows,
twice the bytes, so the F32 gather is what loses to the pair's
all-gather. At the served widths the
deferred form would cost 1.4 us a reduction at 6 rows, 2.7 at 32 and
3.6 at 64: about 0.11, 0.20 and 0.28 ms a step over 77 reductions at
C1, C8 and C16. It saves 4.7, 4.0 and 1.2 us at the 1-, 2- and 4-row
autoregressive decode buckets (about 0.36, 0.31 and 0.10 ms a step),
which DSpark steps do not run.

So the F32-output form should not serve 5 to 192 rows, and this branch
pursues neither direction that keeps the order and the single rounding:

- At 1 to 4 rows, the deferred phased form above pays for autoregressive
  decode ([decode reductions](decode-reductions.md#phased-f32-timing)
  estimated about 0.35 ms of a C1 decode step); it exists as the
  `phased_decode_defer` candidate on another branch.
- Follow-up for the DSpark widths: the HC boundary could take the
  rounding entry's BF16 rows instead of rounding the F32 sum itself. The
  HC mixes convert the sum with the same `cvt.rn.bf16.f32` before any
  other arithmetic, and the rounding entry's owner rounds the same sum
  once with the same instruction, so a fused HC producer that reads
  those BF16 rows computes the same bits. At the rounding entry's
  measured advantage (0.8 us slower at 6 rows, 2.8 faster at 32 and 7.1
  at 64) that is about 0.06 ms slower at C1 and 0.21 and 0.54 ms faster
  at C8 and C16, before the HC entries' own saving from reading half the
  bytes. It needs BF16-input variants of the shared HC producers that
  take the round (`crates/kernels-sm120/cuda/sinkhorn_hc/`), which are
  model-neutral as well, and their own lab pairs.

## Estimates at C1, C8 and C16

Expected savings per step on device 0, in ms (negative is slower), from
the served advances where a candidate removes or replaces a launch the
captures show and from the lab chain timings where it changes one. C1
steps verify 6 rows, C8 steps 32 and C16 steps 64. None of these is
served step-time evidence, which adoption still needs.

| Candidate | C1 | C8 | C16 | Basis |
| --- | ---: | ---: | ---: | --- |
| `verify_row_fusions`, target query and window chains | 0 to 0.05 | 0.15 | 0 | served advances; steps above 32 rows keep the chains |
| `verify_row_fusions`, drafter window ceil | 0.063 | 0.066 | 0.13 | served advances of the serial and parallel ceil32 |
| `attention_merge_ceil` (retired) | -0.011 | -0.022 | 0.024 | lab chains, 43, 43 and 46 merges a step |
| `router_prepare_fusion` (retired) | -0.020 | -0.32 | 0 | lab chains; 64-row steps keep both launches |
| Deferred phased F32 reduction (prototype) | -0.11 | -0.20 | -0.28 | lab calls, 77 reductions a step |
| Rounding entry read by the HC boundary (not built) | -0.06 | 0.21 | 0.54 | lab calls of the comm entry alone |

The C16 inventory is 2,087 launches and 50.2 ms busy a step, so the
largest built candidate's gain is about a quarter of a percent of a C16
step and the
two `verify_row_fusions` parts together at most about 0.8 percent of a C1
step (13.68 ms busy) and 0.7 percent of a C8 step (30.33 ms).

## Rerunning

The ceil pairs (`ceil_parallel`) run on one GPU through the ignored
`bitwise_graph_live_rows_and_timing_on_gpu`, after the ignored
`frozen_modules_are_complete` stages the frozen modules with
`CHOPINFER_CEIL_ARTIFACTS` naming the directory, `CHOPINFER_CEIL_EXPORT`
set and `CHOPINFER_CEIL_TIMING_PROBE` naming the kernel lab's
`probe_sm120.cubin`.

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
