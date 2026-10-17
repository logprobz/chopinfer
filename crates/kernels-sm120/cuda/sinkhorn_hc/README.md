# Sinkhorn-mixed hyper-connections

`sinkhorn_hc.cuh` implements the unfused arms of
`kernel_core::sinkhorn_hyper_connection::SinkhornHyperConnectionQuery`:
`MixCollapse` before each sublayer, `Expand` after it and the
mean-of-streams `Head` before the final norm. The residual holds N BF16
streams of width H per row. This is a different op from the sigmoid-mean
hyper-connection rows (`row::hyper_connection`). The fused all-reduce
epilogue (`AllReduceExpandMixNorm`) is refused by name until it lands.

The planner and catalog live in `src/sinkhorn_hc/mod.rs`. The served
profile `sinkhorn-hc-4x4096` (tag `hc4x4096`) holds 4 streams of 4096, 20
iterations, eps 1e-6, norm eps 1e-5 and BF16 `fn`, at a capacity of 8192
rows. The mixer is replicated, so the same entries serve every rank count.

## ABI

Every entry has 256 threads, no dynamic shared memory and no workspace.
Its grid is sized from the prepared bucket, and `live_rows` (a device
`uint32_t`) bounds the work, so CUDA graphs replay at any live count up to
the bucket. Rows past the live count write nothing.

| Entry | Grid | Parameters |
| --- | --- | --- |
| `mix_collapse` | `(rows)` | `x [rows, N, H]` BF16, `fn [(2+N)N, NH]` BF16 or F32, `base [(2+N)N]` F32, `scale [3]` F32, `collapsed [rows, H]` BF16, `post [rows, N]` BF16, `comb [rows, N, N]` BF16, `live_rows` |
| `mix_collapse_probe` (lab only) | `(rows)` | as above, then `projected [rows, (2+N)N]` F32, `rstd [rows]` F32 and `cells [rows, N, N]` F32 (the Sinkhorn input: row softmax plus eps), before `live_rows` |
| `expand` | `(ceil(H/512), rows)` | `x [rows, N, H]`, `y [rows, H]`, `post`, `comb`, `out [rows, N, H]` (BF16; `out` must not alias `x`), `live_rows` |
| `head` | `(ceil(H/512), rows)` | `x [rows, N, H]`, `out [rows, H]` (BF16), `live_rows` |

`comb[i][j]` is at `i * N + j`. Its column j is the weight of stream i in
output stream j. Every entry calls `griddepcontrol.launch_dependents` first
and `griddepcontrol.wait` before it reads the live count or any activation.
Before the wait, `mix_collapse` prefetches `fn` into L2: CTA b, thread t
touches 128-byte line `(256 b + t) mod lines`.

## Schedule and numerics

All math is F32. Every rounding point is explicit (`__fmul_rn`,
`__fadd_rn`, `__fdiv_rn`, `fmaf`), so nvcc's FMA contraction changes no
result, and every reduction order is fixed. The kernels use no float
atomics and no tensor cores.

`mix_collapse`, one row per CTA:

1. The row's N H streams are staged in static shared memory (BF16, at most
   40 KiB). Each thread FMAs the squares of its elements in order. The
   threads then combine through a warp butterfly and the 8 warp sums in
   warp order. `rstd = __frsqrt_rn(sum / (N H) + norm_eps)` is correctly
   rounded.
2. The projection runs over the raw streams. Warp w owns mixes w, w + 8,
   and so on. For each 128-K chain (K order), each lane takes four products
   from zero with FMA, a butterfly sums the 32 lanes (every lane gets the
   same bits), and the chain is added into the mix's accumulator. No
   chain spans more than 128 K, and the chains are added in K order. The
   logit is `accumulator * rstd`.
3. `pre_i = sigmoid(fma(l_i, scale[0], base[i])) + eps` and
   `post_i = 2 sigmoid(fma(l_{N+i}, scale[1], base[N+i]))`, where sigmoid is
   `common/sigmoid.cuh`: `1 / (1 + expf(-v))` at v >= -80, `expf(v)` below.
4. Sinkhorn uses one thread per cell. It applies the row softmax (the row
   maximum subtracted, `expf`, the sum in index order from zero, then
   `e / sum + eps`), then one column normalization, then
   `iterations - 1` (row, column) rounds. Each normalization divides by
   the line's index-order sum plus eps.
5. `post` and `comb` are rounded once to BF16. `collapsed` is an FMA chain
   from zero in stream order, rounded once.

`expand` gives each thread two columns of every stream:
`X'_j = bf16(bf16(post_j * y) + bf16(sum_i comb[i][j] X_i))`, with the
stream sum an FMA chain from zero in stream order. `head` computes
`bf16((sum_i X_i) / N)`, with the sum from zero in stream order.
Both are exact functions of their BF16 operands.

## Verification

- `kernel_core::sinkhorn_hyper_connection_oracle` holds the f64 oracle of
  `MixCollapse` and the exact F32 forms of `Expand` and `Head`.
- `tests/sinkhorn_hc_compile.rs` builds and audits the served, probe and
  F32-`fn` 4 x 5120 cubins and checks their SASS.
- `crates/kernel-lab/tests/sinkhorn_hc.rs` holds the GPU tests:
  - the probe's projection against Neumaier-compensated f64 within
    2^-22 K S at K = 16,384, and rstd within its bound;
  - post and collapsed against the oracle, both within the gate and tightly
    from the kernel's own logits;
  - comb by its parts, since 20 Sinkhorn iterations have no useful finite
    error enclosure: the probe's softmax cells within a sound bound from the
    kernel's own logits, and comb bitwise equal to the exact F32
    normalizations of those cells (`sinkhorn_f32`);
  - the served entry bitwise equal to the probe;
  - expand and head bitwise against the exact forms;
  - poisoned eager, relaunch, graph and PDL replays.

## Split mix-collapse

`mix_split.cuh` holds `mix_collapse_split`, the `MixCollapse` query spread
over CTAs with output bits equal to `mix_collapse` at every row count. Its
planner is `src/sinkhorn_hc/split.rs`. The one-CTA kernel is
latency-bound at decode: one SM streams the whole `fn` (786 KB at the served
shape) through 128 dependent chain steps per warp. At long prefill launches
one CTA per row already fills the device, so a bundle may select either
entry by row count; both give the same bits.

| Entry | Grid | Parameters |
| --- | --- | --- |
| `mix_collapse_split` | `(chains / 8, rows)` | as `mix_collapse`, then `partials [rows, (2+N)N, chains]` F32 and `tickets [rows]` u32, before `live_rows` |

CTA `(g, row)` computes chains `8 g + w` (warp w) of every mix, with
`mix_collapse`'s lane mapping, products and butterfly, and lane 0 writes
each chain's sum to `partials`. Then every thread fences and thread 0 takes
a ticket (`atom.add.acq_rel.gpu`). The CTA that arrives last resets the
ticket to zero and finishes the row (`mix_finish`). It stages the row and
sums its squares as `mix_collapse` does, adds each mix's chains from zero in
K order, and runs `mix_collapse`'s pre, post, Sinkhorn and collapse
expressions. The ticket only chooses which CTA finishes. No float is added
atomically, and every sum keeps `mix_collapse`'s order. `partials` is a
workspace that every launch rewrites before reading it. `tickets` must be
zeroed once at allocation, and every launch leaves it zeroed. A ticket that
arrives past the CTA count traps.

The comm epilogue's split form (`comm/epilogue/sinkhorn_split.cuh`) uses the
same chain sums, ticket and finish over the expanded streams.

`crates/kernel-lab/tests/sinkhorn_hc_split.rs` checks the split entry
against the one-CTA served and probe cubins at 1, 2, 4, 16, 32, 64, 192,
256, 1024, 2048, 4096 and 8192 rows, and bitwise also one row below each
prefill count from 64 up:

- collapsed, post and comb are bitwise equal, written into poisoned buffers;
- the chain sums, added on the host in K order, are bitwise equal to the
  probe's projection and within the 128-K gate;
- relaunches are bitwise deterministic and leave the tickets zero;
- poisoned eager, relaunch, graph and PDL replays pass at every live count,
  after a producer whose streams are poisoned and checked, with each live
  count's outputs also compared with the one-CTA entry's.

Its timing test runs both entries at 64, 256, 1024, 2048, 4096 and 8192
rows, each launch after an L2 eviction and the producer of its streams, and
records the smallest row count from which the one-CTA entry is faster at
every larger measured count.

## Split mix-collapse with the input norm

`mix_norm_split.cuh` holds `mix_collapse_norm_split`: the split
mix-collapse followed by the sublayer's input RMSNorm of the collapsed row
(`row/row.cuh`'s `rmsnorm<H, 0, 3, false>`, HF order with a BF16
intermediate), in one launch, with output bits equal to the two launches
at every row count. Its planner is `src/sinkhorn_hc/mix_norm.rs`.

| Entry | Grid | Parameters |
| --- | --- | --- |
| `mix_collapse_norm_split` | `(chains / 8, rows)` | `x`, `fn`, `base`, `scale`, the norm's `weight [H]` BF16, `collapsed`, `post`, `comb`, `normed [rows, H]` BF16, `partials`, `tickets`, `live_rows`, then the norm's `eps_bits` (u32) |

The chain sums, ticket, partials and tickets are `mix_collapse_split`'s.
The finishing CTA stages the row, adds each mix's chains and computes pre
and post as `mix_finish` does, then runs the collapse with every thread
keeping its columns in registers. Once every thread has read the streams,
their shared memory holds each collapsed value's square
(`__fmul_rn(v, v)`). Then warp 0 runs `mix_finish`'s Sinkhorn, with its
block barriers as warp barriers because one warp owns all N^2 cells, while
thread 32 adds the 4096 squares from zero in column order and computes
`1 / sqrtf(s / H + eps)`. The serial sum takes about 16K cycles at FADD
latency and hides the Sinkhorn. Every thread then writes
`bf16(bf16(x * inv) * w)` for its columns. Only the order in which
independent values are computed differs from the two launches; every
expression and every reduction order is theirs. Before its dependency
wait, every CTA also prefetches the norm weight into L2.

`crates/kernel-lab/tests/sinkhorn_hc_mix_norm.rs` checks the entry against
the split entry and the row engine's norm at 1, 2, 3, 4, 8, 16, 32 and 192
rows:

- collapsed, post, comb and normed are bitwise equal, written into
  poisoned buffers, over ordinary, wide (binades -30 to 30) and extreme
  common-offset inputs; the chain sums equal the split entry's;
- relaunches are bitwise deterministic and leave the tickets zero;
- poisoned eager, relaunch, graph and PDL replays pass at every live count;
- mutants count as caught only when an output differs: the producer left
  out; the norm's squares added in reverse order (only `normed` may
  differ); and the dependency wait moved after the chain sums' reads,
  behind the lab's live-count PDL producer.

With `CHOPINFER_KERNEL_BENCH=1` it also times graphs of eight boundaries
(expand, then the two launches or the fused entry) at 1 to 32 rows.

## Split mix-collapse with the exact input norm

`mix_norm_exact.cuh` holds `mix_collapse_norm_exact`: the same outputs,
grid and ABI as `mix_collapse_norm_split`, bit for bit, with the norm's
ordered sum computed by `row/ordered_sum.cuh` (see "Exact ordered sum" in
`row/README.md`) instead of one thread's 4096 dependent adds. Its planner
is `src/sinkhorn_hc/mix_norm_exact.rs`.

The chain sums, ticket, staging, rstd and each mix's chains are
`mix_finish`'s. After the logits, warp 0 runs the Sinkhorn (warp barriers,
as in `mix_collapse_norm_split`) and writes comb, while warps 1 to 7 (224
threads, named barrier 1) compute pre and post, the collapse (column pair
`u + 224 p` per thread, each column's FMA chain from zero in stream order),
the squares over the dead streams, the ordered sum, the reciprocal and the
normalized row. The ordered sum's state lives in the staged streams after
the squares, so the static shared memory stays at 46,404 bytes.

`crates/kernel-lab/tests/sinkhorn_hc_mix_norm_exact.rs` checks the entry
against `mix_collapse_norm_split` (and that one against the split entry and
the row engine's norm) at every bucket of 1 to 32 rows and at 192 rows,
over ordinary, wide, extreme common-offset and order-sensitive inputs,
with relaunches, poisoned eager, relaunch, graph and PDL replays, and two
mutants counted only on changed outputs: the ordered sum's ties away from
zero (only `normed` may differ) and the late dependency wait. With
`CHOPINFER_KERNEL_BENCH=1` it times eight boundaries against
`mix_collapse_norm_split`.

## Split mix-collapse with batched loads

`mix_norm_batched.cuh` holds `mix_collapse_norm_batched`: the same outputs,
grid and ABI as `mix_collapse_norm_exact`, bit for bit, with the same
per-element operations in the same orders. Its planner is
`src/sinkhorn_hc/mix_norm_batched.rs`. The exact entry's SASS waits on
memory one access at a time: `chain_sums` loads a mix's fn values only after
storing the previous mix's chain sum (the store may alias fn), so each CTA
makes 24 dependent L2 round trips; `mix_finish` stages the streams and the
chain sums one load per loop iteration (20 more); and every thread runs a
sequentially consistent fence before the ticket. The batched schedule:

- A CTA whose row the live count covers loads its lanes' fn values (48
  registers of BF16 pairs) before the dependency wait. That live count is
  read before the wait as a hint only: a stale hint loads them after the
  wait instead, and the count read after the wait still decides whether the
  row runs. The lane's stream elements are prefetched into L1 beside that
  read.
- The 24 chain sums' butterflies run transposed: at a level whose count of
  sums is even, lanes with the level's bit set keep the upper half and the
  others the lower half, each adding its own value and its partner's as
  `warp_sum` does. Every kept sum is `warp_sum`'s at that lane and level,
  and `warp_sum` ends with the same bits in every lane, so the stored chain
  sums are the exact entry's. This takes 27 shuffles instead of 120.
- The ticket releases the CTA's chain sums with thread 0's acq_rel atomic
  after the block barrier (the arrival of a grid barrier) instead of a fence
  in every thread.
- The finishing CTA issues all its loads (streams, chain sums, norm weights)
  before using any, then stages them with `mix_finish`'s sum of squares in
  its order.
- Warp 0's Sinkhorn exchanges cells by shuffles instead of shared memory and
  warp barriers. Every lane runs cell `lane mod N^2`'s arithmetic, so no lane
  branches, and the lanes below `N^2` write comb.

The rest (logits, pre, post, collapse, squares, ordered sum, reciprocal,
normalized row) is `mix_collapse_norm_exact`'s. 96 registers, 46,340 bytes
of static shared memory, no local memory; two CTAs per SM as before.

The lab's phase breakdown uses two lab-only copies, statement for statement,
with the stamps of `mix_phase_probe.cuh`: `mix_norm_exact_probe.cuh` (of
`mix_collapse_norm_exact`) and `mix_norm_batched_probe.cuh` (of this entry).
A stamp records the global timer when its thread passes a point. Whole-CTA
or whole-row phase ends are either stamps after a barrier that covers the
phase's threads, or the maximum of per-warp stamps: every warp's chain sums,
and every worker warp's normed stores. Stores count when issued, not
drained, and the stamps can move the schedule slightly. The batched copy
takes 116 registers (two CTAs per SM, as served). `mix_probe.cuh`'s stamps
in `mix_norm_batched.cuh`'s `Probe` instantiation are no longer used by the
lab; they stay so that the served header's closure is unchanged.

`crates/kernel-lab/tests/sinkhorn_hc_mix_norm_batched.rs` checks the entry
against `mix_collapse_norm_exact` (outputs and chain sums) at every bucket of
1 to 32 rows and at 192 rows. The inputs are ordinary, wide, extreme
common-offset and order-sensitive, plus non-finite rows at 32 and 192 rows:
NaN streams with several payloads, infinite streams, overflowing streams, a
NaN sublayer output, and the expand's Inf - Inf. Those rows must produce
NaN and infinite chain sums and NaN outputs in the reference. The test also
covers relaunches and the probe entries. It runs poisoned eager, relaunch,
graph and PDL replays: one capacity-32 graph at every live count 0 to 32 for
each of three inputs, and sampled counts elsewhere. It has three mutants,
counted only on changed bits: butterfly levels reversed (the chain sums
differ), ordered-sum ties away from zero (only `normed` differs) and the
late dependency wait. With `CHOPINFER_KERNEL_BENCH=1` it times eight
boundaries against `mix_collapse_norm_exact`. It also reports each phase's
median duration from both probe entries; it fails unless every CTA's stamps
are complete and in program order and every phase has one sample per replay
and boundary.

## Split mix-collapse with the tree norm

`mix_norm_tree.cuh` holds `mix_collapse_norm_tree`: `mix_collapse_norm_batched`
statement for statement (without its probe stamps), with the input norm's
sum of squares in `row/rmsnorm_tree.cuh`'s 256-lane F32 tree
(`RmsNormNumerics::HF_TREE_256`) instead of the exact ordered sum. Its
planner is `src/sinkhorn_hc/mix_norm_tree.rs`; it shares the batched
entry's grid, ABI and plan. The finisher's seven worker warps stage the
squares as before, then run the tree's eight warps over them: worker warp
k runs the tree's warp k, and every worker warp also adds lanes 224 to 255
(the tree's warp 7) so that worker warp 0's two chains interleave; worker
warp 0 stores both, adds the eight warp sums and publishes the reciprocal.
The order depends on the width only, so a row's normed bits equal the
row engine's tree norm of its collapsed row (the prefill chain: split
mix-collapse, then `rmsnorm_hf_tree`) at every bucket and live count. The
exact ordered sum's scan and walker were most of the workers' critical
path (about 6.7 us of 9 at one row); without them the Sinkhorn warp's 40
divisions per cell are the finisher's longest branch.

`crates/kernel-lab/tests/sinkhorn_hc_mix_norm_tree.rs` checks the entry
against the prefill chain (outputs and chain sums) at every bucket of 1 to
32 rows and at 192 rows, on the batched test's ordinary, wide, extreme,
order-sensitive and non-finite inputs, and every normed row of a finite
collapsed row against the CPU emulation of the tree norm. It runs poisoned
eager, relaunch, graph and PDL replays at every live count of a capacity-32
graph, the reversed-levels, reversed-distances and late-wait mutants, and
times it against the batched entry over graphs of eight boundaries.

## Tiled mix-collapse

`mix_tiled.cuh` holds `mix_collapse_tiled`, the `MixCollapse` query with
`Rows` consecutive rows per CTA (4, 8 or 16) and output bits equal to
`mix_collapse` at every row count. Its planner is
`src/sinkhorn_hc/tiled.rs`. Both earlier entries read the whole `fn` once
per row (1.6 GB of L2 reads per mixer at 2048 rows); the tiled entry reads
it once per tile, and sums many chains per butterfly round, for long
prefill launches.

| Entry | Grid | Parameters |
| --- | --- | --- |
| `mix_collapse_tiled<M, Rows, Probe>` | `(ceil(rows / Rows))` | as `mix_collapse` (with `Probe`, its probe's) |

- The tile's streams are staged in 16 KiB chunks of whole 128-K chains
  (`cp.async`, double-buffered). Warp w owns mixes `w, w + 8, ...` of every
  row; lane l loads its four fn values of chain c once and forms each row's
  chain value with `mix_collapse`'s `fmul, fma, fma, fma` over
  `k = 128 c + 4 l .. + 3`.
- The warp sums its `Rows x PerWarp` chain values together. The first
  `log2(Rows)` butterfly levels (offsets 16, 8, ...) halve the set: a lane
  whose level bit is set keeps the upper half and adds its partner's copy
  first, a lane without it adds its own value first. The remaining levels
  run as `warp_sum`. Lane group `l >> (5 - log2(Rows))` holds row
  `l >> (5 - log2(Rows))`'s sums, and the group's first lane holds exactly
  the expression lane 0 of `warp_sum` stores: the same additions, each with
  the lower lanes' partial sum as its first operand. That lane adds them
  into its accumulators in K order and stores them.
- Thread t FMAs the squares of packs `t, t + 256, ...` of every row in
  order, reading them from the staged chunks; the per-row warp sums use the
  same halving, and the warps' sums are added in order, so rstd keeps
  `mix_collapse`'s bits.
- pre, post, the Sinkhorn normalizations and the collapse are
  `mix_collapse`'s expressions. The Sinkhorn runs per row on a group of
  lanes of one warp, which read each other's cells by shuffles. The collapse
  re-reads the streams from global memory, on the warps without Sinkhorn
  rows when there are any.

No float atomics, no cross-CTA communication, a capacity-sized grid with
the live count read on the device; a CTA past the live count returns and
rows past it in a tile write nothing.

`crates/kernel-lab/tests/sinkhorn_hc_tiled.rs` checks every tile against
the one-CTA entry and the split at decode, verify and prefill buckets, the
prefill row parts' sizes and the tile edges (ordinary, extreme and
non-finite inputs), the probe's projection, rstd and Sinkhorn cells
against the one-CTA probe, every live count of a 64-row capacity and the
row parts' live counts at 8192, poisoned eager, relaunch, graph and PDL
replays of each served tile (`SERVED_TILES`, 4 and 8 rows per CTA) at its
tile edges' live counts and the row parts' sizes, and two mutants of each
served tile (the butterfly levels after the halvings in reverse order; the
live count read before the dependency wait). Its timing test runs the
one-CTA entry, the split and every tile at prefill row counts from 64 to
8192, around the schedule's edges and the 4-row tile's one-round limit.

## Lagged placement

`lagged.cuh` holds the unfused arms of the lagged single-pass placement
(`HyperConnectionPlacement::LaggedSinglePass`): `mix_collapse_lagged`, the
F32 expand `expand_f32` (`ExpandF32`) and the sigmoid-weighted head
`weighted_head`. It includes `sinkhorn_hc.cuh` unchanged, so the
per-sublayer cubins keep their source closure. The planner is
`src/sinkhorn_hc/lagged.rs`; the lab profile `sinkhorn-hc-lagged-4x5120`
(tag `hcl4x5120`) holds 4 streams of 5120 (the row's streams fill the 40 KiB
shared-memory row exactly), 20 iterations, eps 1e-6, norm eps 1e-20 and F32
`fn`, at a capacity of 8192 rows.

| Entry | Grid | Parameters |
| --- | --- | --- |
| `lagged_mix_collapse` | `(rows)` | `x [rows, N, H]` BF16, `fn`, `base`, `scale`, `pre_in [rows, N]` F32, `collapsed [rows, H]` BF16, `pre [rows, N]` F32, `post [rows, N]` F32, `comb [rows, N, N]` F32, `live_rows` |
| `lagged_mix_collapse_probe` (lab only) | `(rows)` | as above, then `projected`, `rstd` and `cells` as `mix_collapse_probe`, before `live_rows` |
| `expand_f32` | `(ceil(H/512), rows)` | `x`, `y` BF16, `post [rows, N]` and `comb [rows, N, N]` F32, `out` BF16 (must not alias `x`), `live_rows` |
| `weighted_head` | `(ceil(H/512), rows)` | `x` BF16, `pre [rows, N]` F32, `out [rows, H]` BF16, `live_rows` |

`pre_in` and `pre` must not overlap; the host alternates two buffers between
sublayers and writes the one-hot `(1, 0, ..., 0)` before the first.

The lagged mix-collapse is `mix_collapse` statement for statement in its
staging, sum of squares, rstd, projection chains, pre, post and Sinkhorn.
It differs in three places: it reads the row's collapse weights from
`pre_in` (published by the staging barrier) instead of using its own `pre`;
it stores `pre`, `post` and `comb` as the F32 values, unrounded; and the
collapse is the same FMA chain from zero in stream order over the given
weights. `expand_f32` computes
`X'_j = bf16(fl(post_j * y) + sum_i comb[i][j] X_i)` with the stream sum an
FMA chain from zero in stream order and one rounding to BF16;
`weighted_head` computes `bf16(sum_i pre_i X_i)` as an FMA chain from zero
in stream order. Both are exact functions of their operands.

`tests/sinkhorn_hc_lagged_compile.rs` builds and audits the cubins.
`crates/kernel-lab/tests/sinkhorn_hc_lagged.rs` holds the GPU tests, at 1,
32, 192 and 8192 rows and on the extreme common-offset input:

- the probe's projection within 2^-22 K S at K = 20,480 and rstd within its
  bound;
- the collapsed row bitwise equal to the exact collapse with `pre_in`;
  `pre` and `post` within their F32 bounds from the streams and from the
  kernel's own logits; the comb by its parts (cells within their bound, the
  F32 comb bitwise equal to `sinkhorn_f32` of the cells);
- the per-sublayer twin probe (the same mixer, per-sublayer placement) on
  the same streams: projection, rstd and cells bitwise equal, its BF16 post
  and comb the lagged F32 values rounded once, its collapsed row the exact
  collapse with the lagged entry's stored `pre`;
- the served entry bitwise equal to the probe in poisoned buffers;
- both F32 expands and the head bitwise against the exact forms;
- poisoned eager, relaunch, graph and PDL replays at the live counts of
  `poison::live_counts`.

The conformance kit's case is `kit::cases::sinkhorn_lagged`.
