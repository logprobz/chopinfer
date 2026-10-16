# SM120 PCIe communication

`comm.cuh` contains F32 and BF16 rank-ordered all-reduce and all-gather
entries, and `push.cuh` the fused decode all-reduce. The entries issue
`griddepcontrol` for PDL and keep their epochs on the device, so CUDA graph
replays need no host updates. Every sum runs in rank order in F32, so outputs
are bitwise identical on all ranks and across replays.

## Measured PCIe behaviour

Every protocol here sends to one destination at a time. On four RTX PRO 6000
GPUs behind separate root ports of one CPU (measured 2026-09 with the
kernel-lab `peer_push_fanout` and `peer_push_pull` benches in
`tests/comm_bench.rs`), one destination per GPU runs at 42 to 47 GB/s in
every pairing, while a GPU that writes to (or pulls from) two or more peers
at once collapses to about 2 GB/s in total. Copy-engine copies give no
bandwidth advantage and cannot fuse the reduction, so no path uses them.

## Fused epilogues

A fused all-reduce runs a model's epilogue on each reduced row in the same
launch. The epilogue is a functor the entry supplies; the transport owns the
protocols, flags, row geometry and the order of every sum. Each functor is a
`template <uint32_t Width, uint32_t Streams>` in `epilogue/`, and the
transports take it as their first template argument:

- `epilogue/residual_rmsnorm.cuh`, `ResidualRmsNorm<4096, 1>`: MiMo's
  residual add, BF16 rounding and HF RMSNorm, run by the push transport
  (`push.cuh`: low latency, one-shot, two-shot, wide and the phased pair).
  Per 16-byte vector, `carry` turns the rank-ordered F32 sums into the kept
  BF16 row and a row statistic, the transport sums the statistic in its
  fixed order, and `finish` writes the output.
- `models/qwen38/bundle/cuda/comm/epilogue/hc_inject_norm.cuh`, `HcInjectNorm<2560, 4>`: Qwen3.8's
  hyper-connection inject and per-stream Gemma norm. The hyper-connection
  transports (`hyper_connection.cuh`, `hyper_connection_decode.cuh`,
  `hyper_connection_decode_wide.cuh`)
  hand it the unrounded F32 sums of a token, and it finishes each (row,
  stream) group with one CTA.

The second template argument is the transport's input (`input/`): `Bf16`
(eight elements per 16-byte vector), `F32` (four per vector; the
hyper-connection transports take it, the push transport refuses it at
compile time), and `RoutedCombine<TopK>`, a staged push input that sums each
row's TopK preweighted routed rows in expert order and rounds to BF16, as
`chop_row_combine_mimo4096_v1` does.

Geometry is checked at compile time (`push::Rows`, `qwen38::Geometry`, and
`qwen38_decode::Rows` for the LL layout; a geometry outside it builds the
decode entries without LL, `CHOP_QWEN38_LL_MAX_ROWS` 0). On the host,
`comm::Epilogue` and `comm::Input` name a functor's or input's header,
template and sizes (`mimo26::EPILOGUE`, `qwen38::EPILOGUE`, `input::BF16`,
`input::F32`, `mimo26::ROUTED_COMBINE`), and `qwen38::HcShape` pairs them for
the hyper-connection `_with` functions (`qwen38::SHAPE` for Qwen3.8).
`models/qwen38/contract/tests/sm120/kernel_comm_extension.rs` compiles every
hyper-connection entry for F32 rows 5120 wide with four streams.

A new model adds a functor header (and, for another input type, an input
header) plus host values: F32 sums per stream group take the
hyper-connection transports, a per-row statistic over BF16 rows the push
transport. Each hyper-connection group's CTA runs independently with BF16
residual and parameter tiles, so an epilogue that mixes streams must read
immutable inputs and write separate outputs. No GPU run has exercised an F32
input yet.

## Fused decode all-reduce

`chop_comm_allreduce_oneshot_norm_bf16_v2` reduces BF16 hidden rows (4096
wide) across ranks, adds the residual, rounds to BF16, writes that pre-norm
sum to `residual_output` (which may alias `residual`), and applies HF
RMSNorm. The selector requires `HfBf16Intermediate` and
`fused_residual: true`; v2 argument builders refuse an absent residual.
Normalization rounds to BF16 before multiplying the weight and uses the
protocol's F32 tree reduction. Every call writes all rows of the output.
Launch it with `comm::plan`; `comm::push_buffers` gives the inbox, flag and
epoch sizes. Flags, epochs and the low-latency region of the inbox (its
first `zeroed_inbox_bytes`) start zeroed; the rest of the inbox may hold any
bytes.

`tuning/comm.json` selects the protocol by rows with thresholds per rank
count and feeds the same thresholds to the cubin as defines
(`CHOP_COMM_LL_ROWS_BY_RANKS`, `CHOP_COMM_ONESHOT_ROWS_BY_RANKS`), so launch
geometry and kernel dispatch agree; a mismatch traps:

| Ranks | Low latency | One-shot | Two-shot |
|---|---|---|---|
| 2 | 1 to 2 rows | 3 to 32 | none |
| 4 | 1 to 2 | 3 to 16 | 17 to 32 |
| 8 | 1 to 2 | 3 to 16 | 17 to 32 |
| 1, 3, 5, 6, 7 | 1 to 2 at one rank, none otherwise | the rest | none |

The planner refuses low latency at a rank count that is not a power of two,
or two-shot where a rank's share of a row's 512 vectors is not whole
(`CHOP_COMM_TWO_SHOT_MIN_RANKS`).

- Low latency: one cluster of two 256-thread CTAs per row. Recursive
  doubling sends every input a rank holds to `rank ^ 2^h`; each 8-byte entry
  carries two BF16 values and a flag word whose halves are BF16 NaN
  patterns, so no fence or flag word is needed (the region must start
  zeroed). Inboxes are double-buffered by epoch parity and receivers clear
  what they read.
- One-shot: one 8-CTA cluster pushes the whole input to each peer in turn
  and synchronizes before the next peer; CTA 0 then releases one flag word
  per destination. One CTA per row waits for the flags, reduces, adds the
  residual and normalizes.
- Two-shot: owner `q` reduces `1 / ranks` of the columns of every row and
  pushes the rounded sums to every peer.

Flags of the bulk protocols carry bit 31 and compare for equality. The last
CTA to finish advances the epoch through a local counter.

`chop_comm_allreduce_combine_norm_bf16_v1` (arguments of v2 plus a
`live_rows` pointer, `comm::fused_combine_arguments`) folds the MoE combine
in: it takes the preweighted top-8 routed rows `[rows][8][4096]`, and its
outputs match combine followed by v2 bitwise; rows at or past `*live_rows`
contribute zero. `CHOPINFER_MIMO_FUSED_COMBINE=1` selects it in the
kernel-lab decode fixture (`tests/mimo_layers.rs`).

### Lower-latency decode entries

`chop_comm_allreduce_decode_norm_bf16_v2` and
`chop_comm_allreduce_decode_combine_norm_bf16_v1` (`comm::DECODE_SYMBOL_V2`,
`comm::DECODE_COMBINE_SYMBOL`) take the arguments, plans
(`comm::decode_plan`) and state of the v2 and combine entries and run the
same protocols with earlier loads, so their outputs are bitwise the same and
both families can follow each other on one state. MiMo uses them up to
`comm::DECODE_MAX_ROWS` (12) rows and the v2 and combine entries above. At
low-latency rows both sum the norm statistic with `row_sum_async`, which
pushes CTA sums with `st.async` and adds them in the original CTA order, in
the two-CTA LL geometry. Plain decode v2 has its own cubin,
`comm_decode_v2_sm120` (`comm::DECODE_CUBIN_V2`): in `comm_sm120` its
asynchronous norm state would share one shared-memory layout with the
combine entry's and change the code of both.

Serving qualification of plain decode v2 on 2026-10-04 found bit-identical
C1 outputs and fixed-batch C16 outputs at TP4. Real-host create passed: the
LL and bulk plans needed 2 of 188 and 3 of 46 resident clusters. C1 time per
output token was 0.987 and 0.991 of the replaced v1 entry's at TP2, and
0.990 and 0.954 at TP4 in two runs. C16 and C32 differences were within
measurement noise; the LL protocol runs only at one and two rows.

Qwen3.8's decode entries,
`chop_comm_qwen38_allreduce_hc_decode_r{2,4,8}_v2` and `_v3`
(`qwen38_decode::symbol_v2`, `symbol_v3`), read each stream's norm weights
before the PDL wait and issue a row's local loads before they check the
live count, so no load waits for another. The bundle runs v3 at one or two
prepared rows and v2 for larger buckets.

At one or two prepared rows, v3 keeps recursive doubling and each CTA's
quarter sends, but every stream CTA reads the whole row from its local LL
inbox. It sums in rank order from F32 zero and uses the unchanged inject
and norm epilogue. The local source stays in input to fit eight ranks in
static shared memory. Wider or F32 transport extensions that exceed that
limit keep v2's quarter exchange, as do all one-shot row counts.

Readers leave entries intact. After consuming their values into local sums,
the CTAs rendezvous at a relaxed cluster barrier overlapped with inject and
norm. Only the quarter owner clears each received entry, after the barrier
wait. A slot's next writer uses it two epochs later: completing the
intervening epoch requires this rank's input, which is sent only after the
previous kernel, including its clears, completes. Zero live rows do not
advance their LL epochs. Clearing prevents stale matches after the 12-bit
flag wrap. No F32 quarter copies or mbarrier waits remain in this path.
`crates/kernel-lab/tools/sass_same.sh` checks that other entries' SASS is
unchanged; `tests/comm_latency.rs` checks the bits against v2 and combine v1.

Lab qualification on 2026-10-04 measured one-row calls at 9.47 us for v2
and 6.66 us for v3 at TP2, and 10.98 us and 9.70 us at TP4. Serving
qualification found bit-identical C1 outputs and bit-identical fixed-batch
logits at C1, whose steps prepare one and two rows, and at C16. C1 time per
output token, v3 divided by v2, was 0.966 and 0.945 at TP2 and between 0.964
and 0.993 in four TP4 runs. C16 and C32 run v2 in both arms; their
differences were within start-to-start variation.

### Verify entries for exact rows

`chop_comm_allreduce_verify_norm_bf16_v1` and
`chop_comm_allreduce_verify_combine_norm_bf16_v1` (`comm::VERIFY_SYMBOL`,
`comm::VERIFY_COMBINE_SYMBOL`) are the decode entries with `LlOrder`: the
one-shot's and two-shot's row CTAs sum squares in the low-latency order, so
every row's outputs equal the row's bits in a one-row call at any row count
up to 32 (`comm::verify_entry_rows`). MiMo's exact verify rows and prefill
steps of up to 32 rows take them (`HiddenRoute::ExactRows`,
`HiddenRoute::Prefill`), so a prompt row normalizes as it would alone. They
live in their own cubin, `comm_mimo_verify_sm120`
(`model_mimo26_contract::kernels::comm::verify_cubin_spec`).
Kernel-lab `mimo_verify_rows_bitwise_g1` checks every row of 1- to 16-row
calls against the row reduced alone.

## Wide all-reduce for graphs above 32 rows

Decode and verify graphs of 33 to 256 rows run
`chop_comm_allreduce_wide_norm_bf16_v1`, or with the MoE combine folded in
`chop_comm_allreduce_wide_combine_norm_bf16_v1`. They take the v2 and
combine arguments and match the push path bit for bit on pre-norm rows, and
on normalized rows against the two-shot. `comm::graph_choose` and
`graph_plan` pick them (`Variant::FusedWide`) up to the rank count's
`wide_max_rows` in `tuning/comm.json` (256 at 1, 2, 4 and 8 ranks); `choose`
and prefill keep the phased pair. The wide entry writes all rows of the
output. The MiMo bundle uses the plain entry for every graph above 32 rows;
`CHOPINFER_MIMO_WIDE_COMM=0` restores the phased stages.

The protocol is the push two-shot with slots for 256 rows: owner `q`
reduces columns `[q W, (q + 1) W)`, `W = 512 / ranks` vectors, one
destination at a time. The last reduce-scatter and all-gather phases release
one flag per block of rows, so reduction and normalization overlap the rest
of the phase. Workers (every CTA after the sending cluster; the planner uses
up to 64) normalize rows with one CTA per row; the worker count is a launch
knob (`CHOPINFER_KERNEL_BENCH_WIDE_WORKERS` in `comm_bench`). Every rank of a
call must run the same rows: CTA 0 exchanges its rows and live rows with
every peer and traps on a difference instead of hanging.
`push_protocol` and `push_buffers` answer for the push entries only (up to
32 rows); `graph_protocol` and `graph_buffers` include the wide protocol.
Eight ranks are not yet tuned.

Tests: `tests/comm_wide.rs` (33 to 256 rows against the host residual chain
and the HF RMSNorm oracle, and against v2 on chunks) and `tests/comm_sync.rs`
(growing and shrinking calls mixed with push and phased calls on one state,
under the skew build's lagging rank, across the epoch wrap).

## Low-latency F32 all-reduce

`chop_comm_allreduce_ll_f32_v1` (`allreduce_ll_f32.cuh`,
`comm::ll_f32`) sums F32 rows in rank order from F32 zero, bitwise
`chop_comm_allreduce_oneshot_f32_v1`, for a few rows. Each 4096-element
tile of the one-shot spreads over four CTAs of 1024 elements (one 16-byte
vector per thread), and the CTAs exchange their inputs by the push path's
recursive doubling in low-latency words: each 64-bit word carries one F32
value and an `ll_flag` word, so no fence, flag word or acknowledgment is
needed. Receivers clear what they read; slots alternate by epoch parity.
Each CTA exchanges only with the CTAs of the same index on its peers and
never waits on a CTA of its own grid, but it spins on those counterparts,
so a caller admits the grid against the device's residency first (a bundle
checks its largest grid at create, see "Residency of the spinning comm
entries", and keeps the one-shot otherwise). The entry triggers its PDL
dependents at its first instruction, so they start only after every CTA of
its grid has started. One state (inbox `[parity][source]
[CTA][256][4]` words and one epoch per CTA, both zeroed at allocation) serves
every row count up to its capacity, at most 128 CTAs (32 rows); every rank
passes the same capacity. Power-of-two rank counts only. Kernel-lab
`tests/allreduce_ll_f32.rs` checks it against the one-shot and the host at
1 to 32 rows behind a device producer, poisoned replays, two wait mutants,
a stale-flag mutant, every call around the 12-bit flag wrap and the 32-bit
epoch wrap with fresh inputs, and times both entries.

## Residency of the spinning comm entries

Every peer-to-peer entry here has workers that spin on flags other grids
release. A grid that is only partly resident hangs rather than slows down.
Another process's spinning grids under MPS can hold the SMs a sending
cluster needs.

Each bundle checks at create (`bundle_runtime::residency`) that the
ranks on a device times each spinning entry's clusters at its largest plan
fit what the host reports through `query_cluster_residency_v1`
(`cuOccupancyMaxActiveClusters`). The check fails closed:

- The wide all-reduce is turned off unless it fits (the entries compile for
  `WIDE_CTAS_PER_SM` CTAs per SM), and graphs above 32 rows then take the
  phased pair, with a log line naming the reason.
- An entry without a fallback that falls short fails create.
- A host that reports no cluster count, an SM-limited context or an MPS
  client also fails create: multi-rank serving under MPS is unsupported.
  `CHOPINFER_COMM_RESIDENCY=unchecked` creates anyway with the wide entry
  off; a measured shortfall still fails.

MPS detection is a heuristic (a non-empty `CUDA_MPS_*` variable, or the
daemon's `nvidia-mps/control` pipe in its default directory): a stale pipe
reads as MPS and needs removing or the override. The vocabulary all-gather
(`chop_comm_allgather_*`, no cluster) is not covered.

## Prefill all-reduce

Above 32 rows the planner selects `FusedTwoShot`, and the bundle launches
`chop_comm_reduce_scatter_residual_bf16_v1` then
`chop_comm_allgather_presummed_norm_bf16_v1`
(`comm::PREFILL_ROUND_ONCE_SYMBOL`, `PREFILL_ROUND_ONCE_GATHER_SYMBOL`,
MiMo's `launch::comm::PHASED_SYMBOLS`) with the same `comm::plan`. Unfused
`TwoShot` pairs the plain scatter `chop_comm_reduce_scatter_*_v1` with
`chop_comm_allgather_scattered_*_v1`, also for F32. Both are a phased direct
two-shot in `push.cuh`: owner `q` holds tiles
`[q * tiles / ranks, (q + 1) * tiles / ranks)` of 4096 elements. One 8-CTA
cluster sends to one destination at a time (rank + 1, + 2, ...), and up to
seven more clusters reduce and normalize. The scatter pushes each owner its
tiles; the owner sums all ranks in rank order in F32, adds the residual with
the epilogue's `carry` and rounds once to BF16 into its tiles of `output`.
The gather keeps the owner's pre-norm tiles in `residual_output`, sends them
into every peer's `output`, and workers normalize rows by warp as each
owner's tiles arrive. Each rank sends 2 (ranks - 1) / ranks of the message,
the all-reduce minimum.

The pre-norm rows are bitwise those of the push and wide entries, and the
gather's one-warp norm (`norm_row_warp`) sums a row's squares in the
low-latency order, so its normalized rows are bitwise those of the verify
entries and of a one-row call: a prompt row's bits do not depend on the
step's rows. Kernel-lab `bf16_prefill_phased_matches_verify_rows`
(`tests/comm.rs`) checks this at 1 to 8,192 rows, and
`phased_warp_norm_sums_squares_in_low_latency_order` replays the order on
the CPU.

`comm::phased_buffers(ranks, elements, dtype)` gives the sizes: the inbox
region starts at 8 MiB (`ranks` owner slots follow), the flags need 82 words
and the epoch one word. The stages use fixed flag words with tagged equality
on epoch[0] and need no acknowledgments, so consecutive calls with different
row counts, and push calls in between, are safe; `tests/comm.rs` runs mixed
row counts on one prefill-sized state and in a graph.

## Phased F32 all-reduce from many senders

`allreduce_phased_f32.cuh` (`comm::phased_f32`) runs the F32 phased
pair's protocol in one launch and sends from a launch-chosen number of
CTAs (`senders`, up to 64) instead of one 8-CTA cluster; the pair reached
about 24 GB/s sent per rank on 2,048-row prefill reductions, against 42
to 47 GB/s for one destination per GPU. Owner `q` holds granules
`[q G / ranks, (q + 1) G / ranks)` of the call's `G` 1,024-element
granules; each span splits into blocks of about 64 granules (at least one
per sender, at most 256), interleaved over senders and workers. Sending
CTA `s` moves blocks `s`, `s + senders`, ... to one destination at a
time, `rank + 1`, `rank + 2`, ..., releasing one flag per block, and the
senders meet at a local counter barrier before each next destination, so
a GPU never writes to two peers at once. Workers (the CTAs after the
senders) sum each owner block in rank order from F32 zero as soon as all
its sources arrived, and the senders then gather the summed blocks into
every peer's output. Every element is bitwise the pair's sum.

- `chop_comm_allreduce_phased_f32_v1` writes the F32 sums.
- `chop_comm_allreduce_phased_round_f32_v1` rounds each sum once to BF16
  at its owner (`cvt.rn.bf16.f32`) and gathers BF16, half the gather's
  bytes.
- `chop_comm_allreduce_phased_oneshot_f32_v1` and
  `chop_comm_allreduce_phased_oneshot_round_f32_v1` are a phased one-shot
  for small calls (up to `ONE_SHOT_MAX_ELEMENTS`, 256 granules): each rank
  pushes its whole partial to one peer at a time and sums every element
  itself in the same rank order, three phases at four ranks instead of the
  two-shot's six. Its inbox slots and flags alternate by epoch parity in a
  region of their own, so a rank writes a slot again only two calls later,
  after its intervening call needed the peer's data of that call.

A grid of senders only (no workers) has each sending CTA sum its own
blocks after the scatter and gather them without the reduced-flag handoff;
a single sender skips the counter barrier. Decode-sized calls take these
geometries. The served F32 one-shot (`chop_comm_allreduce_oneshot_f32_v1`)
writes every peer at once and measured 22 us at one row and 373 us at 16
rows on four ranks (about 2 GB/s).

All four take `(input, output, inboxes[ranks], outputs[ranks],
flags[ranks], epoch, elements, row_elements, live_rows, rank, ranks,
senders)` (`phased_f32::arguments`, which takes the entry) and a plan of
`senders + workers` CTAs of 256 threads without clusters
(`phased_f32::plan_with`). With a non-null `live_rows`, a call reduces and
writes only the first `*live_rows` rows of `row_elements` (a multiple of
1,024); a call with nothing to reduce returns without advancing the epoch.
`phased_f32::buffers` gives the state: an inbox of the one-shot region
then one largest-owner-span slot per source, `FLAG_WORDS` flag words and
one epoch word, flags and epoch zeroed. The state belongs to these entries
only; calls of all four may follow each other in any order. Flags carry the
call's tagged epoch and compare for equality; the two-shot needs no
acknowledgments, because a rank sends call `k + 1`'s data only after its
call `k` completed, which needed every peer's gathered blocks, each sent
after that peer's reads of the block. Every CTA spins on flags, so the
whole grid must be resident on every rank.

Kernel-lab `tests/comm_phased_f32.rs` (lock family `comm_phased_f32`)
checks all four entries against the host, the pair and (up to 32 rows) the
served one-shot at 1 to 8,192 rows, the poisoned eager, relaunch, graph
and PDL-producer phases, four mutants (no scatter wait, no gather wait,
reversed rank order, PDL wait after the first reads) and times the served
collectives against every entry per sender and worker count.

## BF16 partials from F32 inputs

`allreduce_phased_bf16.cuh` (`comm::phased_bf16`) and
`allreduce_ll_bf16.cuh` (`comm::ll_bf16`) reduce BF16 partials that each
rank forms from its F32 input: every rank rounds its partial once to BF16
(`cvt.rn.bf16.f32`), the ranks exchange the BF16 values, and every output
element is the F32 sum of the ranks' BF16 values in rank order from F32
zero, rounded once to BF16. The output is BF16 on every rank. This is the
composition of the round row, a widening, any F32 entry above and a second
round row, in one launch with half the payload.

- `chop_comm_allreduce_phased_bf16_v1` and
  `chop_comm_allreduce_phased_oneshot_bf16_v1` are the phased F32 entries'
  two-shot and one-shot protocols, plans, block geometry and flags with
  BF16 inboxes and a BF16 gather. Their state is `phased_bf16::buffers`,
  half the inbox bytes of `phased_f32::buffers` for the same capacity. A
  phased F32 state is large enough for their calls, but a state serves one
  payload format: regions are placed by element, so BF16 and F32 calls on
  one state overlap in bytes, and a state passes between the two headers'
  entries only once every rank finished every call on it.
- `chop_comm_allreduce_ll_bf16_v1` is the low-latency entry with two BF16
  values in each LL word, so a 4096-element row spreads over two CTAs of
  2,048 elements instead of four; its state layout is `ll_f32`'s.

Kernel-lab `tests/comm_phased_bf16.rs` (lock family `comm_phased_bf16`)
checks a pinned matrix of calls over one set of operands, so row r's sum
is the same host fold in every case: the served entry and geometry at
every capacity to 64 rows and every live count, prefill capacities and
row parts to 8,192 rows, the low-latency entry, the fallback (the comm
engine's F32 pair over rounded partials) and the phased F32 round entry
over rounded partials, on one phased state with every rank synchronized
around each call. It also runs poisoned eager, relaunch, graph and
PDL-producer phases, six mutants (no scatter wait, no gather wait,
reversed rank order, late PDL wait, shared one-shot parity, own partial
unrounded), each observed in eager and replayed runs, and back-to-back
generations. The mutants that need a peer to run ahead (no gather wait,
shared parity) are built with a lab-only slow rank, and the served entries
with the same slow rank must stay exact. It times both entries against the
phased F32 round entry and the F32 pair.
`tests/allreduce_ll_bf16.rs` (lock family `allreduce_ll_bf16`) checks the
low-latency entry the same way against the host and the F32 low-latency
entry over host-rounded partials at 1 to 32 rows, around both flag and
epoch wraps, with wait, stale-flag and own-unrounded mutants.

## Phased BF16 all-reduce whose writers take turns

`allreduce_phased_turns_bf16.cuh` (`comm::phased_turns_bf16`,
`chop_comm_allreduce_phased_turns_bf16_v1`) is the phased BF16 two-shot
with the same arguments, rounding, inbox slots, block flags and rank-order
sums, so every output element is bitwise that entry's. It adds one rule on
when a rank starts each of its 2 (ranks - 1) phases (the reduce-scatter's
destinations rank + 1, ..., then the all-gather's in the same order).

When every rank is in the same phase the destinations form a permutation:
each GPU sends to one peer and receives from one. Ranks of a prefill step
start its all-reduces at different times, because the compute before them
differs between ranks (the routed experts' loads). A schedule that each
rank keeps on its own clock then lets a rank that already reduced gather
into a GPU that the late rank is still scattering into, so two or three
writers share one receiving link. Here a rank starts phase t into
destination d only after the rank that wrote into d in phase t - 1 has
issued all its writes. That rank's CTA 0, after its senders' barrier,
releases a turn word in the next writer's flags. The word carries the
call's tagged epoch and is compared for equality, as every flag is. Each
phase therefore waits for the previous phase's writer at the same
destination, which waited for its own predecessor, and so on: one writer
per receiving link, whatever the ranks' start times. Turns order transfers
only. Every value still travels under the block flags, so a turn never
makes a sum read early. Phase 1 waits for no turn, because the previous
call may have been another entry's.

The state is `phased_bf16::buffers`' with `TURN_WORDS` more flag words
(`phased_turns_bf16::buffers`). The phased BF16 entries may share it and
follow this entry in any order.

Kernel-lab `tests/comm_phased_turns_bf16.rs` (lock family
`comm_phased_turns_bf16`) checks every live count from 1 to 8,192 rows of
an 8,192-row call at the served geometry, plus edge counts at ten other
sender and worker counts. Each case is bitwise against the host fold and
the phased BF16 two-shot on one shared state, and every byte past the live
rows must stay poison. The tests also cover poisoned eager, relaunch, graph
and PDL-producer phases, alone and between the phased BF16 entries; five
mutants (no scatter wait, no gather wait with a slow rank, reversed order,
late PDL wait, own partial unrounded), each observed eagerly and replayed;
and generations mixing all three entries with a delayed peer. It times the
entry, its variants (no turns, 32- and 128-granule blocks) and the two-shot
with ranks that start together and with injected skew (one late rank, or
staggered starts), and measures the PCIe write rate per link pattern.

## Qwen3.8 hyper-connection all-reduce

`comm/hyper_connection.cuh` sums the partials of a 2560-wide sublayer output at 2, 4
or 8 ranks in rank order in F32, injects the sum into the four residual
streams with `2 sigmoid(logit / 4)` (below a sigmoid argument of -80 the
product is formed in F64, `common/sigmoid.cuh`) and writes each stream's
grouped Gemma norm, with no BF16 rounding between inject and norm.
`model_qwen38_contract::sm120::comm::launches` picks the entries, plans
and arguments of a call by rows. Every entry takes the rank count as its
last argument and traps on a count other than 2, 4 or 8; the planners
refuse a count that differs from the query's.

- Up to 32 rows (`ONE_SHOT_ROWS`), `chop_comm_qwen38_allreduce_hc_v1` runs
  one 8-CTA cluster that pushes the whole partial to each peer in turn, then
  reduces. It refuses a capacity above 32 on a shared state
  (`standalone_arguments` serves oracles on a state of their own).
- Above 32 rows, `chop_comm_qwen38_reduce_scatter_v2` then
  `chop_comm_qwen38_allgather_hc_v2` run with one sending cluster plus one
  worker per row, rounded to eight CTAs and clamped to 8 through 128 workers.
  The scatter sums the ranks in rank order in F32; the gather releases each
  owner's rows by block and finishes each (row, stream) group with the
  one-shot entry's inject and norm expressions. The state's flags end with
  32 eight-byte gather slots per rank. The model's comm README records the
  pair's qualification.
- Decode graphs (1 to 32 rows) run
  `chop_comm_qwen38_allreduce_hc_decode_r{2,4,8}_v3` at one or two prepared
  rows and `..._decode_r{2,4,8}_v2` above (`comm/hyper_connection_decode.cuh`)
  on a state of their own: low latency with one four-CTA cluster per row up
  to 2 rows, above that one sending cluster and one CTA per (row, stream)
  (see Lower-latency decode entries).
- Decode graphs above 32 rows run
  `chop_comm_qwen38_allreduce_hc_wide_r{2,4,8}_v1`
  (`comm/hyper_connection_decode_wide.cuh`, cubin `comm_qwen38_decode_wide`) on the
  decode state extended by `wide_buffers`, up to 256 rows at two ranks and
  128 at four and eight (`qwen38_decode::graph_takes_wide`); the phased
  pair runs above, where it measured faster (about 125 us against 155 us at
  256 rows on four ranks).

The one-shot and phased paths share one state from
`model_qwen38_contract::sm120::comm::buffers` (127 MB per rank at 8192
rows), with every region at a fixed offset from the capacity. The pair's
flags carry a 64-bit call generation (byte 8 of the epoch buffer, starting
at 1) and receivers compare for equality; the entries trap rather than wrap
it.

Kernel-lab tests: `tests/qwen38_comm.rs` (both paths against the oracle and
each other, alternating on one state and in one graph;
`tp4_mixed_rows_delayed_rank_g1` delays a different rank before each call
with the `comm_qwen38_skew` cubin, whose `chop_comm_qwen38_skew_*` entries
lag one rank through `test_skew`), `tests/qwen38_comm_decode.rs` and
`tests/qwen38_comm_wide.rs`.

## Logits gather

`chop_comm_gather_rows_{bf16,f32}_v1` gathers any number of vocabulary rows
to one rank in one launch; build its arguments with
`comm::gather_rows_arguments` and launch with `comm::gather_rows_plan()` (8
CTAs of 256 threads, PDL):

`(input, outputs[ranks], flags[ranks], epoch, row_indices, rows: u32,
shard: u64, rank: u32, ranks: u32, target: u32)`

`input` holds this rank's shards row-major, `[row][shard]`. Rank q's shard of
row r lands at `outputs[target] + (r * ranks + q) * shard`. `row_indices` is
a device array of `rows` u32 row indices (it may change between graph
replays) or null for rows 0..rows. Flags need `GATHER_ROWS_FLAG_BYTES` (288)
and the epoch `GATHER_ROWS_EPOCH_BYTES` (32), both zeroed and not shared with
the `allgather` entries. Senders write only to the target, and only rank
`target`'s output is written, every row in full. The target acknowledges the
previous call at the start of every call and senders wait for it on every
call, so no epoch value lets a sender overwrite rows a consumer on the
target has not read.

## Logits row exchange

`chop_comm_exchange_rows_{bf16,f32}_v1` gives every rank a block of full
vocabulary rows instead of gathering all rows to one rank. Rows split into
contiguous blocks of `rows.div_ceil(ranks)`, block q owned by rank q
(`kernel_core::row_exchange::exchange_rows_owned`); rank q's shard of row r
lands at `outputs[owner] + (r * ranks + q) * shard`, the gather's layout.
Build the arguments with `row_exchange::exchange_rows_arguments` and launch
with `row_exchange::exchange_rows_plan()` (one cluster of 8 CTAs of 256
threads, PDL):

`(input, outputs[ranks], flags[ranks], epoch, rows: u32, shard: u64,
rank: u32, ranks: u32)`

In phase p each rank writes only to rank (rank + p) % ranks and the cluster
synchronizes before the next phase. Flags need `EXCHANGE_ROWS_FLAG_BYTES`
(512) and the epoch `EXCHANGE_ROWS_EPOCH_BYTES` (32), zeroed and not shared
with other entries. A sender waits for its receiver's acknowledgment of the
previous call before overwriting, on every call. At 32 rows on four GPUs the
exchange took about a third of `gather_rows`' time (`comm_bench` variant
`logits_exchange`).

## Buffer sharing and other entries

Every protocol owns a byte region of the inbox (low latency from 0, one-shot
from 1 MiB, bulk two-shot from 5 MiB, wide all-gather slots from 5.5 MiB,
phased from 8 MiB) with strides for its largest row count, and fixed flag
words that no other protocol's peers write, all on epoch[0] or per-row epoch
cells that every protocol advances monotonically. The wide reduce-scatter
slots share the phased region: both protocols read those bytes before they
send anything a peer needs to finish the call.
`comm::tests::no_protocol_writes_where_another_reads_late` models every
protocol's writes and late reads. Calls with any row count and protocol can
therefore follow each other on one comm state, including in one graph, and
no other output is written. A flag word whose protocol stays unused for 2^31
calls can match a later mark (flags keep 31 epoch bits), so such a state
must be reset first. Distinct comm states must not share inbox bytes; the
MiMo bundle gives each state its own span of the inbox
(`scratch::comm_inbox`).

`chop_comm_allgather_{bf16,f32}_v1` with a target rank writes only to that
rank (the all-ranks mode fans out), but needs one launch per row.
`chop_comm_allreduce_oneshot_{bf16,f32}_v1` and
`chop_comm_allreduce_oneshot_norm_bf16_v1` keep the original one-shot, which
writes to all peers at once; the bundles do not use them. Optional output
quantization is not implemented. An overlapped compute schedule must reserve
SMs for comm workers; the lab probe uses `CHOPINFER_KERNEL_BENCH_OVERLAP` and
`CHOPINFER_KERNEL_BENCH_PREFILL_GRID`.
