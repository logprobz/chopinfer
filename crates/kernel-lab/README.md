# kernel-lab

GPU correctness tests, benchmarks and hardware probes for the SM120 kernels
(`kernels-sm120`) and the in-tree bundles' own kernels, recorded as legacy
evidence. The lab runs on the model-free harness (`crates/kernel-harness`,
re-exported as `kernel_lab::*`): sessions, launches, poison, conformance and
the kit. A new model does not register here; it runs its kernel tests from
its own test crate on `kernel-harness`, as `examples/external-model/lab`
does.

## Running GPU tests

GPU tests run only when exactly one of `CHOPINFER_KERNEL_GPU` or
`CHOPINFER_KERNEL_GPUS` names otherwise idle GPUs; without one they print a
skip notice and Cargo reports them as passed. Under
`CHOPINFER_KERNEL_REQUIRE=1` every skip (missing GPU selection, too few
ordinals, missing checkpoint or data, an opt-in benchmark or diagnostic
switch left unset) fails with the reason instead. Leave out what a strict run
does not cover with `--skip`; for the `ops` binary, these filters skip
exactly its opt-in benchmarks and diagnostics:

```sh
CHOPINFER_KERNEL_REQUIRE=1 CHOPINFER_KERNEL_GPU=0 cargo test -p kernel-lab --release --test ops --locked --offline -- \
    --test-threads=1 --skip benchmarks --skip _bench --skip base_cubins_build \
    --skip matches_base_bit_for_bit --skip linear_e5 --skip 47_layer_graph \
    --skip smalle --skip prefill_boundaries_bitwise --skip candidate_
```

| Variable | Meaning |
|---|---|
| `CHOPINFER_KERNEL_GPU`, `CHOPINFER_KERNEL_GPUS` | one ordinal, or an ordered rank-to-device list (up to eight distinct ordinals) |
| `CHOPINFER_KERNEL_REQUIRE` | `1` turns every skip into a failure; `0` or unset allows skips |
| `CHOPINFER_KERNEL_TIMEOUT_SECS` | whole-test deadline of every multi-GPU test (default 300) |
| `CHOPINFER_KERNEL_RECEIPTS` | receipts directory (see below) |
| `CHOPINFER_KERNEL_SHARED_DEVICE` | `1` labels every measurement `shared-device` |
| `CHOPINFER_KERNEL_BENCH` | `1` enables the opt-in benchmarks |
| `CHOPINFER_QWEN38_RANKS` | Qwen3.8 per-rank tests at 2 or 4 (default) ranks; TP2 runs lean cases |
| `CHOPINFER_MIMO_CHECKPOINT` | MiMo checkpoint for real-weight and verify-drift cases |
| `CHOPINFER_MIMO_AUDIT_DIR`, `CHOPINFER_QWEN38_LAYER_DUMPS` | dumped activations for `gemm::accumulation::gemm_accumulation_entries`, `dense_template` and the model audit |

Some GPU tests need inputs that no script in the repository sets; a GPU run
fails without them (it does not skip), so a strict run either sets them or
skips the tests:

| Input | Tests | What it holds |
|---|---|---|
| `CHOPINFER_BUNDLE_REPORTS` | `norm_weight_forms` (`f32_norm_rows`, `f32_norm_all_reduces`), `ngram_hash_forms` | `:`-separated `kernel-build-report.json` files of the bundle builds holding the declared forms (each bundle crate's build script output directory) |
| `CHOPINFER_KERNEL_OWNED_CUBINS` | the tests that load a bundle's own cubins (`tests/common/owned_cubins.rs`): the bundle rows, speculation rows and `ops` `bundle_invariance` | JSON manifest `{"cubins": {"<name>": {"path": "<cubin path>", "sha256": "<64 hex digest>"}}}` of a pinned bundle build's cubins |
| `CHOPINFER_KERNEL_EVICTION_AB=1`, `CHOPINFER_KERNEL_EVICTION=write` | `timing_receipts` on several GPUs | the paired timing arms the regression compares |

`local.env.example` lists the data paths. `CHOPINFER_QWEN38_LAYER_DUMPS/long_4k`
holds the BF16 taps (`taps/bf16`) of a GPU reference runner run over
`long_4k` ([refrun](../../tools/refrun/README.md)) with `--modes bf16 --taps
--tap-sites x.hc.attn.xn,x.hc.attn,attn.core,x.hc.mlp,x.hc.mlp.xn
--tap-rows` every position from 0 to 4095; the tests read layers 0, 1, 3
and 7.

CPU-only verification and compilation, with both GPU selection variables
unset:

```sh
cargo test -p kernel-lab --release --no-run --locked --offline
cargo test -p kernel-lab --release --lib --locked --offline
cargo clippy -p kernel-lab --release --all-targets --locked --offline -- -D warnings
```

## Lock families

`kernels.lock` gates the existing cubins. New engines lock by source family
in `kernels.lock.d/` ([lock directories](../kernel-build/README.md)), one file
of `families/` per family
([adding a family](../../docs/kernel-library.md#adding-a-kernel-lab-lock-family)).
`channel_decay.lock` gates `channel_decay_h16x128_sm120`, the
per-channel-decay delta rule engine, whose tests are `tests/channel_decay.rs`:

```sh
CHOPINFER_KERNEL_REQUIRE=1 CHOPINFER_KERNEL_GPU=0 cargo test -p kernel-lab --release --locked --offline \
    --test channel_decay -- --test-threads=1
```

`channel_decay_tiled.lock` gates `channel_decay_tiled_h16x128_sm120`, the
same engine's tiled prefill stages, and its five lab-only mutant cubins,
whose tests are `tests/channel_decay_tiled.rs` (bitwise against the staged
stages; the latency table also needs `CHOPINFER_KERNEL_BENCH=1`):

```sh
CHOPINFER_KERNEL_REQUIRE=1 CHOPINFER_KERNEL_GPU=0 cargo test -p kernel-lab --release --locked --offline \
    --test channel_decay_tiled -- --test-threads=1 --skip latency
```

`encoder_vision.lock` gates `encoder_vision_rows_sm120`, the vision encoder's
zero-column pad, q/k RMSNorm, clamped SwiGLU and F32 GELU row entries at their served shapes
(`encoder_vision_shapes.rs`), whose tests are `tests/encoder_vision_rows.rs`:

```sh
CHOPINFER_KERNEL_REQUIRE=1 CHOPINFER_KERNEL_GPU=0 cargo test -p kernel-lab --release --locked --offline \
    --test encoder_vision_rows -- --test-threads=1
```

## Synthetic attention shapes

`shapes.rs` declares lab-only dense, gated, sparse attention and indexer
instances, including head-128 rotary-64 rows and wider indexers. The lab build
emits their cubins; `tests/lab_shapes.rs` checks their catalog contracts on
the CPU, and the `ops` binary launches every one on a GPU against its oracle
(the lab-shape cases of its attention and sparse attention modules, and the
sparse invariance adapter).
See the maintained [kernel instance table](../../docs/kernel-library.md#engines)
for supported shapes and dated GPU qualification results. CPU catalog coverage
alone does not establish GPU correctness or qualify a model checkpoint.

## Sessions and receipts

`Session::open` checks declared free memory and records device identity,
clocks, memory, driver and other compute processes. `Session::receipt`
creates a unique JSON file under `<receipts>/<kind>/`, where `<receipts>` is
`$CHOPINFER_KERNEL_RECEIPTS`, else `$CHOPINFER_DATA/kernel-lab`, else
`chopinfer-kernel-lab` under the system temporary directory (never the
source tree). A measurement with another compute process present, or any
under `CHOPINFER_KERNEL_SHARED_DEVICE=1`, is labelled `shared-device`.

Engine cases use `gpu_test`, `gpu_test_multi`, `Session::upload_f32`,
`upload_bf16`, `download_f32`, `download_bf16`, `module`, `capture`,
`launch::launch`, `check_output` and `time`. `launch` resolves kernel-core
buffer handles and pointer arrays and maps a `LaunchPlan` to runtime-cuda
(tensor maps of rank 1 to 5 and PDL, in `kernel_harness::launch`). `time` gates the
stream, evicts L2 between iterations, and reports at least three
rounds. Timings are evidence only on exclusive, idle GPUs.

## Probe

`kernel-lab probe` (`cargo run -p kernel-lab --release --locked --offline --
probe`) records SM120 cubin execution, cluster DSMEM and control, gather4,
dynamic shared memory, PDL overlap, DRAM and L2 bandwidth, per-format MMA
TFLOPS, and peer pull, push and flag latency for every ordered GPU pair.
`kernel-lab probe p2p` runs just the peer-pair probes and needs at least two
ordinals in `CHOPINFER_KERNEL_GPUS`. CUDA 13.2 lowers cluster TMA multicast
on `sm_120f` to an undefined syscall, so the cubin audit blocks it.

## Generated invariance tests

Every catalog entry, and every bundle's own kernel, declares the exactness it
guarantees (`kernels_sm120::invariance`: chunk, rows, batch, ties, anchors);
for example, prefill rows must not depend on how a prompt is chunked
([numerics](../../docs/numerics.md)). `tests/common/invariance.rs` has one
test per property that iterates the declarations and hands each entry to the
adapter of its op kind, which launches it and returns bytes the harness
compares bit for bit. Every buffer a compared launch writes starts poisoned
and is checked against the adapter's write map; each adapter family has a
CPU negative control for its maps (`*_write_checks_reject_*`). The `ops`
binary runs `invariance_tests::invariance_{chunk,rows,batch,ties,anchors}` on
one GPU and the `comm` binary `invariance_comm_{chunk,rows,batch}` on
several; `CHOPINFER_INVARIANCE_OPS` and `CHOPINFER_INVARIANCE_FILTER` narrow
a run. `invariance_claims_have_adapters` fails on the CPU when a declared
property has no adapter. An entry a run cannot place on its host is reported
with `"qualified": false` and the reason, never as a pass. Collectives need
ranks on distinct devices; an eight-rank collective needs eight GPUs.

## Poisoned replay checks

`kernel_lab::poison` (`kernel_harness::poison`) keeps a relaunch, graph replay or PDL
launch from passing on bytes an earlier launch left behind. It fills buffers
with 0xFF bytes (NaN in BF16, F32, F64 and E4M3, and no valid index, count or
status) through host copies that synchronize the stream. A `WriteMap` holds
the byte ranges a launch writes: every written element must differ from its
fill, and every other byte must keep it. `Replay` runs the `eager`,
`relaunch`, `graph` and `pdl` phases over outputs, scratch (never checked,
or limited by `scratch_within`) and in-place states (starting bytes
restored); each phase after `eager` must match the eager bytes bit for bit.

Use the shared helpers instead of local copies. `replay_all` runs a launch
through all four phases and uses a `LiveProducer` in the `pdl` phase only: the
PDL probe producer that republishes the live count it reads from the device,
never a caller's count. `CopyProducer` instead rewrites a real input. Every
map starts from `WriteMap::of(dtype)`, which counts in elements of the
buffer's dtype (`prefix`, `rows`, `ranges`; `byte_prefix` and `byte_ranges`
place whole elements by byte offset), or `WriteMap::of_records::<N>()` for
records no dtype names, such as 8-byte (score, index) pairs; there is no
byte-sized constructor, so a BF16 map in 1-byte elements does not compile.
`Checks` and `phase_all`
check compared multi-kernel units, whose maps may come from freshly produced
route IDs (`sorted_map`, `routed_map`). Test inputs use
`tests/common/support.rs` (`Rng`, `upload`, `download`, byte conversions).
Accumulation references and gates use `kernel_core::oracle`'s
`compensated_sum`, `accumulation_bound` (2^-22 K S) and `max_bound_use`.

The rules, also in the module doc:

- Poison every buffer a launch writes and the next launch rewrites before
  reading it: outputs, inter-kernel intermediates, and split-K or stream-K
  partial workspaces, before every launch and every PDL producer.
- Restore the bytes a launch reads before it writes them.
- Never poison self-resetting semaphores or counters, the MoE stream-decode
  flag prefix, or comm inboxes, flags, epochs and generations.
- Poison only outside capture, never between a PDL producer and its
  dependent, and across ranks only after every rank's stream is idle.
- Where 0xFF is a valid value, use another fill: packed E2M1 codes take the
  complement of the expected codes, and sorted route rows take 0x80.

`Replay::record` (or `Evidence::record`) writes `deterministic_bitwise`,
`graph_bitwise` and `pdl_bitwise` only for phases that ran poisoned and
matched; `pdl_bitwise` also needs PDL active.
`attention::poisoned_replay_catches_a_graph_writing_elsewhere` is the
negative control.

Coverage includes the MoE decode, prefill, small-expert and parallel
routing suites, including boundary fixtures and sorted routing metadata.
Encoder position and attention conformance fixtures, DFlash row and
argmax entries, and linear-E5 checkpoint and diagnostic fixtures also check
their compared launches. Small-expert diagnostic modes 0 and 1 use fresh poisoned eager
and graph launches outside timing; modes 2 and 3 are timing-only cutouts.

The model adapters cover PLE items, mapped-host n-gram tables, HC row and
projection variants (including E5, up-mix and precompute), parallel routes,
rows, GEMMs and the MoE tail. Concurrency fixtures cover GEMM, HC, MoE,
linear and attention graphs, combine, and phased collectives. Fragment
prefill GEMM partial maps require only the slots the stream-K schedule
publishes; single-segment entries require an empty partial map. Every
unlisted byte must retain its fill, while semaphores remain intact.

KV offload fixtures poison H2D destinations independently of their expected
inputs and check both copied pages and untouched regions. D2H destinations
receive independent fills. Compared graph calls use distinct output banks
where earlier calls must remain observable. Prefill chains, local-context
and frozen later-layer fixtures check outputs, routing intermediates,
attention partials and cache writes, with their prelaunch state restored.
Model-owned selection checks exported contracts and their compared GPU
launches; these examples establish replay mechanics, not qualification of a
served model. The prefill stage maps currently cover nonzero layers, one
sequence, zero prefix and the unfused route path; other configurations need
their own write maps before receiving the same evidence.

Timing-only launches and advancing-state timing graphs stay outside these
checks and earn no replay claims. The full-model audit and drafter timing
fixtures do not gain same-input replay coverage from their existing oracle
or trace checks. Schedule-dependent scratch checks distinguish required
stores from possible publication regions; poisoning an allocation does
not establish that every element was written. Clock diagnostics check
required and untouched slots, but exclude timestamps from bitwise output
comparisons. External E5 diagnostic receipts identify the cubin, generated
source, source closure and SASS digests.

Frozen cubins, captured inputs and numerical snapshots are never regenerated
to resolve a mismatch. Small-expert snapshot tests remain strict by default.
The explicit `CHOPINFER_MOE_SMALLE_REPLAY_ONLY=1` mode checks the numerical
oracles and poisoned replays without historical snapshot comparison; it
rejects simultaneous golden-file or recording settings. Receipts distinguish
that omission from an actual snapshot comparison. All fills, restores,
downloads and staging allocations stay outside stream capture, and gated
copy fixtures allocate release staging and timed events before arming gates.

The live-count tests (`live::` in the `ops` binary: attention, row,
linear and QSA) capture one graph per bucket and replay it at the live
counts of `poison::live_counts` (the capacity, 0, 1, half the capacity and
one below it), with rows past the live count required to keep their poison.
`row::live` fails its oracle check on the computed-angle RoPE entries, whose
F32 angles miss the bound past small positions
(`row::row_rope_computed_angle_position_511_gpu` reproduces it).

```sh
CHOPINFER_KERNEL_GPU=3 cargo test -p kernel-lab --release --test ops --locked --offline -- live:: --test-threads=1 --nocapture
```

## Pool offsets past the 32-bit lines

Tests whose names contain `high_ids` hold live KV pages, ring slots, QSA
block-key pages, linear-state slots and checkpoints, PLE entries and n-gram
table rows past two lines: the first ids whose byte offset exceeds
`i32::MAX`, and the first at 2^32 elements of the pool's element type. Each
pool holds the sentinel 0xff everywhere else, so an offset truncated to 32
bits reads or writes it; the high run must reproduce the low run bit for bit.
`tests/common/pool_offsets.rs` holds the helpers and, in its module
documentation, the audit of every id-scaled product in the kernels. The
tests need up to 25 GiB of free device memory.

```sh
for test in ops qwen38_rows qwen38_bundle_rows; do
    CHOPINFER_KERNEL_GPU=0 cargo test -p kernel-lab --release --locked --offline \
        --test $test high_ids -- --nocapture || exit $?
done
```

## Collectives

The comm tests (`tests/comm.rs`, `comm_sync.rs`, `comm_wide.rs`,
`comm_latency.rs`, `qwen38_comm*.rs`) use the selected rank count. They
cover F32 and BF16 reduction protocols, fused combine, phased prefill, row
gather and all-gather against CPU/f64 references, with cross-rank
comparisons, protocol changes, delayed consumers and epoch wraps;
`CHOPINFER_KERNEL_COMM_TOKENS` selects one row count. The protocols and
their contracts are in the [comm README](../kernels-sm120/cuda/comm/README.md).

Each rank needs a distinct GPU. Duplicate ordinals are refused, and
eight-rank comm tests need eight GPUs.

Every `gpu_test_multi` has a host watchdog (`CHOPINFER_KERNEL_TIMEOUT_SECS`,
default 300 seconds). On expiry it writes `kernel-lab watchdog`, the test
name, rank-to-device map and each rank's last submitted launch to stderr and
exits the test binary with 124 without CUDA cleanup (Cargo then reports exit
101). A watchdog exit is a failed run, not a reason to change grids or
arithmetic.

Correctness matrix, on otherwise idle GPUs without `CHOPINFER_KERNEL_BENCH`:

```sh
unset CHOPINFER_KERNEL_GPU CHOPINFER_KERNEL_BENCH
export CHOPINFER_KERNEL_TIMEOUT_SECS=300
for devices in 0,1 2,3 0,1,2,3; do
    CHOPINFER_KERNEL_GPUS=$devices cargo test -p kernel-lab --release \
        --test comm --test comm_sync --locked --offline -- \
        --test-threads=1 --nocapture --skip low_latency_needs_its_zeroed_region || exit $?
done
```

`low_latency_needs_its_zeroed_region` seeds LL inboxes with stale future
flags, so its malformed first phase may deadlock by design; run it in its own
process per topology (`--exact low_latency_needs_its_zeroed_region`). An exit
124 there proves only that the watchdog fired.

## Benchmarks

Benchmarks are opt-in (`CHOPINFER_KERNEL_BENCH=1`, or an ignored test run
with `--ignored`) and meaningful only on exclusive, idle GPUs.

- `tests/comm_bench.rs`: `CHOPINFER_KERNEL_BENCH_VARIANT` selects a case
  (`push`, `local_push`, `one_shot`, `plain_one_shot`, `prefill`, `logits`,
  `logits_exchange`, `peer_push_fanout`, `peer_push_pull`, `peer_control`,
  `launch_floor` and the `local_*` controls); `CHOPINFER_KERNEL_BENCH_TOKENS`,
  `CHOPINFER_KERNEL_BENCH_ROWS`, `CHOPINFER_KERNEL_BENCH_BLOCKS`,
  `CHOPINFER_KERNEL_BENCH_PREFILL_GRID` and `CHOPINFER_KERNEL_BENCH_OVERLAP`
  narrow or vary it. Receipts under `<receipts>/comm_bench/` report the
  slowest rank's median.
- `tests/comm_latency.rs`: `decode_latency_g4` writes under
  `CHOPINFER_COMM_LAT_OUT`. `tools/sass_same.sh` compares entries' SASS
  between two cubins (`tools/test_sass_same.sh` runs in `ci/cpu.sh`). The
  decode v2/v3 pair alternates A/B and B/A across nine windows, each arm
  independently evicted and gated across ranks. Each window reports three
  samples and their slowest-rank median, all rank clock records, and the
  paired B/A ratio. `CHOPINFER_COMM_LAT_ROWS=1,2` restricts latency to LL
  rows. Correctness still covers every supported bucket and live count.
  Decode comparisons use cancellation inputs with wrong-order controls and
  per-source markers that change for every logical call. Candidates run
  before the oracle on separate comm state. Each mixed-row graph call has
  its own input and output buffers, and every replay gets new inputs and
  poisoned outputs. Skew tests keep state across those replays and check
  every call through the epoch wraps.
- `gemm::chunk_cases::gemm_prefill_chunk_bench` (`CHOPINFER_KERNEL_BENCH_POINTS`
  as `rows@grid` points, `CHOPINFER_KERNEL_BENCH_FILTER` by symbol),
  `attention::chunks::attention_prefill_chunk_bench`,
  `attention_exact_rows_bench`, `grouped_moe_mimo_exact_rows_bench` and
  `qwen38_skinny_prefill_splits_bench`.
- `CHOPINFER_QSA_BENCH=1` times the QSA stages into
  `<receipts>/sparse_attention_bench/`; `CHOPINFER_QSA_CASE=<name,...>`
  selects QSA cases ([SPARSE.md](../kernels-sm120/cuda/attention/SPARSE.md)).

```sh
CHOPINFER_KERNEL_BENCH=1 CHOPINFER_KERNEL_GPUS=0,1,2,3 cargo test -p kernel-lab --release --test comm_bench --locked --offline -- --nocapture
```

The MoE and GEMM switches are listed in the
[MoE](../kernels-sm120/cuda/moe/README.md) and
[GEMM](../kernels-sm120/cuda/gemm/README.md) READMEs.

## Model audits and encoders

`tests/mimo_model_audit.rs` `gpu_hidden_states` (four GPUs,
`CHOPINFER_MIMO_CHECKPOINT`, `CHOPINFER_MIMO_AUDIT_DIR`) runs the MiMo
embedding, all 48 layers and the head with bundle-mimo26's launch choices,
one layer at a time, and dumps hidden states, residuals, attention outputs,
routes and logits; `tools/moe_routing_locality.py` reads the routes. The
real-weight tests read their weights through the host's direct-load
executor (`tests/common/mimo_planned.rs`). Model references come from the GPU
reference runner (`tools/refrun`), whose taps `python3 -m tools.qual.tapdiff`
compares layer by layer ([debugging](../../docs/debugging.md)).

Generated encoder component conformance runs against each bundle's serving
cubins in `tests/encoder_conformance.rs` and
`tests/encoder_conformance_qwen.rs`. Encoder diagnostics use the bundles'
runtime taps (for example `enc.<tower>.B<nn>.out`) through
`CHOPINFER_TAPS_DIR`; the declaration-driven GPU reference is
`python3 -m tools.refrun.encoder --declaration <declaration>`.

## Cache preparation and timing comparisons

Read-sweep eviction is the default. Receipts from before this default switch
used a 256 MiB write flush in `time` and `time_after`, unless read mode was
selected explicitly for an A/B run. `time_multi_graph` previously reused
warm caches; it now uses the shared eviction helper, so its older timings
are not comparable. Write eviction remains selectable. Treat a paired ratio
more than 2% from 1 as a material difference, after inspecting clock and
throttle flags.

On 2026-10-03, 10:54-11:15 UTC, all seven validation jobs passed on four
RTX PRO 6000 Blackwell Max-Q GPUs, with one job assigned to each GPU pair.
The other pair was busy with another kernel-lab or reference job, so these
were assigned-device measurements with concurrent work elsewhere in the
machine. Receipts reported 128 MiB L2
and a 250 W power limit. The comparison alternated write/read and read/write
arms per round on identical work. No measured A/B arm was flagged for power
or thermal throttling or SM clock drift.

Read eviction was equal or faster almost everywhere. GDN delta decode and
verify at batches 8-32 took 5-31% less time; the eager
`delta_verify_c32_r4` median fell from 98.3 to 67.6 us. Full decode MoE
measurements took about 3-6% less time, paged decode at batches 8 and 32
about 5%, and the batch-32 QSA step about 5%. The MiMo output projection at
M=32 took about 7% less time, and the bulk-write prefix calibration about
15.5%. Head GEMMs and small kernels were generally within 1-2%. These
results support the hypothesis that dirty write-flush lines charge DRAM
writebacks to the timed work, and justify the read-sweep default; they do
not directly establish the writeback mechanism.

CUDA event timing on this GPU quantizes near 2.048 us. Differences below
2 us on small kernels are not resolvable with this timing method and should
not be treated as performance improvements.

`CHOPINFER_KERNEL_EVICTION=write|read` selects eviction for the shared helper.
Read mode reuses `chop_probe_read_v1`: it reads `max(2 * L2, 64 MiB)` once,
reduces to one checksum with one atomic per CTA, and performs no bulk writes.
The CUDA L2 attribute is queried for the selected device. The sweep buffer
is initialized once and synchronized before timing. PDL producer, input initialization and delayed
rank controls still use their original writes because those writes are part
of the behavior under test, rather than cache preparation.

`time_prepared` orders eviction, activation production, then the timed target.
Its producer is outside the event interval. Use it to model cold weights and
warm activations in serving; `time` keeps the cold roofline mode. Producers
must enqueue on the supplied stream and avoid reading weights they intend
to keep cold.

`time_ab` independently evicts each arm and alternates A/B, B/A per measured
round. It returns each arm's median and the median of paired B/A ratios.
Even round counts now average the two middle samples.
`CHOPINFER_KERNEL_EVICTION_AB=1` uses this helper inside `time` to compare
write (A) and read (B) eviction on identical work. The ordinary return value
belongs to the selected eviction mode. Inspect `timing_rounds` in the receipt
or run `python3 crates/kernel-lab/tools/eviction_table.py RECEIPT_DIRECTORY`
to print the paired table.

Each measured round records SM and memory clocks, clock event reasons,
pstate, power draw and limit in milliwatts, and temperature through NVML.
Snapshots occur after event completion, outside the measured interval; they
cannot detect a transient throttle event that clears before the query.
Power or thermal reasons (including hardware slowdown) and an SM clock
more than 5% from the session opening clock flag the round. Query failures
are recorded explicitly. Flagged rounds remain in the statistics so the
receipt exposes the complete sample set; they need review before a performance
claim. Timing records are drained into the next receipt for that device on
the timing thread.

Timing operations can use `time_named`, or `with_timing_label` around
`time_after`, to retain their own label with the measurement ID. The A/B table
uses these labels, including launch-floor calibration and individual QSA or
recurrent stages, rather than borrowing the receiving receipt's label.
Older receipts without operation labels are shown as `unlabelled`.

`time_after` collects target-only samples separately in each eviction arm;
the prefix is outside the timed interval. Aggregate graph measurements keep
all rank snapshots in `GraphTiming::gpu_rounds` and in rank 0's receipt,
including flags from any participating rank. Benchmark jobs should generate
an assigned-device `kernel-lab probe` receipt in their own receipt directory
before benchmarking; recurrent rooflines require it and other rooflines use
it in preference to nominal fallback ceilings.

## Non-dense BF16 chain measurements

`non_dense_accumulation` measures encoder BF16 pairs and frozen online-softmax
PV inputs with the separate `probe_non_dense_accumulation_sm120` cubin. It does
not modify a served template. `encoder_pairs_adversarial` compares its long
chain bitwise with the unmodified encoder template before the epilogue, and
checks FMA against a CPU emulator. `pv_adversarial` preserves each captured
32-key tile's probability residuals and rescale schedule. Its short-chain modes
include tile temporaries, at most eight MMA calls, and 128-key temporaries.
A 128-key combined pair temporary contains sixteen calls, so it is measured
separately from the eight-call alternatives.

`encoder_pairs_real` and `pv_real` require `CHOPINFER_ACCUMULATION_MANIFEST`.
`CHOPINFER_ACCUMULATION_CASE` selects one case; an empty selection fails.
`CHOPINFER_ACCUMULATION_K` selects one adversarial K. A manifest uses schema
`chopinfer.non-dense-accumulation.v1` and a `cases` array. Each case states
`id`, `family` (`encoder_pairs` or `pv`), `origin`, `source_sha256`, `n`, `m`,
`k`, mathematical `reduction_k`, `terms`, and file records with relative
`path` and `sha256`. Files are little-endian row-major BF16 `weight[n,k]`
and F32 `activation[m,k]`. PV adds F32 `correction[m,k/32]` and
`denominator[m]`; encoder cases set these fields to null. The activation is
the original F32 encoder input or the unnormalized PV probability at its
own tile maximum. Values for PV are transposed into `weight[n,k]`.

The GPU-only wrapper `tools/capture_non_dense.py` adds bounded operand hooks
to existing reference APIs without editing reference-bound sources. It writes
reference-policy captures, which do not prove equality to served intermediate
operands. Pass the wrapper's arguments before `--`, then the reference CLI
arguments, including explicit CUDA devices. The runner retains its checkpoint
verification and repeat checks. No manifest is published on a failed or empty
capture. `python3 -B tools/test_capture_non_dense.py` checks capture geometry
and empty-publication rejection without importing a GPU framework.

Receipts distinguish decoded-operand accumulation error from original-input
representation error and report max/p99.99 F32 ulps, contract-term ratios,
bias and BF16 misrounding. Small tail samples are marked. Contract failures
are recorded before the test fails. Frozen PV numerator and denominator
results do not qualify full attention, softmax, KV storage or epilogues.
Run assigned-device jobs with `CHOPINFER_KERNEL_REQUIRE=1` so missing inputs
or devices cannot silently pass. Measurement receipts determine a later
numerics recommendation; this harness does not authorize a kernel switch.

`grouped_accumulation` extends the study to expert reductions. It requires
`CHOPINFER_GROUPED_MANIFEST` with schema `chopinfer.grouped-accumulation.v1`;
`CHOPINFER_GROUPED_CASE` selects one fixture and rejects empty selections.
Each fixture has digest-bound gate/up/down banks, actual routed input rows,
a rank topology, and explicit K ownership intervals. Decoded BF16 operands
may chain over an ownership interval. Raw FP8-converted BF16 operands retain
F32 post scales at K128 boundaries; candidate grains do not round scaled
weights to BF16. K32 and K128 baseline promotions model prefill and decode.

`tools/capture_grouped_accumulation.py` accepts an explicit checkpoint/dump
recipe on assigned CUDA devices and loads only selected expert matrices. It
verifies checkpoint files with the existing reference lock machinery, chooses
actual hot/rare routed rows, and records their indices and input digests.
The lab freezes one baseline GPU SwiGLU intermediate, rounded to BF16 once,
for every down candidate. Receipts identify that derivation and do not claim
routed-combine or complete model equivalence. Stream ownership intervals must
come from an actual dispatch trace; equal hypothetical portions are not
presented as served schedules. The measurement does not change MoE numerics.

`grouped_adversarial` checks probe parity and FMA emulation. CPU-only checks
cover unrounded post-scale references and invalid ownership schedules, plus
`python3 -B tools/test_capture_grouped_accumulation.py`. Real NVFP4 checkpoint
operands need a distinct capture; synthetic preparation fixtures cannot stand
in for them.

`matrix_real` accepts `CHOPINFER_MATRIX_MANIFEST` with schema
`chopinfer.matrix-accumulation.v1`. `tools/capture_pv_products.py` supplies
compact encoder pair, windowed and QSA PV matrices on GPUs. It expands each
k16 group in probability-term/value-half order, keeps per-row visible K
separate from the expanded physical reduction, and excludes padded features.
Encoder baselines preserve 64-key tile temporaries. Window and selected-key
baselines measure stable-maximum chains. QSA captures the executed F32 eager
softmax before its BF16 cast and preserves the captured selection set in
ascending native key order; it never recomputes top-k. Text sinks keep their
zero-value denominator mass, while encoder sinks remain masked key-0 biases.

These matrices do not qualify online-softmax, FP8 KV storage or complete
serving parity. Run `grouped_adversarial` first: its strict PV operand-orientation
check must pass before using matrix measurements. The wrappers retain normal
reference repeat checks and restore their hooks in `finally`. CPU geometry
checks are in `tools/test_capture_pv_products.py`.

The grouped controls also call the unchanged TMA `mma_chunk` consumer on its
actual shared-memory swizzle and require bitwise equality to K128 runs. This
checks the eight-call arithmetic shared by the standard dense and SwiGLU
mainloops; it is not a TMA scheduler or loading benchmark.

## F32 norm side tensors

`tests/norm_weights.rs` covers shared grouped RMSNorm rows with F32 weights
and BF16 activations/output: four groups of width 2560, one group of width
10240, and one group of width 2560. PLE uses the same four-group instance
and is covered by that test, without a duplicate case. `GroupedRmsNorm`
takes weight dtype, group width, group count and an explicit reduction order.
`ShuffleDown` retains the per-stream order; `XorThenSerial` retains the
across-stream and embedding order. Both form `(1 + weight)` in F32 with
epsilon 1e-6. The planner refuses grid-x capacities above 2^31-1.

`Epilogue::with_norm_weight_dtype(F32)` selects the shared inject-norm
functor and types the prefill, decode and wide transport weight pointer as
`const float`. BF16 selection retains the original header, functor and
wrapper. F32 decode loads weights at finish: its cached-parameter overload
is an ABI check and forwards to the same arithmetic. The whole-token form
loops over the streams. No serving catalog is changed.

The lab checks f64 oracles with non-BF16-representable weights near -1,
poisoned relaunch and graph replay, inactive rows, chunking and reordered
batches. Activations span BF16 exponents and mantissas. A sparse adversary
has x[0]=4096, x[32*w]=1 for w=1..7, and zero elsewhere. Its F32 square
sums are 16777222 with shuffle-down and 16777216 with XOR/serial; the BF16
weight 0x3a14 makes the final BF16 outputs differ at widths 2560 and 10240.
CPU replay proves that difference, and GPU wrong-tree controls require it
for each shape. The consumer-parity case in the model's speculation-row
tests compares shared BF16 and representable-F32 weights against the
existing HC and MTP symbols and the original inject-norm epilogue. The
legacy CUDA headers are loaded directly, not copied into the test.

Three local comm probes cover rank-ordered sums of 2, 4 and 8 inputs.
Logit -400 covers the inject tail; a separate zero-residual fixture uses
large partials so the F64 tail product remains observable and is checked
against the f64 oracle. The original BF16 tail helper remains untouched
to preserve its source-closure locks. These one-GPU probes do not qualify
peer transport. CPU tests compile and audit F32 prefill, decode and wide
instances at all three rank counts.

The original `zz_norm_weights_sm120` record has six row entries and three
F32 epilogue probes. Two appended cubins add three opposite-tree BF16 rows
and three wrappers of the original BF16 epilogue. The latter record belongs
to the existing model-owned lab lock beside its CUDA source. All earlier lock records
are unchanged. Run the `norm_weights` test binary and the model speculation
binary's `norm_weights_legacy_consumers` case on one idle GPU with
`CHOPINFER_KERNEL_REQUIRE=1`. CPU checks and skipped GPU tests are not GPU
evidence.
