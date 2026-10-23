# Compressed scale bank expansion

`scale_expand.cuh` decodes `scale-row-offset-v1` banks without interpreting
the scale codes. The bit-exact CPU oracle is
`kernel_core::precision::scale_offset::decode_bank_matrix`.

The generated C entries are `chop_moe_scale_expand_w1_v1`,
`chop_moe_scale_expand_w2_v1` and `chop_moe_scale_expand_w4_v1`. Each has
the same signature, with the offset width compiled into its body:

```cpp
extern "C" __global__ void chop_moe_scale_expand_w1_v1(
    const uint8_t* bank,
    const int32_t* active,
    uint8_t* output,
    uint32_t rows,
    uint32_t groups,
    uint32_t projections,
    uint32_t experts);
```

Launch with grid `(ceil(rows / 64), projections, slots)`, block
`(256, 1, 1)`, no dynamic shared memory, no workspace and PDL enabled.
`slots` is the expert count, or with an active list any bound on its
count (`plan_slots`); slots past the count exit at once. Every CTA calls
`griddepcontrol.launch_dependents` and then `griddepcontrol.wait` before
it reads the active list or writes the output, so the expansion follows
its predecessor's completion: the routes that write the list and,
transitively, every earlier reader of the output whose kernel waits for
its own predecessor. The bundle's launch lists admit only PDL launches.
Each CTA owns one 64-row tile of one expert projection. It binary-searches
the sorted exception records for that tile, writes all base-plus-offset
bytes, waits at a block barrier, then applies the exceptions. Output bank
offsets use u64 arithmetic. There are no cross-CTA writes for valid inputs.

The caller supplies a bank produced by the codec's encoder and
`bank_layout`, or validates the equivalent structure before upload. The
bank is 16-byte aligned, has u32 offsets and at most `2^24` scales per
matrix. All matrices share the query's shape and offset width. Actual
bank contents, alignment and allocation overlap are load-time obligations;
the launcher checks geometry, buffer sizes and obvious handle aliases.
Experts and projections are positive and at most 65,535 each for this grid.

The optional active list is 4-byte aligned and has `experts + 1` i32 words.
Its first word is a count in `0..=experts`, followed by unique expert IDs
in `0..experts`, in any order. The rest is unread. An empty list writes
nothing. A null pointer selects all experts. Duplicate IDs violate the
contract because separate CTAs would race on the same output locations.
The output is a disjoint byte buffer of at least
`experts * projections * rows * groups` bytes, with no alignment requirement.
Matrix `m = expert * projections + projection` starts at byte
`m * rows * groups`. Unlisted matrices remain untouched. Bytes 253 through
255 and wrapping base-plus-offset addition are preserved exactly.

## Paired entry

`chop_moe_scale_expand_pair_v1` expands two banks of one expert count in one
launch, each at its own runtime offset width:

```cpp
extern "C" __global__ void chop_moe_scale_expand_pair_v1(
    const uint8_t* first, const uint8_t* second, const int32_t* active,
    uint8_t* first_output, uint8_t* second_output,
    uint32_t first_rows, uint32_t first_groups, uint32_t first_projections,
    uint32_t first_width, uint32_t first_tile_rows,
    uint32_t second_rows, uint32_t second_groups, uint32_t second_projections,
    uint32_t second_width, uint32_t second_tile_rows,
    uint32_t experts);
```

Grid x covers the first bank's tiles, then the second's; grid z holds the
slots, block `(256, 1, 1)`, PDL as above. A null encoded bank takes no
tiles, so either bank may be absent (`pair_arguments` packs null pointers
and a zero shape). `PairBank::tile_rows` sizes each bank's tiles: whole
64-row strips holding at least 8 KiB of scales, so DeepSeek's TP4 gate/up
bank (160 groups) takes 64-row tiles and its down bank (18 groups) 512-row
tiles. Each thread writes four output bytes at a time where the output is
word aligned, with byte loops for an unaligned head and the tail, then
applies the tile's exceptions after a block barrier. The output contract
is the single-bank entries', per bank.

## Registration and bundle use

`kernels_sm120::moe::scale_expand::cubin_spec(name)` returns a standalone
`kernel_build::CubinSpec` with all four entries. Add it once to the bundle's
selection, in a new lock family, and include its header in the exported
source set if that set is explicit. The cubin is independent of model
geometry and topology. Do not add entries or includes to existing MoE
cubins: that would change their locked source closures.

Use `ScaleExpandQuery`, `symbol(query.width)`, `plan(query)` and
`arguments(query, encoded_view, Some(active_view), raw_view)` to resolve,
plan and pack each launch. Passing `None` packs a null active pointer.
Views carry opaque handles and byte extents; the bundle's host services
resolve them as for other kernel-core argument lists. This is a low-level
launcher, with no new generic `Backend` method.

The DeepSeek bundle allocates both raw banks in one scratch region, the
down bank after the gate/up bank at a 256-byte boundary, and launches the
paired entry once per layer at the end of the routes stage, before the
first scale consumer, binding that raw scratch as the existing scale roles.
Gate/up uses `rows=576, groups=160, projections=2` at TP4; down uses
`rows=5120, groups=18, projections=1`; both use 384 experts. The
expansion stays in stream order between the routes and the consumers,
including inside captured graphs, on the routed branch beside the shared
expert. Every consumer the bundle selects reads its weight scales only
after its own `griddepcontrol.wait`; the one-row floor kernel loads them
before its wait, so it must keep binding raw banks.

The direct path already binds `active` to `scratch.active_experts` in
`models/deepseek41/bundle/src/lower/moe_direct.rs`. `route_floor.cuh` and
the fused `router_finish.cuh` / `router_finish_r2.cuh` publish the same
`[count, ids...]` prefix: unique IDs in ascending order. The count is at
most `min(experts, live_tokens * top_k)`. An empty step publishes zero;
unused words can retain old values. These are full expert IDs, independent
of tensor-parallel channel slices. The separate small/wide queues read by
`chained_streamed.cuh` contain `(expert, route_count)` pairs and should not
be passed as the active list. `chained_active_quantized.cuh` already reads
the active prefix. No new route conversion or host count readback is needed.

The lab registers `chop_moe_scale_expand_lab` in the `moe_scale_expand`
family. New lock records are required for that family and, when integrated,
for the bundle's own family and cubin name. Use the pinned-toolchain builder
to initialize them; do not rewrite an existing family's records. No existing
bundle lock or closure needs an update for this standalone header alone.

## Targeted validation

Through the configured shared build queue, build and verify only this family:

```sh
source ~/.config/chopinfer/local.env
set -a; source ci/rust-profile.env; set +a
export CHOPINFER_SCALE_EXPAND_CUBINS="$CHOPINFER_DATA/scale-expand-cubins"
"$CHOPINFER_HEAVY_WRAPPER" --label scale-expand -- \
  cargo run --locked --offline -p kernel-lab-families --example build_family -- \
  moe_scale_expand "$CHOPINFER_SCALE_EXPAND_CUBINS"
"$CHOPINFER_HEAVY_WRAPPER" --label scale-expand -- \
  cargo test --locked --offline -p kernel-lab-families --test moe_scale_expand --no-run
```

`scale_expand_pair_exact_bytes_on_gpu` runs the paired entry over gate/up
and down shapes at differing widths, odd shapes whose words straddle rows,
and each bank alone, with the same active lists and guarded outputs.

`cpu_scale_expand_fixtures_cover_codec_and_write_maps` checks 24 fixtures
against the codec and verifies that the poison differs at every expected
write. The GPU tests cover all widths, random and adversarial scales,
exception-heavy rows, every byte code, wrapping addition, partial row and
byte tails, reversed all-active lists, sparse lists, empty lists and null.
They compare complete small output banks, including inactive rows and
guards, after eager launch and poisoned graph replays. The graph's list
changes between replays without changing its grid. A second GPU test
checks raw matrix offsets at 2 GiB and 4 GiB, adjacent inactive matrices,
and the largest legal exception index. It requests 8 GiB of free memory.

These tests are also suitable for external CUDA memory checking: the normal
suite has input/output guards, an unaligned byte output, very short tails
and poisoned unused active-list words. The high-offset case passes an
exact-size encoded allocation. Device correctness and sanitizer results
must be obtained on idle GPUs; CPU compilation alone establishes neither.

Possible performance work, after byte equality is established on a GPU:
vectorize offset decoding, and share exception-range searches across
tiles. No performance claim accompanies the paired entry until its served
measurement.
