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

Launch with grid `(ceil(rows / 64), projections, experts)`, block
`(256, 1, 1)`, no dynamic shared memory, no workspace and PDL disabled.
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

## Registration and bundle use

`kernels_sm120::moe::scale_expand::cubin_spec(name)` returns a standalone
`kernel_build::CubinSpec` with all three entries. Add it once to the bundle's
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

The DeepSeek integration must allocate compressed bank storage and raw
scratch, launch expansion after Route and before the first corresponding
scale consumer, and bind that raw scratch as the existing scale role.
Gate/up uses `rows=576, groups=160, projections=2` at TP4; down uses
`rows=5120, groups=18, projections=1`; both use 384 experts. Order these
launches with normal stream dependencies, including inside captured graphs.

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
vectorize offset decoding and stores, tune row tiles separately for short
and wide rows, and share exception-range searches across tiles. No
performance claim accompanies this initial implementation.
