# Dense GEMM engine

Dense GEMMs for SM120: fragment-major decode and prefill kernels for BF16 and
FP8 block weights, a TMA mainloop for the other formats, a medium-M stand-in
(`gemm::mid`), the shared cluster decode template (`gemm::cluster`), and
Qwen3.8's model-owned short-K and skinny kernels (`gemm::fused`).

## Formats and MMA paths

The A16 path uses BF16 `mma.sync.m16n8k16`. BF16 weights load directly; FP8
E4M3 weights are dequantized and rounded to BF16 in registers. The A8 path
uses E4M3 `kind::f8f6f4` `mma.sync.m16n8k32`, with F32 promotion after each
128-K scale block. Both paths accumulate in F32, and no MMA chain spans more
than 128 K: BF16 chains restart from zero every eight k16 steps and are added
into the F32 accumulator, as FP8 blocks are promoted (see
[BF16 accumulation](#bf16-accumulation)). The warp tile and conversion
helpers are in `dense.cuh` and `a8_fp8.cuh`.

- MXFP8 A8 uses native `kind::mxf8f6f4.block_scale.scale_vec::1X` MMA with
  row-major UE8M0 scales, one byte per 32 K values for each operand. MXFP8
  A16 expands each scale into exact BF16 weight operands; preparation rejects
  code/scale pairs that cannot be represented exactly.
- MXFP4 and NVFP4 A16 load low-nibble-even packed E2M1 weights through TMA
  (`U4Align8B`; each 256-value box loads 128 bytes), expand each pair to BF16
  fragments and use HMMA. MXFP4 uses UE8M0 scales per 32 K values; its scaled
  conversion traps on non-finite or inexact operands as well as refusing them
  at admission. This dense MXFP4 path is not used by the served fragment
  expert schedules, which rely on admission alone. The specialization in
  `dense_mxfp4.cuh` is included only by cubins with dense MXFP4 A16 entries
  and the operand probe, preserving other formats' source closures. NVFP4 uses
  UE4M3 scales per 16 K values, keeps `code * local_scale` exact in BF16 and
  applies its F32 global scale (a separate device pointer) after the full
  reduction. A 256-K FP4 staging tile is still promoted every 128 K.
- MXFP4 weights with MXFP8 activations use the native E2M1 by E4M3
  `kind::mxf8f6f4.block_scale.scale_vec::1X` MMA, with the FP4 code expanded
  into bits 5:2 of each byte.
- NVFP4 weights with NVFP4 activations use native
  `kind::mxf4nvf4.block_scale.scale_vec::4X` MMA; the two F32 global scales
  multiply the result before the residual epilogue.

| Weight | Activation | Decode | Prefill |
|---|---|---|---|
| BF16 | BF16 | compiled | compiled |
| FP8 E4M3 tensor, channel, or 128x128 F32 blocks | BF16 | compiled | compiled |
| FP8 E4M3 tensor, channel, or 128x128 F32 blocks | FP8 group128 | compiled | compiled |
| MXFP8 | MXFP8 | compiled | compiled |
| MXFP8 | BF16 | compiled | compiled |
| MXFP4 or NVFP4 | BF16 | compiled | compiled |
| MXFP4 | MXFP8 | compiled | compiled |
| NVFP4 | NVFP4 | compiled | compiled |

Both F32 scale modes are accepted, and the final FP8 block scale row may be
partial. Plain BF16 and F32 output plus residual add are compiled. Weighted
MoE combine belongs to the grouped engine. Dense entries keep BF16
activations: a group128 E4M3 activation variant failed the 0.9999 layer
cosine gate on MiMo-shaped inputs.

`quantized_pipeline` composes a BF16 GEMM result with the row engine's
quantization entries on one stream: FP8 group128 (both F32 scale modes) and
MXFP8 use one follow-on entry; NVFP4 first finds row maxima and a global
scale, then writes packed E2M1 codes and UE4M3 scales. It supports output
width 4096 or 8192, including the fused gate/up output. Direct single-kernel
output quantization and other widths are refused.

## Exact F32 decode

`f32_projection::Entry::with_decode` adds a separately locked variant of
`exact_f32.cuh`. Call `PreparedOp::initialize` once after uploading immutable
weights, binding `scratch.weight_domain` as persistent storage of the size
in `OpInfo::bytes`. Replacing weights requires reinitializing this table.
Each U32 flag covers the original 16-output weight tile, including its tail.

At up to 32 rows, one launch classifies each whole activation row. Four
warps per CTA then compute four independent 128-K chains, using 16-byte
async copies where operand addresses and K permit them. Scalar loads cover
unaligned operands and tails. The component split, eight k16 steps, and
hi/mid/lo merge are the original operations. Each chain writes a separate
F32 plane. A final launch adds those planes from zero in increasing K
order, never a sum of CTA subtotals. The declared split count is
`ceil(ceil(K/128)/4)`; callers cannot supply another count.

The whole row and whole weight tile determine tensor eligibility before
any chain runs. Exceptional values and raw F32 weights keep the original
SIMT FMA path and IEEE propagation. Inactive outputs and caller row-status
words remain untouched. Larger launches route to the original locked entry. At 250 W on SM120
(2026-10-19), cached prefill at 64 rows was 1.15-1.17x slower on small-N
projections; 32 is the largest measured winning split bucket. The original
header, entries, and locks remain unchanged.

## Fragment decode

Decode-bound BF16 and FP8 block weights use the fragment-major layout from
`kernel-core::transform::fragment`: each 16-row strip has contiguous 512-byte
MMA A tiles, 16 bytes per lane. Warps load fragments directly from global
memory with `ld.global.nc` and stage the small BF16 activation slab in shared
memory. Each 128-K block accumulates into a temporary, which BF16 adds into
the F32 accumulator and FP8 promotes by one F32 scale.

QKV, output projection and dense MLP decode (`fragment_decode_cluster.cuh`)
give each CTA 2, 4 or 8 weight strips (M <= 4, 8 to 16, and 32 tokens) over
one K chunk. Each warp streams 16 fragment tiles into registers, starting
before the PDL wait; after the wait one thread bulk-copies the chunk's
activation rows into padded shared rows read with conflict-free `ldmatrix`.
The CTAs covering one row tile's K form a thread-block cluster of at most
eight; each writes its F32 partial tile to its own workspace slice, and after
one cluster barrier every rank sums its output rows over all ranks and K
parts in K order. There is no semaphore, and the sum order is fixed.

The router uses 16-row strips with four K splits of eight warps (64 CTAs);
the last split CTA of each strip sums the split partials in fixed order. The
fused router and top-8 entry takes one grid-wide ticket: its last CTA sums
all splits into shared logits and selects top-8 there (ties to the lower
expert ID), matching `chop::moe::route_decode_mimo_body` bit for bit.

Each tiled and fragment decode entry declares its K splits
(`DenseKernel::Tiled { splits }`, `FragmentDecode { splits }`), and the
planner refuses a launch with any other count; the compiled split capacity
only sizes the grid and workspace. Split-K writes F32 partials to
caller-owned workspace; the last CTA for each tile sums them in split order,
writes the result and resets its semaphore. Semaphores must be zero before
the first launch.

Chain-split tiled decode (`DenseKernel::TiledChains { splits }`,
`tiled_chains.cuh`, BF16 and FP8 128x128 weights with BF16 activations, at
most 32 rows, no SwiGLU) spreads a decode tile's K chunks over `splits`
CTAs without changing its arithmetic. Each CTA runs the one-split kernel's
pipeline, warp tiles and MMA calls over its chunks and stores every MMA
chain's temporary (64 K for BF16's 64-K chunks, 128 K for FP8 blocks; no
chain is resegmented) to the workspace, one slot per chunk whatever the
split count. The tile's last CTA (an integer ticket) adds the temporaries
from zero in K order with `run_a16`'s operation (`acc + chain`, or one FMA
with the FP8 block scale), then applies its epilogue, so every output is
bitwise the `Tiled { splits: 1 }` entry's at any split count and live M.
`gemm::chain_splits` declares enough splits for the tiles to fill the SMs,
or one split (a `Tiled` entry) where that is under four: kernel-lab
`projection_latency` measured 4 or more splits up to 5.6 times faster than
one split and 3 splits up to 2 times slower. kernel-lab
`tiled_chain_splits` compares served chain-split entries with the
one-split entries they replace, behind a PDL producer that holds their
activation poisoned through its tail and in back-to-back launches sharing
one workspace and ticket array; lab-only `ChainsMutation` instances (no
or a late dependency wait, shared temporaries, a kept ticket) must fail
the same checks.

Above 32 rows a `TiledChains { splits: 1 }` entry runs the one-split
prefill tiles in warp tiles (`tiled_warps.cuh`, `run_a16_warps`, BF16 or
FP8 128x128 weights, 64 x 64, 64 x 128 or 128 x 128 tiles). Each of the
eight math warps owns a 32-row block of the tile's tokens and a quarter or
half of its columns for the whole K loop, so every chunk is loaded once
(`run_a16` streams K once per wave of 16 x 8 subtiles), and BF16 fragments
come from `ldmatrix` at the addresses `packed_bf16` reads. Its chains,
their adds and the epilogue are `run_a16`'s, so outputs are bitwise the
one-split entry's. The grid has one CTA per tile at the capacity (CTAs past
the live tiles exit), and the producer warp and the math warps run separate
loops over the tile's chunks, so no code runs on both sides of the
`setmaxnreg` split. A persistent version that carried one pipeline across
a CTA's tiles differed from the one-split entry once the tiles outnumbered
the CTAs (GPU run 2026-10-07, n1536 at 1,023 rows). Kernel-lab `tiled_chain_splits`
pairs it with the one-split prefill entry and runs the chain entries'
producer and back-to-back checks on it; its lab-only mutants take
`ChainsMutation::drop_wait` and `late_wait`.

Over FP8 128x128 weights, a `TiledChains { splits: 1 }` prefill entry at
`gemm::decoded_tile_knobs` (128 x 128 tiles, `tile_k` 64; `n` and K whole
128 blocks) runs decoded weight tiles instead (`decoded_tiles.cuh`,
`run_a16_decoded`). Warp tiles decode every B fragment from the raw codes
in each math warp that reads it; here warp 0 loads 64-K stages of
activations (three, 128-byte swizzle) and raw codes (two, unswizzled) by
TMA, warps 1-3 decode each code stage once into a swizzled BF16 stage (two),
and the eight math warps read both operands with `ldmatrix`, each over 32
tokens by 64 weight rows. Every hand-off is an mbarrier. The operands
(`cvt.rn.bf16x2.e4m3x2` of the codes), the MMA chain per 128-K block over
two stages, the scale FMA, the residual add and the conversion are the warp
tiles', so outputs are bitwise theirs and the one-split entry's. Kernel-lab
`tiled_chain_splits` sweeps it (`decoded_tile_shapes.rs`) and its lab-only
`DecodedMutation` instances (no or a late wait, 256-K chains);
`projection_latency` `decoded_tile_latency` measured it 1.4-2.8 times faster
than warp tiles from 64 to 8,192 rows (PERF-B18 probe, 2026-10-09).

Strip-chain decode (`DenseKernel::StripChains { strips, splits }`,
`strip_chains.cuh`, `run_a16_strips`, BF16 weights and activations, at most
32 rows, no SwiGLU) keeps the same arithmetic in a schedule for small row
counts. A CTA holds 1, 2 or 4 16-row weight strips, every token row of the
bucket and one of `splits` K partitions; one math warp per strip reads its
fragments with `ldmatrix` at the addresses `packed_bf16` reads and runs each
64-K chain from zero, and a producer warp streams the chunks through a TMA
ring of as many stages as fit 48 KB. The weight loads of the first stages
are issued before the grid dependency wait (no earlier kernel writes the
weights); activation loads and every workspace, ticket and output access
come after it. At one split a warp adds its chains into a zero accumulator
in K order in registers, as `run_a16` does; over more splits the chains go
to one workspace slot per tile and chunk and the tile's last CTA adds them
as `run_a16_chains` does. Outputs are bitwise the `Tiled { splits: 1 }`
entry's at any strip count, split count and live M. The `tiled_chains.cuh`
consumer packs every fragment register from two 16-bit shared loads and a
swizzle computation per K16 step and runs one warp per 16 x 8 subtile over
all of a CTA's chunks; kernel-lab measured about 1.2 us per 64-K chunk per
CTA at any CTA count (n4096 and the head), so its decode time follows the
chunks per CTA rather than DRAM. Kernel-lab's `strip_chains` family sweeps
strip and split counts beside each shape's entry of today
(`strip_chain_shapes.rs`; `tiled_chain_splits` checks bits, PDL, shared
workspace and the `StripsMutation` mutants, `projection_latency` times
them).

Over FP8 128x128 block weights the same variant runs `run_a16_strips_fp8`
(`strip_chains_fp8.cuh`): 128-K chunks, one scale block each, so a stage
holds 128 swizzled bytes per weight row and 256 unswizzled bytes per
activation row (the tensor maps of the `Tiled` FP8 entry). Each weight
register is the two E4M3 codes `packed_fp8_weight` reads, as one 16-bit
shared load at the swizzled address, converted to BF16 exactly; activation
registers come from `ldmatrix`. Each 128-K chain runs from zero and is added
with `run_a16`'s FMA by the strip's block scale, `acc = fma(chain, scale,
acc)`, in registers at one split or by the tile's last CTA over the chain
slots above one, so outputs are bitwise the FP8 `Tiled { splits: 1 }`
entry's. Kernel-lab's `strip_chains_fp8` family sweeps the served FP8 decode
shapes (`strip_chain_fp8_shapes.rs`) with the same checks, timings and
`Fp8StripsMutation` mutants.

Served strip-chain and warp-tile entries run the `_ordered` variants of
these kernels (`strip_chains_ordered.cuh`, `strip_chains_fp8_ordered.cuh`,
`tiled_warps_ordered.cuh`). The originals release a ring stage (an
mbarrier arrive) right after issuing the step's shared reads of it, with no
fence between those generic-proxy reads and the producer's next TMA write
of the stage, which is in the async proxy; ptxas issues the arrive before
the last `ldmatrix` reads return, so the producer could refill the stage
under a warp still reading it. The vocabulary head's two-split strip chains
(1210 CTAs) stored such corrupted chains in most launches. In the
`_ordered` kernels every math lane runs `fence.proxy.async.shared::cta`
after its last read of a stage and before the warp's release (the PTX
memory model's cross-proxy requirement); nothing else changes, so outputs
keep their bits. Kernel-lab's `strip_chain_race` family runs each served
multi-split schedule, the head's two-split one and two prefill warp tiles
on both kernels beside a contention probe (`tests/strip_chain_race.rs`).

Stacked strip chains (`gemm::stacked`, `strip_chains_stacked.cuh`,
`run_a16_strips_stacked`) run up to six BF16 strip-chain GEMMs that read the
same activation rows in one launch. Each part keeps its own tensor map,
tiles and `run_a16_strips` arithmetic; the parts' tiles are numbered in part
order, a CTA takes one tile and K split, and chain slots and tickets are per
stacked tile, so each part's outputs are bitwise its own strip entry's (and
the one-split `Tiled` entry's) at any strip count, split count, part order
and live M. A part writes its first `width` columns at `row_stride`
elements per token row: several parts fill thirds of one wider buffer, or a
part whose weight rows a format pads writes its rank's columns compacted,
in place of a contiguous output and a layout copy. With `prefetch`, the
producer also issues `cp.async.bulk.prefetch.tensor` L2 requests for every
chunk of its K range the ring does not stage, before its grid dependency
wait, so a launch that starts early behind PDL streams its weights while
the kernels before it run. `gemm::stacked` emits the ordered variant
(`strip_chains_stacked_ordered.cuh`, `run_a16_strips_stacked_ordered`),
which fences each math lane's stage reads before the warp's release as the
`_ordered` strip chains do; `run_a16_strips_stacked` keeps the unfenced
release and is not emitted.

The weight prefetch (`gemm::prefetch`, `weight_prefetch.cuh`) is a separate
PDL launch that releases its dependents, requests up to eight byte ranges
from L2 (bulk `cp.async.bulk.prefetch.L2` pieces or one `prefetch.global.L2`
per line), then waits for the kernel it depends on, so its completion still
implies its predecessor's and the chain's order and outputs are unchanged.
It writes nothing. Kernel-lab's `strip_chains_stacked` family
(`strip_chain_stacked_shapes.rs`, `tests/strip_chains_stacked.rs`) checks
the stacked candidates against each part's one-split entry placed, the
`StackedMutation` mutants and the prefetch's chain order, and times both
under a cold L2 behind idle windows that stand in for the collective and
mixer before a layer's projections.

## Fragment prefill

BF16 and FP8 block prefill consume the same packed weights
(`fragment_prefill.cuh`, with reusable pieces in `prefill_mainloop.cuh`). A
persistent CTA owns 128 tokens by eight 16-row weight strips. A producer
warpgroup fills a ring of 64-K stages (as many as fit 96 KiB: three for BF16
and four for FP8), each holding a 128-byte swizzled activation TMA tile and
one 1-D bulk copy per weight strip. Two math warpgroups each own 64 weight
rows by 32 tokens per warp, read activation fragments with `ldmatrix.x4` and
weight fragments with `LDS.128`. SwiGLU CTAs stage four gate strips and the
four matching up strips, so each warp holds gate and up accumulators for the
same outputs. The first stages' weight copies issue before the PDL wait.

Every tile's K is cut into the same S segments of whole 128-K blocks at every
live M. S is the entry's fixed count (1, 2, 4 or 8), declared as
`DenseKernel::FragmentPrefill { segments }` and read by
`gemm::prefill_segments`; an entry without a declared kernel is refused.
Segment s holds blocks `[k_blocks * s / S, k_blocks * (s + 1) / S)`. An
output element is the sum from zero, in segment order, of its segments' sums,
each the sum from zero in K order of its blocks' temporaries. The counts were
chosen with kernel-lab `gemm_prefill_chunk_bench` (the fastest of 2, 4 and 8
over 512- to 2,048-row chunks at TP4, or 4 when within 1% of it).

Whole tiles are dealt round robin in full waves; if the remaining tiles fill
less than 90% of a wave, their segments are divided evenly over the CTAs
(stream-K over segments). A CTA keeps at most eight partial tiles
(`kMaxSegments`). Every CTA that contributed to a split tile counts its
arrival, waits for the tile's other CTAs, then sums a disjoint set of the
tile's strips in the same order as a whole tile, so a split element and a
whole one have the same bits. Stream-K CTAs are resident (grid within the SM
count), so the wait cannot deadlock, but another process holding SMs on a
shared GPU delays it. The schedule depends on the grid and live M; the output
bits do not: a prompt row gets the same bits in any chunk, at any row of it,
on any grid ([numerics](../../../../docs/numerics.md)), and eager runs, graph
replay and PDL launches are bitwise identical. Fixed segments cost 3% to 11%
per entry against a free stream-K tail at 2,048 rows. The generated
invariance tests (`kernel-lab/tests/ops/gemm_invariance.rs`) check this.

The router prefill entry runs this kernel at every live M. The lab-only
`_clock_v1` cubin records per-CTA `clock64` cycles for barrier waits, math,
epilogue and prologue, stored after the first 8 KiB of the semaphore buffer.

### Medium M (`gemm::mid`)

Verify buckets above the decode batch and decode above 32 sequences run the
fragment prefill entries at a few dozen to a few hundred live rows. For 33 to
64 live rows, `gemm::mid` stands in for them with the same kernel at a
64-token tile, `run_fragment_prefill<W, Swiglu, 64>`, passed its reference
entry's declared segments (`mid::Entry::segments`). Every output element sums
the same segments in the same order with the same epilogue rounding, so the
outputs are bitwise equal to the prefill entry's at the same live M
(`gemm_mid_matches_prefill`). The 33 to 64 range is where the 64-token
kernel was faster than the prefill entry for every shape in the kernel-lab
48-layer graph bench (`gemm_mid_bench`, `kernel-lab/tests/ops/gemm_mid.rs`).

N, K, live M and F32 output are runtime arguments, so six kernels (BF16
weights or FP8 weights with scales per 128 or 64 rows, each with and without
SwiGLU; `chop_gemm_mid_{bf16,fp8b,fp8r64}{,_swiglu}_t64_v1`) serve every
model. `gemm::mid::catalog(catalogs)` holds one entry per fragment prefill entry of
MiMo and Qwen3.8 (all rank counts), each naming its `reference` prefill
symbol and the live rows it is selected for. MiMo's router prefill has no
entry. `mid::cubin_spec_for` builds one cubin with the kernels a set of
profiles selects (MiMo's `gemm_mid_mimo_sm120` serves TP2 and TP4),
and `mid::cubin_specs` one per profile.

Bundles keep their prefill selection and ask the catalog for a stand-in:

```rust
if let Some(mid) = kernels_sm120::gemm::mid::select(catalogs, &prefill_entry.symbol, live_m) {
    // The prefill entry's activation map with mid.activation_box_rows()
    // rows in its box, and the prefill entry's buffers.
    let plan = mid.plan(live_m)?;
    let args = mid.arguments_from(live_m, activation_map, weight_buffer, gemm_buffers)?;
    // launch mid.kernel.symbol() from mid::cubin_name(Some(profile)),
    // configured for mid::dynamic_shared_bytes(symbol)
}
```

The launch uses the prefill entry's grid, the same zeroed semaphore words and
at most its workspace. MiMo launches the stand-ins at 33 to 64 live rows of
its decode and verify buckets through the backend's projection op in its
medium-M form (`DenseForm::Medium`, `bundle-mimo26`'s `sm120/projection.rs`),
and through its own `sm120/dense.rs` builder for the buckets wider than 64
rows; `CHOPINFER_MIMO_VERIFY_DENSE_MID=0` restores the prefill entries and
skips loading the cubin. The Qwen3.8 bundle does not launch the
stand-ins.

## Other formats and SwiGLU

Other formats use a TMA mainloop: one producer warp, two four-warp math
groups, a multi-stage mbarrier pipeline and `setmaxnreg`. Dynamic shared
memory is aligned to 128 bytes before swizzled TMA loads; tiles wider than
the 128-byte swizzle span use an unswizzled descriptor. Fragment decode calls
`griddepcontrol.launch_dependents`, issues its first direct weight loads,
then calls `griddepcontrol.wait` before reading activations; the packed
prefill kernel also uses PDL before its TMA and bulk transfers.

The A16 SwiGLU entry owns paired gate and up tiles in one CTA and writes
SiLU(gate) times up. Concatenated halves use adjacent row tiles (or strips)
from each half. Interleaved columns require the weight tensor map's row
element stride to be 2 and a row box twice the output tile height. Other
SwiGLU entries add residuals to raw columns when requested.

## BF16 accumulation

A BF16 `mma.sync.m16n8k16` adds its 16 exact products and C in one pass
aligned to the largest exponent, keeps 25 fractional bits below it, and
truncates the sum to F32, so along a long chain the error grows and is biased
toward zero. Kernel-lab measures this with the probe
`probe_mma_accumulation_sm120` (`tests/mma_accumulation.rs` sweeps chain
lengths; `gemm::accumulation` in the ops tests and `qwen38_gemm_accumulation`
run the served entries against Neumaier-compensated f64 sums). On checkpoint
weights with dumped activations, one chain over all of K was 8 to 44 times
worse than 128-K runs at p99.99, and served entries such as the MiMo heads
were 20 to 24 times worse, so every BF16 dense kernel uses 128-K temporaries:
at most eight chained MMAs from zero per F32 temporary, added into the
accumulator in K order. No scheme exceeded the contract term `2^-22 K S`.

### Decision, 2026-10-04: keep persistent BF16 chains

Attention PV (prefill,
decode, verify, selected and window), encoder pair GEMMs, encoder pair PV
and MXFP4 expert mainloops keep their current persistent BF16 chains.

The measured errors sit below the precision their outputs are stored at.
On the real captures the current chains err by at most about 4e-5 of the
summed absolute products, against a BF16 half step of about 2e-3, so their only
visible effect is that 0.05 to 0.9% of BF16 outputs land one step from
where an exact sum would round. The large ulp figures come from outputs
near cancellation. The served teacher, paired, encoder parity and media
gates pass with these chains, and an effect this size is below what they
resolve, so no measurable served improvement is expected.

The cost is not free. A compile-only check of the eight-call pending
temporaries raised paged prefill, decode and selected attention entries to
the 255-register limit (no spills), and K64 temporaries in the MXFP4
gate/up prefill mainloop spilled. Every model on these shared templates
would pay that register pressure and any slowdown.

This matches common practice: cuBLAS, CUTLASS and FlashAttention keep BF16
accumulation in the tensor core's F32 accumulator for the whole reduction
and promote only FP8, whose tensor-core accumulation keeps far fewer bits
and whose block scale needs an F32 step anyway. Promotion stays where it is
here already: the dense BF16 kernels' 128-K temporaries and the FP8 block
and post-scale temporaries.

Revisit a family if a model's served gates fail on it, or if a model's
admitted context or chain length pushes the measured toward-zero bias near
BF16 resolution (the synthetic 262144-key PV control reached a bound
fraction of 0.0265, about that level).

## Qwen3.8 Flash at TP4

The bundle keeps every dense Qwen3.8 weight in BF16; only the routed experts
(grouped MoE engine, `models/qwen38/bundle/cuda/moe/qwen.cuh`) and the n-gram table (a row gather)
are FP8. No dense GEMM here reads scales or FP8 activations. Every weight is
a per-rank fragment-major view (`bf16_mma_fragments_16x16_v1`, rows padded
to 16); activations and outputs are row-major BF16 (F32 for the router and
the scored head). Per rank, per token pass: linear attention runs in 36
layers, full attention in 12, each of the 48 layers has two hyper-connection
sublayers plus the final mixer (97 blocks), and PLE runs at layer 1.

| GEMM | Calls | N x K | Decode entry | Prefill entry |
|---|---:|---|---|---|
| linear `in_proj_qkv` (q 512, k 512, v 1536) | 36 | 2560 x 2560 | `lin_in_c{M}` | `lin_qkv_prefill` |
| linear `in_proj_z` | 36 | 1536 x 2560 | `lin_in_c{M}` | `lin_z_prefill` |
| linear `in_proj_a` with `in_proj_b` | 36 | 2 x 12 x 2560 | `lin_in_c{M}` | `lin_ab` (skinny) |
| linear `out_proj`, K-sharded | 36 | 2560 x 1536 | `out_c{M}` | `out_prefill` |
| attention `q_proj` (6 heads of 256 query and 256 gate, interleaved) | 12 | 3072 x 2560 | `attn_in_c{M}` | `attn_qkv_prefill` |
| attention `k_proj`, `v_proj` (KV head r / 2) | 24 | 256 x 2560 | `attn_in_c{M}` | `attn_qkv_prefill` |
| indexer `index_qk_proj`, replicated | 12 | 640 x 2560 | `attn_in_c{M}` | `index_qk_prefill` |
| attention `o_proj`, K-sharded | 12 | 2560 x 1536 | `out_c{M}` | `out_prefill` |
| router `mlp.gate`, replicated, F32 logits | 48 | 512 x 2560 | `router_c{M}` | `router_prefill` |
| hyper-connection `input_mix_weight_down` | 97 | 320 x 10240 | `hc_in_c{M}`; `hc_down_c{M}` (mixer) | `hc_down_prefill` |
| hyper-connection `block_inject_weight` | 96 | 4 x 10240 | `hc_in_c{M}` | `hc_inject` (skinny) |
| hyper-connection `input_mix_weight_up` | 97 | 10240 x 320 | `hc_up_decode` (short K) | `hc_up_prefill` (short K) |
| PLE `key_proj` | 1 | 10240 x 2560 | `ple_key_c{M}` | `ple_key_prefill` |
| PLE `value_proj` | 1 | 2560 x 2560 | `lin_qkv_c{M}` | `lin_qkv_prefill` |
| `lm_head`, vocabulary shard of 248,320 | 1 | 62080 x 2560 | `head_f32_m{M}`, `head_m{M}` (drafter) | `head_prefill`, `head_f32_prefill` |
| shared expert gate/up, zero-padded 160 to 256 | 48 | 512 x 2560, SwiGLU | `shared_gate_up_c{M}` | `shared_gate_up_prefill` |
| shared expert down, zero-padded | 48 | 2560 x 256 | `shared_down_c{M}` | `shared_down_prefill` |
| MTP `fc_embedding`, `fc_hidden` (per stream) | 1 each | 2560 x 2560 | `lin_qkv_c{M}` | `lin_qkv_prefill` |

Entry names drop the `chop_gemm_qwen38_` prefix and `_v1` suffix. The
placement stores each full layer's q_proj, k_proj and v_proj back to back,
so `attn_qkv_prefill` (3584 x 2560) reads them as one fragment view and
writes the packed `[M, 3584]` row (q and gate, then K, then V). The fused
decode entries `lin_in`, `attn_in` and `hc_in` run the weights that read one
activation in one launch, each into its own output with its own row stride.
Decode buckets are M = 1, 2, 4, 8, 16 and 32; prefill entries take any live
M up to the 8192-row chunk. Every target logits row is F32 (`head_f32_m{M}`,
`head_f32_prefill`); the BF16 `head_m{M}` serves only the MTP drafter.
`shared_expert_gate` is a dot product inside the MoE combine, and the vision
tower's GEMMs belong to the multimodal engine.

Entry families:

- Fragment decode and prefill serve every shape whose output is a multiple
  of 64 and whose K is a multiple of 128: all `_prefill` entries and the head
  decode buckets, in the `gemm_qwen38_sm120` cubin.
- Cluster decode (`_c{M}`, M <= 32) runs the shared template below, with
  each entry's blocks per warp declared (`qwen38::WarpBlocks`) and the shape
  chosen by `cluster_decode_shape` (K = 2560 splits five ways, 1536 three or
  six, 10240 five with one strip). A fused entry numbers the strips of its
  weights consecutively, and tokens are staged in two halves when a slab
  would overflow shared memory. The workspace is one F32 partial tile per
  CTA; there is no semaphore.
- Short K: `input_mix_weight_up` has K = 320, which both fragment families
  refuse. `hc_up_decode` and `hc_up_prefill` keep each strip's 20 fragment
  tiles in registers and add every 128-K run (the last is 64 K) into F32. No
  workspace.
- Skinny: `in_proj_a`, `in_proj_b` and `block_inject_weight` have 12 and 4
  rows, below the 64-row output tile. `lin_ab` and `hc_inject` fuse a and b
  into one `[M, stride]` output (a in columns 0 to 11, b in 12 to 23). A CTA
  owns 32 tokens and a K split of whole 256-K chunks. The split count is
  fixed per phase at every live M, so a row's sum does not depend on how many
  rows share its step: graph phases take one 256-K chunk per split (10 for
  `lin_ab`, 40 for `hc_inject`), eager prefill the entry's own count
  (`Entry::prefill_plan`, `prefill_arguments`: 5 and 10). The last CTA of a
  token tile sums the split partials in split order and resets its
  semaphore; `Entry::semaphore_words` gives the zeroed words and
  `workspace_capacity` covers both phases.

The last three families are `gemm::fused` entries (Qwen3.8's
`sm120::gemm::fused_catalog()`) in the `gemm_qwen38_own_sm120` cubin, with
their own planner (`Entry::plan`, `Entry::arguments`, `Buffers`); the short-K
and skinny kernels are in `models/qwen38/bundle/cuda/gemm/qwen38.cuh`, which only
the Qwen cubins include.
All are PDL launches that issue their first weight loads before
`griddepcontrol.wait`. The row engine adds `embedding_qwen38` and
`logits_{argmax,top20}_qwen38_{full,shard}` under the Qwen contract's
`sm120::config::PROFILE`.

Kernel-lab correctness: `ops` tests `gemm_qwen38_shapes`,
`gemm_qwen38_replay`, `gemm_correctness_qwen38_flash`,
`row_router_permute_combine_gpu`, `row_logits_gpu` and
`row_qwen38_full_vocab_embedding_gpu`; the `qwen38_gemm` tests
`qwen38_gemm_g1` (every `gemm::fused::catalog` entry at full and partial
buckets with repeat, graph replay and PDL checked bitwise) and
`qwen38_skinny_prefill_rows_ignore_chunking`. All compare against the
kernel-core f64 GEMM oracle with the numerical bound.
`CHOPINFER_KERNEL_BENCH_PROFILE=qwen38-flash` selects Qwen in
`gemm_bench_profiles`; `qwen38_gemm_g4` is the dedicated bench.

## Shared cluster decode template (`cluster_decode.cuh`, `gemm::cluster`)

MiMo's cluster decode (`run_fragment_decode_cluster`) and Qwen3.8's compute
every output the same way: each warp chains at most eight MMAs from zero per
128-K block, adds (BF16) or scales and adds (FP8) each block into its F32
partial, holding `warp_blocks` consecutive blocks, and the ranks sum the
partials from zero over cluster ranks, then K parts, in K order before the
SwiGLU and the store. They differ only in how work is distributed.
`chop::gemm::cluster::run` is their union, with the epilogue as a template
parameter (`Store` by default; Qwen's `HcBottleneckStore` in
`models/qwen38/bundle/cuda/gemm/qwen38_hc_bottleneck.cuh`). The shape (`gemm::cluster::Shape`) decides the
arithmetic only through the format, K, `warp_blocks` and SwiGLU; strips,
splits, token tile and passes distribute the work.

Qwen3.8 serves the template (`gemm::cluster::Served::runs_template`) for its
`_c{M}` buckets and HC precompute producers. MiMo still serves
`run_fragment_decode_cluster`. `gemm::cluster::lab_instances()` pairs every
served cluster decode entry with its reference: Qwen's with the frozen
pre-template kernel in
`kernel-lab/cuda/baselines/qwen_cluster_decode_97d9ac95e.cuh` (built as the
`chop_gemm_eng_` references, `gemm_references.rs`), and MiMo's local decode
entries with `chop_gemm_tpl_` lab instances at the served shape, ABI,
threads and plan. `kernel-lab/tests/dense_template.rs` compares every pair
bit for bit, and `dense_template_timing` alternates them A B A B.

Qwen's model-owned dense kernels outside the template are the short-K
`hc_up_*` entries, the skinny `lin_ab` and `hc_inject` (whose chunk-strided
chains cannot match a contiguous 128-K template), and the fused HC up mix
(`hc_up_mix_decode`, `hc_up_mix_precomputed`).
