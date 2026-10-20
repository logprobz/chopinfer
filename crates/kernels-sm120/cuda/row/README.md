# SM120 row engine

The row engine provides HF, plain, Gemma and per-head RMSNorm, standalone
activation quantization, SwiGLU, embedding gather, routing, stable expert
permutation, weighted combine, RoPE with KV writes, and logits argmax/top-k.
The catalog covers MiMo V2.6 Flash and standard dense and MoE shapes. A single
compiled geometry serves prepared token capacity buckets from 1 through 8192.

Every entry issues `griddepcontrol.launch_dependents` before reading inputs
and `griddepcontrol.wait` before reading predecessor results. Grids are sized
by prepared capacity; live counts are read from device memory. No kernel uses
atomics for order-dependent output. The CPU compile test audits every entry
for native SM120 cubin symbols and zero local memory, and reports registers.

## RMSNorm and quantization

RMSNorm may fuse residual addition and output quantization. The new BF16
residual stream is written before normalization. HF form keeps the oracle's
ordered F32 sum and rounds the normalized activation to BF16 before the
weight multiplication. Fused output quantization reads the completed BF16
norm row in the same CTA. The quantizer has FP8 1x128 exact and power-of-two
scales, MXFP8, MXFP4 and NVFP4 entries. NVFP4 uses row maxima and a global
scale reduction before element encoding. Its fused variant requires the
caller to supply the global scale for the completed norm output.

`ceil_cached.cuh` is a one-pass form of the warp-parallel decode ceil
E4M3/32 quantizer (`ceil_parallel.cuh`, unchanged). The parallel entry
computes each group twice, for the row status and then for its outputs,
in serial loops. The cached entry prefetches every group's operands into
L1, computes each group once with the same `quantize_group`, keeps its
codes and scale code in one register across the row-status barriers and
stores them, so its outputs equal the parallel entry's. It serves widths
up to 8192 (eight groups per warp); wider rows keep the parallel entry.
`block_quantize::Entry::cached` selects it for a separately named entry.

## Exact ordered sum

`ordered_sum.cuh` computes the HF form's ordered F32 sum (from +0, one
round-to-nearest-even add per term in column order) with many threads and
the same bits for every input. While the running sum stays in one binade,
every add is an integer add of the term rounded to that binade's ulp grid,
ties to the even running index; a worker's chunk becomes one increment per
parity of the running index, and chunks of one binade compose exactly in a
segmented warp scan. One walker applies each composed run only when the
running sum is in the run's binade and stays there; every other run, and
every chunk with a binade crossing or a negative, infinite or NaN term, is
added serially with the served adds. The header comment holds the
construction and the proof. `src/row/ordered_sum` emulates the device
algorithm step for step on the CPU; `tests/ordered_sum.rs` compares it
with the serial sum bit for bit on adversarial and random rows, and
kernel-lab's `ordered_sum` family runs the device method beside a serial
thread with its ties-away and no-fallback mutants.

## HF RMSNorm with a tree sum

`rmsnorm_tree.cuh` is the HF form with its sum of squares in a fixed
256-lane F32 tree (`RmsNormNumerics::HF_TREE_256`) instead of the ordered
sum. Lane l adds the rounded squares of columns 8 l + 2048 i + j from +0,
each warp of 32 lanes adds pairwise at distances 16 to 1, and the eight warp
sums at distances 4, 2 and 1. The order depends on the width only, so the
row engine's entry and the fused mixer norm
(`sinkhorn_hc/mix_norm_tree.cuh`) give a row the same bits at every batch,
bucket and live count. The rest of the HF form is unchanged. For
non-negative terms the tree's relative error is about 23 units of rounding
at most, against the ordered sum's 4095. `src/row/rmsnorm_tree` emulates
the order on the CPU, and kernel-lab's `rmsnorm_tree` family checks the
entry against that emulation and an F64 sum.

## RoPE and KV writes

RoPE accepts NeoX and GPT-J pair order, a partial rotary dimension, and either
computed angles or a position-major F32 cosine/sine table. MiMo uses 64
rotary dimensions out of 192, theta 1e7 for global attention and 1e4 for the
sliding window. Target rows read context-sized tables containing BF16-rounded
cos/sin widened to F32. Folded decode shares those target tables, and DFlash
Q/K shares the SWA table, with positions bounded by the served context.
Rotation uses `p = RN32(y*s)` and `RN32(fma(x,c,+/-p))`, without BF16
products, then one BF16 storage conversion.

Computed-angle entries are comparison variants: inverse frequencies match
Torch 2.14 CPU's F32 pow and reciprocal for all 32 pairs of both theta
values, and trig is evaluated in double on the F32-rounded position product,
rounded to F32 and then BF16. This accurate-trig convention differs from CPU
Torch F32 trig after BF16 rounding at 44 SWA and 42 GA table elements over
the 1,048,576-position domain, by at most 0.00390625; it does not claim
literal CPU Torch trig parity. Pure-HF BF16 products are an unserved
comparison whose distance bound is in [numerics](../../../../docs/numerics.md).
kernel-lab `tests/ops/rope_positions.rs` checks the HF reference and records
table bit differences.

QKV is token major, with Q heads, K heads and V heads packed in that order,
3392 elements for MiMo global attention and 3712 for its sliding window per
rank (TP4). Paged caches use separate K and V base pointers, indexed page,
token, KV head, dimension. The runtime u64 arguments `k_page_stride_bytes`
and `v_page_stride_bytes` advance from one physical page to the next. For
separate tightly packed pools, each stride is its own page-sized K or V block
in bytes. A shared rank pool may interleave all layers inside each page: each
layer supplies pointers to its K and V blocks within page zero, and both
strides equal the full physical page size. Page tables are int32
`[batch_capacity][max_pages_per_seq]` with page size 16, 32 or 64.

Ring caches use slot, position modulo ring size, KV head, dimension. Slot IDs
are int32 `[batch]`. Positions may be int32 or int64. All pool-scaled offsets
use 64 bits. BF16 and FP8 E4M3 cache writes are separate traits; FP8 values
are divided by their layer scale and saturated with round-to-nearest-even. V
is multiplied by the layer value before writing. Decode and paged prefill
write K/V before attention. Ring prefill emits contiguous `k_new`/`v_new`,
then `ring_write` commits only the final ring-sized tail after attention. A
dedicated single-sequence BF16 ring-write entry launches only the final 128
token slots; the general ring-write entry serves batched chunks.

## Routing, permutation and combine

Router output is int32 `topk_ids [tokens][top_k]` and F32 `topk_weights` of
the same shape. Sigmoid bias routing selects on sigmoid plus bias, then
normalizes the unbiased sigmoid scores and applies the scaling factor.
Softmax and grouped top-k have explicit variants. Equal scores choose the
lower expert index.

Permute count, scan and scatter produce int32 `expert_offsets [experts+1]`,
`sorted_rows [padded_rows_capacity]` with -1 padding, and `num_padded_rows
[1]`. The sorted routed-row index is `token * top_k + k`; each expert's rows
are in ascending routed-row order. For 2048-8192 token MiMo prefill, tiled
count, prefix, scan and scatter entries use temporary int32 `tile_counts` and
`tile_offsets` arrays, each `[ceil(token_capacity * 8 / 256)][256]` in
tile-major order, and produce the same interface.

Gather optionally materializes BF16 or MXFP8 activation rows. The down
projection writes token-major routed rows, and combine sums top-k
contributions in ascending k order using F32 before BF16 output. MiMo's down
projection applies `topk_weights` before writing BF16 rows, so its combine
entry sums those rows without another weight multiplication. The standard
4096-wide entry keeps the unweighted-input variant.

## Row copies

`copy.cuh` holds the stream broadcast of `kernel_core::row_copy`: each live
row copied into every stream of `[rows, streams, width]`, in 16-byte
vectors, so a row must hold whole vectors. The column slice is the
launch-shape layout copy (`precision_launch.cuh`, `copy_rows`) over 16-bit
words, so a sliced element must hold whole words; it writes the bytes of an
element slice of any such type. Both are launch-shape entries (one entry
serves every width, stream count, stride and offset), grid one 256-thread
CTA per bucket row, leave rows past the device live count untouched and
compute nothing. Each op's contract states the alignment the host binds
its views at; SM120 refuses a broadcast below 16 bytes and a slice below 2.
`row::copy::lab_cubin_specs` builds them under the shared
symbols in kernel-lab's `row_copy` lock family; kernel-lab's
`tests/row_copy.rs` compares them with their oracles and with the model
kernels they would replace.

## Sharded argmax

`argmax.cuh` holds the shared sharded argmax in its contract's order: a
logit ranks by its IEEE order with -0 as +0 and every NaN below -inf, read
from its bits alone, and a candidate's key packs its rank above its inverted
global id, so the largest key is the largest logit with the lowest id. The
partial (F32 logits, or BF16 widened by their bits) grids one 256-thread
CTA per row and writes the winning logit's bits, reread at its column, and
its global id; it reads `max(min(live_width, vocab), 1)` columns. The final
grids one 32-thread CTA per row, reads every rank's pair through a row and a
rank stride (row-major pairs, or rank-major pairs as an all-gather lays
them out) and writes the id to an optional dense output and up to two
scattered outputs (`out[rows[row]]`, indexes outside the output skipped).
Neither names a shape. `row::argmax::lab_cubin_specs` builds them under the
shared symbols in kernel-lab's `row_argmax` lock family; the conformance
kit's argmax cases run them against `oracle::sharded_argmax_partials` and
`sharded_argmax_final`.

## DFlash drafter rows

The MiMo contract catalogs the DFlash entries at TP2 and TP4
(its `contract/src/sm120/row.rs` and `sm120/gemm.rs`; TP4
keeps the untagged names). The BF16 dense GEMM entries are
`mimo_dflash_{name}_m{1,2,4,8,16,32}` and `mimo_dflash_{name}_prefill`
(capacity 8192) for `q`, `qkv`, `context_kv`, `o`, `gate_up` (concatenated
halves SwiGLU epilogue), `down` and `fc`, on the shared BF16 decode and
fragment prefill kernels.

`dflash_q_norm` and `dflash_k_norm` (`dflash_qk_norm` in `dflash.cuh`)
launch one CTA per token and head (16 Q heads or 2 K heads at TP4). The 128
BF16 norm weights are shared by all heads. `input_stride` and
`output_stride` count BF16 elements between token rows, so `input` may point
into a fused QKV or context-KV result with a compact output. `dflash_rope`
takes the same strides and applies NeoX rotation to the first 64 of 128
dimensions with theta 10000, computed or from a cos/sin table
(`_rope_pos32`, `_rope_pos64`, `_rope_table_pos32`, `_rope_table_pos64`).
Neither writes the KV cache; the attention owner handles the context cache,
sink and V scale.

Sharded argmax: each rank runs `dflash_argmax_partial` on its BF16 logits
with its global vocabulary offset. The caller gathers the per-rank
`(score, id)` pairs into rank order as `[rows, ranks]` device arrays (a device
collective or peer copy, never a host read), then runs `dflash_argmax_final`.
They run the shared argmax (`argmax.cuh`): equal scores choose the lowest
global ID, NaN ranks below every value, and a row of only NaN yields its
lowest global ID.

## Qwen3.8 rows

`model_qwen38_contract::sm120::row::hyper_connection_rows()` lists the
hyper-connection, n-gram and PLE rows (TP4's, plus the n-gram rows of TP2,
whose local hash heads differ); their cubin is `row_{tag}` (`row_qwen38`
at TP4), built by `kernels_sm120::row::hyper_connection::cubin_specs` from
the model's catalogs. The row kernels are model-owned CUDA
in the Qwen bundle's `cuda/row/`. Dense down,
up, key, value and inject projections are GEMMs owned by the GEMM catalog. The fused hyper-connection all-reduce is in the
[comm README](../comm/README.md).

Rows use BF16 inputs, weights and outputs, with F32 arithmetic; the exact
parameter order is in each catalog entry. `Entry::plan(capacity, live)`
accepts capacity 1 through the query's capacity (8192 for Qwen3.8) and
refuses live counts above it; the per-sequence PLE decode entry refuses more
than one live row. A device
`live_rows` scalar controls active work during graph replay; a zero live
count writes nothing. There is no workspace, allocation or host-dependent
launch state. Sigmoids come from `common/sigmoid.cuh`: below a logit of -80
a sigmoid is `exp(x)` with the product formed in F64 and rounded once, so
representable results survive; logits of -80 and above use
`1 / (1 + exp(-x))`.

Shapes at TP4:

- HC grouped norm: X and output `[tokens, 4, 2560]`, weight `[4, 2560]`,
  Gemma `(1 + weight)` scaling, epsilon 1e-6.
- HC bottleneck: `[tokens, 320]`, `silu(down / 4)`.
- HC mix: up and normed X `[tokens, 4, 2560]`, output `[tokens, 2560]`.
- HC inject: X `[tokens, 4, 2560]`, y `[tokens, 2560]`, logits `[tokens, 4]`.
- N-gram: one sequence chunk, raw U32 token IDs and two immutable history
  IDs, oldest first. Request-start history is `[248044, 248044]`.
  Multipliers `[3]`, prime sizes `[16]` and global offsets `[16]` are U64.
  Rank r owns heads `[4*r, 4*r+4)` in bigram-then-trigram order, and its
  FP8 E4M3 table covers the contiguous global rows starting at
  `table_start`; `table_rows` bounds this allocation. The per-tensor scale
  is one BF16 value, multiplied after decoding. Output is `[tokens, 640]`,
  exact global row IDs are `[tokens, 4]`, and output history is two U32s.
  Multiplication, XOR, remainder and address arithmetic are unsigned 64-bit.
- PLE gate: normalized key/query `[tokens, 4, 2560]`, value `[tokens, 2560]`,
  output `[tokens, 4, 2560]`. A zero dot product yields gate 0.5.
- PLE conv: input/output `[tokens, 10240]`, weight `[10240, 4]`, history and
  output history `[9, 10240]`, oldest first; kernel 4, dilation 3, SiLU.
  Decode accepts one token per sequence, prefill one sequence chunk. Each
  sequence has its own history; the decode-items entries (`ngram_decode`
  and the PLE conv items entry) carry every decode bucket item in one
  launch with per-item histories.

N-gram and conv input history must not alias output history, and conv input
must not alias output. The caller selects and commits output history after
successful execution, or uses separate candidate buffers for rollback.
Replay from the same input history is deterministic. Conv history does not
reset at EOT; only the n-gram predecessor walk does.

Kernel-lab coverage is `tests/qwen38_rows.rs` (`hyper_connection_g1`,
`ple_g1`, `ngram_g1`): f64 bounds, exact hashes and state copies, graph
replay, PDL, determinism and inactive rows. The ignored `rows_g4`
performance test needs an otherwise idle GPU (see
[kernel-lab](../../../kernel-lab/README.md)).
