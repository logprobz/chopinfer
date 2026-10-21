# QSA block-sparse attention

Qwen3.8 Flash's full-attention layers attend through a
learned indexer. A row at position p sees p + 1 tokens: complete blocks of
four tokens from position 0, then an incomplete tail of up to three. The
indexer scores every complete block, keeps the 512 best (ties to the lower
block index), and every query head attends to the kept blocks' tokens and
the tail. A row with at most 512 complete blocks, that is at most 2051
visible tokens, keeps all of them, so the same kernels compute dense
attention there. The contracts and f64 oracles are kernel-core's
`SparseAttentionQuery` and `sparse_attention_oracle.rs`; the engine is
`kernels_sm120::attention::sparse` (`indexer_entries`, `attention_entries`),
and the Qwen3.8 contract's `sm120::attention::sparse_catalog()` supplies
the entries for every compiled geometry and its rank profiles. Symbols are
`chop_qsa_{tag}_{name}_v1`, with `chop_qsa_{tag}_attn_..._merge_v1` for the
split merge.

QSA pads each per-rank GQA group to a multiple of eight, up to 16: the
served groups 12 and 6 occupy 16 and 8 columns. It uses these kernels even
when every block is selected. The separate direct dense prefill kernel
accepts groups 1 through 8 or 16, padding only 3, 5, 6 and 7 to 8. Groups
9 through 15 fail its m16 tile assertion. Direct head-256 FP8 prefill also
refuses groups 2 and 4: CUDA 13.2.86 produces 16 bytes of stack and 32 bytes
of spill stores, above the zero-local allowance. This limit does not apply
to QSA's padded groups. Direct paged decode at head dim 128 also refuses
group 16: its 8,448-byte fold exceeds the 8,192-byte K tile. QSA's head-128
group 16 uses two n8 tiles and fits its own scratch layout.

The indexer admits 4, 8 or 16 heads of dimension 128. Attention heads may
have dimension 128 or 256. Both use rotary width 32 or 64, with three
M-RoPE sections summing to half that width. The served bundle's table
follows the geometry's rotary width and theta, which must remain finite
and positive in F32. Gated norm/RoPE rows support the same head and rotary
dimensions, subject to `ceil(local Q heads / 8) + local KV heads <= 8`.
The served catalog instantiates these choices per geometry and rank count;
Flash's four indexer heads, head dim 256 and rotary 64 keep their entries.
Block size and kept-block count are geometry parameters; the counts below
describe Flash's four-token blocks and 512 kept blocks.

Head-128 GPU evidence covers QSA groups 6 and 16 at 512 kept blocks and
groups 6 and 12 at 256 kept blocks with the eight-head, dimension-128,
rotary-32 indexer, theta 5e6 and sections [6,5,5]. The latter expansion
passed all 1,057 required coverage items on 2026-10-04. Both KV formats
are oracle-bounded at prefill, decode and verify, including the
1027/1028-token selection transition. The indexer matrix includes ties,
zero scores and contexts through 262,144.

Serving admission requires block 4, kept counts 256/512 and M-RoPE
sections [6,5,5] at rotary 32 or [11,11,10] at rotary 64. Its finite
attention table is checked against the lab catalog. The
[kernel library](../../../../docs/kernel-library.md#adding-an-op) lists both
GPU-qualified instances, including the closure expansion reconciled on
2026-10-04 with all 3,067 required W7D coverage items qualified. That evidence
covers the listed kernel instances, not model quality or serving performance.
Lab sparse declarations require four exact verify rows.
Unpinned theta uses Rust F32 `powf`/reciprocal and has no Torch table
bit-parity guarantee.

## Rows and caches

Every entry takes packed step rows: `row_batch[row]` is the row's sequence
(its block-table row and ring slot), `positions[row]` its cache position and
`live_rows` the live count. A sequence's rows are contiguous with
consecutive positions. One layout serves decode (one row per sequence),
verify (up to eight rows per item) and prefill chunks. RoPE positions are
separate, `rope_positions[row]` = (T, H, W), and the F32 cos/sin table is
`[position][pair][cos, sin]`, as in the row engine.

- The KV cache is the attention engine's paged layout: separate K and V
  pools, `[page][page_size][kv_head][head_dim]`, BF16 or FP8 E4M3 with per-layer
  F32 scales, and u64 page strides.
- Block keys are BF16 `[page][page_size / 4][128]` through the same block
  table and their own page stride, so they can share the KV page allocation.
  The Qwen profile uses 64-token pages (4 KiB of block keys per page).
- The raw-key ring holds each slot's last 16 raw keys `[slot][16][128]` and
  their RoPE positions `[slot][16][3]`: an open block plus a verify step of
  up to eight rows, so a rejected verify tail cannot overwrite a kept key.

## Kernels

A step runs five kernels in order, each launched with PDL. Each publishes
its dependents at entry and waits before reading anything, so the chain is
transitive.

1. **Query** (`indexer_query`): one warp per indexer head. Gemma RMSNorm
   `x * rsqrt(mean(x^2) + eps) * (1 + w)` in F32, then NeoX RoPE over the
   first 32 or 64 dimensions with interleaved M-RoPE axes, rounded once to BF16.
   The launch uses 128, 256 or 512 threads for 4, 8 or 16 indexer heads.
2. **Key write** (`indexer_key_write`): one warp per row. A row that
   completes a block whose four tokens are all in this launch pools them from
   the projection rows: the F64 sum in position order (the oracle's sum, so
   cancellation between keys of very different magnitudes drops no term)
   times 1/4, rounded to F32 and then BF16 (the mean's BF16 rounding is what
   both references do), then the norm and RoPE at the block's first
   token, written as one BF16 block-key row. The first row of each sequence
   (its leader) pools a block that began in an earlier launch, reading the
   earlier keys and first RoPE position from the ring, and only then writes
   the ring rows of the sequence's last 16 positions. No other warp touches
   that sequence's ring, so the launch is race free for any chunk length.
   When `anchor_slots[b]` is nonnegative, the ring a sequence ending before
   `anchor_ends[b]` would leave is also written to that slot of the anchor
   ring (`anchor_keys`, `anchor_positions`).
3. **Scores** (`block_scores<Rows>`): `sum_h relu(q_h . k_b) / sqrt(128)`
   for every complete block of every row with more than 512. A CTA takes
   `Rows` consecutive rows (1 for decode; 64, 32 or 16 for prefill at 4, 8
   or 16 indexer heads), or one verify item's rows (a CTA scan ranks the
   items), and a chunk of blocks: one 64-block stage for single rows, so a
   small decode batch fills the SMs,
   and 1024 otherwise. Rows of one sequence share each block-key stage, so a
   prefill CTA reads each block key once for its query rows and a verify
   item reads its keys once. The MMA rows are the (row, head) columns and
   the blocks are n8 columns: BF16 `mma.sync.m16n8k16` over 16 KiB cp.async stages of
   64 blocks (three, or two beside a Q tile holding 256 head rows and for
   single rows, so two single-row CTAs share an SM). The ReLU is
   `x > 0 ? x : +0`, so every score is a nonnegative F32 whose bits order
   like the value. Scores are F32
   `[row][score_stride]`.
4. **Top-k** (`block_topk`): one 512-thread CTA per row. A row with at most
   512 complete blocks keeps them all without reading scores. Otherwise a
   radix select finds the 512th largest score T. A warp visits 512-block
   tiles with sixteen coalesced loads per lane in flight. The first pass
   builds a shared histogram of the top 11 bits and a CTA scan finds T's bin;
   the second writes the kept bitmask of the blocks above that bin, one
   ballot per word, and gathers the bin's blocks into shared memory with one
   reservation per warp tile. Digits of 11 and 10 bits over the gathered
   blocks then find T; a bin of more than 8192 blocks (heavy ties) runs those
   passes over the scores instead. The kept blocks are those above T, then
   the lowest-index blocks equal to T, written in ascending order from the
   bitmasks with two block scans. The selection equals the oracle's top-k
   of these F32 scores bit for bit, including all-equal rows. The numerical contract
   accepts that an F32 near tie can keep a different block than the
   oracle's F64 scores; see "Selection precision" below.
5. **Attention** (`attention` and `merge`): grid [row, KV head, split]. A
   row's key list (its 512 or fewer blocks, then the tail) is cut into units
   of 32 keys (16 for prefill, so two CTAs share an SM); split s owns units
   [units s / S, units (s + 1) / S). A CTA stages its units' K and V rows
   through two cp.async stages, 16 bytes per copy, with each key's page
   looked up once; FP8 stages widen to BF16. The keys are the MMA rows, as
   in the dense decode engine: warp w forms S^T = K Q^T for its 16-key tile
   with grouped heads padded to one or two n8 tiles, transposes P in
   registers with `movmatrix`, and accumulates O^T = V^T P^T for its share
   of the V columns. The probability is carried as two BF16 terms (three
   below 128 keys), so the dropped residual stays within the numerical
   bound. With one split the CTA writes BF16 output; otherwise it writes an
   F32 [max, denominator, numerator] partial (or an empty one), and `merge`
   folds the splits: warp 0 reduces the maxima and denominators with a
   fixed shuffle tree, eight warps sum the numerators of interleaved splits,
   and the warps' sums fold in warp order.

The planner sizes decode and verify splits from the prepared row bucket:
`ceil(2 * 188 / (rows * kv_heads))` CTAs per row, rounded up to a power of
two and capped at 64. Prefill always takes one split and writes directly,
whatever the chunk's rows, so a prompt row's sum and output formula do not
depend on its step. Every partial has a fixed owner and every fold a fixed
order: results are bitwise deterministic across relaunch and graph replay.

**Selection precision.** Keep exact top-k of the F32 scores, ties to the
lower block; equality to F64 scores of the same BF16 operands is not
required. GPU diagnostics on 2026-10-03 found one-block differences on
2 of 24,540 sparse rows of 4K prose and 5 of 73,692 of 8K prose. Those
diagnostics use Torch IEEE F32 matmul, not the serving MMA's bits. The
conservative combined score bound included 69.56% of the 4K rows despite
only two actual differences. Their absolute gaps were 1.256e-6 and
5.321e-7, versus a median boundary gap of 0.003738. Do not use the bound
as a flip-frequency estimate.

Switching only score precision changed mean absolute logprobs by
0.00000845 nats on 4K prose and 0.000271 on 8K, with no top-one changes.
The six scored 8K needle positions were more sensitive: 0.01705 nats,
also with no top-one changes. This warrants broader needle evidence
before a global exact-F64 requirement. Prior F64 threshold rescoring
took 73.8 us versus 20.5 us for F32 top-k at decode 65,536 x 32;
double-single scoring with F64 residual tie resolution took 41.0 us.
Neither met the prior 5 us added-time budget and neither ships. Keep
the score bound, deterministic replay and exact lower-block ties.

**Indexer rounding.** The query and key-write kernels keep norm and RoPE
arithmetic in F32 until one BF16 storage rounding, which leaves every
operand within 0.5 BF16 ulp of its f64 value. Rounding the norm output,
each RoPE product and their sum to BF16 (HF's BF16 sequence) was measured
on matched inputs and rejected: it mis-selects 9% to 23% more blocks than
this sequence against f64 selection at 4K to 128K. Evidence is in
[numerics](../../../../docs/numerics.md).

## Capacities

| Entry | Rows | Notes |
| --- | ---: | --- |
| `index_query`, `index_key_write` | 8192 | grid from the prepared rows |
| `block_scores_decode` / `_verify` / `_prefill` | 256 / 256 / 2048 | `Rows` 1 / 8 by item / `256 / index_heads` (64, 32 or 16) |
| `block_topk` | 2048 | |
| `attn_{bf16,fp8}_prefill` | 8192 | one split, direct output |
| `attn_{bf16,fp8}_decode` / `_verify` | 256 / 256 (32 items of 8) | split merge |

Scores cost `rows * ceil(context / 4) * 4` bytes, 512 MiB for 2048 rows at
262,144 tokens, so a bundle scores and selects prefill chunks in sub-chunks
of at most 2048 rows; attention takes the whole chunk.

## Testing

`kernel-lab/tests/ops/sparse_attention.rs` writes each sequence's history
with chunked key writes (blocks straddle launches), then runs one step and
checks it against the oracles: every block key and the final ring rows, the
queries, scores and attention on up to 64 sampled rows per step, and the
selection of every row bit for bit against the oracle's top-k of the
kernel's own scores. Graph replay, a relaunch and a run behind a PDL
producer must match the eager step bit for bit. Focused tests cover
block-key pooling under cancellation, forced near ties, an all-zero
threshold, and the selector alone on distinct, tied and all-equal scores.
Cases cover contexts 1, 2051, 2052, 8192, 65536 and 262,144, batches 1, 8
and 32, verify items of 1 to 8 rows, mixed contexts, a prefill chunk with
two sequences, FP8 KV and differing M-RoPE axes.
`tests/ops/sparse_invariance.rs` checks the declared invariance.

`qsa_f32_selection_on_oracle_operands` reads an export of `qwen38-oracle
qsa-capture --export` from `CHOPINFER_QSA_OPERANDS` and checks the kernels'
queries, block keys, scores and selections against the oracle's emulation
of the same operands, within the F32 bound. Without an export it is skipped
(a failure under `CHOPINFER_KERNEL_REQUIRE=1`). `CHOPINFER_QSA_BENCH=1`
also times stages and whole-step graph replays; run it on an idle GPU.
