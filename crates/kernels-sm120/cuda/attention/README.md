# SM120 attention

Paged, ring (sliding-window), verify, exact-row and gated attention for
SM120. The planner is `kernels_sm120::attention` (`Variant`); block-sparse
attention with a learned indexer is in [SPARSE.md](SPARSE.md).

## Caches

The paged cache has separate K and V base pointers; within one layer a
physical page holds `[token][kv_head][dim]`, and
`block_tables[batch][logical_page]` gives a signed 32-bit physical page ID.
The runtime u64 `k_page_stride_bytes` and `v_page_stride_bytes` advance the
base pointers between pages: one layer's page block for tight pools, or the
whole multi-layer page (with per-layer offset base pointers) for a shared
pool. Page-scaled offsets use u64 math; the planner validates each stride's
minimum size, 16-byte alignment and capacity arithmetic. The ring cache is
`[slot][absolute_position % ring][kv_head][dim]`, selected by `slot_ids`.
K and V dimensions may differ. FP8 E4M3 elements are promoted to BF16 for
MMA, and the per-layer F32 K and V scales apply to the score and the output
numerator.

## Masks, sinks and softmax

Causal and optional left-window masks use absolute positions.
`window_left = 127` includes the current token and 127 earlier tokens.
Softcap applies `cap * tanh(scaled_score / cap)` before the softmax. A sink
is one extra logit per Q head, with zero V contribution. Softmax
exponentials use base two after multiplying logit differences by `log2(e)`.

The probability of each key is carried as one to three BF16 fragments (the
rounded value and its rounded residuals). A row uses three when its own
visible key count K is below 128, two below 32768, and one otherwise; a warp
whose rows differ runs the largest count, with each row's extra terms zero,
so a row's count never depends on the other rows of its tile. The dropped
residual is at most `2^(-8 terms)` of each `|p v|` product, so at those
thresholds it stays within half of the numerical term `2^-22 K S`.

## Decode

Decode runs after the row engine writes the current token: BF16
`q[batch][q_head][qk_dim]` is rotated and `seq_lens` includes that token.
Before `griddepcontrol.wait`, decode kernels read the step inputs (live
batch, `seq_lens`, block tables and slots, which the caller uploads with
stream-ordered copies before the step), plan, and copy the K/V rows before
each row's current token; Q and the current rows are copied after the wait.

Paged decode (`decode`) is a stream-K split-KV kernel whose split is row
local. A row's visible keys form units of 32 positions (16 for 256-wide
heads) aligned to absolute positions. A row of u units is cut into pieces
of `max(split_units, ceil(u / min(split slots, CTAs)))` units from its first
visible unit (`piece_units`), so it has at most the bucket's split slots of
partials (512 for a batch of one, down to 64 for batches of 8 to 32, 32 for
64 and 128 rows and 16 for 256) and its piece bounds depend on its own
length and the bucket only. `split_units`, the bucket's smallest piece, is
1 up to 8 rows, 2 at 16 and 32, 4 at 64 and 128 and 8 at 256
(`tuning/attention.json`). Each CTA scans every live row's unit count from
`seq_lens` into shared prefix sums, takes an even share of the step's units
over the persistent CTAs per KV head (two per SM), finds its first row by
binary search and computes every piece that starts in its share, each to
its end and from a fresh softmax state. Which CTA computes a piece changes
no arithmetic, so a row's output bits do not depend on the number,
lengths, order or values of the other rows of its step
(`attention_decode_rows_ignore_their_companions`). The tables hold 256 rows
(`max_decode_rows`); a larger live count traps, and the merge traps when the
live rows exceed its grid. Verify (`verify.cuh`) uses the same plan per
item.

A CTA streams its units through two 16-byte `cp.async` stages (FP8 widened
to BF16 in shared memory). The keys are the MMA rows: warp w forms
S^T = K Q^T for a 16-key tile with the grouped Q heads as n8 columns,
transposes P with `movmatrix`, and accumulates O^T = V^T P^T. Each piece
writes one F32 `[max, denominator, V numerator]` partial per row and grouped
Q head; the merge counts the row's pieces from its own `seq_lens` entry,
folds them in fixed order, adds the optional sink logit and writes BF16
output.

Windowed ring decode (`decode_window`) gives each row slot and KV head one
CTA of eight warps. Before the wait it copies the row's whole window (up to
128 positions) into dynamic shared memory (81,920 bytes for MiMo SWA) and
loads the sink logits; the warps fold their partials with the sink into
BF16 output, so the planner reports direct output and the ring merge entry
only waits. It requires eight grouped Q heads per KV head and at most 256
rows.

### Folded RoPE and KV write

MiMo decode variants have `_rope_table_v1` entries that fold the row
engine's RoPE and KV write in. Such an entry reads the packed QKV rows
`[row][q_heads * qk | kv_heads * qk | kv_heads * v]` in place of `q`, plus
theta, the V multiplier and a read-only F32 cos/sin table (the same table
the row engine reads; the `_rope_v1` computed entries keep their original
ABI). A row's current position is `seq_lens - 1`. Each CTA rotates its Q
fragments in registers; the CTA that owns a row's last unit loads the
current row from the QKV rows, rotates K, scales V, rounds (or quantizes)
both and writes them to the cache. Output and cache bits equal the row
engine followed by the unfused entry. Verify does not fold.

## Verify

Verify (`verify.cuh`, phase `Verify`) serves items of up to eight query
tokens after a cached prefix: a speculative verify step's current token and
drafts, or a DFlash draft block. It takes the prefill inputs: packed
`q[tokens][q_head][qk_dim]`, `q_start`, `q_len` and `prefix_len` per item
and a device live item count. Token t of an item sits at position
`prefix + t`. Paged items read every key from the cache, so the row engine
writes their rows before attention; ring items read keys before the prefix
from the ring and their own rows from packed `k_new`/`v_new`, so the ring is
written after attention. Causal items show token t the keys up to
`prefix + t`; bidirectional items (`causal = false`, DFlash) show every
token all of its item's rows. A window bounds each token from below at its
own position minus `window_left`.

The kernel uses decode's row-local split per item: an item of u units is
cut into pieces of `piece_units` units from its first visible unit, with
the item bucket's `split_units`, and each piece is read once for all of the
item's tokens. CTAs take even shares of the step's units and compute the
pieces that start in them. An item's bits therefore do not depend on the
other items of its step (`attention_verify_items_ignore_their_companions`).
With one split slot (SWA verify at 32 items or more) each item is one piece
and its CTA writes BF16 output directly. The DFlash drafter's block
attention is a bidirectional verify variant on the same kernel. Its item table (28 bytes per item, in
dynamic shared memory) holds 32 to 256 items; a larger live count, or more
query tokens than the table's items of eight rows, traps before any partial
is written. Partials are packed by query token (at most 256 tokens).

### Exact verify rows

Exact verify rows (MiMo, `Variant::plan_rows`) run a verify step's token
rows through the decode variant, so each row's bits are those of the row
alone in a one-row decode step at its position, whatever the step's shape.
They take the verify inputs. The paged entry `chop_attn_{name}_rows_v1`
runs grid z = token row with the one-row plan; `_rows_merge_v1` folds the
partials. The shared-key entry `chop_attn_{name}_rows_shared_v1` (`decode_rows.cuh`,
`Variant::plan_rows_shared`, paged full attention, at most 16 rows per item)
streams each K/V range once for up to four rows of equal unit count, with
partials equal to `_rows_v1`'s bit for bit; MiMo serves exact rows on it
([cost](../../../../docs/adaptive-speculation.md)). kernel-lab
`attention_exact_rows_equal_one_row_decode` checks every token row against
the unfused one-row decode entry bit for bit.

## Prefill

Prefill uses packed BF16 `q[tokens][q_head][qk_dim]` and device arrays
`q_start`, `q_len` and `prefix_len`, with a device live batch count. Paged
layers read both prefix and new tokens from the cache after the row engine
writes the chunk. Ring layers read the cached in-window prefix from the ring
and the chunk from contiguous post-RoPE `k_new` and scaled `v_new`; the row
engine writes the chunk tail into the ring after attention.

The prefill kernel (`prefill.cuh`) follows FlashAttention-2: eight warps
per 128-row grouped query tile, two shared-memory key stages of 32 keys (16
when `qk + v > 320`) with the 128-byte TMA swizzle, BF16
`mma.sync.m16n8k16` for S = QK^T and O += PV, and the online softmax in
registers. Paged BF16 stages arrive through 4D TMA boxes, FP8 stages are
widened to BF16 in a third buffer, and ring stages use `cp.async`.

Prefill rows do not depend on how the prompt is chunked: a row's arithmetic
follows its absolute position and data only, never the step's live rows,
the row's place in its chunk, the other items in the step or the grid
([numerics](../../../../docs/numerics.md)). Key tiles and paged query tiles
start at absolute multiples of their sizes, and a paged TMA box never
crosses a page.

Paged prefill splits each row's keys into segments at absolute multiples of
L = 256 * ceil(keys / 1024), where `keys` is the last key count of the row's
aligned query tile, so a row has at most four segments (`max_pieces`, the
planner's `PREFILL_SEGMENTS`). Persistent CTAs (one per SM divided among
the KV heads) divide the step's units evenly, visiting each batch's query
tiles in zigzag order. A row with one segment writes BF16 output directly;
a longer row writes one F32 partial per segment, and the merge folds them in
segment order and adds the optional sink. The paged prefill planner refuses
a window (`window_left`).

Ring prefill gives each query tile its own CTA and never splits: it adds
the sink, normalizes and writes BF16 output in the partial kernel; its merge
entry returns at once. Both prefill entries are `_direct_v1` symbols with
`sink` and `out` parameters.

## Qwen3.8 attention

Qwen3.8 Flash's full-attention layers use the paged variants
`qwen38_{bf16,fp8}_{prefill,decode,verify}`: per TP4 rank 6 Q heads and one
KV head of 256, 64-token pages, scale 1/16 and a 262,144-token context.
Rows that see more than 2051 tokens attend through the QSA sparse kernels,
which read the same page layout. Decode pads the group of six to one n8
tile whose padding heads hold zero queries and are never written.

The gated rows (`gated.cuh`, catalog `attention::gated`, same cubin) serve
every Qwen attention path. The packed projection rows are
`[row][6 * 512 | 256 | 256]`, each Q head's query followed by its gate
logits. `chop_attn_qwen38_qk_norm_rope_kv_{bf16,fp8}_v1` takes the QSA step
rows (`row_batch`, `positions`, I32 `rope_positions[row][3]`, U32
`live_rows`): per head it applies the Gemma RMSNorm
`x * rsqrt(mean(x^2) + eps) * (1 + w)` to Q and K, then NeoX RoPE over the
first 64 dimensions with interleaved M-RoPE (pair i reads H when
i % 3 == 1 and i < 33, W when i % 3 == 2 and i < 30, T otherwise) from the
F32 table `cos_sin[position][pair][2]`. It writes Q as BF16 into
`q[row][6][256]` and K and raw V to the cache, dividing by the layer scales
for FP8. `chop_attn_qwen38_output_gate_v1` multiplies BF16 attention output
by `sigmoid(gate)`, in place or not; below a gate logit of -80 the product
is formed in F64 (`common/sigmoid.cuh`). The decode variants' `_rope_v1`
entries fold the norm, M-RoPE and KV write in (`GatedRope`, planned by
`plan_folded_gated`) with bits equal to the row kernel followed by the
unfused entry.

## Launch and determinism

All kernels publish PDL dependents and wait for upstream data before
reading it; entries that return early still wait, so the launch chain stays
transitive. There are no floating atomics or persistent semaphores; every
partial has a fixed owner and merge order, so replay is deterministic. Each
token row of a step writes its heads' outputs (direct, or through its
merge); rows past the step's tokens are never written. The
planner reports each entry's dynamic shared bytes
(`Variant::dynamic_shared_bytes`); launchers raise the function limit above
48 KiB. Windowed decode and verify use dynamic shared memory; every other
entry keeps its stages in static shared memory.

## Testing

The kernel-lab attention suites (`tests/ops/attention*.rs`) check tight and
multi-layer strided pages with decoy data between layers, ring wrap-around
and cached prefixes, BF16 and FP8 KV, sink, softcap, long-context split
decode, mixed per-row decode contexts, graph replay, determinism and PDL.
Paged cases with more than `2048 * 4096` query-key pairs check 48 sampled
query tokens against one-row oracle batches. The generated invariance tests
(`attention_invariance.rs`) check chunked prefill and exact rows bit for
bit. `tests/ops/qwen38_attention.rs` covers Qwen3.8 dense decode at contexts
1 to 262,144, prefill chunks, verify, the gated rows against the
`kernel_core::gated_attention_oracle` and the folds. Build tests audit the
native `sm_120f` cubins for spills and shared-memory limits.
