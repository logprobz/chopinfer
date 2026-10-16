# Pooled top-k indexer

The SM120 engine for `kernel_core::pooled_indexer::PooledIndexerQuery` and
`kernel_core::context_parallel::ContextParallelQuery::TopKMerge`: the key
LayerNorm and gated group pooling write, streaming score and top-k selection
(global or per rank), the exact merge of per-rank selections and the key
list compaction. The f64 oracles are `kernel_core::gated_pooling_oracle` and
`kernel_core::pooled_indexer_oracle`; the planner and cubin specs are
`kernels_sm120::pooled_indexer`.

Every entry is a PDL launch that releases dependents and then waits before
reading any input. Grids are sized for the prepared rows; CTAs past
`*live_rows` return. Rows carry `row_batch[row]` (the sequence: its page
table row and slot) and `positions[row]`; a sequence's rows are contiguous
with consecutive positions. No float atomics: every sum has a fixed order and
every result one owner, so outputs are bitwise identical across relaunch and
graph replay.

## Key write

`key_write<KeyWrite<Width, Group, Page, PositionF32, MaxRows>>`, one warp per
row (lane l holds channels 4l to 4l + 3), 128 threads.

- LayerNorm of the raw key row in F32: the mean and then the variance of the
  deviations (each a per-lane sum, then an xor warp tree), `rstd =
  __frsqrt_rn(var + eps)`, `y = fma((x - mean) * rstd, w, b)`, rounded once to
  BF16. Pooling reads these BF16 values, for tokens of this launch and of the
  tail state alike.
- Pooling: per channel `logit_t = gate_t + position_t`, `m = max`, `e_t =
  expf(logit_t - m)`, the sum in offset order, `w_t = e_t / sum` (IEEE
  division), `pooled = bf16(fma chain over t of w_t * key_t)`. The pooled key
  of pool g goes to row `(g * Group % Page) / Group` of the page holding token
  `g * Group` (see "Pages"); a rank that does not own that page skips the
  store.
- Roles: a row that completes a group whose tokens are all rows of this
  launch pools it. The first row of each sequence pools the group straddling
  the launch start, whose earlier tokens are the live slots of the source
  entry, and then writes the tail state. Every tail read and write of one
  sequence therefore happens in one warp, in program order.
- Tail state: BF16 entries of `GatedGroupPooling::tail_state_elements`, keys
  bank then gate bank, slot `position % Group` for tokens of the incomplete
  group only; dead slots are not written. Prefill and decode
  (`MaxRows == 0`) update `slot_ids[batch]` in place. Verify reads
  `slot_ids[batch]` and writes the whole tail after row r to
  `checkpoint_ids[batch * MaxRows + r]`; checkpoint entries must not alias the
  source entry.

Operands: key and gate rows `[rows, *_stride]` BF16 (128 used), LayerNorm
weight and bias `[128]` BF16, position embedding `[Group][128]` BF16 or F32.

## Selection

`select_split<Select<Heads, Dim, Group, Page, MaxTopK, Tail, PerRank>>`,
grid `[rows, splits]`, 256 threads, 58,256 bytes of dynamic shared memory at
32 heads (launchers opt in above 48 KiB).

- A row at position p sees pools `0..(p + 1) / Group`; its candidates are
  the ones the launch's group rank owns, in local order (see "Pages"). Split
  s of S takes the tile-aligned local range `[s * c, (s + 1) * c)`, `c =
  ceil(owned / S)` rounded up to 64 pools.
- Each 64-pool tile is copied with 16-byte cp.async into a swizzled stage
  (two stages); pools past the range are zero-filled and are not
  candidates. The query `[Heads][128]` stays in
  shared memory.
- Warp w forms `q_h . K_g` for the 8 pools of n8 column w and every head with
  BF16 `mma.sync.m16n8k16` (heads are the M rows) over one 128-K chain from
  zero, then `r = relu(dot * 128^-0.5)` with `x > 0 ? x : +0`. Thread g sums
  `w_h * r_h` with F32 FMA from zero in head order, `w_h = bf16(weight_h) *
  Heads^-0.5`.
- Streaming top-k (`TopK` in common.cuh): a candidate is the 64-bit key
  (score bits mapped to unsigned IEEE order with -0 as +0 and every NaN below
  -inf, then the inverted pool), so larger is better and equal scores rank the
  lower pool first. A NaN score is still a candidate; per-rank pairs carry
  it as `0xffffffff` and zeros as +0. Offers above
  the threshold append to a `2 * MaxTopK / Group` buffer; a bitonic sort keeps
  the best `top_k / Group` and sets the threshold. Candidates are distinct,
  so the kept set does not depend on offer order.
- Workspace: u64 `[rows][splits][MaxTopK / Group]`, each split's kept keys
  descending then zeros. It does not depend on the context: no `[rows,
  pools]` score buffer exists. `splits = min(64, next_pow2(ceil(376 /
  rows)))`, so the workspace is at most 8 MiB for 2048 rows.

`select_merge`, grid `[rows]`, folds the split lists with the same buffer
(each list stops at its first key at or below the threshold). `Global` writes
the key list `[rows, MaxTopK + Group - 1]` i32: kept pools ascending, each
as its `Group` positions, then the tail `[(p + 1) / Group * Group, p]`, then
-1. A `Global` row with no more visible pools than it keeps selects them all
without scoring. `PerRank` writes `[rows, MaxTopK / Group]` u32 pairs (F32
score bits, pool), best first, padded with (-inf, -1).

`topk_merge<Select<...>, Ranks>` folds per-rank pairs gathered rank-major
`[Ranks][rank_rows][MaxTopK / Group][2]` into the `Global` key list.
Per-rank scores use the global arithmetic, so the merge equals the
single-rank selection bit for bit. `compact_selection<Width>` copies
`lists[source[i]]` to row i.

The runtime `top_k` is a launch scalar (a model option): a positive multiple
of `Group` of at most `MaxTopK`.

## Tree merge

`select_merge_tree<Select<...>, 16>` (merge_tree.cuh,
`kernels_sm120::pooled_indexer::merge_tree`) writes `select_merge`'s key
list or per-rank pairs bit for bit from the same workspace, launch ABI and
grid `[rows]`, with 512 threads. The candidates are distinct keys, so the
kept set is the best `top_k / Group` keys of the union of the split lists
whatever the merge order; only the merge's schedule differs.

- A warp holds a list of `MaxTopK / Group` keys in registers, lane l
  holding elements `l * 16` to `l * 16 + 15`. The best keys of two lists
  sorted descending are the elementwise maxima of one and the other
  reversed: a bitonic sequence, sorted again by a bitonic merge of
  `log2(MaxTopK / Group)` compare-exchange stages (warp shuffles across
  lanes, register swaps within).
- Warp w folds split lists w, w + 16, ... from the workspace. The warps
  then fold pairwise through shared memory (the upper half stores its
  lists reversed) until warp 0 holds the row's best keys; a level whose
  upper warps hold no list is skipped.
- The kept keys are the nonzero ones among the first `top_k / Group`, as
  `select_merge`'s `min(live, kept)`. `Global` sorts them pool-major in
  warp 0 and writes the key list with `write_list`; `PerRank` writes the
  pairs, best first. A `Global` row with no more visible pools than it keeps
  selects them all without reading the workspace.

Every compare-exchange takes the full 64-bit key, so equal scores rank the
lower pool first as in the streaming top-k. The lab family
`pooled_indexer_merge_tree` builds the merges of the lab profile's
selections and three mutants: the last split list left out, keys compared by
their score bits alone, and the dependency wait after the reads of the live
count and the lists. Its test (`crates/kernel-lab/tests/
pooled_indexer_merge_tree.rs`) compares both merges with each other and
with the host's selection.

## Radix merge

`select_merge_radix<Select<...>>` (merge_radix.cuh,
`kernels_sm120::pooled_indexer::merge_radix`) writes a `Global` selection's
key list bit for bit from `select_merge`'s workspace, launch ABI and grid
`[rows]`, with 512 threads. The kept set is the `top_k / Group` largest of
the distinct candidate keys, so any exact selection of it writes the same
list; the radix merge finds it without folding every split list.

- `split_range` gives every live split but the last `aligned` pools, so
  split s's list holds exactly `min(pools of s, kept)` keys first, every one
  nonzero. The merge uses only those (at most `splits * kept`), and none of
  an empty split's list or of a list's zero padding.
- A row of at most 4096 candidates (contexts up to 16K tokens at 64 splits)
  copies them to shared memory and runs the radix select below. A row of
  more folds its live lists through the tree merge's register networks, as
  `select_merge_tree` does, each list read as its kept keys then zeros:
  re-reading the workspace on every radix pass costs more there (136 us at
  128K tokens in the lab against the tree's 35).
- The kept-th largest key is found a byte at a time from the top: a
  256-bin histogram of the byte among the keys that match the bytes found
  so far (one shared add per distinct bin of a warp), then warp 0 walks the
  bins from the top to the one that holds the kept-th key. The search stops
  once that bin's keys are exactly the ones still needed; keys are
  distinct, so it stops by the last byte.
- The kept keys are those whose bytes down to the stopping one are at least
  the threshold's. They are stored pool-major, sorted descending in shared
  memory (pools ascending) and written with `write_list`. A row with no
  more visible pools than it keeps selects them all without reading the
  workspace.

The lab family `pooled_indexer_merge_radix` builds the lab profile's radix
merge and three mutants: the last live split's list left out, equal scores
ranking the higher pool first, and the dependency wait after the reads of
the live count and the lists. Its test (`crates/kernel-lab/tests/
pooled_indexer_merge_radix.rs`) compares it with `select_merge`, the tree
merge and the host's selection, reading a workspace whose unused words are
all-ones and, in rows the radix select takes, whose lists are shuffled.

## One-split selection

`select_rows<RowSelect<Select<...>, Rows, Stages>>` (select_rows.cuh,
`kernels_sm120::pooled_indexer::rows`) writes a selection entry's key lists
(`Global`) or per-rank pairs at one split per row, the outputs of
`select_split` and `select_merge` bit for bit, as one kernel: no split
workspace and no merge launch. Same ABI as the selection, with the merge's
`output` in place of the workspace and no split count.

- Every score is the selection's instruction sequence (the same MMA
  operands and k order into one 128-K chain per head, scale, ReLU and the
  head-ordered FMA chain), and candidates are the same distinct keys, so
  the kept set does not depend on the order of offers or reductions.
- Warp w scores n8 column `w % 8` of each 64-pool tile for CTA row
  `w / 8`, its row's query held as A fragments in registers, and sums its
  own 8 pools' head chains on lanes 0 to 7. The dots stay in the warp: one
  barrier per tile, where the selection has four. The dots are
  `[8 pools][heads + 4]` F32, in the 2 KiB of key rows only that warp reads
  (one row per CTA) or in a buffer per warp (two rows, whose warps share
  their key rows); a chain lane loads its pool's dots and the weights four
  at a time before its head-ordered FMAs, and both the fragment stores and
  those loads hit distinct banks.
- `Stages` key stages keep `Stages - 1` tiles in flight; each thread reads
  the page table entries of its copies one tile before it issues them.
- Each warp counts its offers by ballot into one of two per-warp slots, so
  every thread knows each row's buffer fill after the tile barrier; a
  buffer that could overflow is trimmed there, both rows' together.
- A trim keeps a row's best `top_k / Group` keys without sorting: an exact
  radix select from the top in 8-bit digits (digit counts by shared atomic
  adds, which do not depend on their order; a scan finds the digit of the
  kept-th key; the passes end when that digit's keys are exactly the rest
  to keep), then the keys at or above the prefix move to the front. The
  threshold becomes that prefix, a lower bound of the kept-th key, so no
  offer the kept set needs is dropped.
- A `Global` row trims to the best `top_k / Group` keys (when it holds
  more) and writes them in pool order: with one row per CTA through a
  bitmap of its visible pools in the free key stages (`bitmap_pools`, 32
  KiB: a bit per pool of 1,048,576 tokens at pools of 4), each thread
  writing the set bits of its run of words at their rank, while the row's
  visible pools fit; a row with more, and every two-row CTA, sorts them. A
  `PerRank` row trims and sorts them best first and writes its pairs. A
  `Global` row with no more visible pools than it keeps writes them all
  without scoring.
- A trim spreads a row's buffer (twice its kept pools) evenly over the
  row's 256 threads, so the forms take selections that keep at least 128
  pools.

| Form | Threads | Stages | Dynamic shared bytes | CTAs per SM |
| --- | ---: | ---: | ---: | ---: |
| one row (`_rows1`) | 256 | 2 | 42,240 | 2 |
| two rows of one sequence (`_rows2`) | 512 | 3 | 86,528 | 1 |

Two rows of different sequences, or a pair where only one row scores, take
one pass per scoring row. Grid `[ceil(rows / Rows)]`; CTAs past the live
rows return after the PDL wait. The entries build into
`pooled_indexer_rows_{tag}_sm120`, apart from the selection's cubin.

## Head-sum selection

`head_sum_split<HeadSumSelect<4, 128, Group, Page, MaxTopK, Tail, PerRank>>`
(head_sum.cuh, `kernels_sm120::pooled_indexer::head_sum`) is the selection
above with the scores of the block-sparse indexer's `block_scores`
(attention/sparse_indexer.cuh), for
`PooledIndexerQuery::HeadSumSelect`: no head weights, and the same
instruction sequence per score, so the two engines select from the same F32
bits.

- The query tile is one m16 tile: the four heads in rows 0 to 3, zeros
  below. Warp w takes the 8 pools of n8 column w. Two accumulators start at
  zero: `even` takes k16 steps 0, 2, 4 and 6, `odd` steps 1, 3, 5 and 7.
- `x = even + odd`, `r = x > 0 ? x : +0`, then two xor shuffles over lane
  bits 2 and 3 (the rows' head bits), so lanes 0 to 3 hold `(r0 + r1) + (r2
  + r3)` of pools 2c and 2c + 1; one multiply by `rsqrtf(128)`, folded to
  0x3db504f3.
- Splits, the streaming top-k, the split merge (`select_merge`), the
  per-rank pairs and the exact merge (`topk_merge`) are the selection's,
  instantiated with `HeadSumSelect`. Grid `[rows, splits]`, 256 threads,
  45,840 bytes of dynamic shared memory.

Operands are the block-sparse indexer's own: queries `[rows][4][128]` BF16
and block keys `[page][Page / Group][128]` BF16 at their page stride through
the KV page table, with the page ownership of "Pages" under context
parallelism. The entries build into `pooled_indexer_head_sum_{tag}_sm120`,
apart from the weighted selection's cubins.

## Block-id merge

`topk_merge_blocks<HeadSumSelect<...>, Ranks>` (block_merge.cuh,
`kernels_sm120::pooled_indexer::block_merge`, contract
`ContextParallelQuery::TopKMergeBlocks`) folds per-rank head-sum pairs
exactly as `topk_merge` does. It writes the kept pools as i32 block ids
`[rows, MaxTopK / Group]`, ascending, then -1: the list the block-sparse
attention and the selected partials read. Grid `[rows]`, 256 threads, in
its own cubin `block_merge_{tag}_sm120`.

## Pages

Pooled keys live in the KV pages of `runtime_kv::ContextParallelPagesV1`
(docs/serving-kv-pool.md "Context parallelism"). With a context-parallel
group of `cp_size` ranks (1 without context parallelism), physical page j of
a sequence holds tokens `[j * Page, (j + 1) * Page)` and lives on group rank
`j % cp_size`, in that rank's local page `table[j / cp_size]`: each rank's
table row lists its own local page per stripe of `cp_size * Page` tokens. A
page holds `Page / Group` whole pools, so group rank r owns the pools of its
pages; its local pool l is sequence pool `(l / (Page / Group) * cp_size + r) *
(Page / Group) + l % (Page / Group)`, at row `l % (Page / Group)` of local page
`table[l / (Page / Group)]`. At `cp_size` 1 the table is the plain page table.
The key write and the selection take `cp_rank` and `cp_size` as launch
scalars; `Global` selection takes 0 and 1, `PerRank` its group rank and the
entry's rank count. `kernels_sm120::pooled_indexer::{pool_address,
rank_pools, rank_pool}` are the host mirror, tested against
`ContextParallelPagesV1`.

## Capacities

| Entry | Rows | Notes |
| --- | ---: | --- |
| `key_write_prefill` / `_decode` / `_verify` | 2048 / 32 / 32 items x verify rows | |
| `select`, `select_per_rank{2,4}` | 2048 | decode, verify and prefill chunks |
| `topk_merge{2,4}` | 2048 | |
| `compact` | 32 | key list width 2051 |
| `head_sum_select`, `head_sum_select_per_rank{2,4}`, `head_sum_topk_merge{2,4}` | 2048 | |

Contexts reach 1,048,576 tokens (262,144 pools); positions and pool indexes
fit i32, and page offsets are 64-bit.

## Testing

`crates/kernels-sm120/tests/pooled_indexer_compile.rs` builds every entry
through kernel-build (sm_120f, exact symbols, no spills, static shared
memory) and checks the SASS for `HMMA`, `LDSM` and `LDGSTS`;
`pooled_indexer_head_sum_compile.rs` builds the head-sum entries and checks
their two chains of four MMAs and the folded scale; `pooled_rows_build.rs`
builds the one-split forms and their lab mutants (no spills or stack,
registers and shared memory for each form's CTAs per SM).
`crates/kernel-lab/tests/pooled_indexer_rows.rs` compares the one-split
forms with the selection byte for byte on the GPU.
