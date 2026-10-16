# Selected attention partials

Kernel-core contract: `context_parallel::ContextParallelQuery::SelectedAttentionPartials`;
f64 reference `context_parallel_oracle::selected_attention_partials`. The
partials are the inputs of the context-parallel merge
(`../context_parallel/README.md`), which folds them across a group.

The block-sparse attention of `../attention/SPARSE.md` over the keys one
context-parallel group rank owns. A row at position p has `c = (p + 1) /
Block` complete blocks of `Block` tokens from position 0 and the incomplete
tail `c * Block..=p`. Its key set is the tokens of its selected blocks,
`selected[row][0..min(c, TopBlocks)]` of a `[rows, TopBlocks]` i32 list
(ascending distinct blocks below c: the indexer's top-k, or under context
parallelism the merged global list), then the tail, which is always
attended. A row with at most `TopBlocks` complete blocks attends every
visible token. There is no sink, sliding window or softcap; every query
head of a KV head shares the key set. The cache is the attention engine's
paged layout, `[page][Page][kv_head][Dim]` K and V pools behind per-item
block tables, BF16 or FP8 E4M3 with per-layer F32 K and V scales. Under a
group of `cp_size` ranks, physical page j lives on group rank `j %
cp_size` at the rank's local page `table[j / cp_size]`
(`runtime_kv::ContextParallelPagesV1`). A block never straddles a page, so a
rank owns a selected block when it owns its page; the tail lies in one page.

`selected_partials<SelectedPartials<QHeads, KvHeads, Dim, Page, Block,
TopBlocks, Fp8>>`, grid `[rows, KvHeads]`, 256 threads, static shared memory
only. Warp 0 compacts the row's owned blocks in list order with one ballot
per 32 entries; a listed block outside the row's complete blocks traps. The
kept keys (owned blocks' tokens, then the owned tail) are walked in tiles
of 64 keys, each looked up through the rank's table once per tile. The
per-tile arithmetic is that of `../paged_partials/README.md`: lane FMA dot
products folded by xor shuffles, `s = dot * (k_scale * scale)`, the running
maximum and `alpha = exp(m - m')`, the denominator summed by lane 0 in key
order, and per-thread numerators scaled by alpha and FMA-accumulated over the
tile's keys in order. Outputs are F32 `m` and `l` `[rows, heads]` and `o *
v_scale` `[rows, heads, Dim]`, heads by KV head then query head; a row with
no kept key writes -inf, +0 and +0, which the merge weighs 0. No atomics;
the bits do not depend on the grid, the other rows or replay.

These entries live apart from the block-sparse attention engine, whose
kernels and bits stay as they are at context-parallel size 1. The served
shape is a TP4 rank of the block-sparse attention: six query heads over one
KV head of 256, 64-token pages, blocks of 4 and 512 selected blocks; decode
and verify up to 256 rows and prefill up to 8192, BF16 and FP8 KV.
