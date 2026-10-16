# Paged attention partials

Kernel-core contract: `context_parallel::ContextParallelQuery::PagedAttentionPartials`;
f64 reference `context_parallel_oracle::paged_attention_partials`. The
partials are the inputs of the context-parallel merge
(`../context_parallel/README.md`), which folds them across a group.

Causal paged grouped-query attention over the keys one context-parallel
group rank owns. The cache is the attention engine's paged layout
(`../attention/README.md`): K and V pages `[page][kv_head][head_dim]` behind
per-item block tables of signed 32-bit page IDs, u64 page strides, BF16 or
FP8 E4M3 elements with per-layer F32 K and V scales. Under a group of
`cp_size` ranks, physical page j of a sequence lives on group rank
`j % cp_size` at the rank's local page `table[j / cp_size]`
(`runtime_kv::ContextParallelPagesV1`); `cp_size` 1 attends every key. These
entries live apart from the attention engine, whose kernels and bits stay
as they are without context parallelism.

`partials<Partials<QHeads, KvHeads, Dim, Page, Fp8>>`, grid `[rows,
KvHeads]`, 256 threads, static shared memory only. Row r at position
`positions[r]` of item `row_batch[r]` (decode, verify and prefill rows
alike) walks its rank's pages in order, keeping positions up to its own.
Per page, one key tile:

- warp w forms the scores of keys w, w + 8, ...: each lane FMA-sums its
  `Dim / 32` dimensions from zero, the lanes fold by xor shuffles (16, 8, 4,
  2, 1), and `s = dot * (k_scale * scale)`. A chain spans at most `Dim / 32`
  products (8 at 256); no MMA is used;
- warp h, one per query head, takes the tile's maximum, the running maximum
  `m'`, `alpha = exp(m - m')` (0 while m is -inf) and `p = exp(s - m')`
  (exp as `exp2(x * log2 e)`, as the merge computes it); lane 0 forms
  `l = l * alpha + p_0 + p_1 + ...` in key order;
- each thread scales its dimensions' numerators by alpha and FMA-adds
  `p * v` over the tile's keys in order.

The CTA writes F32 `m` and `l` `[rows, heads]` and `o * v_scale` `[rows,
heads, Dim]`, heads ordered by KV head then query head, the merge's
segment layout (`m`, `l`, `o`). A row with no kept key writes -inf, +0 and
+0, which the merge weighs 0. Rows past the live count return after the PDL
wait. No atomics; every reduction has a fixed order, so the bits do not
depend on the grid, the other rows or replay.

The entries for decode and verify (up to 256 rows) and prefill (up to 8192
rows) share the kernel; prefill rows of one item each walk their own causal
prefix. This first schedule is one CTA per row and KV head; it is the
reference for faster schedules, which must match it bit for bit.
