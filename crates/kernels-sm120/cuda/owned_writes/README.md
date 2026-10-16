# Owner-routed cache writes

Kernel-core contracts: `context_parallel::ContextParallelQuery::OwnedQkNormRopeKvWrite`
and `OwnedIndexerKeyWrite`; their placement references are
`context_parallel_oracle::{owned_kv_positions, owned_block_row}`. Under a
context-parallel group of `cp_size` ranks, physical page j of a sequence
lives on group rank `j % cp_size` at its local page `table[j / cp_size]`
(`runtime_kv::ContextParallelPagesV1`). These entries write a step's rows so
that each K/V row and each indexer block key reaches only its owner. They
run the served kernels' arithmetic, so the stored bits are the served
kernels' bits, and live in their own cubin, so the served entries and their
locks stay as they are at context-parallel size 1.

- `owned_qk_norm_rope_kv<...>` (grid `[rows]`, 256 threads) is
  `attention/gated.cuh`'s `qk_norm_rope_kv` over group-local positions:
  `local_positions[row]` is the `LocalizeKeys` list of width 1 for this rank
  (the row's position on its pages, -1 when another rank owns the page).
  Every row's Q heads are written. The K/V warps of a row with -1 return
  before the shared body runs, so it stores no K or V. An owned row stores
  K and V through the rank's local table at `table[l / Page]`, slot
  `l % Page`. The localization runs once per step and serves every layer.
- `owned_index_key_write<Indexer<...>>` (grid `[rows / 4]`, 128 threads) is
  `attention/sparse_indexer.cuh`'s `indexer_key_write` with the block-key
  row lookup replaced: block b lies in physical page `b / (Page / Block)`,
  and only its owner pools and stores it, at its local page. The raw-key
  ring and the anchor ring are written on every rank as the served kernel
  writes them, so each rank can pool a block that straddles launches. It
  takes the global positions and the group rank and size.

Both are PDL launches that release dependents and wait before their first
read; rows past `*live_rows` return. No atomics; every row has one owner.
