//! A model's registered n-gram hash entries as its bundle embeds them. A
//! catalog declares them (`ModelCatalog::ngram_hashes`) so kernel-lab runs
//! their oracle and replay cases without naming a model.
//!
//! The entries share one ABI. Every phase takes `tokens`, `history`,
//! `multipliers`, `sizes`, `offsets`, `ids` and `history_out`; verify then
//! takes `cu_rows`; every phase takes `q_len`; verify then takes
//! `live_items`; every phase takes `rank`, `table_start` and `table_rows`;
//! decode and verify end with `entry_words`. The grid's y axis is the rank's
//! local heads; prefill runs one item, decode one item per x block and
//! verify one item per z block.

/// The hashed tables' parameters, as `kernel_core::ngram_oracle::ngram_tables`
/// takes them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NgramTable {
    pub vocab: u64,
    pub orders: usize,
    pub heads_per_order: usize,
    pub base: u64,
    pub seed: u64,
    pub layer_index: u64,
    pub alignment: u64,
}

/// One rank count's registered hash entries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NgramHashForm {
    /// The embedded cubin holding the entries.
    pub cubin: String,
    pub prefill: String,
    pub decode: String,
    pub verify: String,
    pub ranks: u32,
    /// The heads each rank hashes: a rank owns the table rows of its heads.
    pub local_heads: u32,
    pub table: NgramTable,
    /// The token that resets a sequence's history.
    pub eos: u32,
}
