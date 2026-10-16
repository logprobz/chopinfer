//! Context-parallel merges. Under context parallelism each rank of a group
//! owns a disjoint share of a sequence's keys (whole KV pages), attends over
//! its own keys and scores its own indexer candidates; these entries fold
//! the ranks' results, gathered rank-major by `contract::CommQuery::AllGather`,
//! into what one rank holding every key would produce.
//!
//! The ops ([`PagedPartialsOp`], [`OwnedKvWriteOp`], [`OwnedKeyWriteOp`],
//! [`GroupTransferOp`]) add what the queries leave out and a model shares
//! with its other ops: the cache's context (which fixes a block table row),
//! the softmax scale, the packed rows' indexer heads, and the group
//! transfers that move partials, queries and selections between the
//! members. Every query field is math apart from `capacity`, which the
//! backend states and a model asks for in
//! [`Need::capacity`](crate::backend::Need). A group's size is math (it
//! fixes which rank owns a page); how many ranks a backend's collectives
//! hold, how a transfer reaches its peers (which peer is sent to when, in
//! how many launches) and the attention partials' head width granule are
//! the backend's limits and choices. No context-parallel op's bits depend
//! on the transfers' order: the merges fold the ranks in rank order
//! whatever order their partials arrive in.

use crate::attention::KvCache;
use crate::collective::GatherBuffers;
use crate::contract::{
    AttentionPhase, DType, GatedAttentionQuery, NumericalContract, NumericalContractId, Refusal,
    SparseAttentionQuery, Validate,
};
use crate::pooled_indexer::{key_list_width, validate_selection_shape};

/// Context-parallel merge contracts. `ranks` is the context-parallel group
/// size, at least 2; a backend states the most ranks its collectives hold.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ContextParallelQuery {
    /// Log-sum-exp merge of attention partials
    /// (`latent_attention::LatentAttentionOutput::Partial`): per rank r the
    /// F32 maximum `m_r` and denominator `l_r` `[rows, heads]` and numerator
    /// `o_r` `[rows, heads, width]`. Per (row, head), in F32:
    ///
    /// - `M = max_r m_r`; when every `m_r` is -inf the output is zeros;
    /// - `a_r = exp(m_r - M)` (0 for a rank with `m_r = -inf`);
    /// - `L = sum_r a_r * l_r` and `O = sum_r a_r * o_r`, FMA from zero in
    ///   rank order 0, 1, ...;
    /// - output `bf16(O / L)` `[rows, heads, width]` with an F32 division.
    ///
    /// One rank reproduces the normalized attention output bit for bit
    /// (`a = 1`, and `bf16(o / l)` is that output's rounding).
    PartialMerge {
        ranks: u32,
        heads: u32,
        width: u32,
        capacity: u32,
    },
    /// Exact merge of per-rank indexer selections
    /// (`pooled_indexer::IndexerSelectionMode::PerRank`): rank r supplies
    /// `[rows, max_top_k / group]` (F32 score, i32 pool) pairs. With the
    /// runtime `top_k` and `k = top_k / group`, the merge keeps the first k
    /// of the union ordered by score descending (the selection order: -0
    /// equals +0, NaN below -inf), then pool ascending (padding pairs, pool
    /// -1, never win), and writes the key list exactly
    /// as the single-rank selection does: kept pools ascending, each as its
    /// `group` positions, then the always-selected tail of the row's
    /// incomplete pool, then -1. Each rank's list is its own top k by the
    /// same order and its scores are the single-rank scores, so every global
    /// winner appears in its owner's list and the merged list equals the
    /// single-rank list.
    TopKMerge {
        ranks: u32,
        group: u32,
        max_top_k: u32,
        always_tail: bool,
        capacity: u32,
    },
    /// Key list localization for one group rank: an i32 key list `[rows,
    /// width]` of sequence positions (-1 holes) becomes the rank's local
    /// positions. Over physical pages of `page` tokens, position p lives on
    /// group rank `(p / page) % ranks` (`runtime_kv::ContextParallelPagesV1`);
    /// the owner's entry is its local position `p / (ranks * page) * page +
    /// p % page`, every other entry -1, in place order. Exact.
    LocalizeKeys {
        ranks: u32,
        page: u32,
        width: u32,
        capacity: u32,
    },
    /// Causal paged grouped-query attention over the keys one group rank
    /// owns, as the unnormalized partials `PartialMerge` folds. Each KV head
    /// serves `q_heads` query heads; a rank holds `kv_heads` KV heads (the
    /// same KV head may be replicated on several ranks of a group). The
    /// cache is the paged layout of `contract::AttentionLayout::Paged`: K and
    /// V pages `[page][kv_head][head_dim]` reached through the rank's block
    /// table, in BF16 or FP8 E4M3 with per-layer F32 K and V scales.
    ///
    /// Row r (a decode, verify or prefill token) of item `row_batch[r]` at
    /// absolute position `positions[r]` sees the keys at positions `0..=p`
    /// that launch-time group rank `cp_rank` of `cp_size` owns: position t
    /// lives on group rank `(t / page) % cp_size`, at the rank's local page
    /// `table[t / (cp_size * page)]`, slot `t % page`
    /// (`runtime_kv::ContextParallelPagesV1`; `cp_size` 1 sees every key).
    /// Per query head h and kept key j, in F32:
    ///
    /// - `s_j = (q_h . k_j) * k_scale * scale`, the dot product FMA-summed
    ///   over the dimension in a fixed order;
    /// - `m = max_j s_j`, `l = sum_j exp(s_j - m)` and `o = v_scale * sum_j
    ///   exp(s_j - m) v_j`, accumulated over keys in position order with the
    ///   running maximum (each key tile's earlier sums scaled by
    ///   `exp(m_old - m_new)`);
    /// - a row with no kept key writes `m = -inf`, `l = +0`, `o = +0`.
    ///
    /// Outputs: F32 `m` and `l` `[rows, heads]` and `o` `[rows, heads,
    /// head_dim]`, heads ordered by KV head then query head. Against the f64
    /// reference the maximum holds within one score's bound and `l` and `o`
    /// within `2^-22 K S` of the reference rescaled to the kernel's maximum
    /// (`context_parallel_oracle::paged_attention_partials`).
    PagedAttentionPartials {
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        page: u32,
        kv: DType,
        phase: AttentionPhase,
        capacity: u32,
    },
    /// `PagedAttentionPartials` over a row's selected blocks: the
    /// block-sparse attention of `contract::SparseAttentionQuery::
    /// SparseAttention` restricted to one group rank's keys. A row at
    /// position p has `c = (p + 1) / block` complete blocks of `block`
    /// tokens from position 0 and the incomplete tail `c * block..=p`. Its
    /// key set is the tokens of its selected blocks, i32 `selected[r][0..min(c,
    /// top_blocks)]` of a `[rows, top_blocks]` list (distinct block indices
    /// below c, ascending: the indexer's top-k, or the merged global list
    /// under context parallelism), each as its `block` positions ascending,
    /// then the tail, which is always attended. A row with at most
    /// `top_blocks` complete blocks therefore attends every visible token.
    /// There is no sink, sliding window or softcap, and every query head of a
    /// KV head shares the key set.
    ///
    /// The rank keeps the key set's positions it owns, page placement and
    /// arithmetic as in `PagedAttentionPartials`, accumulated over the kept
    /// keys in position order; a row with no kept key writes `m = -inf`,
    /// `l = +0`, `o = +0`. The outputs and the gate are those of
    /// `PagedAttentionPartials`
    /// (`context_parallel_oracle::selected_attention_partials`). At
    /// `cp_size` 1 the merge of one rank's partials is the selected
    /// attention within the gate; it is not the block-sparse engine's bits.
    SelectedAttentionPartials {
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        page: u32,
        block: u32,
        top_blocks: u32,
        kv: DType,
        phase: AttentionPhase,
        capacity: u32,
    },
    /// `write` (a `GatedAttentionQuery::QkNormRopeKvWrite`) with its K and V
    /// rows routed to their owner. Every row's Q heads are written as
    /// `write` writes them. Each row's cache position comes from a
    /// `LocalizeKeys` list of width 1 for this group rank: -1 writes no K or
    /// V; a local position l writes the row's K and V exactly as `write`
    /// would at position l, at the rank's local page `table[l / page]`, slot
    /// `l % page`. Q, K and V bits are `write`'s.
    OwnedQkNormRopeKvWrite {
        write: GatedAttentionQuery,
        ranks: u32,
    },
    /// `write` (a `SparseAttentionQuery::IndexerKeyWrite`) with its block
    /// keys routed to their owner. The raw-key ring and anchor ring are
    /// written as `write` writes them, on every group rank. Block b's key
    /// lies in physical page `j = b * block / page`; group rank `j % ranks`
    /// stores it, at its local page `table[j / ranks]`, row `b % (page /
    /// block)`, and the others store nothing. Bits are `write`'s.
    OwnedIndexerKeyWrite {
        write: SparseAttentionQuery,
        page: u32,
        ranks: u32,
    },
    /// `TopKMerge` written as block ids: the kept pools of a row, ascending,
    /// as i32 `[rows, max_top_k / group]`, then -1. With `top_k = max_top_k`
    /// this is the block-sparse selection's list
    /// (`SparseAttentionQuery::BlockTopK`): every complete block when there
    /// are at most `max_top_k / group`, otherwise the exact top blocks, ties
    /// to the lower block.
    TopKMergeBlocks {
        ranks: u32,
        group: u32,
        max_top_k: u32,
        capacity: u32,
    },
}

impl ContextParallelQuery {
    /// Entries of a merged key list row; `None` for a partial merge and for
    /// a width above `pooled_indexer::MAX_KEY_LIST_WIDTH`, which validation
    /// refuses.
    pub fn list_width(&self) -> Option<u32> {
        match *self {
            Self::TopKMerge {
                group,
                max_top_k,
                always_tail,
                ..
            } => key_list_width(group, max_top_k, always_tail),
            Self::LocalizeKeys { width, .. } => (width <= i32::MAX as u32).then_some(width),
            Self::PartialMerge { .. }
            | Self::PagedAttentionPartials { .. }
            | Self::SelectedAttentionPartials { .. }
            | Self::OwnedQkNormRopeKvWrite { .. }
            | Self::OwnedIndexerKeyWrite { .. }
            | Self::TopKMergeBlocks { .. } => None,
        }
    }
}

impl Validate for ContextParallelQuery {
    fn validate(&self) -> Result<(), Refusal> {
        let valid_ranks = |ranks: u32| ranks >= 2;
        match *self {
            Self::PartialMerge {
                ranks,
                heads,
                width,
                capacity,
            } => {
                if !valid_ranks(ranks) || heads == 0 || width == 0 || capacity == 0 {
                    return Err(Refusal("invalid context-parallel partial merge geometry"));
                }
            }
            Self::TopKMerge {
                ranks,
                group,
                max_top_k,
                always_tail,
                capacity,
            } => {
                if !valid_ranks(ranks) || capacity == 0 {
                    return Err(Refusal("invalid context-parallel top-k merge geometry"));
                }
                validate_selection_shape(group, max_top_k, always_tail)?;
            }
            Self::LocalizeKeys {
                ranks,
                page,
                width,
                capacity,
            } => {
                if !valid_ranks(ranks)
                    || page == 0
                    || width == 0
                    || width > i32::MAX as u32
                    || capacity == 0
                {
                    return Err(Refusal(
                        "invalid context-parallel key localization geometry",
                    ));
                }
            }
            Self::PagedAttentionPartials {
                q_heads,
                kv_heads,
                head_dim,
                page,
                kv,
                capacity,
                ..
            } => {
                if q_heads == 0 || kv_heads == 0 || head_dim == 0 || page == 0 || capacity == 0 {
                    return Err(Refusal("invalid paged attention partial geometry"));
                }
                if !matches!(kv, DType::Bf16 | DType::Fp8E4M3) {
                    return Err(Refusal("paged attention partials read BF16 or FP8 E4M3 KV"));
                }
            }
            Self::SelectedAttentionPartials {
                q_heads,
                kv_heads,
                head_dim,
                page,
                block,
                top_blocks,
                kv,
                phase,
                capacity,
            } => {
                if block == 0 || top_blocks == 0 || !page.is_multiple_of(block) {
                    return Err(Refusal("invalid selected attention partial blocks"));
                }
                Self::PagedAttentionPartials {
                    q_heads,
                    kv_heads,
                    head_dim,
                    page,
                    kv,
                    phase,
                    capacity,
                }
                .validate()?;
            }
            Self::OwnedQkNormRopeKvWrite { write, ranks } => {
                if !valid_ranks(ranks)
                    || !matches!(write, GatedAttentionQuery::QkNormRopeKvWrite { .. })
                {
                    return Err(Refusal("invalid owner-routed KV write"));
                }
                write.validate()?;
            }
            Self::OwnedIndexerKeyWrite { write, page, ranks } => {
                let SparseAttentionQuery::IndexerKeyWrite { block, .. } = write else {
                    return Err(Refusal("invalid owner-routed indexer key write"));
                };
                if !valid_ranks(ranks) || page == 0 || !page.is_multiple_of(block) {
                    return Err(Refusal("invalid owner-routed indexer key write"));
                }
                write.validate()?;
            }
            Self::TopKMergeBlocks {
                ranks,
                group,
                max_top_k,
                capacity,
            } => {
                if !valid_ranks(ranks) || capacity == 0 {
                    return Err(Refusal("invalid context-parallel top-k merge geometry"));
                }
                validate_selection_shape(group, max_top_k, false)?;
            }
        }
        Ok(())
    }
}

impl NumericalContract for ContextParallelQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        let name = match self {
            Self::PartialMerge { .. } => "row.context-parallel-lse-merge-rank-order.v1",
            Self::TopKMerge { .. } => "sparse.context-parallel-topk-merge-lower-index.v1",
            Self::LocalizeKeys { .. } => "sparse.context-parallel-localize-keys.v1",
            Self::PagedAttentionPartials { .. } => "attention.paged-gqa-partials-position-order.v1",
            Self::SelectedAttentionPartials { .. } => {
                "attention.selected-block-gqa-partials-position-order.v1"
            }
            Self::OwnedQkNormRopeKvWrite { .. } => "row.context-parallel-owner-kv-write.v1",
            Self::OwnedIndexerKeyWrite { .. } => {
                "sparse.context-parallel-owner-indexer-key-write.v1"
            }
            Self::TopKMergeBlocks { .. } => {
                "sparse.context-parallel-topk-merge-block-ids-lower-index.v1"
            }
        };
        Ok(NumericalContractId::new(name.to_owned()))
    }
}

/// One group rank's paged attention partials
/// (`ContextParallelQuery::PagedAttentionPartials`) with logits scaled by
/// `softmax_scale`, over a paged cache of up to `max_context` tokens per
/// item. Its buffers are [`PagedPartialsBuffers`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PagedPartialsOp {
    pub query: ContextParallelQuery,
    pub softmax_scale: f32,
    /// The most tokens of one item's block table row, which holds
    /// `max_context / page` pages, rounded up.
    pub max_context: u32,
}

impl Validate for PagedPartialsOp {
    fn validate(&self) -> Result<(), Refusal> {
        self.query.validate()?;
        if !matches!(
            self.query,
            ContextParallelQuery::PagedAttentionPartials { .. }
        ) {
            return Err(Refusal("paged partials attend by a paged partials query"));
        }
        if !self.softmax_scale.is_finite() || self.softmax_scale <= 0.0 || self.max_context == 0 {
            return Err(Refusal(
                "paged partials need a positive softmax scale and a context",
            ));
        }
        Ok(())
    }
}

/// The owner-routed gated KV write
/// (`ContextParallelQuery::OwnedQkNormRopeKvWrite`) over a cache of up to
/// `max_context` local tokens per item. Its buffers are those of the write
/// it routes (`gated_sparse::GatedKvWriteBuffers`), whose `positions` are
/// the rows' local positions: the `LocalizeKeys` list of width 1 this group
/// rank localized from the rows' cache positions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OwnedKvWriteOp {
    pub query: ContextParallelQuery,
    /// The most local tokens of one item: its block table row holds
    /// `max_context / page_size` pages, rounded up.
    pub max_context: u32,
}

impl Validate for OwnedKvWriteOp {
    fn validate(&self) -> Result<(), Refusal> {
        self.query.validate()?;
        if !matches!(
            self.query,
            ContextParallelQuery::OwnedQkNormRopeKvWrite { .. }
        ) {
            return Err(Refusal(
                "an owner-routed KV write writes by an owner-routed KV write query",
            ));
        }
        if self.max_context == 0 {
            return Err(Refusal("an owner-routed KV write needs a context"));
        }
        Ok(())
    }
}

/// The owner-routed indexer key write
/// (`ContextParallelQuery::OwnedIndexerKeyWrite`) of packed indexer
/// projection rows `[row][index_heads * head_dim | head_dim]` for items of
/// up to `max_context` tokens. Its buffers are those of the write it routes
/// (`gated_sparse::SparseKeyWriteBuffers`): the rings hold every row, and
/// the block table and block-key plane are this group rank's local pages.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OwnedKeyWriteOp {
    pub query: ContextParallelQuery,
    pub index_heads: u32,
    /// The most tokens of one item: its block table row holds
    /// `max_context / page` entries, rounded up.
    pub max_context: u32,
}

impl Validate for OwnedKeyWriteOp {
    fn validate(&self) -> Result<(), Refusal> {
        self.query.validate()?;
        if !matches!(
            self.query,
            ContextParallelQuery::OwnedIndexerKeyWrite { .. }
        ) {
            return Err(Refusal(
                "an owner-routed key write writes by an owner-routed key write query",
            ));
        }
        if self.index_heads == 0 || self.max_context == 0 {
            return Err(Refusal(
                "an owner-routed key write needs index heads and a context",
            ));
        }
        Ok(())
    }
}

/// How a group transfer moves each member's 32-bit words, bit for bit (two
/// BF16 values per word). Group rank q's words land at slot q of a
/// receiving output.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum GroupTransfer {
    /// Every member's input lands in every member's output, which then
    /// holds the group's inputs rank-major `[size][words]`.
    Gather,
    /// Every member's input lands in member `owner`'s output only, rank
    /// major `[size][words]`; the other outputs are untouched. A step that
    /// runs several owner gathers over one state runs them in the same
    /// owner order on every member.
    OwnerGather { owner: u32 },
    /// Block d of every member's destination-major input `[size][words]`
    /// lands in member d's output, which then holds block d of every
    /// member, source-major `[size][words]`.
    Exchange,
    /// A group barrier: a `Gather` whose output no op reads, which ends on
    /// a member only once every member has entered it.
    Barrier,
}

/// A transfer between the members of a context-parallel group of `size`
/// ranks of `words` 32-bit words per member (per destination, for an
/// exchange), whatever the step's rows: every launch moves the whole
/// block, so the outputs' layout is the same at every step.
///
/// Its buffers are a `collective::GatherBuffers` over the group's members
/// only, in group-rank order: each member's output and the state regions
/// the op lists in its info (`OpInfo::state`), which serve this transfer
/// alone. A model keeps one state per call site and per block size. Which
/// member a launch sends to when, and in how many launches, are the
/// backend's choices: they move the same bits to the same places.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct GroupTransferOp {
    pub transfer: GroupTransfer,
    pub size: u32,
    pub words: u64,
}

impl GroupTransferOp {
    /// The bytes a member's input holds: its words, or one block per
    /// member for an exchange.
    pub fn input_bytes(&self) -> Option<u64> {
        let blocks = match self.transfer {
            GroupTransfer::Exchange => u64::from(self.size),
            _ => 1,
        };
        blocks.checked_mul(self.words)?.checked_mul(4)
    }

    /// The bytes a receiving member's output holds: one block per member.
    pub fn output_bytes(&self) -> Option<u64> {
        u64::from(self.size).checked_mul(self.words)?.checked_mul(4)
    }

    /// Whether group rank `member`'s output receives.
    pub fn receives(&self, member: u32) -> bool {
        match self.transfer {
            GroupTransfer::OwnerGather { owner } => member == owner,
            _ => member < self.size,
        }
    }
}

impl Validate for GroupTransferOp {
    fn validate(&self) -> Result<(), Refusal> {
        if self.size < 2 || self.words == 0 {
            return Err(Refusal(
                "a group transfer moves words between at least 2 members",
            ));
        }
        if let GroupTransfer::OwnerGather { owner } = self.transfer
            && owner >= self.size
        {
            return Err(Refusal("an owner gather targets a member of its group"));
        }
        if self.output_bytes().is_none() {
            return Err(Refusal("group transfer bytes overflow u64"));
        }
        Ok(())
    }
}

/// The buffers of a group transfer: a gather's, over the group's members.
pub type GroupTransferBuffers<T> = GatherBuffers<T>;

crate::scratch_record! {
    /// A partial merge's buffers by role: the group's F32 `partials`,
    /// `ranks` segments one after another in rank order, each `m
    /// [rank_rows][input_heads]`, then `l` (the same) and `o
    /// [rank_rows][input_heads][width]`, where row r's head h is pair `r *
    /// input_heads + head_offset + h` of each segment; the device count of
    /// live rows; and the BF16 rows `[row][heads][width]` it writes to
    /// `output`. The host lays the segments out as its group transfers
    /// leave them.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct PartialMergeBuffers {
        pub partials: value,
        pub rank_rows: scalar(u32),
        pub input_heads: scalar(u32),
        pub head_offset: scalar(u32),
        pub live_rows: value,
        pub output: value,
    }
}

crate::scratch_record! {
    /// A key list localization's buffers by role: the I32 `lists` `[row]
    /// [width]` of sequence positions, the device count of live rows and
    /// the I32 local lists (the same shape) it writes to `output`.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct LocalizeKeysBuffers {
        pub lists: value,
        pub live_rows: value,
        pub output: value,
    }
}

crate::scratch_record! {
    /// Paged partials' buffers by role: the BF16 query rows `q`
    /// (`[row][q_heads * kv_heads * head_dim]`), the cache of the rank's
    /// local pages, each item's I32 block table row of local pages, each
    /// row's I32 item (`token_batch`) and absolute position, the device
    /// count of live rows, and the F32 partials it writes: the running
    /// maximum `max` and denominator `den` (`[row][heads]`) and the
    /// numerators `num` (`[row][heads][head_dim]`).
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct PagedPartialsBuffers {
        pub q: value,
        pub cache: record(KvCache),
        pub block_tables: value,
        pub token_batch: value,
        pub positions: value,
        pub live_rows: value,
        pub max: value,
        pub den: value,
        pub num: value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_validate_at_two_and_four_ranks() {
        for ranks in [2, 4] {
            let merge = ContextParallelQuery::PartialMerge {
                ranks,
                heads: 16,
                width: 512,
                capacity: 256,
            };
            merge.validate().unwrap();
            assert_eq!(merge.list_width(), None);
            let topk = ContextParallelQuery::TopKMerge {
                ranks,
                group: 4,
                max_top_k: 2048,
                always_tail: true,
                capacity: 2048,
            };
            topk.validate().unwrap();
            assert_eq!(topk.list_width(), Some(2051));
            assert_ne!(
                merge.numerical_contract_id().unwrap(),
                topk.numerical_contract_id().unwrap()
            );
        }
        assert_eq!(
            ContextParallelQuery::PartialMerge {
                ranks: 1,
                heads: 16,
                width: 512,
                capacity: 256,
            }
            .validate(),
            Err(Refusal("invalid context-parallel partial merge geometry"))
        );
        assert_eq!(
            ContextParallelQuery::TopKMerge {
                ranks: 2,
                group: 4,
                max_top_k: 2050,
                always_tail: true,
                capacity: 1,
            }
            .validate(),
            Err(Refusal(
                "indexer top-k must be a positive multiple of the pool"
            ))
        );
    }

    #[test]
    fn key_localization_validates_from_two_ranks() {
        let localize = |ranks, page, width| ContextParallelQuery::LocalizeKeys {
            ranks,
            page,
            width,
            capacity: 2048,
        };
        for ranks in [2, 4, 8] {
            let query = localize(ranks, 64, 2051);
            query.validate().unwrap();
            assert_eq!(query.list_width(), Some(2051));
            assert_eq!(
                query.numerical_contract_id().unwrap(),
                NumericalContractId::new("sparse.context-parallel-localize-keys.v1".to_owned())
            );
        }
        // A group's size has no ceiling here: a backend's collectives
        // state theirs.
        localize(16, 64, 2051).validate().unwrap();
        for query in [
            localize(1, 64, 2051),
            localize(2, 0, 2051),
            localize(2, 64, 0),
            localize(2, 64, 1 << 31),
        ] {
            assert_eq!(
                query.validate(),
                Err(Refusal(
                    "invalid context-parallel key localization geometry"
                ))
            );
        }
    }

    #[test]
    fn paged_attention_partials_validate_at_a_gqa_rank_shape() {
        let partials = |head_dim, kv| ContextParallelQuery::PagedAttentionPartials {
            q_heads: 6,
            kv_heads: 1,
            head_dim,
            page: 64,
            kv,
            phase: AttentionPhase::Decode,
            capacity: 256,
        };
        for kv in [DType::Bf16, DType::Fp8E4M3] {
            let query = partials(256, kv);
            query.validate().unwrap();
            assert_eq!(query.list_width(), None);
            assert_eq!(
                query.numerical_contract_id().unwrap(),
                NumericalContractId::new(
                    "attention.paged-gqa-partials-position-order.v1".to_owned()
                )
            );
        }
        // The head width granule is a backend's.
        partials(250, DType::Bf16).validate().unwrap();
        assert_eq!(
            partials(0, DType::Bf16).validate(),
            Err(Refusal("invalid paged attention partial geometry"))
        );
        assert_eq!(
            partials(256, DType::F32).validate(),
            Err(Refusal("paged attention partials read BF16 or FP8 E4M3 KV"))
        );
    }

    #[test]
    fn selected_partials_validate_with_their_blocks() {
        let selected = |page, block, top_blocks| ContextParallelQuery::SelectedAttentionPartials {
            q_heads: 6,
            kv_heads: 1,
            head_dim: 256,
            page,
            block,
            top_blocks,
            kv: DType::Fp8E4M3,
            phase: AttentionPhase::Prefill,
            capacity: 8192,
        };
        let query = selected(64, 4, 512);
        query.validate().unwrap();
        assert_eq!(query.list_width(), None);
        assert_eq!(
            query.numerical_contract_id().unwrap(),
            NumericalContractId::new(
                "attention.selected-block-gqa-partials-position-order.v1".to_owned()
            )
        );
        for (page, block, top) in [(64, 0, 512), (64, 4, 0), (66, 4, 512)] {
            assert_eq!(
                selected(page, block, top).validate(),
                Err(Refusal("invalid selected attention partial blocks"))
            );
        }
    }

    #[test]
    fn owner_routed_writes_and_block_merges_validate() {
        let write = GatedAttentionQuery::QkNormRopeKvWrite {
            q_heads: 6,
            kv_heads: 1,
            head_dim: 256,
            rotary_dim: 64,
            theta_bits: 1e7_f32.to_bits(),
            mrope: [11, 11, 10],
            eps_bits: 1e-6_f32.to_bits(),
            page_size: 64,
            kv_dtype: DType::Bf16,
            capacity: 8192,
        };
        let kv = ContextParallelQuery::OwnedQkNormRopeKvWrite { write, ranks: 2 };
        kv.validate().unwrap();
        let key = SparseAttentionQuery::IndexerKeyWrite {
            head_dim: 128,
            rotary_dim: 64,
            theta_bits: 1e7_f32.to_bits(),
            mrope: [11, 11, 10],
            eps_bits: 1e-6_f32.to_bits(),
            block: 4,
            ring: 16,
            capacity: 8192,
        };
        let keys = ContextParallelQuery::OwnedIndexerKeyWrite {
            write: key,
            page: 64,
            ranks: 2,
        };
        keys.validate().unwrap();
        let blocks = ContextParallelQuery::TopKMergeBlocks {
            ranks: 2,
            group: 4,
            max_top_k: 2048,
            capacity: 2048,
        };
        blocks.validate().unwrap();
        let ids = [kv, keys, blocks].map(|q| q.numerical_contract_id().unwrap());
        assert!(ids[0] != ids[1] && ids[1] != ids[2] && ids[0] != ids[2]);
        assert_ne!(
            ids[2],
            merge(4, 2048, false).numerical_contract_id().unwrap()
        );
        assert!(
            ContextParallelQuery::OwnedQkNormRopeKvWrite { write, ranks: 1 }
                .validate()
                .is_err()
        );
        ContextParallelQuery::OwnedIndexerKeyWrite {
            write: key,
            page: 64,
            ranks: 9,
        }
        .validate()
        .unwrap();
        for (page, ranks) in [(66, 2), (64, 1)] {
            assert!(
                ContextParallelQuery::OwnedIndexerKeyWrite {
                    write: key,
                    page,
                    ranks
                }
                .validate()
                .is_err()
            );
        }
        let gate = GatedAttentionQuery::OutputGate {
            heads: 6,
            head_dim: 256,
            capacity: 1,
        };
        assert_eq!(
            ContextParallelQuery::OwnedQkNormRopeKvWrite {
                write: gate,
                ranks: 2
            }
            .validate(),
            Err(Refusal("invalid owner-routed KV write"))
        );
    }

    fn merge(group: u32, max_top_k: u32, always_tail: bool) -> ContextParallelQuery {
        ContextParallelQuery::TopKMerge {
            ranks: 2,
            group,
            max_top_k,
            always_tail,
            capacity: 1,
        }
    }

    #[test]
    fn key_list_width_is_representable_at_its_bounds() {
        // 2^31 - 4 plus a 3-token tail is i32::MAX; the next multiple of 4
        // is not representable.
        let largest = merge(4, 2_147_483_644, true);
        largest.validate().unwrap();
        assert_eq!(largest.list_width(), Some(i32::MAX as u32));
        let first_rejected = merge(4, 2_147_483_648, true);
        assert_eq!(
            first_rejected.validate(),
            Err(Refusal("indexer key list width exceeds the i32 maximum"))
        );
        assert_eq!(first_rejected.list_width(), None);
        // Without the tail the width is max_top_k itself.
        merge(4, 2_147_483_644, false).validate().unwrap();
        assert_eq!(
            merge(4, 2_147_483_648, false).validate(),
            Err(Refusal("indexer key list width exceeds the i32 maximum"))
        );
        // The reviewed overflow: group 3 is outside the pool domain, and the
        // width is never computed with wrapping.
        let wrapping = merge(3, u32::MAX, true);
        assert_eq!(
            wrapping.validate(),
            Err(Refusal(
                "indexer pools hold a power of two from 2 to 16 tokens"
            ))
        );
        assert_eq!(wrapping.list_width(), None);
        assert_eq!(merge(16, 4096, true).list_width(), Some(4111));
        assert!(merge(32, 4096, true).validate().is_err());
    }

    #[test]
    fn ops_validate_their_queries_and_contexts() {
        let partials = ContextParallelQuery::PagedAttentionPartials {
            q_heads: 6,
            kv_heads: 1,
            head_dim: 256,
            page: 64,
            kv: DType::Bf16,
            phase: AttentionPhase::Decode,
            capacity: 256,
        };
        let paged = |query, softmax_scale, max_context| PagedPartialsOp {
            query,
            softmax_scale,
            max_context,
        };
        paged(partials, 0.0625, 8192).validate().unwrap();
        let merge = ContextParallelQuery::PartialMerge {
            ranks: 2,
            heads: 6,
            width: 256,
            capacity: 256,
        };
        assert!(paged(merge, 0.0625, 8192).validate().is_err());
        for (scale, context) in [(0.0, 8192), (f32::NAN, 8192), (0.0625, 0)] {
            assert!(paged(partials, scale, context).validate().is_err());
        }
        let write = GatedAttentionQuery::QkNormRopeKvWrite {
            q_heads: 6,
            kv_heads: 1,
            head_dim: 256,
            rotary_dim: 64,
            theta_bits: 1e7_f32.to_bits(),
            mrope: [11, 11, 10],
            eps_bits: 1e-6_f32.to_bits(),
            page_size: 64,
            kv_dtype: DType::Bf16,
            capacity: 8192,
        };
        let kv = ContextParallelQuery::OwnedQkNormRopeKvWrite { write, ranks: 2 };
        OwnedKvWriteOp {
            query: kv,
            max_context: 8192,
        }
        .validate()
        .unwrap();
        for (query, max_context) in [(kv, 0), (merge, 8192)] {
            assert!(OwnedKvWriteOp { query, max_context }.validate().is_err());
        }
        let key = ContextParallelQuery::OwnedIndexerKeyWrite {
            write: SparseAttentionQuery::IndexerKeyWrite {
                head_dim: 128,
                rotary_dim: 64,
                theta_bits: 1e7_f32.to_bits(),
                mrope: [11, 11, 10],
                eps_bits: 1e-6_f32.to_bits(),
                block: 4,
                ring: 16,
                capacity: 8192,
            },
            page: 64,
            ranks: 2,
        };
        let keys = |query, index_heads, max_context| OwnedKeyWriteOp {
            query,
            index_heads,
            max_context,
        };
        keys(key, 4, 8192).validate().unwrap();
        for op in [keys(key, 0, 8192), keys(key, 4, 0), keys(kv, 4, 8192)] {
            assert!(op.validate().is_err());
        }
    }

    #[test]
    fn group_transfers_size_their_blocks() {
        let op = |transfer, size, words| GroupTransferOp {
            transfer,
            size,
            words,
        };
        let gather = op(GroupTransfer::Gather, 4, 1024);
        gather.validate().unwrap();
        assert_eq!(gather.input_bytes(), Some(4096));
        assert_eq!(gather.output_bytes(), Some(4 * 4096));
        assert!((0..4).all(|member| gather.receives(member)));
        let exchange = op(GroupTransfer::Exchange, 4, 1024);
        assert_eq!(exchange.input_bytes(), exchange.output_bytes());
        let owner = op(GroupTransfer::OwnerGather { owner: 1 }, 2, 8);
        owner.validate().unwrap();
        assert_eq!([0, 1].map(|member| owner.receives(member)), [false, true]);
        for refused in [
            op(GroupTransfer::Gather, 1, 1024),
            op(GroupTransfer::Barrier, 2, 0),
            op(GroupTransfer::OwnerGather { owner: 2 }, 2, 8),
            op(GroupTransfer::Gather, 4, u64::MAX / 4),
        ] {
            assert!(refused.validate().is_err(), "{refused:?}");
        }
    }

    #[test]
    fn records_name_their_roles() {
        use crate::allocation::Traversal;
        let merge = PartialMergeBuffers {
            partials: 1,
            rank_rows: 32,
            input_heads: 16,
            head_offset: 0,
            live_rows: 2,
            output: 3,
        };
        let mut roles = Vec::new();
        merge.visit_named(&mut |role, _| roles.push(role.to_owned()));
        assert_eq!(roles, ["partials", "live_rows", "output"]);
        assert_eq!(merge.map(|v| v * 10).rank_rows, 32);
        let paged = PagedPartialsBuffers {
            q: 1,
            cache: KvCache {
                k: 2,
                v: 3,
                slots: 4,
                k_slot_stride: 64,
                v_slot_stride: 64,
                k_scale: 1.0,
                v_scale: 1.0,
            },
            block_tables: 4,
            token_batch: 5,
            positions: 6,
            live_rows: 7,
            max: 8,
            den: 9,
            num: 10,
        };
        let mut roles = Vec::new();
        paged.visit_named(&mut |role, _| roles.push(role.to_owned()));
        assert_eq!(roles[1..3], ["cache.k", "cache.v"]);
        assert_eq!(roles.len(), 10);
    }
}
