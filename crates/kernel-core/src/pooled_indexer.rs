//! The pooled top-k indexer: per-token keys pooled into groups
//! (`gated_pooling`), scored by weighted ReLU heads, and the top pools
//! expanded back to the token key list a sparse attention reads
//! (`latent_attention::LatentAttentionQuery::Attention`).
//!
//! Visibility and the own-key rule. A row at sequence position p sees the
//! complete pools g with `(g + 1) * group - 1 <= p` and, when the tail is
//! always selected, the raw tokens of its incomplete pool, positions
//! `[(p + 1) / group * group, p]`. The row's own key is therefore attended
//! while it sits in the incomplete tail; the row that completes a pool sees
//! that pool only as a candidate, so its own key is attended only if the
//! pool wins a top-k place.
//!
//! The pooled BF16 indexer has no rotary embedding; its query is the raw projection.
//! The position variant uses [`PositionIndexerQuery`]: compressed positions,
//! tail RoPE, MXFP4 operands and an optional candidate mask.
//! A row's key list is consumed unchanged by later layers that share the
//! selection (cross-layer reuse), and draft steps after the first reuse the
//! draft pass's selection through `CompactSelection`. Reuse is a bundle
//! program property; the list format does not change.
//!
//! Math and backend choices. Every field of a query is math the model
//! relies on or a layout it shares with its attention (the pooling, the
//! key norm's eps, the page size, the key format, the phase's tail state
//! rules, the score arithmetic, top-k with its tie rule, the key list
//! order and width, the per-rank mode), except `capacity`: the rows a
//! backend sizes its launches for, which a model states as
//! `backend::Need::capacity` and a backend may serve from an entry
//! prepared for more rows. Score splits, their merge order, tiles, threads
//! and workspace are a backend's choice: the selection is exact, so they
//! change no output bit. The same holds for the kernels a backend runs a
//! selection with at a given row count: one kernel that scores and keeps a
//! row's pools at one split, or split kernels whose lists a streaming,
//! bitonic-tree or radix-select merge folds, write equal key lists and
//! pairs by construction. The runtime top-k (a model option, a multiple of
//! the pool of at most `max_top_k`), a per-rank selection's group rank and
//! a key write's group rank and size are facts of the instance a backend
//! prepares for.
//!
//! The ops read and write the buffers below by role (`crate::backend`'s
//! pooled indexer methods). Every launch counts its live rows on the device
//! (`live_rows`); a row carries its item (`row_batch`, the item's page
//! table row and tail state) and its sequence position (`positions`), and
//! one item's rows are contiguous with consecutive positions.

use crate::attention::BlockTables;
use crate::contract::{
    NumericalContract, NumericalContractId, RecurrentPhase, Refusal, Validate, valid_eps,
    valid_phase,
};
use crate::gated_pooling::GatedGroupPooling;

/// Largest key list width: the lists hold i32 positions and a row's entries
/// are indexed as i32.
pub const MAX_KEY_LIST_WIDTH: u32 = i32::MAX as u32;

/// The width of a key list row, `max_top_k` plus `group - 1` with the
/// always-selected tail, or `None` above `MAX_KEY_LIST_WIDTH`.
pub(crate) fn key_list_width(group: u32, max_top_k: u32, always_tail: bool) -> Option<u32> {
    let tail = if always_tail {
        group.checked_sub(1)?
    } else {
        0
    };
    max_top_k
        .checked_add(tail)
        .filter(|&width| width <= MAX_KEY_LIST_WIDTH)
}

/// The selection shape shared by `PooledIndexerQuery::Select` and
/// `context_parallel::ContextParallelQuery::TopKMerge`: pools of a power of
/// two from 2 to 16 tokens (a pool never straddles a power-of-two page), a
/// top-k of whole pools, and a representable key list width.
pub(crate) fn validate_selection_shape(
    group: u32,
    max_top_k: u32,
    always_tail: bool,
) -> Result<(), Refusal> {
    if !(2..=16).contains(&group) || !group.is_power_of_two() {
        return Err(Refusal(
            "indexer pools hold a power of two from 2 to 16 tokens",
        ));
    }
    if max_top_k == 0 || !max_top_k.is_multiple_of(group) {
        return Err(Refusal(
            "indexer top-k must be a positive multiple of the pool",
        ));
    }
    if key_list_width(group, max_top_k, always_tail).is_none() {
        return Err(Refusal("indexer key list width exceeds the i32 maximum"));
    }
    Ok(())
}

/// What one candidate of the selection is.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum IndexerGranularity {
    /// Complete pools of `group` tokens.
    Pool,
    /// Every compressed position scored on its own, with `group = 1`.
    /// The enclosing `PositionIndexerQuery` supplies the compression ratio.
    Position,
}

/// Storage of the pooled keys.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum IndexerKeyFormat {
    /// `[page_size / group][head_dim]` BF16 per page.
    Bf16,
    /// MXFP4 position keys: E2M1 codes, low nibble first, and one UE8M0
    /// scale per 32 channels. The position query defines the trained
    /// quantizer. Pooled key writes and pool selection still refuse this.
    Fp4,
}

/// Where a row's candidates come from and what a selection entry writes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum IndexerSelectionMode {
    /// All of the row's candidates; writes the row's key list.
    Global,
    /// Context parallel: group rank r scores only the pools in the KV pages
    /// it owns (physical page j of a sequence on group rank `j % ranks`, in
    /// its local page of stripe `j / ranks`; docs/serving-kv-pool.md
    /// "Context parallelism") and writes its own top-k as (score, pool)
    /// pairs;
    /// `context_parallel::ContextParallelQuery::TopKMerge` merges the ranks'
    /// lists into exactly the `Global` key list.
    PerRank { ranks: u32 },
}

/// Candidate restriction of a position selection. Visibility always applies.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum IndexerCandidates {
    All,
    /// A byte mask per row and compressed position: 0 excludes, 1 admits.
    /// A missing mask is an error, never an all-candidates fallback.
    Mask,
}

/// Position indexer preparation and selection over compressed keys.
///
/// Queries are BF16 projections `[rows, heads, head_dim]`. Keys are BF16
/// projections of the unrotated compressor output. RMSNorm applies to each
/// key, with BF16 weights, F32 accumulation and `key_norm_eps_bits`, then
/// rounds once to BF16. No key bias or mean subtraction is used.
///
/// Rotate interleaved pairs in the last `tail_rope_dims` of q and k, in
/// place, using the supplied F32 cos/sin table. q uses its token position t;
/// compressed key j uses position `j * compression_ratio`. The unrotated
/// prefix stays unchanged. Round rotated values to BF16 before quantization.
/// Both operands use E2M1 RNE, clamped to [-6, 6], in 32-channel blocks:
/// `s = 2^ceil_log2(f32(max(amax, 6*2^-126) * f32(1/6)))`, where amax sees
/// BF16-rounded values. Keys store MXFP4 codes and UE8M0 scales; queries
/// store the same quantized values decoded to BF16. Non-finite operands,
/// scale byte 255 and code/scale products not finite and exact in BF16 are
/// refused. Large scale bytes are not blanket-admitted.
///
/// A token query t sees precisely `j < (t + 1) / compression_ratio`. A
/// supplied mask intersects that set. Scores and their order follow
/// `selection()`, independent of rank count. Select over the kernel's own
/// F32 scores exactly; bound those scores against f64 separately. Agreement
/// with a BF16-score implementation is measured, not an exactness gate.
/// Output lists contain compressed indices ascending, then -1 padding.
/// An attention-source offset is applied by the consumer, after selection.
///
/// The model owns cross-layer reuse: a key owner republishes its cache on
/// every step, including steps with no newly completed compression group.
/// Reindex reads that cache and the source's candidate mask but recomputes
/// scores with its own queries and weights. Neither operation retains an
/// implicit pointer or mask from the previous launch.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PositionIndexerQuery {
    pub heads: u32,
    pub head_dim: u32,
    pub compression_ratio: u32,
    pub tail_rope_dims: u32,
    pub key_norm_eps_bits: u32,
    pub max_top_k: u32,
    /// Compressed positions per page.
    pub page_size: u32,
    pub candidates: IndexerCandidates,
    pub capacity: u32,
}

impl PositionIndexerQuery {
    pub fn selection(&self) -> PooledIndexerQuery {
        PooledIndexerQuery::Select {
            heads: self.heads,
            head_dim: self.head_dim,
            group: 1,
            max_top_k: self.max_top_k,
            always_tail: false,
            granularity: IndexerGranularity::Position,
            key_format: IndexerKeyFormat::Fp4,
            mode: IndexerSelectionMode::Global,
            page_size: self.page_size,
            capacity: self.capacity,
        }
    }

    /// Completed compressed positions, or None for an unrepresentable count
    /// or an invalid zero compression ratio.
    pub fn visible_positions(&self, position: u64) -> Option<u64> {
        position
            .checked_add(1)?
            .checked_div(u64::from(self.compression_ratio))
    }

    /// Packed E2M1 bytes and UE8M0 bytes per page, in separate planes.
    pub fn page_bytes(&self) -> (u64, u64) {
        let elements = u64::from(self.page_size) * u64::from(self.head_dim);
        (elements / 2, elements / 32)
    }
}

impl Validate for PositionIndexerQuery {
    fn validate(&self) -> Result<(), Refusal> {
        self.selection().validate()?;
        if self.compression_ratio == 0
            || self.tail_rope_dims == 0
            || self.tail_rope_dims > self.head_dim
            || !self.tail_rope_dims.is_multiple_of(2)
            || !valid_eps(self.key_norm_eps_bits)
        {
            return Err(Refusal(
                "invalid position indexer compression, RoPE or RMSNorm",
            ));
        }
        Ok(())
    }
}

impl NumericalContract for PositionIndexerQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        let mask = match self.candidates {
            IndexerCandidates::All => "all",
            IndexerCandidates::Mask => "masked",
        };
        Ok(NumericalContractId::new(format!(
            "sparse.position-indexer-rmsnorm-tail-rope-mxfp4-weight-scale-f32-{mask}.v1"
        )))
    }
}

/// First-level candidate filter over a row's own F32 position scores.
///
/// Partition compressed positions into blocks of `block_size` (8). A
/// block's score is the maximum visible position score; pad with -inf.
/// Replace the score of `(visible - 1) / block_size` with +inf when visible
/// is nonzero. This pins the newest block, partial or full, within the
/// `top_k_blocks` budget (2048), never in addition to it. Select top blocks
/// by score, ties to the lower block index, then discard scores <= -inf.
/// Expand each kept block to a byte mask over the stored position width.
/// This can mark invisible positions in its newest block; the second-level
/// selection must still apply visibility. At visible = 0 the mask is zero.
/// NaNs and input +inf are refused; +inf is reserved for pinning. Signed
/// zero scores tie. Filtering is exact over the
/// supplied F32 scores, with no BF16 conversion or f64-score recomputation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CandidateBlocksQuery {
    pub block_size: u32,
    pub top_k_blocks: u32,
    pub capacity: u32,
}

impl Validate for CandidateBlocksQuery {
    fn validate(&self) -> Result<(), Refusal> {
        if self.block_size == 0 || self.top_k_blocks == 0 || self.capacity == 0 {
            return Err(Refusal("invalid candidate block geometry"));
        }
        Ok(())
    }
}

impl NumericalContract for CandidateBlocksQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        Ok(NumericalContractId::new(
            "sparse.candidate-block-max-f32-topk-lower-index-pin-newest.v1".to_owned(),
        ))
    }
}

/// Pooled indexer contracts.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PooledIndexerQuery {
    /// Normalize each row's raw key `[rows, width]` BF16 (the key projection
    /// output) with a LayerNorm with BF16 weight and bias: `mean = sum(x) /
    /// width`, `var = sum((x - mean)^2) / width` (two passes, F32), `y = (x -
    /// mean) * rsqrt(var + eps) * w + b` in F32 with the correctly rounded F32
    /// reciprocal square root, rounded once to BF16. eps is 1e-6 for the
    /// served indexer.
    ///
    /// A row in an incomplete group writes its normalized key and BF16 gate
    /// logits `[rows, width]` to tail slot `position % group` of the
    /// sequence's tail state (`GatedGroupPooling::tail_slot`,
    /// `tail_slot_offsets`, `tail_state_elements`). The row whose position
    /// completes group g writes no tail slot: it pools the group
    /// (`gated_pooling`) from its own key and gate logits plus the `group -
    /// 1` earlier tokens, read from this launch's rows or the live tail
    /// slots, writes the pooled key to the page holding token `g * group` at
    /// row `(g * group % page_size) / group`, and leaves the tail empty
    /// (no live slots).
    ///
    /// The tail state follows `phase` like a recurrent state. A prefill or
    /// decode launch leaves the state after its last row. A verify launch
    /// writes one checkpoint per row, each a copy of the whole tail state
    /// after that row: live slots `0..(position + 1) % group`, empty after
    /// a completing row. Commit keeps row `kept - 1`'s checkpoint. Dead
    /// slots are never read. Pooled rows written for rejected positions lie
    /// beyond the sequence length and are rewritten before any row can see
    /// them.
    ///
    /// In a context-parallel group of `size` ranks (the group rank and size
    /// a backend prepares the write for), physical page j of a sequence
    /// lives on group rank `j % size` in its local page of stripe `j /
    /// size`, as for `IndexerSelectionMode::PerRank`: group rank r writes
    /// only the pooled keys of its own pages, through its own page table,
    /// and every rank writes the whole tail state. A group of one rank owns
    /// every page.
    KeyWrite {
        pooling: GatedGroupPooling,
        key_norm_eps_bits: u32,
        page_size: u32,
        key_format: IndexerKeyFormat,
        phase: RecurrentPhase,
        capacity: u32,
    },
    /// Score and select. Inputs: queries `[rows, heads, head_dim]` BF16,
    /// head weights `[rows, heads]` BF16 (the weight projection output) and
    /// the pooled keys through the page table. For a visible pool g:
    ///
    /// - `r_h = relu(f32(q_h . K_g) * head_dim^-0.5)`: the dot product in
    ///   F32 over 128-wide chains from zero added in order, the scale an F32
    ///   multiply, `relu(x) = x > 0 ? x : +0`;
    /// - `w_h = f32(weight_h) * heads^-0.5` (F32 multiply);
    /// - `score_g = sum_h w_h * r_h`, F32 FMA from zero in head order.
    ///
    /// `Position` instead consumes RoPE'd MXFP4 queries (fake-quantized,
    /// decoded to BF16) and keys, `group = 1`, no tail, and `Global` mode.
    /// Its weights are BF16 projection outputs. The dot is unscaled:
    /// `r_h = relu(f32(q_h . k_j))`, with 128-wide chains as above;
    /// `w_h = f32(weight_h * f32(head_dim^-0.5 * heads^-0.5))` and
    /// `score_j = sum_h w_h * r_h`, F32 FMA in head order. The scale is
    /// applied to the weights after the dot, never to q or the dot.
    /// Positions and lists here are compressed coordinates; the enclosing
    /// `PositionIndexerQuery` defines token visibility and candidate masks.
    ///
    /// With the runtime `top_k` (a multiple of `group`, at most
    /// `max_top_k`; 2048 for the served indexer), `k = top_k / group`
    /// pools are kept: every visible pool when there are at most k,
    /// otherwise the exact top k of the F32 scores with ties to the lower
    /// pool index. Scores compare in IEEE order with -0 equal to +0 and
    /// every NaN below -inf; a NaN-scored pool stays a candidate. The engine
    /// streams the selection and keeps no `[rows, pools]` score buffer.
    ///
    /// `Global` writes the key list `[rows, list_width()]` i32: the kept
    /// pools in ascending pool order, each as its `group` token positions
    /// ascending, then (with `always_tail`) the incomplete tail's positions
    /// ascending, then -1 to the end. `PerRank` writes `[rows, max_top_k /
    /// group]` pairs of F32 score and i32 pool, ordered by score descending
    /// then pool ascending, padded with `(-inf, -1)`; a pair carries +0 for
    /// either zero and `0xffffffff` for a NaN. Per-rank scores use exactly
    /// the `Global` arithmetic, so the merge is exact.
    Select {
        heads: u32,
        head_dim: u32,
        group: u32,
        max_top_k: u32,
        always_tail: bool,
        granularity: IndexerGranularity,
        key_format: IndexerKeyFormat,
        mode: IndexerSelectionMode,
        page_size: u32,
        capacity: u32,
    },
    /// Score and select as `Select`, with the block-sparse indexer's
    /// unweighted head sum (`contract::SparseAttentionQuery::BlockScores`)
    /// for the scores. Inputs: queries `[rows, heads, head_dim]` BF16 and the
    /// pooled BF16 keys through the page table; no head weights. For a
    /// visible pool g and head h, with k16 step s over dimensions `16 s..16
    /// s + 16`:
    ///
    /// - `e_h` and `o_h`: F32 chains from +0 over the even and the odd k16
    ///   steps in step order, each step adding its 16 products (one BF16 MMA
    ///   k16 step);
    /// - `r_h = relu(e_h + o_h)`, an F32 add, `relu(x) = x > 0 ? x : +0`;
    /// - the head sum as an F32 pairwise tree: `(r_0 + r_1) + (r_2 + r_3)` at
    ///   four heads, the four-head sums of heads 0..4 and 4..8 added at
    ///   eight, and at sixteen the eight-head tree over `r_h + r_(h + 8)`;
    /// - `score_g = sum * f32(head_dim^-0.5)`, an F32 multiply by the F32
    ///   nearest to `head_dim^-0.5`.
    ///
    /// Selection, outputs, per-rank pairs and the exact merge
    /// (`context_parallel::ContextParallelQuery::TopKMerge`) are those of
    /// `Select`; scores are nonnegative or +0, so ranks order as the block
    /// selection's raw F32 bits.
    HeadSumSelect {
        heads: u32,
        head_dim: u32,
        group: u32,
        max_top_k: u32,
        always_tail: bool,
        mode: IndexerSelectionMode,
        page_size: u32,
        capacity: u32,
    },
    /// `out[i] = list[source[i]]` for i32 key lists `[rows, width]` with a
    /// launch-supplied source row per output row: the reuse of one pass's
    /// selection by later draft steps.
    CompactSelection { width: u32, capacity: u32 },
}

impl PooledIndexerQuery {
    /// Entries of a `Select` row's key list: `max_top_k`, plus `group - 1`
    /// with the always-selected tail (2051 for top 2048 in groups of 4).
    /// `None` for other entries and for a width above
    /// `MAX_KEY_LIST_WIDTH`, which validation refuses.
    pub fn list_width(&self) -> Option<u32> {
        match *self {
            Self::Select {
                group,
                max_top_k,
                always_tail,
                ..
            }
            | Self::HeadSumSelect {
                group,
                max_top_k,
                always_tail,
                ..
            } => key_list_width(group, max_top_k, always_tail),
            _ => None,
        }
    }

    /// BF16 bytes of the pooled keys in one KV page.
    pub fn pooled_page_bytes(&self) -> Option<u64> {
        match *self {
            Self::KeyWrite {
                pooling, page_size, ..
            } => Some(u64::from(page_size / pooling.group) * u64::from(pooling.width) * 2),
            _ => None,
        }
    }
}

impl Validate for PooledIndexerQuery {
    fn validate(&self) -> Result<(), Refusal> {
        let valid_page = |page_size: u32, group: u32| {
            page_size.is_power_of_two() && page_size >= group && page_size.is_multiple_of(group)
        };
        let refuse_fp4 = |format: IndexerKeyFormat| match format {
            IndexerKeyFormat::Bf16 => Ok(()),
            IndexerKeyFormat::Fp4 => {
                Err(Refusal("FP4 indexer keys are declared but not supported"))
            }
        };
        match *self {
            Self::KeyWrite {
                pooling,
                key_norm_eps_bits,
                page_size,
                key_format,
                phase,
                capacity,
            } => {
                pooling.validate()?;
                if !valid_eps(key_norm_eps_bits)
                    || !valid_page(page_size, pooling.group)
                    || !valid_phase(phase)
                    || capacity == 0
                {
                    return Err(Refusal("invalid indexer key write geometry or phase"));
                }
                refuse_fp4(key_format)
            }
            Self::Select {
                heads,
                head_dim,
                group,
                max_top_k,
                always_tail,
                granularity,
                key_format,
                mode,
                page_size,
                capacity,
            } => {
                if heads == 0 || head_dim == 0 || capacity == 0 {
                    return Err(Refusal("invalid indexer selection geometry"));
                }
                if granularity == IndexerGranularity::Position {
                    if key_format != IndexerKeyFormat::Fp4 {
                        return Err(Refusal("position indexer requires MXFP4 keys"));
                    }
                    if group != 1 || always_tail || mode != IndexerSelectionMode::Global {
                        return Err(Refusal(
                            "position indexer requires group 1, no tail and global selection",
                        ));
                    }
                    if !head_dim.is_multiple_of(32)
                        || max_top_k == 0
                        || key_list_width(group, max_top_k, false).is_none()
                    {
                        return Err(Refusal("invalid position indexer width or top-k"));
                    }
                } else {
                    validate_selection_shape(group, max_top_k, always_tail)?;
                    refuse_fp4(key_format)?;
                }
                if !valid_page(page_size, group) {
                    return Err(Refusal("invalid indexer selection geometry"));
                }
                if let IndexerSelectionMode::PerRank { ranks } = mode
                    && ranks < 2
                {
                    return Err(Refusal("per-rank indexer selection needs at least 2 ranks"));
                }
                Ok(())
            }
            Self::HeadSumSelect {
                heads,
                head_dim,
                group,
                max_top_k,
                always_tail,
                mode,
                page_size,
                capacity,
            } => {
                // The head tree is defined at 4, 8 and 16 heads, the chains
                // over whole k16 steps.
                if !matches!(heads, 4 | 8 | 16)
                    || head_dim == 0
                    || !head_dim.is_multiple_of(16)
                    || capacity == 0
                {
                    return Err(Refusal("invalid head-sum indexer selection geometry"));
                }
                Self::Select {
                    heads,
                    head_dim,
                    group,
                    max_top_k,
                    always_tail,
                    granularity: IndexerGranularity::Pool,
                    key_format: IndexerKeyFormat::Bf16,
                    mode,
                    page_size,
                    capacity,
                }
                .validate()
            }
            Self::CompactSelection { width, capacity } if width == 0 || capacity == 0 => {
                Err(Refusal("invalid selection compaction geometry"))
            }
            Self::CompactSelection { .. } => Ok(()),
        }
    }
}

impl NumericalContract for PooledIndexerQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        let name = match *self {
            Self::KeyWrite { phase, .. } => format!(
                "sparse.pooled-indexer-key-layernorm-gated-pool-write.{}.v1",
                match phase {
                    RecurrentPhase::Prefill => "prefill",
                    RecurrentPhase::Decode => "decode",
                    RecurrentPhase::Verify { .. } => "verify",
                }
            ),
            Self::Select {
                granularity: IndexerGranularity::Position,
                ..
            } => {
                "sparse.position-indexer-mxfp4-relu-weight-scale-f32-topk-lower-index.v1".to_owned()
            }
            Self::Select {
                always_tail, mode, ..
            } => {
                let tail = if always_tail { "-tail" } else { "" };
                let mode = match mode {
                    IndexerSelectionMode::Global => "",
                    IndexerSelectionMode::PerRank { .. } => ".per-rank",
                };
                format!("sparse.pooled-indexer-relu-weighted-topk-lower-index{tail}{mode}.v1")
            }
            Self::HeadSumSelect {
                always_tail, mode, ..
            } => {
                let tail = if always_tail { "-tail" } else { "" };
                let mode = match mode {
                    IndexerSelectionMode::Global => "",
                    IndexerSelectionMode::PerRank { .. } => ".per-rank",
                };
                format!(
                    "sparse.pooled-indexer-relu-head-sum-two-chain-topk-lower-index{tail}{mode}.v1"
                )
            }
            Self::CompactSelection { .. } => "sparse.selection-row-gather.v1".to_owned(),
        };
        Ok(NumericalContractId::new(name))
    }
}

crate::scratch_record! {
    /// MXFP4 position keys with independent code and scale page strides in
    /// bytes. Each page holds `[page_size][head_dim / 2]` code bytes and
    /// `[page_size][head_dim / 32]` scale bytes. Page tables address compressed
    /// positions. Page strides and their address products are 64-bit.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct PositionKeyPages {
        pub codes: value,
        pub scales: value,
        pub pages: scalar(u32),
        pub code_page_stride: scalar(u64),
        pub scale_page_stride: scalar(u64),
        pub page_tables: record(BlockTables),
    }
}

crate::scratch_record! {
    /// Key preparation: BF16 projected keys `[rows][head_dim]` and RMSNorm
    /// weight `[head_dim]`; F32 cos/sin `[tokens][tail_rope_dims / 2][2]`;
    /// I32 compressed key positions and item ids. Negative positions are
    /// holes and write nothing. Only live rows write their addressed page
    /// slots. No pooling or tail state is read here.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct PositionKeyWriteBuffers {
        pub key: value,
        pub norm_weight: value,
        pub cos_sin: value,
        pub positions: value,
        pub row_batch: value,
        pub live_rows: value,
        pub keys: record(PositionKeyPages),
    }
}

crate::scratch_record! {
    /// Query preparation: BF16 q `[rows][heads][head_dim]`, F32 cos/sin as
    /// in the key write, I32 token positions and device live count. Output
    /// has q's shape and holds the MXFP4 fake-quantized values as BF16.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct PositionQueryBuffers {
        pub q: value,
        pub cos_sin: value,
        pub positions: value,
        pub live_rows: value,
        pub output: value,
    }
}

crate::scratch_record! {
    /// Score/select: prepared BF16 q, BF16 head weights, MXFP4 pages,
    /// I32 row item and token position, and device live count. Mask mode
    /// requires `candidate_mask` bytes `[rows][position_capacity]`; All
    /// requires no mask. `scores` is F32 with that same shape: visible,
    /// admitted positions get their scores, excluded positions get -inf.
    /// This output is the exact selection oracle's input. The candidate
    /// source scores All positions, then passes these scores to the block
    /// filter. `output` is I32 `[rows][max_top_k]`, ascending with -1 padding.
    /// No buffers above the live rows are read or written.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct PositionSelectBuffers {
        pub q: value,
        pub head_weights: value,
        pub keys: record(PositionKeyPages),
        pub positions: value,
        pub row_batch: value,
        pub live_rows: value,
        pub candidate_mask: option,
        pub position_capacity: scalar(u32),
        pub scores: value,
        pub output: value,
    }
}

crate::scratch_record! {
    /// Candidate filter: F32 scores `[rows][position_capacity]`, I32 visible
    /// compressed counts `[rows]`, the device live count and the output byte
    /// mask with the score shape. The filter writes every mask byte of each
    /// live row, including an all-zero row when no position is visible.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct CandidateBlocksBuffers {
        pub scores: value,
        pub visible_positions: value,
        pub position_capacity: scalar(u32),
        pub live_rows: value,
        pub output: value,
    }
}

crate::scratch_record! {
    /// Pooled keys in the KV pages, through a page table. Page j's pooled
    /// keys start `j * page_stride` bytes into `keys` and hold
    /// `[page_size / group][width]` BF16 (`PooledIndexerQuery::
    /// pooled_page_bytes`); `keys` holds `pages` pages. Item b's pages are
    /// row b of `page_tables`, one row per item; under context parallelism
    /// each rank's row lists its own local page per stripe
    /// (`IndexerSelectionMode::PerRank`).
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct PooledKeyPages {
        pub keys: value,
        pub pages: scalar(u32),
        pub page_stride: scalar(u64),
        pub page_tables: record(BlockTables),
    }
}

crate::scratch_record! {
    /// The per-sequence tail states of a key write (`PooledIndexerQuery::
    /// KeyWrite`), state the model owns. `state` holds `entries` packed
    /// entries of `GatedGroupPooling::tail_state_elements` BF16 each: the
    /// keys bank `[group - 1][width]`, then the gate logit bank, slot s of
    /// each at `GatedGroupPooling::tail_slot_offsets(s)`. Item b's state is
    /// entry `slot_ids[b]` (I32 `[items]`). A prefill or decode launch
    /// updates it in place; a verify launch of `max_rows` rows per item
    /// reads it and writes the whole state after its row r of item b to
    /// entry `checkpoint_ids[b * max_rows + r]` (I32 `[items][max_rows]`),
    /// which never aliases a source entry. Prefill and decode ignore
    /// checkpoint ids.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct IndexerTailState {
        pub state: value,
        pub entries: scalar(u32),
        pub slot_ids: value,
        pub checkpoint_ids: option,
    }
}

crate::scratch_record! {
    /// A key write's buffers by role: the raw BF16 key rows `key` and gate
    /// logit rows `gate`, `width` used of each row of `key_stride` and
    /// `gate_stride` elements; the LayerNorm's BF16 `norm_weight` and
    /// `norm_bias` `[width]`; the position embedding `position`
    /// `[group][width]` in the pooling's position dtype; each row's I32
    /// item and position; the device count of live rows; the tail states;
    /// and the pooled key pages it writes.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct IndexerKeyWriteBuffers {
        pub key: value,
        pub key_stride: scalar(u32),
        pub gate: value,
        pub gate_stride: scalar(u32),
        pub norm_weight: value,
        pub norm_bias: value,
        pub position: value,
        pub row_batch: value,
        pub positions: value,
        pub live_rows: value,
        pub tail: record(IndexerTailState),
        pub pooled: record(PooledKeyPages),
    }
}

crate::scratch_record! {
    /// A selection's buffers by role (`Select`, `HeadSumSelect`): the BF16
    /// queries `q` `[rows][heads][head_dim]`, a weighted selection's BF16
    /// `head_weights` `[rows][heads]` (a head-sum selection reads none and
    /// ignores a supplied one),
    /// the pooled key pages, each row's I32 item and position, the device
    /// count of live rows, the backend's `workspace` (sized by the op) and
    /// the `output`: the I32 key list `[rows][list_width]` of a `Global`
    /// selection, or the U32 (F32 score bits, pool) pairs `[rows][max_top_k
    /// / group][2]` of a `PerRank` one.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct IndexerSelectBuffers {
        pub q: value,
        pub head_weights: option,
        pub pooled: record(PooledKeyPages),
        pub row_batch: value,
        pub positions: value,
        pub live_rows: value,
        pub workspace: value,
        pub output: value,
    }
}

crate::scratch_record! {
    /// An exact merge's buffers by role
    /// (`context_parallel::ContextParallelQuery::TopKMerge` and
    /// `TopKMergeBlocks`): the ranks' U32 pairs gathered rank-major
    /// `[ranks][rank_rows][max_top_k / group][2]`, each rank's rows
    /// starting `rank_rows` rows after the last; each row's I32 position,
    /// which a key list merge reads for the tail (a block merge ignores
    /// them); the device count of live rows; and the `output`: the I32 key
    /// list `[rows][list_width]` or block ids `[rows][max_top_k / group]`.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct IndexerMergeBuffers {
        pub pairs: value,
        pub rank_rows: scalar(u32),
        pub positions: option,
        pub live_rows: value,
        pub output: value,
    }
}

crate::scratch_record! {
    /// A selection compaction's buffers by role (`CompactSelection`): the
    /// I32 key lists `lists` `[_][width]`, each output row's I32 source row
    /// `source`, the device count of live output rows and the compacted
    /// lists `output` `[rows][width]`.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct SelectionCompactBuffers {
        pub lists: value,
        pub source: value,
        pub live_rows: value,
        pub output: value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::DType;

    fn select(mode: IndexerSelectionMode) -> PooledIndexerQuery {
        PooledIndexerQuery::Select {
            heads: 32,
            head_dim: 128,
            group: 4,
            max_top_k: 2048,
            always_tail: true,
            granularity: IndexerGranularity::Pool,
            key_format: IndexerKeyFormat::Bf16,
            mode,
            page_size: 64,
            capacity: 2048,
        }
    }

    fn key_write(phase: RecurrentPhase) -> PooledIndexerQuery {
        PooledIndexerQuery::KeyWrite {
            pooling: GatedGroupPooling {
                group: 4,
                width: 128,
                position_dtype: DType::F32,
            },
            key_norm_eps_bits: 1e-6_f32.to_bits(),
            page_size: 64,
            key_format: IndexerKeyFormat::Bf16,
            phase,
            capacity: 8192,
        }
    }

    #[test]
    fn served_indexer_validates_with_exact_widths() {
        let global = select(IndexerSelectionMode::Global);
        global.validate().unwrap();
        assert_eq!(global.list_width(), Some(2051));
        let per_rank = select(IndexerSelectionMode::PerRank { ranks: 4 });
        per_rank.validate().unwrap();
        assert_ne!(
            global.numerical_contract_id().unwrap(),
            per_rank.numerical_contract_id().unwrap()
        );
        for phase in [
            RecurrentPhase::Prefill,
            RecurrentPhase::Decode,
            RecurrentPhase::Verify { max_rows: 4 },
        ] {
            let write = key_write(phase);
            write.validate().unwrap();
            assert_eq!(write.pooled_page_bytes(), Some(16 * 128 * 2));
            write.numerical_contract_id().unwrap();
        }
        let compact = PooledIndexerQuery::CompactSelection {
            width: 2051,
            capacity: 256,
        };
        compact.validate().unwrap();
        assert_eq!(compact.list_width(), None);
    }

    #[test]
    fn head_sum_selection_validates_with_its_own_contract() {
        let with = |heads, mode| PooledIndexerQuery::HeadSumSelect {
            heads,
            head_dim: 128,
            group: 4,
            max_top_k: 2048,
            always_tail: true,
            mode,
            page_size: 64,
            capacity: 2048,
        };
        let global = with(4, IndexerSelectionMode::Global);
        global.validate().unwrap();
        assert_eq!(global.list_width(), Some(2051));
        let per_rank = with(4, IndexerSelectionMode::PerRank { ranks: 2 });
        let ids = [
            global.numerical_contract_id().unwrap(),
            per_rank.numerical_contract_id().unwrap(),
            select(IndexerSelectionMode::Global)
                .numerical_contract_id()
                .unwrap(),
        ];
        assert!(ids[0] != ids[1] && ids[0] != ids[2] && ids[1] != ids[2]);
        with(16, IndexerSelectionMode::Global).validate().unwrap();
        for heads in [0, 2, 12, 32] {
            assert_eq!(
                with(heads, IndexerSelectionMode::Global).validate(),
                Err(Refusal("invalid head-sum indexer selection geometry"))
            );
        }
        // A rank ceiling is a backend's limit, not the contract's.
        with(4, IndexerSelectionMode::PerRank { ranks: 9 })
            .validate()
            .unwrap();
    }

    /// Head counts and widths a backend may not build are its refusals,
    /// not the contract's.
    #[test]
    fn selections_check_their_math_only() {
        let PooledIndexerQuery::Select {
            group,
            max_top_k,
            always_tail,
            granularity,
            key_format,
            mode,
            page_size,
            capacity,
            ..
        } = select(IndexerSelectionMode::Global)
        else {
            unreachable!()
        };
        let with = |heads, head_dim| PooledIndexerQuery::Select {
            heads,
            head_dim,
            group,
            max_top_k,
            always_tail,
            granularity,
            key_format,
            mode,
            page_size,
            capacity,
        };
        for (heads, head_dim) in [(80, 128), (24, 192), (1, 8)] {
            with(heads, head_dim).validate().unwrap();
        }
        for (heads, head_dim) in [(0, 128), (32, 0)] {
            assert_eq!(
                with(heads, head_dim).validate(),
                Err(Refusal("invalid indexer selection geometry"))
            );
        }
        let head_sum = |head_dim| PooledIndexerQuery::HeadSumSelect {
            heads: 8,
            head_dim,
            group,
            max_top_k,
            always_tail,
            mode,
            page_size,
            capacity,
        };
        for head_dim in [16, 192, 512] {
            head_sum(head_dim).validate().unwrap();
        }
        for head_dim in [0, 24] {
            assert_eq!(
                head_sum(head_dim).validate(),
                Err(Refusal("invalid head-sum indexer selection geometry"))
            );
        }
    }

    #[test]
    fn declared_variants_refuse_by_name() {
        let PooledIndexerQuery::Select {
            heads,
            head_dim,
            group,
            max_top_k,
            always_tail,
            mode,
            page_size,
            capacity,
            ..
        } = select(IndexerSelectionMode::Global)
        else {
            unreachable!()
        };
        let with = |granularity, key_format| PooledIndexerQuery::Select {
            heads,
            head_dim,
            group,
            max_top_k,
            always_tail,
            granularity,
            key_format,
            mode,
            page_size,
            capacity,
        };
        assert_eq!(
            with(IndexerGranularity::Position, IndexerKeyFormat::Bf16).validate(),
            Err(Refusal("position indexer requires MXFP4 keys"))
        );
        assert_eq!(
            with(IndexerGranularity::Pool, IndexerKeyFormat::Fp4).validate(),
            Err(Refusal("FP4 indexer keys are declared but not supported"))
        );
        let PooledIndexerQuery::KeyWrite {
            pooling,
            key_norm_eps_bits,
            page_size,
            phase,
            capacity,
            ..
        } = key_write(RecurrentPhase::Prefill)
        else {
            unreachable!()
        };
        assert_eq!(
            PooledIndexerQuery::KeyWrite {
                pooling,
                key_norm_eps_bits,
                page_size,
                key_format: IndexerKeyFormat::Fp4,
                phase,
                capacity,
            }
            .validate(),
            Err(Refusal("FP4 indexer keys are declared but not supported"))
        );
        assert_eq!(
            select(IndexerSelectionMode::PerRank { ranks: 1 }).validate(),
            Err(Refusal("per-rank indexer selection needs at least 2 ranks"))
        );
    }

    #[test]
    fn key_write_roles_are_named_by_path() {
        use crate::allocation::Traversal;
        let buffers = IndexerKeyWriteBuffers {
            key: 1,
            key_stride: 128,
            gate: 2,
            gate_stride: 128,
            norm_weight: 3,
            norm_bias: 4,
            position: 5,
            row_batch: 6,
            positions: 7,
            live_rows: 8,
            tail: IndexerTailState {
                state: 9,
                entries: 4,
                slot_ids: 10,
                checkpoint_ids: None,
            },
            pooled: PooledKeyPages {
                keys: 11,
                pages: 8,
                page_stride: 4096,
                page_tables: BlockTables {
                    tables: 12,
                    sequences: 1,
                    max_pages: 2,
                },
            },
        };
        let mut roles = Vec::new();
        buffers.visit_named(&mut |role, &view| roles.push((role.to_owned(), view)));
        let names: Vec<_> = roles.iter().map(|(role, _)| role.as_str()).collect();
        assert_eq!(
            names,
            [
                "key",
                "gate",
                "norm_weight",
                "norm_bias",
                "position",
                "row_batch",
                "positions",
                "live_rows",
                "tail.state",
                "tail.slot_ids",
                "pooled.keys",
                "pooled.page_tables.tables"
            ]
        );
        assert!(roles.iter().map(|&(_, view)| view).eq(1..=12));
    }

    #[test]
    fn selection_key_list_width_is_checked_at_its_bounds() {
        let with = |group, max_top_k| PooledIndexerQuery::Select {
            heads: 32,
            head_dim: 128,
            group,
            max_top_k,
            always_tail: true,
            granularity: IndexerGranularity::Pool,
            key_format: IndexerKeyFormat::Bf16,
            mode: IndexerSelectionMode::Global,
            page_size: 64,
            capacity: 1,
        };
        let largest = with(4, 2_147_483_644);
        largest.validate().unwrap();
        assert_eq!(largest.list_width(), Some(MAX_KEY_LIST_WIDTH));
        let first_rejected = with(4, 2_147_483_648);
        assert_eq!(
            first_rejected.validate(),
            Err(Refusal("indexer key list width exceeds the i32 maximum"))
        );
        assert_eq!(first_rejected.list_width(), None);
        assert_eq!(with(3, u32::MAX).list_width(), None);
        assert_eq!(
            with(3, 2049).validate(),
            Err(Refusal(
                "indexer pools hold a power of two from 2 to 16 tokens"
            ))
        );
    }
}
