//! The gated and sparse attention family's ops and buffer roles: the gated
//! attention rows (the per-head Q/K norm with M-RoPE and the paged KV
//! write, and the sigmoid output gate) and block-sparse attention with its
//! learned indexer (the index query, the raw-key ring and block-key write,
//! block scores, top-k block selection and the attention over the selected
//! blocks), plus one context-parallel group rank's partials of that
//! attention. The math is [`GatedAttentionQuery`] (`gated_attention_oracle`),
//! [`SparseAttentionQuery`] (`sparse_attention_oracle`) and
//! `ContextParallelQuery::SelectedAttentionPartials`
//! (`context_parallel_oracle`).
//!
//! The ops add what the queries leave out and a model shares with its other
//! ops: the cache's context (which fixes a block table row), where the
//! packed projection rows place their gates and raw keys, and the paging of
//! the indexer's block keys. Every query field is math, apart from
//! `capacity`, which the backend states and a model asks for in
//! [`Need::capacity`](crate::backend::Need); split counts, merges, rows per
//! block, score tiles and the scores' row stride are the backend's choices.
//! A model that relies on verify rows equal to decode rows, prompt rows
//! equal over chunk splits at whole blocks, or the top-k's lower-index ties
//! asks for them in [`Need::exact`](crate::backend::Need). Block scores
//! also take the use they serve (`precision::OperationUse`): how a step
//! runs its rows (a graph step or a prompt chunk) is the model's to say, and
//! a backend may score them differently.
//!
//! The block scores and the top-k pass the scores in the backend's layout,
//! which only its own top-k reads: a model prepares both from one backend.
//! The selection they write is math: each row's `top_blocks` I32 slots hold
//! its kept complete blocks' indices, ascending, in the first
//! `min(complete blocks, top_blocks)` slots (every complete block when there
//! are at most `top_blocks`); the attention reads that many.

pub use crate::attention::{FoldInputs, KvCache};
use crate::context_parallel::ContextParallelQuery;
use crate::contract::{GatedAttentionQuery, Refusal, SparseAttentionQuery, Validate};

/// The gated Q/K norm, M-RoPE and paged KV write
/// (`GatedAttentionQuery::QkNormRopeKvWrite`) over a cache of up to
/// `max_context` tokens per item. Packed rows are
/// `[row][q_heads * 2 head_dim | kv_heads * head_dim | kv_heads * head_dim]`,
/// each Q head's query followed by its gate; the op writes the normed and
/// rotated Q heads to `q` and stores K and V at the row's item and position.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatedKvWriteOp {
    pub query: GatedAttentionQuery,
    /// The most tokens of one item: its block table row holds
    /// `max_context / page_size` pages, rounded up.
    pub max_context: u32,
}

impl Validate for GatedKvWriteOp {
    fn validate(&self) -> Result<(), Refusal> {
        self.query.validate()?;
        if !matches!(self.query, GatedAttentionQuery::QkNormRopeKvWrite { .. }) {
            return Err(Refusal("a gated KV write writes by a gated KV write query"));
        }
        if self.max_context == 0 {
            return Err(Refusal("a gated KV write needs a context"));
        }
        Ok(())
    }
}

/// The sigmoid output gate (`GatedAttentionQuery::OutputGate`) whose gate
/// logits are the second half of each Q head of the packed rows of a gated
/// KV write over `kv_heads` KV heads of the gate's head width.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OutputGateOp {
    pub query: GatedAttentionQuery,
    /// The packed rows' KV heads, which fix the rows' width.
    pub kv_heads: u32,
}

impl Validate for OutputGateOp {
    fn validate(&self) -> Result<(), Refusal> {
        self.query.validate()?;
        if !matches!(self.query, GatedAttentionQuery::OutputGate { .. }) {
            return Err(Refusal("an output gate gates by an output gate query"));
        }
        if self.kv_heads == 0 {
            return Err(Refusal("an output gate's packed rows hold KV heads"));
        }
        Ok(())
    }
}

/// The indexer key write (`SparseAttentionQuery::IndexerKeyWrite`) of packed
/// indexer projection rows `[row][index_heads * head_dim | head_dim]` (the
/// query heads, then the raw key) into the raw-key rings and the block-key
/// plane of a paged cache of `page_size`-token pages and up to
/// `max_context` tokens per item.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseKeyWriteOp {
    pub query: SparseAttentionQuery,
    pub index_heads: u32,
    /// Tokens of one cache page: its block-key page holds the keys of its
    /// `page_size / block` blocks.
    pub page_size: u32,
    pub max_context: u32,
}

impl Validate for SparseKeyWriteOp {
    fn validate(&self) -> Result<(), Refusal> {
        self.query.validate()?;
        let SparseAttentionQuery::IndexerKeyWrite { block, .. } = self.query else {
            return Err(Refusal("an indexer key write writes by a key write query"));
        };
        if self.index_heads == 0 || self.max_context == 0 {
            return Err(Refusal(
                "an indexer key write needs index heads and a context",
            ));
        }
        check_pages(self.page_size, block)
    }
}

/// The block scores (`SparseAttentionQuery::BlockScores`) of each row over
/// the block keys a [`SparseKeyWriteOp`] of `block`-token blocks and
/// `page_size`-token pages wrote, for items of up to `max_context` tokens,
/// whose complete blocks are the query's `max_blocks`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockScoresOp {
    pub query: SparseAttentionQuery,
    pub block: u32,
    pub page_size: u32,
    pub max_context: u32,
}

impl Validate for BlockScoresOp {
    fn validate(&self) -> Result<(), Refusal> {
        self.query.validate()?;
        let SparseAttentionQuery::BlockScores { max_blocks, .. } = self.query else {
            return Err(Refusal("block scores score by a block scores query"));
        };
        if self.block < 2 || self.max_context.checked_div(self.block) != Some(max_blocks) {
            return Err(Refusal(
                "block scores cover the complete blocks of the context",
            ));
        }
        check_pages(self.page_size, self.block)
    }
}

/// Sparse attention (`SparseAttentionQuery::SparseAttention`) over a paged
/// cache of up to `max_context` tokens per item.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseAttentionOp {
    pub query: SparseAttentionQuery,
    pub max_context: u32,
}

impl Validate for SparseAttentionOp {
    fn validate(&self) -> Result<(), Refusal> {
        self.query.validate()?;
        if !matches!(self.query, SparseAttentionQuery::SparseAttention { .. }) {
            return Err(Refusal(
                "sparse attention attends by a sparse attention query",
            ));
        }
        if self.max_context == 0 {
            return Err(Refusal("sparse attention needs a context"));
        }
        Ok(())
    }
}

/// One context-parallel group rank's partials of sparse attention
/// (`ContextParallelQuery::SelectedAttentionPartials`) with logits scaled
/// by `softmax_scale`, over a paged cache of up to `max_context` tokens per
/// item.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelectedPartialsOp {
    pub query: ContextParallelQuery,
    pub softmax_scale: f32,
    pub max_context: u32,
}

impl Validate for SelectedPartialsOp {
    fn validate(&self) -> Result<(), Refusal> {
        self.query.validate()?;
        if !matches!(
            self.query,
            ContextParallelQuery::SelectedAttentionPartials { .. }
        ) {
            return Err(Refusal(
                "selected partials attend by a selected partials query",
            ));
        }
        if !self.softmax_scale.is_finite() || self.softmax_scale <= 0.0 || self.max_context == 0 {
            return Err(Refusal(
                "selected partials need a positive softmax scale and a context",
            ));
        }
        Ok(())
    }
}

/// Pages hold whole blocks.
fn check_pages(page_size: u32, block: u32) -> Result<(), Refusal> {
    if page_size == 0 || block == 0 || !page_size.is_multiple_of(block) {
        return Err(Refusal("cache pages hold whole indexer blocks"));
    }
    Ok(())
}

crate::scratch_record! {
    /// The indexer's block-key plane: `slots` pages of the KV cache's page
    /// table, each `page_stride` bytes after the last and holding the BF16
    /// keys `[block][head_dim]` of its `page_size / block` blocks.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct BlockKeys {
        pub keys: value,
        pub slots: scalar(u32),
        pub page_stride: scalar(u64),
    }
}

crate::scratch_record! {
    /// Raw-key rings of `slots` slots of `ring` rows: BF16 `keys`
    /// `[slot][ring][head_dim]` and their I32 (T, H, W) RoPE `positions`
    /// `[slot][ring][3]`, a row at position p in row `p % ring`.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct KeyRings {
        pub keys: value,
        pub positions: value,
        pub slots: scalar(u32),
    }
}

crate::scratch_record! {
    /// A gated KV write's buffers by role: the BF16 packed rows `qkv`, the
    /// BF16 Q rows `q` (`[row][q_heads * head_dim]`) it writes, the cache,
    /// each item's I32 block table row, each row's I32 item
    /// (`token_batch`) and cache position, the device count of live rows,
    /// and in `rope` the BF16 `[head_dim]` norm weights `q_norm` and
    /// `k_norm`, each row's I32 (T, H, W) `rope_positions` and the F32
    /// `[position][rotary / 2][cos, sin]` table `cos_sin`, all of which it
    /// reads (attention's gated fold reads the same).
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct GatedKvWriteBuffers {
        pub qkv: value,
        pub q: value,
        pub cache: record(KvCache),
        pub block_tables: value,
        pub token_batch: value,
        pub positions: value,
        pub live_rows: value,
        pub rope: record(FoldInputs),
    }
}

crate::scratch_record! {
    /// An output gate's buffers by role: the BF16 attention rows
    /// `[row][heads * head_dim]`, the packed rows `qkv` that hold the gate
    /// logits, the BF16 `output` rows (which may be the attention rows) and
    /// the device count of live rows.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct OutputGateBuffers {
        pub attention: value,
        pub qkv: value,
        pub output: value,
        pub live_rows: value,
    }
}

crate::scratch_record! {
    /// An index query's buffers by role: the packed indexer projection rows
    /// `[row][(heads + 1) * head_dim]` (the query heads, then the raw key),
    /// the BF16 `[head_dim]` norm `weight`, each row's I32 (T, H, W)
    /// `rope_positions`, the F32 `cos_sin` table, the device count of live
    /// rows and the BF16 index query rows `q` (`[row][heads * head_dim]`)
    /// it writes.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct IndexQueryBuffers {
        pub projection: value,
        pub weight: value,
        pub rope_positions: value,
        pub cos_sin: value,
        pub live_rows: value,
        pub q: value,
    }
}

crate::scratch_record! {
    /// An indexer key write's buffers by role: the packed projection rows,
    /// norm weight, RoPE positions and table of [`IndexQueryBuffers`], each
    /// row's I32 item (`token_batch`) and position, the device count of
    /// live rows, each item's I32 ring slot (`slot_ids`), the rings, each
    /// item's I32 anchor ring slot (`anchor_slots`, -1 for none) and the
    /// position its anchor ends at (`anchor_ends`), the anchor rings, each
    /// item's I32 block table row and the block-key plane. An anchored item
    /// also leaves in its anchor slot the ring its rows before its anchor
    /// end would leave.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct SparseKeyWriteBuffers {
        pub projection: value,
        pub weight: value,
        pub rope_positions: value,
        pub cos_sin: value,
        pub token_batch: value,
        pub positions: value,
        pub live_rows: value,
        pub slot_ids: value,
        pub rings: record(KeyRings),
        pub anchor_slots: value,
        pub anchor_ends: value,
        pub anchors: record(KeyRings),
        pub block_tables: value,
        pub block_keys: record(BlockKeys),
    }
}

crate::scratch_record! {
    /// Block scores' buffers by role: the index query rows `q`, the
    /// block-key plane, each item's I32 block table row, each row's I32
    /// item (`token_batch`) and position, the device count of live rows and
    /// the `scores` it writes, in the backend's layout of the bytes its info
    /// lists.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct BlockScoresBuffers {
        pub q: value,
        pub block_keys: record(BlockKeys),
        pub block_tables: value,
        pub token_batch: value,
        pub positions: value,
        pub live_rows: value,
        pub scores: value,
    }
}

crate::scratch_record! {
    /// A top-k's buffers by role: the `scores` of the same backend's block
    /// scores, each row's I32 position, the device count of live rows and
    /// the I32 selection `selected` (`[row][top_blocks]`) it writes.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct BlockTopKBuffers {
        pub scores: value,
        pub positions: value,
        pub live_rows: value,
        pub selected: value,
    }
}

crate::scratch_record! {
    /// Sparse attention's buffers by role: the BF16 query rows `q`
    /// (`[row][q_heads * head_dim]`), the cache, each item's I32 block table
    /// row, each row's I32 item (`token_batch`) and position, the device
    /// count of live rows, the selection, the `workspace` of the partials
    /// the op sizes in its info, and the BF16 `output` rows it writes.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct SparseAttentionBuffers {
        pub q: value,
        pub cache: record(KvCache),
        pub block_tables: value,
        pub token_batch: value,
        pub positions: value,
        pub live_rows: value,
        pub selected: value,
        pub workspace: value,
        pub output: value,
    }
}

crate::scratch_record! {
    /// Selected partials' buffers by role: the BF16 query rows `q`
    /// (`[row][q_heads * kv_heads * head_dim]`), the cache, each item's I32
    /// block table row, each row's I32 item (`token_batch`) and position,
    /// the device count of live rows, the selection, and the F32 partials
    /// it writes: the running maximum `max` and denominator `den`
    /// (`[row][heads]`) and the numerators `num` (`[row][heads][head_dim]`).
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct SelectedPartialsBuffers {
        pub q: value,
        pub cache: record(KvCache),
        pub block_tables: value,
        pub token_batch: value,
        pub positions: value,
        pub live_rows: value,
        pub selected: value,
        pub max: value,
        pub den: value,
        pub num: value,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::allocation::Traversal;
    use crate::contract::{AttentionPhase, DType};

    pub(crate) fn kv_write() -> GatedKvWriteOp {
        GatedKvWriteOp {
            query: GatedAttentionQuery::QkNormRopeKvWrite {
                q_heads: 8,
                kv_heads: 1,
                head_dim: 256,
                rotary_dim: 64,
                theta_bits: 1e7_f32.to_bits(),
                mrope: [11, 11, 10],
                eps_bits: 1e-6_f32.to_bits(),
                page_size: 64,
                kv_dtype: DType::Bf16,
                capacity: 8192,
            },
            max_context: 4096,
        }
    }

    pub(crate) fn key_write() -> SparseKeyWriteOp {
        SparseKeyWriteOp {
            query: SparseAttentionQuery::IndexerKeyWrite {
                head_dim: 128,
                rotary_dim: 64,
                theta_bits: 1e7_f32.to_bits(),
                mrope: [11, 11, 10],
                eps_bits: 1e-6_f32.to_bits(),
                block: 4,
                ring: 8,
                capacity: 8192,
            },
            index_heads: 16,
            page_size: 64,
            max_context: 4096,
        }
    }

    #[test]
    fn ops_check_their_math_only() {
        assert_eq!(kv_write().validate(), Ok(()));
        // Any head width is math; a backend refuses what it lacks.
        let GatedAttentionQuery::QkNormRopeKvWrite {
            q_heads,
            kv_heads,
            rotary_dim,
            theta_bits,
            mrope,
            eps_bits,
            page_size,
            kv_dtype,
            capacity,
            ..
        } = kv_write().query
        else {
            unreachable!()
        };
        let wide = GatedKvWriteOp {
            query: GatedAttentionQuery::QkNormRopeKvWrite {
                q_heads,
                kv_heads,
                head_dim: 96,
                rotary_dim,
                theta_bits,
                mrope,
                eps_bits,
                page_size,
                kv_dtype,
                capacity,
            },
            ..kv_write()
        };
        assert_eq!(wide.validate(), Ok(()));
        assert!(
            GatedKvWriteOp {
                max_context: 0,
                ..kv_write()
            }
            .validate()
            .is_err()
        );
        let gate = GatedAttentionQuery::OutputGate {
            heads: 8,
            head_dim: 256,
            capacity: 8192,
        };
        assert!(
            GatedKvWriteOp {
                query: gate,
                ..kv_write()
            }
            .validate()
            .is_err()
        );
        assert_eq!(
            OutputGateOp {
                query: gate,
                kv_heads: 1
            }
            .validate(),
            Ok(())
        );
        assert!(
            OutputGateOp {
                query: gate,
                kv_heads: 0
            }
            .validate()
            .is_err()
        );

        assert_eq!(key_write().validate(), Ok(()));
        assert!(
            SparseKeyWriteOp {
                page_size: 62,
                ..key_write()
            }
            .validate()
            .is_err()
        );
        let scores = BlockScoresOp {
            query: SparseAttentionQuery::BlockScores {
                heads: 16,
                head_dim: 96,
                max_blocks: 1024,
                capacity: 256,
            },
            block: 4,
            page_size: 64,
            max_context: 4096,
        };
        assert_eq!(scores.validate(), Ok(()));
        assert!(
            BlockScoresOp {
                max_context: 8192,
                ..scores
            }
            .validate()
            .is_err()
        );
        let attention = SparseAttentionOp {
            query: SparseAttentionQuery::SparseAttention {
                q_heads: 8,
                kv_heads: 1,
                head_dim: 192,
                block: 4,
                top_blocks: 512,
                softmax_scale_bits: 0.0625_f32.to_bits(),
                page_size: 64,
                kv_dtype: DType::Fp8E4M3,
                phase: AttentionPhase::Decode,
                capacity: 256,
            },
            max_context: 4096,
        };
        assert_eq!(attention.validate(), Ok(()));
        assert!(
            SparseAttentionOp {
                query: scores.query,
                ..attention
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn roles_are_named_by_path() {
        let rings = KeyRings {
            keys: 9,
            positions: 10,
            slots: 4,
        };
        let buffers = SparseKeyWriteBuffers {
            projection: 1,
            weight: 2,
            rope_positions: 3,
            cos_sin: 4,
            token_batch: 5,
            positions: 6,
            live_rows: 7,
            slot_ids: 8,
            rings,
            anchor_slots: 11,
            anchor_ends: 12,
            anchors: KeyRings {
                keys: 13,
                positions: 14,
                slots: 1,
            },
            block_tables: 15,
            block_keys: BlockKeys {
                keys: 16,
                slots: 32,
                page_stride: 1024,
            },
        };
        let mut named = Vec::new();
        buffers.visit_named(&mut |role, &view| named.push((role.to_owned(), view)));
        let roles: Vec<_> = named.iter().map(|(role, _)| role.as_str()).collect();
        assert_eq!(
            roles,
            [
                "projection",
                "weight",
                "rope_positions",
                "cos_sin",
                "token_batch",
                "positions",
                "live_rows",
                "slot_ids",
                "rings.keys",
                "rings.positions",
                "anchor_slots",
                "anchor_ends",
                "anchors.keys",
                "anchors.positions",
                "block_tables",
                "block_keys.keys",
            ]
        );
    }
}
