//! Lowering of the K/V write, the indexer key write, the block selection
//! and the sparse attention for a context-parallel group of 2 ranks
//! (`crate::context_parallel`), through the instance's prepared
//! context-parallel ops (`crate::ops::ContextParallelSet`). Each lowering
//! runs only when the instance's group size is above 1; at size 1 the
//! stages lower as before.

use kernel_core::attention::{BlockTables, FoldInputs};
use kernel_core::backend::{Counts, Prepared, View};
use kernel_core::collective::{GatherBuffers, RankViews};
use kernel_core::context_parallel::{
    GroupTransferBuffers, LocalizeKeysBuffers, PartialMergeBuffers,
};
use kernel_core::contract::Refusal;
use kernel_core::gated_sparse::{
    GatedKvWriteBuffers, KeyRings, SelectedPartialsBuffers, SparseKeyWriteBuffers,
};
use kernel_core::pooled_indexer::{IndexerMergeBuffers, IndexerSelectBuffers, PooledKeyPages};
use model_qwen38_contract::context_parallel::{bucket, bucket_index, dims};
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{RankStage, at, page_slots};
use crate::context_parallel::{ContextParallel, GatherProtocol, RankState};
use crate::contract::{ANCHOR_ITEMS, BF16_BYTES, SCORE_ROWS};
use crate::ops::{Bind, ContextParallelOps, ContextParallelSet, Key, Lowered};
use crate::program::Stage;
use crate::side::PIECES;

const F32: u64 = 4;
const I32: u64 = 4;

/// `span` from `offset` bytes for `length` bytes, which it must hold.
fn view(span: DeviceBufferSpanV1, offset: u64, length: u64) -> Result<DeviceBufferSpanV1, Refusal> {
    if offset + length > span.length {
        return Err(Refusal(
            "a Qwen3.8 context-parallel view exceeds its buffer",
        ));
    }
    Ok(DeviceBufferSpanV1::new(
        span.buffer,
        span.offset + offset,
        length,
    ))
}

/// `span` from `offset` bytes to its end.
fn from(span: DeviceBufferSpanV1, offset: u64) -> Result<DeviceBufferSpanV1, Refusal> {
    view(span, offset, span.length.saturating_sub(offset))
}

/// One piece of a phase: its first row, its rows and its live count.
#[derive(Clone, Copy)]
struct Piece {
    first: u32,
    rows: u32,
    live: DeviceBufferSpanV1,
}

/// A topology rank's place in its group: groups are consecutive topology
/// ranks (`crate::context_parallel`).
#[derive(Clone, Copy)]
struct Group {
    /// The rank within its group.
    rank: u32,
    /// The group's first topology rank.
    first: u32,
    size: u32,
}

impl Group {
    /// The group's topology ranks in group rank order.
    fn members(self) -> std::ops::Range<u32> {
        self.first..self.first + self.size
    }
}

impl RankStage<'_> {
    /// The launches that replace `stage`'s in a group of 2 ranks, or `None`
    /// when the stage lowers as at size 1.
    pub(super) fn context_parallel_stage(
        &self,
        stage: Stage,
    ) -> Result<Option<Vec<Lowered>>, Refusal> {
        let Some(cp) = self.context_parallel.filter(|cp| cp.size > 1) else {
            return Ok(None);
        };
        Ok(Some(match stage {
            Stage::AttnQkNormRopeKv(layer) => self.owner_kv_write(cp, layer)?,
            Stage::IndexKeyWrite(layer) => self.owner_key_write(cp, layer)?,
            Stage::BlockSelect(layer) => {
                let mut launches = Vec::new();
                for piece in self.pieces() {
                    launches.extend(self.group_select(cp, layer, piece)?);
                }
                launches
            }
            Stage::SparseAttention(layer) => {
                let mut launches = Vec::new();
                for piece in self.pieces() {
                    launches.extend(self.group_attention(cp, layer, piece)?);
                }
                launches
            }
            _ => return Ok(None),
        }))
    }

    /// A graph phase's rows, or a prefill's selection pieces of
    /// `SCORE_ROWS` rows with their live counts (as `block_select`).
    fn pieces(&self) -> Vec<Piece> {
        if self.decode() {
            return vec![Piece {
                first: 0,
                rows: self.rows,
                live: self.live_tokens(),
            }];
        }
        (0..PIECES)
            .map(|piece| piece * SCORE_ROWS)
            .take_while(|&first| first < self.rows)
            .map(|first| Piece {
                first,
                rows: (self.rows - first).min(SCORE_ROWS),
                live: self.side.piece_live(first / SCORE_ROWS),
            })
            .collect()
    }

    fn group(&self, cp: &ContextParallel) -> Result<Group, Refusal> {
        if cp.size == 0 || !self.shape.ranks().is_multiple_of(cp.size) {
            return Err(Refusal(
                "context-parallel group does not divide the topology ranks",
            ));
        }
        Ok(Group {
            rank: self.rank % cp.size,
            first: self.rank / cp.size * cp.size,
            size: cp.size,
        })
    }

    fn own<'b>(&self, cp: &'b ContextParallel) -> Result<&'b RankState, Refusal> {
        cp.ranks
            .get(self.rank as usize)
            .ok_or(Refusal("no context-parallel state of this Qwen3.8 rank"))
    }

    fn cp_set(&self) -> Result<&ContextParallelSet, Refusal> {
        self.ops
            .ok_or(Refusal("Qwen3.8 ops are not prepared"))?
            .context_parallel()
    }

    fn cp_ops(&self) -> Result<&ContextParallelOps, Refusal> {
        self.cp_set()?.key(Key::of(self.phase))
    }

    /// Each appended row's position localized to the rank that owns its
    /// page (-1 elsewhere), then the gated norm, RoPE and K/V write of the
    /// owned rows; every row's Q heads are written.
    fn owner_kv_write(&self, cp: &ContextParallel, layer: u32) -> Result<Vec<Lowered>, Refusal> {
        let s = self.scratch();
        let own = self.own(cp)?;
        let group = self.group(cp)?;
        let ops = self.cp_ops()?;
        let rows = self.rows;
        let positions = view(own.positions, 0, u64::from(rows) * I32)?;
        let mut bind = Bind::default();
        let localize = LocalizeKeysBuffers {
            lists: bind.view(self.inputs.positions),
            live_rows: bind.view(self.live_tokens()),
            output: bind.view(positions),
        };
        let mut launches = bind.lower(ops.localize(group.rank)?, &localize, Counts::rows(rows))?;
        let attn = format!(
            "{}.self_attn",
            model_qwen38_contract::views::layer_stem_for(layer, self.shape)
        );
        let (fp8, _) = self.kv_storage(layer);
        let mut bind = Bind::default();
        let buffers = GatedKvWriteBuffers {
            qkv: bind.view(s.attn_qkv),
            q: bind.view(s.attn_q),
            cache: self.cache(&mut bind, layer)?,
            block_tables: bind.view(self.inputs.block_tables),
            token_batch: bind.view(self.inputs.token_batch),
            positions: bind.view(positions),
            live_rows: bind.view(self.live_tokens()),
            rope: FoldInputs {
                cos_sin: Some(bind.view(s.cos_sin)),
                q_norm: Some(bind.view(self.view(&format!("{attn}.q_norm.weight"))?)),
                k_norm: Some(bind.view(self.view(&format!("{attn}.k_norm.weight"))?)),
                rope_positions: Some(bind.view(self.side.rope_positions)),
            },
        };
        launches.extend(bind.lower(ops.kv_write(fp8), &buffers, self.attention_counts(rows))?);
        Ok(launches)
    }

    /// The indexer key write with the group rank and size: both ranks write
    /// the ring, and only a block's owner stores its key.
    fn owner_key_write(&self, cp: &ContextParallel, layer: u32) -> Result<Vec<Lowered>, Refusal> {
        let s = self.scratch();
        let group = self.group(cp)?;
        let (keys, positions) = self.ring_layer(layer);
        let (anchor_keys, anchor_positions) = self.anchor_ring_layer(layer);
        let mut bind = Bind::default();
        let buffers = SparseKeyWriteBuffers {
            projection: bind.view(s.attn_index),
            weight: bind.view(self.view(&format!(
                "{}.self_attn.indexer.k_layernorm.weight",
                model_qwen38_contract::views::layer_stem_for(layer, self.shape)
            ))?),
            rope_positions: bind.view(self.side.rope_positions),
            cos_sin: bind.view(s.cos_sin),
            token_batch: bind.view(self.inputs.token_batch),
            positions: bind.view(self.inputs.positions),
            live_rows: bind.view(self.live_tokens()),
            slot_ids: bind.view(self.inputs.slot_ids),
            rings: KeyRings {
                keys: bind.view(keys),
                positions: bind.view(positions),
                slots: self.ring_layout.slots as u32,
            },
            anchor_slots: bind.view(self.side.anchor_slot),
            anchor_ends: bind.view(self.side.anchor_end),
            anchors: KeyRings {
                keys: bind.view(anchor_keys),
                positions: bind.view(anchor_positions),
                slots: ANCHOR_ITEMS,
            },
            block_tables: bind.view(self.inputs.block_tables),
            block_keys: self.block_keys(&mut bind, layer)?,
        };
        bind.lower(
            self.cp_ops()?.key_write(group.rank)?,
            &buffers,
            self.attention_counts(self.rows),
        )
    }

    /// The group transfer `op` of `input` within the group, into each
    /// member's output `output` picks, over the members' `protocol`s.
    #[allow(clippy::too_many_arguments)]
    fn group_transfer(
        &self,
        cp: &ContextParallel,
        op: &Prepared<GroupTransferBuffers<View>>,
        rows: u32,
        input: DeviceBufferSpanV1,
        output: impl Fn(&RankState) -> Result<DeviceBufferSpanV1, Refusal>,
        protocol: impl Fn(&RankState) -> Result<GatherProtocol, Refusal>,
    ) -> Result<Vec<Lowered>, Refusal> {
        let group = self.group(cp)?;
        let mut bind = Bind::default();
        let buffers = GatherBuffers {
            input: bind.view(input),
            ranks: group
                .members()
                .map(|member| {
                    let state = cp
                        .ranks
                        .get(member as usize)
                        .ok_or(Refusal("a Qwen3.8 group member has no state"))?;
                    let protocol = protocol(state)?;
                    Ok(RankViews {
                        output: bind.view(output(state)?),
                        state: vec![bind.view(protocol.flags), bind.view(protocol.epoch)],
                    })
                })
                .collect::<Result<_, Refusal>>()?,
        };
        bind.lower(op, &buffers, Counts::rows(rows))
    }

    /// One piece's head-sum selection over the rank's own block keys as
    /// (score, block) pairs, the group all-gather of the pairs, and their
    /// exact merge into the ascending block list `SparseAttention` reads.
    fn group_select(
        &self,
        cp: &ContextParallel,
        layer: u32,
        piece: Piece,
    ) -> Result<Vec<Lowered>, Refusal> {
        let s = self.scratch();
        let own = self.own(cp)?;
        let group = self.group(cp)?;
        let ops = self.cp_ops()?;
        let pages = self.pages(layer)?;
        let dims = dims(self.shape);
        let Piece { first, rows, live } = piece;
        let bucket = bucket(rows);
        let index = bucket_index(rows).map_err(|_| Refusal("no Qwen3.8 group bucket"))?;
        let counts = self.attention_counts(rows);
        let first = u64::from(first);
        let index_row = u64::from(self.shape.index_heads() * self.shape.index_dim()) * BF16_BYTES;
        let positions = at(self.inputs.positions, first, I32);
        let pairs = view(own.pairs, 0, dims.pair_words(bucket) * 4)?;
        let mut bind = Bind::default();
        let select = IndexerSelectBuffers {
            q: bind.view(at(s.index_q, first, index_row)),
            head_weights: None,
            pooled: PooledKeyPages {
                keys: bind.view(pages.planes[2]),
                pages: page_slots(&pages)?,
                page_stride: pages.page_stride_bytes,
                page_tables: BlockTables {
                    tables: bind.view(self.inputs.block_tables),
                    sequences: counts.items.max(1),
                    max_pages: self.shape.max_pages(),
                },
            },
            row_batch: bind.view(at(self.inputs.token_batch, first, I32)),
            positions: bind.view(positions),
            live_rows: bind.view(live),
            // The rank's own buffer holds the selection's split workspace.
            workspace: bind.view(own.local),
            output: bind.view(pairs),
        };
        let mut launches = bind.lower(ops.select(group.rank)?, &select, Counts::rows(rows))?;
        launches.extend(self.group_transfer(
            cp,
            self.cp_set()?.bucket(rows)?.pairs(group.rank)?,
            rows,
            pairs,
            |state| {
                state
                    .gathered_pairs
                    .get(index)
                    .copied()
                    .ok_or(Refusal("no Qwen3.8 gathered pairs of this bucket"))
            },
            |state| {
                state
                    .pair_gathers
                    .get(index)
                    .copied()
                    .ok_or(Refusal("no Qwen3.8 pair gather state of this bucket"))
            },
        )?);
        let gathered = own
            .gathered_pairs
            .get(index)
            .copied()
            .ok_or(Refusal("no Qwen3.8 gathered pairs of this bucket"))?;
        let mut bind = Bind::default();
        let merge = IndexerMergeBuffers {
            pairs: bind.view(gathered),
            rank_rows: bucket,
            positions: None,
            live_rows: bind.view(live),
            output: bind.view(at(
                s.selected,
                first,
                u64::from(self.shape.top_blocks()) * I32,
            )),
        };
        launches.extend(bind.lower(&ops.merge, &merge, Counts::rows(rows))?);
        Ok(launches)
    }

    /// One piece's sparse attention of the rank's heads over the group's
    /// keys: the group all-gather of the queries, the selected partials of
    /// each source's heads over the rank's own pages, one all-gather per
    /// owner of the blocks that hold its heads, the merge of the rank's
    /// blocks into `attn_out`, and the group barrier that ends the piece.
    fn group_attention(
        &self,
        cp: &ContextParallel,
        layer: u32,
        piece: Piece,
    ) -> Result<Vec<Lowered>, Refusal> {
        let s = self.scratch();
        let own = self.own(cp)?;
        let group = self.group(cp)?;
        let set = self.cp_set()?;
        let ops = set.key(Key::of(self.phase))?;
        let transfers = set.bucket(piece.rows)?;
        let dims = dims(self.shape);
        let Piece { first, rows, live } = piece;
        let bucket = bucket(rows);
        let index = bucket_index(rows).map_err(|_| Refusal("no Qwen3.8 group bucket"))?;
        let row_bytes = u64::from(dims.heads) * u64::from(dims.width) * BF16_BYTES;
        // 1. The gathered queries: a bucket of rows from the piece's first,
        // moved back to end inside `attn_q`; the piece's rows start `shift`
        // rows into each block.
        let capacity = u32::try_from(s.attn_q.length / row_bytes)
            .map_err(|_| Refusal("Qwen3.8 query rows exceed u32"))?;
        let start = first.min(
            capacity
                .checked_sub(bucket)
                .ok_or(Refusal("a Qwen3.8 group bucket exceeds the query rows"))?,
        );
        let shift = u64::from(first - start);
        let queries = view(
            s.attn_q,
            u64::from(start) * row_bytes,
            u64::from(bucket) * row_bytes,
        )?;
        let mut launches = self.group_transfer(
            cp,
            transfers.queries(group.rank)?,
            rows,
            queries,
            |state| Ok(state.queries),
            |state| {
                state
                    .query_gathers
                    .get(index)
                    .copied()
                    .ok_or(Refusal("no Qwen3.8 query gather state of this bucket"))
            },
        )?;
        // 2. The partials of each source's heads over the rank's pages, into
        // the source's block of the send buffer: source q's queries start
        // `q * bucket * heads * width` values into the gathered queries.
        let (fp8, _) = self.kv_storage(layer);
        let first = u64::from(first);
        let block = dims.block_floats(bucket);
        let pairs = u64::from(bucket) * u64::from(dims.heads) * F32;
        let counts = self.attention_counts(rows);
        let source_values = u64::from(bucket) * u64::from(dims.heads) * u64::from(dims.width);
        for source in 0..cp.size {
            let q = from(
                own.queries,
                u64::from(source) * source_values * BF16_BYTES + shift * row_bytes,
            )?;
            let send = view(own.send, u64::from(source) * block * F32, block * F32)?;
            let mut bind = Bind::default();
            let buffers = SelectedPartialsBuffers {
                q: bind.view(q),
                cache: self.cache(&mut bind, layer)?,
                block_tables: bind.view(self.inputs.block_tables),
                token_batch: bind.view(at(self.inputs.token_batch, first, I32)),
                positions: bind.view(at(self.inputs.positions, first, I32)),
                live_rows: bind.view(live),
                selected: bind.view(at(
                    s.selected,
                    first,
                    u64::from(self.shape.top_blocks()) * I32,
                )),
                max: bind.view(view(send, 0, pairs)?),
                den: bind.view(view(send, pairs, pairs)?),
                num: bind.view(from(send, 2 * pairs)?),
            };
            launches.extend(bind.lower(ops.partials(group.rank, fp8)?, &buffers, counts)?);
        }
        // 3. After every partial: block o of each member's send buffer to
        // group rank o, compact source-major in o's receive buffer, in owner
        // order on every member, over the bucket's state.
        for owner in 0..cp.size {
            let input = view(own.send, u64::from(owner) * block * F32, block * F32)?;
            launches.extend(self.group_transfer(
                cp,
                transfers.owner(group.rank, owner)?,
                rows,
                input,
                |state| Ok(state.receive),
                |state| {
                    state
                        .partial_gathers
                        .get(index)
                        .copied()
                        .ok_or(Refusal("no Qwen3.8 partial gather state of this bucket"))
                },
            )?);
        }
        // 4. The rank's own heads, merged over the group's blocks into the
        // rows the output gate reads: each member's block holds the
        // bucket's rows of every head.
        let received = view(own.receive, 0, u64::from(cp.size) * block * F32)?;
        let mut bind = Bind::default();
        let merge = PartialMergeBuffers {
            partials: bind.view(received),
            rank_rows: bucket,
            input_heads: dims.heads,
            head_offset: 0,
            live_rows: bind.view(live),
            output: bind.view(at(s.attn_out, first, row_bytes)),
        };
        launches.extend(bind.lower(&ops.partial_merge, &merge, Counts::rows(rows))?);
        // 5. The piece ends once every member has merged: no member's next
        // owner gather or query gather overwrites a buffer a peer still
        // reads.
        launches.extend(self.group_transfer(
            cp,
            set.barrier(group.rank)?,
            rows,
            own.barrier_input,
            |state| Ok(state.barrier_output),
            |state| Ok(state.barrier),
        )?);
        Ok(launches)
    }
}
