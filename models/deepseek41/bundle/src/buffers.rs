//! One rank's buffers: the activations of the widest phase, the logits
//! rows, the per-item request state the host allocated, the placed weight
//! views, the serving KV pool and the all-reduce's comm state.
//!
//! Every activation holds `scratch_rows()` rows, the most any phase runs: a
//! prefill chunk, or the widest decode or verify bucket. Each phase binds
//! its first rows. Families add their buffers as they lower stages.

use std::collections::BTreeMap;

use bundle_runtime::state::{Extent, RequestState};
use kernel_core::allocation::Allocation;
use model_deepseek41_contract::own_ops::rope::RopeCosSin;
use model_deepseek41_contract::serving::{
    DECODE_BATCH, MAX_VERIFY_ROWS, PREFILL_TOKENS, SCORED_ROWS,
};
use runtime_bundle_abi::{BundleAbiError, DeviceBufferSpanV1};

use crate::ops::indexer::{FULL_BLOCK, REINDEX_BLOCK};
use crate::shape::Shape;
use model_deepseek41_contract::indexer;

const BF16: u64 = 2;
const F32: u64 = 4;
const I32: u64 = 4;
const U32: u64 = 4;
const U64: u64 = 8;

/// Rows of every activation buffer.
pub fn scratch_rows() -> u32 {
    PREFILL_TOKENS.max(head_rows())
}

/// Most rows the head scores in a decode or verify phase: the widest verify
/// bucket, which holds every decode bucket.
pub fn head_rows() -> u32 {
    crate::ops::verify_rows()
}

/// Cells of one item's staged window: a whole prefill chunk after the
/// window's prior positions.
pub fn staged_slots(shape: &Shape) -> u32 {
    PREFILL_TOKENS + shape.geometry.shape.attention.window - 1
}

/// Rows the local logits shard holds: a prefill's scored rows or a decode
/// or verify phase's.
pub fn logits_rows() -> u32 {
    SCORED_ROWS.max(head_rows())
}

kernel_core::scratch_record! {
    /// One rank's activations and logits.
    #[derive(Clone, Copy, Debug)]
    pub struct Scratch {
        /// The rank's embedding rows of its vocabulary shard (BF16, zero for
        /// other shards' ids).
        pub embedding: value,
        /// The embedding the all-reduce assembled (BF16).
        pub embedded: value,
        /// A sublayer's F32 rank partials, or the embedding's widened
        /// shard rows, which the all-reduce sums.
        pub partial: value,
        /// The sublayer output the all-reduce rounded (BF16).
        pub reduced: value,
        /// The residual streams, `[rows, streams, hidden]` BF16: each
        /// sublayer expands from one of the two into the other, the first
        /// sublayer from `streams`.
        pub streams: value,
        pub expanded: value,
        /// Each mixer's F32 `pre` `[rows, streams]`, which the next
        /// sublayer collapses with: even sublayers write the first, odd the
        /// second. The first sublayer collapses with `first_pre`, the
        /// one-hot `pre` on stream 0 that create uploads.
        pub pre: value,
        pub pre_odd: value,
        pub first_pre: value,
        /// A mixer's F32 `post` `[rows, streams]` and `comb`
        /// `[rows, streams, streams]`, which its sublayer's expand reads.
        pub post: value,
        pub comb: value,
        /// The streams collapsed by a mixer's `pre` (BF16).
        pub collapsed: value,
        /// A sublayer input's RMSNorm (BF16).
        pub normed: value,
        /// The normed rows widened to F32, the exact F32 projections'
        /// input (`ops::exact_f32`), or the F32 final norm's rows, the LM
        /// head's.
        pub normed_wide: value,
        /// The normed rows quantized: E4M3 codes and a UE8M0 scale per 32.
        pub normed_codes: value,
        pub normed_scales: value,
        /// The router's F32 logits and unbiased scores of every expert.
        pub router_logits: value,
        pub router_scores: value,
        /// Each row's bias choice, one U8 (0 text, 1 image), zeroed at
        /// create: the bundle serves text.
        pub bias_select: value,
        /// Each row's top-k expert ids (I32) and route weights (F32).
        pub expert_ids: value,
        pub expert_weights: value,
        /// The rank's route metadata: per-expert counts, offsets and the
        /// experts with routes (`experts + 1` I32 each), and the padded
        /// routed rows (one I32). The sorted rows, whose size the route op
        /// states, are `RankBuffers::sorted_rows`.
        pub route_counts: value,
        pub route_offsets: value,
        pub active_experts: value,
        pub padded_rows: value,
        /// The experts' gate/up output per route, `[rows, top_k, local
        /// intermediate]` BF16, and its E4M3 codes and UE8M0 scales.
        pub expert_intermediate: value,
        pub intermediate_codes: value,
        pub intermediate_scales: value,
        /// The experts' down output per route, `[rows, top_k, hidden]` F32.
        pub routed: value,
        /// The shared expert's F32 gate and up rows `[rows, local
        /// intermediate]`, their activation's E4M3 codes and UE8M0 scales,
        /// and its down projection's F32 partial `[rows, hidden]`, which
        /// the routed sum adds.
        pub shared_gate: value,
        pub shared_up: value,
        pub shared_codes: value,
        pub shared_scales: value,
        pub shared_partial: value,
        /// The query down projection (F32 `[rows, q_rank]`), its F32
        /// RMSNorm, and the norm quantized: E4M3 codes and a UE8M0 scale
        /// per 32.
        pub query_a: value,
        pub query_normed: value,
        pub query_codes: value,
        pub query_scales: value,
        /// The rank's heads' queries (F32 `[rows, heads, head_dim]`), which
        /// the tail RoPE rotates in place and rounds once into the BF16
        /// attention query `native_q`.
        pub query: value,
        pub native_q: value,
        /// The index query (F32 `[rows, index heads, index dim]`) and its
        /// prepared form: tail-rotated and MXFP4 fake-quantized, held
        /// exactly in BF16.
        pub index_query: value,
        pub index_prepared: value,
        /// The window key projection (F32 `[rows, head_dim]`); the window
        /// write's F32 normalized and rotated rows, and their E4M3 codes
        /// and UE8M0 scales per 32.
        pub kv: value,
        pub window_latent: value,
        pub window_codes: value,
        pub window_scales: value,
        /// Each row's window key list (I32 `[rows, window]`), each row's
        /// state slot and each slot's last position in the step (I32),
        /// which the step uploads for the window write.
        pub window_keys: value,
        pub window_rows: value,
        pub window_last: value,
        /// A prefill's staged window, one block of `staged_slots` cells a
        /// phase item: each item's retained ring prefix, gathered, then its
        /// chunk's rows, E4M3 codes `[items, cells, head_dim]` and UE8M0
        /// scales `[items, cells, head_dim / 32]`. BF16 comparison stores
        /// twice as many code bytes and allocates no scale plane.
        pub stage_codes: value,
        pub stage_scales: option,
        /// Each phase item's chunk first and last position, the staged
        /// block's first position and the item itself (the staged
        /// attention's item map), I32 `[items]`; and each state slot's
        /// last ring position before the step, I32 `[slots]`. A prefill
        /// uploads them.
        pub stage_first: value,
        pub stage_last: value,
        pub stage_base: value,
        pub stage_items: value,
        pub ring_last: value,
        /// The step rows' RoPE rows of the window and the compressed table
        /// (F32 `[rows, rope pairs, cos and sin]`).
        pub rope_window: value,
        pub rope_compressed_rows: value,
        /// The index heads' F32 weights `[rows, index heads]`.
        pub index_weights: value,
        /// Reusable F32 scores for full-capacity or bounded eager batches,
        /// or compact `[REINDEX_BLOCK, block slots]` selection.
        pub index_scores: value,
        /// Each row's U64 tag and the candidate source's visible
        /// compressed count (I32), and each row block's U32 live count,
        /// the dense blocks' then the compact blocks', which the step
        /// uploads. Qualified native prefill appends an immutable table of
        /// counts 0..=8192 after that mutable prefix.
        pub row_tags: value,
        pub index_visible: value,
        pub block_live: value,
        /// The candidate source's block IDs that the Reindex sources read:
        /// I32 `[rows, top blocks]`, and each row's U64 tag, I32 visible
        /// count and I32 status.
        pub candidate_ids: value,
        pub candidate_tags: value,
        pub candidate_visible: value,
        pub candidate_status: value,
        /// Each index source's selection, `[index sources, rows, top_k]`
        /// I32 compressed positions, -1 padded, which every layer reading
        /// the source attends over.
        pub index_selected: value,
        /// The rank's heads' attention output (BF16 `[rows, heads,
        /// head_dim]`) and its quantized codes and scales, `wo_a`'s input.
        /// Selected decode/verify KV operands, reused only by ordered attention layers.
        pub selected_cache: option,
        pub attention: value,
        pub attention_codes: value,
        pub attention_scales: value,
        /// The `wo_a` output (F32 `[rows, groups, o_rank]`) and its
        /// quantized codes and scales, `wo_b`'s input.
        pub output_a: value,
        pub output_a_codes: value,
        pub output_a_scales: value,
        /// A ratio-2 KV source's projected F32 keys and gate logits, and a
        /// ratio-1 source's F32 keys, `[rows, head_dim]` each.
        pub compressor_key: value,
        pub compressor_gate: value,
        pub compressor_rows: value,
        /// The normalized compressed latents (F32 `[rows, head_dim]`), and
        /// each row's compressed position (I32, -1 for a row that completes
        /// no group).
        pub compressed: value,
        pub compressed_positions: value,
        /// The projected F32 index keys `[rows, index dim]`.
        pub index_key: value,
        /// Each item's tail entry (I32) the step's pooling reads and
        /// updates, and a verify item's checkpoint entries `[items,
        /// MAX_VERIFY_ROWS]`, uploaded with the step's inputs.
        pub tail_ids: value,
        pub tail_checkpoints: value,
        /// Each index source's I32 status per row, `[index sources, rows]`:
        /// 0, or 1 where the selection met a non-finite score. Zeroed at
        /// create; each step reads its live rows back.
        pub index_status: value,
        /// The phase's U32 status per row: every activation quantizer
        /// (on its own or inside a write, a query preparation or an expert
        /// chain) and every fold GEMM ORs a flag into a row whose input is
        /// invalid and writes that row zeros. The main stream's ops share
        /// the plane; the shared expert has its own plane below. Both are
        /// zeroed before each phase and read back after it.
        pub row_status: value,
        /// The shared expert's independent flags, cleared and checked with
        /// row_status. Neither overlapped branch writes the other's flags.
        pub shared_row_status: value,
        /// Each Engram layer's table rows the host gathered for the step's
        /// rows, `[engram layers, rows, engram_k]` E4M3 codes and
        /// `[engram layers, rows, engram_k / 32]` UE8M0 scales: a row's
        /// hash columns' rows end to end.
        pub engram_codes: value,
        pub engram_scales: value,
        /// One Engram layer's rows decoded to BF16 (`[rows, engram_k]`),
        /// and their quantized codes and scales, `wkv`'s input.
        pub engram_rows: value,
        pub engram_quantized: value,
        pub engram_quantized_scales: value,
        /// The `wkv` output, F32 `[rows, (streams + 1) * hidden]`: a key
        /// per stream, then the value.
        pub engram_kv: value,
        /// The last streams' unrounded F32 collapse, the final norm's input.
        pub head_collapsed: value,
        /// A prefill's F32 final-norm rows that need logits, the head's
        /// input (`[SCORED_ROWS, hidden]`); a decode or verify head reads
        /// `normed_wide`.
        pub head_input: value,
        /// The rank's F32 logits of its vocabulary shard.
        pub head: value,
        /// The decode and verify rows' full F32 logits on the rank that owns
        /// their block.
        pub decode_logits: value,
        /// The prefill's scored rows' full F32 logits, on rank 0.
        pub prefill_logits: value,
    }
}

impl Scratch<Allocation> {
    /// Every buffer of one rank of `shape`.
    pub fn of(shape: &Shape) -> Self {
        Self::of_storage(shape, model_deepseek41_contract::state::KvStorage::Native)
    }
    pub fn of_storage(shape: &Shape, storage: model_deepseek41_contract::state::KvStorage) -> Self {
        let bf16_window = storage == model_deepseek41_contract::state::KvStorage::Bf16;
        let rows = u64::from(scratch_rows());
        let hidden = u64::from(shape.dims().hidden);
        let vocab = u64::from(shape.dims().vocab);
        let shard = u64::from(shape.vocab_rows());
        let streams = u64::from(shape.dims().hc.streams);
        let experts = u64::from(shape.dims().moe.experts);
        let top_k = u64::from(shape.dims().moe.top_k);
        let routed_width = top_k * u64::from(shape.local_inter());
        let attention = &shape.dims().attention;
        let head_dim = u64::from(attention.head_dim);
        let q_rank = u64::from(attention.q_rank);
        let heads = u64::from(shape.local_heads()) * head_dim;
        let output_a = u64::from(shape.local_groups() * attention.o_rank);
        let index = u64::from(shape.dims().indexer.heads * shape.dims().indexer.dim);
        let sources = shape.geometry.shape.csa.index_sources.len() as u64;
        let row = |bytes: u64| Allocation::bytes(rows * bytes);
        // Items of a phase, and state slots: at most the served batch.
        let items = u64::from(DECODE_BATCH);
        let stage_cells = items * u64::from(staged_slots(shape));
        let engram_layers = u64::from(shape.geometry.engram_layers()).max(1);
        let engram_k = u64::from(crate::ops::dense::engram_k(shape.geometry));
        let engram_kv = u64::from(crate::ops::dense::engram_kv_width(shape.geometry));
        Self {
            embedding: row(hidden * BF16),
            embedded: row(hidden * BF16),
            partial: row(hidden * F32),
            reduced: row(hidden * BF16),
            normed_codes: row(hidden),
            normed_scales: row(hidden / 32),
            router_logits: row(experts * F32),
            router_scores: row(experts * F32),
            bias_select: row(1),
            expert_ids: row(top_k * I32),
            expert_weights: row(top_k * F32),
            route_counts: Allocation::bytes((experts + 1) * I32),
            route_offsets: Allocation::bytes((experts + 1) * I32),
            active_experts: Allocation::bytes((experts + 1) * I32),
            padded_rows: Allocation::bytes(I32),
            expert_intermediate: row(routed_width * BF16),
            intermediate_codes: row(routed_width),
            intermediate_scales: row(routed_width / 32),
            routed: row(top_k * hidden * F32),
            shared_gate: row(u64::from(shape.local_inter()) * F32),
            shared_up: row(u64::from(shape.local_inter()) * F32),
            shared_codes: row(u64::from(shape.local_inter())),
            shared_scales: row(u64::from(shape.local_inter()) / 32),
            shared_partial: row(hidden * F32),
            query_a: row(q_rank * F32),
            query_normed: row(q_rank * F32),
            query_codes: row(q_rank),
            query_scales: row(q_rank / 32),
            query: row(heads * F32),
            native_q: row(heads * BF16),
            index_query: row(index * F32),
            index_prepared: row(index * BF16),
            kv: row(head_dim * F32),
            window_latent: row(head_dim * F32),
            window_codes: row(head_dim),
            window_scales: row(head_dim / 32),
            window_keys: row(u64::from(attention.window) * I32),
            window_rows: row(I32),
            // A slot's entry: at most one slot per row.
            window_last: row(I32),
            stage_codes: Allocation::bytes(
                stage_cells * head_dim * if bf16_window { 2 } else { 1 },
            ),
            stage_scales: (!bf16_window).then(|| Allocation::bytes(stage_cells * head_dim / 32)),
            stage_first: Allocation::bytes(items * I32),
            stage_last: Allocation::bytes(items * I32),
            stage_base: Allocation::bytes(items * I32),
            stage_items: Allocation::bytes(items * I32),
            ring_last: Allocation::bytes(items * I32),
            rope_window: row(RopeCosSin::output_bytes(1)),
            rope_compressed_rows: row(RopeCosSin::output_bytes(1)),
            index_selected: row(sources * u64::from(shape.dims().indexer.top_k) * I32),
            index_weights: row(u64::from(shape.dims().indexer.heads) * F32),
            index_scores: Allocation::bytes(index_score_bytes(shape)),
            row_tags: row(U64),
            index_visible: row(I32),
            block_live: Allocation::bytes(
                index_live_bytes()
                    + if crate::ops::indexer::prefill::enabled(shape, storage) {
                        u64::from(PREFILL_TOKENS + 1) * I32
                    } else {
                        0
                    },
            ),
            candidate_ids: row(u64::from(shape.geometry.shape.candidates.top_blocks) * I32),
            candidate_tags: row(U64),
            candidate_visible: row(I32),
            candidate_status: row(I32),
            selected_cache: crate::Kernels::selected_cache_bytes(
                shape.geometry,
                shape.ranks(),
                storage,
            )
            .map(Allocation::bytes),
            attention: row(heads * BF16),
            attention_codes: row(heads),
            attention_scales: row(heads / 32),
            output_a: row(output_a * F32),
            output_a_codes: row(output_a),
            output_a_scales: row(output_a / 32),
            compressor_key: row(head_dim * F32),
            compressor_gate: row(head_dim * F32),
            compressor_rows: row(head_dim * F32),
            compressed: row(head_dim * F32),
            compressed_positions: row(I32),
            index_key: row(u64::from(shape.dims().indexer.dim) * F32),
            tail_ids: row(I32),
            tail_checkpoints: row(u64::from(MAX_VERIFY_ROWS) * I32),
            streams: row(streams * hidden * BF16),
            expanded: row(streams * hidden * BF16),
            pre: row(streams * F32),
            pre_odd: row(streams * F32),
            first_pre: row(streams * F32),
            post: row(streams * F32),
            comb: row(streams * streams * F32),
            collapsed: row(hidden * BF16),
            normed: row(hidden * BF16),
            normed_wide: row(hidden * F32),
            index_status: Allocation::bytes(rows * sources * I32),
            row_status: row(U32),
            shared_row_status: row(U32),
            engram_codes: Allocation::bytes(engram_layers * rows * engram_k),
            engram_scales: Allocation::bytes(engram_layers * rows * engram_k / 32),
            engram_rows: row(engram_k * BF16),
            engram_quantized: row(engram_k),
            engram_quantized_scales: row(engram_k / 32),
            engram_kv: row(engram_kv * F32),
            head_collapsed: row(hidden * F32),
            head_input: Allocation::bytes(u64::from(SCORED_ROWS) * hidden * F32),
            head: Allocation::bytes(u64::from(logits_rows()) * shard * F32),
            decode_logits: Allocation::bytes(u64::from(head_rows()) * vocab * F32),
            prefill_logits: Allocation::bytes(u64::from(SCORED_ROWS) * vocab * F32),
        }
    }
}

/// Row blocks of `rows` rows in the widest phase.
pub fn blocks(rows: u32) -> u32 {
    scratch_rows().div_ceil(rows)
}

/// Bytes of one row block's selection scores: the widest of the dense and
/// the compact selections'.
fn index_score_bytes(shape: &Shape) -> u64 {
    let g = shape.geometry;
    g.shape
        .csa
        .index_sources
        .iter()
        .map(
            |&layer| match indexer::block_selection(g, layer, REINDEX_BLOCK) {
                Some(query) => {
                    u64::from(REINDEX_BLOCK) * u64::from(query.score_slots().unwrap_or(0))
                }
                None => {
                    u64::from(FULL_BLOCK)
                        * u64::from(indexer::position_capacity(g, layer).unwrap_or(0))
                }
            },
        )
        .max()
        .unwrap_or(0)
        * F32
}

/// The U32 live count of row block `block` of `rows` rows in
/// `Scratch::block_live`.
pub fn block_live(live: DeviceBufferSpanV1, rows: u32, block: u32) -> DeviceBufferSpanV1 {
    let first = match rows {
        FULL_BLOCK => 0,
        _ => blocks(FULL_BLOCK),
    };
    DeviceBufferSpanV1::new(
        live.buffer,
        live.offset + u64::from(first + block) * I32,
        I32,
    )
}

fn index_live_bytes() -> u64 {
    u64::from(blocks(FULL_BLOCK) + blocks(REINDEX_BLOCK)) * I32
}

/// Immutable eager counts follow the unchanged mutable 64/4096-row counts.
pub(crate) fn index_prefill_counts(
    live: DeviceBufferSpanV1,
) -> Result<DeviceBufferSpanV1, BundleAbiError> {
    let first = index_live_bytes();
    let bytes = u64::from(PREFILL_TOKENS + 1) * I32;
    if first + bytes > live.length {
        return Err(BundleAbiError::InvalidRange(
            "prefill index count table is absent",
        ));
    }
    let offset = live
        .offset
        .checked_add(first)
        .ok_or(BundleAbiError::InvalidRange(
            "prefill count offset overflows",
        ))?;
    offset
        .checked_add(bytes)
        .ok_or(BundleAbiError::InvalidRange("prefill count end overflows"))?;
    Ok(DeviceBufferSpanV1::new(live.buffer, offset, bytes))
}

pub(crate) fn index_prefill_count(
    live: DeviceBufferSpanV1,
    rows: u32,
) -> Result<DeviceBufferSpanV1, BundleAbiError> {
    if rows > PREFILL_TOKENS {
        return Err(BundleAbiError::InvalidRange(
            "prefill index row count exceeds its table",
        ));
    }
    let table = index_prefill_counts(live)?;
    Ok(DeviceBufferSpanV1::new(
        table.buffer,
        table.offset + u64::from(rows) * I32,
        I32,
    ))
}

/// The per-item request state of one rank: each buffer of the declaration
/// at its offset in the host's state view, in declaration order, by name.
#[derive(Clone, Debug)]
pub struct StateSpans {
    pub buffers: Vec<(&'static str, DeviceBufferSpanV1)>,
}

impl StateSpans {
    /// `state`'s buffers for `extent`, laid end to end in the host's view
    /// `view` of the per-item state.
    pub fn new(
        state: &RequestState,
        extent: Extent,
        view: DeviceBufferSpanV1,
    ) -> Result<Self, BundleAbiError> {
        let mut offset = 0;
        let mut buffers = Vec::with_capacity(state.buffers.len());
        for buffer in &state.buffers {
            let bytes = buffer.bytes(extent);
            if offset + bytes > view.length {
                return Err(BundleAbiError::InvalidRange(
                    "deepseek41 request state exceeds the host's view",
                ));
            }
            buffers.push((
                buffer.name,
                DeviceBufferSpanV1::new(view.buffer, view.offset + offset, bytes),
            ));
            offset += bytes;
        }
        Ok(Self { buffers })
    }

    /// The state buffer `name`, if the declaration has one.
    pub fn named(&self, name: &str) -> Option<DeviceBufferSpanV1> {
        self.buffers
            .iter()
            .find(|(of, _)| *of == name)
            .map(|&(_, span)| span)
    }
}

/// The comm state of one rank's logits moves: the decode and verify
/// exchange's flags and epoch, and the prefill gather's.
#[derive(Clone, Copy, Debug)]
pub struct LogitsComm {
    pub exchange_flags: DeviceBufferSpanV1,
    pub exchange_epoch: DeviceBufferSpanV1,
    pub gather_flags: DeviceBufferSpanV1,
    pub gather_epoch: DeviceBufferSpanV1,
}

/// Separately owned prefill operands and the request's rebased page table.
#[derive(Clone, Copy, Debug)]
pub struct PrefillCache {
    pub data: DeviceBufferSpanV1,
    pub tables: DeviceBufferSpanV1,
}

/// One rank's buffers.
#[derive(Clone, Debug)]
pub struct RankBuffers {
    pub scratch: Scratch<DeviceBufferSpanV1>,
    pub state: StateSpans,
    /// The rank image's placed views, by name.
    pub views: BTreeMap<String, DeviceBufferSpanV1>,
    pub kv_pool: DeviceBufferSpanV1,
    /// The all-reduce's comm state regions, which every rank's op writes
    /// (`ops::ReduceOps::state`), zeroed at create.
    pub comm: Vec<DeviceBufferSpanV1>,
    /// The logits moves' comm state, which every rank's move writes
    /// (`ops::logits::LogitsOps::state`), zeroed at create; `None` without
    /// the logits ops.
    pub logits: Option<LogitsComm>,
    /// The rank's sorted routed rows, as the route ops size them.
    pub sorted_rows: DeviceBufferSpanV1,
    /// The backend workspace the ops take, the most any op states.
    pub workspace: DeviceBufferSpanV1,
    /// Backend scratch reserved for the shared expert on the side stream.
    pub shared_workspace: DeviceBufferSpanV1,
    /// Native single-item prefill operands, separately owned from all live scratch.
    pub prefill_cache: Option<PrefillCache>,
    /// The mixer scratch the mix-collapse lists (`ops::MixScratchBytes`):
    /// its partials and its tickets, zeroed at create.
    pub mix_partials: Option<DeviceBufferSpanV1>,
    pub mix_tickets: Option<DeviceBufferSpanV1>,
    /// The compressed RoPE table (`RopeTable::Compressed`) of every served
    /// position, filled at create.
    pub rope_compressed: DeviceBufferSpanV1,
}

impl RankBuffers {
    /// The placed view `name`, or the error naming it.
    pub fn view(&self, name: &str) -> Result<DeviceBufferSpanV1, String> {
        self.views
            .get(name)
            .copied()
            .ok_or_else(|| format!("the rank image has no view {name}"))
    }
}

/// Index source `source`'s status of the first `rows` rows in `status`
/// (`Scratch::index_status`).
pub fn index_status(status: DeviceBufferSpanV1, source: usize, rows: u32) -> DeviceBufferSpanV1 {
    let stride = u64::from(scratch_rows()) * I32;
    DeviceBufferSpanV1::new(
        status.buffer,
        status.offset + source as u64 * stride,
        u64::from(rows) * I32,
    )
}

/// Index source `source`'s selection of the first `rows` rows in
/// `selected` (`Scratch::index_selected`), `top_k` positions a row.
pub fn index_selected(
    selected: DeviceBufferSpanV1,
    source: usize,
    rows: u32,
    top_k: u32,
) -> DeviceBufferSpanV1 {
    let row = u64::from(top_k) * I32;
    let stride = u64::from(scratch_rows()) * row;
    DeviceBufferSpanV1::new(
        selected.buffer,
        selected.offset + source as u64 * stride,
        u64::from(rows) * row,
    )
}

/// Engram layer `layer`'s (its index among the Engram layers) first `rows`
/// rows of the staged table rows `staged` (`Scratch::engram_codes` or
/// `Scratch::engram_scales`) of an instance of `shape`.
pub fn engram_staged(
    staged: DeviceBufferSpanV1,
    shape: &Shape,
    layer: usize,
    rows: u32,
) -> DeviceBufferSpanV1 {
    let stride = staged.length / u64::from(shape.geometry.engram_layers()).max(1);
    let row = stride / u64::from(scratch_rows());
    DeviceBufferSpanV1::new(
        staged.buffer,
        staged.offset + layer as u64 * stride,
        row * u64::from(rows),
    )
}

/// The first `rows` rows of `span`, a buffer of `scratch_rows()` rows.
pub fn rows(span: DeviceBufferSpanV1, rows: u32) -> DeviceBufferSpanV1 {
    let row = span.length / u64::from(scratch_rows());
    DeviceBufferSpanV1::new(span.buffer, span.offset, row * u64::from(rows))
}

/// A checked row range of a scratch allocation, retaining its row stride.
pub(crate) fn row_range(
    span: DeviceBufferSpanV1,
    first: u32,
    count: u32,
) -> Result<DeviceBufferSpanV1, String> {
    let capacity = scratch_rows();
    let end = first
        .checked_add(count)
        .ok_or("scratch row range overflows")?;
    if count == 0 || end > capacity || !span.length.is_multiple_of(u64::from(capacity)) {
        return Err("scratch row range exceeds its allocation".into());
    }
    let stride = span.length / u64::from(capacity);
    let offset = span
        .offset
        .checked_add(u64::from(first) * stride)
        .ok_or("scratch offset overflows")?;
    let length = u64::from(count) * stride;
    offset.checked_add(length).ok_or("scratch end overflows")?;
    Ok(DeviceBufferSpanV1::new(span.buffer, offset, length))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bundle_runtime::geometry::Ranks;
    use kernel_core::allocation::Traversal;
    use model_deepseek41_contract::geometry::FLASH;

    #[test]
    fn prefill_index_counts_are_disjoint_and_bounded() {
        let handle = runtime_bundle_abi::BundleDeviceBufferHandleV1::from_parts(1, 1);
        let prefix = index_live_bytes();
        let bytes = prefix + u64::from(PREFILL_TOKENS + 1) * 4;
        let full = DeviceBufferSpanV1::new(handle, 256, bytes);
        let table = index_prefill_counts(full).unwrap();
        assert_eq!(table.offset, 256 + prefix);
        for n in [0, 1, 64, 512, 4096, 8192] {
            let count = index_prefill_count(full, n).unwrap();
            assert_eq!(count.offset, table.offset + u64::from(n) * 4);
            assert_eq!(count.length, 4);
            assert!(count.offset + count.length <= table.offset + table.length);
        }
        for size in [FULL_BLOCK, REINDEX_BLOCK] {
            let last = block_live(full, size, blocks(size) - 1);
            assert!(last.offset + last.length <= table.offset);
        }
        assert!(index_prefill_count(full, 8193).is_err());
        assert!(index_prefill_counts(DeviceBufferSpanV1::new(handle, 256, bytes - 1)).is_err());
        assert!(
            index_prefill_counts(DeviceBufferSpanV1::new(handle, u64::MAX - 1, bytes)).is_err()
        );
        assert!(index_prefill_counts(DeviceBufferSpanV1::new(handle, 0, prefix)).is_err());
    }

    #[test]
    fn scratch_ranges_preserve_data_scale_and_status_offsets() {
        let handle = runtime_bundle_abi::BundleDeviceBufferHandleV1::from_parts(1, 1);
        for stride in [4, 64, 2048, 10240, 20480] {
            let full = DeviceBufferSpanV1::new(handle, 256, u64::from(scratch_rows()) * stride);
            let a = row_range(full, 0, 528).unwrap();
            let b = row_range(full, 528, 529).unwrap();
            assert_eq!(a.offset + a.length, b.offset);
            assert_eq!(b.offset + b.length, full.offset + 1057 * stride);
            assert!(row_range(full, scratch_rows(), 1).is_err());
            assert!(row_range(full, u32::MAX, 2).is_err());
            assert!(row_range(full, 0, 0).is_err());
            let invalid = DeviceBufferSpanV1::new(handle, u64::MAX - 1, full.length);
            assert!(row_range(invalid, 0, 1).is_err());
        }
    }

    #[test]
    fn flash_scratch_fits_beside_the_weights() {
        let shape = crate::shape::Shape::new(&FLASH, Ranks::TP4).unwrap();
        let mut bytes = 0;
        Scratch::of(&shape)
            .visit(&mut |allocation: &Allocation| bytes += allocation.allocation_bytes());
        // At a prefill's 8192 rows the experts' F32 routed rows (six
        // 5120-wide routes, 960 MiB), the two stream buffers (640 MiB), the
        // rank's 16 heads' F32 queries and BF16 queries and attention output
        // (768 MiB), one row block's selection scores (256 MiB), the staged
        // windows of 32 items (134 MiB), the index sources' selections
        // (128 MiB), the F32 index queries (128 MiB) and Engram's F32 keys
        // and value (800 MiB) and table rows (243 MiB) dominate the
        // 5.11 GiB.
        assert!(bytes < 11 << 29, "{bytes}");
    }

    #[test]
    fn state_spans_follow_the_declaration() {
        let state = model_deepseek41_contract::state::request_state(
            &FLASH,
            model_deepseek41_contract::state::KvStorage::Native,
            false,
        );
        let extent = Extent {
            served: 2,
            bucket: 2,
            verify_rows: 0,
        };
        let total: u64 = state.buffers.iter().map(|b| b.bytes(extent)).sum();
        let view = DeviceBufferSpanV1::new(
            runtime_bundle_abi::BundleDeviceBufferHandleV1::from_parts(1, 1),
            256,
            total,
        );
        let spans = StateSpans::new(&state, extent, view).unwrap();
        assert_eq!(spans.buffers.len(), state.buffers.len());
        assert_eq!(
            spans.buffers[1].1.offset,
            256 + state.buffers[0].bytes(extent)
        );
        let short = DeviceBufferSpanV1::new(view.buffer, 256, total - 1);
        assert!(StateSpans::new(&state, extent, short).is_err());
    }
    #[test]
    fn bf16_prefill_allocates_one_native_product_plane() {
        use model_deepseek41_contract::state::KvStorage;
        let shape = crate::shape::Shape::new(&FLASH, Ranks::TP4).unwrap();
        let native = Scratch::of_storage(&shape, KvStorage::Native);
        let bf16 = Scratch::of_storage(&shape, KvStorage::Bf16);
        assert_eq!(
            bf16.stage_codes.allocation_bytes(),
            native.stage_codes.allocation_bytes() * 2
        );
        assert!(bf16.stage_scales.is_none());
        assert!(native.stage_scales.unwrap().allocation_bytes() > 0);
        bf16.visit(&mut |a: &Allocation| assert!(a.allocation_bytes() > 0));
    }
}
