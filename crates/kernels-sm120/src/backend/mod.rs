//! The SM120 backend (`kernel_core::backend`): today's typed builders and
//! planners behind the contract families, with SM120's limits applied when
//! an op is prepared. Each family's ops live in their own module.
//!
//! A model constructs [`Sm120`] from its SM120 selection: the symbols of the
//! entries it instantiates (RMSNorm), the row (per-head RMSNorm among them),
//! latent attention and recurrent entries of its catalog, the projection
//! offers its precision documents bind, its attention variants, its gated
//! and sparse attention entries, its encoder catalog, its expert entries,
//! the RoPE and KV write rows, the pooled indexer entries, the Sinkhorn
//! entries it instantiates, its hyper-connection all-reduce transports, its
//! exact fused all-reduce entries, the BF16-partials all-reduce's entries
//! its create admitted and its partial rounding rows, its context-parallel
//! entries and the row copy entries its build compiles. Kernel-base entries
//! (the F32 all-reduce, the all-gather and the row gather and exchange)
//! need no selection.

use kernel_core::attention::{AttentionBuffers, AttentionOp};
use kernel_core::backend::{Backend, Need, Prepared, Refused, View};
use kernel_core::collective::{
    AllReduceBuffers, FusedAllReduceBuffers, GatherBuffers, HcAllReduceBuffers, ReduceInput,
    RowGatherBuffers, ShardedRows,
};
use kernel_core::compressed_kv::{CompressedKvWriteBuffers, CompressedKvWriteQuery};
use kernel_core::context_parallel::{
    ContextParallelQuery, GroupTransferBuffers, GroupTransferOp, LocalizeKeysBuffers,
    OwnedKeyWriteOp, OwnedKvWriteOp, PagedPartialsBuffers, PagedPartialsOp, PartialMergeBuffers,
};
use kernel_core::contract::{
    CommQuery, DType, GemmQuery, LaunchPlan, MoeQuery, MultimodalQuery, NgramQuery, RecurrentQuery,
    Refusal, RowQuery, SigmoidMeanHyperConnectionQuery, SparseAttentionQuery,
};
use kernel_core::gated_mlp::GatedMlpProjectionQuery;
use kernel_core::gated_pooling::{GatedPoolingBuffers, GatedPoolingQuery};
use kernel_core::gated_sparse::{
    BlockScoresBuffers, BlockScoresOp, BlockTopKBuffers, GatedKvWriteBuffers, GatedKvWriteOp,
    IndexQueryBuffers, OutputGateBuffers, OutputGateOp, SelectedPartialsBuffers,
    SelectedPartialsOp, SparseAttentionBuffers, SparseAttentionOp, SparseKeyWriteBuffers,
    SparseKeyWriteOp,
};
use kernel_core::latent_attention::{
    HeadBatchedGemmQuery, HeadGemmBuffers, LatentAttentionBuffers, LatentAttentionQuery,
    LatentKvWriteBuffers,
};
use kernel_core::moe::{
    CombineBuffers, ExpertCombineOp, ProjectedRouteBuffers, RouteBuffers, RoutedSumBuffers,
    RouterBuffers, SharedCombineBuffers,
};
use kernel_core::moe_experts::PreparedExperts;
use kernel_core::multimodal::{
    EmbeddingOverrideBuffers, EncoderAttentionBuffers, EncoderGemmBuffers, EncoderHeadBuffers,
    EncoderRowBuffers,
};
use kernel_core::occupancy::EntryResources;
use kernel_core::per_channel_decay_delta_rule::PerChannelDecayOp;
use kernel_core::pooled_indexer::{
    CandidateBlocksBuffers, CandidateBlocksQuery, IndexerKeyWriteBuffers, IndexerMergeBuffers,
    IndexerSelectBuffers, PooledIndexerQuery, PositionIndexerQuery, PositionKeyWriteBuffers,
    PositionQueryBuffers, PositionSelectBuffers, SelectionCompactBuffers,
};
use kernel_core::position_blocks::{
    CandidateBlockIdsBuffers, PositionBlockIndexerQuery, PositionBlockSelectBuffers,
};
use kernel_core::precision::buffers::ProjectionBuffers;
use kernel_core::precision::execution::Binding;
use kernel_core::precision::{OperationSpec, OperationUse};
use kernel_core::properties::Invariance;
use kernel_core::recurrent::{ConvBuffers, DeltaRuleBuffers, GatedNormBuffers};
use kernel_core::rmsnorm::{RmsNormBuffers, RmsNormLayout, RmsNormRequirement};
use kernel_core::rope_kv::{
    RingWriteBuffers, RingWriteOp, RopeBuffers, RopeKvBuffers, RopeKvOp, RopeOp,
};
use kernel_core::row_copy::{
    ColumnSlice, GatherByIndex, GatherByIndexBuffers, RowCopyBuffers, StreamBroadcast,
};
use kernel_core::rows::{
    ArgmaxFinalBuffers, ArgmaxPartialBuffers, BottleneckBuffers, EmbeddingBuffers,
    GroupedNormBuffers, InjectBuffers, LogitsBuffers, MixBuffers, NgramBuffers, PleGateBuffers,
};
use kernel_core::sinkhorn_hyper_connection::{
    LaggedMixCollapseBuffers, MixCollapseBuffers, MixCollapseNormBuffers,
    ReduceExpandMixNormBuffers, SinkhornHyperConnectionQuery, StreamExpandBuffers,
    StreamHeadBuffers, WeightedStreamHeadBuffers,
};

use crate::linear::{self, channel_decay};

mod all_reduce;
mod attention;
mod bf16_all_reduce;
mod block_quantize;
mod context_parallel;
mod f32_projection;
mod fp4_key_write;
mod fused_all_reduce;
mod gated_pooling;
mod gated_sparse;
mod gather;
mod hc_all_reduce;
mod latent;
mod latent_sources;
mod moe_chained;
mod moe_experts;
mod moe_route;
mod multimodal;
mod mxfp8_fold;
pub mod own;
mod pooled_indexer;
mod position_blocks;
mod position_indexer;
mod position_query;
mod projection;
mod recurrent;
mod rmsnorm;
mod rope_kv;
mod rounded_all_reduce;
mod row_copies;
mod row_gather;
mod rows;
mod select;
mod silu_mul_quantize;
mod sinkhorn;
mod sinkhorn_lagged;

pub use attention::AttentionEntries;
pub use bf16_all_reduce::{
    LOW_LATENCY_ROWS as BF16_LOW_LATENCY_ROWS, TURNS_ROWS as BF16_TURNS_ROWS, Tuned as Bf16Tuned,
    phased_admission_plans as bf16_phased_admission_plans, tuned as bf16_tuned,
};
pub use f32_projection::F32ProjectionOffers;
pub use gated_sparse::GatedSparseEntries;
pub use moe_experts::{TILE_PAIR_MIN_ROWS, grouped_contract, locked_resources};
pub use moe_route::FusedRouterEntry;
pub use projection::{
    DenseForm, NamedOffers, OfferList, ProjectionOffer, ProjectionOffers, ProjectionOp,
};
pub use rope_kv::RopeKvEntry;
pub use rounded_all_reduce::{
    LOW_LATENCY_ROWS as ROUNDED_LOW_LATENCY_ROWS,
    phased_admission_plans as rounded_phased_admission_plans,
};

use crate::encoder::Entry;
use kernel_core::multimodal::EncoderStage;

/// The dynamic shared memory one SM120 block may use.
pub const SHARED_MEMORY_BYTES: u32 = 101_376;

/// The most rows one prepared op serves. Ops prepare their per-row choices
/// up front, so the capacity bounds the tables they hold.
pub const MAX_CAPACITY: u32 = 1 << 16;

/// Checks a plan against SM120: the geometry every CUDA backend shares and
/// SM120's shared memory.
pub fn check_plan(plan: &LaunchPlan) -> Result<(), Refusal> {
    plan.validate_geometry()?;
    if plan.dynamic_shared_bytes > SHARED_MEMORY_BYTES {
        return Err(Refusal("dynamic shared memory exceeds SM120 limit"));
    }
    Ok(())
}

/// The SM120 backend over a model's selection.
#[derive(Default)]
pub struct Sm120 {
    fp4_writes: Vec<crate::gated_pooling::key_write::Entry>,
    pooling: Vec<crate::gated_pooling::Entry>,
    f32_norms: Vec<(crate::gated_pooling::F32Norm, String)>,
    rmsnorms: Vec<(RmsNormRequirement, String)>,
    projections: Option<Box<dyn ProjectionOffers>>,
    f32_projections: Option<F32ProjectionOffers>,
    rows: rows::Selection,
    encoders: Vec<Entry>,
    rope_kv: Vec<RopeKvEntry>,
    attention: Option<Box<AttentionEntries>>,
    latent: latent::Selection,
    latent_sources: Vec<crate::latent::sources::Entry>,
    recurrent: recurrent::Entries,
    experts: moe_experts::Selection,
    chained_quantized: moe_chained::Selection,
    indexer: pooled_indexer::Selection,
    position_block_indexers: Vec<crate::pooled_indexer::position_blocks::Entry>,
    candidate_block_ids: Vec<crate::pooled_indexer::position_blocks::CandidateEntry>,
    position_indexers: Vec<crate::pooled_indexer::position::Entry>,
    position_queries: Vec<crate::pooled_indexer::position_query::Entry>,
    indexer_candidates: Vec<crate::pooled_indexer::position::CandidateEntry>,
    gated_sparse: Option<Box<GatedSparseEntries>>,
    sinkhorn: sinkhorn::Selection,
    hc_all_reduces: Vec<hc_all_reduce::Transport>,
    hc_decode_all_reduces: Vec<hc_all_reduce::DecodeTransport>,
    exact_all_reduces: Vec<fused_all_reduce::ExactEntries>,
    narrow_collectives: bool,
    moe_routes: moe_route::Selection,
    low_latency_all_reduces: Vec<all_reduce::LowLatency>,
    bf16_all_reduce: bf16_all_reduce::Selection,
    rounded_all_reduce: rounded_all_reduce::Selection,
    context_parallel: context_parallel::Selection,
    row_copies: Vec<crate::row::copy::Entry>,
    row_gathers: Vec<String>,
    row_rmsnorms: Vec<crate::row::RowInstantiation>,
    block_quantizers: Vec<crate::row::block_quantize::Entry>,
    silu_mul_quantizers: Vec<crate::row::silu_mul_quantize::Entry>,
    mxfp8_folds: Vec<crate::gemm::mxfp8_fold::Entry>,
}

impl Sm120 {
    pub fn with_mxfp8_fold(mut self, entry: crate::gemm::mxfp8_fold::Entry) -> Self {
        self.mxfp8_folds.push(entry);
        self
    }

    /// Register authenticated exact F32 projection entries.
    pub fn with_f32_projections(mut self, offers: F32ProjectionOffers) -> Self {
        self.f32_projections = Some(offers);
        self
    }

    /// Register a compiled block quantizer. The backend matches its complete
    /// input rounding, scale rule, output form and width at preparation.
    pub fn with_block_quantize(mut self, entry: crate::row::block_quantize::Entry) -> Self {
        self.block_quantizers.push(entry);
        self
    }

    pub fn with_f32_rmsnorm(
        mut self,
        norm: crate::gated_pooling::F32Norm,
        symbol: impl Into<String>,
    ) -> Self {
        self.f32_norms.push((norm, symbol.into()));
        self
    }

    /// A quantized chained compute stage. Gate/up requires the shared
    /// ceil quantizer in the prepared operation before its down input.
    pub fn with_quantized_chained_projection(
        mut self,
        projection: crate::moe::chained::quantized::Projection,
    ) -> Self {
        self.chained_quantized.projections.push(projection);
        self
    }

    pub fn with_row_bias_router(mut self, router: crate::moe::chained::row_bias::Router) -> Self {
        self.chained_quantized.routers.push(router);
        self
    }

    pub fn with_row_bias_route(mut self, route: crate::moe::chained::row_bias::Route) -> Self {
        self.chained_quantized.routes.push(route);
        self
    }

    /// Register a compiled clamped SwiGLU with ceil E4M3/32 quantization
    /// (`row::silu_mul_quantize`). The backend matches its width, clamp and
    /// amax floor at preparation.
    pub fn with_silu_mul_quantize(mut self, entry: crate::row::silu_mul_quantize::Entry) -> Self {
        self.silu_mul_quantizers.push(entry);
        self
    }

    pub fn new() -> Sm120 {
        Sm120::default()
    }

    /// The model's paged MXFP4 index and E2M1/E4M3 compressed-KV writes.
    pub fn with_fp4_key_writes(
        mut self,
        entries: impl IntoIterator<Item = crate::gated_pooling::key_write::Entry>,
    ) -> Self {
        self.fp4_writes.extend(entries);
        self
    }

    /// The model's F32 pair-pooling and masked RMSNorm entry pairs.
    pub fn with_gated_pooling(
        mut self,
        entries: impl IntoIterator<Item = crate::gated_pooling::Entry>,
    ) -> Self {
        self.pooling.extend(entries);
        self
    }

    /// The model's RMSNorm entry `symbol`, built for `requirement`.
    pub fn with_rmsnorm(
        mut self,
        requirement: RmsNormRequirement,
        symbol: impl Into<String>,
    ) -> Self {
        self.rmsnorms.push((requirement, symbol.into()));
        self
    }

    /// The model's projection offers: an [`OfferList`], or the
    /// [`NamedOffers`] of a provider that binds by authenticated identity.
    pub fn with_projections(mut self, offers: Box<dyn ProjectionOffers>) -> Self {
        self.projections = Some(offers);
        self
    }

    /// A row engine entry of the model's catalog: an embedding gather over
    /// a whole table, a logits argmax or top-k, or a sharded argmax partial
    /// or final over row-major pairs into a dense output.
    pub fn with_row(mut self, entry: crate::row::RowInstantiation) -> Self {
        self.rows.engine.push((entry, false));
        self
    }

    /// A sharded argmax entry the model's build compiles
    /// (`row::argmax::cubin_spec`, the shared symbols of
    /// `row::argmax::entries` or its own), which serves every sharded
    /// argmax partial of its logits or every final.
    pub fn with_row_argmax(mut self, entry: crate::row::argmax::Entry) -> Self {
        self.rows.argmax.push(entry);
        self
    }

    /// The model's embedding gather entry over a vocabulary shard
    /// (`row::embedding` with `sharded`), which subtracts the table's first
    /// vocabulary row.
    pub fn with_sharded_embedding(mut self, entry: crate::row::RowInstantiation) -> Self {
        self.rows.engine.push((entry, true));
        self
    }

    /// A hyper-connection or PLE row entry of the model's catalog
    /// (`row::hyper_connection`). n-gram entries register with
    /// [`Sm120::with_ngram`].
    pub fn with_hyper_connection_row(mut self, entry: crate::row::hyper_connection::Entry) -> Self {
        self.rows.hyper.push(entry);
        self
    }

    /// The model's n-gram hash entry (`row::hyper_connection::ngram_entry`)
    /// and the end-of-document token its sources compile in.
    pub fn with_ngram(mut self, entry: crate::row::hyper_connection::Entry, eos: u32) -> Self {
        self.rows.ngrams.push((entry, eos));
        self
    }

    /// The model's grouped norm entry `symbol` over `norm`'s weights (the
    /// F32 form of a hyper-connection grouped norm).
    pub fn with_grouped_norm(
        mut self,
        norm: crate::comm::epilogue::GroupedRmsNorm,
        symbol: impl Into<String>,
    ) -> Self {
        self.rows.grouped_norms.push((norm, symbol.into()));
        self
    }

    /// The model's encoder catalog entries (`encoder::model_catalog`).
    pub fn with_encoders(mut self, entries: impl IntoIterator<Item = Entry>) -> Self {
        self.encoders.extend(entries);
        self
    }

    /// The model's RoPE and KV write rows: its `rope_kv`, `ring_write` and
    /// `dflash_rope` entries, each with the unread ring argument SM120 fills
    /// for it.
    pub fn with_rope_kv(mut self, entries: Vec<RopeKvEntry>) -> Self {
        self.rope_kv.extend(entries);
        self
    }

    /// The model's attention variants and its choices among them.
    pub fn with_attention(mut self, entries: AttentionEntries) -> Self {
        self.attention = Some(Box::new(entries));
        self
    }

    /// Register a native window and paged compressed latent attention entry.
    pub fn with_latent_sources(mut self, entry: crate::latent::sources::Entry) -> Self {
        self.latent_sources.push(entry);
        self
    }

    /// A latent attention entry of the model's (`latent::entries`): a KV
    /// write, an attention with its merge, or a head-batched GEMM.
    pub fn with_latent(mut self, entry: crate::latent::Entry) -> Self {
        self.latent.entries.push(entry);
        self
    }

    /// The slice-widened form of an FP8 one-split attention entry
    /// (`latent::slices::entries`), which its attention op runs in the
    /// entry's place at every row count.
    pub fn with_latent_slices(mut self, entry: crate::latent::slices::Entry) -> Self {
        self.latent.slices.push(entry);
        self
    }

    /// The column merge of a splitting attention entry
    /// (`latent::merge_columns::entries`), which its attention op runs in
    /// place of the entry's split merge at every row count.
    pub fn with_latent_merge_columns(mut self, entry: crate::latent::merge_columns::Entry) -> Self {
        self.latent.merges.push(entry);
        self
    }

    /// The model's linear-attention entries (convolution, gated delta rule
    /// and gated norm).
    pub fn with_linear(mut self, entries: impl IntoIterator<Item = linear::Entry>) -> Self {
        self.recurrent.linear.extend(entries);
        self
    }

    /// The model's per-channel-decay delta rule entries.
    pub fn with_channel_decay(
        mut self,
        entries: impl IntoIterator<Item = channel_decay::Entry>,
    ) -> Self {
        self.recurrent.channel_decay.extend(entries);
        self
    }

    /// A grouped expert entry of the model's catalog (`moe::Variant`),
    /// which serves the gated MLP projection its query, projection and
    /// route weight point state.
    ///
    /// Whether verify steps run exact entries (`Variant::exact`, which sum
    /// in the one-token entry's order) is the model's policy, stated by
    /// registration: a model registers its exact entries only when its
    /// policy runs them. A verify use then takes the exact entry of its
    /// capacity where one is registered and the ordinary entry otherwise;
    /// other uses take the ordinary entry.
    pub fn with_experts(mut self, variant: crate::moe::Variant) -> Self {
        self.experts.grouped.push(variant);
        self
    }

    /// A chained engine projection of the model's selection
    /// (`moe::chained::Projection`), routed or dense.
    pub fn with_chained_projection(mut self, projection: crate::moe::chained::Projection) -> Self {
        self.experts.chained.push(projection);
        self
    }

    /// The resources ptxas reported for entries of the model's cubins, by
    /// symbol (its `kernels.lock`, [`locked_resources`]): a chained op that
    /// runs a tile-pair or decoded-tile form is admitted against them.
    pub fn with_entry_resources(
        mut self,
        resources: impl IntoIterator<Item = (String, EntryResources)>,
    ) -> Self {
        self.experts.resources.extend(resources);
        self
    }

    /// The model's pooled indexer entries (`pooled_indexer::entries`): key
    /// writes, weighted selections and their key list merges, and the
    /// selection compaction.
    pub fn with_pooled_indexer(
        mut self,
        entries: impl IntoIterator<Item = crate::pooled_indexer::Entry>,
    ) -> Self {
        self.indexer.weighted.extend(entries);
        self
    }

    /// The model's head-sum selection entries
    /// (`pooled_indexer::head_sum::entries`) and their key list merges.
    pub fn with_head_sum_indexer(
        mut self,
        entries: impl IntoIterator<Item = crate::pooled_indexer::head_sum::Entry>,
    ) -> Self {
        self.indexer.head_sum.extend(entries);
        self
    }

    /// The one-split forms of the model's weighted selection entries
    /// (`pooled_indexer::rows::entries`), which a selection op runs in
    /// place of its split kernel and merge at the row counts that take one
    /// split.
    pub fn with_indexer_one_split(
        mut self,
        entries: impl IntoIterator<Item = crate::pooled_indexer::rows::Entry>,
    ) -> Self {
        self.indexer.one_split.extend(entries);
        self
    }

    /// The tree merges of the model's weighted selection entries
    /// (`pooled_indexer::merge_tree::entries`), which a selection op runs
    /// in place of its split merge at two or more splits, where no radix
    /// merge is registered.
    pub fn with_indexer_merge_tree(
        mut self,
        entries: impl IntoIterator<Item = crate::pooled_indexer::merge_tree::Entry>,
    ) -> Self {
        self.indexer.trees.extend(entries);
        self
    }

    /// The radix merges of the model's global selection entries
    /// (`pooled_indexer::merge_radix::entries`), which a selection op runs
    /// in place of its split merge at two or more splits.
    pub fn with_indexer_merge_radix(
        mut self,
        entries: impl IntoIterator<Item = crate::pooled_indexer::merge_radix::Entry>,
    ) -> Self {
        self.indexer.radixes.extend(entries);
        self
    }

    /// The model's block-id merges (`pooled_indexer::block_merge::entries`).
    pub fn with_block_merges(
        mut self,
        entries: impl IntoIterator<Item = crate::pooled_indexer::block_merge::Entry>,
    ) -> Self {
        self.indexer.block_merges.extend(entries);
        self
    }

    /// F32 query rotations compiled beside the registered shared quantizer.
    pub fn with_position_queries(
        mut self,
        entries: impl IntoIterator<Item = crate::pooled_indexer::position_query::Entry>,
    ) -> Self {
        self.position_queries.extend(entries);
        self
    }

    /// Compact Position v3 entries compiled by the model.
    pub fn with_position_block_indexer(
        mut self,
        entries: impl IntoIterator<Item = crate::pooled_indexer::position_blocks::Entry>,
    ) -> Self {
        self.position_block_indexers.extend(entries);
        self
    }

    /// Block-ID producers compiled by the model.
    pub fn with_indexer_candidate_block_ids(
        mut self,
        entries: impl IntoIterator<Item = crate::pooled_indexer::position_blocks::CandidateEntry>,
    ) -> Self {
        self.candidate_block_ids.extend(entries);
        self
    }

    /// Position-indexer score/select entries compiled by the model.
    pub fn with_position_indexer(
        mut self,
        entries: impl IntoIterator<Item = crate::pooled_indexer::position::Entry>,
    ) -> Self {
        self.position_indexers.extend(entries);
        self
    }

    /// Candidate-block pre-filter entries compiled by the model.
    pub fn with_indexer_candidate_blocks(
        mut self,
        entries: impl IntoIterator<Item = crate::pooled_indexer::position::CandidateEntry>,
    ) -> Self {
        self.indexer_candidates.extend(entries);
        self
    }

    /// The model's gated rows, block-sparse indexer and attention entries
    /// and selected partials, in the order prepare tries them.
    pub fn with_gated_sparse(mut self, entries: GatedSparseEntries) -> Self {
        self.gated_sparse = Some(Box::new(entries));
        self
    }

    /// The model's one-CTA Sinkhorn entries (`sinkhorn_hc::entries`): the
    /// mix-collapse, expand and head. Probe entries serve no op.
    pub fn with_sinkhorn(
        mut self,
        entries: impl IntoIterator<Item = crate::sinkhorn_hc::Entry>,
    ) -> Self {
        self.sinkhorn.entries.extend(entries);
        self
    }

    /// The model's split mix-collapse (`sinkhorn_hc::split::entry`), which
    /// serves its mix-collapse in place of the one-CTA entry.
    pub fn with_sinkhorn_split(mut self, entry: crate::sinkhorn_hc::split::Entry) -> Self {
        self.sinkhorn.split.push(entry);
        self
    }

    /// The model's tiled mix-collapse entries (`sinkhorn_hc::tiled::entry`),
    /// which serve the row counts SM120's tile schedule gives their tile
    /// (`sinkhorn_hc::tiled::served_prefill_mix`) in place of the split or
    /// one-CTA entry. Probe entries serve no op.
    pub fn with_sinkhorn_tiled(
        mut self,
        entries: impl IntoIterator<Item = crate::sinkhorn_hc::tiled::Entry>,
    ) -> Self {
        self.sinkhorn.tiled.extend(entries);
        self
    }

    /// The model's split mix-collapse fused with the input norm's tree
    /// numerics (`sinkhorn_hc::mix_norm_tree::entry`), which serves its
    /// mix-collapse and norm (`MixCollapseNorm`).
    pub fn with_sinkhorn_mix_norm(
        mut self,
        entry: crate::sinkhorn_hc::mix_norm_tree::Entry,
    ) -> Self {
        self.sinkhorn.mix_norm.push(entry);
        self
    }

    /// The model's lagged Sinkhorn entries (`sinkhorn_hc::lagged::entries`):
    /// the lagged mix-collapse, the F32 expand and the sigmoid-weighted
    /// head. Probe entries serve no op.
    pub fn with_sinkhorn_lagged(
        mut self,
        entries: impl IntoIterator<Item = crate::sinkhorn_hc::lagged::Entry>,
    ) -> Self {
        self.sinkhorn.lagged.extend(entries);
        self
    }

    /// The model's comm epilogue over the F32 all-reduce's sums
    /// (`comm::sinkhorn_split::entry`).
    pub fn with_sinkhorn_epilogue(mut self, entry: crate::comm::sinkhorn_split::Entry) -> Self {
        self.sinkhorn.epilogues.push(entry);
        self
    }

    /// A hyper-connection all-reduce transport of the model's
    /// (`comm::hyper_connection`) and the guarantees the model declares for
    /// its three entries.
    pub fn with_hc_all_reduce(
        mut self,
        transport: crate::comm::hyper_connection::HcTransport,
        invariance: kernel_core::properties::Invariance,
    ) -> Self {
        self.hc_all_reduces.push((transport, invariance));
        self
    }

    /// A hyper-connection decode transport of the model's
    /// (`comm::hyper_connection_decode`) and the guarantees it declares for
    /// its decode (v2 and v3) and its wide entries.
    pub fn with_hc_decode_all_reduce(
        mut self,
        transport: crate::comm::hyper_connection_decode::HcDecodeTransport,
        decode: Invariance,
        wide: Invariance,
    ) -> Self {
        self.hc_decode_all_reduces
            .push(hc_all_reduce::DecodeTransport {
                transport,
                decode,
                wide,
            });
        self
    }

    /// The model's exact fused all-reduce entries, its verify cubin's
    /// (`comm::verify_cubin_spec`) plain and combine entries, and the
    /// guarantees it declares for them.
    pub fn with_exact_all_reduce(
        mut self,
        plain: impl Into<String>,
        combine: impl Into<String>,
        invariance: Invariance,
    ) -> Self {
        self.exact_all_reduces.push(fused_all_reduce::ExactEntries {
            plain: plain.into(),
            combine: combine.into(),
            invariance,
        });
        self
    }

    /// Keeps graphs off the wide all-reduce entries (fused and
    /// hyper-connection), as for a device that cannot keep their grids
    /// resident beside each other's: they take the phased pairs instead.
    pub fn without_wide_collectives(mut self) -> Self {
        self.narrow_collectives = true;
        self
    }

    /// An entry of a model's route cubin
    /// (`moe::expert_parallel::RouteEntry`) other than a decode route: its
    /// router, a prefill permutation kernel or its shared-expert combine.
    ///
    /// # Panics
    ///
    /// On a decode route, which registers with the router kind it computes
    /// ([`Sm120::with_decode_route`]).
    pub fn with_route_entry(mut self, entry: crate::moe::expert_parallel::RouteEntry) -> Self {
        assert!(
            !moe_route::decode_route(&entry),
            "{} is a decode route: register it with with_decode_route and the router kind it computes",
            entry.symbol
        );
        self.moe_routes.entries.push(entry);
        self
    }

    /// A decode route of a model's route cubin, plain or parallel (which
    /// serves its capacity in place of the plain one), and the router kind
    /// it computes: its route cubin's router's, with that router's
    /// parameters. A decode route selects experts itself, yet its entry
    /// records no scoring, so the model names it here, and the route serves
    /// only ops of that router kind (a parallel route also only those its
    /// geometry's scoring computes). It serves without a router entry
    /// registered beside it, and whichever cubin registers its router.
    pub fn with_decode_route(
        mut self,
        entry: crate::moe::expert_parallel::RouteEntry,
        computes: kernel_core::contract::RouterKind,
    ) -> Self {
        self.moe_routes.decode_routes.push((entry, computes));
        self
    }

    /// A chained local route (`moe::chained::LocalRoute`) whose entries the
    /// model's route cubin compiles under `prefix`.
    pub fn with_local_route(
        mut self,
        route: crate::moe::chained::LocalRoute,
        prefix: impl Into<String>,
    ) -> Self {
        self.moe_routes.local.push((route, prefix.into()));
        self
    }

    /// A row engine router entry of the model's catalog
    /// (`row::builders::router_entry`).
    pub fn with_router(mut self, entry: crate::row::RowInstantiation) -> Self {
        self.moe_routes.rows.push(entry);
        self
    }

    /// The model's sigmoid-bias decode route `symbol` of 256 experts, top 8
    /// (`moe::sigmoid_route::decode_route_cubin_spec`), for up to
    /// `tokens_capacity` tokens, with its declared invariance.
    pub fn with_sigmoid_decode_route(
        mut self,
        symbol: impl Into<String>,
        tokens_capacity: u32,
        invariance: crate::invariance::Invariance,
    ) -> Self {
        self.moe_routes
            .sigmoid
            .push((symbol.into(), tokens_capacity, invariance));
        self
    }

    /// The model's tiled permutation of every expert's routes: its row
    /// engine count, prefix, scan and scatter entries
    /// (`chop::row::permute_tile_*`), which route after a row engine router
    /// ([`Sm120::with_router`]).
    pub fn with_tile_permute(mut self, entries: [crate::row::RowInstantiation; 4]) -> Self {
        self.moe_routes.tiles.push(entries);
        self
    }

    /// A row engine expert combine entry of the model's catalog
    /// (`row::builders::expert_combine_entry`).
    pub fn with_expert_combine(mut self, entry: crate::row::RowInstantiation) -> Self {
        self.moe_routes.rows.push(entry);
        self
    }

    /// A chained routed sum (`moe::chained::RoutedSum`) whose entries the
    /// model's route cubin compiles under `prefix`, in either output form.
    pub fn with_routed_sum(
        mut self,
        sum: crate::moe::chained::RoutedSum,
        prefix: impl Into<String>,
    ) -> Self {
        self.moe_routes.sums.push((sum, prefix.into()));
        self
    }

    /// A chained F32 routed sum that adds a row (`moe::chained::RoutedSum`'s
    /// `add_entry`) whose entry the model's cubin compiles under `prefix`:
    /// the routed sum of its geometry launches it at a step that binds an
    /// addend (`kernel_core::moe::RoutedSumBuffers::addend`).
    pub fn with_routed_sum_add(
        mut self,
        sum: crate::moe::chained::RoutedSum,
        prefix: impl Into<String>,
    ) -> Self {
        self.moe_routes.sum_adds.push((sum, prefix.into()));
        self
    }

    /// A router GEMV of one token bucket with the sigmoid-bias route of 256
    /// experts, top 8, fused into its tail
    /// (`moe::sigmoid_route::fused_router_cubin_spec`), which serves
    /// [`Backend::moe_projected_route`] at its bucket with the claims the
    /// model declares for it. A model registers the entries of the rank
    /// count it serves: two entries of one bucket that launch differently
    /// are refused.
    pub fn with_fused_router(mut self, entry: FusedRouterEntry) -> Self {
        self.moe_routes.fused.push(entry);
        self
    }

    /// The low-latency F32 all-reduce (`comm::ll_f32`, from the model's
    /// `ll_f32::cubin_spec` cubin) for launches of up to `ctas` CTAs: the
    /// grid the model's create admitted against every rank's residency,
    /// since each CTA spins on its counterparts on the peer ranks. Plain
    /// all-reduces take it for the row counts that fit the grid; a model
    /// whose devices cannot keep it resident does not register it.
    pub fn with_low_latency_all_reduce(mut self, ctas: u32) -> Self {
        self.low_latency_all_reduces
            .push(all_reduce::LowLatency { ctas });
        self
    }

    /// The low-latency BF16-partials all-reduce (`comm::ll_bf16`, from the
    /// model's `ll_bf16::cubin_spec` cubin) for launches of up to `ctas`
    /// CTAs: the grid the model's create admitted against every rank's
    /// residency ([`BF16_LOW_LATENCY_ROWS`] rows of its width). BF16 all-
    /// reduces take it at the row counts SM120's table gives it that fit
    /// the grid.
    pub fn with_low_latency_bf16_all_reduce(mut self, ctas: u32) -> Self {
        self.bf16_all_reduce
            .low_latency
            .push(bf16_all_reduce::LowLatency { ctas });
        self
    }

    /// The phased BF16-partials all-reduce entries (`comm::phased_bf16`
    /// one-shot and two-shot, `comm::phased_turns_bf16`, from the model's
    /// cubins), whose grids the model's create admitted against every
    /// rank's residency ([`bf16_phased_admission_plans`]). BF16 all-reduces
    /// take them at the row counts SM120's table gives them; without them
    /// those counts take the composed form.
    pub fn with_phased_bf16_all_reduce(mut self) -> Self {
        self.bf16_all_reduce.phased = true;
        self
    }

    /// The low-latency round entry (`comm::ll_f32::ROUND_SYMBOL`, from the
    /// model's `ll_f32::cubin_spec` cubin) for launches of up to `ctas`
    /// CTAs: the grid the model's create admitted against every rank's
    /// residency ([`ROUNDED_LOW_LATENCY_ROWS`] rows of its width). Rounded
    /// all-reduces take it at the row counts SM120's table gives it that
    /// fit the grid.
    pub fn with_low_latency_rounded_all_reduce(mut self, ctas: u32) -> Self {
        self.rounded_all_reduce
            .low_latency
            .push(rounded_all_reduce::LowLatency { ctas });
        self
    }

    /// The phased F32 all-reduce's rounding entries (`comm::phased_f32`
    /// one-shot round and round two-shot, from the model's cubin), whose
    /// grids the model's create admitted against every rank's residency
    /// ([`rounded_phased_admission_plans`]). Rounded all-reduces
    /// (`CommQuery::AllReduceRounded`) take them at every row count of
    /// whole granules; without them they take the composed form.
    pub fn with_phased_f32_all_reduce(mut self) -> Self {
        self.rounded_all_reduce.phased = true;
        self
    }

    /// The model's partial rounding rows over rows of their width
    /// (`row::rounding`: its own compiled entries of that ABI or the shared
    /// ones), which the BF16-partials all-reduce's composed form and its
    /// comm epilogue, and the rounded all-reduce's composed form, launch.
    pub fn with_partial_rounding(
        mut self,
        rounding: crate::row::rounding::PartialRounding,
    ) -> Self {
        self.bf16_all_reduce.rounding.push(rounding);
        self
    }

    /// The model's context-parallel partial merges and key localizations
    /// (`context_parallel::entries`).
    pub fn with_context_parallel(
        mut self,
        entries: impl IntoIterator<Item = crate::context_parallel::Entry>,
    ) -> Self {
        self.context_parallel.entries.extend(entries);
        self
    }

    /// The model's paged attention partials (`paged_partials::entries`).
    pub fn with_paged_partials(
        mut self,
        entries: impl IntoIterator<Item = crate::paged_partials::Entry>,
    ) -> Self {
        self.context_parallel.paged.extend(entries);
        self
    }

    /// The model's owner-routed writes (`owned_writes::entries`).
    pub fn with_owned_writes(
        mut self,
        entries: impl IntoIterator<Item = crate::owned_writes::Entry>,
    ) -> Self {
        self.context_parallel.owned.extend(entries);
        self
    }

    /// The model's phased group gather (`context_parallel::phased`), which
    /// its group gathers and exchanges take in place of the shared
    /// all-gather; owner gathers and barriers stay on the shared one.
    pub fn with_phased_group_gather(mut self) -> Self {
        self.context_parallel.phased = true;
        self
    }

    /// The guarantees the model declares for its context-parallel entries,
    /// by symbol: partial merges, key localizations, owner-routed writes,
    /// paged and selected partials, head-sum selections with their split
    /// merges, and block-id merges. An op reports the claims of the entries
    /// it launches, a head-sum selection its split merge's (the claim of
    /// the composition, whose output the merge writes); an entry without
    /// claims guarantees nothing.
    ///
    /// # Panics
    ///
    /// On a symbol claimed twice with different guarantees.
    pub fn with_context_parallel_claims(
        mut self,
        claims: impl IntoIterator<Item = (String, Invariance)>,
    ) -> Self {
        for (symbol, invariance) in claims {
            self.context_parallel.claims.insert(symbol, invariance);
        }
        self
    }

    /// A row copy entry the model's build compiles (`row::copy::cubin_spec`,
    /// the shared symbols of `row::copy::entries` or its own), which serves
    /// every stream broadcast or every column slice SM120 takes.
    pub fn with_row_copy(mut self, entry: crate::row::copy::Entry) -> Self {
        self.row_copies.push(entry);
        self
    }

    /// A gather by index entry the model's build compiles
    /// (`row::gather::cubin_spec`, the shared symbol `row::gather::SYMBOL`),
    /// which serves every gather by index SM120 takes.
    pub fn with_row_gather(mut self, symbol: impl Into<String>) -> Self {
        self.row_gathers.push(symbol.into());
        self
    }

    /// A per-head RMSNorm row engine entry of the model's catalog, which
    /// serves RMSNorms of `RmsNormLayout::StridedHeads` whose per-head query
    /// is its query and whose heads its `Grid::RowsHeads` grids.
    pub fn with_row_rmsnorm(mut self, entry: crate::row::RowInstantiation) -> Self {
        self.row_rmsnorms.push(entry);
        self
    }
}

/// Families in the trait's sections and order; each delegates to its module.
impl Backend for Sm120 {
    fn name(&self) -> &str {
        "sm120"
    }

    fn mxfp8_fold_gemm(
        &self,
        op: &kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery,
        need: Need,
    ) -> Result<Prepared<kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers<View>>, Refused> {
        mxfp8_fold::prepare(&self.mxfp8_folds, op, need)
    }

    fn block_quantize(
        &self,
        op: &kernel_core::precision::quantizer::BlockQuantizeQuery,
        need: Need,
    ) -> Result<Prepared<kernel_core::precision::quantizer::BlockQuantizeBuffers<View>>, Refused>
    {
        block_quantize::prepare(&self.block_quantizers, op, need)
    }

    fn silu_mul_quantize(
        &self,
        op: &kernel_core::gated_mlp::quantized::SiluMulQuantizeQuery,
        need: Need,
    ) -> Result<Prepared<kernel_core::gated_mlp::quantized::SiluMulQuantizeBuffers<View>>, Refused>
    {
        silu_mul_quantize::prepare(&self.silu_mul_quantizers, op, need)
    }

    // Rows.

    fn rmsnorm(
        &self,
        op: &RmsNormRequirement,
        need: Need,
    ) -> Result<Prepared<RmsNormBuffers<View>>, Refused> {
        // A registered pooling norm serves its exact requirement; any other
        // F32 norm is a row RMSNorm entry's.
        if op.input == kernel_core::contract::DType::F32
            && self
                .f32_norms
                .iter()
                .any(|(norm, _)| norm.requirement() == op)
        {
            return gated_pooling::norm(&self.f32_norms, op, need);
        }
        match op.layout {
            RmsNormLayout::StridedHeads { .. } => {
                rmsnorm::prepare_heads(&self.row_rmsnorms, op, need)
            }
            RmsNormLayout::ContiguousRows { .. } | RmsNormLayout::StridedRows { .. } => {
                rmsnorm::prepare(&self.rmsnorms, op, need)
            }
        }
    }

    fn embedding_gather(
        &self,
        op: &RowQuery,
        row_offset: u32,
        need: Need,
    ) -> Result<Prepared<EmbeddingBuffers<View>>, Refused> {
        rows::embedding(&self.rows, op, row_offset, need)
    }

    fn logits_top_k(
        &self,
        op: &RowQuery,
        need: Need,
    ) -> Result<Prepared<LogitsBuffers<View>>, Refused> {
        rows::logits(&self.rows, op, need)
    }

    fn sharded_argmax_partial(
        &self,
        op: &RowQuery,
        rank: u32,
        need: Need,
    ) -> Result<Prepared<ArgmaxPartialBuffers<View>>, Refused> {
        rows::argmax_partial(&self.rows, op, rank, need)
    }

    fn sharded_argmax_final(
        &self,
        op: &RowQuery,
        need: Need,
    ) -> Result<Prepared<ArgmaxFinalBuffers<View>>, Refused> {
        rows::argmax_final(&self.rows, op, need)
    }

    fn hc_grouped_norm(
        &self,
        op: &SigmoidMeanHyperConnectionQuery,
        weight: DType,
        need: Need,
    ) -> Result<Prepared<GroupedNormBuffers<View>>, Refused> {
        rows::grouped_norm(&self.rows, op, weight, need)
    }

    fn hc_bottleneck(
        &self,
        op: &SigmoidMeanHyperConnectionQuery,
        need: Need,
    ) -> Result<Prepared<BottleneckBuffers<View>>, Refused> {
        rows::bottleneck(&self.rows, op, need)
    }

    fn hc_mix(
        &self,
        op: &SigmoidMeanHyperConnectionQuery,
        need: Need,
    ) -> Result<Prepared<MixBuffers<View>>, Refused> {
        rows::mix(&self.rows, op, need)
    }

    fn hc_inject(
        &self,
        op: &SigmoidMeanHyperConnectionQuery,
        need: Need,
    ) -> Result<Prepared<InjectBuffers<View>>, Refused> {
        rows::inject(&self.rows, op, need)
    }

    fn ngram_hash_gather(
        &self,
        op: &NgramQuery,
        eos: u32,
        rank: u32,
        need: Need,
    ) -> Result<Prepared<NgramBuffers<View>>, Refused> {
        rows::ngram(&self.rows, op, eos, rank, need)
    }

    fn ple_gate(
        &self,
        op: &NgramQuery,
        need: Need,
    ) -> Result<Prepared<PleGateBuffers<View>>, Refused> {
        rows::ple_gate(&self.rows, op, need)
    }

    // Sinkhorn hyper-connection.

    fn sinkhorn_mix_collapse(
        &self,
        op: &SinkhornHyperConnectionQuery,
        need: Need,
    ) -> Result<Prepared<MixCollapseBuffers<View>>, Refused> {
        sinkhorn::mix_collapse(&self.sinkhorn, op, need)
    }

    fn sinkhorn_mix_collapse_norm(
        &self,
        op: &SinkhornHyperConnectionQuery,
        need: Need,
    ) -> Result<Prepared<MixCollapseNormBuffers<View>>, Refused> {
        sinkhorn::mix_collapse_norm(&self.sinkhorn, op, need)
    }

    fn sinkhorn_lagged_mix_collapse(
        &self,
        op: &SinkhornHyperConnectionQuery,
        need: Need,
    ) -> Result<Prepared<LaggedMixCollapseBuffers<View>>, Refused> {
        sinkhorn_lagged::mix_collapse(&self.sinkhorn, op, need)
    }

    fn sinkhorn_expand(
        &self,
        op: &SinkhornHyperConnectionQuery,
        need: Need,
    ) -> Result<Prepared<StreamExpandBuffers<View>>, Refused> {
        match op {
            SinkhornHyperConnectionQuery::ExpandF32 { .. } => {
                sinkhorn_lagged::expand(&self.sinkhorn, op, need)
            }
            _ => sinkhorn::expand(&self.sinkhorn, op, need),
        }
    }

    fn sinkhorn_head(
        &self,
        op: &SinkhornHyperConnectionQuery,
        need: Need,
    ) -> Result<Prepared<StreamHeadBuffers<View>>, Refused> {
        sinkhorn::head(&self.sinkhorn, op, need)
    }

    fn sinkhorn_weighted_head(
        &self,
        op: &SinkhornHyperConnectionQuery,
        need: Need,
    ) -> Result<Prepared<WeightedStreamHeadBuffers<View>>, Refused> {
        sinkhorn_lagged::head(&self.sinkhorn, op, need)
    }

    fn sinkhorn_reduce_expand_mix_norm(
        &self,
        op: &SinkhornHyperConnectionQuery,
        rank: u32,
        need: Need,
    ) -> Result<Prepared<ReduceExpandMixNormBuffers<View>>, Refused> {
        sinkhorn::reduce_expand_mix_norm(
            &self.sinkhorn,
            &self.low_latency_all_reduces,
            &self.bf16_all_reduce,
            op,
            rank,
            need,
        )
    }

    // Projections.

    fn projection(
        &self,
        op: &OperationSpec,
        usage: OperationUse,
        need: Need,
    ) -> Result<Binding<Prepared<ProjectionBuffers<View>>>, Refused> {
        if op
            .activations
            .iter()
            .any(|a| a.format == kernel_core::contract::ActivationFormat::F32)
        {
            f32_projection::prepare(self.f32_projections.as_ref(), op, usage, need)
        } else {
            projection::prepare(self.projections.as_deref(), op, usage, need)
        }
    }

    // Attention.

    fn attention(
        &self,
        op: &AttentionOp,
        need: Need,
    ) -> Result<Prepared<AttentionBuffers<View>>, Refused> {
        attention::prepare(self.attention.as_deref(), op, need)
    }

    // RoPE and KV writes.

    fn rope(&self, op: &RopeOp, need: Need) -> Result<Prepared<RopeBuffers<View>>, Refused> {
        rope_kv::rope(&self.rope_kv, op, need)
    }

    fn rope_kv_write(
        &self,
        op: &RopeKvOp,
        need: Need,
    ) -> Result<Prepared<RopeKvBuffers<View>>, Refused> {
        rope_kv::rope_kv_write(&self.rope_kv, op, need)
    }

    fn ring_write(
        &self,
        op: &RingWriteOp,
        need: Need,
    ) -> Result<Prepared<RingWriteBuffers<View>>, Refused> {
        rope_kv::ring_write(&self.rope_kv, op, need)
    }

    // Gated and sparse attention.

    fn gated_kv_write(
        &self,
        op: &GatedKvWriteOp,
        need: Need,
    ) -> Result<Prepared<GatedKvWriteBuffers<View>>, Refused> {
        gated_sparse::gated_kv_write(self.gated_sparse.as_deref(), op, need)
    }

    fn output_gate(
        &self,
        op: &OutputGateOp,
        need: Need,
    ) -> Result<Prepared<OutputGateBuffers<View>>, Refused> {
        gated_sparse::output_gate(self.gated_sparse.as_deref(), op, need)
    }

    fn sparse_index_query(
        &self,
        op: &SparseAttentionQuery,
        need: Need,
    ) -> Result<Prepared<IndexQueryBuffers<View>>, Refused> {
        gated_sparse::index_query(self.gated_sparse.as_deref(), op, need)
    }

    fn sparse_key_write(
        &self,
        op: &SparseKeyWriteOp,
        need: Need,
    ) -> Result<Prepared<SparseKeyWriteBuffers<View>>, Refused> {
        gated_sparse::key_write(self.gated_sparse.as_deref(), op, need)
    }

    fn sparse_block_scores(
        &self,
        op: &BlockScoresOp,
        usage: OperationUse,
        need: Need,
    ) -> Result<Prepared<BlockScoresBuffers<View>>, Refused> {
        gated_sparse::block_scores(self.gated_sparse.as_deref(), op, usage, need)
    }

    fn sparse_block_top_k(
        &self,
        op: &SparseAttentionQuery,
        need: Need,
    ) -> Result<Prepared<BlockTopKBuffers<View>>, Refused> {
        gated_sparse::block_top_k(self.gated_sparse.as_deref(), op, need)
    }

    fn sparse_attention(
        &self,
        op: &SparseAttentionOp,
        need: Need,
    ) -> Result<Prepared<SparseAttentionBuffers<View>>, Refused> {
        gated_sparse::attention(self.gated_sparse.as_deref(), op, need)
    }

    fn selected_attention_partials(
        &self,
        op: &SelectedPartialsOp,
        group_rank: u32,
        group_size: u32,
        need: Need,
    ) -> Result<Prepared<SelectedPartialsBuffers<View>>, Refused> {
        gated_sparse::selected_partials(
            self.gated_sparse.as_deref(),
            &self.context_parallel.claims,
            op,
            group_rank,
            group_size,
            need,
        )
    }

    // Latent attention.

    fn latent_kv_write(
        &self,
        op: &LatentAttentionQuery,
        need: Need,
    ) -> Result<Prepared<LatentKvWriteBuffers<View>>, Refused> {
        latent::kv_write(&self.latent, op, need)
    }

    fn latent_attention_sources(
        &self,
        op: &LatentAttentionQuery,
        need: Need,
    ) -> Result<Prepared<kernel_core::latent_attention::LatentSourcesBuffers<View>>, Refused> {
        latent_sources::prepare(&self.latent_sources, op, need)
    }

    fn latent_attention(
        &self,
        op: &LatentAttentionQuery,
        need: Need,
    ) -> Result<Prepared<LatentAttentionBuffers<View>>, Refused> {
        latent::attention(&self.latent, op, need)
    }

    fn head_batched_gemm(
        &self,
        op: &HeadBatchedGemmQuery,
        need: Need,
    ) -> Result<Prepared<HeadGemmBuffers<View>>, Refused> {
        latent::head_gemm(&self.latent, op, need)
    }

    // Pooled indexer.

    fn position_indexer_key_write(
        &self,
        op: &PositionIndexerQuery,
        need: Need,
    ) -> Result<Prepared<PositionKeyWriteBuffers<View>>, Refused> {
        fp4_key_write::index(&self.fp4_writes, op, need)
    }

    fn compressed_kv_write(
        &self,
        op: &CompressedKvWriteQuery,
        need: Need,
    ) -> Result<Prepared<CompressedKvWriteBuffers<View>>, Refused> {
        fp4_key_write::compressed(&self.fp4_writes, op, need)
    }

    fn gated_pooling(
        &self,
        op: &GatedPoolingQuery,
        need: Need,
    ) -> Result<Prepared<GatedPoolingBuffers<View>>, Refused> {
        gated_pooling::prepare(&self.pooling, *op, need)
    }

    fn indexer_key_write(
        &self,
        op: &PooledIndexerQuery,
        group_rank: u32,
        group_size: u32,
        need: Need,
    ) -> Result<Prepared<IndexerKeyWriteBuffers<View>>, Refused> {
        pooled_indexer::key_write(&self.indexer, op, group_rank, group_size, need)
    }

    fn indexer_select(
        &self,
        op: &PooledIndexerQuery,
        top_k: u32,
        rank: u32,
        need: Need,
    ) -> Result<Prepared<IndexerSelectBuffers<View>>, Refused> {
        pooled_indexer::select(
            &self.indexer,
            &self.context_parallel.claims,
            op,
            top_k,
            rank,
            need,
        )
    }

    fn indexer_merge(
        &self,
        op: &ContextParallelQuery,
        top_k: u32,
        need: Need,
    ) -> Result<Prepared<IndexerMergeBuffers<View>>, Refused> {
        pooled_indexer::merge(
            &self.indexer,
            &self.context_parallel.claims,
            op,
            top_k,
            need,
        )
    }

    fn indexer_compact(
        &self,
        op: &PooledIndexerQuery,
        need: Need,
    ) -> Result<Prepared<SelectionCompactBuffers<View>>, Refused> {
        pooled_indexer::compact(&self.indexer, op, need)
    }

    fn position_indexer_query(
        &self,
        op: &PositionIndexerQuery,
        need: Need,
    ) -> Result<Prepared<PositionQueryBuffers<View>>, Refused> {
        position_query::prepare(&self.position_queries, self, op, need)
    }

    fn position_indexer(
        &self,
        op: &PositionIndexerQuery,
        top_k: u32,
        position_capacity: u32,
        need: Need,
    ) -> Result<Prepared<PositionSelectBuffers<View>>, Refused> {
        position_indexer::select(&self.position_indexers, op, top_k, position_capacity, need)
    }

    fn indexer_candidate_blocks(
        &self,
        op: &CandidateBlocksQuery,
        position_capacity: u32,
        need: Need,
    ) -> Result<Prepared<CandidateBlocksBuffers<View>>, Refused> {
        position_indexer::candidates(&self.indexer_candidates, op, position_capacity, need)
    }

    fn position_block_indexer(
        &self,
        op: &PositionBlockIndexerQuery,
        top_k: u32,
        position_capacity: u32,
        need: Need,
    ) -> Result<Prepared<PositionBlockSelectBuffers<View>>, Refused> {
        position_blocks::select(
            &self.position_block_indexers,
            op,
            top_k,
            position_capacity,
            need,
        )
    }

    fn indexer_candidate_block_ids(
        &self,
        op: &CandidateBlocksQuery,
        position_capacity: u32,
        need: Need,
    ) -> Result<Prepared<CandidateBlockIdsBuffers<View>>, Refused> {
        position_blocks::candidates(&self.candidate_block_ids, op, position_capacity, need)
    }

    // Recurrent layers.

    fn causal_conv1d(
        &self,
        op: &RecurrentQuery,
        items: u32,
        need: Need,
    ) -> Result<Prepared<ConvBuffers<View>>, Refused> {
        recurrent::causal_conv1d(&self.recurrent, op, items, need)
    }

    fn gated_delta_rule(
        &self,
        op: &RecurrentQuery,
        items: u32,
        need: Need,
    ) -> Result<Prepared<DeltaRuleBuffers<View>>, Refused> {
        recurrent::gated_delta_rule(&self.recurrent, op, items, need)
    }

    fn gated_rmsnorm(
        &self,
        op: &RecurrentQuery,
        need: Need,
    ) -> Result<Prepared<GatedNormBuffers<View>>, Refused> {
        recurrent::gated_rmsnorm(&self.recurrent, op, need)
    }

    fn short_conv(
        &self,
        op: &PerChannelDecayOp,
        items: u32,
        need: Need,
    ) -> Result<Prepared<ConvBuffers<View>>, Refused> {
        recurrent::short_conv(&self.recurrent, op, items, need)
    }

    fn channel_decay_delta_rule(
        &self,
        op: &PerChannelDecayOp,
        items: u32,
        need: Need,
    ) -> Result<Prepared<DeltaRuleBuffers<View>>, Refused> {
        recurrent::channel_decay_delta_rule(&self.recurrent, op, items, need)
    }

    // MoE.

    fn moe_route(
        &self,
        router: &RowQuery,
        routes: &MoeQuery,
        rank: u32,
        route_tile: u32,
        usage: OperationUse,
        need: Need,
    ) -> Result<Prepared<RouteBuffers<View>>, Refused> {
        moe_route::route(
            &self.moe_routes,
            router,
            routes,
            rank,
            route_tile,
            usage,
            need,
        )
    }

    fn moe_projected_route(
        &self,
        projection: &GemmQuery,
        router: &RowQuery,
        routes: &MoeQuery,
        rank: u32,
        route_tile: u32,
        usage: OperationUse,
        need: Need,
    ) -> Result<Prepared<ProjectedRouteBuffers<View>>, Refused> {
        moe_route::projected_route(
            &self.moe_routes,
            projection,
            router,
            routes,
            rank,
            route_tile,
            usage,
            need,
        )
    }

    fn moe_router(
        &self,
        op: &RowQuery,
        need: Need,
    ) -> Result<Prepared<RouterBuffers<View>>, Refused> {
        moe_route::router(&self.moe_routes, op, need)
    }

    fn moe_row_bias_router(
        &self,
        op: &kernel_core::router::RowBiasRouterQuery,
        need: Need,
    ) -> Result<Prepared<kernel_core::router::RowBiasRouterBuffers<View>>, Refused> {
        moe_chained::router(&self.chained_quantized.routers, op, need)
    }

    fn moe_row_bias_route(
        &self,
        router: &kernel_core::router::RowBiasRouterQuery,
        routes: &MoeQuery,
        rank: u32,
        route_tile: u32,
        usage: OperationUse,
        need: Need,
    ) -> Result<Prepared<kernel_core::router::RowBiasRouteBuffers<View>>, Refused> {
        moe_chained::route(
            &self.chained_quantized.routes,
            router,
            routes,
            rank,
            route_tile,
            usage,
            need,
        )
    }

    fn moe_shared_expert_combine(
        &self,
        op: &MoeQuery,
        routes: &MoeQuery,
        rank: u32,
        need: Need,
    ) -> Result<Prepared<SharedCombineBuffers<View>>, Refused> {
        moe_route::shared_expert_combine(&self.moe_routes, op, routes, rank, need)
    }

    fn moe_expert_combine(
        &self,
        op: &ExpertCombineOp,
        need: Need,
    ) -> Result<Prepared<CombineBuffers<View>>, Refused> {
        moe_route::expert_combine(&self.moe_routes, op, need)
    }

    fn moe_routed_sum(
        &self,
        op: &MoeQuery,
        rank: u32,
        need: Need,
    ) -> Result<Prepared<RoutedSumBuffers<View>>, Refused> {
        moe_route::routed_sum(&self.moe_routes, op, rank, need)
    }

    fn gated_mlp_projection(
        &self,
        op: &GatedMlpProjectionQuery,
        usage: OperationUse,
        need: Need,
    ) -> Result<Box<dyn PreparedExperts>, Refused> {
        moe_experts::prepare(&self.experts, &self.moe_routes.local, op, usage, need)
    }

    fn chained_mlp_projection(
        &self,
        op: &kernel_core::moe_chained::ChainedMlpProjectionQuery,
        usage: OperationUse,
        need: Need,
    ) -> Result<Box<dyn PreparedExperts>, Refused> {
        moe_chained::projection(self, &self.chained_quantized.projections, op, usage, need)
    }

    // Collectives.

    fn all_reduce(
        &self,
        op: &CommQuery,
        rank: u32,
        need: Need,
    ) -> Result<Prepared<AllReduceBuffers<View>>, Refused> {
        match op {
            CommQuery::AllReduce {
                dtype: DType::Bf16,
                fused_residual: false,
                fused_rmsnorm: None,
                ..
            } => bf16_all_reduce::prepare(&self.bf16_all_reduce, op, rank, need),
            CommQuery::AllReduceRounded { .. } => rounded_all_reduce::prepare(
                &self.rounded_all_reduce,
                &self.bf16_all_reduce.rounding,
                op,
                rank,
                need,
            ),
            _ => all_reduce::prepare(&self.low_latency_all_reduces, op, rank, need),
        }
    }

    fn fused_all_reduce(
        &self,
        op: &CommQuery,
        input: ReduceInput,
        usage: OperationUse,
        rank: u32,
        need: Need,
    ) -> Result<Prepared<FusedAllReduceBuffers<View>>, Refused> {
        fused_all_reduce::prepare(
            &self.exact_all_reduces,
            !self.narrow_collectives,
            op,
            input,
            usage,
            rank,
            need,
        )
    }

    fn hc_all_reduce(
        &self,
        op: &SigmoidMeanHyperConnectionQuery,
        input: DType,
        weight: DType,
        usage: OperationUse,
        rank: u32,
        need: Need,
    ) -> Result<Prepared<HcAllReduceBuffers<View>>, Refused> {
        hc_all_reduce::prepare(
            &self.hc_all_reduces,
            &self.hc_decode_all_reduces,
            !self.narrow_collectives,
            op,
            input,
            weight,
            usage,
            rank,
            need,
        )
    }

    fn all_gather(
        &self,
        op: &CommQuery,
        rank: u32,
        target: Option<u32>,
        need: Need,
    ) -> Result<Prepared<GatherBuffers<View>>, Refused> {
        gather::all_gather(op, rank, target, need)
    }

    fn gather_rows(
        &self,
        op: &ShardedRows,
        rank: u32,
        target: u32,
        need: Need,
    ) -> Result<Prepared<RowGatherBuffers<View>>, Refused> {
        gather::gather_rows(op, rank, target, need)
    }

    fn exchange_rows(
        &self,
        op: &ShardedRows,
        rank: u32,
        need: Need,
    ) -> Result<Prepared<GatherBuffers<View>>, Refused> {
        gather::exchange_rows(op, rank, need)
    }

    // Context parallelism.

    fn partial_merge(
        &self,
        op: &ContextParallelQuery,
        need: Need,
    ) -> Result<Prepared<PartialMergeBuffers<View>>, Refused> {
        context_parallel::partial_merge(&self.context_parallel, op, need)
    }

    fn localize_keys(
        &self,
        op: &ContextParallelQuery,
        group_rank: u32,
        need: Need,
    ) -> Result<Prepared<LocalizeKeysBuffers<View>>, Refused> {
        context_parallel::localize_keys(&self.context_parallel, op, group_rank, need)
    }

    fn paged_attention_partials(
        &self,
        op: &PagedPartialsOp,
        group_rank: u32,
        group_size: u32,
        need: Need,
    ) -> Result<Prepared<PagedPartialsBuffers<View>>, Refused> {
        context_parallel::paged_partials(&self.context_parallel, op, group_rank, group_size, need)
    }

    fn owned_kv_write(
        &self,
        op: &OwnedKvWriteOp,
        need: Need,
    ) -> Result<Prepared<GatedKvWriteBuffers<View>>, Refused> {
        context_parallel::owned_kv_write(&self.context_parallel, op, need)
    }

    fn owned_key_write(
        &self,
        op: &OwnedKeyWriteOp,
        group_rank: u32,
        need: Need,
    ) -> Result<Prepared<SparseKeyWriteBuffers<View>>, Refused> {
        context_parallel::owned_key_write(&self.context_parallel, op, group_rank, need)
    }

    fn group_transfer(
        &self,
        op: &GroupTransferOp,
        group_rank: u32,
        need: Need,
    ) -> Result<Prepared<GroupTransferBuffers<View>>, Refused> {
        context_parallel::group_transfer(&self.context_parallel, op, group_rank, need)
    }

    // Multimodal encoder stages.

    fn encoder_rows(
        &self,
        op: &MultimodalQuery,
        need: Need,
    ) -> Result<Prepared<EncoderRowBuffers<View>>, Refused> {
        multimodal::prepare(&self.encoders, op, EncoderStage::Rows, None, need)
    }

    fn encoder_heads(
        &self,
        op: &MultimodalQuery,
        need: Need,
    ) -> Result<Prepared<EncoderHeadBuffers<View>>, Refused> {
        multimodal::prepare(&self.encoders, op, EncoderStage::Heads, None, need)
    }

    fn encoder_attention(
        &self,
        op: &MultimodalQuery,
        scale: f32,
        need: Need,
    ) -> Result<Prepared<EncoderAttentionBuffers<View>>, Refused> {
        multimodal::prepare(
            &self.encoders,
            op,
            EncoderStage::Attention,
            Some(scale),
            need,
        )
    }

    fn encoder_gemm(
        &self,
        op: &MultimodalQuery,
        need: Need,
    ) -> Result<Prepared<EncoderGemmBuffers<View>>, Refused> {
        multimodal::prepare(&self.encoders, op, EncoderStage::Gemm, None, need)
    }

    fn embedding_override(
        &self,
        op: &MultimodalQuery,
        need: Need,
    ) -> Result<Prepared<EmbeddingOverrideBuffers<View>>, Refused> {
        multimodal::prepare(&self.encoders, op, EncoderStage::Override, None, need)
    }

    // Row copies.

    fn broadcast_streams(
        &self,
        op: &StreamBroadcast,
        need: Need,
    ) -> Result<Prepared<RowCopyBuffers<View>>, Refused> {
        row_copies::broadcast_streams(&self.row_copies, op, need)
    }

    fn column_slice(
        &self,
        op: &ColumnSlice,
        need: Need,
    ) -> Result<Prepared<RowCopyBuffers<View>>, Refused> {
        row_copies::column_slice(&self.row_copies, op, need)
    }

    fn gather_by_index(
        &self,
        op: &GatherByIndex,
        need: Need,
    ) -> Result<Prepared<GatherByIndexBuffers<View>>, Refused> {
        row_gather::gather_by_index(&self.row_gathers, op, need)
    }
}

#[cfg(test)]
fn view(handle: u64, bytes: u64) -> View {
    View { handle, bytes }
}

#[cfg(test)]
mod tests {
    use super::rmsnorm::hf;
    use super::*;
    use kernel_core::contract::DType;
    use kernel_core::properties::Invariance;

    #[test]
    fn prepare_refuses_inexact_unserved_and_over_limit_ops() {
        let requirement = hf(4096);
        let backend = Sm120::new().with_rmsnorm(requirement, "chop_row_rmsnorm_test_v1");
        let reduce = |dtype| CommQuery::AllReduce {
            ranks: 2,
            elements: 4096,
            dtype,
            fused_residual: false,
            fused_rmsnorm: None,
        };
        // Exactness past what the op guarantees.
        let needed = Invariance::one_row(32);
        assert!(matches!(
            backend.rmsnorm(&requirement, Need::rows(16).with_exact(needed)),
            Err(Refused::Inexact { .. })
        ));
        assert!(
            backend
                .rmsnorm(
                    &requirement,
                    Need::rows(16).with_exact(Invariance::one_row(16))
                )
                .is_ok()
        );
        // A requirement no entry serves, and one past SM120's shared memory.
        assert!(matches!(
            backend.rmsnorm(&hf(2048), Need::rows(16)),
            Err(Refused::Rule(_))
        ));
        let wide = hf(32768);
        let backend = Sm120::new().with_rmsnorm(wide, "chop_row_rmsnorm_wide_v1");
        assert!(matches!(
            backend.rmsnorm(&wide, Need::rows(16)),
            Err(Refused::Rule(rule)) if rule.contains("shared memory")
        ));
        // Sizes that overflow are refused, not wrapped.
        let huge = CommQuery::AllReduce {
            ranks: 2,
            elements: (1 << 63) + 16384,
            dtype: DType::F32,
            fused_residual: false,
            fused_rmsnorm: None,
        };
        assert!(matches!(
            Sm120::new().all_reduce(&huge, 0, Need::rows(2)),
            Err(Refused::Rule(_))
        ));
        // A capacity past what an op prepares tables for is refused before
        // any table is built.
        let narrow = CommQuery::AllReduce {
            ranks: 2,
            elements: 1,
            dtype: DType::F32,
            fused_residual: false,
            fused_rmsnorm: None,
        };
        assert!(matches!(
            Sm120::new().all_reduce(&narrow, 0, Need::rows(u32::MAX)),
            Err(Refused::Rule(rule)) if rule.contains("at most")
        ));
        assert!(
            Sm120::new()
                .all_reduce(&narrow, 0, Need::rows(MAX_CAPACITY))
                .is_ok()
        );
        // The all-reduce sums plain F32 rows only.
        assert!(matches!(
            Sm120::new().all_reduce(&reduce(DType::Bf16), 0, Need::rows(4)),
            Err(Refused::Rule(_))
        ));
        assert!(matches!(
            Sm120::new().all_reduce(&reduce(DType::F32), 2, Need::rows(4)),
            Err(Refused::Rule(_))
        ));
    }
}
