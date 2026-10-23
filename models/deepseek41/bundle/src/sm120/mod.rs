//! The bundle's SM120 module: the entries the build compiles and their
//! registration on `Sm120`, and the `OwnOps` implementation over the model's
//! own cubins. Each engine adds one registration file and its module line
//! here. Generic code names nothing in this module beyond constructing the
//! backend and the own ops.

pub mod ceil_cached;
pub mod ceil_parallel;
pub mod ceil_prefill;
pub mod ceil_split;
pub mod confidence_parallel;
pub mod dense_dual;
mod dense_dual_serving;
pub mod dspark;
pub mod dspark_argmax;
#[allow(dead_code, reason = "the build compiles the drafter auxiliary entries")]
mod dspark_aux;
pub mod dspark_entries;
pub mod dspark_peer_embedding;
pub mod dspark_quantize;
#[allow(dead_code, reason = "the build compiles the drafter router")]
mod dspark_router;
pub mod fold_cluster;
pub mod fold_fused;
pub mod fold_local_rows;
mod fold_local_rows_serving;
pub mod hc_cols;
pub mod hc_completion;
pub mod hc_expand_mix;
pub mod hc_finish_norm;
pub mod hc_finish_twin;
pub mod hc_mixer_probe;
#[allow(dead_code, reason = "the build compiles the measured prefill module")]
mod hc_prefill;
pub mod hc_prefill_expand_mix;
pub mod hc_round_chain;
pub mod hc_round_expand_mix;
pub mod hc_round_rows;
pub mod head_last;
pub mod latent;
#[allow(
    dead_code,
    reason = "the build script compiles the qualified cache catalog"
)]
pub(crate) mod latent_cache;
#[allow(
    dead_code,
    reason = "the build script compiles the qualified grouped catalog"
)]
pub(crate) mod latent_cache_groups;
pub mod latent_decode;
pub mod latent_scheduled;
pub mod markov_row;
pub mod markov_tiled;
pub(crate) mod moe_floor;
pub mod mxfp8_fold;
pub mod mxfp8_fold_direct;
pub mod mxfp8_fold_direct_entries;
pub mod mxfp8_fold_stream;
pub mod norm_prefill;
pub mod norm_quantize;
pub mod norm_quantize_cached;
pub mod norm_quantize_tiles;
pub mod norm_query_cached;
#[allow(dead_code, reason = "the build script compiles the cubin spec")]
mod prefill_attention;
#[allow(dead_code, reason = "the build script compiles the cubin spec")]
pub(crate) mod prefill_cache;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod prefill_heads;
pub mod prefill_matrix;
pub mod prefill_tma_matrix;
pub mod prefill_tma_scales;
pub mod projection_reuse;
pub mod query_prefill;
pub mod query_window;
mod query_window_serving;
pub mod replicated_embedding;
pub mod window_fused;
use kernels_sm120::backend::{F32ProjectionOffers, Sm120};

/// The build's identity of the exact F32 projection cubins (`f32_head`,
/// `f32_dense`), which authenticates their offer.
const F32_DEPENDENCY: &str = include_str!(concat!(env!("OUT_DIR"), "/f32_dependency.txt"));

// The model's own kernels: their entries and the `OwnOps` implementation.
#[allow(
    dead_code,
    reason = "instance creation prepares the own ops once the program lands"
)]
pub(crate) mod own_entries;
#[allow(
    dead_code,
    reason = "instance creation prepares the own ops once the program lands"
)]
pub(crate) mod own_ops;
// The build compiles each registration file's cubins; the bundle registers
// their entries.
pub mod acceptance;
pub mod acceptance_entries;
#[allow(dead_code, reason = "the build compiles the entries and declarations")]
mod compressor;
#[allow(dead_code, reason = "the build compiles the entries and declarations")]
mod compressor_bf16;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod dense;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
pub(crate) mod encoder;
pub(crate) mod encoders;
pub mod engram_hash;
pub mod engram_hash_entries;
pub mod engram_hash_rows;
pub mod f32_blocked;
pub mod f32_decode;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod f32_dense;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
pub mod f32_head;
pub mod f32_prefill;
pub mod f32_split;
pub mod f32_wide;
#[allow(dead_code, reason = "the build script compiles the serving family")]
pub mod f32_zero;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod hc;
pub mod host_acceptance_entries;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod indexer;
mod indexer_cached_serving;
#[allow(dead_code, reason = "the build script compiles the publisher entry")]
mod indexer_candidates_cached;
mod indexer_candidates_cached_serving;
mod indexer_hierarchical_serving;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod indexer_mma;
#[allow(dead_code, reason = "the build script compiles the candidate entries")]
mod indexer_mma_prefill_tile;
#[allow(dead_code, reason = "the build script compiles the measured entries")]
mod indexer_mma_tile;
mod indexer_prefill_tile_serving;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod indexer_query;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod indexer_query_parallel;
#[allow(dead_code, reason = "the build script compiles the selector entries")]
mod indexer_select_early;
mod indexer_select_early_serving;
mod indexer_workload_serving;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod latent_glue;
#[allow(dead_code)]
mod latent_window_bf16;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod latent_window_stage;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
pub(crate) mod moe;
mod moe_direct;
#[allow(dead_code, reason = "the build script compiles the locked entries")]
pub(crate) mod moe_direct_kernels;
#[allow(dead_code, reason = "the build script uses the entry declarations")]
mod moe_grouped_entries;
pub mod moe_grouped_prefill;
pub mod moe_native;
pub mod moe_native_entries;
pub mod moe_pipeline;
#[allow(dead_code)]
mod moe_pipeline_entries;
#[allow(dead_code, reason = "the build compiles the frozen vector declaration")]
mod moe_pipeline_vector;
pub(crate) mod moe_router_finish;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod reduce;
pub mod ring_gather;
pub mod ring_gather_entries;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod routed_sum;
pub mod router_norm;
pub mod router_norm_entries;
pub(crate) mod router_prefill;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod rows;
#[allow(dead_code, reason = "the build script compiles the cubin spec")]
mod scale_expand;
#[allow(dead_code, reason = "the build script compiles the cubin specs")]
mod shared;
mod widen;

/// The serving latent cubins and symbols, for the model's kernel lab.
pub fn latent_cubins()
-> impl Iterator<Item = (&'static str, &'static [u8], &'static [&'static str])> {
    crate::CUBINS
        .iter()
        .filter(|cubin| {
            cubin.name.starts_with("latent_sources_")
                || cubin.name == latent_decode::CUBIN
                || cubin.name == latent_cache::CUBIN
                || cubin.name == latent_cache_groups::CUBIN
        })
        .map(|cubin| (cubin.name, cubin.bytes, cubin.symbols))
}

/// The model-owned module holding a selected fold entry, for module loading
/// and the model's kernel tests. Rank-independent cubins serve every topology.
pub fn fold_cubin_for_symbol(symbol: &str) -> Option<(&'static str, &'static [u8])> {
    crate::CUBINS
        .iter()
        .find(|c| c.symbols.contains(&symbol))
        .map(|c| (c.name, c.bytes))
}

/// Locked PDL producers for the model's kernel lab.
pub fn dependency_probes() -> (&'static [u8], &'static [&'static str]) {
    let cubin = crate::CUBINS
        .iter()
        .find(|cubin| cubin.name == kernels_sm120::probes::CUBIN)
        .expect("embedded dependency probes");
    (cubin.bytes, cubin.symbols)
}

/// Register the exact F32 projections of `geometry` at `ranks`, the head's
/// and the dense ones, as one offer over the symbols their loaded cubins
/// hold, authenticated by their build dependency. The model lab exercises
/// the serving registration through it.
pub fn register_f32_projections(
    backend: Sm120,
    geometry: &model_deepseek41_contract::geometry::Geometry,
    ranks: u32,
) -> Result<Sm120, String> {
    let entries = f32_split::entries(geometry, ranks)?
        .into_iter()
        .map(|entry| {
            f32_prefill::serving_entry(geometry, entry)
                .and_then(|entry| projection_reuse::head(geometry, ranks, entry))
                .and_then(|entry| head_last::head(geometry, ranks, entry))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let cubins = [
        f32_split::cubin_name(geometry),
        f32_decode::cubin_name(geometry),
        projection_reuse::HEAD_CUBIN.into(),
        head_last::CUBIN.into(),
        f32_dense::cubin_name(geometry),
        f32_head::cubin_name(geometry),
        dspark_router::cubin_name(geometry),
        f32_prefill::cubin_name(geometry, 32),
        f32_prefill::cubin_name(geometry, 64),
        f32_blocked::serving_cubin_name(geometry),
        f32_wide::cubin_name(geometry),
    ];
    let available = crate::CUBINS
        .iter()
        .filter(|cubin| cubins.iter().any(|name| cubin.name == name))
        .flat_map(|cubin| cubin.symbols.iter().map(|s| (*s).to_owned()))
        .collect();
    let offers = F32ProjectionOffers::new(entries, available, F32_DEPENDENCY.to_owned())
        .map_err(|e| e.to_string())?;
    Ok(backend.with_f32_projections(offers))
}

include!(concat!(env!("OUT_DIR"), "/argument_access.rs"));

impl crate::Kernels {
    /// Whether a prefill step of `rows` rows runs the router in the
    /// MoE-input norm (`router_norm`).
    pub(crate) fn router_in_norm(
        g: &model_deepseek41_contract::geometry::Geometry,
        ranks: u32,
        rows: u32,
    ) -> bool {
        router_norm::applies(g, ranks, rows)
    }

    /// The certifying MoE-input norm's launch from the served norm's
    /// (`router_norm::norm_launch`).
    pub(crate) fn router_in_norm_norm(
        served: Vec<kernel_core::backend::Launch>,
        certified: kernel_core::backend::View,
    ) -> Result<kernel_core::backend::Launch, kernel_core::contract::Refusal> {
        router_norm::norm_launch(served, certified)
    }

    /// The router's certified projection launch (`router_norm::router_launch`).
    pub(crate) fn router_in_norm_router(
        views: &crate::ops::exact_f32::CertifiedRouterViews,
    ) -> Result<kernel_core::backend::Launch, kernel_core::contract::Refusal> {
        router_norm::router_launch(views)
    }

    pub(crate) fn prefill_router(
        shape: &crate::shape::Shape,
        experts: u32,
        base: kernel_core::backend::Prepared<
            kernel_core::precision::buffers::ProjectionBuffers<kernel_core::backend::View>,
        >,
    ) -> Result<
        kernel_core::backend::Prepared<
            kernel_core::precision::buffers::ProjectionBuffers<kernel_core::backend::View>,
        >,
        kernel_core::backend::Refused,
    > {
        if f32_zero::selected(shape.geometry, shape.ranks(), experts) {
            router_prefill::prepare(base)
        } else {
            Ok(base)
        }
    }

    // SM120 TP4, 250 W, 2026-10-19: the complete 128/top-3 Decode-1
    // graph is bitwise and wins all three routing patterns, 198.13..198.16
    // -> 60.88..60.94 us. Other buckets retain their measured selections.
    pub(crate) fn measured_moe_floor(
        shape: &crate::shape::Shape,
        set: model_deepseek41_contract::moe::Experts,
    ) -> Result<Option<crate::ops::moe::floor::Prepared>, kernel_core::backend::Refused> {
        if !moe::streamed_qualified(shape.geometry, shape.ranks())
            || (set.count, set.top_k) != (128, 3)
        {
            return Ok(None);
        }
        let entry = moe::floor_entries(shape.geometry)
            .map_err(kernel_core::backend::Refused::Rule)?
            .into_iter()
            .find(|e| e.experts() == set.count)
            .expect("declared expert bank");
        moe_floor::prepare(
            entry,
            model_deepseek41_contract::moe::router(shape.geometry, set)
                .scaling_factor
                .to_bits(),
        )
        .map(Some)
    }

    pub(crate) fn stream_fold_qualified(
        g: &model_deepseek41_contract::geometry::Geometry,
        query: kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery,
        rows: u32,
        paired: bool,
    ) -> bool {
        mxfp8_fold_stream::qualified(g, query, rows, paired)
            || mxfp8_fold_direct::selected(g, query, rows, paired).is_some()
    }

    pub(crate) fn measured_moe_direct(
        shape: &crate::shape::Shape,
        set: model_deepseek41_contract::moe::Experts,
    ) -> Result<
        Vec<(
            kernel_core::precision::ExecutionPhase,
            u32,
            crate::ops::moe::direct::Prepared,
        )>,
        kernel_core::backend::Refused,
    > {
        moe_direct::prepare_set(shape, set)
    }

    /// Whether a rank loads `cubin`: the scale expansion only where its
    /// image stores a routed scale bank encoded.
    pub(crate) fn loads_scale_expand(cubin: &str, encoded: bool) -> bool {
        cubin != scale_expand::CUBIN || encoded
    }

    /// The launch expanding the encoded scale bank `encoded` into the raw
    /// bank `output` (`kernels_sm120::moe::scale_expand`): the experts the
    /// `active` list names, over its first `slots` positions, or every
    /// expert without a list. Each CTA waits for its predecessor before it
    /// reads or writes, so the expansion follows the routes and the raw
    /// bank's previous readers; every consumer reads after its own wait.
    pub(crate) fn scale_expand(
        bank: model_deepseek41_contract::moe::ScaleBank,
        width: u32,
        slots: u32,
        [encoded, output]: [kernel_core::backend::View; 2],
        active: Option<kernel_core::backend::View>,
        rows: u32,
    ) -> Result<kernel_core::backend::Launch, kernel_core::contract::Refusal> {
        use kernels_sm120::moe::scale_expand as expand;
        let query = expand::ScaleExpandQuery {
            width,
            rows: bank.rows,
            groups: bank.groups,
            projections: bank.projections,
            experts: bank.experts,
        };
        if active.is_none() && slots != bank.experts {
            return Err(kernel_core::contract::Refusal(
                "an expansion without an active list covers every expert",
            ));
        }
        Ok(kernel_core::backend::Launch {
            symbol: expand::symbol(width)?.into(),
            rows,
            plan: expand::plan_slots(query, slots)?,
            arguments: expand::arguments(query, encoded, active, output)?,
            collective: false,
            persistent: false,
        })
    }

    /// Whether the locked entry declares this pointer argument as const input.
    pub(crate) fn const_input_argument(symbol: &str, index: usize) -> bool {
        CONST_INPUT_ARGUMENTS
            .binary_search_by_key(&symbol, |&(name, _)| name)
            .ok()
            .and_then(|found| CONST_INPUT_ARGUMENTS[found].1.get(index))
            .copied()
            .unwrap_or(false)
    }

    /// Upper row bound of the measured query preparation bucket.
    pub(crate) fn latent_query_small_rows() -> u32 {
        latent_scheduled::QUERY_ROWS
    }

    pub(crate) fn prepare_latent_sources(
        &self,
        geometry: &model_deepseek41_contract::geometry::Geometry,
        ranks: u32,
        query: &kernel_core::latent_attention::LatentAttentionQuery,
        need: kernel_core::backend::Need,
    ) -> Result<
        kernel_core::backend::Prepared<
            kernel_core::latent_attention::LatentSourcesBuffers<kernel_core::backend::View>,
        >,
        kernel_core::backend::Refused,
    > {
        let reference =
            prefill_attention::prepare(self.backend.as_ref(), geometry, ranks, query, need)?;
        latent_decode::prepare(reference, geometry, ranks, query, need)
    }

    pub(crate) fn prepare_prefill_heads(
        &self,
        query: &kernel_core::latent_attention::LatentAttentionQuery,
        need: kernel_core::backend::Need,
    ) -> Result<
        kernel_core::backend::Prepared<
            kernel_core::latent_attention::LatentSourcesBuffers<kernel_core::backend::View>,
        >,
        kernel_core::backend::Refused,
    > {
        prefill_heads::prepare(self.backend.as_ref(), query, need)
    }

    pub(crate) fn loads_latent_attention(
        cubin: &str,
        shape: &crate::shape::Shape,
        storage: model_deepseek41_contract::state::KvStorage,
    ) -> bool {
        prefill_attention::loads(cubin, shape.geometry, shape.ranks(), storage)
            && latent_decode::loads(cubin, shape.geometry, shape.ranks(), storage)
            && (!matches!(
                cubin,
                prefill_cache::CUBIN
                    | prefill_heads::NATIVE_CUBIN
                    | prefill_heads::CACHED_CUBIN
                    | prefill_heads::FINISH_CUBIN
            ) || prefill_cache::eligible(shape.geometry, shape.ranks(), storage))
            && (!matches!(cubin, latent_cache::CUBIN | latent_cache_groups::CUBIN)
                || latent_cache::eligible(shape.geometry, shape.ranks(), storage))
    }

    pub(crate) fn selected_cache_bytes(
        g: &model_deepseek41_contract::geometry::Geometry,
        ranks: u32,
        storage: model_deepseek41_contract::state::KvStorage,
    ) -> Option<u64> {
        latent_cache::eligible(g, ranks, storage).then_some(latent_cache::BYTES)
    }

    pub(crate) fn selected_cache_rows(
        phase: kernel_core::contract::AttentionPhase,
        rows: u32,
    ) -> bool {
        latent_cache::selected(phase, rows)
    }

    pub(crate) fn prepare_selected_cache(
        &self,
        geometry: &model_deepseek41_contract::geometry::Geometry,
        query: &kernel_core::latent_attention::LatentAttentionQuery,
        need: kernel_core::backend::Need,
    ) -> Result<
        kernel_core::backend::Prepared<kernels_sm120::latent::sources_cached::Buffers>,
        kernel_core::backend::Refused,
    > {
        latent_cache::prepare(self.backend.as_ref(), geometry, query, need)
    }

    pub(crate) fn prepare_prefill_cache(
        &self,
        query: &kernel_core::latent_attention::LatentAttentionQuery,
        need: kernel_core::backend::Need,
    ) -> Result<
        kernel_core::backend::Prepared<model_deepseek41_contract::own_ops::prefill_cache::Buffers>,
        kernel_core::backend::Refused,
    > {
        prefill_cache::prepare(self.backend.as_ref(), query, need)
    }

    pub(crate) const PREFILL_CACHE_PAGES: u32 = prefill_cache::PAGE_LIMIT;

    pub(crate) fn prefill_cache_eligible(
        geometry: &model_deepseek41_contract::geometry::Geometry,
        ranks: u32,
        storage: model_deepseek41_contract::state::KvStorage,
    ) -> bool {
        prefill_cache::eligible(geometry, ranks, storage)
    }

    pub(crate) fn prefill_cache_layout(
        slots: u32,
        page: u32,
        pages: u32,
    ) -> Result<
        model_deepseek41_contract::own_ops::prefill_cache::Layout,
        kernel_core::contract::Refusal,
    > {
        let layout = prefill_cache::layout(slots, page, pages)?;
        Ok(model_deepseek41_contract::own_ops::prefill_cache::Layout {
            window: layout.window,
            compressed: layout.compressed,
            marks: layout.marks,
            bytes: layout.bytes,
        })
    }

    /// The largest separately prepared dense decode bucket.
    pub(crate) fn dense_decode_capacity() -> u32 {
        mxfp8_fold::SMALL_CAPACITY
    }

    /// Whether `shape` quantizes one dense row of `width` columns of
    /// `input` with its own one-row op (`ceil_split::serves_width`).
    pub(crate) fn dense_quantize_one_row(
        shape: &crate::shape::Shape,
        width: u32,
        input: kernel_core::contract::DType,
    ) -> bool {
        ceil_split::serves_width(shape.geometry, shape.ranks(), width, input)
    }

    /// The startup lines naming the one-row decode twins `shape` serves,
    /// so a served run shows what it measured.
    pub(crate) fn one_row_twin_lines(shape: &crate::shape::Shape) -> Vec<String> {
        let mut lines = Vec::new();
        if query_window::serves(shape.geometry, shape.ranks()) {
            lines.push(format!(
                "{}: one-row decode runs each layer's query RoPE and window write as one launch",
                query_window::SWITCH
            ));
        }
        if ceil_split::serves(shape.geometry, shape.ranks()) {
            let widths = ceil_split::SERVED
                .iter()
                .map(|(width, input, parts)| format!("w{width} {input:?} over {parts} CTAs"))
                .collect::<Vec<_>>();
            lines.push(format!(
                "{}: one-row dense quantizers split: {}",
                ceil_split::SWITCH,
                widths.join(", ")
            ));
        }
        lines
    }

    /// Select the qualified dense fold query for an exact decode row count.
    pub(crate) fn dense_decode_query(
        query: kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery,
        rows: u32,
    ) -> kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery {
        mxfp8_fold::routed_query(query, rows)
    }

    /// Entries used only by the drafter's smaller expert bank. An AR-only
    /// instance has no consumer for these modules.
    pub fn drafter_only(cubin: &str, g: &model_deepseek41_contract::geometry::Geometry) -> bool {
        // Direct routed MoE needs both expert banks in these shared modules.
        if cubin.starts_with("floor_moe_") || cubin.starts_with("bandwidth_moe_") {
            return false;
        }
        cubin == host_acceptance_entries::CUBIN
            || cubin == confidence_parallel::CUBIN
            || cubin == markov_tiled::CUBIN
            || cubin == markov_row::CUBIN
            || cubin == dspark_argmax::CUBIN
            || cubin == dspark_quantize::cubin_name(g)
            || cubin == dspark_peer_embedding::cubin_name(g)
            || cubin == engram_hash_entries::CUBIN
            || cubin == engram_hash_rows::CUBIN
            || cubin == ring_gather_entries::CUBIN
            || cubin == dspark_router::cubin_name(g)
            || dspark_router::sum_cubins(g)
                .iter()
                .any(|name| name == cubin)
    }

    /// Whether an instance loads `cubin` with the program's cubins: every
    /// cubin but the dependency probes, which only the kernel lab launches,
    /// the unwired acceptance reducer, and the vision encoder's, which only
    /// an encoder instance loads.
    /// Streamed MoE modules load only for the qualified topology.
    pub fn loads(cubin: &str, shape: &crate::shape::Shape) -> bool {
        (![
            projection_reuse::FOLD_CUBIN,
            projection_reuse::HEAD_CUBIN,
            head_last::CUBIN,
            prefill_matrix::CUBIN,
            prefill_tma_matrix::CUBIN,
            prefill_tma_scales::CUBIN,
            fold_local_rows::CUBIN,
            fold_cluster::CUBIN,
            fold_fused::CUBIN,
        ]
        .contains(&cubin)
            || projection_reuse::enabled(shape.geometry, shape.ranks()))
            && (cubin != dense_dual::CUBIN || dense_dual::enabled(shape.geometry, shape.ranks()))
            && (cubin != moe_grouped_entries::CUBIN
                || (shape.geometry.id == model_deepseek41_contract::geometry::FLASH.id
                    && shape.ranks() == 4))
            && cubin != kernels_sm120::probes::CUBIN
            && (!ceil_split::is_cubin(cubin) || ceil_split::serves(shape.geometry, shape.ranks()))
            && (cubin != query_window::CUBIN || query_window::serves(shape.geometry, shape.ranks()))
            && cubin != acceptance_entries::CUBIN
            && (!latent_scheduled::candidate_cubin(cubin)
                || latent_scheduled::qualified(shape.geometry, shape.ranks()))
            && !encoder::is_cubin(cubin)
            && (!cubin.starts_with("bandwidth_moe_")
                || moe::streamed_qualified(shape.geometry, shape.ranks()))
            && (!cubin.starts_with("router_finish_moe_")
                || moe::streamed_qualified(shape.geometry, shape.ranks()))
            && (!cubin.starts_with("floor_moe_")
                || moe::streamed_qualified(shape.geometry, shape.ranks()))
            && (cubin != moe_pipeline_entries::CUBIN
                || moe::streamed_qualified(shape.geometry, shape.ranks()))
            && (cubin != moe_pipeline_vector::CUBIN
                || moe::streamed_qualified(shape.geometry, shape.ranks()))
            && (![
                moe_native::CUBIN,
                moe_native::CEIL_CUBIN,
                moe_native::WIDE_CUBIN,
                moe_native::LEAN_CUBIN,
            ]
            .contains(&cubin)
                || (moe_native::requested() && moe_native::eligible(shape.geometry, shape.ranks())))
            && (!cubin.starts_with("prefill_moe_")
                || (cubin.ends_with("_compute_sm120")
                    && moe::streamed_qualified(shape.geometry, shape.ranks())))
            && (!cubin.starts_with("position_mma_")
                || ((cubin.starts_with("position_mma_guard_")
                    || cubin == indexer_mma_tile::CUBIN
                    || cubin == indexer_mma_prefill_tile::CUBIN
                        && shape.runtime.indexer_prefill_tile)
                    && shape.runtime.indexer_mma))
            && (cubin != indexer_candidates_cached::CUBIN
                || shape.runtime.indexer_prefill_tile && shape.runtime.indexer_mma)
            && (cubin != indexer_select_early::CUBIN
                || shape.runtime.indexer_prefill_tile && shape.runtime.indexer_mma)
            && (!cubin.starts_with("position_hierarchical_") || shape.runtime.indexer_mma)
            && (!cubin.starts_with("position_select_cached_") || shape.runtime.indexer_mma)
            && (!cubin.starts_with("streamed_moe_")
                || moe::streamed_qualified(shape.geometry, shape.ranks()))
            && (cubin != router_norm::CUBIN
                || router_norm::applies(shape.geometry, shape.ranks(), router_norm::ROWS))
    }

    /// Model-qualified preparation ranges, in increasing capacity within
    /// each phase. Unchanged topologies retain their original C1 override.
    pub(crate) fn moe_buckets(
        shape: &crate::shape::Shape,
    ) -> &'static [(kernel_core::precision::ExecutionPhase, u32)] {
        use kernel_core::precision::ExecutionPhase::{Decode, Prefill, Verify};
        if moe::streamed_qualified(shape.geometry, shape.ranks()) {
            &[
                (Decode, 1),
                (Decode, 8),
                (Verify, 8),
                (Verify, 64),
                (Prefill, 512),
                (Prefill, 2048),
                (Prefill, 4096),
                (Prefill, 8192),
            ]
        } else {
            &[(Decode, 1)]
        }
    }

    /// The SM120 backend over the entries the build compiles for `shape`,
    /// and the model's own kernels.
    #[cfg(test)]
    pub(crate) fn new(shape: &crate::shape::Shape) -> Result<Self, String> {
        Self::new_for_kv(shape, model_deepseek41_contract::state::KvStorage::Native)
    }

    pub(crate) fn new_for_kv(
        shape: &crate::shape::Shape,
        storage: model_deepseek41_contract::state::KvStorage,
    ) -> Result<Self, String> {
        let sm120 = rows::register(Sm120::new(), shape.geometry, shape.ranks());
        let sm120 = norm_quantize::register(sm120, shape.geometry)?;
        let sm120 = if shape.geometry.id == model_deepseek41_contract::geometry::FLASH.id {
            sm120
                .with_rmsnorm_quantize_tiles(norm_quantize_tiles::entry(shape.geometry), vec![1, 8])
        } else {
            sm120
        };
        let sm120 = norm_quantize_cached::register(sm120, shape.geometry);
        let sm120 = norm_query_cached::register(sm120, shape.geometry);
        let sm120 = norm_prefill::register(sm120, shape.geometry)?;
        let sm120 = query_prefill::register(sm120, shape.geometry)?;
        let sm120 = window_fused::register(sm120);
        let sm120 = reduce::register(sm120, shape.geometry);
        let sm120 = if crate::ops::bf16_partials_requested(shape, storage) {
            // Prefill DMA carries the rounded BF16 partials of the
            // collectives it replaces (`prefill_dma_payload`).
            // Create admits every resident grid before preparing or running
            // any graph. A failed admission refuses the selected instance.
            sm120.with_phased_bf16_all_reduce()
        } else {
            sm120
        };
        let sm120 = hc::register(sm120, shape.geometry);
        let sm120 = hc_prefill::register(sm120, shape.geometry);
        let sm120 = hc_prefill_expand_mix::register(sm120, shape.geometry);
        let sm120 = hc_expand_mix::register(sm120, shape.geometry);
        let sm120 = hc_round_expand_mix::register(sm120, shape.geometry);
        let sm120 = hc_round_chain::register(sm120, shape.geometry);
        let sm120 = hc_finish_norm::register(sm120, shape.geometry);
        let sm120 = replicated_embedding::register(sm120, shape.geometry, shape.ranks());
        let sm120 = latent_scheduled::register_sources(sm120, shape.geometry, shape.ranks())?;
        let sm120 = indexer::register(sm120, shape.geometry);
        let sm120 = if shape.runtime.indexer_mma {
            sm120.with_coarse_guarded_position_scores()
        } else {
            sm120
        };
        let sm120 = compressor::CompressorKernels::serving(shape.geometry)
            .map_err(|e| e.to_string())?
            .register(sm120);
        let sm120 = compressor_bf16::register(sm120, shape.geometry)?;
        let sm120 = moe::register(sm120, shape.geometry, shape.ranks())?;
        let sm120 = routed_sum::register(sm120, shape.geometry);
        let sm120 = dspark_router::register(sm120, shape.geometry);
        let sm120 = ceil_prefill::register(sm120, shape.geometry, shape.ranks())?;
        let sm120 = shared::register(sm120, shape.geometry, shape.ranks())?;
        let sm120 = mxfp8_fold::register(sm120, shape.geometry, shape.ranks())?;
        let sm120 = mxfp8_fold_stream::register(sm120, shape.geometry, shape.ranks())?;
        let sm120 = mxfp8_fold_direct::register(sm120, shape.geometry, shape.ranks())?;
        let sm120 = projection_reuse::register(sm120, shape.geometry, shape.ranks())?;
        let sm120 = prefill_matrix::register(sm120, shape.geometry, shape.ranks())?;
        let sm120 = prefill_tma_matrix::register(sm120, shape.geometry, shape.ranks())?;
        let sm120 = prefill_tma_scales::register(sm120, shape.geometry, shape.ranks())?;
        let sm120 = indexer_query_parallel::register_parallel(sm120, shape.geometry);
        let sm120 = latent_glue::register(sm120, shape.geometry, shape.ranks())?;
        let sm120 = latent_scheduled::register_query(sm120, shape.geometry, shape.ranks());
        let sm120 = ceil_cached::register(sm120, shape.geometry, shape.ranks())?;
        let sm120 = ceil_split::register(sm120, shape.geometry, shape.ranks())?;
        let sm120 = latent_window_stage::register(sm120, shape.geometry, shape.ranks())?;
        let sm120 = dspark_aux::register(sm120, shape.geometry, shape.ranks())?;
        let sm120 = latent_window_bf16::register(sm120);
        let sm120 = register_f32_projections(sm120, shape.geometry, shape.ranks())?;
        // The vision encoder's shared entries (`crate::encoders`).
        let sm120 = encoder::register(sm120)?;
        Ok(Self {
            backend: Box::new(sm120),
            own: Box::new(own_ops::Sm120Own::new(shape.geometry)),
            gather_state: [
                kernels_sm120::comm::GATHER_ROWS_FLAG_BYTES,
                kernels_sm120::comm::GATHER_ROWS_EPOCH_BYTES,
            ],
        })
    }

    /// The eager indexer uses the same guarded registrations as the target.
    pub(crate) fn prefill_indexer_backend(
        shape: &crate::shape::Shape,
    ) -> Box<dyn kernel_core::backend::Backend> {
        Box::new(
            indexer::register(Sm120::new(), shape.geometry).with_coarse_guarded_position_scores(),
        )
    }

    /// BF16 keys do not use the native scorer or its coarse guard.
    pub(crate) fn loads_indexer(
        cubin: &str,
        storage: model_deepseek41_contract::state::KvStorage,
    ) -> bool {
        !(cubin.starts_with("position_mma_guard_fast_")
            || cubin == indexer_mma_tile::CUBIN
            || cubin == indexer_mma_prefill_tile::CUBIN
            || cubin == indexer_candidates_cached::CUBIN
            || cubin == indexer_select_early::CUBIN
            || cubin.starts_with("position_hierarchical_")
            || cubin.starts_with("position_select_cached_"))
            || storage == model_deepseek41_contract::state::KvStorage::Native
    }

    /// Only the target's optional prefill reduction registers this entry.
    pub(crate) fn prefill_reduction_backend(
        shape: &crate::shape::Shape,
    ) -> Box<dyn kernel_core::backend::Backend> {
        Box::new(reduce::register(Sm120::new(), shape.geometry).with_phased_f32_all_reduce())
    }

    pub(crate) fn prefill_reduction_plans()
    -> Result<Vec<(&'static str, kernel_core::contract::LaunchPlan)>, String> {
        kernels_sm120::backend::rounded_phased_admission_plans().map_err(|error| error.to_string())
    }

    pub(crate) fn bf16_partial_plans()
    -> Result<Vec<(&'static str, kernel_core::contract::LaunchPlan)>, String> {
        kernels_sm120::backend::bf16_phased_admission_plans().map_err(|error| error.to_string())
    }

    /// The payload of the copy-engine reduction that replaces a prefill
    /// collective, or `None` for a collective it does not replace. The
    /// F32-partial contract sends F32 partials; the BF16-partial contract's
    /// two-shot and turns entries send each rank's partial rounded once to
    /// BF16, as they do. With `composed`, the composed form's F32
    /// reduce-scatter (unadmitted phased entries) is replaced with F32
    /// partials: under the BF16-partial contract its preceding launches
    /// have already rounded the partials in place, so the sum is the same.
    pub(crate) fn prefill_dma_payload(
        symbol: &str,
        composed: bool,
    ) -> Option<crate::ops::DmaPayload> {
        use crate::ops::DmaPayload;
        use kernels_sm120::comm::{phased_bf16, phased_f32, phased_turns_bf16};
        if symbol == phased_f32::ROUND_SYMBOL || composed && symbol == COMPOSED_SCATTER {
            Some(DmaPayload::F32)
        } else if symbol == phased_bf16::SYMBOL || symbol == phased_turns_bf16::SYMBOL {
            Some(DmaPayload::Bf16)
        } else {
            None
        }
    }

    pub(crate) fn prefill_dma_sum_info(
        payload: crate::ops::DmaPayload,
    ) -> kernel_core::backend::OpInfo {
        use crate::ops::DmaPayload;
        use kernels_sm120::comm::{owned_sum_bf16, owned_sum_f32};
        kernel_core::backend::OpInfo::new(
            "local-rank-ordered-bf16-owner-sum",
            kernel_core::properties::Invariance::any_rows(8192),
            match payload {
                DmaPayload::F32 => vec![(owned_sum_f32::SYMBOL.into(), 0)],
                DmaPayload::Bf16 => vec![
                    (owned_sum_bf16::ROUND_SYMBOL.into(), 0),
                    (owned_sum_bf16::SUM_SYMBOL.into(), 0),
                ],
            },
        )
    }

    /// The row layout of a launch that may consume part of a reduced
    /// interval: the HC expand that reads the reduced rows, the r8 c4
    /// prefill mix and the cached norm-ceil of the next sublayer. These are
    /// the entries the 2026-10-21 consumer probe ran per owner interval; the
    /// MoE-input norm-ceil may also certify each row's router operands.
    pub(crate) fn prefill_dma_consumer(
        symbol: &str,
        hidden: u64,
        streams: u64,
    ) -> Option<crate::ops::RowLocal> {
        use crate::ops::{RowLocal, RowLocalKind};
        let flat = streams * hidden * 2;
        Some(match symbol {
            // x, y, post, comb, out, live rows; grid (columns, rows).
            "chop_sinkhorn_hc_deepseek41_expand_f32_v1" => RowLocal {
                kind: RowLocalKind::Expand,
                rows: vec![
                    (0, flat),
                    (1, hidden * 2),
                    (2, streams * 4),
                    (3, streams * streams * 4),
                    (4, flat),
                ],
                call_rows: Some(1),
                live: 5,
                row_offset: None,
                optional: vec![],
                capacity_views: false,
                grid: (1, 1),
            },
            // x, fn, base, scale, pre_in, collapsed, pre, post, comb, live
            // rows; grid (row tiles of eight).
            "chop_sinkhorn_hc_deepseek41_prefill_mix_pipeline_r8_c4_v1" => RowLocal {
                kind: RowLocalKind::Mix,
                rows: vec![
                    (0, flat),
                    (4, streams * 4),
                    (5, hidden * 2),
                    (6, streams * 4),
                    (7, streams * 4),
                    (8, streams * streams * 4),
                ],
                call_rows: None,
                live: 9,
                row_offset: None,
                optional: vec![],
                capacity_views: false,
                grid: (0, 8),
            },
            // input, weight, normed, widened (F32 copy of normed, or null),
            // codes, scales, live rows, status, its first row, eps, and in
            // the certifying form one flag per row (`certified[row]`, the
            // router's exact-operand flag); one CTA per row.
            "chop_prefill_norm_ceil_deepseek41_cached_v1"
            | "chop_prefill_norm_ceil_deepseek41_cached_certified_v1" => RowLocal {
                kind: RowLocalKind::Norm,
                rows: if symbol.ends_with("_certified_v1") {
                    vec![
                        (0, hidden * 2),
                        (2, hidden * 2),
                        (4, hidden),
                        (5, hidden / 32),
                        (10, 4),
                    ]
                } else {
                    vec![
                        (0, hidden * 2),
                        (2, hidden * 2),
                        (4, hidden),
                        (5, hidden / 32),
                    ]
                },
                call_rows: None,
                live: 6,
                row_offset: Some(8),
                optional: vec![(3, hidden * 4)],
                capacity_views: false,
                grid: (0, 1),
            },
            _ => return None,
        })
    }

    /// The row layout of `g`'s MoE route-row sum with its shared expert
    /// addend (routed_sum_add.cuh, one CTA per token), the launch that
    /// writes a MoE reduction's F32 partial rows: routed rows, route ids,
    /// live tokens, rank, addend, output; grid one block per token.
    pub(crate) fn prefill_dma_producer(
        symbol: &str,
        g: &model_deepseek41_contract::geometry::Geometry,
    ) -> Option<crate::ops::RowLocal> {
        use crate::ops::{RowLocal, RowLocalKind};
        let sum = routed_sum::sum(g);
        if symbol != sum.add_symbol(&routed_sum::prefix(g)) {
            return None;
        }
        let (top_k, width) = (u64::from(sum.top_k), u64::from(sum.width));
        Some(RowLocal {
            kind: RowLocalKind::Produce,
            rows: vec![
                (0, top_k * width * 4),
                (1, top_k * 4),
                (4, width * 4),
                (5, width * 4),
            ],
            call_rows: Some(5),
            live: 2,
            row_offset: None,
            optional: vec![],
            // The upper cone's sums read compact routed rows, route ids and
            // addends from the start of their capacity buffers and write the
            // cone's partial rows.
            capacity_views: true,
            grid: (0, 1),
        })
    }

    /// Round `input`'s F32 partial rows once to BF16 into `output`.
    pub(crate) fn prefill_dma_round(
        input: runtime_bundle_abi::DeviceBufferSpanV1,
        output: runtime_bundle_abi::DeviceBufferSpanV1,
        rows: u32,
    ) -> Result<bundle_runtime::pipeline::PreparedLaunch, String> {
        use bundle_runtime::dispatch::BufferBindings;
        use kernel_core::contract::{Argument, ArgumentList};
        use kernels_sm120::comm::owned_sum_bf16;
        let elements = output.length / 2;
        if input.length != elements * 4
            || output.length != elements * 2
            || [input, output].iter().any(|s| s.offset % 16 != 0)
        {
            return Err("DMA rounding has inconsistent byte views".into());
        }
        let plan = owned_sum_bf16::round_plan(elements).map_err(|e| e.to_string())?;
        let mut bindings = BufferBindings::default();
        let arguments = ArgumentList(vec![
            Argument::DeviceBuffer(bindings.bind(input)),
            Argument::DeviceBuffer(bindings.bind(output)),
            Argument::U64(elements),
        ]);
        Ok(bundle_runtime::pipeline::PreparedLaunch {
            symbol: owned_sum_bf16::ROUND_SYMBOL.into(),
            rows,
            plan,
            arguments,
            bindings,
        })
    }

    /// The owner's rank-ordered sums. `inbox` holds four rank slots of
    /// `slot` elements in the payload's format.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_dma_sum(
        payload: crate::ops::DmaPayload,
        input: runtime_bundle_abi::DeviceBufferSpanV1,
        inbox: runtime_bundle_abi::DeviceBufferSpanV1,
        output: runtime_bundle_abi::DeviceBufferSpanV1,
        slot: u64,
        rank: u32,
        rows: u32,
    ) -> Result<bundle_runtime::pipeline::PreparedLaunch, String> {
        use crate::ops::DmaPayload;
        use bundle_runtime::dispatch::BufferBindings;
        use kernel_core::contract::{Argument, ArgumentList};
        use kernels_sm120::comm::{owned_sum_bf16, owned_sum_f32};
        let elements = output.length / 2;
        if input.length != elements * 4
            || output.length != elements * 2
            || inbox.length
                != slot
                    .checked_mul(4 * payload.bytes())
                    .ok_or("DMA inbox overflow")?
            || [input, inbox, output].iter().any(|s| s.offset % 16 != 0)
        {
            return Err("DMA owner sum has inconsistent byte views".into());
        }
        let (symbol, plan) = match payload {
            DmaPayload::F32 => (
                owned_sum_f32::SYMBOL,
                owned_sum_f32::plan(elements, slot, rank, 4),
            ),
            DmaPayload::Bf16 => (
                owned_sum_bf16::SUM_SYMBOL,
                owned_sum_bf16::sum_plan(elements, slot, rank, 4),
            ),
        };
        let plan = plan.map_err(|e| e.to_string())?;
        let mut bindings = BufferBindings::default();
        let arguments = ArgumentList(vec![
            Argument::DeviceBuffer(bindings.bind(input)),
            Argument::DeviceBuffer(bindings.bind(inbox)),
            Argument::DeviceBuffer(bindings.bind(output)),
            Argument::U64(elements),
            Argument::U64(slot),
            Argument::U32(rank),
            Argument::U32(4),
        ]);
        Ok(bundle_runtime::pipeline::PreparedLaunch {
            symbol: symbol.into(),
            rows,
            plan,
            arguments,
            bindings,
        })
    }
}

/// The first collective of the all-reduces' composed form.
const COMPOSED_SCATTER: &str = "chop_comm_reduce_scatter_f32_v1";

// The build script includes the registration files, and so does the
// projection assembly test; a test that reads the bundle's own modules
// lives here.
#[cfg(test)]
mod tests {
    use bundle_runtime::geometry::Ranks;
    use model_deepseek41_contract::geometry::GEOMETRIES;

    /// The dense registration quantizes exactly the inputs the dense ops
    /// ask for.
    #[test]
    fn registered_dense_inputs_are_the_dense_ops_inputs() {
        for g in GEOMETRIES {
            for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
                let Ok(shape) = crate::shape::Shape::new(g, ranks) else {
                    continue;
                };
                assert_eq!(
                    super::dense::inputs(g, ranks.get()),
                    crate::ops::dense::quantized_inputs(&shape)
                );
            }
        }
    }

    /// Each rank count's head entry projects its shard of the vocabulary
    /// from the hidden width, the projection the head ops ask for.
    #[test]
    fn the_head_entry_is_the_ranks_vocabulary_shard() {
        use model_deepseek41_contract::geometry::FLASH;
        for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
            let entry = super::f32_head::entry(&FLASH, ranks.get()).unwrap();
            let q = entry.query();
            assert_eq!(
                (q.n * ranks.get(), q.k),
                (FLASH.shape.vocab, FLASH.shape.hidden)
            );
            let shape = crate::shape::Shape::new(&FLASH, ranks).unwrap();
            assert_eq!(q, crate::ops::head::query(&shape));
            assert_eq!(entry.usage(), crate::ops::head::USE);
        }
    }

    /// Every dense F32 projection the ops ask for has an entry of its query
    /// at the use they ask it at, and every entry serves one.
    #[test]
    fn each_f32_projection_has_its_entry() {
        use crate::ops::exact_f32;
        for g in GEOMETRIES {
            let entries = super::f32_dense::entries(g).unwrap();
            let projections = exact_f32::projections(g);
            for dense in &projections {
                let q = exact_f32::query(dense.n, dense.k);
                let entry = entries.iter().find(|e| e.query() == q);
                let entry = entry.unwrap_or_else(|| panic!("{} has no entry", dense.id));
                assert_eq!(entry.usage(), exact_f32::USE);
            }
            for entry in &entries {
                let q = entry.query();
                assert!(projections.iter().any(|d| (d.n, d.k) == (q.n, q.k)));
            }
        }
    }

    /// Every block quantizer the instance registers serves a query no other
    /// entry serves, at every geometry and rank count: SM120 refuses a
    /// query two entries serve.
    #[test]
    fn each_quantizer_query_is_registered_once() {
        use model_deepseek41_contract::admission::{SUPPORTED_RANKS, check_ranks};
        for g in GEOMETRIES {
            for &ranks in SUPPORTED_RANKS {
                if check_ranks(g, ranks).is_err() {
                    continue;
                }
                let entries: Vec<_> = super::moe::quantizers(g, ranks)
                    .unwrap()
                    .into_iter()
                    .chain(super::dense::quantizers(g, ranks).unwrap())
                    .chain(super::latent_glue::quantizers())
                    .chain([super::indexer_query::quantizer(g)])
                    .collect();
                for (at, entry) in entries.iter().enumerate() {
                    let twin = entries[at + 1..].iter().find(|e| e.query == entry.query);
                    assert!(
                        twin.is_none(),
                        "{} tp{ranks}: {} and {} serve one query",
                        g.id,
                        entry.symbol,
                        twin.unwrap().symbol
                    );
                }
            }
        }
    }

    /// Every RMSNorm requirement the instance registers has one entry: the
    /// drafter's F32 head norm is the main head's final norm, registered
    /// once by the rows (`rows::norms`), and the DSpark rows register none.
    #[test]
    fn each_rmsnorm_requirement_is_registered_once() {
        use kernels_sm120::backend::Sm120;
        use model_deepseek41_contract::admission::{SUPPORTED_RANKS, check_ranks};
        for g in GEOMETRIES {
            let mut norms = super::rows::norms(g);
            if SUPPORTED_RANKS
                .iter()
                .any(|&ranks| super::latent_glue::register(Sm120::new(), g, ranks).is_ok())
            {
                norms.push(super::latent_glue::query_norm(g));
            }
            for (at, (requirement, symbol)) in norms.iter().enumerate() {
                let twin = norms[at + 1..].iter().find(|(r, _)| r == requirement);
                assert!(
                    twin.is_none(),
                    "{}: {symbol} and {} serve one requirement",
                    g.id,
                    twin.unwrap().1
                );
            }
            for &ranks in SUPPORTED_RANKS {
                if check_ranks(g, ranks).is_err() {
                    continue;
                }
                let drafter = super::dspark_aux::register(Sm120::new(), g, ranks).unwrap();
                let refused = kernel_core::backend::Backend::rmsnorm(
                    &drafter,
                    &model_deepseek41_contract::rows::final_norm(g),
                    kernel_core::backend::Need::rows(1),
                );
                assert!(
                    refused.is_err(),
                    "{} tp{ranks}: DSpark registers a norm",
                    g.id
                );
            }
        }
    }

    #[test]
    fn each_hc_query_is_registered_once_including_dspark() {
        use kernel_core::sinkhorn_hyper_connection::{
            SinkhornHyperConnectionQuery as Query, StreamHeadMode,
        };
        for g in GEOMETRIES {
            let entries: Vec<_> = super::hc::registered_entries(g)
                .into_iter()
                .map(|entry| (entry.query, entry.symbol))
                .chain(
                    super::dspark_aux::feature_entries(g)
                        .into_iter()
                        .map(|entry| (entry.query, entry.symbol)),
                )
                .collect();
            for (at, (query, symbol)) in entries.iter().enumerate() {
                assert!(
                    !entries[at + 1..].iter().any(|(q, _)| q == query),
                    "{} repeats {symbol}'s query",
                    g.id
                );
            }
            assert_eq!(
                entries
                    .iter()
                    .filter(|(q, _)| matches!(
                        q,
                        Query::Head {
                            mode: StreamHeadMode::SigmoidWeightedF32,
                            ..
                        }
                    ))
                    .count(),
                1
            );
        }
    }
}

#[cfg(test)]
mod window_bf16_tests {
    use super::latent_window_bf16::*;
    use kernel_core::backend::{Backend, Need};
    use kernels_sm120::backend::Sm120;
    use model_deepseek41_contract::{
        geometry::GEOMETRIES, latent_glue, serving::PREFILL_TOKENS, state::KvStorage,
    };
    #[test]
    fn bf16_windows_prepare_through_the_shared_backend() {
        for g in GEOMETRIES {
            let b = super::latent_glue::register(Sm120::new(), g, 4).unwrap();
            let b = register(b);
            let write =
                latent_glue::window_write_for_storage(g, latent_glue::ring(g), KvStorage::Bf16);
            let op = b.latent_window_write(&write, Need::rows(32)).unwrap();
            assert!(op.info().symbols.iter().any(|(s, _)| s == &store().symbol));
            assert_eq!(
                op.info()
                    .bytes
                    .iter()
                    .find(|(s, _)| s == "window.scales")
                    .unwrap()
                    .1,
                0
            );
            let stage =
                latent_glue::window_stage_for_storage(g, PREFILL_TOKENS + 127, KvStorage::Bf16)
                    .unwrap();
            assert!(b.latent_window_stage(&stage, Need::rows(32)).is_ok());
        }
    }
}

#[cfg(test)]
mod one_row_twin_lines {
    use bundle_runtime::geometry::Ranks;
    use model_deepseek41_contract::geometry::FLASH;

    use crate::shape::Shape;

    fn lines(ranks: Ranks, served: bool) -> Vec<String> {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        super::query_window::with_served(served, || {
            super::ceil_split::with_served(served, || crate::Kernels::one_row_twin_lines(&shape))
        })
    }

    /// A served run names each twin it serves with these lines, which the
    /// served pair's job looks for.
    #[test]
    fn served_twins_name_themselves_at_startup() {
        assert_eq!(
            lines(Ranks::TP4, true),
            [
                "CHOPINFER_DEEPSEEK41_QUERY_WINDOW_TWIN: one-row decode runs each layer's query RoPE \
                 and window write as one launch",
                "CHOPINFER_DEEPSEEK41_CEIL_SPLIT: one-row dense quantizers split: w8192 Bf16 over 8 \
                 CTAs, w2048 F32 over 8 CTAs",
            ]
        );
        assert!(lines(Ranks::TP4, false).is_empty());
        assert!(lines(Ranks::TP2, true).is_empty());
    }
}
