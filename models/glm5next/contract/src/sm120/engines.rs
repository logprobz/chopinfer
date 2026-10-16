//! The bundle's entries of the merged engines, shared by its `build.rs`
//! (which compiles and locks them) and its lowering (which plans and binds
//! them): latent attention (`kernels_sm120::latent`) and the
//! per-channel-decay delta rule (`kernels_sm120::linear::channel_decay`).
//! Each engine's cubin gates on its own family of the bundle's lock
//! directory (`bundle/kernels.lock.d`).
//!
//! The pooled indexer (`kernels_sm120::pooled_indexer`) and Sinkhorn mHC
//! (`kernels_sm120::sinkhorn_hc`) entries are instantiated under the same
//! profile. The chained MoE engine (`kernels_sm120::moe::chained`) names
//! its served TP4 instances by shape; the bundle builds its projection
//! cubins and the routed experts' split cubins (`chained::split`) exactly as
//! the engine declares them, so they are the cubins kernel-lab qualifies.

use std::path::PathBuf;

use kernel_build::CubinSpec;
use kernels_sm120::comm::Variant;
use kernels_sm120::comm::ll_bf16;
use kernels_sm120::comm::phased_bf16;
use kernels_sm120::comm::phased_turns_bf16;
use kernels_sm120::comm::sinkhorn_split::Entry as SinkhornCommEntry;
use kernels_sm120::comm::sinkhorn_split_tree;
use kernels_sm120::latent;
use kernels_sm120::linear::channel_decay;
use kernels_sm120::moe::chained;
use kernels_sm120::row::RowInstantiation;
use kernels_sm120::row::argmax;
use kernels_sm120::row::rmsnorm::{RmsNorm, RmsNormTarget};
use kernels_sm120::{Profile, ProfileNames, pooled_indexer, sinkhorn_hc};

use crate::contract::{PAGE_TOKENS, PREFILL_TOKENS, Shape};

/// The bundle's kernel profile: every entry symbol carries its tag.
pub const PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "glm5next",
    tag: "glm5next",
});

/// The lock families of the bundle's lock directory, by engine.
pub const LATENT_FAMILY: &str = "latent_attention";
/// The slice-widened forms of the FP8 prefill latent attention entries
/// (`kernels_sm120::latent::slices`).
pub const LATENT_SLICES_FAMILY: &str = "latent_attention_slices";
/// The column merges of the decode and verify latent attention entries
/// (`kernels_sm120::latent::merge_columns`).
pub const LATENT_MERGE_COLUMNS_FAMILY: &str = "latent_merge_columns";
pub const CHANNEL_DECAY_FAMILY: &str = "channel_decay";
/// The KDA prefill's prepare and scan under the streamed schedule
/// (`kernels_sm120::linear::channel_decay::streamed_entries`).
pub const CHANNEL_DECAY_STREAMED_FAMILY: &str = "channel_decay_streamed";
pub const POOLED_INDEXER_FAMILY: &str = "pooled_indexer";
/// The tree merges of the pooled indexer's selections
/// (`kernels_sm120::pooled_indexer::merge_tree`).
pub const POOLED_INDEXER_MERGE_TREE_FAMILY: &str = "pooled_indexer_merge_tree";
/// The radix merge of the pooled indexer's global selection
/// (`kernels_sm120::pooled_indexer::merge_radix`).
pub const POOLED_INDEXER_MERGE_RADIX_FAMILY: &str = "pooled_indexer_merge_radix";

/// The one-split forms of the pooled indexer selections
/// (`kernels_sm120::pooled_indexer::rows`).
pub const POOLED_INDEXER_ROWS_FAMILY: &str = "pooled_indexer_rows";
pub const SINKHORN_FAMILY: &str = "sinkhorn_hc";
/// The split Sinkhorn mix-collapse (`kernels_sm120::sinkhorn_hc::split`).
pub const SINKHORN_SPLIT_FAMILY: &str = "sinkhorn_hc_split";
/// The Sinkhorn mix-collapse with rows tiled over a CTA
/// (`kernels_sm120::sinkhorn_hc::tiled`).
pub const SINKHORN_TILED_FAMILY: &str = "sinkhorn_hc_tiled";
/// The split Sinkhorn mix-collapse fused with the input RMSNorm whose sum of
/// squares is the 256-lane tree (`kernels_sm120::sinkhorn_hc::mix_norm_tree`).
pub const SINKHORN_MIX_NORM_TREE_FAMILY: &str = "sinkhorn_hc_mix_norm_tree";
/// The dense MLP's and the shared expert's chained projections.
pub const MOE_FAMILY: &str = "moe_chained";
/// The dense MLP's and the shared expert's chained projections in the
/// NVFP4 exports' formats.
pub const MOE_FORMATS_FAMILY: &str = "moe_chained_formats";
/// The routed experts split over the ranks (`chained::split`): their
/// projections in every routed format, the routes over every expert, the
/// route sums and the F32 sum that adds the shared expert's output, which
/// the decode and verify graphs run after the shared expert's down
/// projection ran beside the routed experts. The cubins kernel-lab
/// qualifies in its family of the same name.
pub const MOE_SPLIT_FAMILY: &str = "moe_chained_split";
/// The split routed sum that adds a row.
pub const MOE_SUM_ADD_CUBIN: &str = "chop_moe_chain_e288l288k8_sum_add";
pub const ROW_FAMILY: &str = "row";
/// The shared row copies (`kernels_sm120::row::copy`): the stream
/// expansion's broadcast and the router scores' column slice.
pub const ROW_COPY_FAMILY: &str = "row_copy";
/// Their cubin, which every instance loads after the other text cubins.
pub const ROW_COPY_CUBIN: &str = "row_copy_sm120";
/// The shared stream broadcast, which the stream expansion launches.
pub const BROADCAST_SYMBOL: &str = kernels_sm120::row::copy::BROADCAST_SYMBOL;
/// The shared column slice, which the router's score compaction launches.
pub const SLICE_SYMBOL: &str = kernels_sm120::row::copy::SLICE_SYMBOL;
pub const COMM_FAMILY: &str = "comm";
/// The Sinkhorn mHC comm epilogue split over CTAs, with the input norm's
/// tree sum (`kernels_sm120::comm::sinkhorn_split_tree`).
pub const COMM_EPILOGUE_FAMILY: &str = "sinkhorn_epilogue_split_tree";
/// Its cubin, which an instance loads only when it runs the epilogue, so
/// instances without it load and resolve exactly what they did before.
pub const COMM_EPILOGUE_CUBIN: &str = "sinkhorn_epilogue_split_tree_glm5next_sm120";
/// The low-latency all-reduce of BF16 partials over CTAs
/// (`kernels_sm120::comm::ll_bf16`).
pub const LL_FAMILY: &str = "allreduce_ll_bf16";
pub const LL_CUBIN: &str = "comm_ll_bf16_glm5next_sm120";
/// The phased all-reduce of BF16 partials in one launch
/// (`kernels_sm120::comm::phased_bf16`): both entries, the cubin kernel-lab
/// qualifies under another name.
pub const PHASED_FAMILY: &str = "comm_phased_bf16";
pub const PHASED_CUBIN: &str = "comm_phased_bf16_glm5next_sm120";
/// The phased all-reduce whose writers take turns at each destination
/// (`kernels_sm120::comm::phased_turns_bf16`), the cubin kernel-lab
/// qualifies under another name.
pub const PHASED_TURNS_FAMILY: &str = "comm_phased_turns_bf16";
pub const PHASED_TURNS_CUBIN: &str = "comm_phased_turns_bf16_glm5next_sm120";
/// The low-latency all-reduce of F32 partials (`kernels_sm120::comm::ll_f32`):
/// its F32 entry and the round entry the MoE sublayers' rounded all-reduce
/// runs, the cubin kernel-lab qualifies under another name.
pub const LL_F32_FAMILY: &str = "allreduce_ll_f32";
pub const LL_F32_CUBIN: &str = "comm_ll_f32_glm5next_sm120";
/// The phased all-reduce of F32 partials (`kernels_sm120::comm::phased_f32`):
/// its F32 entries and the rounding entries the MoE sublayers' rounded
/// all-reduce runs, the cubin kernel-lab qualifies under another name.
pub const PHASED_F32_FAMILY: &str = "comm_phased_f32";
pub const PHASED_F32_CUBIN: &str = "comm_phased_f32_glm5next_sm120";
/// The shared sharded argmax (`kernels_sm120::row::argmax`): the F32
/// partial and the final the drafter's argmax runs
/// (`crate::draft_argmax`), under their shared symbols.
pub const ARGMAX_FAMILY: &str = "row_argmax";
pub const ARGMAX_CUBIN: &str = "row_argmax_glm5next_sm120";

/// The latent attention geometry of `shape`: its local heads over the
/// shared latent, the key list of the indexer's compiled top-k plus the
/// incomplete pool tail, and the unrotated head's softmax scale.
pub fn latent_geometry(shape: Shape) -> latent::Geometry {
    let p = shape.profile;
    latent::Geometry {
        heads: p.latent_heads,
        latent: p.kv_rank,
        page_size: PAGE_TOKENS as u32,
        max_keys: shape.latent_keys(),
        softmax_scale: 1.0 / (shape.geometry.qk_head() as f32).sqrt(),
        nope: p.qk_nope,
        v_dim: p.v_head,
    }
}

pub fn latent_entries(shape: Shape) -> Vec<latent::Entry> {
    latent::entries(PROFILE, latent_geometry(shape))
}

/// The slice-widened form of each FP8 prefill attention entry: its outputs
/// bit for bit from the same launch ABI and grid, with two CTAs per SM.
/// Every launch of the entry runs it in its place: kernel-lab timed it
/// faster at 1 to 8192 rows (PERF-B17 job 1), and both forms write the same
/// bits, so the choice does not touch a row's output.
pub fn latent_slices_entries(shape: Shape) -> Vec<latent::slices::Entry> {
    latent::slices::entries(&latent_entries(shape))
}

/// The column merge of each decode and verify attention entry: its split
/// merge's outputs bit for bit from the same workspace and ABI, each thread
/// folding one column after loading every split's value of it. Every launch
/// of the entry's merge runs it in its place.
pub fn latent_merge_columns_entries(shape: Shape) -> Vec<latent::merge_columns::Entry> {
    latent::merge_columns::entries(&latent_entries(shape))
}

/// The KDA geometry of `shape`. The capacities are the engine's served
/// shape, which kernel-lab qualifies (prefill 8192 rows, decode 256
/// sequences, verify 8 rows of 256 sequences); they bound the package's
/// 32 items of up to 6 verify rows.
pub fn channel_decay_geometry(shape: Shape) -> channel_decay::Geometry {
    let p = shape.profile;
    let kernel_core::per_channel_decay_delta_rule::PerChannelDecayDeltaRuleQuery::DeltaRule {
        gate,
        ..
    } = shape.kda_delta_rule(kernel_core::contract::RecurrentPhase::Decode, 1)
    else {
        unreachable!("the delta rule query")
    };
    channel_decay::Geometry {
        heads: p.linear_heads,
        head_dim: p.linear_head_dim,
        kernel: p.conv_taps,
        convs: kernel_core::per_channel_decay_delta_rule::QkvConvs::Separate,
        gate,
        norm_eps_bits: shape.geometry.eps().to_bits(),
        ..channel_decay::SERVED
    }
}

/// The KDA prefill bucket of `rows` rows: an eager prefill may carry every
/// served item.
pub const fn kda_prefill_bucket(rows: u32) -> kernels_sm120::linear::Bucket {
    kernels_sm120::linear::Bucket {
        rows,
        sequences: crate::contract::DECODE_BATCH,
    }
}

pub fn channel_decay_entries(shape: Shape) -> Vec<channel_decay::Entry> {
    channel_decay::entries(PROFILE, channel_decay_geometry(shape))
}

/// The delta-rule prefill's prepare and scan under the streamed schedule,
/// which prefill runs: the staged pair's queries, ABI, grids and workspace,
/// and its workspace, outputs and states bit for bit.
pub fn channel_decay_streamed_entries(shape: Shape) -> Vec<channel_decay::Entry> {
    channel_decay::streamed_entries(PROFILE, channel_decay_geometry(shape))
}

/// The pooled indexer geometry of `shape`: the replicated 32 heads of 128,
/// pools of 4 with the always-selected tail, the compiled top-k, 64-token
/// pages, the BF16 pooling position embedding and verify items of the
/// package's rows.
pub fn pooled_indexer_geometry(shape: Shape) -> pooled_indexer::Geometry {
    let p = shape.profile;
    pooled_indexer::Geometry {
        heads: p.indexer_heads,
        head_dim: p.indexer_dim,
        group: p.index_pool,
        page_size: PAGE_TOKENS as u32,
        max_top_k: p.index_topk_max,
        always_tail: shape.geometry.semantics.index_pool_tail,
        position_dtype: kernel_core::contract::DType::Bf16,
        key_norm_eps: INDEXER_KEY_NORM_EPS,
        verify_rows: crate::contract::MAX_VERIFY_ROWS,
    }
}

/// The indexer key LayerNorm's eps.
pub const INDEXER_KEY_NORM_EPS: f32 = 1e-6;

pub fn pooled_indexer_entries(shape: Shape) -> Vec<pooled_indexer::Entry> {
    pooled_indexer::entries(PROFILE, pooled_indexer_geometry(shape))
}

/// The tree merge of each selection entry: its `select_merge`'s key list or
/// per-rank pairs bit for bit from the same workspace, launch ABI and grid,
/// with the split lists folded in registers. A selection over two or more
/// splits runs it in place of its merge.
pub fn pooled_indexer_merge_tree_entries(shape: Shape) -> Vec<pooled_indexer::merge_tree::Entry> {
    pooled_indexer::merge_tree::entries(&pooled_indexer_entries(shape))
}

/// The radix merge of the global selection: its `select_merge`'s key list
/// bit for bit from the same workspace, launch ABI and grid, by a radix
/// select of the kept keys (rows past `merge_radix::CACHE_KEYS` candidates
/// fold through the tree networks). A global selection over two or more
/// splits runs it in place of the tree merge; the per-rank selections keep
/// the tree merge.
pub fn pooled_indexer_merge_radix_entries(shape: Shape) -> Vec<pooled_indexer::merge_radix::Entry> {
    pooled_indexer::merge_radix::entries(&pooled_indexer_entries(shape))
}

/// Rows per CTA of the served one-split selections: one row in a 256-thread
/// CTA, two CTAs per SM. PERF-B17 job 3 timed it at 0.62 to 0.65 of the
/// selection's split kernel and merge at 2048 and 8192 rows ending at 8K
/// and 32K, and the two-row form at 0.73 to 0.76.
pub const POOLED_INDEXER_ROWS: u32 = 1;

/// The one-split form of each selection entry (global and per rank): the
/// key lists or pairs of the selection's split kernel and merge at one split
/// per row, bit for bit, from one launch. A selection launch whose bucket
/// takes one split (376 rows or more) runs it in place of the pair; both
/// write the same bits, so the choice does not touch a row's output.
pub fn pooled_indexer_rows_entries(shape: Shape) -> Vec<pooled_indexer::rows::Entry> {
    pooled_indexer::rows::entries(&pooled_indexer_entries(shape), POOLED_INDEXER_ROWS)
}

/// The Sinkhorn mixer of `shape`: its streams over the hidden width, the
/// geometry's iterations and eps, the RMS norm eps, BF16 `fn`, a mixer
/// around every sublayer; one entry per op up to the engine's 8192 rows.
pub fn sinkhorn_geometry(shape: Shape) -> sinkhorn_hc::Geometry {
    use kernel_core::sinkhorn_hyper_connection::{
        HyperConnectionPlacement, SinkhornMix, SinkhornOrder,
    };
    let g = shape.geometry;
    sinkhorn_hc::Geometry {
        mix: SinkhornMix {
            streams: shape.profile.hc_streams,
            width: shape.profile.hidden,
            iterations: g.shape.hc.sinkhorn_iterations,
            eps_bits: (g.shape.hc.eps as f32).to_bits(),
            norm_eps_bits: g.eps().to_bits(),
            order: SinkhornOrder::RowSoftmaxThenColumnFirst,
            placement: HyperConnectionPlacement::PerSublayer,
            fn_dtype: kernel_core::contract::DType::Bf16,
        },
        capacity: sinkhorn_hc::MAX_ROWS,
    }
}

pub fn sinkhorn_entries(shape: Shape) -> Vec<sinkhorn_hc::Entry> {
    sinkhorn_hc::entries(PROFILE, sinkhorn_geometry(shape), false)
}

/// The mix-collapse split over CTAs, bitwise the one-CTA entry's, which
/// the mixer stages run except in long prefill chunks (the bundle's
/// lowering picks by row count): `sinkhorn_hc::split::groups` CTAs per row, with
/// the chain sums and tickets of `sinkhorn_hc::split::partials_bytes` and
/// `tickets_bytes`.
pub fn sinkhorn_split_entry(shape: Shape) -> sinkhorn_hc::split::Entry {
    sinkhorn_hc::split::entry(PROFILE, sinkhorn_geometry(shape))
}

/// The tiles of rows per CTA the served prefill schedule runs
/// (`sinkhorn_hc::tiled::served_prefill_mix`).
pub const SINKHORN_TILES: [u32; 2] = sinkhorn_hc::tiled::SERVED_TILES;

/// The mix-collapse with rows tiled over a CTA at each of
/// [`SINKHORN_TILES`], bitwise the one-CTA entry's, with its ABI and no
/// workspace, which prefill mixer stages run where the served schedule
/// picks a tile (the bundle's lowering).
pub fn sinkhorn_tiled_entries(shape: Shape) -> Vec<sinkhorn_hc::tiled::Entry> {
    SINKHORN_TILES
        .iter()
        .map(|&rows| sinkhorn_hc::tiled::entry(PROFILE, sinkhorn_geometry(shape), rows))
        .collect()
}

/// The split mix-collapse fused with the sublayer's input RMSNorm of the
/// collapsed row ([`rmsnorm_requirement`] at the hidden width, whose sum of
/// squares is the 256-lane tree): the split entry's outputs and the row
/// engine's norm of the collapsed row bit for bit in one launch, with the
/// split entry's grid, chain sums and tickets. The decode and verify graphs
/// run it in place of each main layer's mixer and input norm.
pub fn sinkhorn_mix_norm_tree_entry(shape: Shape) -> sinkhorn_hc::mix_norm_tree::Entry {
    sinkhorn_hc::mix_norm_tree::entry(
        PROFILE,
        sinkhorn_geometry(shape),
        shape.rmsnorm_requirement(shape.profile.hidden),
    )
}

/// The Sinkhorn mHC comm epilogue of `shape`'s mixer with the input
/// RMSNorm's eps, split over CTAs, up to the mixer's capacity: it runs over
/// the F32 sums the unchanged all-reduce (the one-shot or the phased pair)
/// wrote, and its outputs match the unfused round, expand, mix-collapse and
/// input norm (the row engine's, with the tree sum) bit for bit.
pub fn comm_epilogue_entry(shape: Shape) -> Result<SinkhornCommEntry, String> {
    let geometry = sinkhorn_geometry(shape);
    sinkhorn_split_tree::entry(
        PROFILE,
        geometry.mix,
        shape.geometry.eps().to_bits(),
        geometry.capacity,
    )
    .map_err(|refusal| format!("Sinkhorn comm epilogue: {refusal}"))
}

/// The chained MoE engine's served shapes of the routed experts split over
/// the ranks (`chained::split::SHAPES`: every expert local, a quarter of
/// its intermediate per rank), checked against `shape`; the dense MLP's and
/// the shared expert's are theirs.
pub fn moe_shapes(shape: Shape) -> Result<chained::ServedShapes, String> {
    let p = shape.profile;
    let served = chained::split::SHAPES;
    let clamp =
        kernel_core::gated_mlp::SwigluClamp::symmetric(shape.geometry.shape.swiglu_limit as f32);
    if (
        served.experts,
        served.local_experts,
        served.top_k,
        served.hidden,
    ) != (
        shape.geometry.shape.moe.experts,
        p.local_experts,
        p.top_k,
        p.hidden,
    ) || served.expert_intermediate != p.expert_intermediate
        || served.dense_intermediates != [p.dense_intermediate, p.shared_intermediate]
        || served.clamp != clamp
    {
        return Err("the chained MoE engine's served shapes differ from the contract".into());
    }
    Ok(served)
}

/// The rank's weighted routes summed per token in F32 and left unrounded:
/// in a prefill the shared expert's down projection adds into it through
/// its F32 residual epilogue; in the decode and verify graphs its
/// [`chained::RoutedSum::add_entry`] adds the shared expert's F32 output
/// (the same add of the same two values). The MLP sublayer rounds once,
/// after the all-reduce.
pub fn routed_sum(shape: Shape) -> Result<chained::RoutedSum, String> {
    let [sum] = moe_shapes(shape)?.sums();
    Ok(sum.with_output(chained::RoutedSumOutput::F32))
}

/// The FP8 projections of the dense MLP and the shared expert, gate/up and
/// down, at the decode and prefill row tiles. Decode runs them as block
/// warps (`chained::Schedule::BlockWarps`), bitwise the column schedule's
/// outputs; every prefill gate/up runs its decoded-tile form
/// ([`moe_decoded_tiles`]).
pub fn moe_projections(shape: Shape) -> Result<Vec<chained::Projection>, String> {
    Ok(fp8_cubin_projections(shape)?
        .into_iter()
        .filter(|projection| projection.query.gemm.grouped.is_none())
        .collect())
}

/// What the bundle's FP8 cubins hold, in their order: the served catalog's
/// FP8 projections (the dense MLP's and the shared expert's) in the column
/// schedule, the decode ones as block warps, then the decoded tiles.
fn fp8_cubin_projections(shape: Shape) -> Result<Vec<chained::Projection>, String> {
    moe_shapes(shape)?;
    Ok(chained::catalog()
        .into_iter()
        .filter(|projection| projection.weights() == chained::Weights::Fp8Block128)
        .map(|projection| projection.block_warps().unwrap_or(projection))
        .chain(moe_decoded_tiles(shape)?)
        .collect())
}

/// The dense MLP's and the shared expert's FP8 prefill gate/up under
/// `chained::Schedule::DecodedTiles`: bitwise the column schedule's
/// outputs, each weight code decoded once per CTA of 128 tokens by 64
/// columns. Their column entries stay in their cubin, unlaunched.
pub fn moe_decoded_tiles(shape: Shape) -> Result<Vec<chained::Projection>, String> {
    moe_shapes(shape)?;
    chained::catalog()
        .into_iter()
        .filter(|projection| {
            projection.weights() == chained::Weights::Fp8Block128
                && projection.query.gemm.grouped.is_none()
                && projection.gate_up()
                && projection.row_tile == 64
        })
        .map(|projection| {
            projection
                .decoded_tiles()
                .map_err(|refusal| format!("{}: {refusal}", projection.symbol))
        })
        .collect()
}

/// The routed experts' projections split over the ranks in every routed
/// format and the schedules they run (`chained::split::forms`): at decode
/// the FP8 gate/up in warp streams and down in block warps up to
/// `chained::split::DOWN_BLOCK_WARP_TOKENS` tokens, else in columns; at
/// prefill the FP8 gate/up and down also in tile pairs, which steps of at
/// least `kernels_sm120::backend::TILE_PAIR_MIN_ROWS` rows run.
pub fn moe_routed_projections(shape: Shape) -> Result<Vec<chained::Projection>, String> {
    moe_shapes(shape)?;
    Ok(chained::split::forms())
}

/// The chained projections the NVFP4 exports' layouts add to
/// [`moe_projections`]: the served catalog's BF16 dense MLP and shared
/// expert projections, the decode ones as block warps (as the FP8 ones,
/// `chained::block_warp_catalog`, which kernel-lab qualifies against the
/// column schedule), then `chained::formats`' NVFP4 dense gate/up.
pub fn moe_format_projections(shape: Shape) -> Result<Vec<chained::Projection>, String> {
    moe_shapes(shape)?;
    Ok(served_format_projections()
        .into_iter()
        .chain(added_format_projections())
        .collect())
}

/// `chained::formats`' dense projections: the NVFP4 dense gate/up.
fn added_format_projections() -> Vec<chained::Projection> {
    chained::formats::catalog()
        .into_iter()
        .filter(|projection| projection.query.gemm.grouped.is_none())
        .collect()
}

/// The served catalog's dense MLP and shared expert projections in the
/// NVFP4 exports' formats, the BF16 decode ones as block warps.
fn served_format_projections() -> Vec<chained::Projection> {
    chained::catalog()
        .into_iter()
        .filter(|projection| {
            projection.weights() != chained::Weights::Fp8Block128
                && projection.query.gemm.grouped.is_none()
        })
        .map(|projection| match projection.weights() {
            chained::Weights::Nvfp4 | chained::Weights::Bf16 => {
                projection.block_warps().unwrap_or(projection)
            }
            _ => projection,
        })
        .collect()
}

/// The RMSNorm of `width` planned at the row capacity bucket `capacity`.
pub fn rmsnorm(shape: Shape, width: u32, capacity: u32) -> Result<RmsNorm, String> {
    RmsNorm::new(
        shape.rmsnorm_requirement(width),
        capacity,
        RmsNormTarget::SM120F,
    )
    .map_err(|refusal| format!("RMSNorm of {width}: {refusal}"))
}

pub fn rmsnorm_symbol(width: u32) -> String {
    format!("chop_row_rmsnorm_{}_w{width}_v1", PROFILE.tag())
}

/// The embedding gather of each rank's vocabulary shard: rows of other
/// shards are zero, which the all-reduce fills.
pub fn embedding_entry(shape: Shape) -> RowInstantiation {
    let p = shape.profile;
    let mut entry = kernels_sm120::row::embedding(
        PROFILE,
        &format!("embedding_{}", PROFILE.tag()),
        p.vocab_rows,
        p.hidden,
        true,
    );
    entry.invariance = kernels_sm120::row::row_local();
    entry
}

/// The comm engine's F32 all-reduce, which the composition an instance
/// falls back to without the low-latency or phased entries runs over
/// partials already rounded to BF16 and widened, the sum rounded once to
/// BF16 afterwards (`ModelRow::Round`). Up to the tuned one-shot bytes the
/// one-shot entry, above them the reduce-scatter then all-gather pair
/// (`kernels_sm120::comm::choose`).
pub const ONE_SHOT_SYMBOL: &str = "chop_comm_allreduce_oneshot_f32_v1";
pub const SCATTER_SYMBOL: &str = "chop_comm_reduce_scatter_f32_v1";
pub const GATHER_SYMBOL: &str = "chop_comm_allgather_scattered_f32_v1";
/// The F32 logits moves: decode and verify rows exchanged to their block
/// owners, prefill rows gathered to rank 0.
pub const EXCHANGE_SYMBOL: &str = "chop_comm_exchange_rows_f32_v1";
pub const GATHER_ROWS_SYMBOL: &str = "chop_comm_gather_rows_f32_v1";
/// Every comm entry the bundle embeds.
pub const COMM_SYMBOLS: [&str; 5] = [
    ONE_SHOT_SYMBOL,
    SCATTER_SYMBOL,
    GATHER_SYMBOL,
    EXCHANGE_SYMBOL,
    GATHER_ROWS_SYMBOL,
];

/// The all-reduce of `rows` hidden rows of F32 partials.
pub fn reduce_query(shape: Shape, rows: u32) -> kernel_core::contract::CommQuery {
    kernel_core::contract::CommQuery::AllReduce {
        ranks: shape.ranks(),
        elements: u64::from(rows) * u64::from(shape.profile.hidden),
        dtype: kernel_core::contract::DType::F32,
        fused_residual: false,
        fused_rmsnorm: None,
    }
}

/// The low-latency all-reduce of the sublayer partials
/// (`kernels_sm120::comm::ll_bf16`): every rank's F32 partial rounded once
/// to BF16, the ranks' BF16 values summed in rank order from F32 zero and
/// rounded once to BF16, in CTAs of 2048 elements, two per row.
/// SM120's table (`kernels_sm120::backend::bf16_tuned`) says which row
/// counts it serves.
pub const LL_SYMBOL: &str = ll_bf16::SYMBOL;

// SM120's form of each sublayer all-reduce row count, the low-latency
// entry's most rows and the phased entries' largest plans, which create
// admits against the devices' residency.
pub use kernels_sm120::backend::{
    BF16_LOW_LATENCY_ROWS, Bf16Tuned, bf16_phased_admission_plans, bf16_tuned,
};

/// One rank's state of the low-latency all-reduce, which every row count
/// up to [`BF16_LOW_LATENCY_ROWS`] shares: its inbox and epochs, both
/// zeroed at create.
pub fn ll_state_bytes(shape: Shape) -> Result<ll_bf16::StateBytes, String> {
    ll_bf16::state_bytes(
        shape.ranks(),
        u64::from(BF16_LOW_LATENCY_ROWS) * u64::from(shape.profile.hidden),
    )
    .map_err(|refusal| refusal.to_string())
}

/// The MoE sublayers' all-reduce (`CommQuery::AllReduceRounded`): every
/// rank's F32 partial summed in F32 in rank order from zero and rounded
/// once to BF16, which the operator chose for the split routed experts
/// (their MoE output 17% closer to an f64 oracle than BF16 partials).
/// SM120's table gives each row count its form: the low-latency round
/// entry up to [`ROUNDED_LOW_LATENCY_ROWS`] rows, the phased F32 rounding
/// entries above, whose largest plans create admits.
pub use kernels_sm120::backend::{ROUNDED_LOW_LATENCY_ROWS, rounded_phased_admission_plans};

/// The low-latency round entry (`kernels_sm120::comm::ll_f32::ROUND_SYMBOL`).
pub const LL_ROUND_SYMBOL: &str = kernels_sm120::comm::ll_f32::ROUND_SYMBOL;

/// The phased F32 rounding entries (`kernels_sm120::comm::phased_f32`), in
/// the order create admits them: the one-shot round entry and the round
/// two-shot.
pub const PHASED_ROUND_SYMBOLS: [&str; 2] = [
    kernels_sm120::comm::phased_f32::ONE_SHOT_ROUND_SYMBOL,
    kernels_sm120::comm::phased_f32::ROUND_SYMBOL,
];

/// One rank's state of the MoE sublayers' low-latency round entry, which
/// every row count up to [`ROUNDED_LOW_LATENCY_ROWS`] shares: its inbox and
/// epochs, both zeroed at create.
pub fn rounded_ll_state_bytes(
    shape: Shape,
) -> Result<kernels_sm120::comm::ll_f32::StateBytes, String> {
    kernels_sm120::comm::ll_f32::state_bytes(
        shape.ranks(),
        u64::from(ROUNDED_LOW_LATENCY_ROWS) * u64::from(shape.profile.hidden),
    )
    .map_err(|refusal| refusal.to_string())
}

/// One rank's state of the MoE sublayers' phased F32 entries, which every
/// row count up to [`PHASED_ROWS`] shares: its own, as a phased state
/// serves one payload format. Flags and epoch zeroed at create.
pub fn rounded_phased_state_bytes(shape: Shape) -> Result<CommBytes, String> {
    let state = kernels_sm120::comm::phased_f32::buffers(
        shape.ranks(),
        u64::from(PHASED_ROWS) * u64::from(shape.profile.hidden),
    )
    .map_err(|refusal| refusal.to_string())?;
    Ok(CommBytes {
        inbox: state.inbox_bytes,
        flags: state.flag_bytes,
        epoch: state.epoch_bytes,
    })
}

/// The phased all-reduce's entries ([`Bf16Tuned::symbol`]), all served:
/// the one-shot and two-shot (`kernels_sm120::comm::phased_bf16`) and the
/// turns entry (`kernels_sm120::comm::phased_turns_bf16`). They are
/// the only entries the phased state takes: a state serves one payload
/// format (all three's BF16), and the fallback's F32 pair runs on the comm
/// engine's own state.
pub const PHASED_SYMBOLS: [&str; 3] = [
    phased_bf16::ONE_SHOT_SYMBOL,
    phased_bf16::SYMBOL,
    phased_turns_bf16::SYMBOL,
];

/// The most rows a phased call takes: its state's capacity, the prefill
/// chunk.
pub const PHASED_ROWS: u32 = PREFILL_TOKENS;

/// Phased calls after which a flag tag repeats. Peers release flags with
/// the call's tagged epoch (`0x80000000 | epoch`,
/// `allreduce_phased_f32.cuh`), receivers compare for equality, and flags
/// stay set after a call; the one-shot picks its inbox and flag parity by
/// `epoch & 1`. A flag a call left behind therefore carries the tag and
/// parity of the call 2^31 calls later, whose early ranks would read the
/// old partials before late peers send theirs (review PERF-B7b-1).
pub const PHASED_TAG_PERIOD: u64 = 1 << 31;

/// The most phased calls one run (one rank's program of a phase) may
/// launch; lowering refuses a program with more. A target decode or
/// verify run launches 91 (one per all-reduce), a drafter pass 3.
pub const PHASED_CALLS_PER_RUN: u64 = 1024;

/// Runs an instance issues before it zeroes every rank's phased flags and
/// epoch while every rank is idle: at most half [`PHASED_TAG_PERIOD`]
/// calls at [`PHASED_CALLS_PER_RUN`] calls a run, so no tag repeats
/// between resets. A call with no live rows does not advance the epoch, so
/// the count of calls is an upper bound.
pub const PHASED_RESET_RUNS: u64 = PHASED_TAG_PERIOD / 2 / PHASED_CALLS_PER_RUN;
const _: () = assert!(PHASED_RESET_RUNS * PHASED_CALLS_PER_RUN <= PHASED_TAG_PERIOD / 2);

/// One rank's state of the phased all-reduce, which every entry and row
/// count up to [`PHASED_ROWS`] shares: the inbox (the one-shot's region,
/// then the two-shot's slots), the flags (the turns entry's, which add its
/// turn words to the phased entries') and the epoch; flags and epoch
/// zeroed at create.
pub fn phased_state_bytes(shape: Shape) -> Result<CommBytes, String> {
    let state = phased_turns_bf16::buffers(
        shape.ranks(),
        u64::from(PHASED_ROWS) * u64::from(shape.profile.hidden),
    )
    .map_err(|refusal| refusal.to_string())?;
    Ok(CommBytes {
        inbox: state.inbox_bytes,
        flags: state.flag_bytes,
        epoch: state.epoch_bytes,
    })
}

/// The most rows of GLM's own all-reduce that take the F32 one-shot, below
/// the shared tuning's byte threshold (16 rows): none. The one-shot writes
/// every peer at once, which on this PCIe fabric collapses to about 2 GB/s;
/// on four ranks it took 22, 44, 83, 159 and 371 us at 1, 2, 4, 8 and 16
/// rows, the phased pair 13.6, 14.6, 14.8, 15.7 and 18.6 us (kernel-lab
/// `comm_phased_f32` timing, 2026-10-08). Both sum the ranks in rank order
/// from F32 zero, so every count gets the same bits from either.
pub const ONE_SHOT_MAX_ROWS: u32 = 0;

/// Whether `rows` rows reduce in one shot: at most [`ONE_SHOT_MAX_ROWS`],
/// and only where the shared tuning allows it.
pub fn one_shot(shape: Shape, rows: u32) -> Result<bool, String> {
    // `rows <= ONE_SHOT_MAX_ROWS`, written so it stays meaningful at zero.
    let within = ONE_SHOT_MAX_ROWS.checked_sub(rows).is_some();
    Ok(within && reduce_variant(shape, rows)? == Variant::OneShot)
}

fn reduce_variant(shape: Shape, rows: u32) -> Result<Variant, String> {
    kernels_sm120::comm::choose(&reduce_query(shape, rows)).map_err(|refusal| refusal.to_string())
}

/// The launch plan of the all-reduce of `rows` rows: the one-shot's or the
/// phased pair's, as [`one_shot`] picks.
pub fn reduce_plan(shape: Shape, rows: u32) -> Result<kernel_core::contract::LaunchPlan, String> {
    let variant = match one_shot(shape, rows)? {
        true => Variant::OneShot,
        false => Variant::TwoShot,
    };
    kernels_sm120::comm::plan_variant(variant, &reduce_query(shape, rows), 1)
        .map_err(|refusal| refusal.to_string())
}

/// One protocol's per-rank comm state: the inbox peers write, the flags
/// peers set and the local epoch, flags and epoch zeroed at create.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommBytes {
    pub inbox: u64,
    pub flags: u64,
    pub epoch: u64,
}

/// The most rows that reduce in one shot; every count below reduces in one
/// shot too.
pub fn one_shot_rows(shape: Shape) -> Result<u32, String> {
    let mut rows = 0;
    while rows < PREFILL_TOKENS && one_shot(shape, rows + 1)? {
        rows += 1;
    }
    Ok(rows)
}

/// The state of `chop_comm_allreduce_oneshot_f32_v1` over `rows` rows, as
/// `allreduce_oneshot` (`crates/kernels-sm120/cuda/comm/comm.cuh`) indexes
/// it: one CTA and one epoch cell per 4096-element tile; per rank and tile
/// a data flag (`rank * tiles + tile`) and an acknowledgment (`ranks *
/// tiles` past it); an inbox slot of the whole reduction per source rank.
pub fn one_shot_bytes(shape: Shape, rows: u32) -> CommBytes {
    let ranks = u64::from(shape.ranks());
    let elements = u64::from(rows) * u64::from(shape.profile.hidden);
    let tiles = elements.div_ceil(4096);
    CommBytes {
        inbox: ranks * elements * 4,
        flags: 2 * ranks * tiles * 4,
        epoch: tiles * 4,
    }
}

/// The phased reduce-scatter and all-gather's state, which holds any
/// reduction up to the prefill capacity: its flags and epoch are fixed
/// (`kernels_sm120::comm::phased_buffers`), its inbox the capacity's.
pub fn two_shot_bytes(shape: Shape) -> Result<CommBytes, String> {
    let elements = u64::from(PREFILL_TOKENS) * u64::from(shape.profile.hidden);
    let phased = kernels_sm120::comm::phased_buffers(
        shape.ranks(),
        elements,
        kernel_core::contract::DType::F32,
    )
    .map_err(|refusal| refusal.to_string())?;
    Ok(CommBytes {
        inbox: phased.inbox_bytes,
        flags: phased.flag_bytes,
        epoch: phased.epoch_bytes,
    })
}

/// The all-reduce states that hold any reduction of up to the prefill
/// capacity: one one-shot state per row count it serves, as its flag
/// offsets and epoch cells follow the tile count (`one_shot[rows - 1]`),
/// and the two-shot's one state of the capacity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommLayout {
    pub one_shot: Vec<CommBytes>,
    pub two_shot: CommBytes,
}

pub fn comm_layout(shape: Shape) -> Result<CommLayout, String> {
    Ok(CommLayout {
        one_shot: (1..=one_shot_rows(shape)?)
            .map(|rows| one_shot_bytes(shape, rows))
            .collect(),
        two_shot: two_shot_bytes(shape)?,
    })
}

/// The comm cubin: the shared comm engine's F32 all-reduce entries and its
/// F32 logits row exchange and gather.
fn comm_cubin_spec() -> CubinSpec {
    let mut spec = kernels_sm120::comm::cubin_spec();
    spec.name = format!("comm_{}_sm120", PROFILE.tag());
    spec.entries
        .retain(|entry| COMM_SYMBOLS.contains(&entry.symbol.as_str()));
    spec
}

/// GLM's own row kernels (`models/glm5next/bundle/cuda/row/glm5next.cuh`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelRow {
    /// An all-reduced F32 row rounded once to BF16.
    Round,
    /// A BF16 row widened into an F32 rank partial.
    Widen,
    /// The rows that need logits, gathered by the prefill's row indices.
    GatherRows,
    /// The MTP drafter's residual add into its single stream.
    ResidualAdd,
    /// The drafter rows' hidden inputs and tokens, by source codes.
    DraftGather,
}

impl ModelRow {
    pub const ALL: [Self; 5] = [
        Self::Round,
        Self::Widen,
        Self::GatherRows,
        Self::ResidualAdd,
        Self::DraftGather,
    ];

    pub fn symbol(self) -> String {
        let name = match self {
            Self::Round => "round",
            Self::Widen => "widen",
            Self::GatherRows => "gather_rows",
            Self::ResidualAdd => "residual_add",
            Self::DraftGather => "draft_gather",
        };
        format!("chop_row_{}_{name}_v1", PROFILE.tag())
    }

    /// The entry over `shape`'s hidden rows.
    pub fn entry(self, shape: Shape) -> kernel_build::EntrySpec {
        use kernel_build::EntryParam;
        let ptr = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
            name: name.into(),
            cxx_type: cxx_type.into(),
        };
        let p = shape.profile;
        let width = p.hidden;
        let ns = "chop::row::glm5next";
        let (params, body) = match self {
            Self::Round => (
                vec![
                    ptr("input", "const float"),
                    ptr("output", "uint16_t"),
                    ptr("live_rows", "const uint32_t"),
                ],
                format!("{ns}::round_rows<{width}>(input,output,live_rows);"),
            ),
            Self::Widen => (
                vec![
                    ptr("input", "const uint16_t"),
                    ptr("output", "float"),
                    ptr("live_rows", "const uint32_t"),
                ],
                format!("{ns}::widen_rows<{width}>(input,output,live_rows);"),
            ),
            Self::GatherRows => (
                vec![
                    ptr("input", "const uint16_t"),
                    ptr("row_indices", "const uint32_t"),
                    ptr("count", "const uint32_t"),
                    ptr("output", "uint16_t"),
                ],
                format!("{ns}::gather_rows<{width}>(input,row_indices,count,output);"),
            ),
            Self::ResidualAdd => (
                vec![
                    ptr("x", "uint16_t"),
                    ptr("y", "const uint16_t"),
                    ptr("live_rows", "const uint32_t"),
                ],
                format!("{ns}::residual_add<{width}>(x,y,live_rows);"),
            ),
            Self::DraftGather => (
                vec![
                    ptr("retained", "const uint16_t"),
                    ptr("last", "const uint16_t"),
                    ptr("codes", "const int32_t"),
                    ptr("drafts", "const uint32_t"),
                    ptr("live_rows", "const uint32_t"),
                    ptr("output", "uint16_t"),
                    ptr("tokens", "uint32_t"),
                ],
                format!(
                    "{ns}::draft_gather<{width}>(retained,last,codes,drafts,live_rows,output,tokens);"
                ),
            ),
        };
        kernel_build::EntrySpec::new(self.symbol(), params, body, 256, 1)
    }

    /// One CTA of 256 threads per row of the capacity bucket `capacity`.
    pub fn plan(self, capacity: u32) -> Result<kernel_core::contract::LaunchPlan, String> {
        if !kernels_sm120::row::capacity_bucket(capacity) {
            return Err(format!("{}: no row bucket of {capacity}", self.symbol()));
        }
        let plan = kernel_core::contract::LaunchPlan {
            grid: [capacity, 1, 1],
            block: [256, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        kernel_core::contract::Validate::validate(&plan).map_err(|r| r.to_string())?;
        Ok(plan)
    }
}

/// The row cubin: the RMSNorms, the embedding gather and GLM's own rows,
/// whose header lives beside the bundle (`models/glm5next/bundle/cuda`).
fn row_cubin_spec(shape: Shape) -> Result<CubinSpec, String> {
    let mut entries = Vec::new();
    for width in shape.rmsnorm_widths() {
        entries.push(rmsnorm(shape, width, PREFILL_TOKENS)?.entry(rmsnorm_symbol(width)));
    }
    let embedding = embedding_entry(shape);
    entries.push(kernel_build::EntrySpec::new(
        embedding.symbol,
        embedding.params,
        embedding.body,
        embedding.threads,
        1,
    ));
    entries.extend(ModelRow::ALL.map(|row| row.entry(shape)));
    Ok(CubinSpec {
        name: format!("row_{}_sm120", PROFILE.tag()),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."),
        include_dirs: vec![
            "crates/kernels-sm120/cuda".into(),
            "models/glm5next/bundle/cuda".into(),
        ],
        headers: vec![
            "#include \"row/row.cuh\"".into(),
            format!("#include \"{}\"", kernels_sm120::row::rmsnorm::TREE_HEADER),
            "#include \"row/glm5next.cuh\"".into(),
        ],
        defines: vec![],
        entries,
    })
}

/// The shared row copies' cubin: both shared entries under their
/// kernel-base symbols, compiled once for every shape (their widths,
/// streams and strides are launch arguments), so no other cubin of the
/// bundle holds them.
pub fn row_copy_cubin_spec() -> CubinSpec {
    kernels_sm120::row::copy::cubin_spec(ROW_COPY_CUBIN, &kernels_sm120::row::copy::entries())
}

pub fn cubin_specs(shape: Shape) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let latent = latent::cubin_spec(PROFILE, &latent_entries(shape))
        .map_err(|refusal| format!("latent attention cubin: {refusal}"))?;
    let latent_slices = latent::slices::cubin_spec(PROFILE, &latent_slices_entries(shape), "", &[])
        .map_err(|refusal| format!("slice-widened latent attention cubin: {refusal}"))?;
    let channel_decay = channel_decay::cubin_spec(PROFILE, &channel_decay_entries(shape));
    let channel_decay_streamed =
        channel_decay::streamed_cubin_spec(PROFILE, &channel_decay_streamed_entries(shape));
    let indexer = pooled_indexer::cubin_spec(PROFILE, &pooled_indexer_entries(shape))
        .map_err(|refusal| format!("pooled indexer cubin: {refusal}"))?;
    let indexer_rows =
        pooled_indexer::rows::cubin_spec(PROFILE, &pooled_indexer_rows_entries(shape), "", &[])
            .map_err(|refusal| format!("one-split pooled indexer cubin: {refusal}"))?;
    let sinkhorn = sinkhorn_hc::cubin_spec(
        format!("sinkhorn_hc_{}_sm120", PROFILE.tag()),
        &sinkhorn_entries(shape),
    );
    let sinkhorn_split = sinkhorn_hc::split::cubin_spec(
        format!("sinkhorn_hc_split_{}_sm120", PROFILE.tag()),
        &[sinkhorn_split_entry(shape)],
    )
    .map_err(|refusal| format!("split Sinkhorn cubin: {refusal}"))?;
    let mut specs = vec![
        (LATENT_FAMILY, latent),
        (CHANNEL_DECAY_FAMILY, channel_decay),
        (CHANNEL_DECAY_STREAMED_FAMILY, channel_decay_streamed),
        (POOLED_INDEXER_FAMILY, indexer),
        (SINKHORN_FAMILY, sinkhorn),
        (SINKHORN_SPLIT_FAMILY, sinkhorn_split),
    ];
    moe_shapes(shape)?;
    // Block warps, columns, then decoded tiles: the cubins load in the
    // order they did before the routed experts moved to their split cubins.
    let mut moe = chained::projection_cubin_specs(chained::PREFIX, &fp8_cubin_projections(shape)?);
    moe.sort_by_key(|spec| !spec.name.ends_with("_bw4"));
    for spec in moe {
        specs.push((MOE_FAMILY, spec));
    }
    // The NVFP4 exports' formats of the dense MLP and the shared expert: the
    // served catalog's, one cubin each with every served entry of its
    // format and schedule (the cubins kernel-lab qualifies), then
    // `chained::formats`' NVFP4 dense gate/up.
    for spec in chained::projection_cubin_specs(chained::PREFIX, &served_format_projections())
        .into_iter()
        .chain(chained::projection_cubin_specs(
            chained::formats::PREFIX,
            &added_format_projections(),
        ))
    {
        specs.push((MOE_FORMATS_FAMILY, spec));
    }
    specs.push((ROW_FAMILY, row_cubin_spec(shape)?));
    specs.extend(super::context_parallel::cubin_specs(shape)?);
    specs.push((super::payload::FAMILY, super::payload::cubin_spec()));
    specs.push((super::state_copy::FAMILY, super::state_copy::cubin_spec()));
    specs.push((COMM_FAMILY, comm_cubin_spec()));
    specs.push((LL_FAMILY, ll_bf16::cubin_spec(LL_CUBIN.into())));
    specs.push((
        COMM_EPILOGUE_FAMILY,
        sinkhorn_split_tree::cubin_spec(
            COMM_EPILOGUE_CUBIN.into(),
            &[comm_epilogue_entry(shape)?],
            &[],
        ),
    ));
    specs.push((
        SINKHORN_MIX_NORM_TREE_FAMILY,
        sinkhorn_hc::mix_norm_tree::cubin_spec(
            format!("sinkhorn_hc_mix_norm_tree_{}_sm120", PROFILE.tag()),
            &[sinkhorn_mix_norm_tree_entry(shape)],
            &[],
        )
        .map_err(|refusal| format!("fused Sinkhorn mix and tree norm cubin: {refusal}"))?,
    ));
    specs.push((
        POOLED_INDEXER_MERGE_TREE_FAMILY,
        pooled_indexer::merge_tree::cubin_spec(
            PROFILE,
            &pooled_indexer_merge_tree_entries(shape),
            "",
            &[],
        )
        .map_err(|refusal| format!("pooled indexer tree merge cubin: {refusal}"))?,
    ));
    specs.push((PHASED_FAMILY, phased_bf16::cubin_spec(PHASED_CUBIN.into())));
    specs.push((
        PHASED_TURNS_FAMILY,
        phased_turns_bf16::cubin_spec(PHASED_TURNS_CUBIN.into()),
    ));
    specs.push((
        SINKHORN_TILED_FAMILY,
        sinkhorn_hc::tiled::cubin_spec(
            format!("sinkhorn_hc_tiled_{}_sm120", PROFILE.tag()),
            &sinkhorn_tiled_entries(shape),
            &[],
        )
        .map_err(|refusal| format!("tiled Sinkhorn mix cubin: {refusal}"))?,
    ));
    specs.push((LATENT_SLICES_FAMILY, latent_slices));
    specs.push((POOLED_INDEXER_ROWS_FAMILY, indexer_rows));
    specs.push((
        LATENT_MERGE_COLUMNS_FAMILY,
        latent::merge_columns::cubin_spec(PROFILE, &latent_merge_columns_entries(shape), "", &[])
            .map_err(|refusal| format!("latent column merge cubin: {refusal}"))?,
    ));
    specs.push((
        POOLED_INDEXER_MERGE_RADIX_FAMILY,
        pooled_indexer::merge_radix::cubin_spec(
            PROFILE,
            &pooled_indexer_merge_radix_entries(shape),
            "",
            &[],
        )
        .map_err(|refusal| format!("pooled indexer radix merge cubin: {refusal}"))?,
    ));
    specs.push((
        ARGMAX_FAMILY,
        argmax::cubin_spec(ARGMAX_CUBIN, &argmax_entries()),
    ));
    // The MoE sublayers' rounded all-reduce of F32 partials.
    specs.push((
        LL_F32_FAMILY,
        kernels_sm120::comm::ll_f32::cubin_spec(LL_F32_CUBIN.into()),
    ));
    specs.push((
        PHASED_F32_FAMILY,
        kernels_sm120::comm::phased_f32::cubin_spec(PHASED_F32_CUBIN.into()),
    ));
    // The routed experts split over the ranks, as kernel-lab builds them:
    // their projections, the routes with both route sums, and the F32 sum
    // that adds a row ([`MOE_SUM_ADD_CUBIN`]).
    for spec in chained::split::cubin_specs()
        .map_err(|refusal| format!("split routed expert cubins: {refusal}"))?
    {
        specs.push((MOE_SPLIT_FAMILY, spec));
    }
    Ok(specs)
}

/// The shared argmax entries the drafter's argmax runs: the partial over
/// F32 logits and the final, under their shared symbols, so each launch
/// claims what the shared entries declare.
pub fn argmax_entries() -> Vec<argmax::Entry> {
    argmax::entries()
        .into_iter()
        .filter(|entry| entry.pattern != argmax::Pattern::PartialBf16)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::RecurrentPhase;
    use kernel_core::rmsnorm::RmsNormNumerics;

    /// The phased state takes one payload format: every phased entry SM120's
    /// table gives a row count of the prefill chunk is a BF16 entry of
    /// `phased_bf16` or `phased_turns_bf16`, none of `phased_f32`'s.
    #[test]
    fn the_phased_state_takes_only_the_bf16_entries() {
        let bf16: Vec<String> = phased_bf16::entries()
            .into_iter()
            .chain(phased_turns_bf16::entries())
            .map(|entry| entry.symbol)
            .collect();
        for rows in 1..=PHASED_ROWS {
            let symbol = bf16_tuned(rows).symbol();
            assert!(
                symbol == LL_SYMBOL || PHASED_SYMBOLS.contains(&symbol),
                "{rows} {symbol}"
            );
        }
        for symbol in PHASED_SYMBOLS {
            assert!(bf16.iter().any(|served| served == symbol), "{symbol}");
            assert!(symbol.ends_with("_bf16_v1"), "{symbol}");
        }
    }

    /// The capacity plan's F32 MoE all-reduce term is the state create
    /// allocates for it: the low-latency round entry's inbox and epochs
    /// (262,160 B) and the phased entries' inbox, flags and epoch
    /// (151,028,748 B).
    #[test]
    fn the_planned_f32_all_reduce_state_is_the_allocated_one() {
        let shape = Shape::TP4;
        let ll = rounded_ll_state_bytes(shape).unwrap();
        let phased = rounded_phased_state_bytes(shape).unwrap();
        assert_eq!(ll.inbox + ll.epochs, 262_160);
        assert_eq!(phased.inbox + phased.flags + phased.epoch, 151_028_748);
        let planned = crate::capacity::workspace(shape.geometry, shape.ranks(), 1).moe_all_reduce;
        assert_eq!(
            planned,
            ll.inbox + ll.epochs + phased.inbox + phased.flags + phased.epoch
        );
    }

    #[test]
    fn engine_geometries_follow_the_contract() {
        let shape = Shape::TP4;
        let g = latent_geometry(shape);
        assert_eq!(
            (g.heads, g.latent, g.max_keys, g.nope, g.v_dim),
            (16, 512, 2051, 256, 256)
        );
        assert_eq!(g.softmax_scale, 0.0625);
        // The engine's served KDA shape is the contract's at TP4.
        assert_eq!(channel_decay_geometry(shape), channel_decay::SERVED);
        let entries = channel_decay_entries(shape);
        for phase in [RecurrentPhase::Prefill, RecurrentPhase::Decode] {
            assert!(entries.iter().any(|entry| entry.query
                == channel_decay::Query::Rule(shape.kda_delta_rule(phase, entry_capacity(phase)))));
        }
        for entry in &entries {
            entry.contract().unwrap();
        }
        // The streamed prepare and scan take the staged pair's queries.
        let streamed = channel_decay_streamed_entries(shape);
        assert_eq!(
            streamed
                .iter()
                .map(|entry| &entry.symbol)
                .collect::<Vec<_>>(),
            [
                "chop_channel_decay_glm5next_delta_prefill_prepare_streamed_v1",
                "chop_channel_decay_glm5next_delta_prefill_scan_streamed_v1",
            ]
        );
        for entry in &streamed {
            entry.contract().unwrap();
            assert!(
                entries
                    .iter()
                    .any(|staged| staged.stage == entry.stage && staged.query == entry.query)
            );
        }
        // The indexer and mixer are the engines' served shapes.
        assert_eq!(sinkhorn_geometry(shape), sinkhorn_hc::SERVED);
        for entry in pooled_indexer_entries(shape) {
            entry.contract().unwrap();
        }
        // A tree merge for the global selection and its per-rank forms, each
        // with its selection's merge grid and ABI.
        let merges = pooled_indexer_merge_tree_entries(shape);
        assert_eq!(
            merges
                .iter()
                .map(pooled_indexer::merge_tree::Entry::symbol)
                .collect::<Vec<_>>(),
            [
                "chop_pooled_indexer_glm5next_select_merge_tree_v1",
                "chop_pooled_indexer_glm5next_select_per_rank2_merge_tree_v1",
                "chop_pooled_indexer_glm5next_select_per_rank4_merge_tree_v1",
            ]
        );
        for merge in &merges {
            merge.contract().unwrap();
            assert_eq!(merge.params(), merge.select.merge_params());
            assert_eq!(
                merge.plan(4).unwrap().grid,
                merge.select.merge_plan(4).unwrap().unwrap().grid
            );
        }
        // A radix merge for the global selection, with its merge grid and
        // ABI.
        let radix = pooled_indexer_merge_radix_entries(shape);
        assert_eq!(
            radix
                .iter()
                .map(pooled_indexer::merge_radix::Entry::symbol)
                .collect::<Vec<_>>(),
            ["chop_pooled_indexer_glm5next_select_merge_radix_v1"]
        );
        for merge in &radix {
            merge.contract().unwrap();
            assert_eq!(merge.params(), merge.select.merge_params());
            assert_eq!(
                merge.plan(4).unwrap().grid,
                merge.select.merge_plan(4).unwrap().unwrap().grid
            );
        }
        for entry in sinkhorn_entries(shape) {
            entry.contract().unwrap();
        }
        let split = sinkhorn_split_entry(shape);
        split.contract().unwrap();
        assert_eq!(
            split.symbol,
            "chop_sinkhorn_hc_glm5next_mix_collapse_split_v1"
        );
        // The fused mixer and input norm: the split entry's mix-collapse and
        // grid, the row engine's norm of the hidden row with the tree sum.
        let fused = sinkhorn_mix_norm_tree_entry(shape);
        fused.contract().unwrap();
        assert_eq!(fused.query, split.query);
        assert_eq!(fused.norm, shape.rmsnorm_requirement(shape.profile.hidden));
        assert_eq!(fused.norm.numerics, Some(RmsNormNumerics::HF_TREE_256));
        assert_eq!(fused.plan(32, 32).unwrap(), split.plan(32, 32).unwrap());
        assert_eq!(
            fused.symbol,
            "chop_sinkhorn_hc_glm5next_mix_collapse_norm_tree_v1"
        );
        // Hidden rows sum their squares in the tree; the query-down and
        // latent rows in order.
        let [hidden, q_rank, kv_rank] = shape.rmsnorm_widths();
        assert_eq!(
            shape.rmsnorm_requirement(hidden).numerics,
            Some(RmsNormNumerics::HF_TREE_256)
        );
        for width in [q_rank, kv_rank] {
            assert_eq!(
                shape.rmsnorm_requirement(width).numerics,
                Some(RmsNormNumerics::HF_ORDERED)
            );
        }
        // The tiled mix-collapse at the served schedule's tiles: the one-CTA
        // entry's query and ABI, one CTA per tile of rows.
        let one_cta = &sinkhorn_entries(shape)[0];
        assert_eq!(one_cta.symbol, "chop_sinkhorn_hc_glm5next_mix_collapse_v1");
        let tiled = sinkhorn_tiled_entries(shape);
        assert_eq!(
            tiled.iter().map(|e| e.symbol.as_str()).collect::<Vec<_>>(),
            [
                "chop_sinkhorn_hc_glm5next_mix_collapse_tiled_r4_v1",
                "chop_sinkhorn_hc_glm5next_mix_collapse_tiled_r8_v1",
            ]
        );
        for entry in &tiled {
            entry.contract().unwrap();
            assert_eq!(entry.query, split.query);
            assert_eq!(entry.params, one_cta.params);
        }
        assert_eq!(tiled[0].plan(952, 952).unwrap().grid, [238, 1, 1]);
        assert_eq!(tiled[1].plan(2048, 2048).unwrap().grid, [256, 1, 1]);
        // Every tile the served schedule picks is embedded.
        for rows in 1..=PREFILL_TOKENS {
            if let sinkhorn_hc::tiled::PrefillMix::Tiled(tile) =
                sinkhorn_hc::tiled::served_prefill_mix(rows)
            {
                assert!(SINKHORN_TILES.contains(&tile), "{rows} rows");
            }
        }
        // Dense and shared, gate/up and down, decode and prefill, then the
        // prefill gate/up's decoded tiles.
        assert_eq!(moe_projections(shape).unwrap().len(), 2 * 2 * 2 + 2);
        let decoded: Vec<_> = moe_decoded_tiles(shape)
            .unwrap()
            .into_iter()
            .map(|p| (p.columns(), p.row_tile, p.schedule))
            .collect();
        assert_eq!(
            decoded,
            [
                (3072, 64, chained::Schedule::DecodedTiles),
                (512, 64, chained::Schedule::DecodedTiles)
            ]
        );
        let plan = moe_decoded_tiles(shape).unwrap()[0]
            .plan(PREFILL_TOKENS)
            .unwrap();
        assert_eq!(
            (plan.block, plan.dynamic_shared_bytes),
            ([384, 1, 1], chained::DECODED_TILE_SHARED_BYTES)
        );
        // No routed projection outside the split: the FP8 cubins hold the
        // dense MLP's and the shared expert's projections alone.
        assert!(
            fp8_cubin_projections(shape)
                .unwrap()
                .iter()
                .all(|projection| projection.query.gemm.grouped.is_none())
        );
        // The routed experts split over the ranks: a quarter of each
        // expert's intermediate on every rank, routed over every expert,
        // with the FP8 prefill gate/up and down in tile pairs at the 64-row
        // route padding.
        let routed = moe_routed_projections(shape).unwrap();
        assert_eq!(routed, chained::split::forms());
        for projection in &routed {
            let grouped = projection.query.gemm.grouped.unwrap();
            assert_eq!((grouped.experts, grouped.top_k), (288, 8));
        }
        let pairs: Vec<_> = routed
            .iter()
            .filter(|p| matches!(p.schedule, chained::Schedule::TilePairs { .. }))
            .map(|p| p.symbol.as_str())
            .collect();
        assert_eq!(
            pairs,
            [
                "chop_moe_chain_gateup_fp8block_e288k8_n1024_k4096_r64_clamp4120000041200000_tp2_v1",
                "chop_moe_chain_down_fp8block_e288k8_n4096_k512_r64_tp0_v1",
            ]
        );
        assert_eq!(moe_shapes(shape).unwrap().routes()[1].pad, 64);
        // BF16 dense and shared at both dense intermediates, then the NVFP4
        // dense gate/up.
        assert_eq!(moe_format_projections(shape).unwrap().len(), 2 * 2 * 2 + 2);
        // The catalog's BF16 dense and shared decode projections run as
        // block warps, the cubins kernel-lab qualifies
        // (`chained::block_warp_catalog`); `chained::formats`' stay in
        // columns.
        let warps: Vec<_> = moe_format_projections(shape)
            .unwrap()
            .into_iter()
            .filter(|p| p.schedule == chained::Schedule::BlockWarps)
            .map(|p| p.symbol)
            .collect();
        let qualified: Vec<_> = chained::block_warp_catalog()
            .into_iter()
            .filter(|p| p.weights() == chained::Weights::Bf16 && p.query.gemm.grouped.is_none())
            .map(|p| p.symbol)
            .collect();
        assert_eq!(warps.len(), 4);
        assert_eq!(warps, qualified);
        for entry in latent_entries(shape) {
            entry.contract().unwrap();
            assert!(entry.symbol().starts_with("chop_latent_glm5next_"));
        }
        // The column merges of the decode and verify entries, each with its
        // entry's merge ABI and workspace.
        let merges = latent_merge_columns_entries(shape);
        assert_eq!(merges.len(), 8);
        for merge in &merges {
            merge.contract().unwrap();
            assert_eq!(merge.params(), merge.attention.merge_params());
            assert_eq!(
                merge.plan(4).unwrap().workspace_bytes,
                merge
                    .attention
                    .merge_plan(4)
                    .unwrap()
                    .unwrap()
                    .workspace_bytes
            );
        }
        assert!(merges.iter().any(|merge| {
            merge.symbol() == "chop_latent_glm5next_attn_fp8_verify_merge_columns_v1"
        }));
        // The slice-widened forms of the FP8 prefill entries, each with its
        // entry's grid and ABI.
        let slices = latent_slices_entries(shape);
        assert_eq!(
            slices
                .iter()
                .map(latent::slices::Entry::symbol)
                .collect::<Vec<_>>(),
            [
                "chop_latent_glm5next_attn_fp8_prefill_slices_v1",
                "chop_latent_glm5next_attn_fp8_prefill_partial_slices_v1",
            ]
        );
        for entry in &slices {
            entry.contract().unwrap();
            assert_eq!(entry.params(), entry.attention.params());
            for rows in [1, 376, 8192] {
                assert_eq!(
                    entry.plan(rows).unwrap().grid,
                    entry.attention.plan(rows).unwrap().grid
                );
            }
        }
        let specs = cubin_specs(shape).unwrap();
        assert_eq!(
            specs
                .iter()
                .map(|(family, spec)| (*family, spec.name.as_str()))
                .collect::<Vec<_>>(),
            [
                (LATENT_FAMILY, "latent_attention_glm5next_sm120"),
                (CHANNEL_DECAY_FAMILY, "channel_decay_glm5next_sm120"),
                (
                    CHANNEL_DECAY_STREAMED_FAMILY,
                    "channel_decay_streamed_glm5next_sm120"
                ),
                (POOLED_INDEXER_FAMILY, "pooled_indexer_glm5next_sm120"),
                (SINKHORN_FAMILY, "sinkhorn_hc_glm5next_sm120"),
                (SINKHORN_SPLIT_FAMILY, "sinkhorn_hc_split_glm5next_sm120"),
                (MOE_FAMILY, "chop_moe_chain_e288l72k8_fp8block_bw4"),
                (MOE_FAMILY, "chop_moe_chain_e288l72k8_fp8block"),
                (MOE_FAMILY, "chop_moe_chain_e288l72k8_fp8block_dt"),
                (MOE_FORMATS_FAMILY, "chop_moe_chain_e288l72k8_bf16_bw4"),
                (MOE_FORMATS_FAMILY, "chop_moe_chain_e288l72k8_bf16"),
                (MOE_FORMATS_FAMILY, "chop_moe_chain_e288l72k8_formats_nvfp4"),
                (ROW_FAMILY, "row_glm5next_sm120"),
                (
                    crate::sm120::context_parallel::FAMILY,
                    "context_parallel_glm5next_sm120"
                ),
                (
                    crate::sm120::context_parallel::FAMILY,
                    "comm_group_glm5next_sm120"
                ),
                (
                    crate::sm120::context_parallel::PHASED_FAMILY,
                    "comm_group_phased_glm5next_sm120"
                ),
                (crate::sm120::payload::FAMILY, crate::sm120::payload::CUBIN),
                (
                    crate::sm120::state_copy::FAMILY,
                    crate::sm120::state_copy::CUBIN
                ),
                (COMM_FAMILY, "comm_glm5next_sm120"),
                (LL_FAMILY, LL_CUBIN),
                (COMM_EPILOGUE_FAMILY, COMM_EPILOGUE_CUBIN),
                (
                    SINKHORN_MIX_NORM_TREE_FAMILY,
                    "sinkhorn_hc_mix_norm_tree_glm5next_sm120"
                ),
                (
                    POOLED_INDEXER_MERGE_TREE_FAMILY,
                    "pooled_indexer_merge_tree_glm5next_sm120"
                ),
                (PHASED_FAMILY, PHASED_CUBIN),
                (PHASED_TURNS_FAMILY, PHASED_TURNS_CUBIN),
                (SINKHORN_TILED_FAMILY, "sinkhorn_hc_tiled_glm5next_sm120"),
                (
                    LATENT_SLICES_FAMILY,
                    "latent_attention_slices_glm5next_sm120"
                ),
                (
                    POOLED_INDEXER_ROWS_FAMILY,
                    "pooled_indexer_rows_glm5next_sm120"
                ),
                (
                    LATENT_MERGE_COLUMNS_FAMILY,
                    "latent_merge_columns_glm5next_sm120"
                ),
                (
                    POOLED_INDEXER_MERGE_RADIX_FAMILY,
                    "pooled_indexer_merge_radix_glm5next_sm120"
                ),
                (ARGMAX_FAMILY, ARGMAX_CUBIN),
                (LL_F32_FAMILY, LL_F32_CUBIN),
                (PHASED_F32_FAMILY, PHASED_F32_CUBIN),
                (MOE_SPLIT_FAMILY, "chop_moe_chain_e288l288k8_fp8block_ws"),
                (MOE_SPLIT_FAMILY, "chop_moe_chain_e288l288k8_fp8block_tp"),
                (MOE_SPLIT_FAMILY, "chop_moe_chain_e288l288k8_fp8block"),
                (MOE_SPLIT_FAMILY, "chop_moe_chain_e288l288k8_fp8block_bw4"),
                (MOE_SPLIT_FAMILY, "chop_moe_chain_e288l288k8_nvfp4_bw4"),
                (MOE_SPLIT_FAMILY, "chop_moe_chain_e288l288k8_nvfp4"),
                (MOE_SPLIT_FAMILY, "chop_moe_chain_e288l288k8_mxfp8"),
                (MOE_SPLIT_FAMILY, "chop_moe_chain_e288l288k8_bf16_bw4"),
                (MOE_SPLIT_FAMILY, "chop_moe_chain_e288l288k8_bf16"),
                (MOE_SPLIT_FAMILY, "chop_moe_chain_e288l288k8_route"),
                (MOE_SPLIT_FAMILY, MOE_SUM_ADD_CUBIN),
            ]
        );
        // The MoE sublayers' rounded all-reduce: the low-latency round entry
        // at one row (four CTAs of 1,024 elements), the phased F32 rounding
        // entries' largest plans, and the cubins holding them.
        assert_eq!(ROUNDED_LOW_LATENCY_ROWS, 1);
        assert_eq!(
            kernels_sm120::comm::ll_f32::plan(shape.ranks(), 4096)
                .unwrap()
                .grid,
            [4, 1, 1]
        );
        let rounded: Vec<_> = rounded_phased_admission_plans()
            .unwrap()
            .into_iter()
            .map(|(symbol, plan)| (symbol, plan.grid))
            .collect();
        assert_eq!(
            rounded,
            [
                (PHASED_ROUND_SYMBOLS[0], [16, 1, 1]),
                (PHASED_ROUND_SYMBOLS[1], [48, 1, 1]),
            ]
        );
        assert_eq!(
            PHASED_ROUND_SYMBOLS,
            [
                kernels_sm120::comm::phased_f32::ONE_SHOT_ROUND_SYMBOL,
                kernels_sm120::comm::phased_f32::ROUND_SYMBOL,
            ]
        );
        let has = |family: &str, symbol: &str| {
            specs
                .iter()
                .any(|(f, spec)| *f == family && spec.entries.iter().any(|e| e.symbol == symbol))
        };
        assert!(has(LL_F32_FAMILY, LL_ROUND_SYMBOL));
        for (symbol, _) in rounded {
            assert!(has(PHASED_F32_FAMILY, symbol), "{symbol}");
        }
        // The split cubins are kernel-lab's, spec for spec.
        let split: Vec<_> = specs
            .iter()
            .filter(|(family, _)| *family == MOE_SPLIT_FAMILY)
            .map(|(_, spec)| format!("{spec:?}"))
            .collect();
        let lab: Vec<_> = chained::split::cubin_specs()
            .unwrap()
            .iter()
            .map(|spec| format!("{spec:?}"))
            .collect();
        assert_eq!(split, lab);
        // The shared argmax cubin holds the F32 partial and the final under
        // their shared symbols.
        assert_eq!(
            specs
                .iter()
                .find(|(family, _)| *family == ARGMAX_FAMILY)
                .unwrap()
                .1
                .entries
                .iter()
                .map(|entry| entry.symbol.as_str())
                .collect::<Vec<_>>(),
            [argmax::PARTIAL_F32_SYMBOL, argmax::FINAL_SYMBOL]
        );
        // The one-split forms of the global and per-rank selections, one
        // row per CTA, each with its selection's capacity.
        let forms = pooled_indexer_rows_entries(shape);
        assert_eq!(
            forms
                .iter()
                .map(pooled_indexer::rows::Entry::symbol)
                .collect::<Vec<_>>(),
            [
                "chop_pooled_indexer_glm5next_select_rows1_v1",
                "chop_pooled_indexer_glm5next_select_per_rank2_rows1_v1",
                "chop_pooled_indexer_glm5next_select_per_rank4_rows1_v1",
            ]
        );
        for form in &forms {
            form.contract().unwrap();
            assert_eq!(form.select.splits(376), 1);
            assert_eq!(form.plan(2048).unwrap().grid, [2048, 1, 1]);
            assert!(form.plan(2049).is_err());
        }
        // The row engine normalizes each width at every capacity bucket and
        // gathers 4096-wide rows from the rank's 38,720-row shard.
        for width in shape.rmsnorm_widths() {
            for capacity in [1, 32, 8192] {
                rmsnorm(shape, width, capacity).unwrap();
            }
        }
        let embedding = embedding_entry(shape);
        embedding.contract().unwrap();
        assert_eq!(embedding.symbol, "chop_row_embedding_glm5next_v1");
        // The MoE sums every expert's routes in F32 and leaves the sum
        // unrounded.
        assert_eq!(
            routed_sum(shape).unwrap().symbol(chained::split::PREFIX),
            "chop_moe_chain_e288l288k8_sum_f32_k8_w4096_l288_t8192_v1"
        );
        // Its form that adds the shared expert's output has its own cubin.
        let sum_add = cubin_specs(shape)
            .unwrap()
            .into_iter()
            .filter(|(_, spec)| spec.name == MOE_SUM_ADD_CUBIN)
            .map(|(_, spec)| spec)
            .collect::<Vec<_>>();
        assert_eq!(sum_add.len(), 1);
        assert_eq!(
            sum_add[0]
                .entries
                .iter()
                .map(|e| e.symbol.as_str())
                .collect::<Vec<_>>(),
            ["chop_moe_chain_e288l288k8_sum_add_f32_k8_w4096_l288_t8192_v1"]
        );
        // Every row count reduces through the phased pair, whose plan has
        // the sending cluster.
        assert_eq!(one_shot_rows(shape).unwrap(), ONE_SHOT_MAX_ROWS);
        for rows in 1..=PREFILL_TOKENS {
            assert!(!one_shot(shape, rows).unwrap(), "{rows} rows");
        }
        // The low-latency state holds one row, the phased state the
        // prefill chunk.
        assert_eq!(BF16_LOW_LATENCY_ROWS, 1);
        assert_eq!(PHASED_ROWS, PREFILL_TOKENS);
        assert_eq!(PHASED_RESET_RUNS, 1 << 20);
        for rows in 1..=BF16_LOW_LATENCY_ROWS {
            let plan = ll_bf16::plan(shape.ranks(), u64::from(rows) * 4096).unwrap();
            assert_eq!(plan.grid, [2 * rows, 1, 1]);
        }
        assert_eq!(
            ll_state_bytes(shape).unwrap(),
            ll_bf16::StateBytes {
                inbox: 2 * 4 * 2 * 8192,
                epochs: 8,
            }
        );
        for rows in [1, 2, 16, 17, 2048] {
            let plan = reduce_plan(shape, rows).unwrap();
            assert_eq!(plan.cluster, [8, 1, 1], "{rows} rows");
            assert!(plan.grid[0] >= 16, "{rows} rows");
        }
        let layout = comm_layout(shape).unwrap();
        assert!(layout.one_shot.is_empty());
        assert_eq!((layout.two_shot.flags, layout.two_shot.epoch), (82 * 4, 4));
        let family = |name| &specs.iter().find(|(family, _)| *family == name).unwrap().1;
        assert_eq!(family(COMM_FAMILY).entries.len(), COMM_SYMBOLS.len());
        assert_eq!(family(LL_FAMILY).entries.len(), 1);
        assert_eq!(family(LL_FAMILY).name, LL_CUBIN);
        // The phased all-reduce: all three entries served; the admitted
        // plans are the one-shot's 16 senders and the two-shot's and turns
        // entry's widest geometry; one state of the prefill chunk holds the
        // one-shot's region and the two-shot's slots in BF16, and the turns
        // entry's flags.
        assert_eq!(family(PHASED_FAMILY).entries.len(), 2);
        assert_eq!(family(PHASED_TURNS_FAMILY).entries.len(), 1);
        for symbol in PHASED_SYMBOLS {
            assert!(
                [PHASED_FAMILY, PHASED_TURNS_FAMILY]
                    .into_iter()
                    .flat_map(|name| &family(name).entries)
                    .any(|entry| entry.symbol == symbol)
            );
        }
        let plans = bf16_phased_admission_plans().unwrap();
        assert_eq!(plans[0].0, phased_bf16::ONE_SHOT_SYMBOL);
        assert_eq!(plans[0].1.grid, [16, 1, 1]);
        assert_eq!(plans[1].0, phased_bf16::SYMBOL);
        assert_eq!(plans[1].1.grid, [48, 1, 1]);
        assert_eq!(plans[2].0, phased_turns_bf16::SYMBOL);
        assert_eq!(plans[2].1.grid, [48, 1, 1]);
        assert!(plans.iter().all(|(_, plan)| plan.cluster == [1, 1, 1]));
        assert_eq!(
            phased_state_bytes(shape).unwrap(),
            CommBytes {
                inbox: (phased_bf16::ONE_SHOT_REGION + 4 * 8192 * 1024) * 2,
                flags: phased_turns_bf16::FLAG_WORDS * 4,
                epoch: 4,
            }
        );
        // The comm epilogue's split rows entry over the mixer's capacity, at
        // the input norm's eps.
        let fused = comm_epilogue_entry(shape).unwrap();
        assert_eq!(
            fused.symbol,
            "chop_sinkhorn_epilogue_glm5next_rows_split_tree_v1"
        );
        assert_eq!(fused.mix, sinkhorn_geometry(shape).mix);
        assert_eq!(f32::from_bits(fused.rmsnorm_eps_bits), 1e-5);
        fused.plan(PREFILL_TOKENS, shape.ranks()).unwrap();
        assert_eq!(family(COMM_EPILOGUE_FAMILY).entries.len(), 1);
    }

    /// Every word `allreduce_oneshot` touches at every served one-shot row
    /// count lies in the state `one_shot_bytes` sizes for it: each CTA's
    /// epoch cell, each rank's data flags and acknowledgments on every peer
    /// and each source's inbox slot.
    #[test]
    fn one_shot_states_hold_the_kernel_tiles_at_every_served_row_count() {
        let shape = Shape::TP4;
        let ranks = u64::from(shape.ranks());
        for rows in 1..=one_shot_rows(shape).unwrap() {
            let bytes = one_shot_bytes(shape, rows);
            let elements = u64::from(rows) * u64::from(shape.profile.hidden);
            let tiles = elements.div_ceil(4096);
            let (mut flag, mut epoch, mut inbox) = (0, 0, 0);
            for rank in 0..ranks {
                for tile in 0..tiles {
                    let index = rank * tiles + tile;
                    epoch = epoch.max(tile);
                    flag = flag.max(ranks * tiles + index).max(index);
                    for source in 0..ranks {
                        flag = flag.max(ranks * tiles + source * tiles + tile);
                        inbox =
                            inbox.max(source * elements + (tile * 4096 + 4095).min(elements - 1));
                    }
                }
            }
            assert_eq!(bytes.flags, (flag + 1) * 4, "{rows} rows");
            assert_eq!(bytes.epoch, (epoch + 1) * 4, "{rows} rows");
            assert_eq!(bytes.inbox, (inbox + 1) * 4, "{rows} rows");
        }
    }

    fn entry_capacity(phase: RecurrentPhase) -> u32 {
        match phase {
            RecurrentPhase::Prefill => channel_decay::SERVED.prefill_rows,
            _ => channel_decay::SERVED.decode_sequences,
        }
    }
}
