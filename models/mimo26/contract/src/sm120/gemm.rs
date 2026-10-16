//! MiMo V2.6 Flash's dense GEMMs on the shared fragment, cluster and router
//! families: TP4's under `crate::sm120::config::PROFILE`, with every rank-invariant
//! shape, and the shapes that change with the rank count under the TP2
//! profile. The router is the fragment router GEMV over every expert.

use super::geometry::{self, GeometryModel};
use crate::geometry::V26_FLASH as GEOMETRY;

use kernel_core::contract::{DType, DenseKernel, SwigluColumns, WeightFormat};
use kernels_sm120::Profile;
use kernels_sm120::gemm::{self, GemmVariant, bf16_query, fp8_block_query, instance};
use kernels_sm120::invariance::{Chunk, Invariance};

/// (N, K) of `name` in MiMo's per-rank operation list at `ranks` ranks.
fn nk(owner: GeometryModel, ranks: u32, name: &str) -> (u32, u32) {
    shapes_for(owner, ranks)
        .into_iter()
        .find(|shape| shape.0 == name)
        .map(|(_, n, k, ..)| (n, k))
        .unwrap_or_else(|| panic!("MiMo operation {name}"))
}

/// The fixed prefill K segments of `name` in MiMo's operation list, the same
/// at every rank count.
fn segments(owner: GeometryModel, name: &str) -> u8 {
    shapes_for(owner, 4)
        .into_iter()
        .find(|shape| shape.0 == name)
        .map(|shape| shape.5)
        .unwrap_or_else(|| panic!("MiMo operation {name}"))
}

/// The decode kernel of `name` in MiMo's operation list at four ranks.
fn decode_kernel(owner: GeometryModel, name: &str) -> DenseKernel {
    shapes_for(owner, 4)
        .into_iter()
        .find(|shape| shape.0 == name)
        .map(|shape| shape.6)
        .unwrap_or_else(|| panic!("MiMo operation {name}"))
}

/// The verify rows MiMo's decode buckets serve exactly: up to eight rows,
/// each with its one-row bits (kernel-lab `ops` `invariance_rows`).
const EXACT_ROWS: u32 = crate::serving::EXACT_VERIFY_ROWS;

/// A MiMo decode bucket of `m` rows that verify steps run as exact rows.
fn exact_rows(item: GemmVariant, m: u32) -> GemmVariant {
    if m <= EXACT_ROWS {
        item.declare(Invariance::one_row(m))
    } else {
        item
    }
}

/// A MiMo BF16 head decode bucket, which also projects the logits rows of
/// prefill steps with few of them when the BF16 head serves them
/// (`CHOPINFER_MIMO_F32_LOGITS=0`): a fragment decode sums all of K for each
/// row alone, so a row keeps its bits whatever rows share the call, and
/// prefill rows keep them at any chunk.
fn bf16_head(item: GemmVariant) -> GemmVariant {
    let invariance = item.invariance.with_chunk(Chunk::Any);
    item.declare(invariance)
}

/// MiMo's TP4 dense entries, then TP2's (`rank_entries`). The TP4
/// shapes come from its operation list at four ranks (`rank_shapes`), with
/// TP4's names, tiles and 128-row block scales.
pub fn catalog() -> Vec<GemmVariant> {
    geometry::validate_names().unwrap_or_else(|error| panic!("{error}"));
    geometry::MODELS
        .iter()
        .flat_map(|owner| catalog_for(*owner))
        .collect()
}

/// The catalog entry of `symbol`, from a catalog built once.
pub fn entry(symbol: &str) -> Option<&'static GemmVariant> {
    static CATALOG: std::sync::OnceLock<Vec<GemmVariant>> = std::sync::OnceLock::new();
    CATALOG
        .get_or_init(catalog)
        .iter()
        .find(|entry| entry.symbol == symbol)
}

pub fn catalog_for(owner: GeometryModel) -> Vec<GemmVariant> {
    let g = owner.geometry;
    let tag = owner.geometry.kernel_tag;
    let mut items = Vec::new();
    let profile = owner.profile(4);
    let mimo = |name: &str| nk(owner, 4, name);
    // The router is rank-invariant: every expert's logit over the hidden.
    let router = (g.experts(), g.hidden());
    for m in [1, 2, 4, 8, 16, 32] {
        let tn = m.max(8) as u16;
        items.push(exact_rows(
            instance(
                profile,
                &format!("{tag}_o_m{m}"),
                bf16_query(m, mimo("o").0, mimo("o").1, DType::Bf16),
                64,
                tn,
                decode_kernel(owner, "o"),
            ),
            m,
        ));
        items.push(exact_rows(
            instance(
                profile,
                &format!("{tag}_router_m{m}"),
                bf16_query(m, router.0, router.1, DType::F32),
                64,
                tn,
                DenseKernel::FragmentRouter {
                    splits: gemm::ROUTER_SPLITS,
                },
            ),
            m,
        ));
        items.push(bf16_head(exact_rows(
            instance(
                profile,
                &format!("{tag}_head_m{m}"),
                bf16_query(m, mimo("head").0, mimo("head").1, DType::Bf16),
                64,
                tn,
                decode_kernel(owner, "head"),
            ),
            m,
        )));
        items.push(exact_rows(
            instance(
                profile,
                &format!("{tag}_head_f32_m{m}"),
                bf16_query(m, mimo("head_f32").0, mimo("head_f32").1, DType::F32),
                64,
                tn,
                decode_kernel(owner, "head_f32"),
            ),
            m,
        ));
        for name in ["qkv_ga", "qkv_swa", "mlp_gate_up", "mlp_down"] {
            let (n, k) = mimo(name);
            let mut query = fp8_block_query(m, n, k);
            if name == "mlp_gate_up" {
                query.epilogues.swiglu = Some(SwigluColumns::ConcatenatedHalves);
            }
            items.push(exact_rows(
                instance(
                    profile,
                    &format!("{tag}_{name}_m{m}"),
                    query,
                    64,
                    tn,
                    decode_kernel(owner, name),
                ),
                m,
            ));
        }
    }
    for m in [1, 2, 4, 8, 16, 32] {
        let tn = m.max(8) as u16;
        for name in ["q", "qkv", "context_kv", "o", "gate_up", "down", "fc"] {
            let (n, k) = mimo(&format!("dflash_{name}"));
            let mut query = bf16_query(m, n, k, DType::Bf16);
            if name == "gate_up" {
                query.epilogues.swiglu = Some(SwigluColumns::ConcatenatedHalves);
            }
            items.push(instance(
                profile,
                &format!("{tag}_dflash_{name}_m{m}"),
                query,
                64,
                tn,
                decode_kernel(owner, &format!("dflash_{name}")),
            ));
        }
    }
    // The router takes eight K segments (as `rank_shapes`).
    for (name, (n, k), output, tm, segments) in [
        ("o", mimo("o"), DType::Bf16, 128, segments(owner, "o")),
        ("router", router, DType::F32, 64, 8),
        (
            "head",
            mimo("head"),
            DType::Bf16,
            128,
            segments(owner, "head"),
        ),
    ] {
        items.push(instance(
            profile,
            &format!("{tag}_{name}_prefill"),
            bf16_query(8192, n, k, output),
            tm,
            128,
            DenseKernel::FragmentPrefill { segments },
        ));
    }
    for name in ["q", "qkv", "context_kv", "o", "down", "fc"] {
        let (n, k) = mimo(&format!("dflash_{name}"));
        items.push(instance(
            profile,
            &format!("{tag}_dflash_{name}_prefill"),
            bf16_query(8192, n, k, DType::Bf16),
            128,
            128,
            DenseKernel::FragmentPrefill {
                segments: segments(owner, &format!("dflash_{name}")),
            },
        ));
    }
    let (n, k) = mimo("dflash_gate_up");
    let mut dflash_gate_up = bf16_query(8192, n, k, DType::Bf16);
    dflash_gate_up.epilogues.swiglu = Some(SwigluColumns::ConcatenatedHalves);
    items.push(instance(
        profile,
        &format!("{tag}_dflash_gate_up_prefill"),
        dflash_gate_up,
        128,
        128,
        DenseKernel::FragmentPrefill {
            segments: segments(owner, "dflash_gate_up"),
        },
    ));
    items.push(instance(
        profile,
        &format!("{tag}_head_f32_prefill"),
        bf16_query(8192, mimo("head_f32").0, mimo("head_f32").1, DType::F32),
        128,
        128,
        DenseKernel::FragmentPrefill {
            segments: segments(owner, "head_f32"),
        },
    ));
    for name in ["qkv_ga", "qkv_swa", "mlp_gate_up", "mlp_down"] {
        let (n, k) = mimo(name);
        let mut query = fp8_block_query(8192, n, k);
        if name == "mlp_gate_up" {
            query.epilogues.swiglu = Some(SwigluColumns::ConcatenatedHalves);
        }
        items.push(instance(
            profile,
            &format!("{tag}_{name}_prefill"),
            query,
            64,
            128,
            DenseKernel::FragmentPrefill {
                segments: segments(owner, name),
            },
        ));
    }
    if !geometry::pinned_names(g) {
        rank_entries(owner, &mut items, 4);
    }
    for ranks in OTHER_RANKS {
        rank_entries(owner, &mut items, ranks);
    }
    items
}

/// MiMo rank counts besides TP4 with a separate profile for per-rank GEMMs.
pub const OTHER_RANKS: [u32; 1] = [2];

/// A MiMo projection: (name, N, K, weight, SwiGLU, prefill K segments,
/// decode kernel).
pub type RankProjection = (&'static str, u32, u32, WeightFormat, bool, u8, DenseKernel);

/// One rank's MiMo projection shapes at `ranks` ranks, derived from
/// `mimo26`'s configuration. The router is the same at every rank count, so
/// TP4's entry serves all of them. The QKV rows are regrouped heads with
/// 64-row scales.
///
/// Each prefill entry's fixed K segments come from the kernel lab's 512-,
/// 1,024- (172 CTAs) and 2,048-row chunk timings at TP4
/// (`gemm_prefill_chunk_bench`, 2026-09-30): the fastest of 2, 4 and 8
/// summed over those chunks, or 4 within 1% of it. More segments balance a
/// stream-K tail of few tiles; fewer write fewer partials.
///
/// Each decode kernel is declared per rank count (TP2, TP4): the
/// cluster decode where it measured faster on GPU 0 (3,000 to 8,192 output
/// rows over K = 2,048 or 4,096), otherwise fragment decode summing all of
/// K per tile, as the bundles launch it.
pub fn rank_shapes(ranks: u32) -> Vec<RankProjection> {
    shapes_for(geometry::MODELS[0], ranks)
}

pub fn shapes_for(owner: GeometryModel, ranks: u32) -> Vec<RankProjection> {
    try_shapes_for(owner, ranks).unwrap_or_else(|error| panic!("{error}"))
}

pub fn try_shapes_for(owner: GeometryModel, ranks: u32) -> Result<Vec<RankProjection>, String> {
    use std::{
        collections::HashMap,
        sync::{Mutex, OnceLock},
    };
    static SHAPES: OnceLock<Mutex<HashMap<Profile, Vec<RankProjection>>>> = OnceLock::new();
    let profile = owner.profile(ranks);
    let mut cache = SHAPES.get_or_init(Default::default).lock().unwrap();
    if let Some(shapes) = cache.get(&profile) {
        return Ok(shapes.clone());
    }
    let shapes = build_shapes(owner, ranks)?;
    cache.insert(profile, shapes.clone());
    Ok(shapes)
}

fn build_shapes(owner: GeometryModel, ranks: u32) -> Result<Vec<RankProjection>, String> {
    let g = owner.geometry;
    let rank = owner.rank(ranks);
    rank.validate()
        .map_err(|error| format!("geometry {} TP{ranks}: {error}", g.id))?;
    let hidden = g.hidden();
    let (ga, swa) = (
        rank.attention(geometry::ga(g)),
        rank.attention(geometry::swa(g)),
    );
    let dflash = rank.attention(geometry::dflash(g));
    let q_heads = ga.q_heads;
    // A regrouped QKV row: the rank's Q heads, then its K and V heads.
    let qkv = |heads: kernel_core::rank_shape::AttentionConfig| {
        heads.q_heads * heads.qk_head + heads.kv_heads * (heads.qk_head + heads.v_head)
    };
    // Target and drafter MLP widths come from their respective configurations.
    let dense = rank.dense(geometry::dense(g)).intermediate;
    let dflash_dense = rank.split(g.dflash().intermediate_size);
    let dflash_hidden = g.dflash().hidden_size;
    let dflash_q = dflash.q_heads * dflash.qk_head;
    let dflash_kv = dflash.kv_heads * dflash.qk_head;
    let bf16 = WeightFormat::Bf16;
    let block = WeightFormat::Fp8Block128x128(kernel_core::contract::ScaleMode::ExactF32);
    let rows = WeightFormat::Fp8Rows64x128(kernel_core::contract::ScaleMode::ExactF32);
    let (bf16_c, fp8_c, fd) = (gemm::CLUSTER_BF16, gemm::CLUSTER_FP8, gemm::WHOLE_K_DECODE);
    // The decode kernel at TP2 and TP4.
    let at = |tp2, tp4| match ranks {
        2 => tp2,
        4 => tp4,
        _ => panic!("MiMo serves 2 or 4 ranks"),
    };
    let mut shapes = vec![
        (
            "o",
            hidden,
            q_heads * ga.v_head,
            bf16,
            false,
            2,
            at(bf16_c, bf16_c),
        ),
        ("head", rank.vocab_shard(), hidden, bf16, false, 4, fd),
        ("head_f32", rank.vocab_shard(), hidden, bf16, false, 4, fd),
        (
            "qkv_ga",
            qkv(ga),
            hidden,
            if ranks == 4 { block } else { rows },
            false,
            8,
            at(fp8_c, fp8_c),
        ),
        (
            "qkv_swa",
            qkv(swa),
            hidden,
            if ranks == 4 { block } else { rows },
            false,
            4,
            at(fp8_c, fp8_c),
        ),
        (
            "mlp_gate_up",
            2 * dense,
            hidden,
            block,
            true,
            2,
            at(fp8_c, fp8_c),
        ),
        ("mlp_down", hidden, dense, block, false, 4, at(fd, fp8_c)),
        (
            "dflash_q",
            dflash_q,
            dflash_hidden,
            bf16,
            false,
            2,
            at(bf16_c, fd),
        ),
        (
            "dflash_qkv",
            dflash_q + 2 * dflash_kv,
            dflash_hidden,
            bf16,
            false,
            4,
            at(bf16_c, fd),
        ),
        (
            "dflash_context_kv",
            g.dflash().num_hidden_layers * 2 * dflash_kv,
            dflash_hidden,
            bf16,
            false,
            2,
            at(bf16_c, fd),
        ),
        (
            "dflash_o",
            dflash_hidden,
            dflash_q,
            bf16,
            false,
            4,
            at(bf16_c, bf16_c),
        ),
        (
            "dflash_gate_up",
            2 * dflash_dense,
            dflash_hidden,
            bf16,
            true,
            2,
            at(bf16_c, bf16_c),
        ),
        (
            "dflash_down",
            dflash_hidden,
            dflash_dense,
            bf16,
            false,
            4,
            at(fd, bf16_c),
        ),
        (
            "dflash_fc",
            dflash_hidden,
            rank.split(g.tap_count() as u32 * g.hidden()),
            bf16,
            false,
            4,
            fd,
        ),
    ];
    if !geometry::pinned_names(g) {
        shapes.push((
            "mtp_eh",
            g.hidden(),
            2 * g.hidden() / ranks,
            bf16,
            false,
            2,
            bf16_c,
        ));
        shapes.push((
            "swa_o",
            g.hidden(),
            g.swa_q_heads() * g.swa_value_head_dim() / ranks,
            bf16,
            false,
            2,
            bf16_c,
        ));
        for projection in &mut shapes {
            super::tuning::dense(owner, ranks, projection)?;
        }
    }
    Ok(shapes)
}

/// Decode (M 1 to 32) and prefill entries of `shapes_for(owner, ranks)`,
/// named `{tag}_{name}_{m<M>|prefill}` (`mimo_tp{ranks}`), with TP4's tiles.
fn rank_entries(owner: GeometryModel, items: &mut Vec<GemmVariant>, ranks: u32) {
    let profile = owner.profile(ranks);
    let tag = profile.tag();
    for (name, n, k, weight, swiglu, segments, decode) in shapes_for(owner, ranks) {
        if ranks == 4 && !["mtp_eh", "swa_o"].contains(&name) {
            continue;
        }
        let query = |m| {
            let mut query = bf16_query(
                m,
                n,
                k,
                if name == "head_f32" {
                    DType::F32
                } else {
                    DType::Bf16
                },
            );
            query.weight = weight;
            if swiglu {
                query.epilogues.swiglu = Some(SwigluColumns::ConcatenatedHalves);
            }
            query
        };
        for m in [1, 2, 4, 8, 16, 32] {
            let item = instance(
                profile,
                &format!("{tag}_{name}_m{m}"),
                query(m),
                64,
                m.max(8) as u16,
                decode,
            );
            // The drafter's projections run no exact verify rows.
            items.push(if name.starts_with("dflash_") {
                item
            } else if name == "head" {
                bf16_head(exact_rows(item, m))
            } else {
                exact_rows(item, m)
            });
        }
        // TP4 prefill tiles: 64 x 128 for FP8, 128 x 128 for BF16.
        let tile_m = if weight == WeightFormat::Bf16 {
            128
        } else {
            64
        };
        items.push(instance(
            profile,
            &format!("{tag}_{name}_prefill"),
            query(8192),
            tile_m,
            128,
            DenseKernel::FragmentPrefill { segments },
        ));
    }
}

/// The fragment prefill entries with a medium-M stand-in
/// (`kernels_sm120::gemm::mid`): TP4's and each other rank count's.
pub fn mid_references() -> Vec<(Profile, String)> {
    geometry::MODELS
        .iter()
        .flat_map(|owner| mid_for(*owner))
        .collect()
}

fn mid_for(owner: GeometryModel) -> Vec<(Profile, String)> {
    let tag = owner.geometry.kernel_tag;
    let mut symbols = Vec::new();
    for name in [
        "qkv_ga",
        "qkv_swa",
        "o",
        "mlp_gate_up",
        "mlp_down",
        "head",
        "head_f32",
        "dflash_q",
        "dflash_qkv",
        "dflash_context_kv",
        "dflash_o",
        "dflash_gate_up",
        "dflash_down",
        "dflash_fc",
    ] {
        symbols.push((
            owner.profile(4),
            format!("chop_gemm_{tag}_{name}_prefill_v1"),
        ));
    }
    if !geometry::pinned_names(owner.geometry) {
        for name in ["mtp_eh", "swa_o"] {
            symbols.push((
                owner.profile(4),
                format!("chop_gemm_{tag}_{name}_prefill_v1"),
            ));
        }
    }
    for ranks in OTHER_RANKS {
        let profile = owner.profile(ranks);
        let tag = profile.tag();
        for (name, ..) in shapes_for(owner, ranks) {
            symbols.push((profile, format!("chop_gemm_{tag}_{name}_prefill_v1")));
        }
    }
    symbols
}

/// The K splits kernel-lab launched these fragment decode projections with,
/// at every decode bucket, before each entry declared the single split MiMo
/// serves them with (`kernels_sm120::gemm::lab_split_catalog`).
pub const LAB_SPLITS: [(&str, u8); 11] = [
    (
        kernels_sm120::const_name!("chop_gemm_", GEOMETRY.kernel_tag, "_head"),
        3,
    ),
    (
        kernels_sm120::const_name!("chop_gemm_", GEOMETRY.kernel_tag, "_head_f32"),
        3,
    ),
    (
        kernels_sm120::const_name!("chop_gemm_", GEOMETRY.kernel_tag, "_dflash_q"),
        32,
    ),
    (
        kernels_sm120::const_name!("chop_gemm_", GEOMETRY.kernel_tag, "_dflash_qkv"),
        32,
    ),
    (
        kernels_sm120::const_name!("chop_gemm_", GEOMETRY.kernel_tag, "_dflash_context_kv"),
        32,
    ),
    (
        kernels_sm120::const_name!("chop_gemm_", GEOMETRY.kernel_tag, "_dflash_fc"),
        24,
    ),
    (
        kernels_sm120::const_name!("chop_gemm_", GEOMETRY.kernel_tag, "_tp2_head"),
        2,
    ),
    (
        kernels_sm120::const_name!("chop_gemm_", GEOMETRY.kernel_tag, "_tp2_head_f32"),
        2,
    ),
    (
        kernels_sm120::const_name!("chop_gemm_", GEOMETRY.kernel_tag, "_tp2_mlp_down"),
        24,
    ),
    (
        kernels_sm120::const_name!("chop_gemm_", GEOMETRY.kernel_tag, "_tp2_dflash_fc"),
        24,
    ),
    (
        kernels_sm120::const_name!("chop_gemm_", GEOMETRY.kernel_tag, "_tp2_dflash_down"),
        24,
    ),
];

/// Projection roles whose default geometry happens to share a kernel.
#[derive(Clone, Copy, Debug)]
pub enum Role {
    AttentionOut,
    SlidingOut,
    MtpEh,
}
pub fn role_name(g: crate::geometry::Geometry, role: Role) -> &'static str {
    match role {
        Role::AttentionOut => "o",
        Role::SlidingOut | Role::MtpEh if geometry::pinned_names(g) => "o",
        Role::SlidingOut => "swa_o",
        Role::MtpEh => "mtp_eh",
    }
}
