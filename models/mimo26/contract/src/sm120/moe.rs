//! MiMo V2.6 Flash's grouped expert entries on the MXFP4 fragment and packed
//! engines (`kernels_sm120::moe`), its sigmoid-bias route cubins
//! (`kernels_sm120::moe::sigmoid_route`) and the fused down combine, with its
//! own tuning table (`tuning/moe.json`).

use super::geometry::{self, GeometryModel};
use crate::geometry::V26_FLASH as GEOMETRY;

use kernel_build::CubinSpec;
use kernel_build::tuning::{Table, TableSource};
use kernel_core::contract::{LaunchPlan, Refusal, RuntimeCounts, WeightFormat};
use kernels_sm120::Profile;
use kernels_sm120::backend::FusedRouterEntry;
use kernels_sm120::gemm::GemmVariant;
use kernels_sm120::invariance::{Invariance, Ties};
use kernels_sm120::moe::{
    self, InputPath, Phase, Projection, Shape, Tuning, Variant, WeightLayout, sigmoid_route,
    template,
};

/// The bytes of the parallel route's ticket, which MiMo's scratch holds.
pub use kernels_sm120::moe::parallel::PARALLEL_ROUTE_TICKET_BYTES;

/// MiMo A16 decode buckets with exact entries: verify steps of up to eight
/// rows (every C1 verify step) run them so each row's experts give the bits
/// of the row alone. A bucket whose decode entry already has bucket 1's
/// reduction order (down at every bucket) has none.
pub const EXACT_BUCKETS: [u32; 3] = [2, 4, 8];

/// MiMo A16 decode buckets: up to `serving::DECODE_EXPERT_ROWS` rows. Wider
/// decode and verify buckets route as prefill does and run the prefill
/// entries.
pub const DECODE_BUCKETS: [u32; 5] = [1, 2, 4, 8, 16];

pub const DECODE_ROUTE_CUBIN: &str =
    kernels_sm120::const_name!("chop_moe_", GEOMETRY.kernel_tag, "_decode_route_b16");
pub const DECODE_ROUTE_SYMBOL: &str =
    kernels_sm120::const_name!("chop_moe_", GEOMETRY.kernel_tag, "_decode_route_b16_v1");
pub const FUSED_ROUTER_PREFIX: &str =
    kernels_sm120::const_name!("chop_moe_", GEOMETRY.kernel_tag, "_router_route");
pub const DOWN_COMBINE_PREFIX: &str = kernels_sm120::const_name!(
    "chop_moe_",
    GEOMETRY.kernel_tag,
    "_down_combine_decode_mxfp4_a16"
);

/// MiMo's expert shape on one TP4 rank: all 256 experts, each with a
/// quarter of the 2048 intermediate columns.
pub const SHAPE: Shape = match crate::sm120::config::PROFILE.rank_shape() {
    Some(rank) => Shape::of(rank, crate::sm120::config::MOE),
    None => panic!("MiMo has per-rank shapes"),
};

/// MiMo's expert shape on one rank of `profile`: TP4's `SHAPE`, or the
/// 2048-wide expert intermediate split over 2 ranks.
pub fn shape(profile: Profile) -> Option<Shape> {
    let owner = geometry::owner(profile)?;
    Some(Shape::of(
        profile.rank_shape()?,
        geometry::moe(owner.geometry),
    ))
}

/// The model's grouped-template schedule table (`tuning/schedules.json`).
pub fn schedules() -> &'static Table {
    &resolved().1
}

pub fn tuning() -> Tuning {
    let values: serde_json::Value =
        serde_json::from_str(include_str!("../../tuning/moe.json")).unwrap();
    Tuning::parse(&values[GEOMETRY.id].to_string()).expect("default MoE tuning")
}

fn resolved() -> &'static (Vec<Variant>, Table) {
    static RESOLVED: std::sync::OnceLock<(Vec<Variant>, Table)> = std::sync::OnceLock::new();
    RESOLVED.get_or_init(|| {
        let mut all = Vec::new();
        for owner in geometry::MODELS {
            all.extend(catalog_for(*owner).unwrap_or_else(|e| panic!("{e}")));
        }
        let table = super::tuning::schedules(&all).unwrap_or_else(|e| panic!("{e}"));
        (
            template::scheduled(all, &table).unwrap_or_else(|e| panic!("{e}")),
            table,
        )
    })
}

pub fn catalog_for(owner: GeometryModel) -> Result<Vec<Variant>, String> {
    catalog_with_tables(owner, super::tuning::moe_tables())
}

fn catalog_with_tables(
    owner: GeometryModel,
    tables: &std::collections::BTreeMap<String, Tuning>,
) -> Result<Vec<Variant>, String> {
    let g = owner.geometry;
    let (variants, source) = kernel_build::tuning::build_nearest(
        g.id,
        "MoE",
        tables,
        |id| {
            let source = crate::geometry::GEOMETRIES
                .iter()
                .find(|g| g.id == id)
                .expect("tuned geometry is listed");
            u64::from(
                source.hidden().abs_diff(g.hidden())
                    + source
                        .expert_intermediate()
                        .abs_diff(g.expert_intermediate())
                    + source.experts().abs_diff(g.experts()),
            )
        },
        moe::default_tuning,
        |table| raw_for(owner, table),
    )?;
    let status = match source {
        TableSource::Own => return Ok(variants),
        TableSource::Inherited(id) => format!("inherited {id} MoE table"),
        TableSource::Default => "engine default MoE table".into(),
    };
    for variant in &variants {
        super::tuning::record(owner, variant.symbol.clone(), status.clone());
    }
    Ok(variants)
}

/// Every MiMo profile's entries: TP4's MXFP4 experts on both input paths
/// (with the MXFP8-quantized gate/up of the A8 path and the exact decode
/// buckets of the A16 path), then TP2's A16 experts. A16 decode buckets
/// without an exact entry sum in bucket 1's order, so exact verify rows run
/// them.
pub fn catalog() -> Vec<Variant> {
    resolved().0.clone()
}

fn raw_for(owner: GeometryModel, table: &Tuning) -> Result<Vec<Variant>, String> {
    let weight = WeightFormat::Mxfp4;
    let layout = WeightLayout::Mxfp4Fragment;
    let mut all = Vec::new();
    let profile = owner.profile(4);
    for (class, _) in moe::classes(profile) {
        for path in [InputPath::A16Gather, InputPath::A8Pregather] {
            for projection in [Projection::GateUp, Projection::Down] {
                let at = |phase, tokens| {
                    moe::variant(
                        profile, class, layout, projection, phase, tokens, weight, path, table,
                    ).map_err(|e| format!("geometry {} class {class} projection {projection:?} phase {phase:?} bucket {tokens} TP4 path {path:?}: {e}", owner.geometry.id))
                };
                let quantize = projection == Projection::GateUp && path == InputPath::A8Pregather;
                let buckets: &[u32] = if path == InputPath::A16Gather {
                    &DECODE_BUCKETS
                } else {
                    &moe::DECODE_BUCKETS
                };
                for &tokens in buckets {
                    let base = at(Phase::Decode, tokens)?;
                    if quantize {
                        all.push(moe::quantized(&base));
                    }
                    all.push(base);
                }
                if path == InputPath::A16Gather {
                    for tokens in EXACT_BUCKETS {
                        all.extend(moe::exact_variant(
                            profile, class, projection, tokens, table,
                        ).map_err(|e| format!("geometry {} class {class} projection {projection:?} phase Decode bucket {tokens} TP{}: {e}", owner.geometry.id, profile.ranks().unwrap()))?);
                    }
                }
                let base = at(Phase::Prefill, 8192)?;
                if quantize {
                    all.push(moe::quantized(&base));
                }
                all.push(base);
            }
        }
    }
    // MiMo at 2 ranks: the MXFP4 BF16-input experts the bundle runs.
    for profile in owner.profiles().into_iter().skip(1) {
        for (class, _) in moe::classes(profile) {
            for projection in [Projection::GateUp, Projection::Down] {
                for (phase, tokens) in DECODE_BUCKETS
                    .map(|tokens| (Phase::Decode, tokens))
                    .into_iter()
                    .chain([(Phase::Prefill, 8192)])
                {
                    all.push(moe::variant(
                        profile,
                        class,
                        layout,
                        projection,
                        phase,
                        tokens,
                        weight,
                        InputPath::A16Gather,
                        table,
                    ).map_err(|e| format!("geometry {} class {class} projection {projection:?} phase {phase:?} bucket {tokens} TP2: {e}", owner.geometry.id))?);
                }
                for tokens in EXACT_BUCKETS {
                    all.extend(moe::exact_variant(
                        profile, class, projection, tokens, table,
                    ).map_err(|e| format!("geometry {} class {class} projection {projection:?} phase Decode bucket {tokens} TP2: {e}", owner.geometry.id))?);
                }
            }
        }
    }
    let exact = all
        .iter()
        .filter(|v| v.exact)
        .map(|v| (v.profile, v.class, v.projection, v.tokens_capacity))
        .collect::<Vec<_>>();
    for v in &mut all {
        if v.path == InputPath::A16Gather
            && v.phase == Phase::Decode
            && !v.exact
            && (v.tokens_capacity == 1 || EXACT_BUCKETS.contains(&v.tokens_capacity))
            && !exact.contains(&(v.profile, v.class, v.projection, v.tokens_capacity))
        {
            v.invariance = Invariance::one_row(v.tokens_capacity);
        }
    }
    Ok(all)
}

/// The decode buckets that run the fused router and route; wider buckets
/// route as prefill does (`serving::prefill_route`).
pub fn prefer_fused_router(tokens_capacity: u32) -> bool {
    matches!(tokens_capacity, 1 | 2 | 4 | 8)
}

const ROUTER_GEMV_PREFIX: &str =
    kernels_sm120::const_name!("chop_gemm_", GEOMETRY.kernel_tag, "_router_m");

#[cfg(test)]
static ROUTER_VARIANT_BUILDS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// The router GEMV entries of the GEMM catalog, built once per process.
/// The catalog is a pure function of compile-time tables and lists every
/// topology's entries, and the router is rank-invariant (one entry per token
/// bucket, owned by TP4's profile), so one copy serves every rank count.
/// Eager launches plan on every step, and rebuilding the catalog on every
/// call cost 0.3 ms per MoE layer, rank and step.
fn router_variants() -> &'static [GemmVariant] {
    static VARIANTS: std::sync::OnceLock<Vec<GemmVariant>> = std::sync::OnceLock::new();
    VARIANTS.get_or_init(|| {
        #[cfg(test)]
        ROUTER_VARIANT_BUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        super::gemm::catalog()
            .into_iter()
            .filter(|item| item.symbol.starts_with(ROUTER_GEMV_PREFIX))
            .collect()
    })
}

/// The router GEMV's final CTA can select top-8 and build stable expert
/// metadata without a second launch for the small decode buckets.
pub fn fused_router_plan(tokens_capacity: u32, live_tokens: u32) -> Result<LaunchPlan, Refusal> {
    // The route tail's token tile holds at most 32 tokens and traps above it.
    if tokens_capacity > sigmoid_route::ROUTE_MAX_TOKENS
        || live_tokens > sigmoid_route::ROUTE_MAX_TOKENS
    {
        return Err(Refusal(
            "fused MiMo router tail holds at most 32 tokens (ROUTE_MAX_TOKENS)",
        ));
    }
    if ![1, 2, 4, 8, 16, 32].contains(&tokens_capacity) || live_tokens > tokens_capacity {
        return Err(Refusal("fused MiMo router exceeds prepared token bucket"));
    }
    let symbol = format!("{ROUTER_GEMV_PREFIX}{tokens_capacity}_v1");
    let variant = router_variants()
        .iter()
        .find(|item| item.symbol == symbol)
        .ok_or(Refusal("fused MiMo router GEMV entry is absent"))?;
    kernels_sm120::gemm::plan(
        &variant.key,
        RuntimeCounts {
            m: tokens_capacity,
            batch: tokens_capacity,
            context: 0,
            routed_rows: 0,
            split_k: kernels_sm120::gemm::choose_split_k(&variant.key)?,
        },
    )
}

/// The fused router and route entries of the buckets that prefer them
/// (`prefer_fused_router`), whose symbols start with `prefix` (the
/// geometry's `FUSED_ROUTER_PREFIX`): each with the router GEMV entry of
/// its bucket, which plans its launch, and its declaration
/// (`fused_router_declarations`).
pub fn fused_router_entries(prefix: &str) -> Result<Vec<FusedRouterEntry>, Refusal> {
    let declared = fused_router_declarations();
    [1_u32, 2, 4, 8]
        .into_iter()
        .filter(|&tokens| prefer_fused_router(tokens))
        .map(|tokens| {
            let symbol = format!("{prefix}_t{tokens}_v1");
            let gemm = router_variants()
                .iter()
                .find(|item| item.symbol == format!("{ROUTER_GEMV_PREFIX}{tokens}_v1"))
                .cloned()
                .ok_or(Refusal("fused MiMo router GEMV entry is absent"))?;
            let invariance = declared
                .iter()
                .find(|(declared, _)| *declared == symbol)
                .map(|&(_, invariance)| invariance)
                .ok_or(Refusal("fused MiMo router entry is undeclared"))?;
            Ok(FusedRouterEntry {
                symbol,
                gemm,
                invariance,
            })
        })
        .collect()
}

/// The fused router cubin's entries.
pub fn fused_router_cubin_spec() -> CubinSpec {
    sigmoid_route::fused_router_cubin_spec(FUSED_ROUTER_PREFIX)
}

/// The fused router entries' declarations: a token bucket of at most eight
/// rows gives every row the router logits, top-8 experts (equal scores keep
/// the lower expert, `route.cuh`) and weights it has alone, which exact
/// verify rows need. These claims cover those per-token outputs; the route
/// metadata it also writes depends on every route of the launch.
pub fn fused_router_declarations() -> Vec<(String, Invariance)> {
    geometry::MODELS
        .iter()
        .flat_map(|owner| fused_router_declarations_for(*owner))
        .collect()
}

fn fused_router_declarations_for(owner: GeometryModel) -> Vec<(String, Invariance)> {
    let prefix = format!("chop_moe_{}_router_route", owner.geometry.kernel_tag);
    [1_u32, 2, 4, 8, 16, 32]
        .into_iter()
        .map(|tokens| {
            let invariance = if tokens <= 8 {
                Invariance::one_row(tokens).with_ties(Ties::LowerIndex)
            } else {
                Invariance::NONE
            };
            (format!("{prefix}_t{tokens}_v1"), invariance)
        })
        .collect()
}

/// One token saves a combine launch. Larger buckets reuse expert weights in
/// the grouped down kernel and should retain the separate fixed-order combine.
pub fn prefer_fused_down_combine(tokens_capacity: u32) -> bool {
    tokens_capacity == 1
}

/// Policy: four warps cover the fused down-combine routes.
const DOWN_COMBINE_THREADS: u32 = 128;

pub fn fused_down_combine_cubin_spec() -> CubinSpec {
    let prefix = DOWN_COMBINE_PREFIX;
    CubinSpec {
        name: prefix.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."),
        include_dirs: vec!["crates/kernels-sm120/cuda".into(), "crates/kernels-sm120/cuda/common".into(), "crates/kernels-sm120/cuda/moe".into()],
        headers: vec!["#include \"models/mimo26/bundle/cuda/moe/mimo26_fragments.cuh\"".into()],
        defines: vec![
            ("CHOP_MIMO_HIDDEN".into(), SHAPE.hidden.to_string()),
            ("CHOP_MIMO_EXPERT_INTERMEDIATE".into(), SHAPE.intermediate.to_string()),
            ("CHOP_MIMO_EXPERTS".into(), SHAPE.experts.to_string()),
            ("CHOP_MIMO_TOP_K".into(), SHAPE.top_k.to_string()),
        ],
        entries: DECODE_BUCKETS.iter().map(|&tokens| kernel_build::EntrySpec::new(
            format!("{prefix}_t{tokens}_v1"),
            vec![
                moe::ptr("activation", "const uint16_t"),
                moe::ptr("weight_bytes", "const uint8_t"),
                moe::ptr("weight_scales", "const uint8_t"),
                moe::ptr("topk_ids", "const int32_t"),
                moe::ptr("topk_weights", "const float"),
                moe::ptr("output", "uint16_t"),
                moe::ptr("live_tokens", "const int32_t"),
            ],
            format!("chop::moe::run_fragment_down_combine<{tokens}>(activation, weight_bytes, weight_scales, topk_ids, topk_weights, output, live_tokens);"),
            DOWN_COMBINE_THREADS,
            1,
        )).collect(),
    }
}

pub fn decode_route_cubin_spec() -> CubinSpec {
    sigmoid_route::decode_route_cubin_spec(DECODE_ROUTE_CUBIN, DECODE_ROUTE_SYMBOL)
}

/// The route cubins TP4's profile holds (they serve every rank count): the
/// decode route, the fused router and the fused down combine.
pub(super) fn route_cubins(profile: Option<Profile>) -> Vec<CubinSpec> {
    geometry::MODELS
        .iter()
        .filter(|owner| profile.is_none_or(|p| p == owner.profile(4)))
        .flat_map(|owner| route_cubins_for(*owner))
        .collect()
}

pub fn route_cubins_for(owner: GeometryModel) -> Vec<CubinSpec> {
    let g = owner.geometry;
    assert_eq!(
        (g.hidden(), g.experts(), g.top_k(), g.expert_intermediate()),
        (4096, 256, 8, 2048),
        "{}: specialized route domain",
        g.id
    );
    let tag = g.kernel_tag;
    let decode = format!("chop_moe_{tag}_decode_route_b16");
    let mut down = fused_down_combine_cubin_spec();
    let prefix = format!("chop_moe_{tag}_down_combine_decode_mxfp4_a16");
    down.name = prefix.clone();
    for entry in &mut down.entries {
        entry.symbol = entry.symbol.replace(DOWN_COMBINE_PREFIX, &prefix);
    }
    vec![
        sigmoid_route::decode_route_cubin_spec(&decode, &format!("{decode}_v1")),
        sigmoid_route::fused_router_cubin_spec(&format!("chop_moe_{tag}_router_route")),
        down,
    ]
}

#[cfg(test)]
mod measured_refusal_tests {
    use super::*;

    #[test]
    fn own_measured_table_refusal_does_not_inherit() {
        let mut values: serde_json::Value =
            serde_json::from_str(include_str!("../../tuning/moe.json")).unwrap();
        values[GEOMETRY.id]["decode"]["1"]["warps"] = serde_json::json!(3);
        let tables = std::collections::BTreeMap::from([(
            GEOMETRY.id.to_owned(),
            Tuning::parse(&values[GEOMETRY.id].to_string()).unwrap(),
        )]);
        let error = catalog_with_tables(geometry::MODELS[0], &tables).unwrap_err();
        assert!(
            error.contains("geometry v26-flash measured MoE table refused"),
            "{error}"
        );
        assert!(
            error.contains("class moe projection GateUp phase Decode bucket 1 TP4"),
            "{error}"
        );
    }
}
