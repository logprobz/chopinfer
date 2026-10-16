//! Qwen3.8 Flash's grouped expert and route entries on the shared FP8
//! fragment expert engine and route kernels (`kernels_sm120::moe::expert_parallel`):
//! softmax top-10 routing over 512 experts, 128 FP8 block-scaled experts
//! per rank with replicated activations, and the shared expert's
//! sigmoid-gated combine. TP2 and TP4 entries hold 256 and 128
//! local experts (`local_experts`), which their cubins define as
//! `CHOP_QWEN38_LOCAL_EXPERTS`.

use kernel_build::tuning::Table;
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{MoeQuery, Refusal, RouterKind, RowQuery};
use kernels_sm120::Profile;
use kernels_sm120::invariance::{Invariance, Ties};
use kernels_sm120::moe::expert_parallel::{
    self as engine, Numerics, RouteContract, Tuning, ptr, rank_arg,
};
use kernels_sm120::moe::parallel::{ParallelRoute, RouteScoring};
use kernels_sm120::moe::{self, Phase, Projection, RouteWeight, Variant, template};

use serde::Deserialize;
use std::collections::BTreeMap;

pub use kernels_sm120::moe::expert_parallel::{
    PREFIX_THREADS, ROUTE_BLOCK, ROUTE_TILE, RouteEntry, WEIGHT,
};

pub const GLOBAL_EXPERTS: u32 = crate::sm120::config::MOE.experts;
/// Local experts of a TP4 rank; `local_experts` gives any Qwen profile's.
pub const LOCAL_EXPERTS: u32 = crate::sm120::config::RankShape { ranks: 4 }.local_experts();
pub const TOP_K: u32 = crate::sm120::config::MOE.top_k;
pub const HIDDEN: u32 = crate::sm120::config::HIDDEN;
pub const INTERMEDIATE: u32 = crate::sm120::config::MOE.intermediate;
pub const PREFILL_CAPACITY: u32 = crate::contract::PREFILL_TOKENS;
pub const DECODE_BUCKETS: [u32; 6] = crate::contract::DECODE_ENTRY_BUCKETS;
pub const ROUTE_CUBIN: &str = "chop_moe_qwen38_route";

/// Qwen3.8's experts weight routes after the down projection and do not
/// clamp SwiGLU inputs.
const NUMERICS: Numerics = Numerics {
    route_weight: RouteWeight::AfterDown,
    swiglu_clamp: None,
};

/// The local experts of a Qwen3.8 profile's rank, derived from the model's
/// configuration: 512 over its rank count.
pub fn local_experts(profile: Profile) -> Result<u32, Refusal> {
    super::geometry::shape(profile)
        .map(|shape| shape.local_experts())
        .ok_or(Refusal("not a Qwen3.8 MoE profile"))
}

/// The CUDA defines of a Qwen3.8 profile's MoE cubins: the derived local
/// experts. Other profiles have none.
pub fn cubin_defines(profile: Profile) -> Vec<(String, String)> {
    local_experts(profile)
        .map(engine::cubin_defines)
        .unwrap_or_default()
}

#[derive(Deserialize)]
struct Tables {
    compute_capability: u32,
    sm_count: u32,
    decode: BTreeMap<String, DecodeTuning>,
    prefill: BTreeMap<String, PrefillTuning>,
}

#[derive(Deserialize)]
struct DecodeTuning {
    /// Warps and K splits of the gate/up and down entries.
    gate_up: (u8, u8),
    down: (u8, u8),
}

#[derive(Deserialize)]
struct PrefillTuning {
    weight_stages: u8,
    activation_stages: u8,
}

fn tables(g: crate::geometry::Geometry) -> Tables {
    let tables: Tables = serde_json::from_str(g.tuning.expect("measured geometry").moe)
        .expect("Qwen3.8 MoE tuning table is valid JSON");
    assert_eq!(
        (tables.compute_capability, tables.sm_count),
        (120, moe::SM_COUNT)
    );
    tables
}

/// The tuning of a bucket's entry.
fn tuning(
    tables: &Tables,
    projection: Projection,
    phase: Phase,
    tokens: u32,
) -> Result<Tuning, Refusal> {
    match phase {
        Phase::Decode => {
            let tuned = tables
                .decode
                .get(&tokens.to_string())
                .ok_or(Refusal("Qwen3.8 MoE decode bucket is not tuned"))?;
            let (warps, split) = match projection {
                Projection::GateUp => tuned.gate_up,
                Projection::Down => tuned.down,
            };
            Ok(Tuning::Decode { warps, split })
        }
        Phase::Prefill => {
            let tuned = tables
                .prefill
                .get(&tokens.to_string())
                .ok_or(Refusal("Qwen3.8 MoE prefill bucket is not tuned"))?;
            Ok(Tuning::Prefill {
                weight_stages: tuned.weight_stages,
                activation_stages: tuned.activation_stages,
            })
        }
    }
}

/// The claims of the decode expert and route entries of a token bucket.
/// The buckets an exact K3 verify step takes (`EXACT_VERIFY_ROWS` rows and
/// below) give each token its bits through the bucket-1 entries: a token's
/// routed rows sit in their experts' 16-row tiles, each row of a tile
/// accumulates its own MMA column in one K order, the gate/up K splits (ten
/// at buckets 1 to 4, `tuning/flash/moe.json`) add in split order, and the route
/// selects each token's experts on a warp of its own. Larger buckets claim
/// nothing.
fn decode_invariance(tokens_capacity: u32) -> Invariance {
    if tokens_capacity <= crate::sm120::config::EXACT_VERIFY_ROWS {
        Invariance::one_row(tokens_capacity)
    } else {
        Invariance::NONE
    }
}

/// The model's grouped-template schedule table (`tuning/flash/schedules.json`):
/// its FP8 experts' rows and its packed ring rows.
pub fn schedules() -> &'static Table {
    &scheduled_catalog().0
}

pub fn tuning_status() -> &'static BTreeMap<String, String> {
    &scheduled_catalog().1
}

fn scheduled_catalog() -> &'static (Table, BTreeMap<String, String>) {
    static TABLE: std::sync::OnceLock<(Table, BTreeMap<String, String>)> =
        std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let measured = crate::geometry::GEOMETRIES
            .iter()
            .filter_map(|g| {
                g.tuning.map(|data| {
                    (
                        g.id.to_owned(),
                        template::parse_table(data.schedules).expect("geometry schedule table"),
                    )
                })
            })
            .collect::<Vec<_>>();
        let (raw, mut statuses) = raw_catalog();
        let table = template::resolve_table(
            &raw,
            &measured,
            |variant| super::geometry::owner(variant.profile).map(|owner| owner.geometry.id),
            |variant, status| {
                let g = super::geometry::owner(variant.profile).unwrap().geometry;
                if !super::tuning::measured(g) {
                    statuses.entry(variant.symbol.clone()).and_modify(|s| {
                        s.push_str("; ");
                        s.push_str(&status);
                    });
                }
            },
        )
        .unwrap_or_else(|e| panic!("{e}"));
        (table, statuses)
    })
}

/// The expert entries of every listed geometry and rank count.
pub fn catalog() -> Vec<Variant> {
    template::scheduled(raw_catalog().0, schedules()).unwrap_or_else(|e| panic!("{e}"))
}

// Model-chosen conservative defaults. The engine validates them for every bucket.
fn defaults(phase: Phase) -> Tuning {
    match phase {
        Phase::Decode => Tuning::Decode { warps: 4, split: 1 },
        Phase::Prefill => Tuning::Prefill {
            weight_stages: 2,
            activation_stages: 2,
        },
    }
}

fn choose_tuning(
    g: crate::geometry::Geometry,
    profile: Profile,
    class: (&'static str, moe::Shape),
    projection: Projection,
    phase: Phase,
    tokens: u32,
    sources: &[crate::geometry::Geometry],
) -> Result<super::tuning::Decision<Tuning>, Refusal> {
    let buckets = if phase == Phase::Decode && tokens <= super::config::EXACT_VERIFY_ROWS {
        DECODE_BUCKETS
            .into_iter()
            .filter(|&n| n <= super::config::EXACT_VERIFY_ROWS)
            .collect::<Vec<_>>()
    } else {
        vec![tokens]
    };
    let accepts = |knobs, capacity| {
        engine::variant(
            profile,
            class,
            projection,
            capacity,
            knobs,
            NUMERICS,
            if phase == Phase::Decode {
                decode_invariance(capacity)
            } else {
                Invariance::PREFILL
            },
        )
        .is_ok()
    };
    for &candidate in sources {
        let table = tables(candidate);
        let choices = buckets
            .iter()
            .map(|&n| tuning(&table, projection, phase, n))
            .collect::<Result<Vec<_>, _>>()?;
        let same_split = choices.iter().all(|choice| match (choices[0], *choice) {
            (Tuning::Decode { split: a, .. }, Tuning::Decode { split: b, .. }) => a == b,
            _ => true,
        });
        if same_split
            && choices
                .iter()
                .zip(&buckets)
                .all(|(&choice, &n)| accepts(choice, n))
        {
            return Ok(super::tuning::Decision {
                value: tuning(&table, projection, phase, tokens)?,
                status: (!super::tuning::measured(g)).then(|| format!("untuned: inherited {} row moe.{phase:?}.{projection:?} buckets {buckets:?}", candidate.id)),
            });
        }
        if candidate.id == g.id {
            return Err(Refusal(
                "measured MoE tuning is illegal or changes exact-bucket K splits",
            ));
        }
    }
    let value = defaults(phase);
    if !buckets.iter().all(|&n| accepts(value, n)) {
        return Err(Refusal("model MoE default is illegal for the bucket group"));
    }
    Ok(super::tuning::Decision {
        value,
        status: Some(format!(
            "untuned: model default moe.{phase:?}.{projection:?} buckets {buckets:?}"
        )),
    })
}

fn raw_catalog() -> (Vec<Variant>, BTreeMap<String, String>) {
    let mut all = Vec::new();
    let mut statuses = BTreeMap::new();
    for profile in crate::sm120::PROFILES {
        let g = super::geometry::owner(profile).unwrap().geometry;
        for class in moe::classes(profile) {
            for projection in [Projection::GateUp, Projection::Down] {
                for (phase, tokens) in DECODE_BUCKETS
                    .map(|tokens| (Phase::Decode, tokens))
                    .into_iter()
                    .chain([(Phase::Prefill, PREFILL_CAPACITY)])
                {
                    let choice = choose_tuning(
                        g,
                        profile,
                        class,
                        projection,
                        phase,
                        tokens,
                        &super::tuning::nearest(g),
                    )
                    .expect("legal MoE tuning");
                    let variant = engine::variant(
                        profile,
                        class,
                        projection,
                        tokens,
                        choice.value,
                        NUMERICS,
                        if phase == Phase::Decode {
                            decode_invariance(tokens)
                        } else {
                            Invariance::PREFILL
                        },
                    )
                    .expect("catalog MoE variant");
                    if let Some(status) = choice.status {
                        statuses.insert(variant.symbol.clone(), status);
                    }
                    all.push(variant);
                }
            }
        }
    }
    (all, statuses)
}

/// Packed NVFP4 and MXFP8 decode instances at TP4's expert shape, every
/// decode bucket and both projections. The served NVFP4 entries are these
/// template instances under their own symbols (`precision::expert_offer`);
/// MXFP8 has no served entry and is compared with the packed engine at the
/// same shape.
pub fn packed_lab_instances() -> Vec<template::PackedInstance> {
    template::packed_lab_instances(schedules(), crate::sm120::config::PROFILE, packed_shape())
}

/// The TP4 expert shape of the packed ring rows.
fn packed_shape() -> moe::Shape {
    crate::sm120::config::PROFILE
        .rank_shape()
        .map(|rank| moe::Shape::of(rank, crate::sm120::config::MOE))
        .expect("Qwen3.8 has experts")
}

/// Every key Qwen3.8's schedule table may hold: its template entries' and
/// the packed ring rows of its packed decode buckets.
pub fn schedule_keys() -> Vec<kernel_build::tuning::Key> {
    let mut keys = template::schedule_keys(&catalog());
    keys.extend(template::packed_keys(packed_shape()));
    keys
}

fn entry_name(profile: Profile, operation: &str) -> String {
    format!("chop_moe_{}_{operation}_v1", profile.tag())
}

/// A route entry of `profile` with Qwen3.8's experts.
#[allow(clippy::too_many_arguments)]
fn route(
    profile: Profile,
    symbol: String,
    contract: RouteContract,
    tokens_capacity: u32,
    threads: u32,
    params: Vec<kernel_build::EntryParam>,
    body: String,
    invariance: Invariance,
) -> RouteEntry {
    RouteEntry {
        profile,
        experts: super::geometry::owner(profile).unwrap().geometry.experts(),
        local_experts: local_experts(profile).expect("Qwen3.8 route profile"),
        top_k: super::geometry::owner(profile).unwrap().geometry.top_k(),
        symbol,
        contract,
        tokens_capacity,
        threads,
        params,
        body,
        parallel: None,
        invariance,
    }
}

/// Router, decode route, prefill permutation and combine entries: TP4's,
/// then TP2's, which hold every entry that depends on the local
/// expert count (all but the router).
pub fn route_catalog() -> Vec<RouteEntry> {
    super::geometry::MODELS
        .iter()
        .flat_map(route_catalog_for)
        .collect()
}

fn route_catalog_for(owner: &super::geometry::GeometryModel) -> Vec<RouteEntry> {
    let profile = owner.model.at(4);
    let g = owner.geometry;
    let router = RowQuery::Router {
        experts: g.experts(),
        top_k: g.top_k(),
        kind: RouterKind::SoftmaxTopK { renormalize: true },
    };
    let mut entries = vec![route(
        profile,
        entry_name(profile, "router"),
        RouteContract::Router(router),
        PREFILL_CAPACITY,
        256,
        vec![
            ptr("logits", "const float"),
            ptr("ids", "int32_t"),
            ptr("weights", "float"),
            ptr("live_tokens", "const int32_t"),
        ],
        "chop::moe::qwen::router(logits, ids, weights, live_tokens);".into(),
        Invariance::PREFILL.with_ties(Ties::LowerIndex),
    )];
    for profile in owner.model.profiles() {
        entries.extend(local_route_entries(profile));
    }
    entries
}

/// The route entries of one Qwen3.8 profile that depend on its local
/// experts.
fn local_route_entries(profile: Profile) -> Vec<RouteEntry> {
    let g = super::geometry::owner(profile).unwrap().geometry;
    let hidden = g.hidden();
    let local_count = local_experts(profile).expect("Qwen3.8 route profile");
    let local = |capacity| {
        RouteContract::Moe(MoeQuery::LocalRoutes {
            experts: g.experts(),
            local_experts: local_count,
            top_k: g.top_k(),
            capacity,
        })
    };
    let entry_name = |operation: &str| entry_name(profile, operation);
    let mut entries = Vec::new();
    for tokens in DECODE_BUCKETS {
        entries.push(route(
            profile,
            entry_name(&format!("decode_route_t{tokens}")),
            local(tokens),
            tokens,
            (tokens * 32).max(local_count),
            vec![
                ptr("logits", "const float"),
                ptr("ids", "int32_t"),
                ptr("weights", "float"),
                ptr("counts", "int32_t"),
                ptr("offsets", "int32_t"),
                ptr("sorted_rows", "int32_t"),
                ptr("num_padded_rows", "int32_t"),
                ptr("live_tokens", "const int32_t"),
                ptr("active_experts", "int32_t"),
                rank_arg(),
            ],
            format!(
                "chop::moe::qwen::route_decode<{tokens}>(logits, ids, weights, counts, offsets, sorted_rows, num_padded_rows, live_tokens, active_experts, rank);"
            ),
            decode_invariance(tokens),
        ));
    }
    entries.extend([
        route(
            profile,
            entry_name("permute_count"),
            local(PREFILL_CAPACITY),
            PREFILL_CAPACITY,
            ROUTE_TILE,
            vec![
                ptr("ids", "const int32_t"),
                ptr("tile_counts", "int32_t"),
                ptr("live_tokens", "const int32_t"),
                rank_arg(),
            ],
            "chop::moe::qwen::permute_count(ids, tile_counts, live_tokens, rank);".into(),
            Invariance::PREFILL.of_routed_operation(),
        ),
        route(
            profile,
            entry_name("permute_prefix"),
            local(PREFILL_CAPACITY),
            PREFILL_CAPACITY,
            PREFIX_THREADS,
            vec![
                ptr("tile_counts", "const int32_t"),
                ptr("counts", "int32_t"),
                ptr("tile_offsets", "int32_t"),
                ptr("live_tokens", "const int32_t"),
            ],
            "chop::moe::qwen::permute_prefix(tile_counts, counts, tile_offsets, live_tokens);"
                .into(),
            Invariance::PREFILL.of_routed_operation(),
        ),
        route(
            profile,
            entry_name("permute_scan"),
            local(PREFILL_CAPACITY),
            PREFILL_CAPACITY,
            local_count,
            vec![
                ptr("counts", "const int32_t"),
                ptr("offsets", "int32_t"),
                ptr("sorted_rows", "int32_t"),
                ptr("num_padded_rows", "int32_t"),
                ptr("active_experts", "int32_t"),
            ],
            format!(
                "chop::moe::qwen::permute_scan<{ROUTE_BLOCK}>(counts, offsets, sorted_rows, num_padded_rows, active_experts);"
            ),
            Invariance::PREFILL.of_routed_operation(),
        ),
        route(
            profile,
            entry_name("permute_scatter"),
            local(PREFILL_CAPACITY),
            PREFILL_CAPACITY,
            ROUTE_TILE,
            vec![
                ptr("ids", "const int32_t"),
                ptr("tile_offsets", "const int32_t"),
                ptr("offsets", "const int32_t"),
                ptr("sorted_rows", "int32_t"),
                ptr("live_tokens", "const int32_t"),
                rank_arg(),
            ],
            "chop::moe::qwen::permute_scatter(ids, tile_offsets, offsets, sorted_rows, live_tokens, rank);"
                .into(),
            Invariance::PREFILL.of_routed_operation(),
        ),
        // The combine keeps this rank's routes, so it depends on the local
        // experts too. A CTA combines one row from that row's routes and
        // inputs alone.
        route(
            profile,
            entry_name("combine"),
            RouteContract::Moe(MoeQuery::SharedExpertGateCombine {
                width: hidden,
                capacity: PREFILL_CAPACITY,
            }),
            PREFILL_CAPACITY,
            hidden / 8,
            vec![
                ptr("routed", "const uint16_t"),
                ptr("ids", "const int32_t"),
                ptr("shared_out", "const uint16_t"),
                ptr("x", "const uint16_t"),
                ptr("gate_weight", "const uint16_t"),
                ptr("output", "uint16_t"),
                ptr("live_tokens", "const int32_t"),
                rank_arg(),
            ],
            format!(
                "chop::moe::qwen::combine<{hidden}>(routed, ids, shared_out, x, gate_weight, output, live_tokens, rank);"
            ),
            Invariance::any_rows(PREFILL_CAPACITY),
        ),
    ]);
    entries
}

/// TP4's route entry of `operation`, e.g. `permute_count`.
pub fn route_entry(operation: &str) -> Result<RouteEntry, Refusal> {
    route_entry_for(crate::sm120::config::PROFILE, operation)
}

/// The route entry of `operation` for a Qwen3.8 profile. The router is
/// TP4's at every rank count.
pub fn route_entry_for(profile: Profile, operation: &str) -> Result<RouteEntry, Refusal> {
    let owner = super::geometry::owner(profile).ok_or(Refusal("not a Qwen profile"))?;
    let profile = if operation == "router" {
        owner.model.at(4)
    } else {
        profile
    };
    let symbol = entry_name(profile, operation);
    if super::geometry::owner(profile).is_none() {
        return Err(Refusal("not a Qwen3.8 route profile"));
    }
    route_catalog()
        .into_iter()
        .chain(parallel_route_entries(profile))
        .find(|entry| entry.symbol == symbol)
        .ok_or(Refusal("unknown Qwen3.8 route entry"))
}

/// TP4's route cubin (`ROUTE_CUBIN`), with the router.
pub fn route_cubin_spec() -> CubinSpec {
    route_cubin_spec_for(crate::sm120::config::PROFILE).expect("TP4 route cubin")
}

/// The route cubin of a Qwen3.8 profile, `chop_moe_{tag}_route`, or `None`
/// for other profiles.
pub fn route_cubin_spec_for(profile: Profile) -> Option<CubinSpec> {
    let g = super::geometry::owner(profile)?.geometry;
    Some(CubinSpec {
        name: format!("chop_moe_{}_route", profile.tag()),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."),
        include_dirs: vec![
            "crates/kernels-sm120/cuda".into(),
            "crates/kernels-sm120/cuda/common".into(),
            "crates/kernels-sm120/cuda/moe".into(),
            "models/qwen38/bundle/cuda".into(),
        ],
        headers: vec!["#include \"moe/qwen.cuh\"".into()],
        defines: {
            let mut defines = cubin_defines(profile);
            defines.extend([
                ("CHOP_QWEN38_GLOBAL_EXPERTS".into(), g.experts().to_string()),
                ("CHOP_QWEN38_TOP_K".into(), g.top_k().to_string()),
            ]);
            defines
        },
        entries: route_catalog()
            .into_iter()
            .filter(|entry| entry.profile == profile)
            .map(|entry| EntrySpec::new(entry.symbol, entry.params, entry.body, entry.threads, 1))
            .collect(),
    })
}

/// C16 measured no gain; only the qualified C32 shape selects the parallel
/// decode route.
pub fn use_parallel_route(capacity: u32) -> bool {
    capacity == 32
}

/// The parallel decode route entries of a profile, in a cubin of their own
/// so the qualified route and expert cubins keep their source closure.
fn parallel_route_entries(profile: Profile) -> Vec<RouteEntry> {
    let g = super::geometry::owner(profile).unwrap().geometry;
    let local = local_experts(profile).expect("Qwen route profile");
    local_route_entries(profile)
        .into_iter()
        .filter(|entry| {
            matches!(
                entry.contract,
                RouteContract::Moe(MoeQuery::LocalRoutes { .. })
            ) && use_parallel_route(entry.tokens_capacity)
        })
        .map(|mut entry| {
            let tokens = entry.tokens_capacity;
            entry.symbol = entry_name(profile, &format!("decode_route_parallel_t{tokens}"));
            let geometry = ParallelRoute {
                experts: g.experts(),
                top_k: g.top_k(),
                local_experts: local,
                capacity: tokens,
                padding: 16,
                groups: 1,
                take_groups: 1,
                scoring: RouteScoring::Softmax,
            };
            entry.threads = geometry.plan(tokens).unwrap().block[0];
            entry.parallel = Some(geometry);
            entry
                .params
                .insert(entry.params.len() - 1, ptr("ticket", "uint32_t"));
            entry.body = geometry.body("rank", "nullptr", "1.f").unwrap();
            entry
        })
        .collect()
}

/// Parallel route entries across the two supported expert shards.
pub fn parallel_route_catalog() -> Vec<RouteEntry> {
    crate::sm120::PROFILES
        .into_iter()
        .flat_map(parallel_route_entries)
        .collect()
}

/// Parallel entries use a dedicated zero-initialized four-byte ticket.
/// Original route and expert GEMM cubins keep their source closure.
pub fn parallel_route_cubin_spec_for(profile: Profile) -> Option<CubinSpec> {
    super::geometry::owner(profile)?;
    Some(CubinSpec {
        name: format!("chop_moe_{}_route_parallel", profile.tag()),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."),
        include_dirs: vec![
            "crates/kernels-sm120/cuda".into(),
            "crates/kernels-sm120/cuda/common".into(),
            "crates/kernels-sm120/cuda/moe".into(),
            "models/qwen38/bundle/cuda".into(),
        ],
        headers: vec!["#include \"moe/route_parallel.cuh\"".into()],
        defines: Vec::new(),
        entries: parallel_route_entries(profile)
            .into_iter()
            .map(|entry| EntrySpec::new(entry.symbol, entry.params, entry.body, entry.threads, 1))
            .collect(),
    })
}

/// The route and parallel route entries.
pub(super) fn routes() -> Vec<RouteEntry> {
    route_catalog()
        .into_iter()
        .chain(parallel_route_catalog())
        .collect()
}

/// The route and parallel route cubins of each Qwen3.8 profile, or of
/// `profile` alone.
pub(super) fn route_cubins(profile: Option<Profile>) -> Vec<CubinSpec> {
    crate::sm120::PROFILES
        .into_iter()
        .filter(|&owner| profile.is_none_or(|p| p == owner))
        .flat_map(|owner| {
            route_cubin_spec_for(owner)
                .into_iter()
                .chain(parallel_route_cubin_spec_for(owner))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::{NumericalContract, RuntimeCounts, Validate};
    use kernels_sm120::moe::SM_COUNT;

    /// Shared memory of an FP8 fragment entry besides its dynamic bytes
    /// (`RESERVED_SHARED` in the engine).
    const RESERVED_SHARED: u32 = 1024 + 128;

    #[test]
    fn exact_buckets_fall_back_together_when_one_measured_bucket_refuses() {
        let mut source = crate::geometry::FLASH;
        source.tuning = Some(crate::geometry::TuningData {
            moe: r#"{"compute_capability":120,"sm_count":188,"decode":{"1":{"gate_up":[10,10],"down":[8,1]},"2":{"gate_up":[10,10],"down":[8,1]},"4":{"gate_up":[10,9],"down":[8,1]}},"prefill":{}}"#,
            ..source.tuning.unwrap()
        });
        let g = crate::geometry::Geometry {
            id: "unmeasured",
            tuning: None,
            ..source
        };
        let profile = super::super::config::PROFILE;
        let class = moe::classes(profile)[0];
        let table = tables(source);
        assert!(
            engine::variant(
                profile,
                class,
                Projection::GateUp,
                1,
                tuning(&table, Projection::GateUp, Phase::Decode, 1).unwrap(),
                NUMERICS,
                decode_invariance(1)
            )
            .is_ok()
        );
        for tokens in [1, 2, 4] {
            let choice = choose_tuning(
                g,
                profile,
                class,
                Projection::GateUp,
                Phase::Decode,
                tokens,
                &[source],
            )
            .unwrap();
            assert_eq!(choice.value, Tuning::Decode { warps: 4, split: 1 });
            assert!(choice.status.unwrap().contains("model default"));
        }
    }

    fn find(projection: Projection, phase: Phase, tokens: u32) -> Variant {
        catalog()
            .into_iter()
            .find(|v| {
                v.fp8_fragment_weights()
                    && v.projection == projection
                    && v.phase == phase
                    && v.tokens_capacity == tokens
            })
            .unwrap()
    }

    fn counts(v: &Variant) -> RuntimeCounts {
        RuntimeCounts {
            m: v.tokens_capacity,
            batch: 0,
            context: 0,
            routed_rows: v.query.m_capacity,
            split_k: v.knobs.split_k_capacity,
        }
    }

    #[test]
    fn qwen_catalog_covers_every_bucket_and_matches_the_q4_contract() {
        let qwen = catalog();
        assert!(qwen.iter().all(Variant::fp8_fragment_weights));
        assert_eq!(
            qwen.len(),
            crate::sm120::PROFILES.len() * 2 * (DECODE_BUCKETS.len() + 1)
        );
        for v in &qwen {
            let local = local_experts(v.profile).unwrap();
            let g = super::super::geometry::owner(v.profile).unwrap().geometry;
            assert!(v.instantiation().validate().is_ok());
            assert_eq!(v.query.weight, WEIGHT);
            let grouped = v.query.grouped.unwrap();
            assert_eq!((grouped.experts, grouped.top_k), (local, g.top_k()));
            let expected = if v.projection == Projection::GateUp {
                (2 * g.expert_intermediate(), g.hidden())
            } else {
                (g.hidden(), g.expert_intermediate())
            };
            assert_eq!((v.query.n, v.query.k), expected);
            assert!(
                v.symbol
                    .starts_with(&format!("chop_moe_{}_", v.profile.tag()))
            );
            assert_eq!(v.reduction_symbol(), None);
            let plan = v.plan(counts(v)).unwrap();
            assert!(plan.pdl);
            assert_eq!(plan.workspace_bytes, 0);
            assert!(plan.dynamic_shared_bytes + RESERVED_SHARED <= 101_376);
            let layout = v.fp8_fragment_layout().unwrap();
            assert_eq!(layout.rows, local as usize * v.query.n as usize);
        }
        assert_eq!(
            find(Projection::GateUp, Phase::Decode, 1)
                .query
                .numerical_contract_id()
                .unwrap()
                .as_str(),
            "gemm.w-fp8-block128-f32.a-bf16.acc-f32.v1"
        );
    }

    #[test]
    fn decode_plans_size_rings_and_grids_from_capacity() {
        let gate = find(Projection::GateUp, Phase::Decode, 1);
        let plan = gate.engine_plan(counts(&gate)).unwrap();
        // Ten warps split K ten ways on one 16-column strip: 40 strips per
        // expert for up to ten active experts; each warp's ring holds two K64
        // steps of gate and up, then the 5 KB token row.
        assert_eq!(plan.grid, [10 * 40, 1, 1]);
        assert_eq!(plan.block, [320, 1, 1]);
        assert_eq!(plan.dynamic_shared_bytes, 10 * 2 * 2048 + 5120);
        let down = find(Projection::Down, Phase::Decode, 1);
        let plan = down.engine_plan(counts(&down)).unwrap();
        assert_eq!(plan.grid, [10 * 20, 1, 1]);
        assert_eq!(plan.dynamic_shared_bytes, 8 * 5 * 1024 + 1280);
        let wide = find(Projection::GateUp, Phase::Decode, 32);
        assert_eq!(
            wide.engine_plan(counts(&wide)).unwrap().grid,
            [128 * 40, 1, 1]
        );
        assert!(
            wide.engine_plan(RuntimeCounts {
                m: 33,
                ..counts(&wide)
            })
            .is_err()
        );
        let prefill = find(Projection::Down, Phase::Prefill, PREFILL_CAPACITY);
        let plan = prefill.engine_plan(counts(&prefill)).unwrap();
        assert_eq!((plan.grid, plan.block), ([SM_COUNT, 1, 1], [384, 1, 1]));
    }

    #[test]
    fn route_entries_carry_the_q4_contracts_and_capacity_plans() {
        let entries = route_catalog();
        assert_eq!(
            entries.len(),
            super::super::geometry::MODELS.len()
                + crate::sm120::PROFILES.len() * (DECODE_BUCKETS.len() + 5)
        );
        for entry in &entries {
            entry.instantiation().unwrap();
        }
        for profile in crate::sm120::PROFILES {
            let g = super::super::geometry::owner(profile).unwrap().geometry;
            let local = local_experts(profile).unwrap();
            let spec = route_cubin_spec_for(profile).unwrap();
            assert_eq!(
                spec.defines,
                vec![
                    ("CHOP_QWEN38_LOCAL_EXPERTS".into(), local.to_string()),
                    ("CHOP_QWEN38_GLOBAL_EXPERTS".into(), g.experts().to_string()),
                    ("CHOP_QWEN38_TOP_K".into(), g.top_k().to_string()),
                ]
            );
            let count = route_entry_for(profile, "permute_count").unwrap();
            assert_eq!(
                count.plan(PREFILL_CAPACITY, 1).unwrap().workspace_bytes,
                u64::from((PREFILL_CAPACITY * g.top_k()).div_ceil(ROUTE_TILE) * local * 8)
            );
            let combine = route_entry_for(profile, "combine").unwrap();
            assert_eq!(
                combine.plan(512, 300).unwrap().block,
                [g.hidden() / 8, 1, 1]
            );
        }
        let router = route_entry("router").unwrap();
        assert_eq!(router.symbol, "chop_moe_qwen38_router_v1");
        assert_eq!(
            router.numerical_contract_id().unwrap().as_str(),
            "row.router-softmax-topk.v1"
        );
        assert_eq!(router.plan(8192, 8000).unwrap().grid, [1024, 1, 1]);
        let route = route_entry("decode_route_t32").unwrap();
        assert_eq!(
            route.numerical_contract_id().unwrap().as_str(),
            "row.moe-local-expert-routes.v1"
        );
        assert_eq!(route.plan(32, 17).unwrap().block, [1024, 1, 1]);
        assert!(route.plan(64, 1).is_err());
        let count = route_entry("permute_count").unwrap();
        let plan = count.plan(8192, 8192).unwrap();
        assert_eq!(plan.grid, [320, 1, 1]);
        assert_eq!(plan.workspace_bytes, 320 * 128 * 8);
        assert!(count.arguments(&[1, 2, 3], Some(3)).is_ok());
        assert!(count.arguments(&[1, 2, 3], Some(4)).is_err());
        assert!(count.arguments(&[1, 2, 3], None).is_err());
        let combine = route_entry("combine").unwrap();
        assert_eq!(
            combine.numerical_contract_id().unwrap().as_str(),
            "row.moe-shared-expert-sigmoid-combine.v1"
        );
        assert_eq!(combine.plan(512, 300).unwrap().block, [320, 1, 1]);
        assert_eq!(
            route_cubin_spec().entries.len(),
            1 + DECODE_BUCKETS.len() + 5
        );
        assert_eq!(route_cubin_spec().name, ROUTE_CUBIN);
        assert_eq!(
            route_cubin_spec().defines,
            vec![
                ("CHOP_QWEN38_LOCAL_EXPERTS".into(), "128".into()),
                ("CHOP_QWEN38_GLOBAL_EXPERTS".into(), "512".into()),
                ("CHOP_QWEN38_TOP_K".into(), "10".into()),
            ]
        );
        // TP2: the local routes of 256 experts, no router.
        let (profile, local, ranks) = (crate::sm120::config::MODEL.at(2), 256, 2);
        let spec = route_cubin_spec_for(profile).unwrap();
        assert_eq!(spec.entries.len(), DECODE_BUCKETS.len() + 5);
        assert_eq!(
            spec.defines,
            vec![
                ("CHOP_QWEN38_LOCAL_EXPERTS".into(), local.to_string()),
                ("CHOP_QWEN38_GLOBAL_EXPERTS".into(), "512".into()),
                ("CHOP_QWEN38_TOP_K".into(), "10".into()),
            ]
        );
        let tag = profile.tag();
        let count = route_entry_for(profile, "permute_count").unwrap();
        assert_eq!(count.symbol, format!("chop_moe_{tag}_permute_count_v1"));
        assert_eq!(
            count.plan(8192, 8192).unwrap().workspace_bytes,
            320 * local * 8
        );
        assert!(count.arguments(&[1, 2, 3], Some(ranks - 1)).is_ok());
        assert!(count.arguments(&[1, 2, 3], Some(ranks)).is_err());
        let prefix = route_entry_for(profile, "permute_prefix").unwrap();
        assert_eq!(prefix.plan(8192, 1).unwrap().grid, [local as u32, 1, 1]);
        let scan = route_entry_for(profile, "permute_scan").unwrap();
        assert_eq!(scan.plan(8192, 1).unwrap().block, [local as u32, 1, 1]);
        let route = route_entry_for(profile, "decode_route_t1").unwrap();
        assert_eq!(
            route.plan(1, 1).unwrap().block,
            [local.max(32) as u32, 1, 1]
        );
        assert_eq!(
            route_entry_for(profile, "router").unwrap().symbol,
            "chop_moe_qwen38_router_v1"
        );
    }

    /// The decode experts and routes of the exact K3 verify buckets claim
    /// one-row bits at every profile, larger buckets nothing; the combine
    /// is row-local and the permutation keeps the routed operation's scope.
    #[test]
    fn decode_buckets_of_exact_verify_rows_claim_one_row_bits() {
        use kernels_sm120::invariance::{Chunk, Scope};
        let exact = |tokens: u32| {
            if tokens <= crate::sm120::config::EXACT_VERIFY_ROWS {
                Invariance::one_row(tokens)
            } else {
                Invariance::NONE
            }
        };
        let mut claimed = 0;
        for v in catalog() {
            let expected = match v.phase {
                Phase::Decode => exact(v.tokens_capacity),
                Phase::Prefill => Invariance::PREFILL,
            };
            assert_eq!(v.invariance, expected, "{}", v.symbol);
            claimed += usize::from(v.phase == Phase::Decode && v.invariance != Invariance::NONE);
        }
        // Gate/up and down at each exact verify bucket of every profile.
        assert_eq!(claimed, crate::sm120::PROFILES.len() * 2 * 3);
        for profile in crate::sm120::PROFILES {
            for tokens in DECODE_BUCKETS {
                let route = route_entry_for(profile, &format!("decode_route_t{tokens}")).unwrap();
                assert_eq!(route.invariance, exact(tokens), "{}", route.symbol);
            }
            let combine = route_entry_for(profile, "combine").unwrap().invariance;
            assert_eq!(combine, Invariance::any_rows(PREFILL_CAPACITY));
            assert_eq!(combine.chunk, Chunk::Any);
            for operation in [
                "permute_count",
                "permute_prefix",
                "permute_scan",
                "permute_scatter",
            ] {
                let entry = route_entry_for(profile, operation).unwrap();
                assert_eq!(entry.invariance.scope, Scope::RoutedOperation);
            }
        }
        for entry in parallel_route_catalog() {
            assert_eq!(entry.invariance, Invariance::NONE, "{}", entry.symbol);
        }
    }

    #[test]
    fn parallel_routes_preserve_the_contract_and_add_one_ticket() {
        for profile in crate::sm120::PROFILES {
            for entry in parallel_route_entries(profile) {
                let rows = entry.tokens_capacity;
                assert_eq!(
                    entry.symbol,
                    format!(
                        "chop_moe_{}_decode_route_parallel_t{rows}_v1",
                        profile.tag()
                    ),
                );
                let plan = entry.plan(rows, rows - 1).unwrap();
                assert_eq!(plan.grid, [rows, 1, 1]);
                assert_eq!(
                    plan.block[0],
                    (rows
                        * super::super::geometry::owner(profile)
                            .unwrap()
                            .geometry
                            .top_k())
                    .next_multiple_of(32)
                    .max(entry.local_experts())
                );
                assert_eq!(entry.arguments(&[1; 10], Some(0)).unwrap().0.len(), 11);
                assert!(entry.arguments(&[1; 9], Some(0)).is_err());
                assert!(entry.plan(rows + 1, rows).is_err());
                entry.instantiation().unwrap();
            }
        }
    }
}
