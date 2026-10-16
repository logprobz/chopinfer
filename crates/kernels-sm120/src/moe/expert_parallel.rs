//! The FP8 block-scaled fragment expert engine and the softmax top-k
//! routing kernels Qwen3.8 Flash serves: expert-sharded MoE with replicated
//! activations, the per-rank route, permutation and shared-expert combine
//! (`cuda/moe/qwen.cuh`), and the expert GEMMs as grouped-MoE template
//! instances. Entries carry their shapes (local experts, top-k, hidden and
//! intermediate) and their cubins define the local experts as
//! `CHOP_QWEN38_LOCAL_EXPERTS` ([`cubin_defines`]). Which entries exist is
//! the serving model's catalog, registered with `crate::registry`.

use super::{FRAGMENT_PREFILL_THREADS, InputPath, Phase, Projection, SM_COUNT, Shape, Variant};
use crate::Profile;
use crate::invariance::Invariance;
use kernel_build::EntryParam;
use kernel_core::contract::{
    ActivationFormat, Argument, ArgumentList, CompileKey, CompileKnobs, DType, Epilogues,
    GemmQuery, GroupedGemm, InstantiationSpec, LaunchPlan, MoeQuery, NumericalContract,
    NumericalContractId, OpFamily, Refusal, RoutedCapacity, RowQuery, RuntimeCounts, ScaleMode,
    StaticQuery, SwigluColumns, Target, Validate, WeightFormat,
};
use kernel_core::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};

pub use super::ptr;

/// Expert ranges are padded to 16 routed rows in decode and prefill.
pub const ROUTE_BLOCK: u32 = 16;
/// The experts' weight format.
pub const WEIGHT: WeightFormat = WeightFormat::Fp8Block128x128(ScaleMode::ExactF32);

/// Routes per prefill permutation tile, and the prefix scan's thread count,
/// which bounds the live tiles.
pub const ROUTE_TILE: u32 = 256;
pub const PREFIX_THREADS: u32 = 512;
/// Decode ring budgets: two CTAs per SM, or one when two would leave a ring
/// shallower than two steps.
const TWO_CTA_SHARED: u32 = 50_176;
const ONE_CTA_SHARED: u32 = 99_000;
/// Padded activation row stride in a decode ring slot (`kFragmentRowBytes`).
const DECODE_ROW_BYTES: u32 = 128 + 16;
/// Prefill stage and table sizes, as `prefill_shared_bytes` in qwen.cuh.
const PREFILL_WEIGHT_SLOT_BYTES: u32 = 16 * 1024 + 16;
const PREFILL_ACTIVATION_BYTES: u32 = 128 * 128;
const RESERVED_SHARED: u32 = 1024 + 128;

/// The CUDA defines of a cubin of entries over `local_experts` experts
/// per rank.
pub fn cubin_defines(local_experts: u32) -> Vec<(String, String)> {
    vec![(
        "CHOP_QWEN38_LOCAL_EXPERTS".into(),
        local_experts.to_string(),
    )]
}

/// The expert tables, route slots and staging of a prefill CTA
/// (`kPrefillTableBytes` and the fixed terms of `prefill_shared_bytes`).
fn prefill_fixed_bytes(local: u32) -> u32 {
    (4 * local + 4) * 4 + 2 * 128 * 8 + 8 * 1024
}

/// The tuning of an entry: its phase's knobs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tuning {
    /// Warps and K splits of a decode entry.
    Decode { warps: u8, split: u8 },
    /// Weight and activation ring depths of a prefill entry.
    Prefill {
        weight_stages: u8,
        activation_stages: u8,
    },
}

fn decode_slot_bytes(projection: Projection, tokens_capacity: u32) -> u32 {
    let parts = if projection == Projection::GateUp {
        2
    } else {
        1
    };
    parts * 1024
        + if tokens_capacity > 1 {
            tokens_capacity * DECODE_ROW_BYTES
        } else {
            0
        }
}

fn decode_row_bytes(k: u32, tokens_capacity: u32) -> u32 {
    if tokens_capacity == 1 { k * 2 } else { 0 }
}

/// Ring depth in K64 steps: as deep as the warp's K range allows within two
/// CTAs per SM, else within one.
fn decode_depth(projection: Projection, k: u32, warps: u8, split: u8, tokens: u32) -> u8 {
    let slot = decode_slot_bytes(projection, tokens) * u32::from(warps);
    let steps = k / u32::from(split) / 64;
    let fit = |budget: u32| (budget - decode_row_bytes(k, tokens)) / slot;
    let depth = if fit(TWO_CTA_SHARED) >= 2 {
        fit(TWO_CTA_SHARED)
    } else {
        fit(ONE_CTA_SHARED)
    };
    depth.min(steps) as u8
}

fn decode_shared_bytes(v: &Variant) -> u32 {
    u32::from(v.knobs.warps)
        * u32::from(v.knobs.stages)
        * decode_slot_bytes(v.projection, v.tokens_capacity)
        + decode_row_bytes(v.query.k, v.tokens_capacity)
}

fn prefill_shared_bytes(weight_stages: u8, activation_stages: u8, local: u32) -> u32 {
    u32::from(weight_stages) * (PREFILL_WEIGHT_SLOT_BYTES + 16)
        + u32::from(activation_stages) * (PREFILL_ACTIVATION_BYTES + 16)
        + prefill_fixed_bytes(local)
}

/// The schedule of Qwen3.8's per-model engine, which the template instance
/// reproduces: the ring decode with row reloads and the persistent prefill,
/// trimming the gather and bulk-loading weights for TP4's gate/up.
fn engine_schedule(v: &Variant) -> super::template::ScheduleKnobs {
    use super::template::{PrefillKnobs, RingKnobs, ScheduleKnobs};
    let gate_up = v.projection == Projection::GateUp;
    match v.phase {
        Phase::Decode => ScheduleKnobs::Ring(RingKnobs {
            split_k: u32::from(v.knobs.split_k_capacity),
            warps: u32::from(v.knobs.warps),
            depth: u32::from(v.knobs.stages),
            early_row: false,
            reload_rows: true,
        }),
        Phase::Prefill => {
            let tp4 = gate_up && v.query.grouped.is_some_and(|g| g.experts == 128);
            ScheduleKnobs::Prefill(PrefillKnobs {
                weight_stages: u32::from(v.weight_stages),
                activation_stages: u32::from(v.knobs.stages),
                trim_gather: tp4,
                bulk_weights: tp4,
            })
        }
    }
}

/// The numerics an expert entry declares: where the route weight scales
/// the output and the SwiGLU input clamps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Numerics {
    pub route_weight: super::RouteWeight,
    pub swiglu_clamp: Option<super::SwigluClamp>,
}

/// The FP8 fragment expert entry of `profile`'s MoE block `class` (with
/// its rank `shape`) at `projection` and token capacity, with its declared
/// tuning, numerics and invariance. Refuses an entry the engine does not
/// support.
#[allow(clippy::too_many_arguments)]
pub fn variant(
    profile: Profile,
    (class, shape): (&'static str, Shape),
    projection: Projection,
    tokens_capacity: u32,
    tuning: Tuning,
    numerics: Numerics,
    invariance: Invariance,
) -> Result<Variant, Refusal> {
    let (n, k) = shape.projection(projection);
    let (swiglu, weighted) =
        super::expert_epilogue(projection, numerics.route_weight, numerics.swiglu_clamp)?;
    let (phase, knobs, weight_stages) = match tuning {
        Tuning::Decode { warps, split } => {
            let knobs = CompileKnobs {
                tile_m: ROUTE_BLOCK as u16,
                tile_n: 16,
                tile_k: 64,
                stages: decode_depth(projection, k, warps, split.max(1), tokens_capacity),
                swizzle: 0,
                warps,
                split_k_capacity: split,
            };
            (Phase::Decode, knobs, 0)
        }
        Tuning::Prefill {
            weight_stages,
            activation_stages,
        } => {
            let knobs = CompileKnobs {
                tile_m: ROUTE_BLOCK as u16,
                tile_n: 128,
                tile_k: 64,
                stages: activation_stages,
                swizzle: 0,
                warps: 8,
                split_k_capacity: 1,
            };
            (Phase::Prefill, knobs, weight_stages)
        }
    };
    let local = shape.experts;
    let rows = tokens_capacity
        .checked_mul(shape.top_k)
        .ok_or(Refusal("MoE capacity overflow"))?;
    let capacity = super::padded_capacity(rows, local, ROUTE_BLOCK)?;
    let query = GemmQuery {
        m_capacity: rows,
        n,
        k,
        weight: WEIGHT,
        activation: ActivationFormat::Bf16,
        output: DType::Bf16,
        epilogues: Epilogues {
            swiglu,
            weighted_moe_combine: weighted,
            ..Epilogues::default()
        },
        grouped: Some(GroupedGemm {
            experts: local,
            top_k: shape.top_k,
            capacity: RoutedCapacity::TotalRows(capacity),
        }),
    };
    let projection_name = match projection {
        Projection::GateUp => "gateup",
        Projection::Down => "down",
    };
    let phase_name = match phase {
        Phase::Decode => "decode",
        Phase::Prefill => "prefill",
    };
    let variant = Variant {
        profile,
        class,
        layout: super::WeightLayout::Fp8Fragment,
        projection,
        phase,
        tokens_capacity,
        path: InputPath::A16Gather,
        query,
        knobs,
        weight_stages,
        exact: false,
        route_weight: numerics.route_weight,
        swiglu_clamp: numerics.swiglu_clamp,
        engine: None,
        schedule: None,
        symbol: format!(
            "chop_moe_{}_{projection_name}_{phase_name}_fp8block_a16_t{tokens_capacity}_v1",
            super::class_tag(profile, class)
        ),
        invariance,
    };
    supported(&variant)?;
    Ok(Variant {
        engine: Some(engine_schedule(&variant)),
        ..variant
    })
}

/// The rank shape of `v`'s profile and MoE block class.
fn class_shape(v: &Variant) -> Result<Shape, Refusal> {
    v.profile
        .rank_shape()
        .and_then(|rank| Shape::of_class(rank, v.class))
        .ok_or(Refusal(
            "FP8 fragment experts need an MoE class of a model profile",
        ))
}

pub(super) fn supported(v: &Variant) -> Result<(), Refusal> {
    let query = &v.query;
    query.validate()?;
    v.knobs.validate()?;
    let grouped = query
        .grouped
        .ok_or(Refusal("MoE engine requires grouped GEMM"))?;
    let shape = class_shape(v)?;
    let local = shape.experts;
    if grouped.experts != local
        || grouped.top_k != shape.top_k
        || !matches!(grouped.capacity, RoutedCapacity::TotalRows(_))
        || query.weight != WEIGHT
        || query.activation != ActivationFormat::Bf16
        || query.output != DType::Bf16
        || query.epilogues.residual_add
        || query.epilogues.output_quantization.is_some()
        || query.epilogues.swiglu.is_some() == query.epilogues.weighted_moe_combine
        || query.epilogues.swiglu == Some(SwigluColumns::Interleaved)
        || v.path != InputPath::A16Gather
        || !query.k.is_multiple_of(128)
        || !query.n.is_multiple_of(256)
    {
        return Err(Refusal(
            "FP8 fragment experts need their class's local FP8 block experts, top-k and BF16 input",
        ));
    }
    let out_n = if v.projection == Projection::GateUp {
        query.n / 2
    } else {
        query.n
    };
    match v.phase {
        Phase::Decode => {
            let (warps, split) = (
                u32::from(v.knobs.warps),
                u32::from(v.knobs.split_k_capacity),
            );
            let strip_columns = warps / split.max(1) * 16;
            if v.tokens_capacity > 32
                || split == 0
                || !warps.is_multiple_of(split)
                || !query.k.is_multiple_of(split * 128)
                || strip_columns == 0
                || !out_n.is_multiple_of(strip_columns)
                || (v.projection == Projection::Down && split != 1)
            {
                return Err(Refusal(
                    "expert-parallel MoE decode warps or K splits are invalid",
                ));
            }
            let steps = query.k / split / 64;
            let parts = if v.projection == Projection::GateUp {
                2
            } else {
                1
            };
            let token_tiles = v.tokens_capacity.div_ceil(8);
            let ring =
                u32::from(v.knobs.stages) * decode_slot_bytes(v.projection, v.tokens_capacity);
            if v.knobs.stages == 0
                || u32::from(v.knobs.stages) > steps
                || ring < 32 * parts * token_tiles * 16
                || decode_shared_bytes(v) > ONE_CTA_SHARED
            {
                return Err(Refusal("expert-parallel MoE decode ring does not fit"));
            }
        }
        Phase::Prefill => {
            if v.knobs.split_k_capacity != 1
                || v.knobs.stages < 2
                || v.weight_stages < 2
                || prefill_shared_bytes(v.weight_stages, v.knobs.stages, local) + RESERVED_SHARED
                    > 101_376
            {
                return Err(Refusal("expert-parallel MoE prefill rings do not fit"));
            }
        }
    }
    Ok(())
}

pub(super) fn plan(v: &Variant, counts: RuntimeCounts) -> Result<LaunchPlan, Refusal> {
    supported(v)?;
    if counts.m > v.tokens_capacity
        || counts.routed_rows > v.query.m_capacity
        || counts.split_k != v.knobs.split_k_capacity
    {
        return Err(Refusal(
            "live MoE counts exceed prepared capacity or split-K differs",
        ));
    }
    let (grid, block, dynamic_shared_bytes) = match v.phase {
        Phase::Decode => {
            let out_n = if v.projection == Projection::GateUp {
                v.query.n / 2
            } else {
                v.query.n
            };
            let strip_columns = u32::from(v.knobs.warps) / u32::from(v.knobs.split_k_capacity) * 16;
            let grouped = v
                .query
                .grouped
                .ok_or(Refusal("MoE engine requires grouped GEMM"))?;
            let active = (grouped.top_k * v.tokens_capacity).min(grouped.experts);
            (
                active * (out_n / strip_columns),
                u32::from(v.knobs.warps) * 32,
                decode_shared_bytes(v),
            )
        }
        Phase::Prefill => (
            SM_COUNT,
            FRAGMENT_PREFILL_THREADS,
            prefill_shared_bytes(
                v.weight_stages,
                v.knobs.stages,
                v.query
                    .grouped
                    .ok_or(Refusal("MoE engine requires grouped GEMM"))?
                    .experts,
            ),
        ),
    };
    let plan = LaunchPlan {
        grid: [grid, 1, 1],
        block: [block, 1, 1],
        dynamic_shared_bytes,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
}

impl Variant {
    /// Offline layout of Qwen3.8 FP8 expert weights: every local expert's
    /// rows (gate rows, then up rows) in fragment-major E4M3 tiles. The F32
    /// block scales stay row-major, `[experts * n / 128, k / 128]`.
    pub fn fp8_fragment_layout(&self) -> Option<FragmentWeightLayout> {
        self.fp8_fragment_weights().then(|| FragmentWeightLayout {
            rows: (self.query.grouped.expect("grouped GEMM").experts * self.query.n) as usize,
            k: self.query.k as usize,
            format: FragmentWeightFormat::Fp8E4M3,
        })
    }
}

/// The contract a route entry implements.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RouteContract {
    Router(RowQuery),
    Moe(MoeQuery),
}

/// A routing, permutation or combine entry of a route cubin.
#[derive(Clone, Debug)]
pub struct RouteEntry {
    /// The profile whose local experts the entry serves. The router does
    /// not depend on them and is TP4's only.
    pub profile: Profile,
    /// Routed experts of the model, the router's width.
    pub experts: u32,
    /// Experts each rank serves; `experts / local_experts` ranks.
    pub local_experts: u32,
    pub top_k: u32,
    pub symbol: String,
    pub contract: RouteContract,
    /// Token capacity of a decode route entry, otherwise the prefill capacity.
    pub tokens_capacity: u32,
    pub threads: u32,
    pub params: Vec<EntryParam>,
    pub body: String,
    pub parallel: Option<super::parallel::ParallelRoute>,
    /// The router and the combine serve prefill rows: each row's route and
    /// combined output ignore the chunk split and the other rows; the
    /// combine, row-local, also gives a row its one-row bits at any row
    /// count. The permutation's counts, offsets and sorted rows depend on
    /// every route of the launch; its claims are the routed operation's.
    /// A decode route entry of an exact verify bucket (`decode_invariance`)
    /// claims one-row bits for its per-token outputs, each token's expert
    /// IDs and route weights; the route metadata it also writes (expert
    /// counts, offsets, sorted rows, active experts) depends on every route
    /// of the launch.
    pub invariance: Invariance,
}

impl RouteEntry {
    pub fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        match self.contract {
            RouteContract::Router(query) => query.numerical_contract_id(),
            RouteContract::Moe(query) => query.numerical_contract_id(),
        }
    }

    pub fn instantiation(&self) -> Result<InstantiationSpec, Refusal> {
        let query = match self.contract {
            RouteContract::Router(_) => StaticQuery::Row {
                family: "row",
                width_capacity: self.experts,
                dtype: DType::F32,
                traits_digest: self
                    .body
                    .bytes()
                    .fold(0xcbf29ce484222325_u64, |hash, byte| {
                        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
                    }),
            },
            RouteContract::Moe(query) => StaticQuery::Moe(query),
        };
        let spec = InstantiationSpec {
            family: OpFamily::Row,
            key: CompileKey {
                query,
                knobs: CompileKnobs {
                    tile_m: 1,
                    tile_n: self.threads as u16,
                    tile_k: 1,
                    stages: 1,
                    swizzle: 0,
                    warps: (self.threads / 32) as u8,
                    split_k_capacity: 1,
                },
                dense: None,
                target: Target::Sm120f,
            },
            symbol: self.symbol.clone(),
        };
        spec.validate()?;
        Ok(spec)
    }

    /// The experts each rank serves.
    pub fn local_experts(&self) -> u32 {
        self.local_experts
    }

    /// Grids derive from the prepared token capacity; the kernels read the
    /// live token count on the device.
    pub fn plan(&self, capacity: u32, live_tokens: u32) -> Result<LaunchPlan, Refusal> {
        self.instantiation()?;
        let local = self.local_experts();
        if capacity == 0 || capacity > self.tokens_capacity || live_tokens > capacity {
            return Err(Refusal("route token count exceeds capacity"));
        }
        if let Some(geometry) = self.parallel {
            return geometry.plan(live_tokens);
        }
        let tiles = (capacity * self.top_k).div_ceil(ROUTE_TILE);
        let name = self.symbol.as_str();
        let (grid, workspace_bytes) = if name.contains("_decode_route_") {
            (1, 0)
        } else if name.ends_with("_router_v1") {
            (capacity.div_ceil(self.threads / 32), 0)
        } else if name.ends_with("_permute_count_v1") || name.ends_with("_permute_scatter_v1") {
            // Tile counts and tile offsets, each [tiles, local experts] int32.
            (tiles, u64::from(tiles) * u64::from(local) * 4 * 2)
        } else if name.ends_with("_permute_prefix_v1") {
            if tiles > PREFIX_THREADS {
                return Err(Refusal("prefill routes exceed the prefix scan"));
            }
            (local, 0)
        } else if name.ends_with("_permute_scan_v1") {
            (1, 0)
        } else {
            (capacity, 0)
        };
        let plan = LaunchPlan {
            grid: [grid, 1, 1],
            block: [self.threads, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// Device buffers in entry order, then the rank for entries that filter
    /// this rank's experts.
    pub fn arguments(&self, handles: &[u64], rank: Option<u32>) -> Result<ArgumentList, Refusal> {
        let pointers = self
            .params
            .iter()
            .filter(|param| matches!(param, EntryParam::DevicePointer { .. }))
            .count();
        let takes_rank = self
            .params
            .iter()
            .any(|param| matches!(param, EntryParam::U32 { .. }));
        if handles.len() != pointers || takes_rank != rank.is_some() {
            return Err(Refusal("route argument count mismatch"));
        }
        if rank.is_some_and(|rank| rank >= self.experts / self.local_experts) {
            return Err(Refusal("rank exceeds the expert shards"));
        }
        let mut values = handles
            .iter()
            .map(|&handle| Argument::DeviceBuffer(handle))
            .collect::<Vec<_>>();
        if let Some(rank) = rank {
            values.push(Argument::U32(rank));
        }
        let args = ArgumentList(values);
        args.validate()?;
        Ok(args)
    }
}

/// The rank argument of entries that keep this rank's experts.
pub fn rank_arg() -> EntryParam {
    EntryParam::U32 {
        name: "rank".into(),
    }
}
