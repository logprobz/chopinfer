//! Grouped SM120 expert GEMM instantiations and capacity-based launches.

mod catalog;
mod construction;
pub use construction::{
    GlobalScaleLayout, GroupedRequest, PackedContract, ScaleBindings, cubin_spec, instantiate,
};
pub mod mxfp4_fragment;
mod packed;
pub use packed::{
    PACKED_MAX_WARPS, PACKED_RING_TILE_K, PACKED_STAGED_TILE_K, PackedRing, PackedStaged,
    packed_ring_knobs,
};
pub mod chained;
pub mod expert_parallel;
pub mod grouped_bf16;
pub mod native_grouped;
pub mod parallel;
pub mod sigmoid_route;
pub mod template;

pub use catalog::{
    Tuning, catalog, default_tuning, exact_variant, interleaved, own_catalog, quantized, variant,
};
pub use mxfp4_fragment::*;
pub use sigmoid_route::*;

use crate::arithmetic::{self, DeclaredCubin};
use crate::invariance::Invariance;
use crate::{Catalogs, Profile, RankShape};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    ActivationFormat, Argument, ArgumentList, CompileKey, CompileKnobs, DType, Epilogues,
    GemmQuery, GroupedGemm, InstantiationSpec, LaunchPlan, OpFamily, OutputQuantization, Refusal,
    RoutedCapacity, RuntimeCounts, ScaleMode, StaticQuery, SwigluColumns, Target, TensorMap,
    TensorMapElement, TensorMapInterleave, TensorMapL2Promotion, TensorMapOobFill,
    TensorMapSwizzle, Validate, WeightFormat,
};
use kernel_core::rank_shape::{Block, MoeConfig};
use kernel_core::transform::{Mxfp4MoeFragmentLayout, ScaleTileLayout};
use std::collections::BTreeMap;
use std::path::PathBuf;

pub const SM_COUNT: u32 = 188;
/// Threads of a split-K reduction CTA, one output column each.
const REDUCE_THREADS: u32 = 128;
/// Decode token buckets.
pub const DECODE_BUCKETS: [u32; 6] = [1, 2, 4, 8, 16, 32];
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Phase {
    Decode,
    Prefill,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Projection {
    GateUp,
    Down,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum InputPath {
    A16Gather,
    A8Pregather,
}

/// Weight and scale storage selected by the catalog, independent of profile and expert count.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum WeightLayout {
    Packed,
    Mxfp4Fragment,
    Fp8Fragment,
}

/// Where an expert's route weight scales its output, which decides output
/// bits. Each entry declares it; no projection implies it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum RouteWeight {
    /// The down projection's output, in its weighted combine epilogue.
    AfterDown,
    /// The SwiGLU intermediate, before the down projection. No grouped
    /// engine implements it yet.
    BeforeDown,
}

impl RouteWeight {
    /// Whether `projection`'s epilogue applies the route weight.
    pub fn applied_by(self, projection: Projection) -> Result<bool, Refusal> {
        match self {
            RouteWeight::AfterDown => Ok(projection == Projection::Down),
            RouteWeight::BeforeDown => Err(Refusal(
                "grouped engines apply route weights after the down projection",
            )),
        }
    }
}

/// Clamps of the SwiGLU inputs: the gate at most `gate_max`, the up half
/// within `[-up_limit, up_limit]`. No grouped engine implements them yet.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SwigluClamp {
    pub gate_max: f32,
    pub up_limit: f32,
}

/// The query epilogues of an expert projection with the declared route
/// weight point and SwiGLU clamps: (SwiGLU columns, weighted combine).
fn expert_epilogue(
    projection: Projection,
    route_weight: RouteWeight,
    swiglu_clamp: Option<SwigluClamp>,
) -> Result<(Option<SwigluColumns>, bool), Refusal> {
    if swiglu_clamp.is_some() {
        return Err(Refusal("grouped engines do not clamp SwiGLU inputs"));
    }
    let swiglu = (projection == Projection::GateUp).then_some(SwigluColumns::ConcatenatedHalves);
    Ok((swiglu, route_weight.applied_by(projection)?))
}

/// Model dimensions declared together; no dimension has an implicit default.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Shape {
    pub top_k: u32,
    pub hidden: u32,
    pub experts: u32,
    pub intermediate: u32,
}

impl Shape {
    /// One rank's expert shape for the MoE block class `moe` of `rank`'s
    /// model: the rank's local experts and their intermediate columns.
    pub const fn of(rank: RankShape, moe: MoeConfig) -> Shape {
        let experts = rank.experts(moe);
        Shape {
            top_k: experts.top_k,
            hidden: rank.config().hidden,
            experts: experts.experts,
            intermediate: experts.intermediate,
        }
    }

    /// The MoE block classes of `rank`'s model with one rank's shape each,
    /// in declaration order.
    pub fn classes(rank: RankShape) -> Vec<(&'static str, Shape)> {
        rank.config()
            .blocks
            .iter()
            .filter_map(|class| match class.block {
                Block::Moe(moe) => Some((class.name, Shape::of(rank, moe))),
                Block::Dense(_) => None,
            })
            .collect()
    }

    /// [`Shape::of`] the MoE block class named `class`; `None` where
    /// `rank`'s model has no such MoE class.
    pub fn of_class(rank: RankShape, class: &str) -> Option<Shape> {
        match rank.config().block(class)?.block {
            Block::Moe(moe) => Some(Shape::of(rank, moe)),
            Block::Dense(_) => None,
        }
    }

    /// (N, K) of `projection`.
    pub fn projection(self, projection: Projection) -> (u32, u32) {
        match projection {
            Projection::GateUp => (2 * self.intermediate, self.hidden),
            Projection::Down => (self.hidden, self.intermediate),
        }
    }
}

/// The standard profile's one synthetic expert class: 128 experts (top 8)
/// of intermediate 384 at hidden 4096.
const STD_CLASS: (&str, Shape) = (
    "moe",
    Shape {
        top_k: 8,
        hidden: 4096,
        experts: 128,
        intermediate: 384,
    },
);

/// The MoE block classes of `profile` with one rank's expert shape each, in
/// declaration order: its model's MoE classes, or the standard profile's
/// synthetic class. Catalogs generate entries per class.
pub fn classes(profile: Profile) -> Vec<(&'static str, Shape)> {
    match profile.rank_shape() {
        Some(rank) => Shape::classes(rank),
        None => vec![STD_CLASS],
    }
}

/// The symbol tag of `class`'s entries at `profile`: the profile's tag,
/// then the class name where the model has more than one MoE class.
fn class_tag(profile: Profile, class: &str) -> String {
    if classes(profile).len() > 1 {
        format!("{}_{class}", profile.tag())
    } else {
        profile.tag().to_string()
    }
}

#[derive(Clone, Debug)]
pub struct Variant {
    pub profile: Profile,
    /// The MoE block class of the profile's model whose experts the entry
    /// runs ([`classes`]).
    pub class: &'static str,
    pub layout: WeightLayout,
    pub projection: Projection,
    pub phase: Phase,
    pub tokens_capacity: u32,
    pub path: InputPath,
    pub query: GemmQuery,
    pub knobs: CompileKnobs,
    /// Weight ring depth of MiMo fragment prefill entries, otherwise zero.
    /// Their activation ring depth is `knobs.stages`.
    pub weight_stages: u8,
    /// A decode-bucket entry in the one-token entry's reduction order (a
    /// model's exact buckets): bucket 1's K splits and ring schedule, so
    /// every token's output bits are those of the token alone.
    pub exact: bool,
    /// Where the route weight scales the expert output.
    pub route_weight: RouteWeight,
    /// The gate/up projection's SwiGLU input clamps, if any.
    pub swiglu_clamp: Option<SwigluClamp>,
    /// The schedule of the per-model engine a template entry replaced (the
    /// frozen kernel-lab reference), set by the model's definition; `None`
    /// for entries on the packed engine.
    pub engine: Option<template::ScheduleKnobs>,
    /// The template schedule the entry is served with: its row of the
    /// model's schedule table (`template::scheduled`); `None` for entries on
    /// the packed engine.
    pub schedule: Option<template::ScheduleKnobs>,
    pub symbol: String,
    /// Served prefill entries are chunk invariant (every routed row sums K
    /// in one order whatever rows share its expert); MiMo's one-row and
    /// exact decode buckets are one-row exact.
    pub invariance: Invariance,
}

impl Variant {
    fn operation(&self) -> GroupedOperation<&str> {
        GroupedOperation {
            layout: self.layout,
            projection: self.projection,
            phase: self.phase,
            tokens_capacity: self.tokens_capacity,
            path: self.path,
            query: self.query,
            knobs: self.knobs,
            weight_stages: self.weight_stages,
            packed_contract: None,
            symbol: &self.symbol,
        }
    }
    /// Refuses a route weight point or SwiGLU clamps the grouped engines do
    /// not implement, query epilogues that disagree with them, or
    /// interleaved gate/up columns on an engine that reads concatenated
    /// halves. Planning, arguments and code generation check this before
    /// choosing an engine.
    pub fn check_numerics(&self) -> Result<(), Refusal> {
        let (swiglu, weighted) =
            expert_epilogue(self.projection, self.route_weight, self.swiglu_clamp)?;
        if self.query.epilogues.swiglu.is_some() != swiglu.is_some()
            || self.query.epilogues.weighted_moe_combine != weighted
        {
            return Err(Refusal(
                "expert query epilogues differ from the declared route weight point",
            ));
        }
        // Only the packed engine reads interleaved columns; the template
        // schedules and fragment layouts read the up half at column + N/2.
        if self.query.epilogues.swiglu == Some(SwigluColumns::Interleaved)
            && (self.engine.is_some() || self.layout != WeightLayout::Packed)
        {
            return Err(Refusal(
                "only the packed grouped engine reads interleaved gate/up columns",
            ));
        }
        Ok(())
    }

    /// The counts of a launch over `tokens` tokens and `routed_rows`
    /// routed rows, at the variant's K split capacity.
    pub fn counts(&self, tokens: u32, routed_rows: u32) -> RuntimeCounts {
        RuntimeCounts {
            m: tokens,
            batch: 1,
            context: 1,
            routed_rows,
            split_k: self.knobs.split_k_capacity,
        }
    }

    /// The dynamic shared memory a launch takes at the variant's token and
    /// routed-row capacity: what its symbol is configured for at load.
    pub fn capacity_shared_bytes(&self) -> Result<u32, Refusal> {
        Ok(self
            .plan(self.counts(self.tokens_capacity, self.query.m_capacity))?
            .dynamic_shared_bytes)
    }

    /// The served entry's launch plan: the template's at its schedule-table
    /// row for entries the template serves, else the engine's.
    pub fn plan(&self, counts: RuntimeCounts) -> Result<LaunchPlan, Refusal> {
        self.check_numerics()?;
        if template::covers(self) {
            template::plan(self, counts)
        } else {
            self.engine_plan(counts)
        }
    }

    /// The per-model or packed engine's launch plan at the variant's knobs.
    pub fn engine_plan(&self, counts: RuntimeCounts) -> Result<LaunchPlan, Refusal> {
        self.check_numerics()?;
        if self.fp8_fragment_weights() {
            expert_parallel::plan(self, counts)
        } else {
            self.operation().plan(counts)
        }
    }

    pub fn reduction_plan(&self, counts: RuntimeCounts) -> Result<Option<LaunchPlan>, Refusal> {
        self.check_numerics()?;
        if self.fp8_fragment_weights() {
            self.plan(counts)?;
            Ok(None)
        } else {
            self.operation().reduction_plan(counts)
        }
    }
    pub fn fp8_fragment_weights(&self) -> bool {
        self.operation().fp8_fragment_weights()
    }
    pub fn fragment_weights(&self) -> bool {
        self.operation().fragment_weights()
    }
    pub fn fragment_decode(&self) -> bool {
        self.operation().fragment_decode()
    }
    pub fn fragment_prefill(&self) -> bool {
        self.operation().fragment_prefill()
    }
    pub fn packed_ring(&self) -> bool {
        self.operation().packed_ring()
    }
    /// Whether the build can compile the entry ([`declared_cubin_specs`]),
    /// or why not: numerics no engine implements, no grouped geometry, a
    /// decode symbol naming no bucket for its cubin, code the template
    /// cannot build ([`template::try_implementation`]), such as an engine
    /// and schedule that disagree or a scheduled entry without a rank
    /// shape, or, where the template has no instance, weights the packed
    /// engine does not run. The backend serves no entry it refuses.
    pub fn buildable(&self) -> Result<(), Refusal> {
        self.check_numerics()?;
        self.query
            .grouped
            .ok_or(Refusal("MoE requires grouped geometry"))?;
        if self.phase == Phase::Decode
            && self
                .symbol
                .trim_end_matches("_v1")
                .rsplit_once("_t")
                .is_none()
        {
            return Err(Refusal("a decode entry's symbol names no bucket"));
        }
        if template::try_implementation(self)?.is_none() {
            try_packed_body(&self.operation())?;
        }
        Ok(())
    }
    pub fn instantiation(&self) -> InstantiationSpec {
        self.operation().instantiation()
    }
    pub fn weight_tensor_map(&self, base_buffer: u64) -> Result<TensorMap, Refusal> {
        self.check_numerics()?;
        self.operation().weight_tensor_map(base_buffer)
    }
    pub fn weight_fragment_layout(&self) -> Option<Mxfp4MoeFragmentLayout> {
        self.operation().weight_fragment_layout()
    }
    pub fn weight_scale_layout(&self) -> Option<ScaleTileLayout> {
        self.operation().weight_scale_layout()
    }
    pub fn reduction_symbol(&self) -> Option<String> {
        self.operation().reduction_symbol()
    }
    pub fn arguments(
        &self,
        handles: [u64; 15],
        weight_map: TensorMap,
    ) -> Result<ArgumentList, Refusal> {
        self.check_numerics()?;
        self.operation().arguments(handles, weight_map)
    }
    pub fn reduction_arguments(&self, handles: [u64; 15]) -> Result<Option<ArgumentList>, Refusal> {
        self.check_numerics()?;
        self.operation().reduction_arguments(handles)
    }
}

/// A retained grouped kernel binding with no model-profile requirement.
#[derive(Clone, Debug)]
pub struct GroupedOperation<S = String> {
    pub layout: WeightLayout,
    pub projection: Projection,
    pub phase: Phase,
    pub tokens_capacity: u32,
    pub path: InputPath,
    pub query: GemmQuery,
    pub knobs: CompileKnobs,
    /// Weight ring depth of MiMo fragment prefill entries, otherwise zero.
    /// Their activation ring depth is `knobs.stages`.
    pub weight_stages: u8,
    /// Explicit packed-engine semantics. Legacy catalog entries retain their
    /// existing template defaults and tensor-map geometry.
    pub packed_contract: Option<PackedContract>,
    pub symbol: S,
}

/// Worst-case stable expert padding, including experts with no routed rows.
pub fn padded_capacity(routed_capacity: u32, experts: u32, block: u32) -> Result<u32, Refusal> {
    if routed_capacity == 0 || experts == 0 || block == 0 || !block.is_power_of_two() {
        return Err(Refusal("invalid MoE capacity or row block"));
    }
    routed_capacity
        .checked_add(
            experts
                .min(routed_capacity)
                .checked_mul(block - 1)
                .ok_or(Refusal("MoE capacity overflow"))?,
        )
        .ok_or(Refusal("MoE capacity overflow"))
}

/// Byte offset of one scale in the packed 16-row by 8-block MMA tile.
pub fn packed_scale_offset(row: usize, k_block: usize, k_blocks: usize) -> usize {
    let logical = (row % 16) * 8 + k_block % 8;
    ((row / 16) * k_blocks.div_ceil(8) + k_block / 8) * 128 + (logical % 32) * 4 + logical / 32
}

fn supported(
    query: &GemmQuery,
    phase: Phase,
    knobs: CompileKnobs,
    layout: WeightLayout,
) -> Result<(), Refusal> {
    query.validate()?;
    knobs.validate()?;
    let Some(group) = query.grouped else {
        return Err(Refusal("MoE engine requires grouped GEMM"));
    };
    if !matches!(group.capacity, RoutedCapacity::TotalRows(_)) {
        return Err(Refusal("MoE engine needs total padded row capacity"));
    }
    if query.output != DType::Bf16 || query.epilogues.residual_add {
        return Err(Refusal("MoE output must be BF16 without residual fusion"));
    }
    if query.epilogues.swiglu.is_some() == query.epilogues.weighted_moe_combine {
        return Err(Refusal(
            "MoE projection must select SwiGLU or weighted down output",
        ));
    }
    if let Some(quantization) = query.epilogues.output_quantization
        && (quantization != OutputQuantization::Mxfp8
            || query.epilogues.swiglu.is_none()
            || query.activation != ActivationFormat::Mxfp8
            || !matches!(query.weight, WeightFormat::Mxfp4 | WeightFormat::Mxfp8))
    {
        return Err(Refusal("unsupported MoE output quantization"));
    }
    let pair = matches!(
        (query.weight, query.activation),
        (
            WeightFormat::Mxfp4,
            ActivationFormat::Bf16 | ActivationFormat::Mxfp8
        ) | (
            WeightFormat::Mxfp8,
            ActivationFormat::Bf16 | ActivationFormat::Mxfp8
        ) | (
            WeightFormat::Nvfp4,
            ActivationFormat::Bf16 | ActivationFormat::Nvfp4
        ) | (
            WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
            ActivationFormat::Bf16 | ActivationFormat::Fp8Group128(ScaleMode::ExactF32)
        )
    );
    if !pair {
        return Err(Refusal("unsupported MoE weight and activation pairing"));
    }
    if fragment_prefill_query(query, layout) {
        supported_fragment_shape(query)?;
    }
    let fragment_prefill = phase == Phase::Prefill && fragment_prefill_query(query, layout);
    let packed_ring = packed::selects_ring(query, phase, knobs, layout);
    let tiles = if fragment_prefill {
        (FRAGMENT_PREFILL_TOKEN_TILE, 64)
    } else if packed_ring {
        (knobs.tile_n, knobs.tile_k)
    } else {
        (16, 32)
    };
    if query.n % 16 != 0 || query.k % 32 != 0 || (knobs.tile_n, knobs.tile_k) != tiles {
        return Err(Refusal("MoE dimensions require N16 and K32 tiles"));
    }
    let valid_shape = match phase {
        Phase::Decode => {
            knobs.tile_m == 16
                && (matches!(knobs.warps, 2 | 4 | 8 | 16) || packed_ring && knobs.warps <= 16)
        }
        Phase::Prefill => knobs.tile_m == 64 && knobs.warps == 8,
    };
    if !valid_shape {
        return Err(Refusal("MoE phase row tile or warp count is unsupported"));
    }
    let fragment_decode = phase == Phase::Decode && fragment_prefill_query(query, layout);
    if !(matches!(knobs.split_k_capacity, 1 | 2 | 4)
        || fragment_decode && knobs.split_k_capacity <= 16)
        || phase == Phase::Prefill && knobs.split_k_capacity != 1
        || query.k % u32::from(knobs.split_k_capacity) != 0
    {
        return Err(Refusal("unsupported MoE split-K capacity"));
    }
    Ok(())
}

fn fragment_prefill_query(query: &GemmQuery, layout: WeightLayout) -> bool {
    query.weight == WeightFormat::Mxfp4
        && query.activation == ActivationFormat::Bf16
        && layout == WeightLayout::Mxfp4Fragment
}

fn format_label(weight: WeightFormat) -> &'static str {
    match weight {
        WeightFormat::Mxfp4 => "mxfp4",
        WeightFormat::Mxfp8 => "mxfp8",
        WeightFormat::Nvfp4 => "nvfp4",
        WeightFormat::Fp8Block128x128(_) => "fp8block",
        _ => unreachable!(),
    }
}

impl<S: AsRef<str>> GroupedOperation<S> {
    pub fn fp8_fragment_weights(&self) -> bool {
        self.layout == WeightLayout::Fp8Fragment
    }

    pub fn fragment_weights(&self) -> bool {
        self.layout == WeightLayout::Mxfp4Fragment && self.query.weight == WeightFormat::Mxfp4
    }

    pub fn fragment_decode(&self) -> bool {
        self.fragment_weights() && self.phase == Phase::Decode && self.path == InputPath::A16Gather
    }

    pub fn fragment_prefill(&self) -> bool {
        self.fragment_weights() && self.phase == Phase::Prefill && self.path == InputPath::A16Gather
    }

    pub fn instantiation(&self) -> InstantiationSpec {
        InstantiationSpec {
            family: OpFamily::Gemm,
            key: CompileKey {
                query: StaticQuery::Gemm(self.query),
                knobs: self.knobs,
                dense: None,
                target: Target::Sm120f,
            },
            symbol: self.symbol.as_ref().into(),
        }
    }

    /// The weight map views row-major weights as [K bytes, expert N]. MiMo
    /// fragment entries retain this ABI map but read `base_buffer` directly.
    pub fn weight_tensor_map(&self, base_buffer: u64) -> Result<TensorMap, Refusal> {
        let experts = self.query.grouped.expect("validated grouped GEMM").experts;
        let bytes_per_row =
            if matches!(self.query.weight, WeightFormat::Mxfp4 | WeightFormat::Nvfp4) {
                self.query.k / 2
            } else {
                self.query.k
            };
        let map = TensorMap {
            base_buffer,
            dims: vec![
                u64::from(bytes_per_row),
                u64::from(experts) * u64::from(self.query.n),
            ],
            strides_bytes: vec![1, u64::from(bytes_per_row)],
            box_dims: vec![
                if self.phase == Phase::Decode {
                    self.packed_contract.map_or_else(
                        || if self.query.k == 384 { 64 } else { 128 },
                        |contract| u32::from(contract.decode_weight_bytes),
                    )
                } else if bytes_per_row == self.query.k
                    || self.path == InputPath::A8Pregather
                        && self.query.weight == WeightFormat::Nvfp4
                {
                    32
                } else {
                    16
                },
                if self.query.epilogues.swiglu == Some(SwigluColumns::Interleaved) {
                    32
                } else {
                    16
                },
            ],
            element_strides: vec![1, 1],
            swizzle: TensorMapSwizzle::None,
            interleave: TensorMapInterleave::None,
            l2_promotion: TensorMapL2Promotion::B256,
            oob_fill: TensorMapOobFill::Zero,
            element: TensorMapElement::U8,
        };
        map.validate()?;
        Ok(map)
    }

    /// Offline layout used by every MiMo MXFP4 projection and activation path.
    pub fn weight_fragment_layout(&self) -> Option<Mxfp4MoeFragmentLayout> {
        self.fragment_weights().then(|| Mxfp4MoeFragmentLayout {
            experts: self.query.grouped.expect("grouped GEMM").experts as usize,
            rows_per_expert: self.query.n as usize,
            k: self.query.k as usize,
        })
    }

    /// Legacy MX and NV scale tile geometry, used outside MiMo MXFP4.
    pub fn weight_scale_layout(&self) -> Option<ScaleTileLayout> {
        let block = match self.query.weight {
            WeightFormat::Mxfp4 | WeightFormat::Mxfp8 => 32,
            WeightFormat::Nvfp4 => 16,
            _ => return None,
        };
        let experts = self.query.grouped.expect("validated grouped GEMM").experts;
        Some(ScaleTileLayout {
            rows: (experts * self.query.n) as usize,
            k_blocks: (self.query.k / block) as usize,
            tile_rows: 16,
            k_blocks_per_tile: 8,
        })
    }

    /// Grid derives from prepared capacity. Device pointers carry all live counts.
    pub fn plan(&self, counts: RuntimeCounts) -> Result<LaunchPlan, Refusal> {
        if self.fp8_fragment_weights() {
            return Err(Refusal(
                "specialized FP8 fragments require their owning model planner",
            ));
        }
        supported(&self.query, self.phase, self.knobs, self.layout)?;
        if let Some(contract) = self.packed_contract {
            construction::validate_packed(self, contract)?;
        }
        if self.fragment_prefill() {
            supported_fragment_prefill(self.knobs, self.weight_stages)?;
        }
        if counts.m > self.tokens_capacity
            || counts.routed_rows > self.query.m_capacity
            || counts.split_k != self.knobs.split_k_capacity
        {
            return Err(Refusal(
                "live MoE counts exceed prepared capacity or split-K differs",
            ));
        }
        if self.packed_ring() {
            return self.packed_ring_plan();
        }
        let group = self.query.grouped.expect("validated grouped GEMM");
        let padded = match group.capacity {
            RoutedCapacity::TotalRows(rows) => rows,
            RoutedCapacity::PerExpert(_) => unreachable!(),
        };
        let output_n = if self.projection == Projection::GateUp {
            self.query.n / 2
        } else {
            self.query.n
        };
        let output_tile = if self.fragment_decode() {
            // Gate/up warps split K within the CTA.
            u32::from(self.knobs.warps) / u32::from(self.knobs.split_k_capacity) * 16
        } else if self.query.epilogues.output_quantization.is_some() {
            32
        } else {
            u32::from(self.knobs.tile_n)
        };
        let rows = self.query.m_capacity;
        let work_owners = if self.fragment_decode() {
            group.experts.min(rows)
        } else if self.phase == Phase::Decode {
            group.experts.min(rows) * u32::from(self.knobs.split_k_capacity)
        } else {
            padded.div_ceil(u32::from(self.knobs.tile_m))
        };
        let tiles = work_owners
            .checked_mul(output_n / output_tile)
            .ok_or(Refusal("MoE grid capacity overflow"))?;

        let workspace_bytes = if self.knobs.split_k_capacity > 1 && !self.fragment_decode() {
            u64::from(rows)
                .checked_mul(u64::from(output_n))
                .and_then(|v| {
                    v.checked_mul(if self.projection == Projection::GateUp {
                        2
                    } else {
                        1
                    })
                })
                .and_then(|v| v.checked_mul(u64::from(self.knobs.split_k_capacity) * 4))
                .ok_or(Refusal("MoE workspace capacity overflow"))?
        } else {
            0
        };
        let dynamic_shared_bytes = if self.fragment_decode() {
            fragment_decode_shared_bytes(
                self.projection,
                self.query.k,
                self.knobs.split_k_capacity,
                self.knobs.warps,
                self.tokens_capacity,
            )
        } else if self.phase == Phase::Decode {
            let weight_bytes =
                if matches!(self.query.weight, WeightFormat::Mxfp4 | WeightFormat::Nvfp4) {
                    self.query.k / 2
                } else {
                    self.query.k
                };
            weight_bytes
                .checked_mul(if self.projection == Projection::GateUp {
                    2
                } else {
                    1
                })
                .and_then(|v| {
                    v.checked_mul(if self.query.epilogues.output_quantization.is_some() {
                        2
                    } else {
                        1
                    })
                })
                .map(|v| v / u32::from(self.knobs.split_k_capacity))
                .and_then(|v| v.checked_mul(16))
                .ok_or(Refusal("MoE weight cache capacity overflow"))?
        } else {
            0
        };
        let (grid, dynamic_shared_bytes) = if self.fragment_prefill() {
            (
                SM_COUNT,
                fragment_prefill_shared_bytes(self.weight_stages, self.knobs.stages),
            )
        } else if self.fragment_decode() {
            (tiles, dynamic_shared_bytes)
        } else {
            (tiles.min(SM_COUNT * 2), dynamic_shared_bytes)
        };
        let plan = LaunchPlan {
            grid: [grid, 1, 1],
            block: [
                if self.fragment_prefill() {
                    FRAGMENT_PREFILL_THREADS
                } else {
                    u32::from(self.knobs.warps) * 32
                },
                1,
                1,
            ],
            dynamic_shared_bytes,
            workspace_bytes,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// The second launch sums split partials in a fixed order. Its grid is
    /// sized for routed-row capacity and skips inactive rows on the device.
    pub fn reduction_plan(&self, counts: RuntimeCounts) -> Result<Option<LaunchPlan>, Refusal> {
        self.plan(counts)?;
        if self.knobs.split_k_capacity == 1 || self.fragment_decode() || self.fp8_fragment_weights()
        {
            return Ok(None);
        }
        let output_n = if self.projection == Projection::GateUp {
            self.query.n / 2
        } else {
            self.query.n
        };
        let reduction_tiles = self
            .query
            .m_capacity
            .checked_mul(output_n)
            .ok_or(Refusal("MoE reduction grid capacity overflow"))?
            / REDUCE_THREADS;
        let plan = LaunchPlan {
            grid: [reduction_tiles, 1, 1],
            block: [REDUCE_THREADS, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(Some(plan))
    }

    pub fn reduction_symbol(&self) -> Option<String> {
        (self.knobs.split_k_capacity > 1 && !self.fragment_decode() && !self.fp8_fragment_weights())
            .then(|| self.symbol.as_ref().replace("_v1", "_reduce_v1"))
    }

    /// ABI argument order follows the generated entry signature.
    pub fn arguments(
        &self,
        handles: [u64; 15],
        weight_map: TensorMap,
    ) -> Result<ArgumentList, Refusal> {
        let args = vec![
            Argument::DeviceBuffer(handles[0]),
            Argument::DeviceBuffer(handles[1]),
            Argument::DeviceBuffer(handles[2]),
            Argument::DeviceBuffer(handles[3]),
            Argument::DeviceBuffer(handles[4]),
            Argument::DeviceBuffer(handles[5]),
            Argument::DeviceBuffer(handles[6]),
            Argument::DeviceBuffer(handles[7]),
            Argument::DeviceBuffer(handles[8]),
            Argument::DeviceBuffer(handles[9]),
            Argument::DeviceBuffer(handles[10]),
            Argument::DeviceBuffer(handles[11]),
            if self.query.epilogues.output_quantization.is_some() {
                Argument::DeviceBuffer(handles[12])
            } else {
                Argument::NullDevicePointer
            },
            if self.query.epilogues.output_quantization.is_some() {
                Argument::DeviceBuffer(handles[13])
            } else {
                Argument::NullDevicePointer
            },
            if self.knobs.split_k_capacity > 1
                && !self.fragment_decode()
                && !self.fp8_fragment_weights()
            {
                Argument::DeviceBuffer(handles[14])
            } else {
                Argument::NullDevicePointer
            },
            Argument::DeviceBuffer(weight_map.base_buffer),
            Argument::TensorMap(weight_map),
        ];
        let list = ArgumentList(args);
        list.validate()?;
        Ok(list)
    }

    pub fn reduction_arguments(&self, handles: [u64; 15]) -> Result<Option<ArgumentList>, Refusal> {
        if self.knobs.split_k_capacity == 1 || self.fragment_decode() || self.fp8_fragment_weights()
        {
            return Ok(None);
        }
        let list = ArgumentList(vec![
            Argument::DeviceBuffer(handles[14]),
            Argument::DeviceBuffer(handles[5]),
            Argument::DeviceBuffer(handles[6]),
            Argument::DeviceBuffer(handles[10]),
            Argument::DeviceBuffer(handles[2]),
            Argument::DeviceBuffer(handles[4]),
            if self.query.epilogues.output_quantization.is_some() {
                Argument::DeviceBuffer(handles[12])
            } else {
                Argument::NullDevicePointer
            },
            if self.query.epilogues.output_quantization.is_some() {
                Argument::DeviceBuffer(handles[13])
            } else {
                Argument::NullDevicePointer
            },
            Argument::DeviceBuffer(handles[7]),
            Argument::DeviceBuffer(handles[8]),
            Argument::DeviceBuffer(handles[9]),
            Argument::DeviceBuffer(handles[11]),
        ]);
        list.validate()?;
        Ok(Some(list))
    }
}

/// A device pointer entry parameter.
pub fn ptr(name: &str, cxx_type: &str) -> EntryParam {
    EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    }
}

/// The entry ABI every grouped expert entry declares.
pub fn parameters() -> Vec<EntryParam> {
    vec![
        ptr("activation", "const uint8_t"),
        ptr("activation_scales", "const uint8_t"),
        ptr("activation_f32_scales", "const float"),
        ptr("weight_scales", "const uint8_t"),
        ptr("weight_f32_scales", "const float"),
        ptr("output", "uint16_t"),
        ptr("topk_weights", "const float"),
        ptr("expert_offsets", "const int32_t"),
        ptr("sorted_rows", "const int32_t"),
        ptr("num_padded_rows", "const int32_t"),
        ptr("live_tokens", "const int32_t"),
        ptr("active_experts", "const int32_t"),
        ptr("quantized_output", "uint8_t"),
        ptr("quantized_scales", "uint8_t"),
        ptr("partials", "float"),
        ptr("weight_bytes", "const uint8_t"),
        EntryParam::TensorMap {
            name: "weight_map".into(),
        },
    ]
}

/// The packed engine's weight format id, or why it runs no such weights.
fn device_format(format: WeightFormat) -> Result<u8, Refusal> {
    match format {
        WeightFormat::Mxfp4 => Ok(0),
        WeightFormat::Fp8Block128x128(_) => Ok(1),
        WeightFormat::Nvfp4 => Ok(2),
        WeightFormat::Mxfp8 => Ok(3),
        _ => Err(Refusal(
            "the packed engine runs MXFP4, FP8 block, NVFP4 and MXFP8 weights only",
        )),
    }
}

/// The packed engine's entry body. Panics where [`try_packed_body`]
/// refuses the operation.
fn packed_body(v: &GroupedOperation<impl AsRef<str>>) -> String {
    try_packed_body(v).unwrap_or_else(|Refusal(reason)| panic!("{}: {reason}", v.symbol.as_ref()))
}

/// [`packed_body`], or why the packed engine cannot build the operation:
/// weights it does not run, or no grouped geometry.
fn try_packed_body(v: &GroupedOperation<impl AsRef<str>>) -> Result<String, Refusal> {
    let group = v
        .query
        .grouped
        .ok_or(Refusal("MoE requires grouped geometry"))?;
    let extra = v.packed_contract.map_or_else(String::new, |contract| {
        format!(
            ", {}, {}",
            contract.decode_weight_bytes,
            u8::from(contract.scales.weight_globals == GlobalScaleLayout::ExpertProjection)
        )
    });
    Ok(format!(
        "chop::moe::run<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}{}>(activation, activation_scales, activation_f32_scales, weight_scales, weight_f32_scales, output, topk_weights, expert_offsets, sorted_rows, num_padded_rows, live_tokens, active_experts, quantized_output, quantized_scales, partials, weight_bytes, &weight_map);",
        device_format(v.query.weight)?,
        u8::from(v.layout == WeightLayout::Mxfp4Fragment),
        group.top_k,
        u8::from(v.path == InputPath::A8Pregather),
        u8::from(v.projection == Projection::GateUp),
        u8::from(v.query.epilogues.swiglu == Some(SwigluColumns::Interleaved)),
        u8::from(v.query.epilogues.output_quantization == Some(OutputQuantization::Mxfp8)),
        group.experts,
        v.query.n,
        v.query.k,
        v.knobs.tile_m,
        v.knobs.warps,
        v.knobs.split_k_capacity,
        extra,
    ))
}

/// The header of the split-K reductions (`moe/reduce.cuh`), which only
/// cubins with a reduction entry include.
pub(crate) const REDUCE_HEADER: &str = "#include \"moe/reduce.cuh\"";

fn reduction_entry(v: &GroupedOperation<impl AsRef<str>>) -> Option<EntrySpec> {
    let reduction_symbol = v.reduction_symbol()?;
    let output_n = if v.projection == Projection::GateUp {
        v.query.n / 2
    } else {
        v.query.n
    };
    let reduce_body = format!(
        "chop::moe::reduce_owned<{}, {}, {}, {}, {}, {}, {}, {}, {REDUCE_THREADS}>(partials, output, topk_weights, live_tokens, activation_f32_scales, weight_f32_scales, quantized_output, quantized_scales, expert_offsets, sorted_rows, num_padded_rows, active_experts);",
        u8::from(v.projection == Projection::GateUp),
        u8::from(v.query.epilogues.output_quantization.is_some()),
        if v.query.weight != WeightFormat::Nvfp4 {
            0
        } else if v.path == InputPath::A8Pregather {
            1
        } else {
            2
        },
        v.query.grouped.expect("grouped").top_k,
        output_n,
        v.tokens_capacity,
        v.knobs.split_k_capacity,
        v.query.grouped.expect("grouped").experts,
    );
    Some(EntrySpec::new(
        reduction_symbol,
        vec![
            ptr("partials", "const float"),
            ptr("output", "uint16_t"),
            ptr("topk_weights", "const float"),
            ptr("live_tokens", "const int32_t"),
            ptr("activation_f32_scales", "const float"),
            ptr("weight_f32_scales", "const float"),
            ptr("quantized_output", "uint8_t"),
            ptr("quantized_scales", "uint8_t"),
            ptr("expert_offsets", "const int32_t"),
            ptr("sorted_rows", "const int32_t"),
            ptr("num_padded_rows", "const int32_t"),
            ptr("active_experts", "const int32_t"),
        ],
        reduce_body,
        REDUCE_THREADS,
        1,
    ))
}

/// A grouped entry's device code as its cubin compiles it: the template
/// instance or packed engine body, header, defines and threads, and its
/// reduction entry. The backend's launch key guard compares it.
#[cfg(test)]
pub(crate) fn device_code(v: &Variant) -> String {
    let implementation = template::implementation(v).unwrap_or_else(|| Implementation {
        body: packed_body(&v.operation()),
        header: "#include \"moe/moe.cuh\"",
        defines: Vec::new(),
        threads: u32::from(v.knobs.warps) * 32,
    });
    format!(
        "{} {} {:?} {} {:?}",
        implementation.body,
        implementation.header,
        implementation.defines,
        implementation.threads,
        reduction_entry(&v.operation())
    )
}

/// A grouped entry's device code: the entry body, the header defining it and
/// the defines of its cubin.
pub(crate) struct Implementation {
    pub(crate) body: String,
    pub(crate) header: &'static str,
    pub(crate) defines: Vec<(String, String)>,
    /// The entry's CTA threads.
    pub(crate) threads: u32,
}

/// One cubin per profile, format, path, projection and phase, with decode bucket entries together.
/// Fragment-layout grouped entries are grouped-MoE template instances
/// (`template::implementation`); the standard profile and the A8 paths use
/// the packed engine (`moe.cuh`).
pub fn cubin_specs(catalogs: &Catalogs, profile: Option<Profile>) -> Vec<CubinSpec> {
    declared_cubin_specs(catalogs, profile)
        .into_iter()
        .map(DeclaredCubin::into_spec)
        .collect()
}

/// [`cubin_specs`] with each grouped entry's and its reduction's declared
/// arithmetic ([`arithmetic::grouped`]); the route cubins compile theirs in.
pub fn declared_cubin_specs(catalogs: &Catalogs, profile: Option<Profile>) -> Vec<DeclaredCubin> {
    let mut groups: Vec<CubinSpec> = Vec::new();
    let mut declared = BTreeMap::new();
    for v in catalog(catalogs)
        .into_iter()
        .filter(|v| profile.is_none_or(|p| p == v.profile))
    {
        if let Err(Refusal(reason)) = v.check_numerics() {
            panic!("{}: {reason}", v.symbol);
        }
        let arithmetic = arithmetic::grouped(&v);
        if let Some(reduction) = v.reduction_symbol() {
            declared.insert(reduction, format!("reduction; {arithmetic}"));
        }
        declared.insert(v.symbol.clone(), arithmetic);
        let base = v.symbol.trim_end_matches("_v1");
        let name = if v.phase == Phase::Decode {
            base.rsplit_once("_t")
                .expect("decode bucket suffix")
                .0
                .to_string()
        } else {
            base.to_string()
        };
        let implementation = template::implementation(&v).unwrap_or_else(|| Implementation {
            body: packed_body(&v.operation()),
            header: "#include \"moe/moe.cuh\"",
            defines: Vec::new(),
            threads: u32::from(v.knobs.warps) * 32,
        });
        let entry = EntrySpec::new(
            v.symbol.clone(),
            parameters(),
            implementation.body,
            implementation.threads,
            1,
        );
        let mut entries = vec![entry];
        let mut headers = vec![implementation.header.to_string()];
        if let Some(entry) = reduction_entry(&v.operation()) {
            entries.push(entry);
            headers.push(REDUCE_HEADER.into());
        }
        if let Some(group) = groups.iter_mut().find(|group| group.name == name) {
            group.entries.extend(entries);
            for header in headers {
                if !group.headers.contains(&header) {
                    group.headers.push(header);
                }
            }
        } else {
            groups.push(CubinSpec {
                name,
                sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
                include_dirs: vec![PathBuf::from("common"), PathBuf::from("moe")],
                headers,
                defines: implementation.defines,
                entries,
            });
        }
    }
    for model in catalogs.models() {
        groups.extend((model.moe_route_cubins)(profile));
    }
    DeclaredCubin::all(groups, declared).expect("grouped entries declare their arithmetic")
}

#[cfg(test)]
mod tests {
    use super::*;

    use kernel_core::transform::Transform;

    #[test]
    fn declared_shapes_and_layout_do_not_inherit_profile_dimensions() {
        let shape = Shape {
            top_k: 3,
            hidden: 2048,
            experts: 256,
            intermediate: 256,
        };
        let (n, k) = shape.projection(Projection::GateUp);
        let (swiglu, weighted) =
            expert_epilogue(Projection::GateUp, RouteWeight::AfterDown, None).unwrap();
        let rows = 2 * shape.top_k;
        let mut v = Variant {
            profile: Profile::StdDenseMoe,
            class: "moe",
            layout: WeightLayout::Packed,
            projection: Projection::GateUp,
            phase: Phase::Decode,
            tokens_capacity: 2,
            path: InputPath::A16Gather,
            query: GemmQuery {
                m_capacity: rows,
                n,
                k,
                weight: WeightFormat::Mxfp4,
                activation: ActivationFormat::Bf16,
                output: DType::Bf16,
                epilogues: Epilogues {
                    swiglu,
                    weighted_moe_combine: weighted,
                    ..Epilogues::default()
                },
                grouped: Some(GroupedGemm {
                    experts: shape.experts,
                    top_k: shape.top_k,
                    capacity: RoutedCapacity::TotalRows(
                        padded_capacity(rows, shape.experts, 16).unwrap(),
                    ),
                }),
            },
            knobs: CompileKnobs {
                tile_m: 16,
                tile_n: 16,
                tile_k: 32,
                stages: 1,
                swizzle: 0,
                warps: 2,
                split_k_capacity: 2,
            },
            weight_stages: 0,
            exact: false,
            route_weight: RouteWeight::AfterDown,
            swiglu_clamp: None,
            engine: None,
            schedule: None,
            symbol: "synthetic_experts".into(),
            invariance: Invariance::NONE,
        };
        let counts = RuntimeCounts {
            m: 2,
            batch: 0,
            context: 0,
            routed_rows: rows,
            split_k: 2,
        };
        assert_eq!((v.query.n, v.query.k), (512, 2048));
        assert!(!v.fragment_weights());
        assert!(v.weight_fragment_layout().is_none());
        let plan = v.plan(counts).unwrap();
        assert_eq!(plan.workspace_bytes, u64::from(rows) * 256 * 2 * 2 * 4);
        v.profile = Profile::synthetic(crate::ProfileNames {
            name: "other",
            tag: "other",
        });
        assert_eq!(v.plan(counts).unwrap(), plan);
        assert!(!v.fragment_weights());
        assert_eq!(shape.projection(Projection::Down), (2048, 256));
        assert_eq!(
            expert_epilogue(Projection::Down, RouteWeight::AfterDown, None),
            Ok((None, true))
        );
        assert!(expert_epilogue(Projection::GateUp, RouteWeight::BeforeDown, None).is_err());
        let clamp = SwigluClamp {
            gate_max: 10.0,
            up_limit: 10.0,
        };
        assert!(expert_epilogue(Projection::GateUp, RouteWeight::AfterDown, Some(clamp)).is_err());
    }

    #[test]
    fn decode_route_refuses_batches_above_32_tokens() {
        assert_eq!(ROUTE_MAX_TOKENS, 32);
        assert!(decode_route_plan(32, 32).is_ok());
        for rows in [33, 64, 256] {
            for (capacity, live) in [(rows, rows), (rows, 1), (32, rows)] {
                let refusal = decode_route_plan(capacity, live).unwrap_err();
                assert!(refusal.0.contains("at most 32 tokens"), "{}", refusal.0);
            }
        }
    }

    #[test]
    fn capacity_and_catalog_are_static() {
        assert_eq!(padded_capacity(8, 256, 16).unwrap(), 128);
        // The standard profile: three formats, two paths, two projections,
        // six decode buckets and prefill, with interleaved gate/up.
        assert_eq!(own_catalog().len(), 3 * 2 * (2 + 1) * 7);
        assert!(
            own_catalog()
                .iter()
                .all(|v| v.instantiation().validate().is_ok())
        );
        // Per format and path: gate/up and down decode and prefill, and the
        // interleaved gate/up decode and prefill.
        assert_eq!(cubin_specs(&Catalogs::base(), None).len(), 3 * 2 * 6);
        assert_eq!(decode_route_plan(32, 17).unwrap().block[0], 1024);
        assert_eq!(decode_route_plan(1, 1).unwrap().block[0], 256);
        assert!(decode_route_plan(8, 9).is_err());
        assert!(decode_route_arguments([1; 10], 0.0).is_err());
    }

    #[test]
    fn unsupported_pair_and_split_are_refused() {
        let v = &own_catalog()[0];
        let mut query = v.query;
        query.activation = ActivationFormat::Nvfp4;
        assert_eq!(
            supported(&query, v.phase, v.knobs, v.layout),
            Err(Refusal("unsupported MoE weight and activation pairing"))
        );
        let mut knobs = v.knobs;
        knobs.tile_m = 32;
        assert_eq!(
            supported(&v.query, v.phase, knobs, v.layout),
            Err(Refusal("MoE phase row tile or warp count is unsupported"))
        );
        let mut query = v.query;
        query.epilogues.output_quantization = Some(OutputQuantization::Nvfp4);
        assert_eq!(
            supported(&query, v.phase, v.knobs, v.layout),
            Err(Refusal("unsupported MoE output quantization"))
        );
    }

    #[test]
    fn packed_scale_offsets_follow_kernel_core_transform() {
        let layout = ScaleTileLayout {
            rows: 32,
            k_blocks: 16,
            tile_rows: 16,
            k_blocks_per_tile: 8,
        };
        let source: Vec<u8> = (0..layout.rows * layout.k_blocks)
            .map(|i| i as u8)
            .collect();
        let packed = layout.pack(&source).unwrap();
        for row in 0..layout.rows {
            for kb in 0..layout.k_blocks {
                assert_eq!(
                    packed.bytes[packed_scale_offset(row, kb, layout.k_blocks)],
                    source[row * layout.k_blocks + kb]
                );
            }
        }
        assert_eq!(layout.unpack(&packed).unwrap(), source);
    }
}
