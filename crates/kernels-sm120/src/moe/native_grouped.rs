//! Native block-scaled MXFP4 x MXFP8 expert tiles for prefill
//! (`cuda/moe/native_grouped.cuh`), in any expert geometry.
//!
//! These entries change bits against the BF16-decoded engines and qualify
//! on the within-bound track (docs/numerics.md, "A speedup that changes
//! bits"): every product is exact and the 128-K chains fold in ascending K
//! order as before, but the instruction reduces each 32-K group itself.
//! No catalog or backend selects them; a model binds them explicitly.
//!
//! A projection is two launches: one warp packs the active experts into
//! row tiles of [`Tile::rows`], then a persistent grid takes (tile, column
//! strip) items from a ticket. A row's arithmetic depends only on its own
//! routes and data, never on the tile, the item order or the grid.
//!
//! A gate/up compute entry also has a build reading its bank in 64-row by
//! 128-K tiles ([`row_tiles_entry`]); each copy moves the same bytes into
//! the same shared slot, so its outputs are the row-major entry's.
use crate::moe::{SM_COUNT, ptr};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::backend::Launch;
use kernel_core::contract::{
    ActivationFormat, Argument, ArgumentList, DType, LaunchPlan, OutputQuantization, Refusal,
    SwigluColumns, Validate, WeightFormat,
};
use kernel_core::gated_mlp::{GatedMlpProjection, RouteWeightPoint};
use kernel_core::moe_chained::ChainedMlpProjectionQuery;
use std::path::PathBuf;

/// SM120's opt-in dynamic shared memory per block.
pub const MAX_SHARED_BYTES: u32 = 101_376;
/// Shared memory per SM, for the resident-CTA estimate of the grid.
const SM_SHARED_BYTES: u32 = 102_400;
const QUEUE_HEADER: u64 = 4;

/// One projection's dimensions: `n` output columns per half (gate/up has a
/// gate and an up half over the same columns), reduction `k`, and the
/// token capacity of the routes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub experts: u32,
    pub top_k: u32,
    pub k: u32,
    pub n: u32,
    pub gate_up: bool,
    pub capacity: u32,
    /// Gate maximum and up limit bits of the SwiGLU clamp (gate/up only).
    pub clamp: Option<(u32, u32)>,
}

impl Shape {
    pub fn new(
        experts: u32,
        top_k: u32,
        k: u32,
        n: u32,
        gate_up: bool,
        capacity: u32,
        clamp: Option<(u32, u32)>,
    ) -> Result<Self, Refusal> {
        let shape = Self {
            experts,
            top_k,
            k,
            n,
            gate_up,
            capacity,
            clamp,
        };
        shape.validate()?;
        Ok(shape)
    }

    /// The shape of a chained MXFP4 / ceil-MXFP8 projection whose routes
    /// are weighted before down, as the exact prefill engines read it.
    pub fn from_query(query: &ChainedMlpProjectionQuery, capacity: u32) -> Result<Self, Refusal> {
        query.validate()?;
        let p = query.projection;
        let g = p.gemm;
        let grouped = g
            .grouped
            .ok_or(Refusal("native tiles need grouped experts"))?;
        let gate_up = p.projection == GatedMlpProjection::GateUp;
        let expected = if gate_up {
            (
                DType::Bf16,
                Some(SwigluColumns::ConcatenatedHalves),
                Some(OutputQuantization::Mxfp8),
            )
        } else {
            (DType::F32, None, None)
        };
        if g.weight != WeightFormat::Mxfp4
            || g.activation != ActivationFormat::Mxfp8
            || (
                g.output,
                g.epilogues.swiglu,
                g.epilogues.output_quantization,
            ) != expected
            || p.route_weight != Some(RouteWeightPoint::BeforeDown)
            || query.k.stored != g.k
            || (!gate_up && p.swiglu_clamp.is_some())
        {
            return Err(Refusal(
                "native tiles serve MXFP4 x ceil-MXFP8 projections weighted before down",
            ));
        }
        if u64::from(capacity) * u64::from(grouped.top_k) > u64::from(g.m_capacity) {
            return Err(Refusal("native tiles need the routes' total rows"));
        }
        Self::new(
            grouped.experts,
            grouped.top_k,
            g.k,
            if gate_up { g.n / 2 } else { g.n },
            gate_up,
            capacity,
            p.swiglu_clamp.map(|c| (c.gate_max_bits, c.up_limit_bits)),
        )
    }

    pub fn halves(&self) -> u32 {
        if self.gate_up { 2 } else { 1 }
    }

    fn tag(&self) -> String {
        let clamp = self
            .clamp
            .map_or(String::new(), |(g, u)| format!("_clamp{g:08x}{u:08x}"));
        format!(
            "{}_e{}k{}_n{}_k{}_t{}{clamp}",
            if self.gate_up { "gateup" } else { "down" },
            self.experts,
            self.top_k,
            self.n,
            self.k,
            self.capacity
        )
    }

    /// The kernel's `Config` type.
    pub fn config(&self) -> String {
        let clamp = self.clamp.map_or_else(
            || "chop::moe::clamp::SwigluClamp<false,0u,0u>".into(),
            |(g, u)| format!("chop::moe::clamp::SwigluClamp<true,{g:#x}u,{u:#x}u>"),
        );
        format!(
            "chop::moe::native_grouped::Config<{},{},{},{},{},{},{clamp}>",
            self.experts,
            self.top_k,
            self.k,
            self.n,
            self.halves(),
            self.capacity
        )
    }
}

impl Validate for Shape {
    fn validate(&self) -> Result<(), Refusal> {
        let routes = u64::from(self.capacity) * u64::from(self.top_k);
        if self.experts == 0
            || self.experts > 4096
            || self.top_k == 0
            || self.top_k > 32
            || self.capacity == 0
            || routes >= 1 << 31
            || self.k == 0
            || !self.k.is_multiple_of(32)
            || self.n == 0
            || !self.n.is_multiple_of(16)
            || u64::from(self.experts) * u64::from(self.halves()) * u64::from(self.n) >= 1 << 31
            || (self.clamp.is_some() && !self.gate_up)
        {
            return Err(Refusal(
                "native expert tile shape outside its compiled domain",
            ));
        }
        Ok(())
    }
}

/// A CTA's warp grid: `row_warps` of 64 routed rows by `col_warps` of 16
/// (gate/up, gate and up halves) or 32 (down) output columns, with
/// `stages` 128-K operand stages in flight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    pub row_warps: u32,
    pub col_warps: u32,
    pub stages: u32,
}

impl Tile {
    pub fn rows(&self) -> u32 {
        self.row_warps * 64
    }
    pub fn threads(&self) -> u32 {
        self.row_warps * self.col_warps * 32
    }
    pub fn cols(&self, shape: &Shape) -> u32 {
        self.col_warps * if shape.gate_up { 16 } else { 32 }
    }
    pub fn strips(&self, shape: &Shape) -> u32 {
        shape.n.div_ceil(self.cols(shape))
    }
    fn name(&self) -> String {
        format!("r{}w{}s{}", self.row_warps, self.col_warps, self.stages)
    }
    fn check(&self, shape: &Shape) -> Result<(), Refusal> {
        shape.validate()?;
        if !(1..=4).contains(&self.row_warps)
            || self.col_warps == 0
            || self.threads() > 1024
            || !(2..=4).contains(&self.stages)
            || shared_bytes(shape, self) > MAX_SHARED_BYTES
        {
            return Err(Refusal("native expert tile exceeds its CTA limits"));
        }
        Ok(())
    }
}

/// Gate/up tiles measured against each other in the lab.
pub const GATE_UP_TILES: [Tile; 3] = [
    Tile {
        row_warps: 2,
        col_warps: 4,
        stages: 3,
    },
    Tile {
        row_warps: 1,
        col_warps: 6,
        stages: 3,
    },
    Tile {
        row_warps: 2,
        col_warps: 6,
        stages: 3,
    },
];
/// Down tiles measured against each other in the lab.
pub const DOWN_TILES: [Tile; 2] = [
    Tile {
        row_warps: 2,
        col_warps: 4,
        stages: 3,
    },
    Tile {
        row_warps: 1,
        col_warps: 8,
        stages: 3,
    },
];

/// `sizeof(Shared<C, T>)`, which each compute entry asserts.
pub fn shared_bytes(shape: &Shape, tile: &Tile) -> u32 {
    (tile.stages * stage_bytes(shape, tile) + tile.rows() * 8 + 4).next_multiple_of(16)
}

/// `sizeof(Shared<C, T>::Stage)`.
fn stage_bytes(shape: &Shape, tile: &Tile) -> u32 {
    let words = if (shape.k / 32).is_multiple_of(4) {
        1
    } else {
        2
    };
    let weight_rows = shape.halves() * tile.cols(shape);
    (tile.rows() * 128 + weight_rows * 64 + (tile.rows() + weight_rows) * 4 * words)
        .next_multiple_of(16)
}

pub fn queue_slots(shape: &Shape, tile: &Tile) -> u64 {
    (u64::from(shape.capacity) * u64::from(shape.top_k)).div_ceil(u64::from(tile.rows()))
        + u64::from(shape.experts)
}

/// The queue's bytes: its header and one (expert, first, rows) per tile.
pub fn workspace_bytes(shape: &Shape, tile: &Tile) -> u64 {
    ((QUEUE_HEADER + 3 * queue_slots(shape, tile)) * 4).next_multiple_of(256)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Pack,
    Compute,
}

/// A compute entry's lab variant. Serving binds only [`Variant::Serving`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Serving,
    /// Drops the first 128-K chain: a negative control.
    OmitFirst,
    /// Gate/up only: also writes the F32 gate and up sums for the oracle.
    Raw,
}

pub fn symbol(shape: &Shape, tile: &Tile, stage: Stage, variant: Variant) -> String {
    format!(
        "chop_moe_native_{}_{}_{}{}_v1",
        shape.tag(),
        tile.name(),
        match stage {
            Stage::Pack => "pack",
            Stage::Compute => "tiles",
        },
        match variant {
            Variant::Serving => "",
            Variant::OmitFirst => "_omit_first",
            Variant::Raw => "_raw",
        }
    )
}

/// The lab's single-instruction probe (`native_grouped::probe`).
pub const PROBE: &str = "chop_moe_native_mma_probe_v1";

/// The pack and compute entries; `lab` adds the lab variants.
pub fn entries(shape: &Shape, tile: &Tile, lab: bool) -> Result<Vec<EntrySpec>, Refusal> {
    tile.check(shape)?;
    let c = shape.config();
    let t = format!(
        "chop::moe::native_grouped::Tile<{},{},{}>",
        tile.row_warps, tile.col_warps, tile.stages
    );
    let dependency = "chop::ptx::griddep_launch_dependents(); chop::ptx::griddep_wait();";
    let mut entries = vec![EntrySpec::new(
        symbol(shape, tile, Stage::Pack, Variant::Serving),
        vec![
            ptr("counts", "const int"),
            ptr("offsets", "const int"),
            ptr("active", "const int"),
            ptr("live", "const int"),
            ptr("scratch", "int"),
        ],
        format!(
            "{dependency} chop::moe::native_grouped::pack<{c},{t}>(counts,offsets,active,live,scratch);"
        ),
        32,
        1,
    )];
    let mut variants = vec![Variant::Serving];
    if lab {
        variants.push(Variant::OmitFirst);
        if shape.gate_up {
            variants.push(Variant::Raw);
        }
    }
    for variant in variants {
        let mut params = vec![
            ptr("input", "const uint8_t"),
            ptr("weight", "const uint8_t"),
            ptr("weight_scales", "const uint8_t"),
            ptr("activation_scales", "const uint8_t"),
            ptr("sorted", "const int"),
            ptr("padded", "const int"),
            ptr("live", "const int"),
            ptr("scratch", "int"),
            ptr("route_weights", "const float"),
            ptr("output", "void"),
        ];
        let raw = variant == Variant::Raw;
        if raw {
            params.push(ptr("raw", "float"));
        }
        entries.push(EntrySpec::new(
            symbol(shape, tile, Stage::Compute, variant),
            params,
            format!(
                "static_assert(sizeof(chop::moe::native_grouped::Shared<{c},{t}>) == {}); \
                 extern __shared__ __align__(16) uint8_t arena[]; {dependency} \
                 chop::moe::native_grouped::compute<{c},{t},{},{raw}>(arena,input,weight,\
                 weight_scales,activation_scales,sorted,padded,live,scratch,route_weights,output{});",
                shared_bytes(shape, tile),
                variant == Variant::OmitFirst,
                if raw { ",raw" } else { "" }
            ),
            tile.threads(),
            1,
        ));
    }
    Ok(entries)
}

/// A compute entry reading its bank in row tiles.
pub fn row_tiles_symbol(symbol: &str) -> String {
    format!("{}_row_tiles_v1", symbol.trim_end_matches("_v1"))
}

/// `entry`, a compute entry of `shape` (served, tuned or fused ceil), with
/// its bank read in 64-row by 128-K tiles (`row-tiles-64x128-groups`,
/// `Config`'s `Tiles`): the same ABI, launch and arithmetic.
pub fn row_tiles_entry(shape: &Shape, entry: &EntrySpec) -> Result<EntrySpec, Refusal> {
    if !shape.gate_up
        || !shape.k.is_multiple_of(128)
        || !(shape.halves() * shape.n).is_multiple_of(64)
    {
        return Err(Refusal(
            "row tiles cover whole 64-row by 128-K tiles of a gate/up bank",
        ));
    }
    let weight =
        |p: &EntryParam| matches!(p, EntryParam::DevicePointer { name, .. } if name == "weight");
    let config = shape.config();
    if !entry.params.iter().any(weight) || !entry.body.contains(&config) {
        return Err(Refusal("the entry reads no weights of this shape"));
    }
    let tiled = format!("{},true>", &config[..config.len() - 1]);
    let mut e = entry.clone();
    e.symbol = row_tiles_symbol(&entry.symbol);
    e.body = entry.body.replace(&config, &tiled);
    Ok(e)
}

/// The symbol of compute entry `symbol`'s quad-lane build.
pub fn lanes_symbol(symbol: &str) -> String {
    format!("{}_lanes_v1", symbol.trim_end_matches("_v1"))
}

/// [`row_tiles_entry`] with the tiles' code bytes in quad-lane order
/// (`row-tiles-64x128-lanes`, `Config`'s `Lanes`): each thread copies whole
/// tile rows and restores their row-tiled order in shared memory before
/// the stage barrier, so its ABI, launch and arithmetic are the row-tiled
/// entry's.
pub fn lanes_entry(shape: &Shape, entry: &EntrySpec) -> Result<EntrySpec, Refusal> {
    let mut e = row_tiles_entry(shape, entry)?;
    let config = shape.config();
    let tiles = format!("{},true>", &config[..config.len() - 1]);
    assert!(e.body.contains(&tiles));
    e.symbol = lanes_symbol(&entry.symbol);
    e.body = e.body.replace(
        &tiles,
        &format!("{},true,true>", &config[..config.len() - 1]),
    );
    Ok(e)
}

/// One cubin of `instances`; `lab` adds the lab variants and the probe.
pub fn cubin_spec(
    name: &str,
    instances: &[(Shape, Tile)],
    lab: bool,
) -> Result<CubinSpec, Refusal> {
    let mut entries = Vec::new();
    for (shape, tile) in instances {
        entries.extend(self::entries(shape, tile, lab)?);
    }
    if lab {
        entries.push(EntrySpec::new(
            PROBE.into(),
            vec![
                ptr("a", "const uint32_t"),
                ptr("b", "const uint32_t"),
                ptr("scales", "const uint32_t"),
                ptr("d", "float"),
            ],
            "chop::moe::native_grouped::probe(a,b,scales,d);".into(),
            32,
            1,
        ));
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: Vec::new(),
        headers: vec!["#include \"moe/native_grouped.cuh\"".into()],
        defines: Vec::new(),
        entries,
    })
}

/// The persistent grid: as many CTAs as shared memory lets each SM hold,
/// at most two. Extra CTAs only take later tickets.
pub fn grid(shape: &Shape, tile: &Tile) -> u32 {
    let per_sm = if 2 * (shared_bytes(shape, tile) + 1024) <= SM_SHARED_BYTES {
        2
    } else {
        1
    };
    SM_COUNT * per_sm
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Buffers {
    pub input: u64,
    pub weights: u64,
    pub weight_scales: u64,
    pub activation_scales: u64,
    pub offsets: u64,
    pub sorted_rows: u64,
    pub padded_rows: u64,
    pub live_tokens: u64,
    pub route_weights: u64,
    pub output: u64,
    pub counts: u64,
    pub active: u64,
    pub scratch: u64,
    pub raw: u64,
}

/// The pack and compute launches of a step of `rows` tokens.
/// Device handles of one projection. `route_weights` may be zero for down,
/// and `raw` is read only by [`Variant::Raw`].
pub fn launches(
    shape: &Shape,
    tile: &Tile,
    rows: u32,
    b: &Buffers,
    variant: Variant,
) -> Result<Vec<Launch>, Refusal> {
    tile.check(shape)?;
    if rows == 0 || rows > shape.capacity {
        return Err(Refusal("native expert launch exceeds its token capacity"));
    }
    let required = [
        b.input,
        b.weights,
        b.weight_scales,
        b.activation_scales,
        b.offsets,
        b.sorted_rows,
        b.padded_rows,
        b.live_tokens,
        b.output,
        b.counts,
        b.active,
        b.scratch,
    ];
    if required.contains(&0)
        || (shape.gate_up && b.route_weights == 0)
        || (variant == Variant::Raw) != (b.raw != 0)
        || (variant == Variant::Raw && !shape.gate_up)
    {
        return Err(Refusal("native expert launch is missing a buffer"));
    }
    let arguments = |handles: Vec<u64>| -> Result<ArgumentList, Refusal> {
        let list = ArgumentList(
            handles
                .into_iter()
                .map(|h| {
                    if h == 0 {
                        Argument::NullDevicePointer
                    } else {
                        Argument::DeviceBuffer(h)
                    }
                })
                .collect(),
        );
        list.validate()?;
        Ok(list)
    };
    let plan = |grid: u32, threads: u32, shared: u32| -> Result<LaunchPlan, Refusal> {
        let plan = LaunchPlan {
            grid: [grid, 1, 1],
            block: [threads, 1, 1],
            dynamic_shared_bytes: shared,
            workspace_bytes: workspace_bytes(shape, tile),
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    };
    Ok(vec![
        Launch {
            symbol: symbol(shape, tile, Stage::Pack, Variant::Serving),
            rows,
            plan: plan(1, 32, 0)?,
            arguments: arguments(vec![
                b.counts,
                b.offsets,
                b.active,
                b.live_tokens,
                b.scratch,
            ])?,
            collective: false,
            persistent: false,
        },
        Launch {
            symbol: symbol(shape, tile, Stage::Compute, variant),
            rows,
            plan: plan(grid(shape, tile), tile.threads(), shared_bytes(shape, tile))?,
            arguments: arguments(
                [
                    b.input,
                    b.weights,
                    b.weight_scales,
                    b.activation_scales,
                    b.sorted_rows,
                    b.padded_rows,
                    b.live_tokens,
                    b.scratch,
                    b.route_weights,
                    b.output,
                ]
                .into_iter()
                .chain((variant == Variant::Raw).then_some(b.raw))
                .collect(),
            )?,
            collective: false,
            persistent: false,
        },
    ])
}

/// The lab's tuning variants of the served inner loop (`native_grouped.cuh`,
/// `Served`). [`Tune::Lean`] keeps the served outputs and admission bit for
/// bit; the others are timing ablations whose outputs are not meaningful.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tune {
    /// Scale bytes by the MMA's byte selectors; NaN codes checked by one
    /// column warp per row group.
    Lean,
    /// No admission or NaN checks.
    NoChecks,
    /// No tensor instructions.
    NoMma,
    /// No global loads after the prologue's stages (and no checks).
    NoGlobal,
    /// MMAs on register operands only: no loads and no checks.
    MmaOnly,
    /// The lean loop without the activation NaN check, behind a
    /// [`nan_scan_symbol`] launch over the same input: the served outputs
    /// and refusals bit for bit.
    NoNan,
    /// The lean loop without any activation admission (NaN codes and the
    /// scale-range product check), behind an [`admission_scan_symbol`]
    /// launch over the same input and scales: the served outputs and
    /// refusals bit for bit.
    Scanned,
}
impl Tune {
    pub const ALL: [Tune; 6] = [
        Tune::Lean,
        Tune::NoChecks,
        Tune::NoMma,
        Tune::NoGlobal,
        Tune::MmaOnly,
        Tune::NoNan,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Tune::Lean => "lean",
            Tune::NoChecks => "no_checks",
            Tune::NoMma => "no_mma",
            Tune::NoGlobal => "no_global",
            Tune::MmaOnly => "mma_only",
            Tune::NoNan => "no_nan",
            Tune::Scanned => "scanned",
        }
    }
    fn policy(self) -> &'static str {
        match self {
            Tune::Lean => "Lean",
            Tune::NoChecks => "NoChecks",
            Tune::NoMma => "NoMma",
            Tune::NoGlobal => "NoGlobal",
            Tune::MmaOnly => "MmaOnly",
            Tune::NoNan => "NoNan",
            Tune::Scanned => "Scanned",
        }
    }
}

pub fn tune_symbol(shape: &Shape, tile: &Tile, tune: Tune, variant: Variant) -> String {
    symbol(shape, tile, Stage::Compute, variant)
        .replace("_tiles_", &format!("_tiles_{}_", tune.name()))
}

/// The `variant` compute entries of `tunes` at `tile`, beside the lab's
/// entries of that variant.
pub fn tune_entries(
    shape: &Shape,
    tile: &Tile,
    tunes: &[Tune],
    variant: Variant,
) -> Result<Vec<EntrySpec>, Refusal> {
    let base_symbol = symbol(shape, tile, Stage::Compute, variant);
    let base = entries(shape, tile, variant != Variant::Serving)?
        .into_iter()
        .find(|e| e.symbol == base_symbol)
        .ok_or(Refusal("native compute entry is missing"))?;
    let from = ">(arena,";
    if base.body.matches(from).count() != 1 {
        return Err(Refusal("native compute entry body changed"));
    }
    Ok(tunes
        .iter()
        .map(|&tune| {
            let mut e = base.clone();
            e.symbol = tune_symbol(shape, tile, tune, variant);
            e.body = base.body.replace(
                from,
                &format!(
                    ",chop::moe::native_grouped::NoCeil,chop::moe::native_grouped::{}>(arena,",
                    tune.policy()
                ),
            );
            e
        })
        .collect())
}

/// The `variant` launches with the compute entry of `tune`; [`Tune::NoNan`]
/// starts with the NaN scan of the input, [`Tune::Scanned`] with the
/// admission scan of the input and its scales.
pub fn tune_launches(
    shape: &Shape,
    tile: &Tile,
    rows: u32,
    b: &Buffers,
    tune: Tune,
    variant: Variant,
) -> Result<Vec<Launch>, Refusal> {
    let mut ops = launches(shape, tile, rows, b, variant)?;
    ops[1].symbol = tune_symbol(shape, tile, tune, variant);
    if tune == Tune::NoNan {
        let mut scan = ops[0].clone();
        scan.symbol = nan_scan_symbol(shape);
        scan.plan.grid = [SM_COUNT * NAN_SCAN_BLOCKS_PER_SM, 1, 1];
        scan.plan.block = [NAN_SCAN_THREADS, 1, 1];
        scan.plan.validate()?;
        scan.arguments = ArgumentList(vec![
            Argument::DeviceBuffer(b.input),
            Argument::DeviceBuffer(b.live_tokens),
        ]);
        scan.arguments.validate()?;
        ops.insert(0, scan);
    }
    if tune == Tune::Scanned {
        let mut scan = ops[0].clone();
        scan.symbol = admission_scan_symbol(shape);
        scan.plan.grid = [SM_COUNT * NAN_SCAN_BLOCKS_PER_SM, 1, 1];
        scan.plan.block = [NAN_SCAN_THREADS, 1, 1];
        scan.plan.validate()?;
        scan.arguments = ArgumentList(vec![
            Argument::DeviceBuffer(b.input),
            Argument::DeviceBuffer(b.activation_scales),
            Argument::DeviceBuffer(b.live_tokens),
        ]);
        scan.arguments.validate()?;
        ops.insert(0, scan);
    }
    Ok(ops)
}

const NAN_SCAN_THREADS: u32 = 512;
const NAN_SCAN_BLOCKS_PER_SM: u32 = 2;

pub fn nan_scan_symbol(shape: &Shape) -> String {
    format!("chop_moe_native_{}_nan_scan_v1", shape.tag())
}

/// The NaN scan of `shape`'s input (`native_grouped::nan_scan`).
pub fn nan_scan_entry(shape: &Shape) -> EntrySpec {
    EntrySpec::new(
        nan_scan_symbol(shape),
        vec![ptr("input", "const uint8_t"), ptr("live", "const int")],
        format!(
            "chop::ptx::griddep_launch_dependents(); chop::ptx::griddep_wait(); \
             chop::moe::native_grouped::nan_scan<{}>(input,live);",
            shape.config()
        ),
        NAN_SCAN_THREADS,
        1,
    )
}

pub fn admission_scan_symbol(shape: &Shape) -> String {
    format!("chop_moe_native_{}_admission_scan_v1", shape.tag())
}

/// The activation admission scan of `shape`'s input and scales
/// (`native_grouped::admission_scan`).
pub fn admission_scan_entry(shape: &Shape) -> EntrySpec {
    EntrySpec::new(
        admission_scan_symbol(shape),
        vec![
            ptr("input", "const uint8_t"),
            ptr("activation_scales", "const uint8_t"),
            ptr("live", "const int"),
        ],
        format!(
            "chop::ptx::griddep_launch_dependents(); chop::ptx::griddep_wait(); \
             chop::moe::native_grouped::admission_scan<{}>(input,activation_scales,live);",
            shape.config()
        ),
        NAN_SCAN_THREADS,
        1,
    )
}

/// Gate/up tiles with the E4M3/32 ceil quantizer in their epilogue
/// (`cuda/moe/native_ceil.cuh`): the serving intermediate quantizer's
/// codes, scales and status, bit for bit, without its separate pass. The
/// tiles write each group as if its token row were clean; the fixup launch
/// after them rewrites tokens whose status is set. `floor_bits` is the
/// quantizer's amax floor (`BlockQuantization::E4M3Ceil32`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ceil {
    pub floor_bits: u32,
}

/// Threads of a fixup CTA, one CTA per token.
pub const FIXUP_THREADS: u32 = 128;

impl Ceil {
    fn check(&self, shape: &Shape, tile: &Tile) -> Result<(), Refusal> {
        tile.check(shape)?;
        if !shape.gate_up
            || !tile.cols(shape).is_multiple_of(32)
            || !shape.n.is_multiple_of(32)
            || u64::from(tile.rows()) * u64::from(tile.cols(shape)) * 2
                > u64::from(tile.stages) * u64::from(stage_bytes(shape, tile))
        {
            return Err(Refusal(
                "the fused ceil needs gate/up tiles of whole groups",
            ));
        }
        Ok(())
    }

    pub fn tiles_symbol(&self, shape: &Shape, tile: &Tile) -> String {
        format!(
            "{}_ceil{:08x}_v1",
            symbol(shape, tile, Stage::Compute, Variant::Serving).trim_end_matches("_v1"),
            self.floor_bits
        )
    }

    pub fn fixup_symbol(&self, shape: &Shape) -> String {
        format!(
            "chop_moe_native_ceil_fixup_w{}_t{}_v1",
            shape.top_k * shape.n,
            shape.capacity
        )
    }

    /// The fused tiles and the fixup.
    pub fn entries(&self, shape: &Shape, tile: &Tile) -> Result<Vec<EntrySpec>, Refusal> {
        self.check(shape, tile)?;
        let mut serving = entries(shape, tile, false)?;
        let tiles = &mut serving[1];
        tiles.symbol = self.tiles_symbol(shape, tile);
        tiles.params.extend([
            ptr("codes", "uint8_t"),
            ptr("scales", "uint8_t"),
            ptr("status", "uint32_t"),
        ]);
        let call = "route_weights,output);";
        assert!(tiles.body.ends_with(call), "the compute call ends the body");
        tiles.body = format!(
            "{}route_weights,output,nullptr,chop::moe::native_ceil::Ceil<{:#x}u>{{codes,scales,status}});",
            tiles.body.strip_suffix(call).expect("checked above"),
            self.floor_bits
        );
        let fixup = EntrySpec::new(
            self.fixup_symbol(shape),
            vec![
                ptr("live", "const int"),
                ptr("status", "const uint32_t"),
                ptr("codes", "uint8_t"),
                ptr("scales", "uint8_t"),
            ],
            format!(
                "chop::moe::native_ceil::fixup<{}>(live,status,codes,scales);",
                shape.top_k * shape.n
            ),
            FIXUP_THREADS,
            1,
        );
        Ok(vec![serving.swap_remove(1), fixup])
    }

    /// The pack, the fused tiles and the fixup of a step of `rows` tokens:
    /// [`launches`]' pack and tile handles, then the quantizer's codes,
    /// scales and status.
    pub fn launches(
        &self,
        shape: &Shape,
        tile: &Tile,
        rows: u32,
        b: &Buffers,
        (codes, scales, status): (u64, u64, u64),
    ) -> Result<Vec<Launch>, Refusal> {
        self.check(shape, tile)?;
        if [codes, scales, status].contains(&0) {
            return Err(Refusal("the fused ceil needs its quantizer's buffers"));
        }
        let mut ops = launches(shape, tile, rows, b, Variant::Serving)?;
        ops[1].symbol = self.tiles_symbol(shape, tile);
        ops[1]
            .arguments
            .0
            .extend([codes, scales, status].map(Argument::DeviceBuffer));
        ops[1].arguments.validate()?;
        let plan = LaunchPlan {
            grid: [rows, 1, 1],
            block: [FIXUP_THREADS, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        let arguments = ArgumentList(
            [b.live_tokens, status, codes, scales]
                .map(Argument::DeviceBuffer)
                .to_vec(),
        );
        arguments.validate()?;
        ops.push(Launch {
            symbol: self.fixup_symbol(shape),
            rows,
            plan,
            arguments,
            collective: false,
            persistent: false,
        });
        Ok(ops)
    }
}

/// One cubin of the fused ceil entries of `instances` (the pack entries
/// stay in the native cubin, which these launches also use).
pub fn ceil_cubin_spec(
    name: &str,
    ceil: &Ceil,
    instances: &[(Shape, Tile)],
) -> Result<CubinSpec, Refusal> {
    let mut entries: Vec<EntrySpec> = Vec::new();
    for (shape, tile) in instances {
        for entry in ceil.entries(shape, tile)? {
            if entries.iter().all(|e| e.symbol != entry.symbol) {
                entries.push(entry);
            }
        }
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: Vec::new(),
        headers: vec!["#include \"moe/native_ceil.cuh\"".into()],
        defines: Vec::new(),
        entries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flash(gate_up: bool) -> Shape {
        if gate_up {
            Shape::new(
                384,
                6,
                5120,
                576,
                true,
                8192,
                Some((0x40e0_0000, 0x40e0_0000)),
            )
            .unwrap()
        } else {
            Shape::new(384, 6, 576, 5120, false, 8192, None).unwrap()
        }
    }

    #[test]
    fn tiles_fit_shared_memory_and_cover_the_columns() {
        for (shape, tiles) in [
            (flash(true), &GATE_UP_TILES[..]),
            (flash(false), &DOWN_TILES[..]),
        ] {
            for tile in tiles {
                tile.check(&shape).unwrap();
                assert!(shared_bytes(&shape, tile) <= MAX_SHARED_BYTES);
                assert!(tile.strips(&shape) * tile.cols(&shape) >= shape.n);
                let lab = if shape.gate_up { 4 } else { 3 };
                assert_eq!(entries(&shape, tile, true).unwrap().len(), lab);
                assert_eq!(entries(&shape, tile, false).unwrap().len(), 2);
            }
        }
        // Gate/up's 576 columns split evenly at 64 and 96; down's 5120 at 128 and 256.
        assert_eq!(GATE_UP_TILES.map(|t| t.strips(&flash(true))), [9, 6, 6]);
        assert_eq!(DOWN_TILES.map(|t| t.strips(&flash(false))), [40, 20]);
        // Down's 18 scale bytes per row take two words when a row is unaligned.
        assert_eq!(
            shared_bytes(&flash(false), &DOWN_TILES[0]),
            3 * (128 * 128 + 128 * 64 + 256 * 8) + 1024 + 16
        );
        assert_eq!(
            shared_bytes(&flash(true), &GATE_UP_TILES[0]),
            3 * (128 * 128 + 128 * 64 + 256 * 4) + 1024 + 16
        );
    }

    #[test]
    fn fused_ceil_extends_the_serving_tiles_and_adds_a_fixup() {
        let ceil = Ceil {
            floor_bits: 1e-4_f32.to_bits(),
        };
        let shape = flash(true);
        let tile = GATE_UP_TILES[0];
        let entries = ceil.entries(&shape, &tile).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(
            entries[0].symbol,
            "chop_moe_native_gateup_e384k6_n576_k5120_t8192_clamp40e0000040e00000_r2w4s3_tiles_ceil38d1b717_v1"
        );
        assert_eq!(entries[0].params.len(), 13);
        assert!(entries[0].body.ends_with(
            "route_weights,output,nullptr,chop::moe::native_ceil::Ceil<0x38d1b717u>{codes,scales,status});"
        ));
        assert_eq!(
            entries[1].symbol,
            "chop_moe_native_ceil_fixup_w3456_t8192_v1"
        );
        // Down tiles and partial groups are refused.
        assert!(ceil.entries(&flash(false), &DOWN_TILES[0]).is_err());
        let small = Shape::new(128, 3, 2048, 80, true, 512, None).unwrap();
        assert!(ceil.entries(&small, &tile).is_err());
        let b = Buffers {
            input: 1,
            weights: 2,
            weight_scales: 3,
            activation_scales: 4,
            offsets: 5,
            sorted_rows: 6,
            padded_rows: 7,
            live_tokens: 8,
            route_weights: 9,
            output: 10,
            counts: 11,
            active: 12,
            scratch: 13,
            raw: 0,
        };
        let served = launches(&shape, &tile, 8192, &b, Variant::Serving).unwrap();
        let ops = ceil.launches(&shape, &tile, 777, &b, (14, 15, 16)).unwrap();
        assert_eq!(ops.len(), 3);
        assert_eq!(ops[0].symbol, served[0].symbol);
        assert_eq!(ops[1].symbol, entries[0].symbol);
        assert_eq!(ops[1].arguments.0[..10], served[1].arguments.0[..]);
        assert_eq!(
            ops[1].arguments.0[10..],
            [14, 15, 16].map(Argument::DeviceBuffer)
        );
        assert_eq!(ops[2].symbol, entries[1].symbol);
        assert_eq!(ops[2].plan.grid, [777, 1, 1]);
        assert_eq!(
            ops[2].arguments.0,
            [8, 16, 14, 15].map(Argument::DeviceBuffer).to_vec()
        );
        assert!(ceil.launches(&shape, &tile, 777, &b, (14, 0, 16)).is_err());
    }

    #[test]
    fn launches_plan_the_queue_and_refuse_missing_buffers() {
        let b = Buffers {
            input: 1,
            weights: 2,
            weight_scales: 3,
            activation_scales: 4,
            offsets: 5,
            sorted_rows: 6,
            padded_rows: 7,
            live_tokens: 8,
            route_weights: 9,
            output: 10,
            counts: 11,
            active: 12,
            scratch: 13,
            raw: 0,
        };
        let shape = flash(true);
        let tile = GATE_UP_TILES[0];
        let ops = launches(&shape, &tile, 8192, &b, Variant::Serving).unwrap();
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].plan.grid, [1, 1, 1]);
        assert_eq!(ops[1].plan.block, [256, 1, 1]);
        assert_eq!(ops[1].plan.grid, [SM_COUNT, 1, 1]);
        assert_eq!(
            ops[1].plan.dynamic_shared_bytes,
            shared_bytes(&shape, &tile)
        );
        // 8192 * 6 / 128 full tiles plus at most one tail per expert.
        assert_eq!(queue_slots(&shape, &tile), 384 + 384);
        assert_eq!(
            workspace_bytes(&shape, &tile),
            ((4 + 3 * 768) * 4_u64).next_multiple_of(256)
        );
        assert!(ops[1].symbol.ends_with("_tiles_v1"));
        assert!(
            launches(&shape, &tile, 8192, &b, Variant::OmitFirst).unwrap()[1]
                .symbol
                .ends_with("_omit_first_v1")
        );
        // The raw variant needs its buffer and an extra argument, and only gate/up has it.
        assert!(launches(&shape, &tile, 8192, &b, Variant::Raw).is_err());
        let raw = Buffers { raw: 14, ..b };
        assert!(launches(&shape, &tile, 8192, &raw, Variant::Serving).is_err());
        assert_eq!(
            launches(&shape, &tile, 8192, &raw, Variant::Raw).unwrap()[1]
                .arguments
                .0
                .len(),
            11
        );
        assert!(launches(&flash(false), &DOWN_TILES[0], 8192, &raw, Variant::Raw).is_err());
        assert!(launches(&shape, &tile, 0, &b, Variant::Serving).is_err());
        assert!(launches(&shape, &tile, 8193, &b, Variant::Serving).is_err());
        for field in 0..13 {
            let mut bad = b;
            let slot = match field {
                0 => &mut bad.input,
                1 => &mut bad.weights,
                2 => &mut bad.weight_scales,
                3 => &mut bad.activation_scales,
                4 => &mut bad.offsets,
                5 => &mut bad.sorted_rows,
                6 => &mut bad.padded_rows,
                7 => &mut bad.live_tokens,
                8 => &mut bad.route_weights,
                9 => &mut bad.output,
                10 => &mut bad.counts,
                11 => &mut bad.active,
                _ => &mut bad.scratch,
            };
            *slot = 0;
            assert!(launches(&shape, &tile, 8192, &bad, Variant::Serving).is_err());
            // Down reads no route weights.
            assert_eq!(
                launches(&flash(false), &DOWN_TILES[0], 8192, &bad, Variant::Serving).is_ok(),
                field == 8
            );
        }
    }

    #[test]
    fn shapes_refuse_their_unsupported_domain() {
        assert!(Shape::new(384, 6, 5120, 576, true, 8192, None).is_ok());
        assert!(Shape::new(384, 6, 5136, 576, true, 8192, None).is_err());
        assert!(Shape::new(384, 6, 5120, 584, true, 8192, None).is_err());
        assert!(Shape::new(384, 6, 576, 5120, false, 8192, Some((1, 1))).is_err());
        assert!(Shape::new(0, 6, 576, 5120, false, 8192, None).is_err());
        assert!(Shape::new(384, 0, 576, 5120, false, 8192, None).is_err());
        let tile = Tile {
            row_warps: 4,
            col_warps: 8,
            stages: 4,
        };
        assert!(tile.check(&flash(true)).is_err());
        assert_ne!(
            symbol(
                &flash(true),
                &GATE_UP_TILES[0],
                Stage::Compute,
                Variant::Serving
            ),
            symbol(
                &flash(true),
                &GATE_UP_TILES[1],
                Stage::Compute,
                Variant::Serving
            )
        );
    }

    #[test]
    fn tune_entries_change_only_the_loop_policy() {
        for (shape, tile, variant) in [
            (flash(true), GATE_UP_TILES[2], Variant::Serving),
            (flash(true), GATE_UP_TILES[2], Variant::Raw),
            (flash(false), DOWN_TILES[0], Variant::Serving),
        ] {
            let lab = entries(&shape, &tile, true).unwrap();
            let base = lab
                .iter()
                .find(|e| e.symbol == symbol(&shape, &tile, Stage::Compute, variant))
                .unwrap();
            let tuned = tune_entries(&shape, &tile, &Tune::ALL, variant).unwrap();
            assert_eq!(tuned.len(), Tune::ALL.len());
            for (entry, tune) in tuned.iter().zip(Tune::ALL) {
                assert_eq!(entry.symbol, tune_symbol(&shape, &tile, tune, variant));
                assert!(lab.iter().all(|e| e.symbol != entry.symbol));
                assert_eq!(entry.params, base.params);
                assert_eq!(
                    entry.body.replace(
                        &format!(
                            ",chop::moe::native_grouped::NoCeil,chop::moe::native_grouped::{}>",
                            tune.policy()
                        ),
                        ">"
                    ),
                    base.body
                );
            }
        }
        assert_eq!(
            tune_symbol(
                &flash(true),
                &GATE_UP_TILES[2],
                Tune::Lean,
                Variant::Serving
            ),
            symbol(
                &flash(true),
                &GATE_UP_TILES[2],
                Stage::Compute,
                Variant::Serving
            )
            .replace("_tiles_v1", "_tiles_lean_v1")
        );
    }
}
