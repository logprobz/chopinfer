//! The fused-weight GEMM kernels of `cuda/gemm/qwen38.cuh`, which serve
//! weights beside the fragment families:
//!
//! - a cluster decode for K in whole 128-element blocks, fusing up to four
//!   weights that read the same activation when the split fits the cluster
//!   and shared memory; its entries are instances of the
//!   shared cluster decode template (`gemm::cluster`);
//! - a short-K kernel for the hyper-connection up projection (K = 320 is not
//!   a whole number of 128-K blocks);
//! - a skinny kernel for outputs of at most 32 rows per weight (in_proj_a
//!   with in_proj_b, and the inject projection), below the 64-row prefill
//!   tile;
//! - the hyper-connection up-mix and its precompute producers.
//!
//! Which entries exist is the serving model's catalog
//! (`crate::ModelCatalog::gemm_fused`); [`catalog`] lists a consumer's.

use super::{SM_COUNT, cluster};
use crate::arithmetic::{self, DeclaredCubin};
use crate::invariance::Invariance;
use crate::{Catalogs, Profile};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, CompileKey, CompileKnobs, DType, GemmQuery, InstantiationSpec,
    LaunchPlan, OpFamily, Refusal, StaticQuery, Target, Validate,
};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Model-supplied geometry and entry name of the hyper-connection tail.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HcGeometry {
    /// One stream's width.
    pub width: u32,
    /// Stream count.
    pub streams: u32,
    /// Bottleneck rank, the up projection's K dimension.
    pub rank: u32,
    /// Entry symbol of the up projection and stream mix.
    pub up_mix_symbol: &'static str,
}

/// Kernel family of an entry in this module's catalog.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    /// Decode for M <= 32: `strips` 16-row strips per CTA, `warp_blocks`
    /// 128-K blocks per warp, one cluster per row tile over K, and the
    /// tokens staged in `passes` halves. Each weight of the entry's segments
    /// writes its own output.
    ClusterDecode {
        strips: u32,
        warp_blocks: u32,
        passes: u32,
    },
    /// Whole K in registers per strip; `tokens` rows staged per CTA, with
    /// (token tile, `warps` strips) items shared evenly over at most one
    /// persistent CTA per SM.
    ShortK { tokens: u32, warps: u32 },
    /// `weights` weights with `rows` live rows each (at most 32, one or two
    /// fragment strips). Graph phases split K into one 256-K chunk per
    /// split; eager prefill phases into `prefill_splits` at every live M.
    Skinny {
        rows: u32,
        weights: u32,
        prefill_splits: u32,
    },
}

#[derive(Clone, Debug)]
pub struct Entry {
    /// The owning catalog's hyper-connection shape, when it has one.
    pub hc: Option<HcGeometry>,
    pub profile: Profile,
    pub symbol: String,
    /// N is the sum of the segments.
    pub query: GemmQuery,
    pub kind: Kind,
    /// Output rows of each weight, in weight order.
    pub segments: Vec<u32>,
    /// Skinny and short-K prefill entries are chunk invariant: their
    /// prefill schedule fixes each row's K order at every live M. Cluster
    /// decode buckets of up to four rows are one-row exact (`entry_for`).
    pub invariance: Invariance,
}

/// Opaque buffers of one launch; one weight per segment. Cluster decode
/// segment s writes `outputs[s]` rows of `output_strides[s]` elements.
/// Short-K writes `[M, N]` rows to `outputs[0]`. Skinny writes every
/// segment into `outputs[0]` rows of `output_strides[0]` elements, segment
/// s at columns `[s * rows, (s + 1) * rows)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Buffers {
    pub weights: [u64; 4],
    pub activation: u64,
    pub outputs: [u64; 4],
    pub output_strides: [u32; 4],
    pub partial: u64,
    pub semaphores: u64,
}

const THREADS: u32 = 256;
const SKINNY_TOKENS: u32 = 32;

/// Fragment strips a skinny entry accumulates: each weight's rows in 16-row
/// strips (`SkinnyTraits::units`).
fn skinny_units(rows: u32, weights: u32) -> u32 {
    weights * rows.div_ceil(16)
}
/// Static shared memory besides the dynamic allocation, with slack.
const STATIC_SHARED: u32 = 1152;

fn token_tile(m_capacity: u32) -> u32 {
    m_capacity.next_power_of_two().max(8)
}

/// 128-K blocks each warp of a cluster decode entry sums, which decide its
/// output bits (`cluster::Shape::arithmetic`), for buckets of at most four
/// tokens and for larger buckets. The larger buckets prefer four strips per
/// CTA to two, which halves the blocks a CTA covers per warp block: one
/// block per warp splits K = 2,560 into five CTAs at two strips but ten at
/// four, more than a cluster holds.
pub type WarpBlocks = (u32, u32);

/// A cluster decode entry: (name, rows of each weight, K, output, SwiGLU,
/// blocks per warp).
pub type ClusterShape<Rows> = (&'static str, Rows, u32, DType, bool, WarpBlocks);

fn bucket_warp_blocks(warp_blocks: WarpBlocks, m_capacity: u32) -> u32 {
    if m_capacity <= 4 {
        warp_blocks.0
    } else {
        warp_blocks.1
    }
}

/// Strips per CTA and token passes for a cluster decode whose warps each
/// sum `warp_blocks` 128-K blocks, or `None` when no split of K fits a
/// cluster of at most eight. Buckets above 4 tokens prefer 4 strips, which
/// share the activation slab (8 strips left `out` at 32 tokens with 120
/// CTAs); a K of fewer blocks than K parts takes more strips; a slab too
/// large for shared memory is staged in two token halves.
pub fn cluster_decode_shape(k: u32, m_capacity: u32, warp_blocks: u32) -> Option<(u32, u32)> {
    if m_capacity > 32 || !k.is_multiple_of(128) {
        return None;
    }
    let blocks = k / 128;
    let tokens = token_tile(m_capacity);
    let preferred = if m_capacity <= 4 { 2 } else { 4 };
    // The preference, fewer strips, then more (a short K needs more).
    let candidates = [preferred, preferred / 2, preferred / 4, 4, 8]
        .into_iter()
        .filter(|&s| s >= 1);
    for passes in [1, 2] {
        if passes == 2 && tokens < 16 {
            break;
        }
        for strips in candidates.clone() {
            let chunk = warp_blocks * (8 / strips);
            if blocks.is_multiple_of(chunk)
                && blocks / chunk <= 8
                && cluster_shared_bytes(tokens, chunk, passes) + STATIC_SHARED <= 101_376
            {
                return Some((strips, passes));
            }
        }
    }
    None
}

/// The activation slab of one pass plus the F32 partial tile.
fn cluster_shared_bytes(tokens: u32, chunk_blocks: u32, passes: u32) -> u32 {
    tokens / passes * (chunk_blocks * 256 + 16) + 8 * 16 * tokens * 4
}

impl Entry {
    pub fn threads(&self) -> u32 {
        match self.kind {
            Kind::ShortK { warps, .. } => warps * 32,
            _ => THREADS,
        }
    }

    /// Fragment strips over all segments.
    pub fn strips(&self) -> u32 {
        self.segments.iter().map(|r| r.div_ceil(16)).sum()
    }

    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        self.query.validate()?;
        if self.segments.is_empty()
            || self.segments.len() > 4
            || self.segments.iter().sum::<u32>() != self.query.n
        {
            return Err(Refusal("GEMM segments do not sum to N"));
        }
        let spec = InstantiationSpec {
            family: OpFamily::Gemm,
            key: CompileKey {
                query: StaticQuery::Gemm(self.query),
                knobs: CompileKnobs {
                    tile_m: match self.kind {
                        Kind::ClusterDecode { strips, .. } => (strips * 16) as u16,
                        _ => 16,
                    },
                    tile_n: match self.kind {
                        Kind::ClusterDecode { .. } => token_tile(self.query.m_capacity) as u16,
                        Kind::ShortK { tokens, .. } => tokens as u16,
                        Kind::Skinny { .. } => SKINNY_TOKENS as u16,
                    },
                    tile_k: match self.kind {
                        Kind::ClusterDecode { .. } => 128,
                        _ => 16,
                    },
                    stages: 1,
                    swizzle: 0,
                    warps: (self.threads() / 32) as u8,
                    split_k_capacity: self.max_splits() as u8,
                },
                dense: None,
                target: Target::Sm120f,
            },
            symbol: self.symbol.clone(),
        };
        spec.validate()?;
        if let Kind::ClusterDecode {
            strips,
            warp_blocks,
            passes,
        } = self.kind
            && (cluster_decode_shape(self.query.k, self.query.m_capacity, warp_blocks)
                != Some((strips, passes))
                || (self.query.epilogues.swiglu.is_some()
                    && (strips < 2 || self.segments.len() != 1)))
        {
            return Err(Refusal("cluster decode shape is not the planned split"));
        }
        Ok(spec)
    }

    fn max_splits(&self) -> u32 {
        match self.kind {
            Kind::ClusterDecode {
                strips,
                warp_blocks,
                ..
            } => self.query.k / 128 / (warp_blocks * (8 / strips)),
            Kind::ShortK { .. } => 1,
            // At least one 256-K chunk per split.
            Kind::Skinny { .. } => self.query.k / 256,
        }
    }

    fn dynamic_shared_bytes(&self) -> u32 {
        let tokens = token_tile(self.query.m_capacity);
        match self.kind {
            Kind::ClusterDecode {
                strips,
                warp_blocks,
                passes,
            } => cluster_shared_bytes(tokens, warp_blocks * (8 / strips), passes),
            Kind::ShortK { tokens, .. } => tokens * (self.query.k * 2 + 16),
            // Two 256-K activation stages, reused for the warp partial sums.
            Kind::Skinny { rows, weights, .. } => (2 * SKINNY_TOKENS * (256 * 2 + 16))
                .max(THREADS / 32 * skinny_units(rows, weights) * 16 * SKINNY_TOKENS * 4),
        }
    }

    /// Tiles and CTAs per tile: row tiles by the cluster, one persistent
    /// range of items, or token tiles by K splits.
    fn schedule(&self, live_m: u32, prefill: bool) -> Result<(u32, u32), Refusal> {
        self.contract()?;
        if live_m == 0 || live_m > self.query.m_capacity {
            return Err(Refusal("live M exceeds prepared GEMM capacity"));
        }
        Ok(match self.kind {
            Kind::ClusterDecode { strips, .. } => {
                (self.strips().div_ceil(strips), self.max_splits())
            }
            // At most one CTA per SM, sharing the items evenly.
            Kind::ShortK { tokens, warps } => (
                1,
                (live_m.div_ceil(tokens) * (self.query.n / (16 * warps))).min(SM_COUNT),
            ),
            // A fixed split count per phase at every live M, so a row's sum
            // does not depend on how many rows share the step: one 256-K
            // chunk per split in graph phases (the count decode buckets up to
            // nine token tiles always took), and the entry's prefill count in
            // eager prefill phases.
            Kind::Skinny { prefill_splits, .. } => (
                live_m.div_ceil(SKINNY_TOKENS),
                if prefill {
                    prefill_splits
                } else {
                    self.max_splits()
                },
            ),
        })
    }

    /// Cluster decode plans from its capacity; the others size their grid
    /// and workspace for live M. For graph phases (decode, verify).
    pub fn plan(&self, live_m: u32) -> Result<LaunchPlan, Refusal> {
        self.plan_phase(live_m, false)
    }

    /// `plan` for an eager prefill phase.
    pub fn prefill_plan(&self, live_m: u32) -> Result<LaunchPlan, Refusal> {
        self.plan_phase(live_m, true)
    }

    fn plan_phase(&self, live_m: u32, prefill: bool) -> Result<LaunchPlan, Refusal> {
        let (tiles, per_tile) = self.schedule(live_m, prefill)?;
        let tokens = token_tile(self.query.m_capacity);
        let (workspace_bytes, cluster) = match self.kind {
            Kind::ClusterDecode { .. } if per_tile > 1 => {
                (u64::from(tiles * per_tile * 8 * 16 * tokens) * 4, per_tile)
            }
            Kind::Skinny { rows, weights, .. } if per_tile > 1 => (
                u64::from(tiles * per_tile * skinny_units(rows, weights) * 16 * SKINNY_TOKENS) * 4,
                1,
            ),
            _ => (0, 1),
        };
        let plan = LaunchPlan {
            grid: [tiles * per_tile, 1, 1],
            block: [self.threads(), 1, 1],
            dynamic_shared_bytes: self.dynamic_shared_bytes(),
            workspace_bytes,
            cluster: [cluster, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// The largest workspace `plan` or `prefill_plan` asks for at any live
    /// M.
    pub fn workspace_capacity(&self) -> Result<u64, Refusal> {
        let tile = match self.kind {
            Kind::Skinny { .. } => SKINNY_TOKENS,
            _ => self.query.m_capacity,
        };
        (1..=self.query.m_capacity.div_ceil(tile))
            .flat_map(|tiles| [(tiles, false), (tiles, true)])
            .map(|(tiles, prefill)| {
                self.plan_phase((tiles * tile).min(self.query.m_capacity), prefill)
                    .map(|plan| plan.workspace_bytes)
            })
            .try_fold(0, |most, bytes| bytes.map(|bytes| most.max(bytes)))
    }

    /// Zeroed semaphore words the caller must provide (one per token tile).
    pub fn semaphore_words(&self) -> u32 {
        match self.kind {
            Kind::Skinny { .. } => self.query.m_capacity.div_ceil(SKINNY_TOKENS),
            _ => 0,
        }
    }

    /// The arguments of `plan`.
    pub fn arguments(&self, live_m: u32, b: Buffers) -> Result<ArgumentList, Refusal> {
        self.arguments_phase(live_m, b, false)
    }

    /// The arguments of `prefill_plan`.
    pub fn prefill_arguments(&self, live_m: u32, b: Buffers) -> Result<ArgumentList, Refusal> {
        self.arguments_phase(live_m, b, true)
    }

    fn arguments_phase(
        &self,
        live_m: u32,
        b: Buffers,
        prefill: bool,
    ) -> Result<ArgumentList, Refusal> {
        let (_, per_tile) = self.schedule(live_m, prefill)?;
        let output_f32 = u32::from(self.query.output == DType::F32);
        let values = match self.kind {
            Kind::ClusterDecode { .. } => {
                let count = self.segments.len();
                let halves = 1 + u32::from(self.query.epilogues.swiglu.is_some());
                if (0..count).any(|s| b.output_strides[s] < self.segments[s] / halves) {
                    return Err(Refusal("output stride is shorter than its rows"));
                }
                let mut values = Vec::new();
                values.extend((0..count).map(|s| Argument::DeviceBuffer(b.weights[s])));
                values.extend((0..count).map(|s| Argument::DeviceBuffer(b.outputs[s])));
                values.extend((0..count).map(|s| Argument::U32(b.output_strides[s])));
                values.extend([
                    Argument::DeviceBuffer(b.activation),
                    Argument::DeviceBuffer(b.partial),
                    Argument::U32(live_m),
                    Argument::U32(output_f32),
                ]);
                values
            }
            Kind::ShortK { .. } => vec![
                Argument::DeviceBuffer(b.weights[0]),
                Argument::DeviceBuffer(b.activation),
                Argument::DeviceBuffer(b.outputs[0]),
                Argument::U32(live_m),
                Argument::U32(self.query.n),
                Argument::U32(output_f32),
            ],
            Kind::Skinny { weights, .. } => {
                let stride = b.output_strides[0];
                if stride < self.query.n || !stride.is_multiple_of(2) {
                    return Err(Refusal("skinny output stride is shorter than its columns"));
                }
                vec![
                    Argument::DeviceBuffer(b.weights[0]),
                    Argument::DeviceBuffer(b.weights[usize::from(weights == 2)]),
                    Argument::DeviceBuffer(b.activation),
                    Argument::DeviceBuffer(b.outputs[0]),
                    Argument::DeviceBuffer(b.partial),
                    Argument::DeviceBuffer(b.semaphores),
                    Argument::U32(live_m),
                    Argument::U32(stride),
                    Argument::U32(per_tile),
                    Argument::U32(output_f32),
                ]
            }
        };
        let args = ArgumentList(values);
        args.validate()?;
        Ok(args)
    }

    fn params(&self) -> Vec<EntryParam> {
        let ptr = |name: &str, ty: &str| EntryParam::DevicePointer {
            name: name.into(),
            cxx_type: ty.into(),
        };
        let scalar = |name: &str| EntryParam::U32 { name: name.into() };
        match self.kind {
            Kind::ClusterDecode { .. } => {
                let count = self.segments.len();
                let mut params = Vec::new();
                params.extend((0..count).map(|s| ptr(&format!("weight{s}"), "const uint8_t")));
                params.extend((0..count).map(|s| ptr(&format!("output{s}"), "void")));
                params.extend((0..count).map(|s| scalar(&format!("stride{s}"))));
                params.extend([
                    ptr("activation", "const uint16_t"),
                    ptr("partial", "float"),
                    scalar("live_m"),
                    scalar("output_f32"),
                ]);
                params
            }
            Kind::ShortK { .. } => vec![
                ptr("weight", "const uint8_t"),
                ptr("activation", "const uint16_t"),
                ptr("output", "void"),
                scalar("live_m"),
                scalar("n"),
                scalar("output_f32"),
            ],
            Kind::Skinny { .. } => vec![
                ptr("weight0", "const uint8_t"),
                ptr("weight1", "const uint8_t"),
                ptr("activation", "const uint16_t"),
                ptr("output", "void"),
                ptr("partial", "float"),
                ptr("semaphores", "uint32_t"),
                scalar("live_m"),
                scalar("output_stride"),
                scalar("splits"),
                scalar("output_f32"),
            ],
        }
    }

    fn body(&self) -> String {
        let k = self.query.k;
        match self.kind {
            Kind::ClusterDecode { .. } => {
                self.template()
                    .expect("cluster decode entries run the shared template")
                    .1
            }
            Kind::ShortK { tokens, warps } => format!(
                "chop::gemm::qwen38::run_short_k<{k}, {tokens}, {warps}>(weight, activation, output, live_m, n, output_f32);"
            ),
            Kind::Skinny { rows, weights, .. } => format!(
                "chop::gemm::qwen38::run_skinny<{k}, {rows}, {weights}>(weight0, weight1, activation, output, partial, semaphores, live_m, output_stride, splits, output_f32);"
            ),
        }
    }

    /// A cluster decode entry on the shared template (`gemm::cluster`,
    /// `cuda/gemm/cluster_decode.cuh`): its shape and a body with this
    /// entry's ABI, bitwise the arithmetic of `run_cluster_decode`.
    pub fn template(&self) -> Option<(cluster::Shape, String)> {
        self.template_with(None)
    }

    fn template_with(
        &self,
        epilogue: Option<(String, String)>,
    ) -> Option<(cluster::Shape, String)> {
        let Kind::ClusterDecode {
            strips,
            warp_blocks,
            passes,
        } = self.kind
        else {
            return None;
        };
        let count = self.segments.len();
        let shape = cluster::Shape {
            weights: cluster::Weights::Bf16,
            token_tile: token_tile(self.query.m_capacity),
            k_blocks: self.query.k / 128,
            strips,
            warp_blocks,
            passes,
            segments: count as u32,
            swiglu: self.query.epilogues.swiglu.is_some(),
        };
        let list = |f: &dyn Fn(usize) -> String| (0..count).map(f).collect::<Vec<_>>();
        let args = cluster::Arguments {
            weights: list(&|s| format!("weight{s}")),
            outputs: list(&|s| format!("output{s}")),
            rows: list(&|s| format!("{}u", self.segments[s])),
            strides: list(&|s| format!("stride{s}")),
            activation: "activation".into(),
            workspace: "partial".into(),
            live_m: "live_m".into(),
            output_f32: "output_f32".into(),
            scales: None,
            epilogue,
        };
        Some((shape, shape.call(&args)))
    }

    pub fn entry_spec(&self) -> EntrySpec {
        EntrySpec::new(
            self.symbol.clone(),
            self.params(),
            self.body(),
            self.threads(),
            // Two cluster decode CTAs share each SM.
            if matches!(self.kind, Kind::ClusterDecode { .. }) {
                2
            } else {
                1
            },
        )
    }
}

impl Entry {
    /// The entry `chop_gemm_{tag}_{name}_v1` of `profile`, with its declared
    /// invariance. Panics where the kernel refuses it.
    pub fn new(
        profile: Profile,
        name: &str,
        query: GemmQuery,
        kind: Kind,
        segments: &[u32],
        invariance: Invariance,
    ) -> Entry {
        let entry = Entry {
            hc: None,
            profile,
            symbol: format!("chop_gemm_{}_{name}_v1", profile.tag()),
            query,
            kind,
            segments: segments.to_vec(),
            invariance,
        };
        entry.contract().expect("GEMM catalog entry validates");
        entry
    }
}

/// A cluster decode entry's kind at its declared blocks per warp.
pub fn cluster_kind(k: u32, m_capacity: u32, warp_blocks: WarpBlocks) -> Kind {
    let warp_blocks = bucket_warp_blocks(warp_blocks, m_capacity);
    let (strips, passes) = cluster_decode_shape(k, m_capacity, warp_blocks)
        .expect("every cluster decode K has a split at its blocks per warp");
    Kind::ClusterDecode {
        strips,
        warp_blocks,
        passes,
    }
}

/// The models' entries in `catalogs`, in order.
pub fn catalog(catalogs: &Catalogs) -> Vec<Entry> {
    catalogs
        .models()
        .iter()
        .flat_map(|model| (model.gemm_fused)())
        .collect()
}

/// The decode hyper-connection input tail (`cuda/gemm/qwen38_hc.cuh`): the
/// bottleneck, the up projection and the stream mix in one launch, bitwise
/// as `hc_bottleneck`, `hc_up_decode` and `hc_mix` in turn. CTA x owns
/// columns `[16x, 16x + 16)` of the four streams for every decode row.
const UP_MIX_TOKENS: u32 = 32;

/// The catalog geometry of a named hyper-connection mix.
pub fn hc_geometry(catalogs: &Catalogs, symbol: &str) -> Option<HcGeometry> {
    catalog(catalogs)
        .iter()
        .filter_map(|entry| entry.hc)
        .find(|hc| hc.up_mix_symbol == symbol)
}

/// Rows up to `UP_MIX_TOKENS`; the live count is a device scalar.
pub fn hc_up_mix_plan(hc: HcGeometry, rows: u32) -> Result<LaunchPlan, Refusal> {
    if rows == 0 || rows > UP_MIX_TOKENS {
        return Err(Refusal(
            "hyper-connection up mix serves decode buckets only",
        ));
    }
    let activation = UP_MIX_TOKENS * (hc.rank * 2 + 16);
    let terms = hc.streams * 16 * UP_MIX_TOKENS * 4;
    let plan = LaunchPlan {
        grid: [hc.width / 16, 1, 1],
        block: [hc.streams * 32, 1, 1],
        dynamic_shared_bytes: activation + terms,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
}

/// `weight` is input_mix_weight_up in fragments, `down` the `[rows, hc.rank]`
/// down projection, `normed` the `[rows, hc.streams, hc.width]` normalized streams and
/// `output` the `[rows, hc.width]` mix.
pub fn hc_up_mix_arguments(
    weight: u64,
    down: u64,
    normed: u64,
    output: u64,
    live_rows: u64,
) -> Result<ArgumentList, Refusal> {
    let args = ArgumentList(
        [weight, down, normed, output, live_rows]
            .into_iter()
            .map(Argument::DeviceBuffer)
            .collect(),
    );
    args.validate()?;
    Ok(args)
}

fn hc_up_mix_spec(hc: HcGeometry) -> EntrySpec {
    let ptr = |name: &str, ty: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    };
    EntrySpec::new(
        hc.up_mix_symbol.into(),
        vec![
            ptr("weight", "const uint8_t"),
            ptr("down", "const uint16_t"),
            ptr("normed", "const uint16_t"),
            ptr("output", "uint16_t"),
            ptr("live_rows", "const uint32_t"),
        ],
        format!(
            "chop::gemm::qwen38::run_hc_up_mix<{UP_MIX_TOKENS}>(weight, down, normed, output, live_rows);"
        ),
        hc.streams * 32,
        1,
    )
}

/// Separate module: the producer writes a rounded HC bottleneck once.
pub const HC_PRECOMPUTE_CUBIN: &str = "gemm_qwen38_hc_precompute_sm120";
pub const HC_PRECOMPUTED_MIX: &str = "chop_gemm_qwen38_hc_up_mix_precomputed_v1";
/// Opt-in layout candidate; the v1 entry remains available for exact comparison.
pub const HC_PADDED_MIX: &str = "chop_gemm_qwen38_hc_up_mix_precomputed_v2";

pub fn hc_padded_mix_plan(hc: HcGeometry, rows: u32) -> Result<LaunchPlan, Refusal> {
    let mut plan = hc_up_mix_plan(hc, rows)?;
    plan.dynamic_shared_bytes += hc.streams * 16 * 4;
    plan.validate()?;
    Ok(plan)
}

/// The qualified producer retains the base entry's complete ABI and appends
/// one `[capacity, hc.rank]` BF16 bottleneck output. Only the 16/32-row HC
/// entries, which the model's base profile holds for every rank count
/// (their shapes do not depend on it).
pub fn hc_precompute_spec(entry: &Entry) -> Result<EntrySpec, Refusal> {
    let name = hc_precompute_name(entry)?;
    let m = entry.query.m_capacity;
    let mut spec = entry.entry_spec();
    spec.symbol = format!(
        "chop_gemm_{}_hc_{name}_precompute_c{m}_v1",
        entry.profile.tag()
    );
    spec.params.push(EntryParam::DevicePointer {
        name: "bottleneck".into(),
        cxx_type: "uint16_t".into(),
    });
    spec.body = hc_precompute_template(entry)?.1;
    Ok(spec)
}

/// `in` or `down` for a qualified HC producer of a model's base profile
/// (TP4's in-tree) at 16 or 32 rows.
fn hc_precompute_name(entry: &Entry) -> Result<&'static str, Refusal> {
    let hc = entry
        .hc
        .ok_or(Refusal("HC producer requires catalog geometry"))?;
    let m = entry.query.m_capacity;
    let inject = entry.segments == [hc.rank, hc.streams];
    let name = if inject { "in" } else { "down" };
    if !matches!(m, 16 | 32)
        || !entry.profile.is_base()
        || entry.symbol != format!("chop_gemm_{}_hc_{name}_c{m}_v1", entry.profile.tag())
        || entry.query.k != hc.width * hc.streams
        || entry.query.output != DType::Bf16
        || entry.query.epilogues.swiglu.is_some()
        || entry.kind
            != (Kind::ClusterDecode {
                strips: 1,
                warp_blocks: 2,
                passes: m / 16,
            })
        || (!inject && entry.segments != [hc.rank])
    {
        return Err(Refusal("HC precompute requires a qualified TP4 producer"));
    }
    Ok(name)
}

/// A qualified HC precompute producer (`hc_precompute_spec`) on the shared
/// cluster decode template, with Qwen's bottleneck epilogue
/// (`cuda/gemm/qwen38_hc_bottleneck.cuh`): its shape and body.
pub fn hc_precompute_template(entry: &Entry) -> Result<(cluster::Shape, String), Refusal> {
    hc_precompute_name(entry)?;
    let store = "chop::gemm::qwen38::HcBottleneckStore";
    Ok(entry
        .template_with(Some((store.into(), format!("{store}{{bottleneck}}"))))
        .expect("HC precompute producers are cluster decode entries"))
}

/// Shared memory configuration for the precompute module's symbols.
pub fn hc_precompute_shared(catalogs: &Catalogs, symbol: &str) -> Option<u32> {
    for entry in catalog(catalogs) {
        if hc_precompute_name(&entry).is_err() {
            continue;
        }
        let hc = entry.hc?;
        if symbol == format!("chop_gemm_{}_hc_up_mix_precomputed_v2", entry.profile.tag()) {
            return Some(
                hc_padded_mix_plan(hc, 32)
                    .expect("padded mix")
                    .dynamic_shared_bytes,
            );
        }
        if symbol == format!("chop_gemm_{}_hc_up_mix_precomputed_v1", entry.profile.tag()) {
            return Some(
                hc_up_mix_plan(hc, 32)
                    .expect("qualified mix")
                    .dynamic_shared_bytes,
            );
        }
    }
    if !symbol.contains("_hc_") || !symbol.contains("precompute") {
        return None;
    }
    catalog(catalogs).iter().find_map(|entry| {
        let spec = hc_precompute_spec(entry).ok()?;
        (spec.symbol == symbol).then(|| {
            entry
                .plan(entry.query.m_capacity)
                .unwrap()
                .dynamic_shared_bytes
        })
    })
}

// Panic on generator drift: filtering refused catalog entries must never hide
// a missing rewrite and silently omit a qualified producer from the cubin.
fn replace_hc_body(body: &str, from: &str, to: &str) -> String {
    assert_eq!(
        body.matches(from).count(),
        1,
        "HC spec rewrite must match once: {from}"
    );
    body.replace(from, to)
}

fn hc_precompute_cubin(catalogs: &Catalogs, owner: Profile) -> CubinSpec {
    let mut entries: Vec<_> = catalog(catalogs)
        .iter()
        .filter(|entry| entry.profile == owner)
        .filter_map(|entry| hc_precompute_spec(entry).ok())
        .collect();
    let hc = catalog(catalogs)
        .iter()
        .filter(|entry| entry.profile == owner)
        .find_map(|entry| hc_precompute_name(entry).ok().and(entry.hc))
        .expect("precompute cubin has catalog geometry");
    let mut mix = hc_up_mix_spec(hc);
    mix.symbol = format!("chop_gemm_{}_hc_up_mix_precomputed_v1", owner.tag());
    mix.body = replace_hc_body(&mix.body, "run_hc_up_mix<32>", "run_hc_up_mix<32, true>");
    let mut padded = mix.clone();
    padded.symbol = format!("chop_gemm_{}_hc_up_mix_precomputed_v2", owner.tag());
    padded.body = replace_hc_body(
        &padded.body,
        "run_hc_up_mix<32, true>",
        "run_hc_up_mix<32, true, true>",
    );
    entries.push(mix);
    entries.push(padded);
    let mut spec = CubinSpec {
        name: format!("gemm_{}_hc_precompute_sm120", owner.tag()),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec![
            "#include \"gemm/qwen38.cuh\"".into(),
            "#include \"gemm/qwen38_hc.cuh\"".into(),
            cluster::HEADER.into(),
            cluster::HC_EPILOGUE_HEADER.into(),
        ],
        defines: vec![],
        entries,
    };
    catalogs.fused_sources(owner, &mut spec);
    spec
}

/// The cubin of this module's catalog per profile with entries,
/// `gemm_{tag}_own_sm120`, in catalog order. `None` builds them all.
pub fn cubin_specs(catalogs: &Catalogs, profile: Option<Profile>) -> Vec<CubinSpec> {
    declared_cubin_specs(catalogs, profile)
        .into_iter()
        .map(DeclaredCubin::into_spec)
        .collect()
}

/// [`cubin_specs`] with each catalog entry's and HC precompute producer's
/// declared arithmetic ([`arithmetic::fused_gemm`]); the HC mixes compile
/// theirs in.
pub fn declared_cubin_specs(catalogs: &Catalogs, profile: Option<Profile>) -> Vec<DeclaredCubin> {
    let all = catalog(catalogs);
    let mut declared = BTreeMap::new();
    for entry in all
        .iter()
        .filter(|entry| profile.is_none_or(|selected| selected == entry.profile))
    {
        let arithmetic = arithmetic::fused_gemm(entry);
        if let Ok(spec) = hc_precompute_spec(entry) {
            declared.insert(spec.symbol, format!("hc-precompute; {arithmetic}"));
        }
        declared.insert(entry.symbol.clone(), arithmetic);
    }
    let mut owners = Vec::new();
    for entry in &all {
        if !owners.contains(&entry.profile) {
            owners.push(entry.profile);
        }
    }
    let specs = owners
        .iter()
        .copied()
        .filter(|&owner| profile.is_none_or(|selected| selected == owner))
        .map(|owner| {
            let mut entries: Vec<_> = all
                .iter()
                .filter(|entry| entry.profile == owner)
                .map(Entry::entry_spec)
                .collect();
            let mut headers = vec![
                "#include \"gemm/qwen38.cuh\"".into(),
                cluster::HEADER.into(),
            ];
            if let Some(hc) = all.iter().find_map(|entry| {
                (entry.profile == owner && owner.is_base())
                    .then_some(entry.hc)
                    .flatten()
            }) {
                entries.push(hc_up_mix_spec(hc));
                headers.push("#include \"gemm/qwen38_hc.cuh\"".into());
            }
            let mut spec = CubinSpec {
                name: format!("gemm_{}_own_sm120", owner.tag()),
                sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
                include_dirs: vec![],
                headers,
                defines: vec![],
                entries,
            };
            catalogs.fused_sources(owner, &mut spec);
            spec
        })
        .chain(
            owners
                .iter()
                .copied()
                .filter(|&owner| {
                    profile.is_none_or(|p| p == owner)
                        && all.iter().any(|entry| {
                            entry.profile == owner && hc_precompute_spec(entry).is_ok()
                        })
                })
                .map(|owner| hc_precompute_cubin(catalogs, owner)),
        )
        .collect();
    DeclaredCubin::all(specs, declared).expect("fused GEMM entries declare their arithmetic")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_mix_preserves_geometry_and_reserves_every_term() {
        let hc = HcGeometry {
            width: 1024,
            streams: 4,
            rank: 128,
            up_mix_symbol: "chop_hc_mix_test",
        };
        assert!(hc_padded_mix_plan(hc, 0).is_err());
        assert!(hc_padded_mix_plan(hc, 33).is_err());
        for rows in 1..=32 {
            let base = hc_up_mix_plan(hc, rows).unwrap();
            let mut padded = hc_padded_mix_plan(hc, rows).unwrap();
            assert_eq!(padded.dynamic_shared_bytes, 32 * 272 + 4 * 16 * 33 * 4);
            assert_eq!(base.grid, [64, 1, 1]);
            padded.dynamic_shared_bytes = base.dynamic_shared_bytes;
            assert_eq!(padded, base);
        }
        assert_eq!(hc_precompute_shared(&Catalogs::base(), HC_PADDED_MIX), None);
    }

    #[test]
    #[should_panic(expected = "HC spec rewrite must match once")]
    fn hc_spec_rewrite_refuses_missing_pattern() {
        replace_hc_body("changed template", "expected template", "replacement");
    }

    #[test]
    #[should_panic(expected = "HC spec rewrite must match once")]
    fn hc_spec_rewrite_refuses_duplicate_pattern() {
        replace_hc_body("template template", "template", "replacement");
    }

    #[test]
    fn cluster_decode_splits_every_declared_k() {
        // (K, M, strips, warp blocks, passes)
        for (k, m, strips, blocks, passes) in [
            (2560, 1, 2, 1, 1),
            (2560, 8, 4, 2, 1),
            (2560, 32, 4, 2, 1),
            (1536, 1, 2, 1, 1),
            (1536, 8, 4, 1, 1),
            (1536, 32, 4, 1, 1),
            (10240, 1, 1, 2, 1),
            (10240, 16, 1, 2, 1),
            // A 16-block chunk for 32 tokens overflows shared memory.
            (10240, 32, 1, 2, 2),
            // Two blocks need at least four strips.
            (256, 1, 4, 1, 1),
        ] {
            assert_eq!(
                cluster_decode_shape(k, m, blocks),
                Some((strips, passes)),
                "K={k} M={m}"
            );
        }
        // One block per warp at eight tokens takes two strips, and K =
        // 10,240 does not split into eight one-block chunks.
        assert_eq!(cluster_decode_shape(2560, 8, 1), Some((2, 1)));
        assert_eq!(cluster_decode_shape(10240, 1, 1), None);
    }
}
