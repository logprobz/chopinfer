//! The chained expert engine (`cuda/moe/chained_experts.cuh`,
//! `cuda/moe/route_local.cuh`, `cuda/moe/route_local_scan.cuh`,
//! `cuda/moe/routed_sum_add.cuh`, `cuda/moe/swiglu_clamp.cuh`): grouped and
//! dense SwiGLU MLP projections with 128-K accumulation chains and an
//! optional clamped SwiGLU epilogue (`kernel_core::gated_mlp`), sigmoid-bias
//! routes for any count of local experts, and the F32 route sum with one
//! BF16 cast or an F32 output, or with one F32 row added. The served dense
//! instances are listed by shape in [`catalog`]; the routed experts', split
//! over the ranks, in [`split`].
//!
//! Weights keep their checkpoint layouts, per expert and row-major: FP8 E4M3
//! codes `[rows, K]` with F32 block scales `[rows / 128, K / 128]`; NVFP4
//! E2M1 pairs `[rows, K / 2]` (even K in the low nibble) with UE4M3 scales
//! `[rows, K / 16]` and F32 globals (gate and up per expert for gate/up, one
//! per expert for down); MXFP8 E4M3 `[rows, K]` with UE8M0 `[rows, K / 32]`;
//! or BF16 `[rows, K]`. Gate/up rows are the N gate rows, then the N up rows.
//!
//! Declared exactness ([`Projection::invariance`],
//! [`LocalRoute::invariance`], [`RoutedSum::invariance`]): every kernel is
//! token-local in its outputs.
//!
//! - A projection's output element of a route (or token, dense) adds each
//!   128-K block as one MMA chain from zero over ascending K and folds the
//!   blocks in K order (`fmaf` with the FP8 residual, `__fadd_rn`
//!   otherwise), with no split K and no atomics; it reads only the route's
//!   own input row and its expert's weights and writes at the route's
//!   index. The row count, the route padding, the row tile, the slot a
//!   route takes and the other tokens' routes only change which CTA and
//!   which MMA column computes it. Every schedule computes the same chains
//!   (`Schedule`), so a projection claims one-row bits up to its token
//!   capacity, any chunk split and token independence, its one-row bits
//!   being the column schedule's for one token.
//! - The routes select each token's experts from its own logits (strict
//!   greater, or equal with the lower expert first; a NaN never wins) and
//!   build integer metadata without atomics; the metadata depends on every
//!   route by design, so the backend takes their claims about the routed
//!   operation. They claim one-row bits, any chunk split, token
//!   independence and `Ties::LowerIndex` up to their capacity.
//! - A routed sum adds a token's local routes in top-k order from +0 (and,
//!   in its add form, one addend row after them): one-row bits, any chunk
//!   split and token independence.
//!
//! Lab mutants (`add_entry_with` other than `none`) declare nothing.

use super::ptr;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    ActivationFormat, Argument, ArgumentList, DType, LaunchPlan, Refusal, RoutedCapacity,
    ScaleMode, SwigluColumns, Validate, WeightFormat,
};
use kernel_core::gated_mlp::{
    GatedMlpProjection, GatedMlpProjectionQuery, RouteWeightPoint, SwigluClamp,
};
use kernel_core::properties::{Invariance, Ties};
use std::path::PathBuf;

pub mod formats;
pub mod quantized;
pub mod row_bias;
pub mod split;

/// Threads of a projection CTA: four warps of 16 output columns.
pub const THREADS: u32 = 128;
/// Output columns of a projection CTA.
pub const TILE_N: u32 = 64;
/// Threads of a route-sum CTA and of the prefill route kernels.
pub const ROW_THREADS: u32 = 256;
/// Most tokens of the one-CTA decode route: a warp per token.
pub const DECODE_ROUTE_MAX_TOKENS: u32 = 32;
/// Row tiles the projection compiles: decode (16) and prefill (64) routed
/// rows, or 32.
pub const ROW_TILES: [u32; 3] = [16, 32, 64];
/// Output columns of a [`Schedule::BlockWarps`] CTA.
pub const BLOCK_WARPS_TILE_N: u32 = 16;
/// 128-K blocks a [`Schedule::BlockWarps`] round spreads over its warps.
pub const BLOCK_WARPS: u32 = 4;
/// Threads of a [`Schedule::TilePairs`] CTA: four warps per output tile.
pub const TILE_PAIR_THREADS: u32 = 256;
/// Output columns of a [`Schedule::TilePairs`] column pair.
pub const TILE_PAIR_COLUMNS: u32 = 2 * TILE_N;
/// Output columns of a [`Schedule::WarpStreams`] item.
pub const WARP_STREAMS_TILE_N: u32 = 16;
/// K bytes each warp of a [`Schedule::WarpStreams`] CTA streams per row and
/// stage: one whole 128-K block of FP8 codes.
pub const WARP_STREAMS_STAGE_K: u32 = 128;
/// The persistent CTAs of a persistent [`Schedule::WarpStreams`] launch
/// ([`WarpStreamLayout::persistent`]): two on each of the 188 SMs of an RTX
/// PRO 6000 (the static shared memory fits twice per SM). The grid never
/// changes an output bit; another SM count only leaves CTAs waiting or SMs
/// idle.
pub const WARP_STREAMS_CTAS: u32 = 2 * 188;

/// How a [`Schedule::WarpStreams`] projection stages and spreads its work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WarpStreamLayout {
    /// Stages of each warp's `cp.async` ring.
    pub stages: u32,
    /// Rows XOR-swizzled by 16-byte chunk instead of padded by 16 bytes.
    pub swizzle: bool,
    /// [`WARP_STREAMS_CTAS`] persistent CTAs over the items, else one item
    /// per CTA of a `[columns / 16, tile capacity]` grid.
    pub persistent: bool,
}

/// Threads of a [`Schedule::DecodedTiles`] CTA: a loader warp, three decode
/// warps and eight math warps.
pub const DECODED_TILE_THREADS: u32 = 384;
/// Tokens of a [`Schedule::DecodedTiles`] CTA.
pub const DECODED_TILE_TOKENS: u32 = 128;
/// Dynamic shared bytes of a [`Schedule::DecodedTiles`] CTA: three
/// activation stages, two raw code and two decoded weight stages, and 1024
/// bytes of alignment (`decoded_tiles::Layout`).
pub const DECODED_TILE_SHARED_BYTES: u32 = 99_328;

/// Static shared bytes of a [`Schedule::DecodedTiles`] CTA over `k`, as
/// the compiler reports them: the 1,024 bytes it reserves beside mbarriers,
/// then the ten mbarriers and the tile's gate and up F32 block scales per
/// 128-K block (`chained_decoded.cuh`), padded to the 128-byte alignment of
/// the dynamic stages after them (1,408 bytes at K = 4096).
pub const fn decoded_tile_static_bytes(k: u32) -> u32 {
    1024 + (10 * 8 + 2 * 4 * k.div_ceil(128)).next_multiple_of(128)
}

/// Refuses a [`Schedule::DecodedTiles`] CTA over `k` whose static bytes and
/// [`DECODED_TILE_SHARED_BYTES`] exceed SM120's shared memory.
fn decoded_tile_fits(k: u32) -> Result<(), Refusal> {
    if decoded_tile_static_bytes(k) + DECODED_TILE_SHARED_BYTES
        > crate::backend::SHARED_MEMORY_BYTES
    {
        return Err(Refusal(
            "chained decoded tiles' static block scales and stages exceed SM120 shared memory",
        ));
    }
    Ok(())
}

/// How a projection spreads its work over CTAs. All compute every output
/// element with the same operands, chains and K-ordered adds, so their
/// outputs are bitwise equal; only the CTA count and the load schedule
/// differ.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Schedule {
    /// `chained_experts.cuh`: four warps on 64 output columns, K in order.
    Columns,
    /// `chained_block_warps.cuh` (`chained_block_warps_fp4.cuh` for NVFP4):
    /// four warps on the 128-K blocks of 16 output columns, with a
    /// `cp.async` stage ring; 16-row tiles over FP8 128x128, NVFP4 or BF16
    /// weights. Its grid covers the most padded rows the live routes can
    /// need, not every expert's padding.
    BlockWarps,
    /// `chained_tile_pairs.cuh`: eight warps on two 64 x 64 output tiles of
    /// the 64-row route padding that share an operand. An expert with at
    /// least `min_pair_tiles` row tiles (0: none) pairs its row tiles under
    /// one weight tile; other row tiles pair adjacent column tiles under one
    /// activation tile. Grouped FP8 128x128 projections; the column
    /// schedule's grid with half its column tiles.
    TilePairs { min_pair_tiles: u32 },
    /// `chained_warp_streams.cuh`: block warps' items (16-row tile x 16
    /// output columns), one per CTA or over [`WARP_STREAMS_CTAS`]
    /// persistent CTAs, each warp streaming its own 128-K blocks' weight
    /// rows through its own `cp.async` ring ([`Projection::warp_stream_layout`]);
    /// the CTA meets only at the four-block chain handovers. Grouped FP8
    /// 128x128 projections at the 16-row decode padding.
    WarpStreams,
    /// `chained_decoded.cuh`: 128 tokens by 64 output columns per CTA, a
    /// loader warp and three decode warps feeding eight math warps through
    /// mbarrier rings, each weight code decoded once per CTA. Dense FP8
    /// 128x128 gate/up at the prefill row tile; the grid covers the token
    /// capacity in 128-token tiles.
    DecodedTiles,
}

/// The cubin name suffix and the header of a schedule's projections (NVFP4
/// block warps take `chained_block_warps_fp4.cuh`; [`projection_cubin_specs`]).
pub fn schedule_cubin(schedule: Schedule) -> (&'static str, &'static str) {
    match schedule {
        Schedule::Columns => ("", "chained_experts.cuh"),
        Schedule::BlockWarps => ("_bw4", "chained_block_warps.cuh"),
        Schedule::TilePairs { .. } => ("_tp", "chained_tile_pairs.cuh"),
        Schedule::WarpStreams => ("_ws", "chained_warp_streams.cuh"),
        Schedule::DecodedTiles => ("_dt", "chained_decoded.cuh"),
    }
}

/// A weight format the engine decodes into exact BF16 operands.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Weights {
    Fp8Block128,
    Nvfp4,
    Mxfp8,
    Bf16,
}

impl Weights {
    pub fn of(format: WeightFormat) -> Result<Self, Refusal> {
        match format {
            WeightFormat::Fp8Block128x128(ScaleMode::ExactF32) => Ok(Self::Fp8Block128),
            WeightFormat::Nvfp4 => Ok(Self::Nvfp4),
            WeightFormat::Mxfp8 => Ok(Self::Mxfp8),
            WeightFormat::Bf16 => Ok(Self::Bf16),
            _ => Err(Refusal(
                "the chained expert engine takes FP8 128x128 F32, NVFP4, MXFP8 or BF16 weights",
            )),
        }
    }

    fn cxx(self) -> &'static str {
        match self {
            Self::Fp8Block128 => "fp8_block128",
            Self::Nvfp4 => "nvfp4",
            Self::Mxfp8 => "mxfp8",
            Self::Bf16 => "bf16",
        }
    }

    /// The short tag in cubin and symbol names.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Fp8Block128 => "fp8block",
            Self::Nvfp4 => "nvfp4",
            Self::Mxfp8 => "mxfp8",
            Self::Bf16 => "bf16",
        }
    }
}

/// Admission of the F32 scales a projection reads (FP8 block scales and
/// NVFP4 globals): every scale is finite and not negative; zero is allowed
/// and makes its block contribute exactly zero (`split_scale` in
/// `chained_experts.cuh`).
pub fn admit_scales(scales: &[f32]) -> Result<(), Refusal> {
    if scales.iter().any(|&s| !s.is_finite() || s < 0.0) {
        return Err(Refusal(
            "chained F32 weight scales must be finite and not negative",
        ));
    }
    Ok(())
}

/// One compiled projection: a gated MLP projection contract, the row tile
/// (and route padding) it reads, and the token capacity its grid covers.
#[derive(Clone, Debug, PartialEq)]
pub struct Projection {
    pub symbol: String,
    pub query: GatedMlpProjectionQuery,
    pub row_tile: u32,
    pub tokens_capacity: u32,
    pub schedule: Schedule,
}

impl Projection {
    /// Checks the contract against what the engine computes.
    pub fn new(
        query: GatedMlpProjectionQuery,
        row_tile: u32,
        tokens_capacity: u32,
    ) -> Result<Self, Refusal> {
        query.validate()?;
        let gemm = query.gemm;
        let weights = Weights::of(gemm.weight)?;
        if gemm.activation != ActivationFormat::Bf16 {
            return Err(Refusal("the chained expert engine reads BF16 activations"));
        }
        let e = gemm.epilogues;
        if e.residual_add || e.bias || e.gelu.is_some() || e.output_quantization.is_some() {
            return Err(Refusal(
                "the chained expert engine has no residual, bias, GELU or output quantization",
            ));
        }
        if query.route_weight == Some(RouteWeightPoint::BeforeDown) {
            return Err(Refusal(
                "the chained expert engine applies route weights after the down projection",
            ));
        }
        let gate_up = query.projection == GatedMlpProjection::GateUp;
        if gate_up && e.swiglu != Some(SwigluColumns::ConcatenatedHalves) {
            return Err(Refusal(
                "chained gate/up reads concatenated gate and up halves",
            ));
        }
        let grouped_down = gemm.grouped.is_some() && !gate_up;
        let output = if grouped_down {
            DType::F32
        } else {
            DType::Bf16
        };
        if gemm.output != output {
            return Err(Refusal(
                "chained grouped down writes F32 weighted routes; other projections write BF16",
            ));
        }
        let n = if gate_up { gemm.n / 2 } else { gemm.n };
        if !n.is_multiple_of(TILE_N)
            || !gemm.k.is_multiple_of(128)
            || (weights == Weights::Fp8Block128 && !n.is_multiple_of(128))
        {
            return Err(Refusal(
                "chained projections need 64-column tiles, whole 128-K blocks and whole FP8 row blocks",
            ));
        }
        if !ROW_TILES.contains(&row_tile) || tokens_capacity == 0 {
            return Err(Refusal("chained row tile or token capacity unsupported"));
        }
        if let Some(grouped) = gemm.grouped {
            let routes = tokens_capacity.checked_mul(grouped.top_k);
            if routes.is_none_or(|routes| routes > gemm.m_capacity)
                || !matches!(grouped.capacity, RoutedCapacity::TotalRows(_))
            {
                return Err(Refusal(
                    "chained routed rows exceed the query's capacity or are per-expert",
                ));
            }
        } else if tokens_capacity > gemm.m_capacity {
            return Err(Refusal("chained dense rows exceed the query's capacity"));
        }
        let symbol = Self::symbol_of(&query, weights, row_tile, Schedule::Columns);
        Ok(Self {
            symbol,
            query,
            row_tile,
            tokens_capacity,
            schedule: Schedule::Columns,
        })
    }

    /// The projection's declared exactness (see the module docs): one-row
    /// bits up to its token capacity, any chunk split and token
    /// independence, under every schedule and row tile.
    pub fn invariance(&self) -> Invariance {
        Invariance::any_rows(self.tokens_capacity)
    }

    /// This projection under [`Schedule::BlockWarps`]: 16-row tiles, FP8
    /// 128x128, NVFP4 or BF16 weights and K in whole rounds of four 128-K
    /// blocks. NVFP4 runs `chained_block_warps_fp4.cuh`.
    pub fn block_warps(&self) -> Result<Self, Refusal> {
        if !matches!(
            self.weights(),
            Weights::Fp8Block128 | Weights::Nvfp4 | Weights::Bf16
        ) || self.row_tile != 16
            || !(self.query.gemm.k / 128).is_multiple_of(BLOCK_WARPS)
        {
            return Err(Refusal(
                "chained block warps take 16-row tiles over FP8 128x128, NVFP4 or BF16 weights and K in 512s",
            ));
        }
        Ok(Self {
            symbol: Self::symbol_of(
                &self.query,
                self.weights(),
                self.row_tile,
                Schedule::BlockWarps,
            ),
            schedule: Schedule::BlockWarps,
            ..self.clone()
        })
    }

    /// This projection under [`Schedule::TilePairs`]: grouped, 64-row
    /// tiles, FP8 128x128 weights and whole column pairs.
    pub fn tile_pairs(&self, min_pair_tiles: u32) -> Result<Self, Refusal> {
        if self.weights() != Weights::Fp8Block128
            || self.query.gemm.grouped.is_none()
            || self.row_tile != 64
            || !self.columns().is_multiple_of(TILE_PAIR_COLUMNS)
        {
            return Err(Refusal(
                "chained tile pairs take grouped 64-row tiles over FP8 128x128 weights and 128-column pairs",
            ));
        }
        let schedule = Schedule::TilePairs { min_pair_tiles };
        Ok(Self {
            symbol: Self::symbol_of(&self.query, self.weights(), self.row_tile, schedule),
            schedule,
            ..self.clone()
        })
    }

    /// This projection under [`Schedule::WarpStreams`]: grouped, 16-row
    /// tiles, FP8 128x128 weights and K in whole rounds of four 128-K
    /// blocks, with at least two rounds and enough steps after an item's
    /// first round to fill its [`WarpStreamLayout`]'s ring (the kernel's
    /// `Total - subs >= Stages - 1`): K >= 1024 for gate/up's two stages,
    /// K >= 2048 for down's four.
    pub fn warp_streams(&self) -> Result<Self, Refusal> {
        let blocks = self.query.gemm.k / 128;
        let rounds = blocks / BLOCK_WARPS;
        let subs = 128 / WARP_STREAMS_STAGE_K;
        let stages = self.warp_stream_layout().stages;
        if self.weights() != Weights::Fp8Block128
            || self.query.gemm.grouped.is_none()
            || self.row_tile != 16
            || !self.query.gemm.k.is_multiple_of(128)
            || !blocks.is_multiple_of(BLOCK_WARPS)
            || rounds < 2
            || (rounds - 1) * subs < stages - 1
        {
            return Err(Refusal(
                "chained warp streams take grouped 16-row tiles over FP8 128x128 weights and K in whole 512s, two or more and enough to fill the layout's ring",
            ));
        }
        Ok(Self {
            symbol: Self::symbol_of(
                &self.query,
                self.weights(),
                self.row_tile,
                Schedule::WarpStreams,
            ),
            schedule: Schedule::WarpStreams,
            ..self.clone()
        })
    }

    /// The [`WarpStreamLayout`] of a [`Schedule::WarpStreams`] projection,
    /// as measured fastest (PERF-B20): gate/up two padded stages (45,056
    /// bytes of rings and chains) one item per CTA, down four swizzled
    /// stages (36,864 bytes) over persistent CTAs.
    pub fn warp_stream_layout(&self) -> WarpStreamLayout {
        WarpStreamLayout {
            stages: if self.gate_up() { 2 } else { 4 },
            swizzle: !self.gate_up(),
            persistent: !self.gate_up(),
        }
    }

    /// This projection under [`Schedule::DecodedTiles`]: dense gate/up at
    /// the 64-row prefill tile over FP8 128x128 weights.
    pub fn decoded_tiles(&self) -> Result<Self, Refusal> {
        if self.weights() != Weights::Fp8Block128
            || self.query.gemm.grouped.is_some()
            || !self.gate_up()
            || self.row_tile != 64
        {
            return Err(Refusal(
                "chained decoded tiles take dense 64-row gate/up over FP8 128x128 weights",
            ));
        }
        decoded_tile_fits(self.query.gemm.k)?;
        Ok(Self {
            symbol: Self::symbol_of(
                &self.query,
                self.weights(),
                self.row_tile,
                Schedule::DecodedTiles,
            ),
            schedule: Schedule::DecodedTiles,
            ..self.clone()
        })
    }

    fn symbol_of(
        query: &GatedMlpProjectionQuery,
        weights: Weights,
        row_tile: u32,
        schedule: Schedule,
    ) -> String {
        let gemm = query.gemm;
        let projection = match query.projection {
            GatedMlpProjection::GateUp => "gateup",
            GatedMlpProjection::Down => "down",
        };
        let experts = gemm.grouped.map_or("dense".to_string(), |g| {
            format!("e{}k{}", g.experts, g.top_k)
        });
        let clamp = query.swiglu_clamp.map_or(String::new(), |c| {
            format!("_clamp{:08x}{:08x}", c.gate_max_bits, c.up_limit_bits)
        });
        let schedule = match schedule {
            Schedule::Columns => String::new(),
            Schedule::BlockWarps => "_bw4".into(),
            Schedule::TilePairs { min_pair_tiles } => format!("_tp{min_pair_tiles}"),
            Schedule::WarpStreams => "_ws".into(),
            Schedule::DecodedTiles => "_dt".into(),
        };
        format!(
            "chop_moe_chain_{projection}_{}_{experts}_n{}_k{}_r{row_tile}{clamp}{schedule}_v1",
            weights.tag(),
            gemm.n,
            gemm.k
        )
    }

    pub fn weights(&self) -> Weights {
        Weights::of(self.query.gemm.weight).expect("checked at construction")
    }

    pub fn gate_up(&self) -> bool {
        self.query.projection == GatedMlpProjection::GateUp
    }

    /// Output columns: the intermediate for gate/up, else N.
    pub fn columns(&self) -> u32 {
        let n = self.query.gemm.n;
        if self.gate_up() { n / 2 } else { n }
    }

    /// Padded route rows of a grouped launch at `tokens`, or token rows when
    /// dense, rounded to whole row tiles: each expert's range pads to a
    /// multiple of the row tile.
    pub fn padded_rows(&self, tokens: u32) -> u32 {
        let rows = match self.query.gemm.grouped {
            Some(g) => tokens * g.top_k + g.experts * (self.row_tile - 1),
            None => tokens,
        };
        rows.next_multiple_of(self.row_tile)
    }

    /// The most padded rows a grouped launch at `tokens` can read: each
    /// expert with a route pads its range by less than a row tile, and at
    /// most `tokens * top_k` experts have one. Token rows when dense.
    pub fn live_padded_rows(&self, tokens: u32) -> u32 {
        let rows = match self.query.gemm.grouped {
            Some(g) => {
                let routes = tokens * g.top_k;
                routes + routes.min(g.experts) * (self.row_tile - 1)
            }
            None => tokens,
        };
        rows.next_multiple_of(self.row_tile)
    }

    /// Output columns of one CTA's column tiles (a column pair under
    /// [`Schedule::TilePairs`]).
    pub fn tile_columns(&self) -> u32 {
        match self.schedule {
            Schedule::Columns | Schedule::DecodedTiles => TILE_N,
            Schedule::BlockWarps => BLOCK_WARPS_TILE_N,
            Schedule::TilePairs { .. } => TILE_PAIR_COLUMNS,
            Schedule::WarpStreams => WARP_STREAMS_TILE_N,
        }
    }

    /// Threads of one CTA.
    pub fn threads(&self) -> u32 {
        match self.schedule {
            Schedule::Columns | Schedule::BlockWarps | Schedule::WarpStreams => THREADS,
            Schedule::TilePairs { .. } => TILE_PAIR_THREADS,
            Schedule::DecodedTiles => DECODED_TILE_THREADS,
        }
    }

    /// The capacity-sized launch for a prepared token capacity; the live
    /// counts are read on the device.
    pub fn plan(&self, tokens_capacity: u32) -> Result<LaunchPlan, Refusal> {
        if tokens_capacity == 0 || tokens_capacity > self.tokens_capacity {
            return Err(Refusal("chained launch exceeds the entry's token capacity"));
        }
        let rows = match self.schedule {
            Schedule::Columns | Schedule::TilePairs { .. } | Schedule::DecodedTiles => {
                self.padded_rows(tokens_capacity)
            }
            Schedule::BlockWarps | Schedule::WarpStreams => self.live_padded_rows(tokens_capacity),
        };
        let grid = match self.schedule {
            Schedule::WarpStreams if self.warp_stream_layout().persistent => {
                [WARP_STREAMS_CTAS, 1, 1]
            }
            Schedule::WarpStreams => [
                self.columns() / self.tile_columns(),
                rows / self.row_tile,
                1,
            ],
            // 128-token tiles over the token capacity.
            Schedule::DecodedTiles => {
                decoded_tile_fits(self.query.gemm.k)?;
                [
                    tokens_capacity.div_ceil(DECODED_TILE_TOKENS),
                    self.columns() / self.tile_columns(),
                    1,
                ]
            }
            _ => [
                rows / self.row_tile,
                self.columns() / self.tile_columns(),
                1,
            ],
        };
        let plan = LaunchPlan {
            grid,
            block: [self.threads(), 1, 1],
            dynamic_shared_bytes: if self.schedule == Schedule::DecodedTiles {
                DECODED_TILE_SHARED_BYTES
            } else {
                0
            },
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// Arguments in entry order: input, weight, block scales, local scales,
    /// globals, offsets, sorted rows, padded row count, live tokens, route
    /// weights and output. A zero handle passes a null pointer, which only
    /// the parameters a format or mode does not read may take.
    pub fn arguments(&self, handles: [u64; 11]) -> Result<ArgumentList, Refusal> {
        let weights = self.weights();
        let grouped = self.query.gemm.grouped.is_some();
        let required = [
            true,
            true,
            weights == Weights::Fp8Block128,
            matches!(weights, Weights::Nvfp4 | Weights::Mxfp8),
            weights == Weights::Nvfp4,
            grouped,
            grouped,
            grouped,
            !grouped,
            grouped && !self.gate_up(),
            true,
        ];
        let mut values = Vec::with_capacity(handles.len());
        for (handle, needed) in handles.into_iter().zip(required) {
            values.push(match (handle, needed) {
                (0, true) => return Err(Refusal("chained projection argument missing")),
                (0, false) => Argument::NullDevicePointer,
                (handle, _) => Argument::DeviceBuffer(handle),
            });
        }
        let args = ArgumentList(values);
        args.validate()?;
        Ok(args)
    }

    pub fn entry(&self) -> EntrySpec {
        let gemm = self.query.gemm;
        let (experts, top_k) = gemm.grouped.map_or((1, 1), |g| (g.experts, g.top_k));
        let clamp = match self.query.swiglu_clamp {
            Some(SwigluClamp {
                gate_max_bits,
                up_limit_bits,
            }) => format!(
                "chop::moe::clamp::SwigluClamp<true,{gate_max_bits:#x}u,{up_limit_bits:#x}u>"
            ),
            None => "chop::moe::clamp::SwigluClamp<false,0u,0u>".into(),
        };
        let projection = if self.gate_up() { "gate_up" } else { "down" };
        let (run, extra) = match (self.schedule, self.weights()) {
            (Schedule::Columns, _) => ("chop::moe::chained::run", String::new()),
            (Schedule::BlockWarps, Weights::Nvfp4) => {
                ("chop::moe::block_warps_fp4::run", String::new())
            }
            (Schedule::BlockWarps, _) => ("chop::moe::block_warps::run", String::new()),
            (Schedule::TilePairs { min_pair_tiles }, _) => {
                ("chop::moe::tile_pairs::run", format!(",{min_pair_tiles}"))
            }
            (Schedule::WarpStreams, _) => {
                let WarpStreamLayout {
                    stages,
                    swizzle,
                    persistent,
                } = self.warp_stream_layout();
                let max_tiles = self.live_padded_rows(self.tokens_capacity) / self.row_tile;
                (
                    "chop::moe::warp_streams::run",
                    format!(",{WARP_STREAMS_STAGE_K},{stages},{swizzle},{persistent},{max_tiles}"),
                )
            }
            (Schedule::DecodedTiles, _) => ("chop::moe::decoded_tiles::run", String::new()),
        };
        let body = format!(
            "{run}<chop::moe::chained::Config<chop::moe::chained::Weights::{},chop::moe::chained::Projection::{projection},{experts},{top_k},{},{},{},{},{clamp}>{extra}>(input,weight,block_scales,local_scales,globals,offsets,sorted_rows,num_padded_rows,live_tokens,route_weights,output);",
            self.weights().cxx(),
            self.columns(),
            gemm.k,
            self.row_tile,
            gemm.grouped.is_some(),
        );
        EntrySpec::new(
            self.symbol.clone(),
            vec![
                ptr("input", "const uint16_t"),
                ptr("weight", "const uint8_t"),
                ptr("block_scales", "const float"),
                ptr("local_scales", "const uint8_t"),
                ptr("globals", "const float"),
                ptr("offsets", "const int32_t"),
                ptr("sorted_rows", "const int32_t"),
                ptr("num_padded_rows", "const int32_t"),
                ptr("live_tokens", "const int32_t"),
                ptr("route_weights", "const float"),
                ptr("output", "void"),
            ],
            body,
            self.threads(),
            1,
        )
    }
}

/// Sigmoid-bias routes of `experts` global experts, top `top_k`, filtered to
/// `local_experts` per rank (any count) with route padding `pad`, for up to
/// `capacity` tokens; the scaling factor is a launch scalar.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LocalRoute {
    pub experts: u32,
    pub local_experts: u32,
    pub top_k: u32,
    pub capacity: u32,
    pub pad: u32,
}

/// The kernels of a [`LocalRoute`]: the one-CTA decode route
/// (`route_local_scan.cuh`: warp-parallel counts, offsets and scatter), or
/// prefill's row selection, per-expert count and per-expert scatter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RouteKernel {
    Decode,
    Select,
    Count,
    Scatter,
}

impl LocalRoute {
    pub fn validate(self) -> Result<(), Refusal> {
        let Self {
            experts,
            local_experts,
            top_k,
            capacity,
            pad,
        } = self;
        if experts == 0
            || !experts.is_multiple_of(32)
            || local_experts == 0
            || local_experts > 1024
            || !experts.is_multiple_of(local_experts)
            || !(1..=32).contains(&top_k)
            || top_k > experts
            || !ROW_TILES.contains(&pad)
            || capacity == 0
        {
            return Err(Refusal("unsupported local route geometry"));
        }
        Ok(())
    }

    /// The kernels a launch at this capacity runs, in order.
    pub fn kernels(self) -> &'static [RouteKernel] {
        if self.capacity <= DECODE_ROUTE_MAX_TOKENS {
            &[RouteKernel::Decode]
        } else {
            &[
                RouteKernel::Select,
                RouteKernel::Count,
                RouteKernel::Scatter,
            ]
        }
    }

    /// The declared exactness of each of the route's kernels (see the module
    /// docs): one-row bits, any chunk split, token independence and the
    /// lower-expert tie rule up to its capacity. The kernels that write
    /// route metadata (the decode route, the count and the scatter: expert
    /// counts, offsets and sorted routes depend on every token by design)
    /// make these claims about the routed operation's rows
    /// (`Scope::RoutedOperation`); the selection writes each token's ids and
    /// weights alone.
    pub fn invariance(self, kernel: RouteKernel) -> Invariance {
        let claims = match self.validate() {
            Ok(()) => Invariance::any_rows(self.capacity).with_ties(Ties::LowerIndex),
            Err(_) => return Invariance::NONE,
        };
        match kernel {
            RouteKernel::Select => claims,
            RouteKernel::Decode | RouteKernel::Count | RouteKernel::Scatter => {
                claims.of_routed_operation()
            }
        }
    }

    /// Route rows the metadata needs: every route plus each expert's padding.
    pub fn sorted_rows_capacity(self) -> u32 {
        self.capacity * self.top_k + self.local_experts * (self.pad - 1)
    }

    pub fn plan(self, kernel: RouteKernel, live: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        if live > self.capacity || !self.kernels().contains(&kernel) {
            return Err(Refusal(
                "local route kernel or live count outside its capacity",
            ));
        }
        let (grid, block) = match kernel {
            RouteKernel::Decode => (
                1,
                (self.capacity * 32).max(self.local_experts.next_multiple_of(32)),
            ),
            RouteKernel::Select => (self.capacity.div_ceil(8), ROW_THREADS),
            RouteKernel::Count | RouteKernel::Scatter => (self.local_experts, ROW_THREADS),
        };
        let plan = LaunchPlan {
            grid: [grid, 1, 1],
            block: [block, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn symbol(self, prefix: &str, kernel: RouteKernel) -> String {
        let name = match kernel {
            RouteKernel::Decode => "decode",
            RouteKernel::Select => "select",
            RouteKernel::Count => "count",
            RouteKernel::Scatter => "scatter",
        };
        format!(
            "{prefix}_{name}_e{}l{}k{}_t{}_p{}_v1",
            self.experts, self.local_experts, self.top_k, self.capacity, self.pad
        )
    }

    pub fn entries(self, prefix: &str) -> Result<Vec<EntrySpec>, Refusal> {
        self.validate()?;
        let Self {
            experts,
            local_experts,
            top_k,
            capacity,
            pad,
        } = self;
        let score = "chop::moe::SigmoidBias{bias,__uint_as_float(scale_bits)}";
        let u32 = |name: &str| EntryParam::U32 { name: name.into() };
        self.kernels()
            .iter()
            .map(|&kernel| {
                let block = self.plan(kernel, 0)?.block[0];
                let (params, body) = match kernel {
                    RouteKernel::Decode => (
                        vec![
                            ptr("logits", "const float"),
                            ptr("bias", "const float"),
                            ptr("ids", "int32_t"),
                            ptr("weights", "float"),
                            ptr("counts", "int32_t"),
                            ptr("offsets", "int32_t"),
                            ptr("sorted_rows", "int32_t"),
                            ptr("num_padded_rows", "int32_t"),
                            ptr("live_tokens", "const int32_t"),
                            ptr("active_experts", "int32_t"),
                            u32("rank"),
                            u32("scale_bits"),
                        ],
                        format!("chop::moe::local::route_decode_scan<{experts},{top_k},{local_experts},{capacity},{pad}>(logits,ids,weights,counts,offsets,sorted_rows,num_padded_rows,live_tokens,active_experts,rank,{score});"),
                    ),
                    RouteKernel::Select => (
                        vec![
                            ptr("logits", "const float"),
                            ptr("bias", "const float"),
                            ptr("ids", "int32_t"),
                            ptr("weights", "float"),
                            ptr("live_tokens", "const int32_t"),
                            u32("scale_bits"),
                        ],
                        format!("chop::moe::local::route_select_rows<{experts},{top_k},{capacity}>(logits,ids,weights,live_tokens,{score});"),
                    ),
                    RouteKernel::Count => (
                        vec![
                            ptr("ids", "const int32_t"),
                            ptr("counts", "int32_t"),
                            ptr("live_tokens", "const int32_t"),
                            u32("rank"),
                        ],
                        format!("chop::moe::local::route_count_local<{experts},{top_k},{local_experts},{capacity}>(ids,counts,live_tokens,rank);"),
                    ),
                    RouteKernel::Scatter => (
                        vec![
                            ptr("ids", "const int32_t"),
                            ptr("counts", "const int32_t"),
                            ptr("offsets", "int32_t"),
                            ptr("sorted_rows", "int32_t"),
                            ptr("num_padded_rows", "int32_t"),
                            ptr("live_tokens", "const int32_t"),
                            ptr("active_experts", "int32_t"),
                            u32("rank"),
                        ],
                        format!("chop::moe::local::route_scatter_local<{experts},{top_k},{local_experts},{capacity},{pad}>(ids,counts,offsets,sorted_rows,num_padded_rows,live_tokens,active_experts,rank);"),
                    ),
                };
                Ok(EntrySpec::new(self.symbol(prefix, kernel), params, body, block, 1))
            })
            .collect()
    }
}

/// What a [`RoutedSum`] writes per token: the F32 sum cast to BF16 once, or
/// the F32 sum itself for a later F32 add (a shared expert's residual
/// epilogue, then the cross-rank F32 reduction).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoutedSumOutput {
    Bf16,
    F32,
}

/// A rank's F32 weighted routes summed per token in top-k order
/// (`chained::routed_sum`) and written in `output`'s form. Both forms run
/// the same F32 adds, so the BF16 row is the F32 row cast to BF16.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoutedSum {
    pub top_k: u32,
    pub width: u32,
    pub local_experts: u32,
    pub capacity: u32,
    pub output: RoutedSumOutput,
}

impl RoutedSum {
    /// This sum in the other output form.
    pub fn with_output(self, output: RoutedSumOutput) -> Self {
        Self { output, ..self }
    }

    /// Output bytes per token.
    pub fn output_row_bytes(self) -> u64 {
        let element = match self.output {
            RoutedSumOutput::Bf16 => 2,
            RoutedSumOutput::F32 => 4,
        };
        u64::from(self.width) * element
    }

    /// The launch for a prepared token capacity up to the entry's: one CTA
    /// per token. The entry does not depend on the capacity, so one compiled
    /// sum serves every smaller capacity; the live count is read on the
    /// device.
    pub fn plan(self, tokens_capacity: u32) -> Result<LaunchPlan, Refusal> {
        if self.top_k == 0
            || self.local_experts == 0
            || !self.width.is_multiple_of(4 * ROW_THREADS)
            || tokens_capacity == 0
            || tokens_capacity > self.capacity
        {
            return Err(Refusal("unsupported routed sum geometry or token capacity"));
        }
        let plan = LaunchPlan {
            grid: [tokens_capacity, 1, 1],
            block: [ROW_THREADS, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// The declared exactness of the sum and of its add form (see the
    /// module docs): one-row bits, any chunk split and token independence
    /// up to its capacity.
    pub fn invariance(self) -> Invariance {
        match self.plan(self.capacity) {
            Ok(_) => Invariance::any_rows(self.capacity),
            Err(_) => Invariance::NONE,
        }
    }

    /// `{prefix}_sum_...` for the BF16 form, `{prefix}_sum_f32_...` for F32.
    pub fn symbol(self, prefix: &str) -> String {
        let output = match self.output {
            RoutedSumOutput::Bf16 => "",
            RoutedSumOutput::F32 => "_f32",
        };
        format!(
            "{prefix}_sum{output}_k{}_w{}_l{}_t{}_v1",
            self.top_k, self.width, self.local_experts, self.capacity
        )
    }

    pub fn entry(self, prefix: &str) -> Result<EntrySpec, Refusal> {
        self.plan(self.capacity)?;
        Ok(EntrySpec::new(
            self.symbol(prefix),
            vec![
                ptr("routed", "const float"),
                ptr("ids", "const int32_t"),
                ptr("live_tokens", "const int32_t"),
                EntryParam::U32 {
                    name: "rank".into(),
                },
                ptr(
                    "output",
                    match self.output {
                        RoutedSumOutput::Bf16 => "uint16_t",
                        RoutedSumOutput::F32 => "float",
                    },
                ),
            ],
            format!(
                "chop::moe::chained::routed_sum<{},{},{}>(routed,ids,live_tokens,rank,output);",
                self.top_k, self.width, self.local_experts
            ),
            ROW_THREADS,
            1,
        ))
    }

    /// `{prefix}_sum_add_f32_...`: the F32 sum's symbol for
    /// [`Self::add_entry`].
    pub fn add_symbol(self, prefix: &str) -> String {
        format!(
            "{prefix}_sum_add_f32_k{}_w{}_l{}_t{}_v1",
            self.top_k, self.width, self.local_experts, self.capacity
        )
    }

    /// The F32 sum followed by one F32 add of an `addend` row per token
    /// (`chained::routed_sum_add`, routed_sum_add.cuh): the sum's adds are
    /// [`Self::entry`]'s, then `sum + addend`, written as F32. Same launch
    /// as the sum ([`Self::plan`]). Only the F32 form adds.
    pub fn add_entry(self, prefix: &str) -> Result<EntrySpec, Refusal> {
        self.add_entry_with(prefix, "none")
    }

    /// [`Self::add_entry`] instantiated with the `chained::SumAddMutation`
    /// named `mutation` (kernel-lab's deliberate defects; `none` is the
    /// served kernel), under [`Self::add_symbol`] with `_{mutation}` before
    /// the version unless it is `none`.
    pub fn add_entry_with(self, prefix: &str, mutation: &str) -> Result<EntrySpec, Refusal> {
        if self.output != RoutedSumOutput::F32 {
            return Err(Refusal("only the F32 routed sum adds a row"));
        }
        self.plan(self.capacity)?;
        let mut symbol = self.add_symbol(prefix);
        if mutation != "none" {
            symbol = symbol.replace("_v1", &format!("_{mutation}_v1"));
        }
        Ok(EntrySpec::new(
            symbol,
            vec![
                ptr("routed", "const float"),
                ptr("ids", "const int32_t"),
                ptr("live_tokens", "const int32_t"),
                EntryParam::U32 {
                    name: "rank".into(),
                },
                ptr("addend", "const float"),
                ptr("output", "float"),
            ],
            format!(
                "chop::moe::chained::routed_sum_add<{},{},{},chop::moe::chained::SumAddMutation::{mutation}>(routed,ids,live_tokens,rank,addend,output);",
                self.top_k, self.width, self.local_experts
            ),
            ROW_THREADS,
            1,
        ))
    }
}

/// The cubin `name` of the F32 routed sums that add a row
/// ([`RoutedSum::add_entry`] of each of `sums` under `prefix`), from
/// routed_sum_add.cuh alone.
pub fn routed_sum_add_cubin_spec(
    name: String,
    prefix: &str,
    sums: &[RoutedSum],
) -> Result<CubinSpec, Refusal> {
    let entries = sums
        .iter()
        .map(|sum| sum.add_entry(prefix))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(spec(name, "routed_sum_add.cuh", entries))
}

fn spec(name: String, header: &str, entries: Vec<EntrySpec>) -> CubinSpec {
    CubinSpec {
        name,
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: Vec::new(),
        headers: vec![format!("#include \"moe/{header}\"")],
        defines: Vec::new(),
        entries,
    }
}

/// The projections' cubins, one per weight format and schedule present,
/// named `{prefix}_{format}{suffix}` with the schedule's suffix
/// ([`schedule_cubin`]: none for [`Schedule::Columns`], `_bw4` for
/// [`Schedule::BlockWarps`], `_tp` for [`Schedule::TilePairs`], `_ws` for
/// [`Schedule::WarpStreams`]).
pub fn projection_cubin_specs(prefix: &str, projections: &[Projection]) -> Vec<CubinSpec> {
    let mut specs: Vec<CubinSpec> = Vec::new();
    for projection in projections {
        let (suffix, header) = match (projection.schedule, projection.weights()) {
            (Schedule::BlockWarps, Weights::Nvfp4) => ("_bw4", "chained_block_warps_fp4.cuh"),
            (schedule, _) => schedule_cubin(schedule),
        };
        let name = format!("{prefix}_{}{suffix}", projection.weights().tag());
        match specs.iter_mut().find(|s| s.name == name) {
            Some(spec) => spec.entries.push(projection.entry()),
            None => specs.push(spec(name, header, vec![projection.entry()])),
        }
    }
    specs
}

/// The routes' and route sums' cubin `{prefix}_route`.
pub fn route_cubin_spec(
    prefix: &str,
    routes: &[LocalRoute],
    sums: &[RoutedSum],
) -> Result<CubinSpec, Refusal> {
    let mut entries = Vec::new();
    for route in routes {
        entries.extend(route.entries(prefix)?);
    }
    for sum in sums {
        entries.push(sum.entry(prefix)?);
    }
    let mut spec = spec(format!("{prefix}_route"), "route_local_scan.cuh", entries);
    spec.headers
        .push("#include \"moe/chained_experts.cuh\"".into());
    Ok(spec)
}

/// The served shapes of one rank of a sharded top-k MoE with a clamped
/// SwiGLU, its dense MLP layers and its shared expert.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ServedShapes {
    pub experts: u32,
    pub local_experts: u32,
    pub top_k: u32,
    pub hidden: u32,
    /// Expert intermediate of one rank: the whole expert's when experts are
    /// distributed, a quarter of it when each is split over four ranks
    /// ([`split::SHAPES`]).
    pub expert_intermediate: u32,
    /// Per-rank intermediates of the dense MLP and the shared expert.
    pub dense_intermediates: [u32; 2],
    pub clamp: SwigluClamp,
    /// Routed expert weight formats and dense weight formats.
    pub expert_weights: [WeightFormat; 3],
    pub dense_weights: [WeightFormat; 2],
    pub decode_tokens: u32,
    pub prefill_tokens: u32,
}

/// Rank shapes at TP4 of 288 experts, top 8, hidden 4096, dense
/// intermediates 3072 and 512 per rank, and a symmetric clamp of 10: the
/// dense projections of [`catalog`]. Its routed fields are an unsplit
/// expert's (intermediate 2048, 72 experts a rank when experts are
/// distributed); the served routed layout, [`split::SHAPES`], splits every
/// expert over the four ranks.
pub const TP4_288_TOP8: ServedShapes = ServedShapes {
    experts: 288,
    local_experts: 72,
    top_k: 8,
    hidden: 4096,
    expert_intermediate: 2048,
    dense_intermediates: [3072, 512],
    clamp: SwigluClamp {
        gate_max_bits: 0x4120_0000,
        up_limit_bits: 0x4120_0000,
    },
    expert_weights: [
        WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
        WeightFormat::Nvfp4,
        WeightFormat::Mxfp8,
    ],
    dense_weights: [
        WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
        WeightFormat::Bf16,
    ],
    decode_tokens: 32,
    prefill_tokens: 8192,
};

impl ServedShapes {
    fn query(
        &self,
        weight: WeightFormat,
        projection: GatedMlpProjection,
        grouped: bool,
        intermediate: u32,
        tokens: u32,
    ) -> GatedMlpProjectionQuery {
        use kernel_core::contract::{Epilogues, GemmQuery, GroupedGemm};
        let gate_up = projection == GatedMlpProjection::GateUp;
        let (n, k) = if gate_up {
            (2 * intermediate, self.hidden)
        } else {
            (self.hidden, intermediate)
        };
        let rows = if grouped { tokens * self.top_k } else { tokens };
        GatedMlpProjectionQuery {
            gemm: GemmQuery {
                m_capacity: rows,
                n,
                k,
                weight,
                activation: ActivationFormat::Bf16,
                output: if grouped && !gate_up {
                    DType::F32
                } else {
                    DType::Bf16
                },
                epilogues: Epilogues {
                    swiglu: gate_up.then_some(SwigluColumns::ConcatenatedHalves),
                    weighted_moe_combine: grouped && !gate_up,
                    ..Epilogues::default()
                },
                grouped: grouped.then_some(GroupedGemm {
                    experts: self.local_experts,
                    top_k: self.top_k,
                    capacity: RoutedCapacity::TotalRows(rows),
                }),
            },
            projection,
            swiglu_clamp: gate_up.then_some(self.clamp),
            route_weight: grouped.then_some(RouteWeightPoint::AfterDown),
        }
    }

    /// Every served dense projection: the dense MLP and the shared expert
    /// in each dense format, gate/up and down, at the decode (row tile 16)
    /// and prefill (row tile 64) capacity. The routed experts' projections
    /// are [`split::catalog`]'s.
    pub fn projections(&self) -> Result<Vec<Projection>, Refusal> {
        let mut out = Vec::new();
        let phases = [(16, self.decode_tokens), (64, self.prefill_tokens)];
        let mut add = |weight, grouped, intermediate| -> Result<(), Refusal> {
            for projection in [GatedMlpProjection::GateUp, GatedMlpProjection::Down] {
                for (row_tile, tokens) in phases {
                    let query = self.query(weight, projection, grouped, intermediate, tokens);
                    out.push(Projection::new(query, row_tile, tokens)?);
                }
            }
            Ok(())
        };
        for weight in self.dense_weights {
            for intermediate in self.dense_intermediates {
                add(weight, false, intermediate)?;
            }
        }
        Ok(out)
    }

    /// The decode and prefill routes, with the paddings of their projections.
    pub fn routes(&self) -> [LocalRoute; 2] {
        let route = |capacity, pad| LocalRoute {
            experts: self.experts,
            local_experts: self.local_experts,
            top_k: self.top_k,
            capacity,
            pad,
        };
        [
            route(self.decode_tokens, 16),
            route(self.prefill_tokens, 64),
        ]
    }

    /// The BF16 route sum at the prefill capacity, which also serves decode.
    /// Its F32 form is [`RoutedSum::with_output`]; a route cubin holds the
    /// forms its caller passes.
    pub fn sums(&self) -> [RoutedSum; 1] {
        [RoutedSum {
            top_k: self.top_k,
            width: self.hidden,
            local_experts: self.local_experts,
            capacity: self.prefill_tokens,
            output: RoutedSumOutput::Bf16,
        }]
    }
}

/// The cubin name prefix of the served dense chained instances (named for
/// [`TP4_288_TOP8`], whose locked cubins keep it).
pub const PREFIX: &str = "chop_moe_chain_e288l72k8";

/// The served dense projections of [`TP4_288_TOP8`].
pub fn catalog() -> Vec<Projection> {
    TP4_288_TOP8
        .projections()
        .expect("served chained projections are supported")
}

/// [`catalog`]'s projections that [`Projection::block_warps`] takes (its
/// FP8 128x128 and BF16 decode projections) under that schedule.
pub fn block_warp_catalog() -> Vec<Projection> {
    catalog()
        .iter()
        .filter_map(|projection| projection.block_warps().ok())
        .collect()
}

/// The cubins of [`block_warp_catalog`], `{PREFIX}_{format}_bw4`.
pub fn block_warp_cubin_specs() -> Vec<CubinSpec> {
    projection_cubin_specs(PREFIX, &block_warp_catalog())
}

/// Every served dense chained cubin: [`catalog`]'s projections per weight
/// format.
pub fn cubin_specs() -> Vec<CubinSpec> {
    projection_cubin_specs(PREFIX, &catalog())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The route kernels that write metadata (counts, offsets, sorted
    /// routes, which follow every token) claim about the routed operation's
    /// rows; the prefill selection, which writes each token's ids and
    /// weights alone, about its own.
    #[test]
    fn route_metadata_kernels_claim_the_routed_operation() {
        use kernel_core::properties::Scope;
        let [decode, prefill] = split::routes();
        let scope = |route: LocalRoute, kernel| route.invariance(kernel).scope;
        assert_eq!(scope(decode, RouteKernel::Decode), Scope::RoutedOperation);
        assert_eq!(scope(prefill, RouteKernel::Select), Scope::Entry);
        assert_eq!(scope(prefill, RouteKernel::Count), Scope::RoutedOperation);
        assert_eq!(scope(prefill, RouteKernel::Scatter), Scope::RoutedOperation);
        for (route, kernel) in [
            (decode, RouteKernel::Decode),
            (prefill, RouteKernel::Select),
            (prefill, RouteKernel::Count),
            (prefill, RouteKernel::Scatter),
        ] {
            let claims = route.invariance(kernel);
            assert_eq!(
                (claims.rows, claims.ties),
                (
                    kernel_core::properties::Rows::OneRow {
                        max: route.capacity
                    },
                    Ties::LowerIndex
                )
            );
        }
    }

    #[test]
    fn served_catalog_covers_every_format_projection_and_phase() {
        let projections = catalog();
        // Two dense formats over two dense shapes, two projections and two
        // phases each; no routed projection.
        assert_eq!(projections.len(), 2 * 2 * 2 * 2);
        assert!(projections.iter().all(|p| p.query.gemm.grouped.is_none()));
        let mut symbols = projections
            .iter()
            .map(|p| p.symbol.clone())
            .collect::<Vec<_>>();
        symbols.sort();
        symbols.dedup();
        assert_eq!(symbols.len(), projections.len());
        // A routed projection of the split: every expert local, a quarter of
        // the intermediate.
        let gate_up = split::catalog()
            .into_iter()
            .find(|p| p.weights() == Weights::Nvfp4 && p.gate_up() && p.row_tile == 16)
            .unwrap();
        assert_eq!(
            gate_up.symbol,
            "chop_moe_chain_gateup_nvfp4_e288k8_n1024_k4096_r16_clamp4120000041200000_v1"
        );
        assert!(
            gate_up
                .entry()
                .body
                .contains("SwigluClamp<true,0x41200000u,0x41200000u>")
        );
        assert!(gate_up.entry().body.contains(
            "Weights::nvfp4,chop::moe::chained::Projection::gate_up,288,8,512,4096,16,true"
        ));
        // Decode: 256 routes and 288 experts padded to 16 rows; 8 column tiles.
        assert_eq!(
            gate_up.plan(32).unwrap().grid,
            [(256 + 288 * 15_u32).div_ceil(16), 8, 1]
        );
        assert_eq!(
            gate_up.plan(1).unwrap().grid,
            [(8 + 288 * 15_u32).div_ceil(16), 8, 1]
        );
        assert!(gate_up.plan(33).is_err());
        let dense_down = projections
            .iter()
            .find(|p| p.query.gemm.grouped.is_none() && !p.gate_up() && p.query.gemm.k == 512)
            .unwrap();
        assert_eq!(dense_down.columns(), 4096);
        assert!(dense_down.entry().body.contains("SwigluClamp<false,0u,0u>"));
        let specs = cubin_specs();
        let entries = specs.iter().map(|s| s.entries.len()).sum::<usize>();
        // Projections only: the routes and sums are the split's.
        assert_eq!(entries, projections.len());
        assert!(specs.iter().all(|s| !s.name.ends_with("_route")));
    }

    /// The split's served FP8 decode gate/up in warp streams, one item per
    /// CTA, and its decode down's query at K 2048 (warp streams refuse the
    /// split's K 512): four swizzled stages over a fixed grid of persistent
    /// CTAs.
    #[test]
    fn warp_streams_take_the_routed_fp8_decode_projections_on_a_fixed_grid() {
        let decode = |gate_up: bool| {
            split::catalog()
                .into_iter()
                .find(|p| {
                    p.weights() == Weights::Fp8Block128
                        && p.row_tile == 16
                        && p.gate_up() == gate_up
                })
                .unwrap()
        };
        let mut long_down = decode(false).query;
        long_down.gemm.k = 2048;
        let streams = [
            decode(true).warp_streams().unwrap(),
            Projection::new(long_down, 16, 32)
                .unwrap()
                .warp_streams()
                .unwrap(),
        ];
        assert!(decode(false).warp_streams().is_err());
        for p in &streams {
            assert_eq!(p.schedule, Schedule::WarpStreams);
            assert!(p.query.gemm.grouped.is_some() && p.row_tile == 16);
            assert!(p.symbol.ends_with("_ws_v1"), "{}", p.symbol);
            // Down: a fixed grid of persistent CTAs at every live count;
            // gate/up: one item per CTA over the live count's tile bound.
            let layout = p.warp_stream_layout();
            for tokens in [1, 7, 32] {
                let plan = p.plan(tokens).unwrap();
                let routes = tokens * 8;
                let tiles = (routes + routes.min(288) * 15).div_ceil(16);
                let grid = if layout.persistent {
                    [WARP_STREAMS_CTAS, 1, 1]
                } else {
                    [p.columns() / 16, tiles, 1]
                };
                assert_eq!(plan.grid, grid, "{}", p.symbol);
                assert_eq!(plan.block, [THREADS, 1, 1]);
                assert_eq!(plan.dynamic_shared_bytes, 0);
            }
            assert!(p.plan(33).is_err());
            let WarpStreamLayout {
                stages,
                swizzle,
                persistent,
            } = layout;
            // 256 routes reach at most 256 of the 288 experts.
            let tiles = (256 + 256 * 15_u32).div_ceil(16);
            assert!(p.entry().body.starts_with(
                "chop::moe::warp_streams::run<chop::moe::chained::Config<chop::moe::chained::Weights::fp8_block128,"
            ));
            assert!(p.entry().body.contains(&format!(
                ">,128,{stages},{swizzle},{persistent},{tiles}>(input,"
            )));
        }
        let [gate_up, down] = streams;
        assert_eq!(
            gate_up.warp_stream_layout(),
            WarpStreamLayout {
                stages: 2,
                swizzle: false,
                persistent: false
            }
        );
        assert_eq!(
            down.warp_stream_layout(),
            WarpStreamLayout {
                stages: 4,
                swizzle: true,
                persistent: true
            }
        );
        assert_eq!(
            gate_up.symbol,
            "chop_moe_chain_gateup_fp8block_e288k8_n1024_k4096_r16_clamp4120000041200000_ws_v1"
        );
        // The shortest K each layout's ring admits (the kernel's static
        // assertion refuses shorter ones): two rounds for gate/up's two
        // stages, four for down's four.
        for (p, shortest) in [(&gate_up, 1024), (&down, 2048)] {
            for k in [512, 768, 1024, 1536, 2048, 2560, 4096] {
                let mut query = p.query;
                query.gemm.k = k;
                let taken = Projection::new(query, p.row_tile, p.tokens_capacity)
                    .unwrap()
                    .warp_streams();
                let admitted = k >= shortest && k % 512 == 0;
                assert_eq!(taken.is_ok(), admitted, "{} at K {k}", p.symbol);
                if let Ok(taken) = taken {
                    assert_eq!(taken.warp_stream_layout(), p.warp_stream_layout());
                }
            }
        }
        // Only grouped FP8 decode projections with K in two or more 512s:
        // of the split's, its gate/up.
        for p in catalog().into_iter().chain(split::catalog()) {
            let taken = p.weights() == Weights::Fp8Block128
                && p.query.gemm.grouped.is_some()
                && p.row_tile == 16
                && p.gate_up();
            assert_eq!(p.warp_streams().is_ok(), taken, "{}", p.symbol);
        }
        let specs = projection_cubin_specs(split::PREFIX, &[gate_up]);
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "chop_moe_chain_e288l288k8_fp8block_ws");
        assert_eq!(
            specs[0].headers,
            ["#include \"moe/chained_warp_streams.cuh\""]
        );
    }

    #[test]
    fn block_warps_take_decode_fp8_nvfp4_and_bf16_projections_with_live_grids() {
        let warps = block_warp_catalog();
        // The FP8 and BF16 dense MLP and shared expert, gate/up and down.
        assert_eq!(warps.len(), 4 + 4);
        assert!(
            warps
                .iter()
                .all(|p| p.schedule == Schedule::BlockWarps && p.row_tile == 16)
        );
        assert!(warps.iter().all(|p| p.query.gemm.grouped.is_none()));
        // The split's served NVFP4 decode gate/up in block warps.
        let gate_up = split::forms()
            .into_iter()
            .find(|p| p.gate_up() && p.weights() == Weights::Nvfp4 && p.row_tile == 16)
            .unwrap();
        assert_eq!(gate_up.schedule, Schedule::BlockWarps);
        assert_eq!(
            gate_up.symbol,
            "chop_moe_chain_gateup_nvfp4_e288k8_n1024_k4096_r16_clamp4120000041200000_bw4_v1"
        );
        // One token's 8 routes reach at most 8 experts: 8 + 8 * 15 rows in
        // 8 tiles, where the column schedule covers all 288 experts' padding.
        assert_eq!(gate_up.plan(1).unwrap().grid, [8, 32, 1]);
        let columns = split::catalog()
            .into_iter()
            .find(|p| p.symbol == gate_up.symbol.replace("_bw4", ""))
            .unwrap();
        assert_eq!(columns.plan(1).unwrap().grid, [271, 8, 1]);
        // At 32 tokens 256 routes reach at most 256 of the 288 experts.
        assert_eq!(gate_up.plan(32).unwrap().grid, [256, 32, 1]);
        assert_eq!(columns.plan(32).unwrap().grid, [286, 8, 1]);
        let shared = warps
            .iter()
            .find(|p| p.gate_up() && p.columns() == 512 && p.weights() == Weights::Fp8Block128)
            .unwrap();
        assert_eq!(shared.plan(1).unwrap().grid, [1, 32, 1]);
        // Prefill tiles and MXFP8 stay on the column schedule.
        for p in catalog().into_iter().chain(split::catalog()) {
            let taken = p.row_tile == 16
                && matches!(
                    p.weights(),
                    Weights::Fp8Block128 | Weights::Nvfp4 | Weights::Bf16
                );
            assert_eq!(p.block_warps().is_ok(), taken, "{}", p.symbol);
        }
        let specs = block_warp_cubin_specs();
        assert_eq!(
            specs
                .iter()
                .map(|s| (s.name.as_str(), s.headers[0].as_str()))
                .collect::<Vec<_>>(),
            [
                (
                    "chop_moe_chain_e288l72k8_fp8block_bw4",
                    "#include \"moe/chained_block_warps.cuh\""
                ),
                (
                    "chop_moe_chain_e288l72k8_bf16_bw4",
                    "#include \"moe/chained_block_warps.cuh\""
                ),
            ]
        );
        assert!(specs.iter().all(|s| s.headers.len() == 1));
        // The NVFP4 block warps run their own kernel.
        assert!(
            gate_up
                .entry()
                .body
                .starts_with("chop::moe::block_warps_fp4::run<")
        );
        let nvfp4 = projection_cubin_specs(split::PREFIX, &[gate_up]);
        assert_eq!(
            nvfp4[0].headers,
            ["#include \"moe/chained_block_warps_fp4.cuh\""]
        );
    }

    #[test]
    fn tile_pairs_take_grouped_fp8_prefill_projections_on_half_the_column_tiles() {
        let all = catalog()
            .into_iter()
            .chain(split::catalog())
            .collect::<Vec<_>>();
        let prefill = all.iter().filter(|p| p.row_tile == 64).collect::<Vec<_>>();
        for p in &prefill {
            let taken = p.weights() == Weights::Fp8Block128 && p.query.gemm.grouped.is_some();
            assert_eq!(p.tile_pairs(2).is_ok(), taken, "{}", p.symbol);
        }
        for p in all.iter().filter(|p| p.row_tile == 16) {
            assert!(p.tile_pairs(2).is_err(), "{}", p.symbol);
        }
        let gate_up = prefill
            .iter()
            .find(|p| {
                p.gate_up() && p.weights() == Weights::Fp8Block128 && p.query.gemm.grouped.is_some()
            })
            .unwrap();
        let pairs = gate_up.tile_pairs(4).unwrap();
        assert_eq!(pairs.schedule, Schedule::TilePairs { min_pair_tiles: 4 });
        assert_eq!(
            pairs.symbol,
            "chop_moe_chain_gateup_fp8block_e288k8_n1024_k4096_r64_clamp4120000041200000_tp4_v1"
        );
        assert_eq!((pairs.query, pairs.row_tile), (gate_up.query, 64));
        // The column schedule's row tiles, half its column tiles, 256 threads.
        let columns = gate_up.plan(2048).unwrap();
        let plan = pairs.plan(2048).unwrap();
        assert_eq!(plan.grid, [columns.grid[0], columns.grid[1] / 2, 1]);
        assert_eq!((plan.block, columns.block), ([256, 1, 1], [128, 1, 1]));
        let entry = pairs.entry();
        assert!(entry.body.starts_with("chop::moe::tile_pairs::run<"));
        assert!(
            entry
                .body
                .contains("SwigluClamp<true,0x41200000u,0x41200000u>>,4>(input,")
        );
        assert_eq!(entry.max_threads, 256);
        assert_eq!(
            schedule_cubin(pairs.schedule),
            ("_tp", "chained_tile_pairs.cuh")
        );
        let specs = projection_cubin_specs("p", &[pairs.clone(), pairs.tile_pairs(0).unwrap()]);
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "p_fp8block_tp");
        assert_eq!(specs[0].entries.len(), 2);
        assert_eq!(
            specs[0].headers,
            ["#include \"moe/chained_tile_pairs.cuh\""]
        );
    }

    #[test]
    fn decoded_tiles_take_dense_fp8_prefill_gate_up_on_128_token_tiles() {
        for p in catalog() {
            let taken = p.weights() == Weights::Fp8Block128
                && p.query.gemm.grouped.is_none()
                && p.gate_up()
                && p.row_tile == 64;
            assert_eq!(p.decoded_tiles().is_ok(), taken, "{}", p.symbol);
        }
        let shared = catalog()
            .into_iter()
            .find(|p| {
                p.weights() == Weights::Fp8Block128
                    && p.query.gemm.grouped.is_none()
                    && p.gate_up()
                    && p.row_tile == 64
                    && p.columns() == 512
            })
            .unwrap();
        let decoded = shared.decoded_tiles().unwrap();
        assert_eq!(decoded.schedule, Schedule::DecodedTiles);
        assert_eq!(
            decoded.symbol,
            "chop_moe_chain_gateup_fp8block_dense_n1024_k4096_r64_clamp4120000041200000_dt_v1"
        );
        assert_eq!((decoded.query, decoded.row_tile), (shared.query, 64));
        // 128-token tiles over the capacity, 64-column tiles, 384 threads and
        // the stages' dynamic shared memory.
        let plan = decoded.plan(2048).unwrap();
        assert_eq!(plan.grid, [16, 8, 1]);
        assert_eq!(plan.block, [384, 1, 1]);
        assert_eq!(plan.dynamic_shared_bytes, DECODED_TILE_SHARED_BYTES);
        assert_eq!(decoded.plan(8192).unwrap().grid, [64, 8, 1]);
        assert_eq!(decoded.plan(33).unwrap().grid, [1, 8, 1]);
        assert_eq!(decoded.plan(129).unwrap().grid, [2, 8, 1]);
        assert!(decoded.plan(8193).is_err());
        assert_eq!(shared.plan(2048).unwrap().dynamic_shared_bytes, 0);
        let entry = decoded.entry();
        assert!(entry.body.starts_with("chop::moe::decoded_tiles::run<"));
        assert!(entry.body.contains(
            "Weights::fp8_block128,chop::moe::chained::Projection::gate_up,1,1,512,4096,64,false"
        ));
        assert_eq!(entry.max_threads, 384);
        let specs = projection_cubin_specs("p", &[decoded]);
        assert_eq!(specs[0].name, "p_fp8block_dt");
        assert_eq!(specs[0].headers, ["#include \"moe/chained_decoded.cuh\""]);
    }

    /// Decoded tiles keep their K-sized static block scales and mbarriers
    /// (with the compiler's reserved 1,024 bytes) beside the 99,328 dynamic
    /// stage bytes within SM120's 101,376: K up to 15,104 is admitted, a
    /// longer K is refused when the schedule is taken and when a projection
    /// under it is planned.
    #[test]
    fn decoded_tiles_refuse_footprints_past_shared_memory() {
        // The served K = 4096 entries report 1,408 static shared bytes.
        assert_eq!(decoded_tile_static_bytes(4096), 1408);
        assert_eq!(decoded_tile_static_bytes(15_104), 2048);
        assert_eq!(
            decoded_tile_static_bytes(15_104) + DECODED_TILE_SHARED_BYTES,
            crate::backend::SHARED_MEMORY_BYTES
        );
        let served = catalog()
            .into_iter()
            .find(|p| p.decoded_tiles().is_ok())
            .unwrap();
        let at = |k: u32| {
            let mut query = served.query;
            query.gemm.k = k;
            Projection::new(query, served.row_tile, served.tokens_capacity).unwrap()
        };
        assert!(at(15_104).decoded_tiles().unwrap().plan(128).is_ok());
        let refusal = Refusal(
            "chained decoded tiles' static block scales and stages exceed SM120 shared memory",
        );
        for k in [15_232, 32_768] {
            let columns = at(k);
            assert_eq!(columns.decoded_tiles().unwrap_err(), refusal, "{k}");
            let forced = Projection {
                schedule: Schedule::DecodedTiles,
                ..columns
            };
            assert_eq!(forced.plan(128).unwrap_err(), refusal, "{k}");
        }
    }

    #[test]
    fn unsupported_contracts_refuse_by_name() {
        let shapes = split::SHAPES;
        let base = shapes.query(
            shapes.expert_weights[0],
            GatedMlpProjection::Down,
            true,
            shapes.expert_intermediate,
            32,
        );
        let before = GatedMlpProjectionQuery {
            route_weight: Some(RouteWeightPoint::BeforeDown),
            ..base
        };
        assert_eq!(
            Projection::new(before, 16, 32).unwrap_err(),
            Refusal("the chained expert engine applies route weights after the down projection")
        );
        let mut bf16_down = base;
        bf16_down.gemm.output = DType::Bf16;
        assert!(Projection::new(bf16_down, 16, 32).is_err());
        let mut a8 = base;
        a8.gemm.activation = ActivationFormat::Mxfp8;
        assert_eq!(
            Projection::new(a8, 16, 32).unwrap_err(),
            Refusal("the chained expert engine reads BF16 activations")
        );
        let mut mxfp4 = base;
        mxfp4.gemm.weight = WeightFormat::Mxfp4;
        assert!(Projection::new(mxfp4, 16, 32).is_err());
        assert!(Projection::new(base, 8, 32).is_err());
        assert!(Projection::new(base, 16, 33).is_err());
        let projection = Projection::new(base, 16, 32).unwrap();
        let mut handles = [1_u64; 11];
        handles[3] = 0; // local scales: FP8 reads none.
        handles[4] = 0; // globals
        handles[8] = 0; // live tokens: grouped reads none.
        assert!(projection.arguments(handles).is_ok());
        handles[9] = 0; // route weights: grouped down needs them.
        assert!(projection.arguments(handles).is_err());
    }

    #[test]
    fn admission_refuses_non_finite_and_negative_scales() {
        let edge = [
            0.0,
            -0.0,
            f32::MIN_POSITIVE / 8.0,
            2_f32.powi(-126),
            2_f32.powi(100),
            f32::MAX,
        ];
        admit_scales(&edge).unwrap();
        for bad in [f32::INFINITY, f32::NAN, -1.0, -f32::MIN_POSITIVE] {
            assert_eq!(
                admit_scales(&[1.0, bad]),
                Err(Refusal(
                    "chained F32 weight scales must be finite and not negative"
                ))
            );
        }
    }

    #[test]
    fn one_routed_sum_serves_every_smaller_capacity() {
        let [sum] = split::sums();
        assert_eq!(sum.plan(32).unwrap().grid, [32, 1, 1]);
        assert_eq!(sum.plan(8192).unwrap().grid, [8192, 1, 1]);
        assert!(sum.plan(0).is_err());
        assert!(sum.plan(8193).is_err());
        let f32_sum = sum.with_output(RoutedSumOutput::F32);
        assert_eq!(f32_sum.plan(32), sum.plan(32));
    }

    #[test]
    fn routed_sum_forms_differ_only_in_output_type_and_symbol() {
        const PREFIX: &str = split::PREFIX;
        let [bf16] = split::sums();
        let f32_sum = bf16.with_output(RoutedSumOutput::F32);
        assert_eq!(
            bf16.symbol(PREFIX),
            "chop_moe_chain_e288l288k8_sum_k8_w4096_l288_t8192_v1"
        );
        assert_eq!(
            f32_sum.symbol(PREFIX),
            "chop_moe_chain_e288l288k8_sum_f32_k8_w4096_l288_t8192_v1"
        );
        assert_eq!(
            (bf16.output_row_bytes(), f32_sum.output_row_bytes()),
            (8192, 16384)
        );
        let (a, b) = (bf16.entry(PREFIX).unwrap(), f32_sum.entry(PREFIX).unwrap());
        // One body: the output pointer's type picks the form.
        assert_eq!(a.body, b.body);
        assert_eq!(
            a.body,
            "chop::moe::chained::routed_sum<8,4096,288>(routed,ids,live_tokens,rank,output);"
        );
        let output = |e: &EntrySpec| match e.params.last() {
            Some(EntryParam::DevicePointer { name, cxx_type }) if name == "output" => {
                cxx_type.clone()
            }
            other => panic!("{other:?}"),
        };
        assert_eq!(
            (output(&a), output(&b)),
            ("uint16_t".into(), "float".into())
        );
        assert_eq!(a.params[..4], b.params[..4]);
        // A route cubin holds the forms its caller passes; the served
        // cubins hold both.
        let sums_of = |spec: &CubinSpec| {
            spec.entries
                .iter()
                .filter(|e| e.symbol.contains("_sum"))
                .map(|e| e.symbol.clone())
                .collect::<Vec<_>>()
        };
        let routes = split::routes();
        let spec = route_cubin_spec(PREFIX, &routes, &[bf16]).unwrap();
        assert_eq!(sums_of(&spec), [bf16.symbol(PREFIX)]);
        let served = split::cubin_specs().unwrap();
        let route = served.iter().find(|s| s.name.ends_with("_route")).unwrap();
        assert_eq!(
            sums_of(route),
            [bf16.symbol(PREFIX), f32_sum.symbol(PREFIX)]
        );
    }

    #[test]
    fn local_routes_take_any_local_expert_count() {
        for local_experts in [72, 96] {
            let decode = LocalRoute {
                experts: local_experts * 4,
                local_experts,
                top_k: 8,
                capacity: 32,
                pad: 16,
            };
            assert_eq!(decode.kernels(), [RouteKernel::Decode]);
            assert_eq!(
                decode.plan(RouteKernel::Decode, 32).unwrap().block,
                [1024, 1, 1]
            );
            assert!(decode.plan(RouteKernel::Decode, 33).is_err());
            assert!(decode.plan(RouteKernel::Select, 1).is_err());
            let small = LocalRoute {
                capacity: 1,
                ..decode
            };
            assert_eq!(
                small.plan(RouteKernel::Decode, 1).unwrap().block[0],
                local_experts.next_multiple_of(32)
            );
            let prefill = LocalRoute {
                capacity: 8192,
                pad: 64,
                ..decode
            };
            assert_eq!(
                prefill.plan(RouteKernel::Select, 0).unwrap().grid,
                [1024, 1, 1]
            );
            assert_eq!(
                prefill.plan(RouteKernel::Scatter, 8192).unwrap().grid,
                [local_experts, 1, 1]
            );
            assert_eq!(
                prefill.sorted_rows_capacity(),
                8192 * 8 + local_experts * 63
            );
            assert_eq!(prefill.entries("p").unwrap().len(), 3);
            // The decode route builds its metadata across warps.
            let entries = decode.entries("p").unwrap();
            assert_eq!(entries.len(), 1);
            assert!(
                entries[0]
                    .body
                    .starts_with("chop::moe::local::route_decode_scan<"),
                "{}",
                entries[0].body
            );
        }
        let bad = LocalRoute {
            experts: 300,
            local_experts: 75,
            top_k: 8,
            capacity: 32,
            pad: 16,
        };
        assert!(bad.validate().is_err());
    }
}
