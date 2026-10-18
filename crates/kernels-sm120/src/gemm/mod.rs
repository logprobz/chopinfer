//! Dense SM120 GEMM planning and native cubin instantiations.

pub mod cluster;
mod construction;
pub mod fused;
pub mod mid;
pub mod prefetch;
pub mod stacked;
pub use construction::{DenseRequest, instantiate};

use crate::arithmetic::{self, DeclaredCubin};
use crate::invariance::Invariance;
use crate::{Catalogs, Profile, row};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    ActivationFormat, Argument, ArgumentList, CompileKey, CompileKnobs, DType, DenseKernel,
    Epilogues, GemmQuery, InstantiationSpec, LaunchPlan, NumericalContract, OpFamily,
    OutputQuantization, Refusal, RuntimeCounts, StaticQuery, SwigluColumns, Target, TensorMap,
    TensorMapElement, TensorMapInterleave, TensorMapL2Promotion, TensorMapOobFill,
    TensorMapSwizzle, Validate, WeightFormat,
};
use std::ops::Deref;
use std::path::PathBuf;

/// A dense GEMM instantiation and the model profile that owns it.
#[derive(Clone, Debug)]
pub struct GemmVariant {
    pub profile: Profile,
    pub instantiation: InstantiationSpec,
    /// Fragment prefill entries are chunk invariant; decode entries claim
    /// one-row rows where the catalog declares them.
    pub invariance: Invariance,
}

impl GemmVariant {
    /// The entry with `invariance` declared.
    pub fn declare(self, invariance: Invariance) -> Self {
        GemmVariant { invariance, ..self }
    }
}

impl Deref for GemmVariant {
    type Target = InstantiationSpec;

    fn deref(&self) -> &Self::Target {
        &self.instantiation
    }
}

pub const SM_COUNT: u32 = 188;
const MATH_WARPS: u32 = 8;
const THREADS: u32 = (MATH_WARPS + 4) * 32;
/// Partial slots per CTA of a fragment prefill entry (`kMaxSegments` in
/// fragment_prefill.cuh): its K splits into at most this many fixed
/// segments, and a CTA keeps one partial for the tile its stream-K range
/// ends in and one per segment of the tile it starts in.
const PREFILL_SEGMENTS: u8 = 8;
/// K partitions of the fragment router (`kRouterSplits` in
/// fragment_router.cuh).
pub const ROUTER_SPLITS: u8 = 4;
/// Router logits (experts) and K of the fragment router kernel.
const ROUTER_N: u32 = 256;
const ROUTER_K: u32 = 4096;
const TUNING: &str = include_str!("../../tuning/gemm.json");
/// Weight strips per CTA a strip-chain entry may hold
/// (`DenseKernel::StripChains`, `StripTraits` in strip_chains.cuh).
pub const STRIP_COUNTS: [u8; 3] = [1, 2, 4];
/// The TMA ring of a strip-chain entry: as many stages as fit this many
/// bytes (two CTAs share an SM's shared memory), within 2 to 16.
const STRIP_RING_BYTES: u32 = 48 * 1024;
const STRIP_MAX_STAGES: u32 = 16;

/// The kernel `key` declares. A dense entry without one is refused, never
/// given a kernel by its shape.
pub fn kernel(key: &CompileKey) -> Result<DenseKernel, Refusal> {
    key.dense
        .ok_or(Refusal("a dense GEMM entry must declare its kernel"))
}

/// The fixed K segments of fragment prefill entry `key`.
pub fn prefill_segments(key: &CompileKey) -> Result<u8, Refusal> {
    match kernel(key)? {
        DenseKernel::FragmentPrefill { segments } => Ok(segments),
        _ => Err(Refusal("not a fragment prefill entry")),
    }
}

/// Fragment decode kernels: plain, cluster and router.
fn is_fragment_decode(kernel: DenseKernel) -> bool {
    matches!(
        kernel,
        DenseKernel::FragmentDecode { .. }
            | DenseKernel::FragmentCluster { .. }
            | DenseKernel::FragmentRouter { .. }
    )
}

fn is_fragment_prefill(kernel: DenseKernel) -> bool {
    matches!(kernel, DenseKernel::FragmentPrefill { .. })
}

fn is_fragment(kernel: DenseKernel) -> bool {
    is_fragment_decode(kernel) || is_fragment_prefill(kernel)
}

fn is_cluster(kernel: DenseKernel) -> bool {
    matches!(kernel, DenseKernel::FragmentCluster { .. })
}

fn is_router(kernel: DenseKernel) -> bool {
    matches!(kernel, DenseKernel::FragmentRouter { .. })
}

/// The output columns (weight rows) of a strip-chain CTA, 16 per strip.
fn strip_rows(kernel: DenseKernel) -> Option<u32> {
    match kernel {
        DenseKernel::StripChains { strips, .. } => Some(16 * u32::from(strips)),
        _ => None,
    }
}

/// The ring stages of a strip-chain entry with `tile_m` weight rows and
/// `tile_n` token rows per stage: one K chunk, 128 bytes a weight row (64
/// BF16 or 128 FP8 values) and `tile_k` BF16 values an activation row.
fn strip_stages(tile_m: u32, tile_n: u32, tile_k: u32) -> u8 {
    (STRIP_RING_BYTES / (tile_m * 128 + tile_n * tile_k * 2)).clamp(2, STRIP_MAX_STAGES) as u8
}

/// The 64-K stages of the activation ring of a decoded-tile entry
/// (`DecodedTiles::a_stages` in decoded_tiles.cuh; its two raw-code and two
/// decoded weight stages are fixed) and of a staged-tile entry's ring
/// (`StagedTiles::stages` in staged_tiles.cuh).
const DECODED_STAGES: u8 = 3;
/// A decoded-tile entry's dynamic shared memory (`DecodedTiles::shared_bytes`):
/// three 128 x 64 BF16 activation stages, two 128 x 64 FP8 code stages, two
/// 128 x 64 BF16 decoded weight stages and 1024 bytes of alignment. A
/// staged-tile entry's three activation and three weight stages take the
/// same (`StagedTiles::shared_bytes`).
const DECODED_SHARED_BYTES: u32 = 3 * 128 * 128 + 2 * 128 * 64 + 2 * 128 * 128 + 1024;

/// Decoded weight tiles (`decoded_tiles.cuh`): an FP8 128x128 prefill entry
/// of `DenseKernel::TiledChains` in 64-K stages ([`decoded_tile_knobs`]).
fn is_decoded_tiles(
    kernel: DenseKernel,
    query: GemmQuery,
    knobs: CompileKnobs,
    decode: bool,
) -> bool {
    matches!(kernel, DenseKernel::TiledChains { .. })
        && !decode
        && matches!(query.weight, WeightFormat::Fp8Block128x128(_))
        && knobs.tile_k == 64
}

/// Balanced decoded weight tiles (`decoded_balanced.cuh`): decoded tiles at
/// [`balanced_decoded_tile_knobs`], whose 64-row `tile_m` is the half tile
/// a final partial wave is split into.
fn is_balanced_decoded(
    kernel: DenseKernel,
    query: GemmQuery,
    knobs: CompileKnobs,
    decode: bool,
) -> bool {
    is_decoded_tiles(kernel, query, knobs, decode) && knobs.tile_m == 64
}

/// Staged weight tiles (`staged_tiles.cuh`): a BF16 prefill entry of
/// `DenseKernel::TiledChains` at 128 x 128 tiles ([`staged_tile_knobs`]),
/// where the warp tiles take 64-row tiles.
fn is_staged_tiles(
    kernel: DenseKernel,
    query: GemmQuery,
    knobs: CompileKnobs,
    decode: bool,
) -> bool {
    matches!(kernel, DenseKernel::TiledChains { .. })
        && !decode
        && query.weight == WeightFormat::Bf16
        && (knobs.tile_m, knobs.tile_n) == (128, 128)
}

/// The knobs that run a BF16 prefill `DenseKernel::TiledChains { splits: 1
/// }` entry (more than 32 rows, `n` a multiple of 128, K of 64) in staged
/// weight tiles (`staged_tiles.cuh`): 128 x 128 tiles in 64-K stages that a
/// loader lane fills by TMA and eight math warps read with `ldmatrix.x4`,
/// the final partial wave split into 64-row halves when they fit the SMs
/// (`common/balanced_tiles.cuh`). Each output element keeps the warp
/// tiles' 64-K chains and K-order sums, so its bits equal the entry's at its
/// default knobs.
pub fn staged_tile_knobs() -> CompileKnobs {
    CompileKnobs {
        tile_m: 128,
        ..decoded_tile_knobs()
    }
}

/// The knobs that run an FP8 128x128 prefill `DenseKernel::TiledChains {
/// splits: 1 }` entry (more than 32 rows, `n` and `k` multiples of 128) in
/// decoded weight tiles (`decoded_tiles.cuh`): 128 x 128 tiles, 64-K
/// stages, each weight code decoded to BF16 once per CTA. Each output
/// element keeps the warp tiles' 128-K chains and scale FMAs, so its bits
/// equal the entry's at its default knobs.
pub fn decoded_tile_knobs() -> CompileKnobs {
    CompileKnobs {
        tile_m: 128,
        tile_n: 128,
        tile_k: 64,
        stages: DECODED_STAGES,
        swizzle: tuning_number("swizzle") as u8,
        warps: (THREADS / 32) as u8,
        split_k_capacity: 1,
    }
}

/// [`decoded_tile_knobs`] with the final partial wave over the SMs split
/// into 64-row halves when they fit them (`decoded_balanced.cuh`,
/// `common/balanced_tiles.cuh`); the activation box is one 64-row half.
/// Each output element keeps the decoded tiles' chains and scale FMAs, so
/// its bits equal theirs.
pub fn balanced_decoded_tile_knobs() -> CompileKnobs {
    CompileKnobs {
        tile_m: 64,
        ..decoded_tile_knobs()
    }
}

/// Refuses a declared kernel that cannot compute `query`.
fn kernel_supported(kernel: DenseKernel, query: GemmQuery) -> Result<(), Refusal> {
    let decode = query.m_capacity <= 32;
    let fragment_operands = query.activation == ActivationFormat::Bf16
        && matches!(
            query.weight,
            WeightFormat::Bf16 | WeightFormat::Fp8Block128x128(_) | WeightFormat::Fp8Rows64x128(_)
        )
        && matches!(
            query.epilogues.swiglu,
            None | Some(SwigluColumns::ConcatenatedHalves)
        )
        && !query.epilogues.residual_add;
    if let DenseKernel::Tiled { splits: 0 }
    | DenseKernel::TiledChains { splits: 0 }
    | DenseKernel::StripChains { splits: 0, .. }
    | DenseKernel::FragmentDecode { splits: 0 } = kernel
    {
        return Err(Refusal("a dense entry declares at least one K split"));
    }
    match kernel {
        DenseKernel::Tiled { .. } => Ok(()),
        DenseKernel::StripChains { strips, .. } => {
            if !decode {
                Err(Refusal("strip chains serve at most 32 rows"))
            } else if !STRIP_COUNTS.contains(&strips) {
                Err(Refusal("strip chains hold 1, 2 or 4 weight strips per CTA"))
            } else if query.activation != ActivationFormat::Bf16
                || !matches!(
                    query.weight,
                    WeightFormat::Bf16 | WeightFormat::Fp8Block128x128(_)
                )
                || query.epilogues.swiglu.is_some()
            {
                Err(Refusal(
                    "strip chains need BF16 activations, BF16 or FP8 128x128 weights \
                     and no SwiGLU",
                ))
            } else {
                Ok(())
            }
        }
        DenseKernel::TiledChains { splits } => {
            if !decode && splits != 1 {
                Err(Refusal(
                    "tiled chains above 32 rows run one split in warp tiles",
                ))
            } else if query.activation != ActivationFormat::Bf16
                || !matches!(
                    query.weight,
                    WeightFormat::Bf16 | WeightFormat::Fp8Block128x128(_)
                )
                || query.epilogues.swiglu.is_some()
            {
                Err(Refusal(
                    "tiled chain splits need BF16 activations, BF16 or FP8 128x128 \
                     weights and no SwiGLU",
                ))
            } else {
                Ok(())
            }
        }
        _ if !fragment_operands => Err(Refusal(
            "fragment GEMM needs BF16 activations, BF16 or 128-K FP8 weights, \
             concatenated SwiGLU halves and no residual",
        )),
        DenseKernel::FragmentPrefill { segments } => {
            if decode {
                Err(Refusal("fragment prefill serves more than 32 rows"))
            } else if !(1..=PREFILL_SEGMENTS).contains(&segments) {
                Err(Refusal("fragment prefill takes 1 to 8 K segments"))
            } else {
                Ok(())
            }
        }
        _ if !decode => Err(Refusal("fragment decode serves at most 32 rows")),
        DenseKernel::FragmentDecode { .. } => Ok(()),
        DenseKernel::FragmentRouter { splits } => {
            if query.epilogues.swiglu.is_some() {
                Err(Refusal("router GEMM does not implement SwiGLU"))
            } else if (query.n, query.k, query.weight, query.output, splits)
                != (
                    ROUTER_N,
                    ROUTER_K,
                    WeightFormat::Bf16,
                    DType::F32,
                    ROUTER_SPLITS,
                )
            {
                Err(Refusal(
                    "the fragment router computes 256 F32 logits from BF16 weights \
                     over K = 4096 in 4 splits",
                ))
            } else {
                Ok(())
            }
        }
        DenseKernel::FragmentCluster { warp_blocks } => {
            let weights = cluster::Weights::of(query.weight)
                .ok_or(Refusal("cluster decode weights are BF16 or 128-K FP8"))?;
            if warp_blocks * weights.tiles_per_block() != 16 {
                return Err(Refusal(
                    "fragment cluster decode holds 16 fragment tiles per warp",
                ));
            }
            if !query.k.is_multiple_of(128) {
                return Err(Refusal("cluster decode K is whole 128-K blocks"));
            }
            local_shape(warp_blocks, query).validate()
        }
    }
}

/// Weight strips per CTA for local decode (measured on GPU 0). Wider token
/// buckets share the staged activation slab across more strips.
fn local_strips(warp_blocks: u32, query: GemmQuery) -> u32 {
    let mut strips = match query.m_capacity {
        0..=4 => 2,
        5..=16 => 4,
        _ => 8,
    };
    // Clusters hold at most eight CTAs.
    while cluster_for_strips(warp_blocks, query, strips) > 8 {
        strips /= 2;
    }
    strips
}

/// `warp_blocks` 128-K blocks per warp, and 8 / strips warps share a strip.
fn cluster_for_strips(warp_blocks: u32, query: GemmQuery, strips: u32) -> u32 {
    (query.k / 128) / (warp_blocks * (8 / strips))
}

fn local_warp_blocks(kernel: DenseKernel) -> u32 {
    match kernel {
        DenseKernel::FragmentCluster { warp_blocks } => warp_blocks,
        _ => unreachable!("only cluster decode entries have a cluster"),
    }
}

/// K-chunk CTAs per decode row tile; they form one thread-block cluster.
fn local_cluster(kernel: DenseKernel, query: GemmQuery) -> u32 {
    let warp_blocks = local_warp_blocks(kernel);
    cluster_for_strips(warp_blocks, query, local_strips(warp_blocks, query))
}

fn local_row_tiles(kernel: DenseKernel, query: GemmQuery) -> u32 {
    let rows = local_strips(local_warp_blocks(kernel), query) * 16;
    output_n(query).div_ceil(if query.epilogues.swiglu.is_some() {
        rows / 2
    } else {
        rows
    })
}

/// A cluster decode entry's shape on the shared cluster decode template.
fn local_shape(warp_blocks: u32, query: GemmQuery) -> cluster::Shape {
    cluster::Shape {
        weights: cluster::Weights::of(query.weight).expect("cluster decode weights"),
        token_tile: query.m_capacity.next_power_of_two().max(8),
        k_blocks: query.k / 128,
        strips: local_strips(warp_blocks, query),
        warp_blocks,
        passes: 1,
        segments: 1,
        swiglu: query.epilogues.swiglu.is_some(),
    }
}

/// A cluster decode entry (`run_fragment_decode_cluster`) on the shared
/// cluster decode template: its shape and a body with the entry's ABI.
fn local_decode_template(variant: &GemmVariant) -> Option<(cluster::Shape, String)> {
    let StaticQuery::Gemm(query) = variant.key.query else {
        return None;
    };
    let DenseKernel::FragmentCluster { warp_blocks } = variant.key.dense? else {
        return None;
    };
    let shape = local_shape(warp_blocks, query);
    let swiglu = shape.swiglu;
    let args = cluster::Arguments {
        weights: vec!["map_a".into()],
        outputs: vec!["output".into()],
        rows: vec!["n".into()],
        strides: vec![if swiglu { "n / 2" } else { "n" }.into()],
        activation: "map_b".into(),
        workspace: "partial".into(),
        live_m: "live_m".into(),
        output_f32: "output_f32".into(),
        scales: (shape.weights != cluster::Weights::Bf16).then_some("weight_scales"),
        epilogue: None,
    };
    Some((shape, shape.call(&args)))
}

/// The K partitions every launch of `key` must pass as its live split
/// count: the declared splits of a tiled, fragment decode or router entry.
/// Cluster decode and fragment prefill entries take their reduction from the
/// entry itself, whatever the count.
pub fn required_split_k(key: &CompileKey) -> Option<u8> {
    match key.dense {
        Some(
            DenseKernel::Tiled { splits }
            | DenseKernel::TiledChains { splits }
            | DenseKernel::StripChains { splits, .. }
            | DenseKernel::FragmentDecode { splits }
            | DenseKernel::FragmentRouter { splits },
        ) => Some(splits),
        _ => None,
    }
}

fn fragment_decode_tile_m(kernel: DenseKernel, query: GemmQuery) -> u16 {
    match kernel {
        DenseKernel::FragmentCluster { warp_blocks } => {
            (local_strips(warp_blocks, query) * 16) as u16
        }
        DenseKernel::FragmentRouter { .. } => 16,
        _ if query.epilogues.swiglu.is_some() => 64,
        _ => 128,
    }
}

fn fragment_decode_grid_target(kernel: DenseKernel) -> u32 {
    match kernel {
        // 16 router strips by its K splits in fragment_router.cuh.
        DenseKernel::FragmentRouter { splits } => ROUTER_N / 16 * u32::from(splits),
        _ => SM_COUNT * 4,
    }
}

/// Dense weights for this key use the lane-major MMA fragment layout.
pub fn fragment_weight_enabled(key: &CompileKey) -> bool {
    key.dense.is_some_and(is_fragment)
}

fn tuning_number(key: &'static str) -> u16 {
    // Shapes and plans read these on every call: scan the table once per key.
    static NUMBERS: std::sync::Mutex<Vec<(&str, u16)>> = std::sync::Mutex::new(Vec::new());
    let mut numbers = NUMBERS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(&(_, number)) = numbers.iter().find(|(name, _)| *name == key) {
        return number;
    }
    let number = scan_tuning_number(key);
    numbers.push((key, number));
    number
}

fn scan_tuning_number(key: &str) -> u16 {
    let field = format!("\"{key}\":");
    let tail = TUNING
        .split_once(&field)
        .unwrap_or_else(|| panic!("GEMM tuning field {key} is absent"))
        .1
        .trim_start();
    let digits: String = tail.chars().take_while(char::is_ascii_digit).collect();
    digits
        .parse()
        .unwrap_or_else(|_| panic!("GEMM tuning field {key} is not a number"))
}

fn div_up(n: u32, d: u32) -> u32 {
    n.div_ceil(d)
}

fn output_n(query: GemmQuery) -> u32 {
    if query.epilogues.swiglu.is_some() {
        query.n / 2
    } else {
        query.n
    }
}

fn tile_shared_bytes(
    kernel: DenseKernel,
    query: GemmQuery,
    knobs: CompileKnobs,
    decode: bool,
) -> u32 {
    if is_router(kernel) {
        return 8 * 16 * u32::from(knobs.tile_n) * 4;
    }
    if is_decoded_tiles(kernel, query, knobs, decode)
        || is_staged_tiles(kernel, query, knobs, decode)
    {
        return DECODED_SHARED_BYTES;
    }
    if is_cluster(kernel) {
        // Padded activation rows for one K chunk plus the F32 partial tile.
        let token_tile = u32::from(knobs.tile_n);
        let chunk_bytes = query.k / local_cluster(kernel, query) * 2;
        return token_tile * (chunk_bytes + 16) + 128 * token_tile * 4;
    }
    if is_fragment_decode(kernel) {
        return u32::from(knobs.tile_n) * 128 * 2;
    }
    if is_fragment_prefill(kernel) {
        // One 64-K stage: a 128-byte swizzled activation row per token plus
        // each weight strip's contiguous fragment run; 1024 bytes of slack
        // align the ring for the 128-byte TMA swizzle.
        let strip_stage = if query.weight == WeightFormat::Bf16 {
            2048
        } else {
            1024
        };
        let stage = u32::from(knobs.tile_m) * 128 + (u32::from(knobs.tile_n) / 16) * strip_stage;
        return u32::from(knobs.stages) * stage + 1024;
    }
    let tile_k = u32::from(knobs.tile_k);
    let activation_row_bytes = match query.activation {
        ActivationFormat::Bf16 => tile_k * 2,
        ActivationFormat::Nvfp4 => tile_k / 2,
        _ => tile_k,
    };
    let weight_row_bytes = match query.weight {
        WeightFormat::Bf16 => tile_k * 2,
        WeightFormat::Mxfp4 | WeightFormat::Nvfp4 => tile_k / 2,
        _ => tile_k,
    };
    let (a_row_bytes, b_row_bytes) = if decode {
        (weight_row_bytes, activation_row_bytes)
    } else {
        (activation_row_bytes, weight_row_bytes)
    };
    u32::from(knobs.stages)
        * (u32::from(knobs.tile_m) * a_row_bytes + u32::from(knobs.tile_n) * b_row_bytes)
        + 128
}

fn expected_stages(kernel: DenseKernel, query: GemmQuery, knobs: CompileKnobs, decode: bool) -> u8 {
    if let Some(rows) = strip_rows(kernel) {
        return strip_stages(rows, u32::from(knobs.tile_n), u32::from(knobs.tile_k));
    }
    if is_decoded_tiles(kernel, query, knobs, decode)
        || is_staged_tiles(kernel, query, knobs, decode)
    {
        return DECODED_STAGES;
    }
    if is_fragment_decode(kernel) {
        return 8;
    }
    if is_fragment_prefill(kernel) {
        return if query.weight == WeightFormat::Bf16 {
            3
        } else {
            4
        };
    }
    let wanted = tuning_number(if decode {
        "decode_stages"
    } else {
        "prefill_stages"
    }) as u8;
    let one_stage =
        tile_shared_bytes(kernel, query, CompileKnobs { stages: 1, ..knobs }, decode) - 128;
    if one_stage * u32::from(wanted) + 128 + 1152 <= 101_376 {
        wanted
    } else {
        2
    }
}

fn dense_supported(kernel: DenseKernel, query: &GemmQuery) -> Result<(), Refusal> {
    query.validate()?;
    if query.grouped.is_some() {
        return Err(Refusal("grouped GEMM belongs to the MoE engine"));
    }
    kernel_supported(kernel, *query)?;
    // Only the fragment kernels address scales per 64 rows, and their output
    // rows must be whole scale groups.
    if matches!(query.weight, WeightFormat::Fp8Rows64x128(_))
        && (!is_fragment(kernel) || !output_n(*query).is_multiple_of(64))
    {
        return Err(Refusal(
            "64-row FP8 scales need a fragment GEMM over whole row groups",
        ));
    }
    let fp8_weight = matches!(
        query.weight,
        WeightFormat::Fp8PerTensor
            | WeightFormat::Fp8PerChannel
            | WeightFormat::Fp8Block128x128(_)
            | WeightFormat::Fp8Rows64x128(_)
    );
    match query.activation {
        ActivationFormat::F32 => {
            return Err(Refusal(
                "F32 projections require a registered F32 implementation",
            ));
        }
        ActivationFormat::Bf16
            if query.weight == WeightFormat::Bf16
                || fp8_weight
                || matches!(
                    query.weight,
                    WeightFormat::Mxfp8 | WeightFormat::Mxfp4 | WeightFormat::Nvfp4
                ) => {}
        ActivationFormat::Fp8Group128(_) if fp8_weight => {}
        ActivationFormat::Mxfp8
            if matches!(query.weight, WeightFormat::Mxfp8 | WeightFormat::Mxfp4) => {}
        ActivationFormat::Nvfp4 if query.weight == WeightFormat::Nvfp4 => {}
        _ => {
            return Err(Refusal(
                "dense engine does not support this weight and activation pair",
            ));
        }
    }
    if query.epilogues.swiglu.is_some() && query.activation != ActivationFormat::Bf16 {
        return Err(Refusal("dense SwiGLU needs BF16 activations"));
    }
    if query.epilogues.output_quantization.is_some()
        || query.epilogues.weighted_moe_combine
        || query.epilogues.bias
        || query.epilogues.gelu.is_some()
    {
        return Err(Refusal("dense GEMM epilogues are not implemented"));
    }
    Ok(())
}

fn shape(key: &CompileKey) -> Result<(DenseKernel, GemmQuery, CompileKnobs, bool), Refusal> {
    key.knobs.validate()?;
    if key.target != Target::Sm120f {
        return Err(Refusal("dense GEMM requires SM120f"));
    }
    let StaticQuery::Gemm(query) = key.query else {
        return Err(Refusal("compile key is not GEMM"));
    };
    let kernel = kernel(key)?;
    dense_supported(kernel, &query)?;
    let decode = query.m_capacity <= 32;
    let (tile_m, tile_n, tile_k) = (
        u32::from(key.knobs.tile_m),
        u32::from(key.knobs.tile_n),
        u32::from(key.knobs.tile_k),
    );
    if query.epilogues.swiglu.is_some()
        && !(if decode {
            output_n(query).is_multiple_of(tile_m)
        } else {
            output_n(query).is_multiple_of(tile_n)
        })
    {
        return Err(Refusal("SwiGLU halves must align to output tiles"));
    }
    let fragment = is_fragment(kernel);
    let decoded = is_decoded_tiles(kernel, query, key.knobs, decode);
    if key.knobs.stages != expected_stages(kernel, query, key.knobs, decode)
        || key.knobs.warps
            != if is_router(kernel) || is_cluster(kernel) {
                8
            } else if is_fragment_decode(kernel) {
                (tile_m / 16) as u8
            } else if fragment {
                8
            } else if let Some(rows) = strip_rows(kernel) {
                (rows / 16 + 1) as u8
            } else {
                (THREADS / 32) as u8
            }
        || key.knobs.swizzle != 128
        || tile_k
            != if is_fragment_prefill(kernel) {
                64
            } else if fragment {
                128
            } else {
                match query.weight {
                    WeightFormat::Bf16 => 64,
                    WeightFormat::Mxfp4 | WeightFormat::Nvfp4 => 256,
                    _ if decoded => 64,
                    _ => 128,
                }
            }
        || !tile_m.is_multiple_of(16)
        || !tile_n.is_multiple_of(8)
        || !(if is_fragment_decode(kernel) {
            [16, 32, 64, 128].contains(&tile_m)
        } else if let Some(rows) = strip_rows(kernel) {
            tile_m == rows
        } else {
            [64, 128].contains(&tile_m)
        })
        || ![8, 16, 32, 64, 128, 256].contains(&tile_n)
        || (decode
            && (tile_m
                != if fragment {
                    u32::from(fragment_decode_tile_m(kernel, query))
                } else {
                    strip_rows(kernel).unwrap_or(64)
                }
                || tile_n != query.m_capacity.next_power_of_two().max(8)))
        || (is_fragment_prefill(kernel)
            && ((tile_m, tile_n) != (128, 128)
                || !output_n(query).is_multiple_of(64)
                || !query.k.is_multiple_of(128)))
        || (!decode
            && !fragment
            && !(if query.weight == WeightFormat::Bf16
                || matches!(
                    query.activation,
                    ActivationFormat::Fp8Group128(_)
                        | ActivationFormat::Mxfp8
                        | ActivationFormat::Nvfp4
                ) {
                [(64, 64), (64, 128), (128, 128), (128, 256)].contains(&(tile_m, tile_n))
            } else if decoded {
                [(64, 128), (128, 128)].contains(&(tile_m, tile_n))
            } else {
                [(64, 64), (64, 128)].contains(&(tile_m, tile_n))
            }))
    {
        return Err(Refusal("unsupported dense GEMM tile or pipeline knobs"));
    }
    if matches!(kernel, DenseKernel::TiledChains { .. })
        && !decode
        && ![(64, 64), (64, 128), (128, 128)].contains(&(tile_m, tile_n))
    {
        return Err(Refusal(
            "warp-tiled chains take 64 x 64, 64 x 128 or 128 x 128 prefill tiles",
        ));
    }
    if is_staged_tiles(kernel, query, key.knobs, decode)
        && (!query.n.is_multiple_of(128)
            || !query.k.is_multiple_of(64)
            || query.epilogues.swiglu.is_some())
    {
        return Err(Refusal(
            "staged weight tiles take whole 128 x 64 weight blocks without SwiGLU",
        ));
    }
    if decoded && (!query.n.is_multiple_of(128) || !query.k.is_multiple_of(128)) {
        return Err(Refusal(
            "decoded weight tiles take whole 128 x 128 weight blocks",
        ));
    }
    let smem = tile_shared_bytes(kernel, query, key.knobs, decode) + 1152;
    if smem > 101_376 {
        return Err(Refusal("dense GEMM shared memory exceeds SM120 limit"));
    }
    if let DenseKernel::FragmentPrefill { segments } = kernel
        && key.knobs.split_k_capacity != segments
    {
        return Err(Refusal(
            "fragment prefill split capacity is its declared K segments",
        ));
    }
    if let DenseKernel::FragmentRouter { splits } = kernel
        && key.knobs.split_k_capacity != splits
    {
        return Err(Refusal("router split capacity is its declared K splits"));
    }
    if let DenseKernel::Tiled { splits }
    | DenseKernel::TiledChains { splits }
    | DenseKernel::StripChains { splits, .. }
    | DenseKernel::FragmentDecode { splits } = kernel
    {
        if key.knobs.split_k_capacity < splits {
            return Err(Refusal(
                "dense split capacity is below the declared K splits",
            ));
        }
        if u32::from(splits) > div_up(query.k, tile_k) {
            return Err(Refusal("declared K splits exceed the K chunks"));
        }
    }
    Ok((kernel, query, key.knobs, decode))
}

/// The K partitions a launch of `key` uses: its declared splits, or one for
/// cluster decode and fragment prefill entries, whose arguments carry their
/// own reduction (`split_k_capacity` segments for prefill).
pub fn choose_split_k(key: &CompileKey) -> Result<u8, Refusal> {
    shape(key)?;
    Ok(required_split_k(key).unwrap_or(1))
}

/// The counts of a launch of `key` over `m` rows: one batch and context,
/// the eight routed rows a dense GEMM takes, and the entry's declared K
/// splits (`required_split_k`, else one).
pub fn row_counts(key: &CompileKey, m: u32) -> RuntimeCounts {
    RuntimeCounts {
        m,
        batch: 1,
        context: 1,
        routed_rows: 8,
        split_k: required_split_k(key).unwrap_or(1),
    }
}

/// The dynamic shared memory a launch of `key` takes at its full row
/// capacity: what its symbol is configured for at load.
pub fn capacity_shared_bytes(key: &CompileKey) -> Result<u32, Refusal> {
    let StaticQuery::Gemm(query) = key.query else {
        return Err(Refusal("dense entry has no GEMM query"));
    };
    Ok(plan(key, row_counts(key, query.m_capacity))?.dynamic_shared_bytes)
}

/// The scratch workspace a launch of `key` takes at its full row capacity:
/// the largest any of its launches asks for.
pub fn capacity_workspace_bytes(key: &CompileKey) -> Result<u64, Refusal> {
    let StaticQuery::Gemm(query) = key.query else {
        return Err(Refusal("dense entry has no GEMM query"));
    };
    Ok(plan(key, row_counts(key, query.m_capacity))?.workspace_bytes)
}

/// The tensor maps a bundle binds for a dense entry over row-major BF16
/// activations of `m_capacity` rows (planner buffer `activation`) and its
/// `k` x `n` weights (`weight`), in the order the entry takes them: a
/// `decode` entry the weight first. Fragment-major BF16 weights bind
/// unswizzled; the entry reads them as a plain buffer, and the map only
/// carries the logical descriptor the engine validates. An activation box
/// wider than 64 columns is unswizzled.
pub fn fragment_maps(
    key: &CompileKey,
    query: &GemmQuery,
    activation: u64,
    weight: u64,
    decode: bool,
) -> [TensorMap; 2] {
    let knobs = key.knobs;
    let map = |id, rows, tile_rows, element, swizzle| {
        let bytes = if element == TensorMapElement::Bf16 {
            2
        } else {
            1
        };
        TensorMap {
            base_buffer: id,
            dims: vec![u64::from(query.k), u64::from(rows)],
            strides_bytes: vec![bytes, u64::from(query.k) * bytes],
            box_dims: vec![u32::from(knobs.tile_k), u32::from(tile_rows)],
            element_strides: vec![1, 1],
            swizzle,
            interleave: TensorMapInterleave::None,
            l2_promotion: TensorMapL2Promotion::B256,
            oob_fill: TensorMapOobFill::Zero,
            element,
        }
    };
    let bf16 = query.weight == WeightFormat::Bf16;
    let weight_map = map(
        weight,
        query.n,
        if decode { knobs.tile_m } else { knobs.tile_n },
        if bf16 {
            TensorMapElement::Bf16
        } else {
            TensorMapElement::Fp8E4M3
        },
        if fragment_weight_enabled(key) && bf16 {
            TensorMapSwizzle::None
        } else {
            TensorMapSwizzle::B128
        },
    );
    let activation_map = map(
        activation,
        query.m_capacity,
        if decode { knobs.tile_n } else { knobs.tile_m },
        TensorMapElement::Bf16,
        if knobs.tile_k > 64 {
            TensorMapSwizzle::None
        } else {
            TensorMapSwizzle::B128
        },
    );
    if decode {
        [weight_map, activation_map]
    } else {
        [activation_map, weight_map]
    }
}

/// Build the two TMA maps in the order expected by the swapped decode kernel.
/// An unswizzled inner box is required when its byte width exceeds 128.
pub fn tensor_maps(
    key: &CompileKey,
    weight_buffer: u64,
    activation_buffer: u64,
) -> Result<[TensorMap; 2], Refusal> {
    let (kernel, query, knobs, decode) = shape(key)?;
    let weight_element = match query.weight {
        WeightFormat::Bf16 => TensorMapElement::Bf16,
        WeightFormat::Mxfp4 | WeightFormat::Nvfp4 => TensorMapElement::U4Align8B,
        _ => TensorMapElement::Fp8E4M3,
    };
    let activation_element = match query.activation {
        ActivationFormat::Bf16 => TensorMapElement::Bf16,
        ActivationFormat::Nvfp4 => TensorMapElement::U4Align8B,
        _ => TensorMapElement::Fp8E4M3,
    };
    let make = |base_buffer, rows, tile_rows: u16, element, row_stride, unswizzled| {
        let bytes_per_row = match element {
            TensorMapElement::Bf16 => u64::from(query.k) * 2,
            TensorMapElement::U4Align8B => u64::from(query.k) / 2,
            _ => u64::from(query.k),
        };
        let box_bytes = match element {
            TensorMapElement::Bf16 => u32::from(knobs.tile_k) * 2,
            TensorMapElement::U4Align8B => u32::from(knobs.tile_k) / 2,
            _ => u32::from(knobs.tile_k),
        };
        TensorMap {
            base_buffer,
            dims: vec![u64::from(query.k), u64::from(rows)],
            strides_bytes: vec![
                if element == TensorMapElement::Bf16 {
                    2
                } else {
                    1
                },
                bytes_per_row,
            ],
            box_dims: vec![u32::from(knobs.tile_k), u32::from(tile_rows) * row_stride],
            element_strides: vec![1, row_stride],
            swizzle: if unswizzled || box_bytes > 128 {
                TensorMapSwizzle::None
            } else {
                TensorMapSwizzle::B128
            },
            interleave: kernel_core::contract::TensorMapInterleave::None,
            l2_promotion: kernel_core::contract::TensorMapL2Promotion::B128,
            oob_fill: kernel_core::contract::TensorMapOobFill::Zero,
            element,
        }
    };
    let stride = if query.epilogues.swiglu == Some(SwigluColumns::Interleaved) {
        2
    } else {
        1
    };
    let weight = make(
        weight_buffer,
        query.n,
        if decode { knobs.tile_m } else { knobs.tile_n },
        weight_element,
        stride,
        (is_fragment(kernel) && weight_element == TensorMapElement::Bf16)
            || is_decoded_tiles(kernel, query, knobs, decode),
    );
    let activation = make(
        activation_buffer,
        query.m_capacity,
        if decode { knobs.tile_n } else { knobs.tile_m },
        activation_element,
        1,
        false,
    );
    Ok(if decode {
        [weight, activation]
    } else {
        [activation, weight]
    })
}

/// Plan from prepared capacity. M at launch may be smaller than that capacity.
pub fn plan(key: &CompileKey, counts: RuntimeCounts) -> Result<LaunchPlan, Refusal> {
    let (kernel, query, knobs, decode) = shape(key)?;
    if required_split_k(key).is_some_and(|required| counts.split_k != required) {
        return Err(Refusal("a launch must use the entry's declared K splits"));
    }
    if counts.m == 0 || counts.m > query.m_capacity {
        return Err(Refusal("live M exceeds prepared GEMM capacity"));
    }
    if counts.split_k == 0 || counts.split_k > knobs.split_k_capacity {
        return Err(Refusal("split-K exceeds prepared capacity"));
    }
    if u32::from(counts.split_k) > div_up(query.k, u32::from(knobs.tile_k)) {
        return Err(Refusal("split-K exceeds K chunks"));
    }
    let tiles_a = if decode {
        div_up(output_n(query), u32::from(knobs.tile_m))
    } else {
        div_up(query.m_capacity, u32::from(knobs.tile_m))
    };
    let tiles_b = if is_cluster(kernel) {
        1
    } else if decode {
        div_up(query.m_capacity, u32::from(knobs.tile_n))
    } else {
        div_up(output_n(query), u32::from(knobs.tile_n))
    };
    let tiles = u64::from(tiles_a) * u64::from(tiles_b);
    let scheduled_units = tiles
        .checked_mul(u64::from(knobs.split_k_capacity))
        .and_then(|n| u32::try_from(n).ok())
        .ok_or(Refusal("GEMM grid exceeds u32"))?;
    let grid = if is_balanced_decoded(kernel, query, knobs, decode)
        || is_staged_tiles(kernel, query, knobs, decode)
    {
        // One item per CTA: the capacity's 128-row tiles and a wave of
        // halves (`balanced_tiles.cuh`).
        u32::try_from(
            u64::from(div_up(query.m_capacity, 128)) * u64::from(tiles_b) + u64::from(SM_COUNT),
        )
        .map_err(|_| Refusal("GEMM grid exceeds u32"))?
    } else if matches!(kernel, DenseKernel::TiledChains { .. }) && !decode {
        // Warp tiles run one tile per CTA (`run_a16_warps`).
        scheduled_units
    } else if let DenseKernel::StripChains { splits, .. } = kernel {
        // One CTA per tile and declared K split (`run_a16_strips`).
        u32::try_from(tiles * u64::from(splits)).map_err(|_| Refusal("GEMM grid exceeds u32"))?
    } else {
        scheduled_units.min(if is_fragment_decode(kernel) {
            fragment_decode_grid_target(kernel)
        } else {
            SM_COUNT
        })
    };
    if is_cluster(kernel) {
        let cluster = local_cluster(kernel, query);
        let row_tiles = local_row_tiles(kernel, query);
        let result = LaunchPlan {
            grid: [row_tiles * cluster, 1, 1],
            block: [256, 1, 1],
            dynamic_shared_bytes: tile_shared_bytes(kernel, query, knobs, decode),
            // One F32 partial tile per clustered CTA.
            workspace_bytes: if cluster > 1 {
                u64::from(row_tiles * cluster) * 128 * u64::from(knobs.tile_n) * 4
            } else {
                0
            },
            cluster: [cluster, 1, 1],
            pdl: true,
        };
        result.validate()?;
        return Ok(result);
    }
    if is_fragment_prefill(kernel) {
        // Stream-K keeps PREFILL_SEGMENTS F32 partial tiles per CTA.
        let partial_floats =
            u64::from(grid) * u64::from(PREFILL_SEGMENTS) * u64::from(MATH_WARPS) * 16 * 4 * 32;
        let result = LaunchPlan {
            grid: [grid, 1, 1],
            block: [THREADS, 1, 1],
            dynamic_shared_bytes: tile_shared_bytes(kernel, query, knobs, decode),
            workspace_bytes: partial_floats * 4,
            cluster: [1, 1, 1],
            pdl: true,
        };
        result.validate()?;
        return Ok(result);
    }
    let scratch_units = if let DenseKernel::TiledChains { .. } = kernel
        && !decode
    {
        // Warp tiles write their outputs directly (`run_a16_warps`).
        0
    } else if let DenseKernel::StripChains { splits, .. } = kernel {
        // One split adds its chains in registers; more keep one F32
        // temporary tile per K chunk, whatever the count (`run_a16_strips`).
        if splits == 1 {
            0
        } else {
            tiles * u64::from(div_up(query.k, u32::from(knobs.tile_k)))
        }
    } else if let DenseKernel::TiledChains { .. } = kernel {
        // One F32 temporary tile per K chunk's 128-K block, whatever the
        // split count (`run_a16_chains`).
        tiles
            * u64::from(div_up(query.k, u32::from(knobs.tile_k)))
            * u64::from(knobs.tile_k.div_ceil(128))
    } else if query.epilogues.swiglu.is_some() && knobs.split_k_capacity == 1 {
        u64::from(grid)
    } else {
        u64::from(scheduled_units)
    };
    let projection_count = if query.epilogues.swiglu.is_some() {
        2
    } else {
        1
    };
    let partial_elements = scratch_units
        .checked_mul(u64::from(knobs.tile_m) * u64::from(knobs.tile_n))
        .and_then(|n| n.checked_mul(projection_count))
        .ok_or(Refusal("GEMM workspace size overflow"))?;
    let workspace = partial_elements
        .checked_add(tiles)
        .and_then(|n| n.checked_mul(4))
        .ok_or(Refusal("GEMM workspace size overflow"))?;
    let result = LaunchPlan {
        grid: [grid, 1, 1],
        block: [
            if is_router(kernel) || is_cluster(kernel) {
                256
            } else if is_fragment_decode(kernel) {
                u32::from(knobs.tile_m / 16) * 32
            } else if is_fragment_prefill(kernel) {
                256
            } else if let Some(rows) = strip_rows(kernel) {
                (rows / 16 + 1) * 32
            } else {
                THREADS
            },
            1,
            1,
        ],
        dynamic_shared_bytes: tile_shared_bytes(kernel, query, knobs, decode),
        workspace_bytes: workspace,
        cluster: [1, 1, 1],
        pdl: true,
    };
    result.validate()?;
    Ok(result)
}

/// The parameter order matches the generated C wrapper exactly.
#[derive(Clone, Copy, Debug)]
pub struct GemmBuffers {
    pub output: u64,
    pub partial: u64,
    pub semaphores: u64,
    pub weight_scales: Option<u64>,
    pub activation_scales: Option<u64>,
    pub residual: Option<u64>,
    pub weight_global_scale: Option<u64>,
    pub activation_global_scale: Option<u64>,
}

impl GemmBuffers {
    /// Prepared external semaphore bytes, initialized to zero by the owner.
    /// Fragment prefill reserves a sufficient bound across all live row counts.
    pub fn semaphore_capacity_bytes(key: &CompileKey) -> Result<u64, Refusal> {
        let (kernel, query, knobs, decode) = shape(key)?;
        let prepared = plan(key, row_counts(key, query.m_capacity))?;
        let words = match kernel {
            DenseKernel::Tiled { splits: 1 }
            | DenseKernel::FragmentDecode { splits: 1 }
            | DenseKernel::FragmentCluster { .. } => 0,
            // Warp-tiled prefill chains take no ticket.
            DenseKernel::TiledChains { .. } if !decode => 0,
            // One-split strip chains add in registers.
            DenseKernel::StripChains { splits: 1, .. } => 0,
            // Chain splits take a ticket per tile at every split count.
            DenseKernel::Tiled { .. }
            | DenseKernel::TiledChains { .. }
            | DenseKernel::StripChains { .. } => {
                let (a, b) = if decode {
                    (
                        output_n(query).div_ceil(u32::from(knobs.tile_m)),
                        query.m_capacity.div_ceil(u32::from(knobs.tile_n)),
                    )
                } else {
                    (
                        query.m_capacity.div_ceil(u32::from(knobs.tile_m)),
                        output_n(query).div_ceil(u32::from(knobs.tile_n)),
                    )
                };
                u64::from(a) * u64::from(b)
            }
            DenseKernel::FragmentDecode { .. } => {
                u64::from(output_n(query).div_ceil(u32::from(knobs.tile_m)))
            }
            DenseKernel::FragmentRouter { .. } => u64::from(ROUTER_N / 16),
            DenseKernel::FragmentPrefill { segments } => {
                // Smaller live batches can need tail counters when the full
                // capacity has only whole waves. Each owner uses two words.
                if u32::from(segments).min(query.k / 128) > 1 {
                    u64::from(prepared.grid[0]) * 2
                } else {
                    0
                }
            }
        };
        words
            .checked_mul(4)
            .ok_or(Refusal("GEMM semaphore capacity overflow"))
    }
}

/// Validate descriptors and pack only ABI argument kinds.
pub fn arguments(
    key: &CompileKey,
    counts: RuntimeCounts,
    maps: [TensorMap; 2],
    buffers: GemmBuffers,
) -> Result<ArgumentList, Refusal> {
    let [map_a, map_b] = maps;
    let (kernel, query, knobs, decode) = shape(key)?;
    plan(key, counts)?;
    if (query.weight == WeightFormat::Bf16 && buffers.weight_scales.is_some())
        || (query.weight != WeightFormat::Bf16 && buffers.weight_scales.is_none())
        || (query.activation == ActivationFormat::Bf16 && buffers.activation_scales.is_some())
        || (query.activation != ActivationFormat::Bf16 && buffers.activation_scales.is_none())
    {
        return Err(Refusal("scale pointer does not match GEMM format"));
    }
    if query.epilogues.residual_add != buffers.residual.is_some() {
        return Err(Refusal("residual pointer does not match GEMM epilogue"));
    }
    if (query.weight == WeightFormat::Nvfp4) != buffers.weight_global_scale.is_some()
        || (query.activation == ActivationFormat::Nvfp4)
            != buffers.activation_global_scale.is_some()
    {
        return Err(Refusal(
            "NVFP4 global scale pointer does not match GEMM format",
        ));
    }
    let weight_element = match query.weight {
        WeightFormat::Bf16 => TensorMapElement::Bf16,
        WeightFormat::Mxfp4 | WeightFormat::Nvfp4 => TensorMapElement::U4Align8B,
        _ => TensorMapElement::Fp8E4M3,
    };
    let activation_element = match query.activation {
        ActivationFormat::Bf16 => TensorMapElement::Bf16,
        ActivationFormat::Nvfp4 => TensorMapElement::U4Align8B,
        _ => TensorMapElement::Fp8E4M3,
    };
    let map_matches = |map: &TensorMap, rows: u32, tile_rows: u16, element, swizzle, row_stride| {
        map.dims == [u64::from(query.k), u64::from(rows)]
            && map.box_dims == [u32::from(knobs.tile_k), u32::from(tile_rows) * row_stride]
            && map.swizzle == swizzle
            && map.element == element
    };
    let activation_swizzle = if (query.activation == ActivationFormat::Bf16 && knobs.tile_k > 64)
        || (matches!(
            query.activation,
            ActivationFormat::Fp8Group128(_) | ActivationFormat::Mxfp8
        ) && knobs.tile_k > 128)
    {
        TensorMapSwizzle::None
    } else {
        TensorMapSwizzle::B128
    };
    let weight_stride = if query.epilogues.swiglu == Some(SwigluColumns::Interleaved) {
        2
    } else {
        1
    };
    let valid_maps = if decode {
        map_matches(
            &map_a,
            query.n,
            knobs.tile_m,
            weight_element,
            if is_fragment(kernel) && weight_element == TensorMapElement::Bf16 {
                TensorMapSwizzle::None
            } else {
                TensorMapSwizzle::B128
            },
            weight_stride,
        ) && map_matches(
            &map_b,
            query.m_capacity,
            knobs.tile_n,
            activation_element,
            activation_swizzle,
            1,
        ) && map_a.element_strides == [1, weight_stride]
            && map_b.element_strides == [1, 1]
    } else {
        map_matches(
            &map_a,
            query.m_capacity,
            knobs.tile_m,
            activation_element,
            activation_swizzle,
            1,
        ) && map_matches(
            &map_b,
            query.n,
            knobs.tile_n,
            weight_element,
            if (is_fragment(kernel) && weight_element == TensorMapElement::Bf16)
                || is_decoded_tiles(kernel, query, knobs, decode)
            {
                TensorMapSwizzle::None
            } else {
                TensorMapSwizzle::B128
            },
            weight_stride,
        ) && map_a.element_strides == [1, 1]
            && map_b.element_strides == [1, weight_stride]
    };
    if !valid_maps {
        return Err(Refusal("tensor maps do not match dense GEMM tiles"));
    }
    let args = ArgumentList(vec![
        if is_fragment_decode(kernel) {
            Argument::DeviceBuffer(map_a.base_buffer)
        } else {
            Argument::TensorMap(map_a)
        },
        if is_fragment(kernel) {
            Argument::DeviceBuffer(map_b.base_buffer)
        } else {
            Argument::TensorMap(map_b)
        },
        Argument::DeviceBuffer(buffers.output),
        Argument::DeviceBuffer(buffers.partial),
        Argument::DeviceBuffer(buffers.semaphores),
        Argument::U32(counts.m),
        Argument::U32(query.n),
        Argument::U32(query.k),
        // Fragment prefill: the entry's fixed K segments at every live M.
        Argument::U32(u32::from(if is_fragment_prefill(kernel) {
            knobs.split_k_capacity
        } else {
            counts.split_k
        })),
        Argument::U32(u32::from(query.output == DType::F32)),
        buffers
            .weight_scales
            .map_or(Argument::NullDevicePointer, Argument::DeviceBuffer),
        buffers
            .activation_scales
            .map_or(Argument::NullDevicePointer, Argument::DeviceBuffer),
        buffers
            .residual
            .map_or(Argument::NullDevicePointer, Argument::DeviceBuffer),
        buffers
            .weight_global_scale
            .map_or(Argument::NullDevicePointer, Argument::DeviceBuffer),
        buffers
            .activation_global_scale
            .map_or(Argument::NullDevicePointer, Argument::DeviceBuffer),
    ]);
    args.validate()?;
    Ok(args)
}

/// Opaque buffers for a GEMM followed by a row quantization epilogue.
#[derive(Clone, Copy, Debug)]
pub struct QuantizedBuffers {
    pub gemm: GemmBuffers,
    pub codes: u64,
    pub scales: u64,
    pub live_rows: u64,
    pub row_amax: Option<u64>,
    pub global_scale: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct QuantizedStage {
    pub symbol: String,
    pub launch: LaunchPlan,
    pub arguments: ArgumentList,
}

fn row_quant_stage(
    catalog: &[row::RowInstantiation],
    symbol: &str,
    capacity: u32,
    counts: RuntimeCounts,
    values: Vec<Argument>,
) -> Result<QuantizedStage, Refusal> {
    let entry = catalog
        .iter()
        .find(|entry| entry.symbol == symbol)
        .ok_or(Refusal("row quantization entry is not instantiated"))?;
    Ok(QuantizedStage {
        symbol: symbol.into(),
        launch: entry.plan(capacity, counts)?,
        arguments: entry.pack_arguments(values)?,
    })
}

/// Compose a BF16 GEMM result with audited row quantizers. The caller launches
/// these stages in order on one stream and owns every buffer and live-row count.
pub fn quantized_pipeline(
    item: &InstantiationSpec,
    format: OutputQuantization,
    counts: RuntimeCounts,
    maps: [TensorMap; 2],
    buffers: QuantizedBuffers,
) -> Result<Vec<QuantizedStage>, Refusal> {
    if item.family != OpFamily::Gemm {
        return Err(Refusal("quantized pipeline requires dense GEMM"));
    }
    let (_, query, _, _) = shape(&item.key)?;
    let width = output_n(query);
    if query.output != DType::Bf16 || ![4096, 8192].contains(&width) {
        return Err(Refusal(
            "row quantization requires BF16 output width 4096 or 8192",
        ));
    }
    let label = match format {
        OutputQuantization::Fp8Group128(kernel_core::contract::ScaleMode::ExactF32) => "fp8_exact",
        OutputQuantization::Fp8Group128(kernel_core::contract::ScaleMode::PowerOfTwoF32) => {
            "fp8_pow2"
        }
        OutputQuantization::Mxfp8 => "mxfp8",
        OutputQuantization::Nvfp4 => "nvfp4",
        OutputQuantization::Mxfp4 => {
            return Err(Refusal(
                "MXFP4 output quantization is not enabled for dense GEMM",
            ));
        }
    };
    if format == OutputQuantization::Nvfp4
        && (buffers.row_amax.is_none() || buffers.global_scale.is_none())
    {
        return Err(Refusal(
            "NVFP4 output needs row maxima and global scale buffers",
        ));
    }
    let mut stages = vec![QuantizedStage {
        symbol: item.symbol.clone(),
        launch: plan(&item.key, counts)?,
        arguments: arguments(&item.key, counts, maps, buffers.gemm)?,
    }];
    let mut rows =
        row::builders::quantize_entries(Profile::StdDenseMoe, width, &width.to_string())?;
    rows.push(row::builders::nvfp4_global_scale(Profile::StdDenseMoe)?);
    if format == OutputQuantization::Nvfp4 {
        stages.push(row_quant_stage(
            &rows,
            &format!("chop_row_nvfp4_row_amax_{width}_v1"),
            query.m_capacity,
            counts,
            vec![
                Argument::DeviceBuffer(buffers.gemm.output),
                Argument::DeviceBuffer(buffers.row_amax.unwrap()),
                Argument::DeviceBuffer(buffers.live_rows),
            ],
        )?);
        stages.push(row_quant_stage(
            &rows,
            "chop_row_nvfp4_global_scale_v1",
            query.m_capacity,
            counts,
            vec![
                Argument::DeviceBuffer(buffers.row_amax.unwrap()),
                Argument::DeviceBuffer(buffers.global_scale.unwrap()),
                Argument::DeviceBuffer(buffers.live_rows),
            ],
        )?);
    }
    stages.push(row_quant_stage(
        &rows,
        &format!("chop_row_quant_{label}_{width}_v1"),
        query.m_capacity,
        counts,
        vec![
            Argument::DeviceBuffer(buffers.gemm.output),
            Argument::DeviceBuffer(buffers.codes),
            Argument::DeviceBuffer(buffers.scales),
            buffers
                .global_scale
                .map_or(Argument::NullDevicePointer, Argument::DeviceBuffer),
            Argument::DeviceBuffer(buffers.live_rows),
        ],
    )?);
    Ok(stages)
}

fn params(fragment: bool, decode: bool) -> Vec<EntryParam> {
    let result = vec![
        if fragment && decode {
            EntryParam::DevicePointer {
                name: "map_a".into(),
                cxx_type: "const uint8_t".into(),
            }
        } else {
            EntryParam::TensorMap {
                name: "map_a".into(),
            }
        },
        if fragment {
            EntryParam::DevicePointer {
                name: "map_b".into(),
                cxx_type: if decode {
                    "const uint16_t"
                } else {
                    "const uint8_t"
                }
                .into(),
            }
        } else {
            EntryParam::TensorMap {
                name: "map_b".into(),
            }
        },
        EntryParam::DevicePointer {
            name: "output".into(),
            cxx_type: "void".into(),
        },
        EntryParam::DevicePointer {
            name: "partial".into(),
            cxx_type: "float".into(),
        },
        EntryParam::DevicePointer {
            name: "semaphores".into(),
            cxx_type: "uint32_t".into(),
        },
        EntryParam::U32 {
            name: "live_m".into(),
        },
        EntryParam::U32 { name: "n".into() },
        EntryParam::U32 { name: "k".into() },
        EntryParam::U32 {
            name: "splits".into(),
        },
        EntryParam::U32 {
            name: "output_f32".into(),
        },
        EntryParam::DevicePointer {
            name: "weight_scales".into(),
            cxx_type: "const float".into(),
        },
        EntryParam::DevicePointer {
            name: "activation_scales".into(),
            cxx_type: "const float".into(),
        },
        EntryParam::DevicePointer {
            name: "residual".into(),
            cxx_type: "const void".into(),
        },
        EntryParam::DevicePointer {
            name: "weight_global_scale".into(),
            cxx_type: "const float".into(),
        },
        EntryParam::DevicePointer {
            name: "activation_global_scale".into(),
            cxx_type: "const float".into(),
        },
    ];
    result
}

pub fn cubin_spec(instantiations: &[InstantiationSpec]) -> Result<CubinSpec, Refusal> {
    let mut entries = Vec::new();
    let mut mxfp4_a16 = false;
    let mut tiled_chains = false;
    let mut tiled_warps = false;
    let mut decoded_tiles = false;
    let mut decoded_balanced = false;
    let mut staged_tiles = false;
    let mut strip_chains = false;
    let mut strip_chains_fp8 = false;
    for item in instantiations {
        item.validate()?;
        if item.family != OpFamily::Gemm {
            return Err(Refusal("non-GEMM instantiation"));
        }
        let (kernel, query, knobs, decode) = shape(&item.key)?;
        let kind = match query.weight {
            WeightFormat::F32 => {
                return Err(Refusal(
                    "F32 projections require a registered F32 implementation",
                ));
            }
            WeightFormat::Fp8Block32x32 => {
                return Err(Refusal("32x32 UE8M0 weights require the MXFP8 fold engine"));
            }
            WeightFormat::Bf16 => "bf16",
            WeightFormat::Fp8PerTensor => "fp8_tensor",
            WeightFormat::Fp8PerChannel => "fp8_channel",
            WeightFormat::Fp8Block128x128(_) => "fp8_block128",
            WeightFormat::Fp8Rows64x128(_) => "fp8_row64",
            WeightFormat::Mxfp8 => "mxfp8",
            WeightFormat::Mxfp4 => "mxfp4",
            WeightFormat::Nvfp4 => "nvfp4",
        };
        let residual = query.epilogues.residual_add;
        let fragment = is_fragment(kernel);
        mxfp4_a16 |= !fragment
            && query.weight == WeightFormat::Mxfp4
            && query.activation == ActivationFormat::Bf16;
        let body = if is_router(kernel) {
            format!(
                "chop::gemm::run_fragment_router<{}>(map_a, map_b, output, partial, semaphores, live_m, n, k, splits, output_f32, weight_scales, residual);",
                knobs.tile_n
            )
        } else if let DenseKernel::FragmentCluster { warp_blocks } = kernel {
            format!(
                "chop::gemm::run_fragment_decode_cluster<chop::gemm::WeightKind::{kind}, {}, {}, {}, {}>(map_a, map_b, output, partial, semaphores, live_m, n, k, splits, output_f32, weight_scales, residual);",
                knobs.tile_n,
                query.epilogues.swiglu.is_some(),
                query.k / 128,
                local_strips(warp_blocks, query)
            )
        } else if is_fragment_decode(kernel) {
            format!(
                "chop::gemm::run_fragment_decode<chop::gemm::WeightKind::{kind}, {}, false, {}, {}>(map_a, map_b, output, partial, semaphores, live_m, n, k, splits, output_f32, weight_scales, residual);",
                knobs.tile_n,
                query.epilogues.swiglu.is_some(),
                knobs.tile_m
            )
        } else if is_fragment_prefill(kernel) {
            format!(
                "chop::gemm::run_fragment_prefill<chop::gemm::WeightKind::{kind}, {}, 128>(map_a, map_b, output, partial, semaphores, live_m, n, k, splits, output_f32, weight_scales, residual);",
                query.epilogues.swiglu.is_some(),
            )
        } else if let Some(layout) = query.epilogues.swiglu {
            let interleaved = layout == SwigluColumns::Interleaved;
            format!(
                "chop::gemm::run_a16_swiglu<chop::gemm::WeightKind::{kind}, {decode}, {residual}, {interleaved}, {}, {}, {}, {}>(map_a, map_b, output, partial, semaphores, live_m, n, k, splits, output_f32, weight_scales, residual, weight_global_scale);",
                knobs.tile_m, knobs.tile_n, knobs.tile_k, knobs.stages
            )
        } else if query.activation == ActivationFormat::Nvfp4 {
            format!(
                "chop::gemm::run_nvfp4_a4<{decode}, {residual}, {}, {}, {}, {}>(map_a, map_b, output, partial, semaphores, live_m, n, k, splits, output_f32, weight_scales, activation_scales, residual, weight_global_scale, activation_global_scale);",
                knobs.tile_m, knobs.tile_n, knobs.tile_k, knobs.stages
            )
        } else if matches!(
            query.activation,
            ActivationFormat::Fp8Group128(_) | ActivationFormat::Mxfp8
        ) {
            format!(
                "chop::gemm::run_fp8_a8<chop::gemm::WeightKind::{kind}, {decode}, {residual}, {}, {}, {}, {}>(map_a, map_b, output, partial, semaphores, live_m, n, k, splits, output_f32, weight_scales, activation_scales, residual);",
                knobs.tile_m, knobs.tile_n, knobs.tile_k, knobs.stages
            )
        } else if is_balanced_decoded(kernel, query, knobs, decode) {
            decoded_balanced = true;
            format!(
                "chop::gemm::run_a16_decoded_balanced<{residual}, {SM_COUNT}>(map_a, map_b, output, partial, semaphores, live_m, n, k, splits, output_f32, weight_scales, residual, weight_global_scale);",
            )
        } else if is_staged_tiles(kernel, query, knobs, decode) {
            staged_tiles = true;
            format!(
                "chop::gemm::run_a16_staged<{residual}, {SM_COUNT}>(map_a, map_b, output, partial, semaphores, live_m, n, k, splits, output_f32, weight_scales, residual, weight_global_scale);",
            )
        } else if is_decoded_tiles(kernel, query, knobs, decode) {
            decoded_tiles = true;
            format!(
                "chop::gemm::run_a16_decoded<{residual}>(map_a, map_b, output, partial, semaphores, live_m, n, k, splits, output_f32, weight_scales, residual, weight_global_scale);",
            )
        } else if let DenseKernel::TiledChains { .. } = kernel
            && !decode
        {
            tiled_warps = true;
            format!(
                "chop::gemm::run_a16_warps_ordered<chop::gemm::WeightKind::{kind}, {residual}, {}, {}, {}, {}>(map_a, map_b, output, partial, semaphores, live_m, n, k, splits, output_f32, weight_scales, residual, weight_global_scale);",
                knobs.tile_m, knobs.tile_n, knobs.tile_k, knobs.stages
            )
        } else if let DenseKernel::StripChains { strips, .. } = kernel
            && query.weight != WeightFormat::Bf16
        {
            strip_chains_fp8 = true;
            format!(
                "chop::gemm::run_a16_strips_fp8_ordered<{residual}, {strips}, {}, {}>(map_a, map_b, output, partial, semaphores, live_m, n, k, splits, output_f32, weight_scales, residual, weight_global_scale);",
                knobs.tile_n, knobs.stages
            )
        } else if let DenseKernel::StripChains { strips, .. } = kernel {
            strip_chains = true;
            format!(
                "chop::gemm::run_a16_strips_ordered<{residual}, {strips}, {}, {}>(map_a, map_b, output, partial, semaphores, live_m, n, k, splits, output_f32, weight_scales, residual, weight_global_scale);",
                knobs.tile_n, knobs.stages
            )
        } else if let DenseKernel::TiledChains { .. } = kernel {
            tiled_chains = true;
            format!(
                "chop::gemm::run_a16_chains<chop::gemm::WeightKind::{kind}, {decode}, {residual}, {}, {}, {}, {}>(map_a, map_b, output, partial, semaphores, live_m, n, k, splits, output_f32, weight_scales, residual, weight_global_scale);",
                knobs.tile_m, knobs.tile_n, knobs.tile_k, knobs.stages
            )
        } else {
            format!(
                "chop::gemm::run_a16<chop::gemm::WeightKind::{kind}, {decode}, {residual}, {}, {}, {}, {}>(map_a, map_b, output, partial, semaphores, live_m, n, k, splits, output_f32, weight_scales, residual, weight_global_scale);",
                knobs.tile_m, knobs.tile_n, knobs.tile_k, knobs.stages
            )
        };
        let mut entry = EntrySpec::new(
            item.symbol.clone(),
            params(fragment, decode),
            body,
            if is_router(kernel) || is_cluster(kernel) {
                256
            } else if is_fragment_decode(kernel) {
                u32::from(knobs.tile_m / 16) * 32
            } else if let Some(rows) = strip_rows(kernel) {
                (rows / 16 + 1) * 32
            } else {
                THREADS
            },
            // Two cluster decode or strip-chain CTAs share each SM.
            if is_cluster(kernel) || strip_rows(kernel).is_some() {
                2
            } else {
                1
            },
        );
        // nvcc may reserve a small stack frame for packed prefill bodies,
        // and FP8 warp tiles (`tiled_warps.cuh`) keep a 72-byte stack; the
        // bound covers both.
        entry.allowed_local_bytes = 1024;
        entries.push(entry);
    }
    let mut spec = CubinSpec {
        name: "chop_dense_gemm".into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec![
            "#include \"gemm/a8_fp8.cuh\"".into(),
            "#include \"gemm/a4_nvfp4.cuh\"".into(),
            "#include \"gemm/swiglu.cuh\"".into(),
            "#include \"gemm/fragment_decode.cuh\"".into(),
            "#include \"gemm/fragment_router.cuh\"".into(),
            "#include \"gemm/fragment_decode_cluster.cuh\"".into(),
            "#include \"gemm/fragment_prefill.cuh\"".into(),
        ],
        defines: vec![],
        entries,
    };
    if mxfp4_a16 {
        spec.headers
            .push("#include \"gemm/dense_mxfp4.cuh\"".into());
    }
    // Only cubins with chain-split or warp-tiled entries include their
    // headers, so other cubins keep their source closures.
    if tiled_chains {
        spec.headers
            .push("#include \"gemm/tiled_chains.cuh\"".into());
    }
    // Warp tiles run the kernel that releases a ring stage only after every
    // read of it (tiled_warps_ordered.cuh, FIX-SC1).
    if tiled_warps {
        spec.headers
            .push("#include \"gemm/tiled_warps_ordered.cuh\"".into());
    }
    if decoded_tiles {
        spec.headers
            .push("#include \"gemm/decoded_tiles.cuh\"".into());
    }
    if decoded_balanced {
        spec.headers
            .push("#include \"gemm/decoded_balanced.cuh\"".into());
    }
    if staged_tiles {
        spec.headers
            .push("#include \"gemm/staged_tiles.cuh\"".into());
    }
    // Strip chains run the kernels that release a ring stage only after
    // every read of it (strip_chains_ordered.cuh, FIX-SC1).
    if strip_chains {
        spec.headers
            .push("#include \"gemm/strip_chains_ordered.cuh\"".into());
    }
    if strip_chains_fp8 {
        spec.headers
            .push("#include \"gemm/strip_chains_fp8_ordered.cuh\"".into());
    }
    Ok(spec)
}

/// The entry `chop_gemm_{name}_v1` of `profile` on `kernel`, with tiles
/// and split capacity for its kernel (`tile_m` and `tile_n` apply to
/// prefill tiles of the TMA kernels only). Fragment prefill entries declare
/// chunk invariance, others nothing. Panics where the kernel refuses
/// `query`.
pub fn instance(
    profile: Profile,
    name: &str,
    query: GemmQuery,
    tile_m: u16,
    tile_n: u16,
    kernel: DenseKernel,
) -> GemmVariant {
    let symbol = format!("chop_gemm_{name}_v1");
    if let Err(Refusal(reason)) = dense_supported(kernel, &query) {
        panic!("{symbol}: {reason}");
    }
    // Each prefill row sums its fixed K segments in one order whatever the
    // live M, grid and row position (`gemm_prefill_rows_ignore_chunking`).
    let invariance = if is_fragment_prefill(kernel) {
        Invariance::PREFILL
    } else {
        Invariance::NONE
    };
    GemmVariant {
        profile,
        instantiation: construct(symbol, query, tile_m, tile_n, kernel),
        invariance,
    }
}

/// A format compilation probe, planned with room for its declared K splits
/// only.
fn probe(
    profile: Profile,
    name: &str,
    query: GemmQuery,
    tile_m: u16,
    tile_n: u16,
    kernel: DenseKernel,
) -> GemmVariant {
    let mut item = instance(profile, name, query, tile_m, tile_n, kernel);
    item.instantiation.key.knobs.split_k_capacity =
        required_split_k(&item.key).expect("probes declare their K splits");
    item
}

/// An entry of `kernel` (already checked against `query`) with tiles and
/// split capacity for its kernel; `tile_m` and `tile_n` apply to prefill
/// tiles of the TMA kernels only.
fn construct(
    symbol: String,
    query: GemmQuery,
    tile_m: u16,
    tile_n: u16,
    kernel: DenseKernel,
) -> InstantiationSpec {
    let _ = query
        .numerical_contract_id()
        .expect("catalog query must validate");
    assert_eq!(tuning_number("sm_count"), SM_COUNT as u16);
    assert!(TUNING.contains("\"compute_capability\": \"12.0\""));
    let decode = query.m_capacity <= 32;
    let fragment = is_fragment(kernel);
    let chosen_tile_m = if is_fragment_prefill(kernel) {
        128
    } else if let Some(rows) = strip_rows(kernel) {
        rows as u16
    } else if decode {
        if fragment {
            fragment_decode_tile_m(kernel, query)
        } else {
            tuning_number("tile_m")
        }
    } else {
        tile_m
    };
    let chosen_tile_n = if is_fragment_prefill(kernel) {
        128
    } else if is_cluster(kernel) {
        u16::try_from(query.m_capacity.next_power_of_two().max(8))
            .expect("decode token tile fits u16")
    } else {
        tile_n
    };
    let chosen_tile_k = if is_fragment_prefill(kernel) {
        64
    } else if fragment {
        128
    } else {
        match query.weight {
            WeightFormat::Bf16 => tuning_number("tile_k_bf16"),
            WeightFormat::Mxfp4 | WeightFormat::Nvfp4 => tuning_number("tile_k_fp4"),
            _ => tuning_number("tile_k_fp8"),
        }
    };
    let tiles_a = if decode {
        div_up(output_n(query), u32::from(chosen_tile_m))
    } else {
        div_up(query.m_capacity, u32::from(chosen_tile_m))
    };
    let tiles_b = if is_cluster(kernel) {
        1
    } else if decode {
        div_up(query.m_capacity, u32::from(chosen_tile_n))
    } else {
        div_up(output_n(query), u32::from(chosen_tile_n))
    };
    let target_splits = u64::from(if is_fragment_decode(kernel) {
        fragment_decode_grid_target(kernel)
    } else {
        SM_COUNT
    })
    .div_ceil(u64::from(tiles_a) * u64::from(tiles_b));
    let max_splits = u64::from(tuning_number(if decode {
        "decode_split_cap"
    } else {
        "prefill_split_cap"
    }));
    // Tiled and fragment decode entries plan room for the K partitions that
    // would fill the SMs; their launches always pass the declared splits.
    let selected_splits = match kernel {
        DenseKernel::FragmentPrefill { segments } => segments,
        DenseKernel::FragmentRouter { splits } => splits,
        DenseKernel::FragmentCluster { .. } => 1,
        DenseKernel::Tiled { splits } | DenseKernel::FragmentDecode { splits } => (target_splits
            .min(max_splits)
            .min(u64::from(div_up(query.k, u32::from(chosen_tile_k))))
            as u8)
            .max(splits),
        // Bits do not depend on the split count; the grid covers the
        // declared splits only.
        DenseKernel::TiledChains { splits } | DenseKernel::StripChains { splits, .. } => splits,
    };
    let mut knobs = CompileKnobs {
        tile_m: chosen_tile_m,
        tile_n: chosen_tile_n,
        tile_k: chosen_tile_k,
        stages: 1,
        swizzle: tuning_number("swizzle") as u8,
        warps: if is_router(kernel) || is_cluster(kernel) {
            8
        } else if is_fragment_decode(kernel) {
            (chosen_tile_m / 16) as u8
        } else if fragment {
            8
        } else if strip_rows(kernel).is_some() {
            (chosen_tile_m / 16 + 1) as u8
        } else {
            tuning_number("warps") as u8
        },
        split_k_capacity: selected_splits,
    };
    knobs.stages = expected_stages(kernel, query, knobs, decode);
    InstantiationSpec {
        family: OpFamily::Gemm,
        key: CompileKey {
            query: query.into(),
            knobs,
            dense: Some(kernel),
            target: Target::Sm120f,
        },
        symbol,
    }
}

/// The K splits a chain-split decode entry of `query`
/// (`DenseKernel::TiledChains`) declares: enough splits of each decode
/// tile's K chunks that the tiles fill `SM_COUNT` CTAs, at most one per
/// chunk and `decode_split_cap`; 1 (keep `Tiled { splits: 1 }`) below
/// `MIN_CHAIN_SPLITS`. Its bits do not depend on the count.
///
/// kernel-lab `projection_latency` (RTX PRO 6000, 1 to 32 rows): with 4 or
/// more splits the chain entries ran 1.0 to 5.6 times faster than one
/// split (n64 to n2048 at K = 4096); with 3 splits (n4096) the temporaries
/// and their reduction made them up to 2.0 times slower.
pub fn chain_splits(query: &GemmQuery) -> u8 {
    let tile_k = match query.weight {
        WeightFormat::Bf16 => tuning_number("tile_k_bf16"),
        WeightFormat::Mxfp4 | WeightFormat::Nvfp4 => tuning_number("tile_k_fp4"),
        _ => tuning_number("tile_k_fp8"),
    };
    let tiles = div_up(output_n(*query), u32::from(tuning_number("tile_m")))
        * div_up(
            query.m_capacity,
            query.m_capacity.next_power_of_two().max(8),
        );
    let splits = SM_COUNT
        .div_ceil(tiles)
        .min(div_up(query.k, u32::from(tile_k)))
        .min(u32::from(tuning_number("decode_split_cap")));
    if splits < u32::from(MIN_CHAIN_SPLITS) {
        1
    } else {
        splits as u8
    }
}

/// The fewest K splits a chain-split decode entry gains from
/// (`chain_splits`).
pub const MIN_CHAIN_SPLITS: u8 = 4;

/// A BF16 weight and activation query of `m` rows.
pub fn bf16_query(m: u32, n: u32, k: u32, output: DType) -> GemmQuery {
    GemmQuery {
        m_capacity: m,
        n,
        k,
        weight: WeightFormat::Bf16,
        activation: ActivationFormat::Bf16,
        output,
        epilogues: Epilogues::default(),
        grouped: None,
    }
}

/// An FP8 128x128 block-scaled weight query with BF16 activations.
pub fn fp8_block_query(m: u32, n: u32, k: u32) -> GemmQuery {
    GemmQuery {
        weight: WeightFormat::Fp8Block128x128(kernel_core::contract::ScaleMode::ExactF32),
        ..bf16_query(m, n, k, DType::Bf16)
    }
}

fn fp8_a8_query(m: u32, n: u32, k: u32) -> GemmQuery {
    GemmQuery {
        activation: ActivationFormat::Fp8Group128(kernel_core::contract::ScaleMode::ExactF32),
        ..fp8_block_query(m, n, k)
    }
}

/// The cluster decode of BF16 weights (two 128-K blocks per warp) and of
/// FP8 weights (four), whose warps each hold 16 fragment tiles.
pub const CLUSTER_BF16: DenseKernel = DenseKernel::FragmentCluster { warp_blocks: 2 };
pub const CLUSTER_FP8: DenseKernel = DenseKernel::FragmentCluster { warp_blocks: 4 };
/// Fragment decode summing all of K per tile, as the served models' fragment
/// decode entries run.
pub const WHOLE_K_DECODE: DenseKernel = DenseKernel::FragmentDecode { splits: 1 };

/// The kernel base's dense entries, then the models' in `catalogs`.
pub fn catalog(catalogs: &Catalogs) -> Vec<GemmVariant> {
    let mut items = own_catalog();
    for model in catalogs.models() {
        items.extend((model.gemm)());
    }
    items
}

/// The kernel base's dense entries: the standard profile's and the format
/// compilation probes.
pub fn own_catalog() -> Vec<GemmVariant> {
    let mut items = Vec::new();
    for m in [1, 2, 4, 8, 16, 32] {
        let tn = m.max(8) as u16;
        items.push(instance(
            Profile::StdDenseMoe,
            &format!("std_dense_m{m}"),
            bf16_query(m, 4096, 4096, DType::Bf16),
            64,
            tn,
            CLUSTER_BF16,
        ));
        for (profile, name, format) in [
            (Profile::StdDenseMoe, "std_dense_mxfp4", WeightFormat::Mxfp4),
            (Profile::StdDenseMoe, "std_dense_nvfp4", WeightFormat::Nvfp4),
        ] {
            let mut query = bf16_query(m, 4096, 4096, DType::Bf16);
            query.weight = format;
            items.push(instance(
                profile,
                &format!("{name}_m{m}"),
                query,
                64,
                tn,
                DenseKernel::Tiled { splits: 3 },
            ));
        }
        let mut mxfp4_a8 = bf16_query(m, 4096, 4096, DType::Bf16);
        mxfp4_a8.weight = WeightFormat::Mxfp4;
        mxfp4_a8.activation = ActivationFormat::Mxfp8;
        items.push(instance(
            Profile::StdDenseMoe,
            &format!("std_dense_mxfp4_a8_m{m}"),
            mxfp4_a8,
            64,
            tn,
            DenseKernel::Tiled { splits: 3 },
        ));
        let mut nvfp4_a4 = bf16_query(m, 4096, 4096, DType::Bf16);
        nvfp4_a4.weight = WeightFormat::Nvfp4;
        nvfp4_a4.activation = ActivationFormat::Nvfp4;
        items.push(instance(
            Profile::StdDenseMoe,
            &format!("std_dense_nvfp4_a4_m{m}"),
            nvfp4_a4,
            64,
            tn,
            DenseKernel::Tiled { splits: 3 },
        ));
        items.push(instance(
            Profile::StdDenseMoe,
            &format!("std_wide_m{m}"),
            bf16_query(m, 8192, 8192, DType::Bf16),
            64,
            tn,
            DenseKernel::FragmentDecode { splits: 12 },
        ));
        items.push(instance(
            Profile::StdDenseMoe,
            &format!("std_mlp_7168_m{m}"),
            bf16_query(m, 7168, 4096, DType::Bf16),
            64,
            tn,
            CLUSTER_BF16,
        ));
        items.push(instance(
            Profile::StdDenseMoe,
            &format!("std_dense_fp8_m{m}"),
            fp8_block_query(m, 4096, 4096),
            64,
            tn,
            CLUSTER_FP8,
        ));
        items.push(instance(
            Profile::StdDenseMoe,
            &format!("std_dense_a8_m{m}"),
            fp8_a8_query(m, 4096, 4096),
            64,
            tn,
            DenseKernel::Tiled { splits: 3 },
        ));
        items.push(instance(
            Profile::StdDenseMoe,
            &format!("std_wide_fp8_m{m}"),
            fp8_block_query(m, 8192, 8192),
            64,
            tn,
            DenseKernel::FragmentDecode { splits: 12 },
        ));
        items.push(instance(
            Profile::StdDenseMoe,
            &format!("std_mlp_14336_fp8_m{m}"),
            fp8_block_query(m, 14336, 8192),
            64,
            tn,
            DenseKernel::FragmentDecode { splits: 7 },
        ));
    }
    // The router takes eight K segments.
    for (profile, name, m, n, k, tm, tn, segments) in [
        (
            Profile::StdDenseMoe,
            "std_router_prefill",
            8192,
            256,
            4096,
            64,
            128,
            8,
        ),
        (
            Profile::StdDenseMoe,
            "std_dense_prefill",
            8192,
            4096,
            4096,
            128,
            128,
            4,
        ),
        (
            Profile::StdDenseMoe,
            "std_wide_prefill",
            8192,
            8192,
            8192,
            128,
            128,
            4,
        ),
        (
            Profile::StdDenseMoe,
            "std_mlp_7168_prefill",
            8192,
            7168,
            4096,
            128,
            128,
            4,
        ),
    ] {
        let output = if name.contains("router") {
            DType::F32
        } else {
            DType::Bf16
        };
        items.push(instance(
            profile,
            name,
            bf16_query(m, n, k, output),
            tm,
            tn,
            DenseKernel::FragmentPrefill { segments },
        ));
    }
    for (profile, name, format) in [
        (
            Profile::StdDenseMoe,
            "std_dense_mxfp4_prefill",
            WeightFormat::Mxfp4,
        ),
        (
            Profile::StdDenseMoe,
            "std_dense_nvfp4_prefill",
            WeightFormat::Nvfp4,
        ),
    ] {
        let mut query = bf16_query(8192, 4096, 4096, DType::Bf16);
        query.weight = format;
        items.push(instance(
            profile,
            name,
            query,
            64,
            128,
            DenseKernel::Tiled { splits: 1 },
        ));
    }
    let mut mxfp4_a8 = bf16_query(8192, 4096, 4096, DType::Bf16);
    mxfp4_a8.weight = WeightFormat::Mxfp4;
    mxfp4_a8.activation = ActivationFormat::Mxfp8;
    items.push(instance(
        Profile::StdDenseMoe,
        "std_dense_mxfp4_a8_prefill",
        mxfp4_a8,
        64,
        128,
        DenseKernel::Tiled { splits: 1 },
    ));
    let mut nvfp4_a4 = bf16_query(8192, 4096, 4096, DType::Bf16);
    nvfp4_a4.weight = WeightFormat::Nvfp4;
    nvfp4_a4.activation = ActivationFormat::Nvfp4;
    items.push(instance(
        Profile::StdDenseMoe,
        "std_dense_nvfp4_a4_prefill",
        nvfp4_a4,
        128,
        128,
        DenseKernel::Tiled { splits: 1 },
    ));
    items.push(instance(
        Profile::StdDenseMoe,
        "std_dense_fp8_prefill",
        fp8_block_query(8192, 4096, 4096),
        64,
        128,
        DenseKernel::FragmentPrefill { segments: 4 },
    ));
    items.push(instance(
        Profile::StdDenseMoe,
        "std_dense_a8_prefill",
        fp8_a8_query(8192, 4096, 4096),
        128,
        128,
        DenseKernel::Tiled { splits: 1 },
    ));
    items.push(instance(
        Profile::StdDenseMoe,
        "std_wide_fp8_prefill",
        fp8_block_query(8192, 8192, 8192),
        64,
        128,
        DenseKernel::FragmentPrefill { segments: 4 },
    ));
    items.push(instance(
        Profile::StdDenseMoe,
        "std_mlp_14336_fp8_prefill",
        fp8_block_query(8192, 14336, 8192),
        64,
        128,
        DenseKernel::FragmentPrefill { segments: 4 },
    ));
    let mut tensor = fp8_block_query(1, 256, 128);
    tensor.weight = WeightFormat::Fp8PerTensor;
    items.push(probe(
        Profile::StdDenseMoe,
        "format_fp8_tensor_a16",
        tensor,
        64,
        8,
        DenseKernel::Tiled { splits: 1 },
    ));
    tensor.weight = WeightFormat::Fp8PerChannel;
    items.push(probe(
        Profile::StdDenseMoe,
        "format_fp8_channel_a16",
        tensor,
        64,
        8,
        DenseKernel::Tiled { splits: 1 },
    ));
    tensor.weight = WeightFormat::Fp8Block128x128(kernel_core::contract::ScaleMode::PowerOfTwoF32);
    items.push(probe(
        Profile::StdDenseMoe,
        "format_fp8_block_pow2_a16",
        tensor,
        64,
        8,
        DenseKernel::FragmentDecode { splits: 1 },
    ));
    tensor.activation = ActivationFormat::Fp8Group128(kernel_core::contract::ScaleMode::ExactF32);
    items.push(probe(
        Profile::StdDenseMoe,
        "format_fp8_block_pow2_a8",
        tensor,
        64,
        8,
        DenseKernel::Tiled { splits: 1 },
    ));
    tensor.weight = WeightFormat::Fp8PerTensor;
    items.push(probe(
        Profile::StdDenseMoe,
        "format_fp8_tensor_a8",
        tensor,
        64,
        8,
        DenseKernel::Tiled { splits: 1 },
    ));
    tensor.weight = WeightFormat::Fp8PerChannel;
    items.push(probe(
        Profile::StdDenseMoe,
        "format_fp8_channel_a8",
        tensor,
        64,
        8,
        DenseKernel::Tiled { splits: 1 },
    ));
    tensor.activation =
        ActivationFormat::Fp8Group128(kernel_core::contract::ScaleMode::PowerOfTwoF32);
    items.push(probe(
        Profile::StdDenseMoe,
        "format_fp8_group_pow2_a8",
        tensor,
        64,
        8,
        DenseKernel::Tiled { splits: 1 },
    ));
    let mut interleaved = fp8_block_query(1, 128, 128);
    interleaved.epilogues.swiglu = Some(SwigluColumns::Interleaved);
    items.push(probe(
        Profile::StdDenseMoe,
        "format_swiglu_interleaved_decode",
        interleaved,
        64,
        8,
        DenseKernel::Tiled { splits: 1 },
    ));
    interleaved.m_capacity = 128;
    interleaved.n = 256;
    items.push(probe(
        Profile::StdDenseMoe,
        "format_swiglu_interleaved_prefill",
        interleaved,
        64,
        128,
        DenseKernel::Tiled { splits: 1 },
    ));
    let mxfp8 = GemmQuery {
        weight: WeightFormat::Mxfp8,
        activation: ActivationFormat::Mxfp8,
        ..bf16_query(1, 256, 128, DType::Bf16)
    };
    items.push(probe(
        Profile::StdDenseMoe,
        "format_mxfp8_decode",
        mxfp8,
        64,
        8,
        DenseKernel::Tiled { splits: 1 },
    ));
    items.push(probe(
        Profile::StdDenseMoe,
        "format_mxfp8_a16_decode",
        GemmQuery {
            activation: ActivationFormat::Bf16,
            ..mxfp8
        },
        64,
        8,
        DenseKernel::Tiled { splits: 1 },
    ));
    items.push(probe(
        Profile::StdDenseMoe,
        "format_mxfp8_prefill",
        GemmQuery {
            m_capacity: 128,
            ..mxfp8
        },
        128,
        128,
        DenseKernel::Tiled { splits: 1 },
    ));
    items.push(probe(
        Profile::StdDenseMoe,
        "format_mxfp8_a16_prefill",
        GemmQuery {
            m_capacity: 128,
            activation: ActivationFormat::Bf16,
            ..mxfp8
        },
        64,
        128,
        DenseKernel::Tiled { splits: 1 },
    ));
    for (name, weight, activation) in [
        ("mxfp4_a16", WeightFormat::Mxfp4, ActivationFormat::Bf16),
        ("nvfp4_a16", WeightFormat::Nvfp4, ActivationFormat::Bf16),
        ("mxfp4_a8", WeightFormat::Mxfp4, ActivationFormat::Mxfp8),
        ("nvfp4_a4", WeightFormat::Nvfp4, ActivationFormat::Nvfp4),
    ] {
        let mut query = bf16_query(1, 256, 256, DType::Bf16);
        query.weight = weight;
        query.activation = activation;
        items.push(probe(
            Profile::StdDenseMoe,
            &format!("format_{name}_decode"),
            query,
            64,
            8,
            DenseKernel::Tiled { splits: 1 },
        ));
        query.m_capacity = 128;
        items.push(probe(
            Profile::StdDenseMoe,
            &format!("format_{name}_prefill"),
            query,
            if activation == ActivationFormat::Bf16 {
                64
            } else {
                128
            },
            128,
            DenseKernel::Tiled { splits: 1 },
        ));
    }
    let mut residual = bf16_query(1, 4096, 4096, DType::Bf16);
    residual.epilogues.residual_add = true;
    items.push(probe(
        Profile::StdDenseMoe,
        "format_residual_decode",
        residual,
        64,
        8,
        DenseKernel::Tiled { splits: 1 },
    ));
    let mut residual_prefill = fp8_a8_query(8192, 4096, 4096);
    residual_prefill.epilogues.residual_add = true;
    items.push(probe(
        Profile::StdDenseMoe,
        "format_residual_prefill",
        residual_prefill,
        128,
        128,
        DenseKernel::Tiled { splits: 1 },
    ));
    items
}

/// [`cubin_spec`] with each entry's declared arithmetic
/// ([`arithmetic::dense`]), from the instantiation that built it.
pub fn declared_cubin_spec(instantiations: &[InstantiationSpec]) -> Result<DeclaredCubin, Refusal> {
    let spec = cubin_spec(instantiations)?;
    let arithmetic = instantiations
        .iter()
        .map(|item| (item.symbol.clone(), arithmetic::dense(item)))
        .collect();
    DeclaredCubin::new(spec, arithmetic)
}

/// One audited cubin per profile with GEMM entries in `catalogs`. `None`
/// builds them all.
pub fn cubin_specs(catalogs: &Catalogs, profile: Option<Profile>) -> Vec<CubinSpec> {
    declared_cubin_specs(catalogs, profile)
        .into_iter()
        .map(DeclaredCubin::into_spec)
        .collect()
}

/// [`cubin_specs`] with each entry's declared arithmetic.
pub fn declared_cubin_specs(catalogs: &Catalogs, profile: Option<Profile>) -> Vec<DeclaredCubin> {
    let all = catalog(catalogs);
    let mut owners = catalogs.profiles();
    for item in &all {
        if !owners.contains(&item.profile) {
            owners.push(item.profile);
        }
    }
    owners
        .into_iter()
        .filter(|&owner| profile.is_none_or(|selected| selected == owner))
        .filter_map(|owner| {
            let items = all
                .iter()
                .filter(|item| item.profile == owner)
                .map(|item| item.instantiation.clone())
                .collect::<Vec<_>>();
            if items.is_empty() {
                return None;
            }
            let cubin = declared_cubin_spec(&items).expect("catalog entries are valid");
            Some(cubin.with_name(format!("gemm_{}_sm120", owner.tag())))
        })
        .collect()
}

/// Lab-only: the decode buckets of the models' lab K splits
/// (`ModelCatalog::gemm_lab_splits`) at those splits, beside their served single-split entries. Each keeps its served
/// entry's symbol and compiled wrapper (the split count is a launch
/// argument), so it runs from the served cubin and adds no entry.
pub fn lab_split_catalog(catalogs: &Catalogs) -> Vec<GemmVariant> {
    let splits = catalogs
        .models()
        .iter()
        .flat_map(|model| {
            model
                .gemm_lab_splits
                .iter()
                .map(|&(name, splits)| (name.to_owned(), splits))
                .chain((model.gemm_lab_splits_for)())
        })
        .collect::<Vec<_>>();
    catalog(catalogs)
        .into_iter()
        .filter_map(|mut item| {
            let (stem, bucket) = item.symbol.strip_suffix("_v1")?.rsplit_once("_m")?;
            bucket.parse::<u32>().ok()?;
            let splits = splits.iter().find(|(name, _)| *name == stem)?.1;
            assert_eq!(item.key.dense, Some(WHOLE_K_DECODE), "{}", item.symbol);
            item.instantiation.key.dense = Some(DenseKernel::FragmentDecode { splits });
            Some(item)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An entry's capacity workspace and shared memory are its plan's at
    /// its full row capacity.
    #[test]
    fn capacity_needs_are_the_full_capacity_plans() {
        let query = bf16_query(4, 1536, 4096, DType::Bf16);
        let entry = instantiate(DenseRequest {
            query,
            kernel: DenseKernel::TiledChains {
                splits: chain_splits(&query),
            },
            target: Target::Sm120f,
            symbol: "capacity_probe_v1".into(),
            knobs: None,
        })
        .unwrap();
        let plan = plan(&entry.key, row_counts(&entry.key, query.m_capacity)).unwrap();
        assert!(plan.workspace_bytes > 0);
        assert_eq!(
            capacity_workspace_bytes(&entry.key),
            Ok(plan.workspace_bytes)
        );
        assert_eq!(
            capacity_shared_bytes(&entry.key),
            Ok(plan.dynamic_shared_bytes)
        );
    }

    /// Chain-split entries keep the one-split tiles, take one ticket per
    /// tile, size their workspace by K chunks whatever the split count, and
    /// compile the one-split body's chain-keeping twin from their own header.
    #[test]
    fn chain_split_entries_plan_chunk_temporaries() {
        let make = |query: GemmQuery, kernel| {
            instantiate(DenseRequest {
                query,
                kernel,
                target: Target::Sm120f,
                symbol: "chain_probe_v1".into(),
                knobs: None,
            })
        };
        let mut fp8 = bf16_query(4, 1536, 4096, DType::Bf16);
        fp8.weight = WeightFormat::Fp8Block128x128(kernel_core::contract::ScaleMode::ExactF32);
        for (query, chunks) in [(bf16_query(1, 2048, 4096, DType::Bf16), 64), (fp8, 32)] {
            let splits = chain_splits(&query);
            let tiles = query.n.div_ceil(64);
            assert_eq!(u32::from(splits), 188u32.div_ceil(tiles).min(chunks));
            assert!(splits >= MIN_CHAIN_SPLITS);
            let one = make(query, DenseKernel::Tiled { splits: 1 }).unwrap();
            let chains = make(query, DenseKernel::TiledChains { splits }).unwrap();
            let mut knobs = one.key.knobs;
            knobs.split_k_capacity = splits;
            assert_eq!(chains.key.knobs, knobs);
            let tile = u64::from(knobs.tile_m) * u64::from(knobs.tile_n);
            let plan = plan(&chains.key, row_counts(&chains.key, query.m_capacity)).unwrap();
            assert_eq!(plan.grid[0], (tiles * u32::from(splits)).min(SM_COUNT));
            assert_eq!(
                plan.workspace_bytes,
                (u64::from(tiles) * u64::from(chunks) * tile + u64::from(tiles)) * 4
            );
            assert_eq!(
                GemmBuffers::semaphore_capacity_bytes(&chains.key).unwrap(),
                u64::from(tiles) * 4
            );
            let one_spec = cubin_spec(std::slice::from_ref(&one)).unwrap();
            let chain_spec = cubin_spec(std::slice::from_ref(&chains)).unwrap();
            assert_eq!(
                chain_spec.entries[0].body,
                one_spec.entries[0]
                    .body
                    .replace("run_a16<", "run_a16_chains<")
            );
            assert_eq!(chain_spec.entries[0].params, one_spec.entries[0].params);
            assert!(!one_spec.headers.iter().any(|h| h.contains("tiled_chains")));
            assert!(
                chain_spec
                    .headers
                    .iter()
                    .any(|h| h.contains("tiled_chains"))
            );
        }
        // 64 tiles would take 3 splits: they keep one split.
        assert_eq!(chain_splits(&bf16_query(1, 4096, 4096, DType::Bf16)), 1);
        assert_eq!(chain_splits(&bf16_query(1, 2048, 128, DType::Bf16)), 1);
        let mut swiglu = bf16_query(1, 2048, 4096, DType::Bf16);
        swiglu.epilogues.swiglu = Some(SwigluColumns::ConcatenatedHalves);
        let mut nvfp4 = bf16_query(1, 2048, 4096, DType::Bf16);
        nvfp4.weight = WeightFormat::Nvfp4;
        for query in [bf16_query(64, 2048, 4096, DType::Bf16), swiglu, nvfp4] {
            assert!(make(query, DenseKernel::TiledChains { splits: 2 }).is_err());
        }
        assert!(
            make(
                bf16_query(1, 2048, 4096, DType::Bf16),
                DenseKernel::TiledChains { splits: 0 }
            )
            .is_err()
        );
    }

    /// Strip-chain entries take 16 weight rows per strip, the bucket's token
    /// tile and 64-K chunks, a producer warp and a math warp per strip, the
    /// ring stages that fit `STRIP_RING_BYTES`, one CTA per tile and
    /// declared split, and above one split a chain temporary per tile and K
    /// chunk and a ticket per tile (neither at one); they compile from their
    /// own header.
    #[test]
    fn strip_chain_entries_plan_one_cta_per_tile_and_split() {
        let make = |query: GemmQuery, kernel| {
            instantiate(DenseRequest {
                query,
                kernel,
                target: Target::Sm120f,
                symbol: "strip_probe_v1".into(),
                knobs: None,
            })
        };
        for (m, strips, splits) in [(1u32, 1u8, 1u8), (4, 2, 5), (32, 4, 11), (16, 1, 23)] {
            let query = bf16_query(m, 2048, 4096, DType::Bf16);
            let spec = make(query, DenseKernel::StripChains { strips, splits }).unwrap();
            let knobs = spec.key.knobs;
            let rows = 16 * u32::from(strips);
            let tile_n = m.next_power_of_two().max(8);
            assert_eq!(
                (
                    u32::from(knobs.tile_m),
                    u32::from(knobs.tile_n),
                    knobs.tile_k
                ),
                (rows, tile_n, 64)
            );
            assert_eq!(knobs.warps, strips + 1);
            assert_eq!(knobs.split_k_capacity, splits);
            let stage = (rows + tile_n) * 128;
            assert_eq!(u32::from(knobs.stages), (48 * 1024 / stage).clamp(2, 16));
            let tiles = 2048 / rows;
            let plan = plan(&spec.key, row_counts(&spec.key, m)).unwrap();
            assert_eq!(plan.grid[0], tiles * u32::from(splits));
            assert_eq!(plan.block[0], (u32::from(strips) + 1) * 32);
            assert_eq!(
                plan.dynamic_shared_bytes,
                u32::from(knobs.stages) * stage + 128
            );
            let temporaries = if splits == 1 {
                0
            } else {
                u64::from(tiles) * 64 * u64::from(rows * tile_n)
            };
            assert_eq!(plan.workspace_bytes, (temporaries + u64::from(tiles)) * 4);
            assert_eq!(
                GemmBuffers::semaphore_capacity_bytes(&spec.key).unwrap(),
                if splits == 1 { 0 } else { u64::from(tiles) * 4 }
            );
            let cubin = cubin_spec(std::slice::from_ref(&spec)).unwrap();
            assert!(cubin.entries[0].body.starts_with(&format!(
                "chop::gemm::run_a16_strips_ordered<false, {strips}, {tile_n}, {}>(map_a",
                knobs.stages
            )));
            assert!(
                cubin
                    .headers
                    .iter()
                    .any(|h| h.contains("strip_chains_ordered.cuh"))
            );
            let one = make(query, DenseKernel::Tiled { splits: 1 }).unwrap();
            let one = cubin_spec(std::slice::from_ref(&one)).unwrap();
            assert!(!one.headers.iter().any(|h| h.contains("strip_chains")));
        }
        let mut rows64 = bf16_query(1, 2048, 4096, DType::Bf16);
        rows64.weight = WeightFormat::Fp8Rows64x128(kernel_core::contract::ScaleMode::ExactF32);
        let mut swiglu = bf16_query(1, 2048, 4096, DType::Bf16);
        swiglu.epilogues.swiglu = Some(SwigluColumns::ConcatenatedHalves);
        let bf16 = bf16_query(1, 2048, 4096, DType::Bf16);
        for (query, strips, splits) in [
            (bf16_query(64, 2048, 4096, DType::Bf16), 1, 1),
            (bf16, 3, 1),
            (bf16, 8, 1),
            (bf16, 1, 0),
            (bf16_query(1, 2048, 128, DType::Bf16), 1, 3),
            (rows64, 1, 1),
            (swiglu, 1, 1),
        ] {
            assert!(
                make(query, DenseKernel::StripChains { strips, splits }).is_err(),
                "{strips} strips, {splits} splits over {query:?}"
            );
        }
    }

    /// FP8 128x128 strip-chain entries take 128-K chunks (one scale block
    /// each): 128-byte weight rows and 256-byte activation rows per stage in
    /// the ring, a chain temporary per tile and chunk above one split, and
    /// the FP8 body from its own header.
    #[test]
    fn fp8_strip_chain_entries_take_128_k_chunks() {
        for (m, n, k, strips, splits) in [
            (1u32, 4096u32, 4096u32, 1u8, 1u8),
            (8, 1536, 4096, 2, 4),
            (32, 4096, 1536, 4, 3),
            (16, 512, 4096, 1, 6),
        ] {
            let query = fp8_block_query(m, n, k);
            let spec = instantiate(DenseRequest {
                query,
                kernel: DenseKernel::StripChains { strips, splits },
                target: Target::Sm120f,
                symbol: "fp8_strip_probe_v1".into(),
                knobs: None,
            })
            .unwrap();
            let knobs = spec.key.knobs;
            let rows = 16 * u32::from(strips);
            let tile_n = m.next_power_of_two().max(8);
            assert_eq!(
                (
                    u32::from(knobs.tile_m),
                    u32::from(knobs.tile_n),
                    knobs.tile_k
                ),
                (rows, tile_n, 128)
            );
            let stage = rows * 128 + tile_n * 256;
            assert_eq!(u32::from(knobs.stages), (48 * 1024 / stage).clamp(2, 16));
            let tiles = n / rows;
            let plan = plan(&spec.key, row_counts(&spec.key, m)).unwrap();
            assert_eq!(plan.grid[0], tiles * u32::from(splits));
            assert_eq!(
                plan.dynamic_shared_bytes,
                u32::from(knobs.stages) * stage + 128
            );
            let temporaries = if splits == 1 {
                0
            } else {
                u64::from(tiles) * u64::from(k / 128) * u64::from(rows * tile_n)
            };
            assert_eq!(plan.workspace_bytes, (temporaries + u64::from(tiles)) * 4);
            let maps = tensor_maps(&spec.key, 1, 2).unwrap();
            assert_eq!(maps[0].box_dims, [128, rows]);
            assert_eq!(maps[0].swizzle, TensorMapSwizzle::B128);
            assert_eq!(maps[1].box_dims, [128, tile_n]);
            assert_eq!(maps[1].swizzle, TensorMapSwizzle::None);
            let cubin = cubin_spec(std::slice::from_ref(&spec)).unwrap();
            assert!(cubin.entries[0].body.starts_with(&format!(
                "chop::gemm::run_a16_strips_fp8_ordered<false, {strips}, {tile_n}, {}>(map_a",
                knobs.stages
            )));
            assert!(
                cubin
                    .headers
                    .iter()
                    .any(|h| h.contains("strip_chains_fp8_ordered.cuh"))
            );
            assert!(
                !cubin
                    .headers
                    .iter()
                    .any(|h| h.contains("strip_chains_ordered.cuh"))
            );
        }
    }

    /// Above 32 rows a chain entry runs one split in warp tiles: the
    /// one-split prefill tiles and stages, one CTA per tile at the capacity,
    /// no chain workspace or tickets, and the one-split body's warp-tiled
    /// twin from its own header.
    #[test]
    fn warp_tiled_prefill_chains_keep_the_one_split_tiles() {
        let make = |query: GemmQuery, kernel| {
            instantiate(DenseRequest {
                query,
                kernel,
                target: Target::Sm120f,
                symbol: "warp_probe_v1".into(),
                knobs: None,
            })
        };
        let mut fp8 = bf16_query(8192, 4096, 4096, DType::F32);
        fp8.weight = WeightFormat::Fp8Block128x128(kernel_core::contract::ScaleMode::ExactF32);
        fp8.epilogues.residual_add = true;
        let narrow = bf16_query(64, 320, 4096, DType::F32);
        for query in [bf16_query(8192, 2048, 4096, DType::Bf16), fp8, narrow] {
            let one = make(query, DenseKernel::Tiled { splits: 1 }).unwrap();
            let warps = make(query, DenseKernel::TiledChains { splits: 1 }).unwrap();
            let mut knobs = one.key.knobs;
            knobs.split_k_capacity = 1;
            assert_eq!(warps.key.knobs, knobs);
            let tiles = query.m_capacity.div_ceil(u32::from(knobs.tile_m))
                * query.n.div_ceil(u32::from(knobs.tile_n));
            let plan = plan(&warps.key, row_counts(&warps.key, query.m_capacity)).unwrap();
            assert_eq!(plan.grid[0], tiles);
            assert_eq!(plan.workspace_bytes, u64::from(tiles) * 4);
            assert_eq!(
                GemmBuffers::semaphore_capacity_bytes(&warps.key).unwrap(),
                0
            );
            let one_spec = cubin_spec(std::slice::from_ref(&one)).unwrap();
            let warp_spec = cubin_spec(std::slice::from_ref(&warps)).unwrap();
            let residual = query.epilogues.residual_add;
            assert_eq!(
                warp_spec.entries[0].body,
                one_spec.entries[0]
                    .body
                    .replace("run_a16<", "run_a16_warps_ordered<")
                    .replace(&format!(", false, {residual},"), &format!(", {residual},"))
            );
            assert_eq!(warp_spec.entries[0].params, one_spec.entries[0].params);
            assert!(!one_spec.headers.iter().any(|h| h.contains("tiled_warps")));
            assert!(warp_spec.headers.iter().any(|h| h.contains("tiled_warps")));
            assert!(!warp_spec.headers.iter().any(|h| h.contains("tiled_chains")));
        }
        let wide = CompileKnobs {
            tile_m: 128,
            tile_n: 256,
            ..make(
                bf16_query(8192, 2048, 4096, DType::Bf16),
                DenseKernel::TiledChains { splits: 1 },
            )
            .unwrap()
            .key
            .knobs
        };
        assert!(
            instantiate(DenseRequest {
                query: bf16_query(8192, 2048, 4096, DType::Bf16),
                kernel: DenseKernel::TiledChains { splits: 1 },
                target: Target::Sm120f,
                symbol: "warp_probe_v1".into(),
                knobs: Some(wide),
            })
            .is_err()
        );
    }

    /// FP8 128x128 prefill chains at `decoded_tile_knobs` run decoded weight
    /// tiles: 128 x 128 tiles in 64-K stages, one CTA per tile at the
    /// capacity, no workspace beyond the warp tiles' or tickets, swizzled
    /// activation and unswizzled code boxes, and the decoded body from its own
    /// header; decode buckets, SwiGLU, other tiles and blocks that are not
    /// whole are refused; BF16 weights at these knobs run staged tiles, and
    /// the default knobs keep the warp tiles.
    #[test]
    fn decoded_tile_prefill_chains_take_their_own_tiles() {
        let make = |query: GemmQuery, knobs| {
            instantiate(DenseRequest {
                query,
                kernel: DenseKernel::TiledChains { splits: 1 },
                target: Target::Sm120f,
                symbol: "decoded_probe_v1".into(),
                knobs,
            })
        };
        let fp8 = |m, n, k, output| {
            let mut query = fp8_block_query(m, n, k);
            query.output = output;
            query
        };
        let mut residual = fp8(8192, 4096, 512, DType::F32);
        residual.epilogues.residual_add = true;
        for query in [
            fp8(8192, 4096, 4096, DType::F32),
            fp8(8192, 1536, 4096, DType::Bf16),
            fp8(8192, 512, 4096, DType::Bf16),
            fp8(8192, 4096, 1536, DType::Bf16),
            residual,
            fp8(64, 4096, 3072, DType::F32),
        ] {
            let warps = make(query, None).unwrap();
            assert_eq!(warps.key.knobs.tile_k, 128);
            let spec = make(query, Some(decoded_tile_knobs())).unwrap();
            let knobs = spec.key.knobs;
            assert_eq!((knobs.tile_m, knobs.tile_n, knobs.tile_k), (128, 128, 64));
            let tiles = query.m_capacity.div_ceil(128) * query.n.div_ceil(128);
            let plan = plan(&spec.key, row_counts(&spec.key, query.m_capacity)).unwrap();
            assert_eq!(plan.grid, [tiles, 1, 1]);
            assert_eq!(plan.block, [THREADS, 1, 1]);
            assert_eq!(plan.dynamic_shared_bytes, 99_328);
            assert_eq!(plan.workspace_bytes, u64::from(tiles) * 4);
            assert_eq!(GemmBuffers::semaphore_capacity_bytes(&spec.key).unwrap(), 0);
            let maps = tensor_maps(&spec.key, 1, 2).unwrap();
            assert_eq!(maps[0].box_dims, [64, 128]);
            assert_eq!(maps[0].swizzle, TensorMapSwizzle::B128);
            assert_eq!(maps[1].box_dims, [64, 128]);
            assert_eq!(maps[1].swizzle, TensorMapSwizzle::None);
            let counts = row_counts(&spec.key, query.m_capacity);
            let buffers = GemmBuffers {
                output: 3,
                partial: 4,
                semaphores: 5,
                weight_scales: Some(6),
                activation_scales: None,
                residual: query.epilogues.residual_add.then_some(7),
                weight_global_scale: None,
                activation_global_scale: None,
            };
            assert!(arguments(&spec.key, counts, maps, buffers).is_ok());
            let mut swizzled = tensor_maps(&spec.key, 1, 2).unwrap();
            swizzled[1].swizzle = TensorMapSwizzle::B128;
            assert!(arguments(&spec.key, counts, swizzled, buffers).is_err());
            let cubin = cubin_spec(std::slice::from_ref(&spec)).unwrap();
            let residual = query.epilogues.residual_add;
            assert_eq!(
                cubin.entries[0].body,
                format!(
                    "chop::gemm::run_a16_decoded<{residual}>(map_a, map_b, output, partial, \
                     semaphores, live_m, n, k, splits, output_f32, weight_scales, residual, \
                     weight_global_scale);"
                )
            );
            assert_eq!(cubin.entries[0].max_threads, THREADS);
            assert!(cubin.headers.iter().any(|h| h.contains("decoded_tiles")));
            assert!(!cubin.headers.iter().any(|h| h.contains("tiled_warps")));
            let warp_cubin = cubin_spec(std::slice::from_ref(&warps)).unwrap();
            assert!(
                !warp_cubin
                    .headers
                    .iter()
                    .any(|h| h.contains("decoded_tiles"))
            );
        }
        let knobs = Some(decoded_tile_knobs());
        // BF16 weights at these knobs are staged tiles.
        let bf16 = make(bf16_query(8192, 4096, 4096, DType::Bf16), knobs).unwrap();
        let cubin = cubin_spec(std::slice::from_ref(&bf16)).unwrap();
        assert!(
            cubin.entries[0]
                .body
                .starts_with("chop::gemm::run_a16_staged<")
        );
        assert!(make(fp8(32, 4096, 4096, DType::Bf16), knobs).is_err());
        assert!(make(fp8(8192, 4160, 4096, DType::Bf16), knobs).is_err());
        assert!(make(fp8(8192, 4096, 4160, DType::Bf16), knobs).is_err());
        let mut swiglu = fp8(8192, 2048, 4096, DType::Bf16);
        swiglu.epilogues.swiglu = Some(SwigluColumns::ConcatenatedHalves);
        assert!(make(swiglu, knobs).is_err());
        let narrow = CompileKnobs {
            tile_n: 64,
            ..decoded_tile_knobs()
        };
        assert!(make(fp8(8192, 4096, 4096, DType::Bf16), Some(narrow)).is_err());
    }

    /// FP8 128x128 prefill chains at `balanced_decoded_tile_knobs` run the
    /// balanced decoded tiles: 64-row activation boxes, a CTA per 128-row
    /// tile at the capacity and a wave of halves, the decoded tiles' shared
    /// memory, and the balanced body from its own header (which includes the
    /// decoded tiles'); the same shapes are refused as at the decoded-tile
    /// knobs.
    #[test]
    fn balanced_decoded_tiles_split_the_final_wave() {
        let make = |query: GemmQuery, knobs| {
            instantiate(DenseRequest {
                query,
                kernel: DenseKernel::TiledChains { splits: 1 },
                target: Target::Sm120f,
                symbol: "balanced_probe_v1".into(),
                knobs,
            })
        };
        let fp8 = |m, n, k, output| {
            let mut query = fp8_block_query(m, n, k);
            query.output = output;
            query
        };
        let knobs = balanced_decoded_tile_knobs();
        assert_eq!((knobs.tile_m, knobs.tile_n, knobs.tile_k), (64, 128, 64));
        assert_eq!(
            CompileKnobs {
                tile_m: 128,
                ..knobs
            },
            decoded_tile_knobs()
        );
        let mut residual = fp8(8192, 4096, 512, DType::F32);
        residual.epilogues.residual_add = true;
        for (query, grid) in [
            (fp8(8192, 4096, 4096, DType::F32), 64 * 32 + 188),
            (fp8(8192, 512, 4096, DType::Bf16), 64 * 4 + 188),
            (residual, 64 * 32 + 188),
            (fp8(2048, 512, 4096, DType::Bf16), 16 * 4 + 188),
            (fp8(65, 4096, 3072, DType::F32), 32 + 188),
        ] {
            let spec = make(query, Some(knobs)).unwrap();
            let plan = plan(&spec.key, row_counts(&spec.key, query.m_capacity)).unwrap();
            assert_eq!(plan.grid, [grid, 1, 1], "{query:?}");
            assert_eq!(plan.block, [THREADS, 1, 1]);
            assert_eq!(plan.dynamic_shared_bytes, 99_328);
            let maps = tensor_maps(&spec.key, 1, 2).unwrap();
            assert_eq!(maps[0].box_dims, [64, 64]);
            assert_eq!(maps[0].swizzle, TensorMapSwizzle::B128);
            assert_eq!(maps[1].box_dims, [64, 128]);
            assert_eq!(maps[1].swizzle, TensorMapSwizzle::None);
            let cubin = cubin_spec(std::slice::from_ref(&spec)).unwrap();
            let residual = query.epilogues.residual_add;
            assert_eq!(
                cubin.entries[0].body,
                format!(
                    "chop::gemm::run_a16_decoded_balanced<{residual}, 188>(map_a, map_b, \
                     output, partial, semaphores, live_m, n, k, splits, output_f32, \
                     weight_scales, residual, weight_global_scale);"
                )
            );
            assert!(cubin.headers.iter().any(|h| h.contains("decoded_balanced")));
            assert!(!cubin.headers.iter().any(|h| h.contains("tiled_warps")));
            // Decoded tiles of the same shape take their own header only.
            let decoded = make(query, Some(decoded_tile_knobs())).unwrap();
            let decoded = cubin_spec(std::slice::from_ref(&decoded)).unwrap();
            assert!(
                !decoded
                    .headers
                    .iter()
                    .any(|h| h.contains("decoded_balanced"))
            );
        }
        let knobs = Some(knobs);
        assert!(make(fp8(32, 4096, 4096, DType::Bf16), knobs).is_err());
        assert!(make(fp8(8192, 4160, 4096, DType::Bf16), knobs).is_err());
        assert!(make(fp8(8192, 4096, 4160, DType::Bf16), knobs).is_err());
        let mut swiglu = fp8(8192, 2048, 4096, DType::Bf16);
        swiglu.epilogues.swiglu = Some(SwigluColumns::ConcatenatedHalves);
        assert!(make(swiglu, knobs).is_err());
    }

    /// BF16 prefill chains at `staged_tile_knobs` (128 x 128 tiles) run
    /// staged weight tiles: a CTA per 128-row tile at the capacity and a wave
    /// of halves, 128-row swizzled activation and weight boxes, three 32 KiB
    /// stages, and the staged body from its own header; FP8 weights at these
    /// knobs, decode buckets, SwiGLU and columns that are not whole 128-row
    /// blocks are refused, and the default knobs keep the warp tiles.
    #[test]
    fn staged_tiles_take_bf16_prefill_chains() {
        let make = |query: GemmQuery, knobs| {
            instantiate(DenseRequest {
                query,
                kernel: DenseKernel::TiledChains { splits: 1 },
                target: Target::Sm120f,
                symbol: "staged_probe_v1".into(),
                knobs,
            })
        };
        let knobs = staged_tile_knobs();
        for (query, grid) in [
            (bf16_query(8192, 2048, 4096, DType::Bf16), 64 * 16 + 188),
            (bf16_query(8192, 4096, 2048, DType::Bf16), 64 * 32 + 188),
            (bf16_query(2048, 128, 4096, DType::Bf16), 16 + 188),
            (bf16_query(8192, 2048, 128, DType::F32), 64 * 16 + 188),
        ] {
            let spec = make(query, Some(knobs)).unwrap();
            let plan = plan(&spec.key, row_counts(&spec.key, query.m_capacity)).unwrap();
            assert_eq!(plan.grid, [grid, 1, 1], "{query:?}");
            assert_eq!(plan.block, [THREADS, 1, 1]);
            assert_eq!(plan.dynamic_shared_bytes, 99_328);
            assert_eq!(GemmBuffers::semaphore_capacity_bytes(&spec.key).unwrap(), 0);
            let maps = tensor_maps(&spec.key, 1, 2).unwrap();
            assert_eq!(maps[0].box_dims, [64, 128]);
            assert_eq!(maps[0].swizzle, TensorMapSwizzle::B128);
            assert_eq!(maps[1].box_dims, [64, 128]);
            assert_eq!(maps[1].swizzle, TensorMapSwizzle::B128);
            let cubin = cubin_spec(std::slice::from_ref(&spec)).unwrap();
            assert_eq!(
                cubin.entries[0].body,
                "chop::gemm::run_a16_staged<false, 188>(map_a, map_b, output, partial, \
                 semaphores, live_m, n, k, splits, output_f32, weight_scales, residual, \
                 weight_global_scale);"
            );
            assert_eq!(cubin.entries[0].max_threads, THREADS);
            assert!(cubin.headers.iter().any(|h| h.contains("staged_tiles")));
            assert!(!cubin.headers.iter().any(|h| h.contains("tiled_warps")));
            let warps = make(query, None).unwrap();
            assert_eq!(warps.key.knobs.warps, 12);
            let warp_cubin = cubin_spec(std::slice::from_ref(&warps)).unwrap();
            assert!(
                !warp_cubin
                    .headers
                    .iter()
                    .any(|h| h.contains("staged_tiles"))
            );
        }
        let knobs = Some(knobs);
        // FP8 weights at 128 x 128 tiles and 64-K stages are decoded tiles.
        let fp8 = make(fp8_block_query(8192, 2048, 4096), knobs).unwrap();
        let fp8 = cubin_spec(std::slice::from_ref(&fp8)).unwrap();
        assert!(
            fp8.entries[0]
                .body
                .starts_with("chop::gemm::run_a16_decoded<")
        );
        assert!(make(bf16_query(32, 2048, 4096, DType::Bf16), knobs).is_err());
        assert!(make(bf16_query(8192, 320, 4096, DType::F32), knobs).is_err());
        let mut swiglu = bf16_query(8192, 2048, 4096, DType::Bf16);
        swiglu.epilogues.swiglu = Some(SwigluColumns::ConcatenatedHalves);
        assert!(make(swiglu, knobs).is_err());
        // The kernel traps on a K tail, so preparation refuses one.
        for k in [32, 96, 160] {
            let refused = make(bf16_query(65, 128, k, DType::Bf16), knobs);
            assert!(refused.is_err(), "k {k}");
        }
        for k in [64, 128] {
            make(bf16_query(65, 128, k, DType::Bf16), knobs).unwrap();
        }
    }

    #[test]
    fn external_semaphore_capacity_covers_all_dense_engines() {
        let make = |capacity, n, k, output, kernel| {
            instantiate(DenseRequest {
                query: bf16_query(capacity, n, k, output),
                kernel,
                target: Target::Sm120f,
                symbol: "semaphore_probe_v1".into(),
                knobs: None,
            })
            .unwrap()
        };
        let tiled = make(8, 256, 1024, DType::Bf16, DenseKernel::Tiled { splits: 2 });
        assert_eq!(
            GemmBuffers::semaphore_capacity_bytes(&tiled.key).unwrap(),
            16
        );
        let decode = make(
            8,
            256,
            1024,
            DType::Bf16,
            DenseKernel::FragmentDecode { splits: 2 },
        );
        assert_eq!(decode.key.knobs.tile_m, 128);
        assert_eq!(
            GemmBuffers::semaphore_capacity_bytes(&decode.key).unwrap(),
            8
        );
        let router = make(
            8,
            256,
            4096,
            DType::F32,
            DenseKernel::FragmentRouter {
                splits: ROUTER_SPLITS,
            },
        );
        assert_eq!(
            GemmBuffers::semaphore_capacity_bytes(&router.key).unwrap(),
            64
        );
        for kernel in [
            DenseKernel::Tiled { splits: 1 },
            DenseKernel::FragmentDecode { splits: 1 },
            DenseKernel::FragmentCluster { warp_blocks: 2 },
        ] {
            let spec = make(8, 256, 1024, DType::Bf16, kernel);
            assert_eq!(GemmBuffers::semaphore_capacity_bytes(&spec.key).unwrap(), 0);
        }
        let prefill = make(
            9472,
            256,
            1024,
            DType::Bf16,
            DenseKernel::FragmentPrefill { segments: 2 },
        );
        let bytes = GemmBuffers::semaphore_capacity_bytes(&prefill.key).unwrap();
        let grid = plan(&prefill.key, row_counts(&prefill.key, 9472))
            .unwrap()
            .grid[0];
        assert_eq!(bytes, u64::from(grid) * 8);
        // Smaller live batches can have stream-K tails even if a full batch
        // needs no counters. Reserve both counter words for every grid owner.
        for rows in (128..=9472).step_by(128) {
            let tiles = rows / 128 * 2;
            let tail = tiles % grid;
            assert!(bytes >= u64::from(tail) * 8);
        }
        let single = make(
            512,
            256,
            1024,
            DType::Bf16,
            DenseKernel::FragmentPrefill { segments: 1 },
        );
        assert_eq!(
            GemmBuffers::semaphore_capacity_bytes(&single.key).unwrap(),
            0
        );
    }

    fn counts(m: u32, split_k: u8) -> RuntimeCounts {
        RuntimeCounts {
            m,
            batch: m,
            context: 0,
            routed_rows: 0,
            split_k,
        }
    }

    #[test]
    fn catalog_profile_is_independent_of_symbol_spelling() {
        let entry = instance(
            Profile::StdDenseMoe,
            "model_arbitrary_name",
            bf16_query(1, 128, 128, DType::Bf16),
            64,
            8,
            DenseKernel::Tiled { splits: 1 },
        );
        assert_eq!(entry.profile, Profile::StdDenseMoe);
    }

    /// Every fragment prefill entry compiles its declared K segments as its
    /// split capacity, and its arguments carry them at every live M.
    #[test]
    fn prefill_entries_carry_their_declared_segments() {
        let mut prefill = 0;
        for item in own_catalog() {
            let Ok(segments) = prefill_segments(&item.key) else {
                continue;
            };
            assert_eq!(item.key.knobs.split_k_capacity, segments, "{}", item.symbol);
            prefill += 1;
        }
        assert_eq!(prefill, 7);
        let segments = |symbol: &str| {
            let item = own_catalog()
                .into_iter()
                .find(|item| item.symbol == symbol)
                .unwrap_or_else(|| panic!("{symbol}"));
            prefill_segments(&item.key).unwrap()
        };
        assert_eq!(segments("chop_gemm_std_dense_prefill_v1"), 4);
        assert_eq!(segments("chop_gemm_std_router_prefill_v1"), 8);
    }

    /// No shape, row count or name gives an entry a kernel: a key without
    /// one is refused, and a declared kernel must fit its query.
    #[test]
    fn entries_without_a_declared_kernel_are_refused() {
        let item = own_catalog()
            .into_iter()
            .find(|item| item.symbol == "chop_gemm_std_dense_prefill_v1")
            .unwrap();
        let mut key = item.key.clone();
        key.dense = None;
        let expected = Refusal("a dense GEMM entry must declare its kernel");
        assert_eq!(plan(&key, counts(8192, 1)), Err(expected));
        assert_eq!(choose_split_k(&key), Err(expected));
        assert!(!fragment_weight_enabled(&key));
        let query = bf16_query(8192, 256, 4096, DType::F32);
        for (kernel, refusal) in [
            (
                DenseKernel::FragmentPrefill { segments: 9 },
                "fragment prefill takes 1 to 8 K segments",
            ),
            (WHOLE_K_DECODE, "fragment decode serves at most 32 rows"),
        ] {
            assert_eq!(dense_supported(kernel, &query), Err(Refusal(refusal)));
        }
        let router = |n| bf16_query(1, n, 4096, DType::F32);
        let kernel = DenseKernel::FragmentRouter {
            splits: ROUTER_SPLITS,
        };
        assert!(dense_supported(kernel, &router(256)).is_ok());
        assert!(dense_supported(kernel, &router(512)).is_err());
        let wrong = DenseKernel::FragmentCluster { warp_blocks: 1 };
        assert_eq!(
            dense_supported(wrong, &bf16_query(8, 4096, 4096, DType::Bf16)),
            Err(Refusal(
                "fragment cluster decode holds 16 fragment tiles per warp"
            ))
        );
    }

    #[test]
    fn refuses_capacity_and_unimplemented_formats() {
        let item = &own_catalog()[0];
        assert!(plan(&item.key, counts(2, 1)).is_err());
        let mut key = item.key.clone();
        let StaticQuery::Gemm(ref mut q) = key.query else {
            unreachable!()
        };
        q.weight = WeightFormat::Nvfp4;
        q.activation = ActivationFormat::Mxfp8;
        assert!(plan(&key, counts(1, 1)).is_err());
        let mx = own_catalog()
            .into_iter()
            .find(|s| s.symbol == "chop_gemm_format_mxfp8_decode_v1")
            .unwrap();
        assert!(plan(&mx.key, counts(1, 1)).is_ok());
    }

    #[test]
    fn packed_fp4_prefill_fits_shared_memory() {
        let item = own_catalog()
            .into_iter()
            .find(|s| s.symbol == "chop_gemm_std_dense_nvfp4_prefill_v1")
            .unwrap();
        let launch = plan(&item.key, counts(8192, 1)).unwrap();
        assert_eq!(launch.dynamic_shared_bytes, 98_432);
        assert_eq!(launch.grid[0], 188);
        let a4 = own_catalog()
            .into_iter()
            .find(|s| s.symbol == "chop_gemm_std_dense_nvfp4_a4_prefill_v1")
            .unwrap();
        let a4_launch = plan(&a4.key, counts(8192, 1)).unwrap();
        assert_eq!(a4_launch.dynamic_shared_bytes, 98_432);
        assert_eq!(a4_launch.grid[0], 188);
    }
}
