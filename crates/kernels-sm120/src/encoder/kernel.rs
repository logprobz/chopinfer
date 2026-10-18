//! One encoder kernel instantiation from its `MultimodalQuery`: the entry
//! body, its fixed ABI, the header holding its template, its launch plan
//! and its contract. Generated catalogs build every serving entry through
//! these functions. Retired BF16-legacy queries cannot build GPU entries.

use std::path::PathBuf;

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, CompileKey, CompileKnobs, DType, GeluMode, GemmQuery,
    InstantiationSpec, LaunchPlan, MultimodalQuery, NumericalContract, OpFamily, Refusal,
    RotaryStyle, StaticQuery, Target, Validate,
};

use super::attention;

/// Persistent CTAs of `encoder_gemm.cuh`: 128 tokens by 128 weight rows (64
/// output columns for SwiGLU), eight math warps and a four-warp producer
/// group, three 64-K stages of both operands in shared memory.
pub const GEMM_THREADS: u32 = 384;
/// `encoder_gemm_pairs.cuh`: the same CTAs, two stages of 128 F32
/// activation rows and 128 BF16 weight rows of 64 K.
pub const PAIRS_GEMM_SHARED_BYTES: u32 = 2 * (128 * 64 * 4 + 128 * 64 * 2) + 1024;
/// Threads of every row entry.
pub const ROW_THREADS: u32 = 256;

pub fn is_attention(query: MultimodalQuery) -> bool {
    matches!(
        query,
        MultimodalQuery::ContiguousAttention { .. }
            | MultimodalQuery::ContiguousAttentionPairs { .. }
    )
}

pub fn is_gemm(query: MultimodalQuery) -> bool {
    matches!(
        query,
        MultimodalQuery::Gemm(_) | MultimodalQuery::GemmBf16(_) | MultimodalQuery::GemmPairs(_)
    )
}

/// SM120's limits on a valid query: the head dims of the contiguous
/// attention engine (`attention/varlen.cuh`), of the q/k RMSNorm's
/// 32-lane heads (`encoder/rows/qk_rms_f32.cuh`) and the row gather's
/// float4 blocks (`encoder/rows/gather_rows_f32.cuh`).
pub fn limits(query: MultimodalQuery) -> Result<(), Refusal> {
    match query {
        MultimodalQuery::ContiguousAttention { head_dim, .. }
        | MultimodalQuery::ContiguousAttentionPairs { head_dim, .. }
            if !matches!(head_dim, 64 | 72) =>
        {
            Err(Refusal(
                "SM120 contiguous attention has head dims 64 and 72",
            ))
        }
        MultimodalQuery::QkRmsNormF32 { head_dim, .. }
            if !head_dim.is_multiple_of(32) || head_dim > 256 =>
        {
            Err(Refusal(
                "SM120 q/k RMSNorm heads are 32-lane steps up to 256 dims",
            ))
        }
        MultimodalQuery::GatherRowsF32 { width, .. } if !width.is_multiple_of(4) => Err(Refusal(
            "SM120 row gathers copy float4 blocks of a width divisible by 4",
        )),
        _ => Ok(()),
    }
}

/// The instantiation contract of `symbol` running `query`.
pub fn contract(symbol: &str, query: MultimodalQuery) -> Result<InstantiationSpec, Refusal> {
    query.validate()?;
    limits(query)?;
    header(query)?;
    let family = if is_attention(query) {
        OpFamily::Attention
    } else if is_gemm(query) {
        OpFamily::Gemm
    } else {
        OpFamily::Row
    };
    let spec = InstantiationSpec {
        family,
        key: CompileKey {
            query: StaticQuery::Multimodal(query),
            knobs: CompileKnobs {
                tile_m: 1,
                tile_n: 128,
                tile_k: 1,
                stages: 1,
                swizzle: 0,
                warps: 4,
                split_k_capacity: 1,
            },
            dense: None,
            target: Target::Sm120f,
        },
        symbol: symbol.to_owned(),
    };
    spec.validate()?;
    query.numerical_contract_id()?;
    Ok(spec)
}

/// The prepared row capacity (tokens for attention, M for GEMM).
pub fn capacity(query: MultimodalQuery) -> u32 {
    match query {
        MultimodalQuery::AudioRopeBf16 { capacity }
        | MultimodalQuery::EmbeddingSumBf16 { capacity, .. }
        | MultimodalQuery::ResidualAdd { capacity, .. }
        | MultimodalQuery::Bf16ToF32 { capacity, .. }
        | MultimodalQuery::GeluRows { capacity, .. }
        | MultimodalQuery::SplitQkv { capacity, .. }
        | MultimodalQuery::ContiguousAttention { capacity, .. }
        | MultimodalQuery::LayerNorm { capacity, .. }
        | MultimodalQuery::ScratchRope { capacity, .. }
        | MultimodalQuery::Permute { capacity, .. }
        | MultimodalQuery::Im2col1d { capacity, .. }
        | MultimodalQuery::EmbeddingSum { capacity, .. }
        | MultimodalQuery::EmbeddingOverride { capacity, .. }
        | MultimodalQuery::InterpolatedPositionAddF32 { capacity, .. }
        | MultimodalQuery::LayerNormF32 { capacity, .. }
        | MultimodalQuery::RopeQkvPairs { capacity, .. }
        | MultimodalQuery::ContiguousAttentionPairs { capacity, .. }
        | MultimodalQuery::RmsNormF32 { capacity, .. }
        | MultimodalQuery::PermuteF32 { capacity, .. }
        | MultimodalQuery::GatherRowsF32 { capacity, .. }
        | MultimodalQuery::SwigluF32 { capacity, .. }
        | MultimodalQuery::ResidualAddF32 { capacity, .. }
        | MultimodalQuery::Im2col1dF32 { capacity, .. }
        | MultimodalQuery::QkRmsNormF32 { capacity, .. }
        | MultimodalQuery::SwigluClampedF32 { capacity, .. }
        | MultimodalQuery::GeluRowsF32 { capacity, .. }
        | MultimodalQuery::PadColumnsF32 { capacity, .. }
        | MultimodalQuery::CastBf16F32 { capacity, .. } => capacity,
        MultimodalQuery::Gemm(query)
        | MultimodalQuery::GemmBf16(query)
        | MultimodalQuery::GemmPairs(query) => query.m_capacity,
    }
}

/// `query` prepared for `rows` rows.
pub fn with_capacity(query: MultimodalQuery, rows: u32) -> MultimodalQuery {
    let mut query = query;
    match &mut query {
        MultimodalQuery::AudioRopeBf16 { capacity }
        | MultimodalQuery::EmbeddingSumBf16 { capacity, .. }
        | MultimodalQuery::ResidualAdd { capacity, .. }
        | MultimodalQuery::Bf16ToF32 { capacity, .. }
        | MultimodalQuery::GeluRows { capacity, .. }
        | MultimodalQuery::SplitQkv { capacity, .. }
        | MultimodalQuery::ContiguousAttention { capacity, .. }
        | MultimodalQuery::LayerNorm { capacity, .. }
        | MultimodalQuery::ScratchRope { capacity, .. }
        | MultimodalQuery::Permute { capacity, .. }
        | MultimodalQuery::Im2col1d { capacity, .. }
        | MultimodalQuery::EmbeddingSum { capacity, .. }
        | MultimodalQuery::EmbeddingOverride { capacity, .. }
        | MultimodalQuery::InterpolatedPositionAddF32 { capacity, .. }
        | MultimodalQuery::LayerNormF32 { capacity, .. }
        | MultimodalQuery::RopeQkvPairs { capacity, .. }
        | MultimodalQuery::ContiguousAttentionPairs { capacity, .. }
        | MultimodalQuery::RmsNormF32 { capacity, .. }
        | MultimodalQuery::PermuteF32 { capacity, .. }
        | MultimodalQuery::GatherRowsF32 { capacity, .. }
        | MultimodalQuery::SwigluF32 { capacity, .. }
        | MultimodalQuery::ResidualAddF32 { capacity, .. }
        | MultimodalQuery::Im2col1dF32 { capacity, .. }
        | MultimodalQuery::QkRmsNormF32 { capacity, .. }
        | MultimodalQuery::SwigluClampedF32 { capacity, .. }
        | MultimodalQuery::GeluRowsF32 { capacity, .. }
        | MultimodalQuery::PadColumnsF32 { capacity, .. }
        | MultimodalQuery::CastBf16F32 { capacity, .. } => *capacity = rows,
        MultimodalQuery::Gemm(query)
        | MultimodalQuery::GemmBf16(query)
        | MultimodalQuery::GemmPairs(query) => query.m_capacity = rows,
    }
    query
}

/// The launch of `symbol` over `live_rows`: every entry grids for its
/// prepared capacity and reads the live count on the device.
pub fn plan(symbol: &str, query: MultimodalQuery, live_rows: u32) -> Result<LaunchPlan, Refusal> {
    contract(symbol, query)?;
    let capacity = capacity(query);
    if live_rows == 0 || live_rows > capacity {
        return Err(Refusal("encoder rows exceed prepared capacity"));
    }
    if is_attention(query) {
        return attention::plan(query);
    }
    if let MultimodalQuery::GemmPairs(query) = query {
        return gemm_plan(query, PAIRS_GEMM_SHARED_BYTES);
    }
    let plan = LaunchPlan {
        grid: [capacity, 1, 1],
        block: [ROW_THREADS, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
}

fn gemm_plan(query: GemmQuery, shared_bytes: u32) -> Result<LaunchPlan, Refusal> {
    let width = if query.epilogues.swiglu.is_some() {
        query.n / 2
    } else {
        query.n
    };
    let feature_tiles = width
        / if query.epilogues.swiglu.is_some() {
            64
        } else {
            128
        };
    // Grid for the prepared capacity; the kernel reads the live row count on
    // the device and its persistent CTAs loop over the live tiles.
    let tiles = query.m_capacity.div_ceil(128) * feature_tiles;
    let plan = LaunchPlan {
        grid: [tiles.clamp(1, crate::gemm::SM_COUNT), 1, 1],
        block: [GEMM_THREADS, 1, 1],
        dynamic_shared_bytes: shared_bytes,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
}

/// Dynamic shared memory an entry needs, for symbol configuration at load
/// time. Plain attention entries need none and are not configured.
pub fn dynamic_shared_bytes(query: MultimodalQuery) -> Option<u32> {
    match query {
        MultimodalQuery::GemmPairs(_) => Some(PAIRS_GEMM_SHARED_BYTES),
        MultimodalQuery::ContiguousAttention { .. } => None,
        MultimodalQuery::ContiguousAttentionPairs { .. } => attention::plan(query)
            .ok()
            .map(|plan| plan.dynamic_shared_bytes),
        _ => Some(0),
    }
}

/// Checks `values` against an entry's ABI and packs them.
pub fn pack_arguments(
    params: &[EntryParam],
    values: Vec<Argument>,
) -> Result<ArgumentList, Refusal> {
    if values.len() != params.len() {
        return Err(Refusal("encoder argument count mismatch"));
    }
    for (param, value) in params.iter().zip(&values) {
        if !matches!(
            (param, value),
            (
                EntryParam::DevicePointer { .. },
                Argument::DeviceBuffer(_) | Argument::NullDevicePointer
            ) | (EntryParam::U32 { .. }, Argument::U32(_))
                | (EntryParam::U64 { .. }, Argument::U64(_))
        ) {
            return Err(Refusal("encoder argument ABI kind mismatch"));
        }
    }
    let args = ArgumentList(values);
    args.validate()?;
    Ok(args)
}

/// Threads per CTA.
pub fn threads(query: MultimodalQuery) -> u32 {
    if is_gemm(query) {
        GEMM_THREADS
    } else if is_attention(query) {
        attention::THREADS
    } else {
        ROW_THREADS
    }
}

/// The `#include` line of the header holding the query's template. Each
/// row template has a header of its own (`cuda/encoder/rows/`), so a cubin's
/// source closure holds only the row templates its entries call.
pub fn header(query: MultimodalQuery) -> Result<&'static str, Refusal> {
    Ok(match query {
        MultimodalQuery::GemmPairs(_) => "#include \"encoder/linear_pairs.cuh\"",
        MultimodalQuery::ContiguousAttentionPairs { .. } => {
            "#include \"attention/varlen_pairs.cuh\""
        }
        MultimodalQuery::EmbeddingOverride { .. } => {
            "#include \"encoder/rows/embedding_override.cuh\""
        }
        MultimodalQuery::InterpolatedPositionAddF32 { .. } => {
            "#include \"encoder/rows/interpolated_position_add_f32.cuh\""
        }
        MultimodalQuery::LayerNormF32 { .. } => "#include \"encoder/rows/layernorm_f32.cuh\"",
        MultimodalQuery::RopeQkvPairs { .. } => "#include \"encoder/rows/rope_qkv_pairs.cuh\"",
        MultimodalQuery::RmsNormF32 { .. } => "#include \"encoder/rows/rms_f32.cuh\"",
        MultimodalQuery::PermuteF32 { .. } => "#include \"encoder/rows/permute_f32.cuh\"",
        MultimodalQuery::GatherRowsF32 { .. } => "#include \"encoder/rows/gather_rows_f32.cuh\"",
        MultimodalQuery::SwigluF32 { .. } => "#include \"encoder/rows/swiglu_f32.cuh\"",
        MultimodalQuery::ResidualAddF32 { .. } => "#include \"encoder/rows/residual_add_f32.cuh\"",
        MultimodalQuery::Im2col1dF32 { .. } => "#include \"encoder/rows/im2col_f32.cuh\"",
        MultimodalQuery::QkRmsNormF32 { .. } => "#include \"encoder/rows/qk_rms_f32.cuh\"",
        MultimodalQuery::SwigluClampedF32 { .. } => {
            "#include \"encoder/rows/swiglu_clamped_f32.cuh\""
        }
        MultimodalQuery::GeluRowsF32 { .. } => "#include \"encoder/rows/gelu_f32.cuh\"",
        MultimodalQuery::PadColumnsF32 { .. } => "#include \"encoder/rows/pad_columns_f32.cuh\"",
        MultimodalQuery::CastBf16F32 { .. } => "#include \"encoder/rows/cast_bf16_f32.cuh\"",
        _ => return Err(Refusal("BF16-legacy encoder kernels are retired")),
    })
}

/// The `#include` lines `entries` need, sorted.
fn headers<'a>(queries: impl Iterator<Item = &'a MultimodalQuery>) -> Result<Vec<String>, Refusal> {
    let mut headers = queries
        .map(|query| header(*query).map(str::to_owned))
        .collect::<Result<Vec<_>, _>>()?;
    headers.sort();
    headers.dedup();
    Ok(headers)
}

/// A cubin of `entries` named `name`, including only their headers.
pub fn cubin_spec<'a>(
    name: &str,
    include_dirs: Vec<PathBuf>,
    entries: impl Iterator<Item = (&'a str, &'a MultimodalQuery, &'a [EntryParam])> + Clone,
) -> Result<CubinSpec, Refusal> {
    Ok(CubinSpec {
        name: name.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs,
        headers: headers(entries.clone().map(|(_, query, _)| query))?,
        defines: vec![],
        entries: entries
            .map(|(symbol, query, params)| {
                Ok(EntrySpec::new(
                    symbol.to_owned(),
                    params.to_vec(),
                    body(*query)?,
                    threads(*query),
                    1,
                ))
            })
            .collect::<Result<_, Refusal>>()?,
    })
}

/// The encoder GEMM templates' GELU selector.
fn gelu_index(gelu: Option<GeluMode>) -> u32 {
    match gelu {
        None => 0,
        Some(GeluMode::Erf) => 1,
        Some(GeluMode::Tanh) => 2,
    }
}

/// The entry body calling the query's template.
pub fn body(query: MultimodalQuery) -> Result<String, Refusal> {
    Ok(match query {
        MultimodalQuery::ContiguousAttentionPairs { .. } => attention::body(query)?,
        MultimodalQuery::EmbeddingOverride { width, .. } => format!(
            "chop::multimodal::embedding_override<{width}>(activation,arena,source,live_rows,arena_rows);"
        ),
        MultimodalQuery::InterpolatedPositionAddF32 {
            width, table_rows, ..
        } => format!(
            "chop::multimodal::interpolated_position_add_f32<{width},{table_rows}>(input,table,indices,weights,output,live_rows);"
        ),
        MultimodalQuery::LayerNormF32 { width, .. } => format!(
            "chop::multimodal::layernorm_f32<{width}>(input,weight,bias,output,live_rows,__uint_as_float(eps_bits));"
        ),
        MultimodalQuery::RopeQkvPairs {
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            style,
            ..
        } => format!(
            "chop::multimodal::rope_qkv_pairs<{q_heads},{kv_heads},{head_dim},{rotary_dim},{}>(rows,cos,sin,live_rows);",
            style == RotaryStyle::Neox
        ),
        MultimodalQuery::GemmPairs(query) => format!(
            "chop::gemm::encoder_gemm_pairs<{},{},chop::encoder::LinearPairs<{},{},{},{}>>(activation,weight,bias,output,live_rows);",
            query.n,
            query.k,
            query.epilogues.bias,
            gelu_index(query.epilogues.gelu),
            query.epilogues.residual_add,
            if query.output == DType::F32 {
                "float"
            } else {
                "uint16_t"
            },
        ),
        MultimodalQuery::RmsNormF32 { width, .. } => format!(
            "chop::multimodal::rms_f32<{width}>(input,weight,output,live_rows,__uint_as_float(eps_bits));"
        ),
        MultimodalQuery::PermuteF32 { width, .. } => {
            format!("chop::multimodal::permute_f32<{width}>(input,indices,output,live_rows);")
        }
        MultimodalQuery::GatherRowsF32 { width, group, .. } => format!(
            "chop::multimodal::gather_rows_f32<{width},{group}>(input,indices,output,live_rows);"
        ),
        MultimodalQuery::SwigluF32 { hidden, .. } => {
            format!("chop::multimodal::swiglu_f32<{hidden}>(input,output,live_rows);")
        }
        MultimodalQuery::ResidualAddF32 { width, .. } => {
            format!("chop::multimodal::residual_add_f32<{width}>(input,residual,output,live_rows);")
        }
        MultimodalQuery::Im2col1dF32 {
            channels,
            kernel,
            stride,
            pad_left,
            pad_right,
            ..
        } => format!(
            "chop::multimodal::im2col_f32<{channels},{kernel},{stride},{pad_left},{pad_right}>(input,output,live_frames);"
        ),
        MultimodalQuery::QkRmsNormF32 {
            q_heads,
            kv_heads,
            head_dim,
            ..
        } => format!(
            "chop::multimodal::qk_rms_f32<{q_heads},{kv_heads},{head_dim}>(rows,q_weight,k_weight,live_rows,__uint_as_float(eps_bits));"
        ),
        MultimodalQuery::SwigluClampedF32 { hidden, .. } => format!(
            "chop::multimodal::swiglu_clamped_f32<{hidden}>(input,output,live_rows,__uint_as_float(limit_bits));"
        ),
        MultimodalQuery::GeluRowsF32 { width, .. } => {
            format!("chop::multimodal::gelu_f32<{width}>(input,output,live_rows);")
        }
        MultimodalQuery::PadColumnsF32 { input, output, .. } => {
            format!("chop::multimodal::pad_columns_f32<{input},{output}>(input,output,live_rows);")
        }
        MultimodalQuery::CastBf16F32 { width, .. } => {
            format!("chop::multimodal::cast_bf16_f32<{width}>(input,output,live_rows);")
        }
        _ => return Err(Refusal("BF16-legacy encoder kernels are retired")),
    })
}

fn ptr(name: &str, ty: &str) -> EntryParam {
    EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    }
}

fn u32_arg(name: &str) -> EntryParam {
    EntryParam::U32 { name: name.into() }
}

/// The fixed ABI of the query's entry: device pointers, then scalars. The
/// live row count is always a device pointer, so entries run under graph
/// capture.
pub fn params(query: MultimodalQuery) -> Vec<EntryParam> {
    let live = || ptr("live_rows", "const uint32_t");
    match query {
        MultimodalQuery::ContiguousAttentionPairs { .. } => attention::pairs_params(),
        MultimodalQuery::EmbeddingOverride { .. } => vec![
            ptr("activation", "uint16_t"),
            ptr("arena", "const uint16_t"),
            ptr("source", "const int64_t"),
            live(),
            EntryParam::U64 {
                name: "arena_rows".into(),
            },
        ],
        MultimodalQuery::InterpolatedPositionAddF32 { .. } => vec![
            ptr("input", "const float"),
            ptr("table", "const uint16_t"),
            ptr("indices", "const int32_t"),
            ptr("weights", "const float"),
            ptr("output", "float"),
            live(),
        ],
        MultimodalQuery::LayerNormF32 { .. } => vec![
            ptr("input", "const float"),
            ptr("weight", "const uint16_t"),
            ptr("bias", "const uint16_t"),
            ptr("output", "float"),
            live(),
            u32_arg("eps_bits"),
        ],
        MultimodalQuery::RopeQkvPairs { .. } => vec![
            ptr("rows", "float"),
            ptr("cos", "const float"),
            ptr("sin", "const float"),
            live(),
        ],
        MultimodalQuery::RmsNormF32 { .. } => vec![
            ptr("input", "const float"),
            ptr("weight", "const uint16_t"),
            ptr("output", "float"),
            live(),
            u32_arg("eps_bits"),
        ],
        MultimodalQuery::PermuteF32 { .. } | MultimodalQuery::GatherRowsF32 { .. } => vec![
            ptr("input", "const float"),
            ptr("indices", "const int32_t"),
            ptr("output", "float"),
            live(),
        ],
        MultimodalQuery::SwigluF32 { .. } => {
            vec![ptr("input", "const float"), ptr("output", "float"), live()]
        }
        MultimodalQuery::ResidualAddF32 { .. } => vec![
            ptr("input", "const float"),
            ptr("residual", "const float"),
            ptr("output", "float"),
            live(),
        ],
        MultimodalQuery::Im2col1dF32 { .. } => vec![
            ptr("input", "const float"),
            ptr("output", "float"),
            ptr("live_frames", "const uint32_t"),
        ],
        MultimodalQuery::QkRmsNormF32 { .. } => vec![
            ptr("rows", "float"),
            ptr("q_weight", "const uint16_t"),
            ptr("k_weight", "const uint16_t"),
            live(),
            u32_arg("eps_bits"),
        ],
        MultimodalQuery::SwigluClampedF32 { .. } => vec![
            ptr("input", "const float"),
            ptr("output", "float"),
            live(),
            u32_arg("limit_bits"),
        ],
        MultimodalQuery::GeluRowsF32 { .. } | MultimodalQuery::PadColumnsF32 { .. } => {
            vec![ptr("input", "const float"), ptr("output", "float"), live()]
        }
        MultimodalQuery::CastBf16F32 { .. } => {
            vec![
                ptr("input", "const float"),
                ptr("output", "uint16_t"),
                live(),
            ]
        }
        MultimodalQuery::GemmPairs(query) => vec![
            ptr("activation", "const float"),
            ptr("weight", "const uint16_t"),
            ptr("bias", "const uint16_t"),
            ptr(
                "output",
                if query.output == DType::F32 {
                    "float"
                } else {
                    "uint16_t"
                },
            ),
            live(),
        ],
        _ => Vec::new(),
    }
}
