//! Row builders with explicit profiles, names and geometry.

use super::*;
use kernel_core::contract::{ArgmaxLayout, RotaryStyle};

/// Build the rmsnorm entry from explicit geometry. Refuses what the typed
/// RMSNorm (`rmsnorm::RmsNorm::new`) refuses.
pub fn rmsnorm_entry(
    profile: Profile,
    name: &str,
    width: u32,
    head: u32,
    kind: u32,
    residual: bool,
) -> Result<RowInstantiation, Refusal> {
    let norm_kind = match kind {
        1 => RmsNormKind::GemmaOnePlusWeight,
        2 => RmsNormKind::QkPerHead { head_dim: head },
        3 => RmsNormKind::HfBf16Intermediate,
        _ => RmsNormKind::Plain,
    };
    let numerics = rmsnorm::RmsNorm::numerics(norm_kind);
    let requirement =
        kernel_core::rmsnorm::RmsNormRequirement::bf16(width, 1e-6, norm_kind, residual, numerics);
    let selected = rmsnorm::RmsNorm::new(requirement, 8192, rmsnorm::RmsNormTarget::SM120F)?;
    let entry = selected.entry(format!("chop_row_rms_{name}_v1"));
    assert_eq!(tuning().row_threads, entry.max_threads);
    checked(RowInstantiation {
        profile,
        symbol: entry.symbol,
        query: requirement.query(),
        width,
        grid: Grid::Rows,
        threads: entry.max_threads,
        params: entry.params,
        body: entry.body,
        invariance: Invariance::NONE,
    })
}

/// Build the quantize entries from explicit geometry: `quantize` of each
/// format but MXFP8, one CTA of the quantizer tuning per group;
/// `quantize_mxfp8_rows`, one CTA of the row tuning per row; and
/// `nvfp4_row_amax`. Refuses a width that is not whole 32-wide groups (the
/// MX and NVFP4 scales index whole groups, and the MXFP8 rows drop a
/// partial one), whose group count or `nvfp4_row_amax` loop passes the
/// templates' `int`s, whose group grid (NVFP4's `width / 16`) passes
/// grid.y's limit (`launchable`), and tunings whose CTAs miss a group or a
/// maximum.
pub fn quantize_entries(
    profile: Profile,
    width: u32,
    suffix: &str,
) -> Result<Vec<RowInstantiation>, Refusal> {
    require(
        width > 0 && width.is_multiple_of(32),
        "the row quantizers take whole 32-wide groups",
    )?;
    require(
        u64::from(width) + 127 <= i32::MAX as u64 && loop_fits(width, tuning().row_threads),
        "the row quantizers' group counts and loops are ints",
    )?;
    // Each group's CTA reduces the maxima of the lanes that load it, eight
    // values each.
    require(
        [128, 32, 16]
            .iter()
            .all(|group| tree_keeps(tuning().quant_threads, group / 8)),
        "the quantizer tuning's CTA does not reduce every loading lane",
    )?;
    require(
        mxfp8_rows_cover(width, tuning().row_threads),
        "the row tuning's CTA does not quantize every MXFP8 group",
    )?;
    require(
        tree_keeps(tuning().row_threads, width),
        "the row tuning's CTA does not reduce every NVFP4 row maximum",
    )?;
    let mut rows = Vec::new();
    for (format, label, query_format) in [
        (
            0,
            "fp8_exact",
            OutputQuantization::Fp8Group128(ScaleMode::ExactF32),
        ),
        (
            1,
            "fp8_pow2",
            OutputQuantization::Fp8Group128(ScaleMode::PowerOfTwoF32),
        ),
        (2, "mxfp8", OutputQuantization::Mxfp8),
        (3, "mxfp4", OutputQuantization::Mxfp4),
        (4, "nvfp4", OutputQuantization::Nvfp4),
    ] {
        let group = if format < 2 {
            128
        } else if format == 4 {
            16
        } else {
            32
        };
        add(
            &mut rows,
            profile,
            &format!("quant_{label}_{suffix}"),
            RowQuery::Quantize {
                width,
                format: query_format,
            },
            width,
            if format == 2 {
                (Grid::Rows, 256)
            } else {
                (Grid::RowsGroups(width.div_ceil(group)), 32)
            },
            vec![
                ptr("input", "const uint16_t"),
                ptr("codes", "uint8_t"),
                ptr("scales", "void"),
                ptr("global_scale", "const float"),
                ptr("live_rows", "const uint32_t"),
            ],
            if format == 2 {
                format!(
                    "chop::row::quantize_mxfp8_rows<{width}>(input,codes,reinterpret_cast<uint8_t*>(scales),live_rows);"
                )
            } else {
                format!(
                    "chop::row::quantize<{width},{format}>(input,codes,scales,global_scale,live_rows);"
                )
            },
        );
    }
    add(
        &mut rows,
        profile,
        &format!("nvfp4_row_amax_{suffix}"),
        RowQuery::Quantize {
            width,
            format: OutputQuantization::Nvfp4,
        },
        width,
        (Grid::Rows, 256),
        vec![
            ptr("input", "const uint16_t"),
            ptr("row_amax", "float"),
            ptr("live_rows", "const uint32_t"),
        ],
        format!("chop::row::nvfp4_row_amax<{width}>(input,row_amax,live_rows);"),
    );
    launchable(rows.into_iter().map(checked).collect::<Result<_, _>>()?)
}

/// Build the router entry from explicit geometry. Refuses an unknown kind,
/// a sigmoid router (the warp entry) of other than 256 experts, experts
/// whose softmax or grouped scores pass the static shared memory
/// (`router_shared_bytes`), and a grouped router whose selected groups hold
/// fewer than its eight routes.
pub fn router_entry(
    profile: Profile,
    name: &str,
    experts: u32,
    kind: u32,
    grouped: bool,
) -> Result<RowInstantiation, Refusal> {
    require(kind <= 2, "unsupported router kind")?;
    let n_group = if grouped { 8 } else { 1 };
    let topk_group = if grouped { 4 } else { 1 };
    if kind == 1 {
        require(
            experts == 256,
            "the sigmoid router warp ranks exactly 256 experts",
        )?;
    } else {
        require(
            router_shared_bytes(experts, kind, n_group) <= STATIC_SHARED_BYTES,
            "the router's scores exceed the static shared memory",
        )?;
    }
    // The grouped kind ranks only the experts of its `topk_group` best
    // groups, so they must hold the eight routes (the f64 oracle's
    // selected-group capacity).
    require(
        kind != 2 || experts / n_group * topk_group >= 8,
        "the grouped router's selected groups hold fewer experts than its top 8",
    )?;
    let mut rows = Vec::new();
    add(
        &mut rows,
        profile,
        &format!("router_{name}"),
        RowQuery::Router {
            experts,
            top_k: 8,
            kind: match kind {
                1 => RouterKind::SigmoidBiasSelectNormalized {
                    scaling_factor: 1.0,
                },
                2 => RouterKind::GroupLimitedSelectedSoftmax {
                    n_group,
                    topk_group,
                },
                _ => RouterKind::SoftmaxTopK { renormalize: true },
            },
        },
        experts,
        (Grid::Rows, if kind == 1 { 32 } else { 256 }),
        vec![
            ptr("logits", "const float"),
            ptr("bias", "const float"),
            ptr("ids", "int32_t"),
            ptr("weights", "float"),
            ptr("live_rows", "const uint32_t"),
            u32_arg("scale_bits"),
        ],
        if kind == 1 {
            format!(
                "chop::row::router_sigmoid_warp<{experts},8>(logits,bias,ids,weights,live_rows,__uint_as_float(scale_bits));"
            )
        } else {
            format!(
                "chop::row::router<{experts},8,{kind},true,{},{}>(logits,bias,ids,weights,live_rows,__uint_as_float(scale_bits));",
                n_group, topk_group
            )
        },
    );
    checked(rows.remove(0))
}

/// Build the expert permute entries from explicit geometry: the count,
/// scan and scatter of 8 routes per row into `block`-row expert blocks,
/// then the BF16 and MXFP8 gathers. Refuses sizes past the templates'
/// `int`s and padded route offsets, with the scatter's 32-slot tail step,
/// that pass them (`padded_routes_fit`), and gathers whose planned grid
/// passes grid.x's limit at a row capacity bucket (`launchable`).
pub fn expert_permute_entries(
    profile: Profile,
    name: &str,
    experts: u32,
    block: u32,
) -> Result<Vec<RowInstantiation>, Refusal> {
    require(
        template_ints(&[experts, block]),
        "expert permutation shapes are template ints",
    )?;
    require(
        padded_routes_fit(experts, 8, block, 32),
        "the expert permutation's padded route offsets are 32-bit",
    )?;
    let mut rows = Vec::new();
    let query = RowQuery::ExpertPermute {
        experts,
        block_size: block,
    };
    let common = [
        ptr("ids", "const int32_t"),
        ptr("counts", "int32_t"),
        ptr("live_rows", "const uint32_t"),
    ];
    add(
        &mut rows,
        profile,
        &format!("permute_count_{name}"),
        query,
        experts,
        (Grid::Experts(experts), 1024),
        common.to_vec(),
        format!("chop::row::permute_count<{experts},8,{block}>(ids,counts,live_rows);"),
    );
    add(
        &mut rows,
        profile,
        &format!("permute_scan_{name}"),
        query,
        experts,
        (Grid::One, 32),
        vec![
            ptr("counts", "const int32_t"),
            ptr("offsets", "int32_t"),
            ptr("padded_rows", "int32_t"),
        ],
        format!("chop::row::permute_scan<{experts},{block}>(counts,offsets,padded_rows);"),
    );
    add(
        &mut rows,
        profile,
        &format!("permute_scatter_{name}"),
        query,
        experts,
        (Grid::Experts(experts), 32),
        vec![
            ptr("ids", "const int32_t"),
            ptr("offsets", "const int32_t"),
            ptr("sorted_rows", "int32_t"),
            ptr("live_rows", "const uint32_t"),
        ],
        format!(
            "chop::row::permute_scatter<{experts},8,{block}>(ids,offsets,sorted_rows,live_rows);"
        ),
    );
    add(
        &mut rows,
        profile,
        &format!("gather_{name}"),
        query,
        4096,
        (
            Grid::PaddedRows {
                experts,
                top_k: 8,
                block,
            },
            256,
        ),
        vec![
            ptr("input", "const uint16_t"),
            ptr("sorted_rows", "const int32_t"),
            ptr("output", "uint16_t"),
            ptr("padded_rows", "const int32_t"),
        ],
        "chop::row::gather<4096,8>(input,sorted_rows,output,padded_rows);".into(),
    );
    add(
        &mut rows,
        profile,
        &format!("gather_mxfp8_{name}"),
        query,
        4096,
        (
            Grid::PaddedRows {
                experts,
                top_k: 8,
                block,
            },
            256,
        ),
        vec![
            ptr("input", "const uint16_t"),
            ptr("sorted_rows", "const int32_t"),
            ptr("codes", "uint8_t"),
            ptr("scales", "uint8_t"),
            ptr("padded_rows", "const int32_t"),
        ],
        "chop::row::gather_mxfp8<4096,8>(input,sorted_rows,codes,scales,padded_rows);".into(),
    );
    launchable(rows.into_iter().map(checked).collect::<Result<_, _>>()?)
}

/// Build the expert combine entry from explicit geometry. Refuses a width
/// that is not whole 16-byte vectors or whose vector loop passes an `int`.
pub fn expert_combine_entry(
    profile: Profile,
    name: &str,
    width: u32,
    experts: u32,
    preweighted: bool,
) -> Result<RowInstantiation, Refusal> {
    // The query carries no width, so the contract never sees it.
    require(
        width > 0 && width.is_multiple_of(8),
        "the expert combine adds whole 16-byte vectors",
    )?;
    require(
        vector_loop_fits(width, tuning().row_threads),
        "the expert combine's vector loop steps within an int",
    )?;
    let mut rows = Vec::new();
    add(
        &mut rows,
        profile,
        &format!("combine_{name}"),
        RowQuery::ExpertCombine { experts, top_k: 8 },
        width,
        (Grid::Rows, 256),
        vec![
            ptr("routed", "const uint16_t"),
            ptr("weights", "const float"),
            ptr("output", "uint16_t"),
            ptr("live_rows", "const uint32_t"),
        ],
        format!(
            "chop::row::combine<{width},8,{}>(routed,weights,output,live_rows);",
            if preweighted { "true" } else { "false" }
        ),
    );
    checked(rows.remove(0))
}

/// Build the swiglu entries from explicit geometry. The template indexes a
/// row's gate and up values as `int`s up to `2 width - 1`, so it refuses
/// widths past 2^30.
pub fn swiglu_entries(profile: Profile, width: u32) -> Result<Vec<RowInstantiation>, Refusal> {
    require(
        width <= 1 << 30,
        "a SwiGLU row's gate and up indices are ints",
    )?;
    let mut rows = Vec::new();
    for (name, columns, interleaved) in [
        ("halves", SwigluColumns::ConcatenatedHalves, false),
        ("interleaved", SwigluColumns::Interleaved, true),
    ] {
        add(
            &mut rows,
            profile,
            &format!("swiglu_{name}_{width}"),
            RowQuery::Swiglu { width, columns },
            width,
            (Grid::Rows, 256),
            vec![
                ptr("input", "const uint16_t"),
                ptr("output", "uint16_t"),
                ptr("live_rows", "const uint32_t"),
            ],
            format!(
                "chop::row::swiglu<{width},{}>(input,output,live_rows);",
                if interleaved { "true" } else { "false" }
            ),
        );
    }
    rows.into_iter().map(checked).collect()
}

/// Build the global NVFP4 scale reduction used by the row quantizer. Every
/// lane holds a row maximum once the live rows reach the CTA, so it
/// refuses a row tuning whose reduction drops a lane.
pub fn nvfp4_global_scale(profile: Profile) -> Result<RowInstantiation, Refusal> {
    let largest = tuning().buckets.iter().copied().max().unwrap_or(0);
    require(
        tree_keeps(tuning().row_threads, largest),
        "the row tuning's CTA does not reduce every row maximum",
    )?;
    let mut rows = Vec::new();
    add(
        &mut rows,
        profile,
        "nvfp4_global_scale",
        RowQuery::Quantize {
            width: 4096,
            format: OutputQuantization::Nvfp4,
        },
        4096,
        (Grid::One, 256),
        vec![
            ptr("row_amax", "const float"),
            ptr("scale", "float"),
            ptr("live_rows", "const uint32_t"),
        ],
        "chop::row::nvfp4_global_scale(row_amax,scale,live_rows);".into(),
    );
    checked(rows.remove(0))
}

/// The arguments every `rope_kv` entry passes its template.
pub(crate) const ROPE_KV_ARGUMENTS: &str = "(qkv,q,k_new,v_new,k_pool,v_pool,block_tables,slot_ids,token_batch,positions,cos_sin,live_tokens,max_pages,ring_size,k_page_stride_bytes,v_page_stride_bytes,__uint_as_float(theta_bits),__uint_as_float(v_multiplier_bits),__uint_as_float(k_scale_bits),__uint_as_float(v_scale_bits));";
/// The arguments every `ring_write` entry passes its template.
pub(crate) const RING_WRITE_ARGUMENTS: &str = "(k_new,v_new,k_ring,v_ring,slot_ids,token_batch,positions,last_positions,live_tokens,ring_size,__uint_as_float(k_scale_bits),__uint_as_float(v_scale_bits));";
/// The arguments of a `dflash_rope` entry, then those of a table-backed one.
pub(crate) const ROPE_ARGUMENTS: [&str; 2] = [
    "(input,output,positions,live_tokens,input_stride,output_stride);",
    "(input,output,positions,live_tokens,input_stride,output_stride,cos_sin);",
];

/// The head width `dflash_qk_norm` and `dflash_rope` compile in, one
/// thread per element.
pub(crate) const HEAD_DIM: u32 = 128;
/// The rotated width and the computed theta `dflash_rope` compiles in.
pub(crate) const ROPE_ROTARY: u32 = 64;
pub(crate) const ROPE_THETA: f32 = 1e4;

/// The Q heads one `rope_kv` block rotates from 512 rows on, 64 lanes each,
/// so its blocks hold exactly `ROPE_KV_THREADS` threads.
pub(crate) const ROPE_KV_HEAD_GROUP: u32 = 4;
pub(crate) const ROPE_KV_THREADS: u32 = 256;

/// The routes one tile of the tiled permutation counts, and its threads:
/// the count, prefix and scan kernels give one thread to each of 256
/// experts or route tiles.
const PERMUTE_TILE: u32 = 256;

/// The static shared memory a CTA may declare in `__shared__` arrays.
const STATIC_SHARED_BYTES: u64 = 48 * 1024;

/// The static shared memory `router<experts, 8, kind, _, groups, _>` holds
/// for a softmax (0) or grouped (2) kind: its scores and selected values,
/// and the grouped kind's group scores. Its `unbiased` array is read only
/// by the sigmoid kind, which takes the warp entry, and nvcc drops it.
fn router_shared_bytes(experts: u32, kind: u32, groups: u32) -> u64 {
    4 * u64::from(experts) + 4 * 8 + if kind == 2 { 4 * u64::from(groups) } else { 0 }
}

/// Whether every value fits the `int` template parameter it becomes.
fn template_ints(values: &[u32]) -> bool {
    values.iter().all(|&value| i32::try_from(value).is_ok())
}

/// The static shared memory of `rmsnorm_quant<width, 3, _, format>`: the
/// HF norm's ordered squares (`float[width]`) and its eight warp sums,
/// then for the formats past FP8 the quantizer's 256 group maxima.
fn rmsnorm_quant_shared_bytes(width: u32, format: u32) -> u64 {
    4 * u64::from(width) + 4 * 8 + if format >= 2 { 4 * 256 } else { 0 }
}

/// Whether the CTA tree reduction (`a[t] = op(a[t], a[t + step])` for
/// `t < step`, halving `step` from `threads / 2` to one) leaves the values
/// of lanes `0..lanes`, the lanes that contribute one, in `a[0]`. A
/// power-of-two CTA keeps every lane; others drop some lanes, which only
/// matters where those lanes contribute.
fn tree_keeps(threads: u32, lanes: u32) -> bool {
    let threads = threads as usize;
    // The lanes each slot holds, as a bit set over at most 256 lanes.
    let mut held: Vec<[u64; 4]> = (0..threads)
        .map(|lane| {
            let mut set = [0; 4];
            set[lane / 64] = 1 << (lane % 64);
            set
        })
        .collect();
    let mut step = threads / 2;
    while step > 0 {
        for lane in 0..step {
            let other = held[lane + step];
            for (word, more) in held[lane].iter_mut().zip(other) {
                *word |= more;
            }
        }
        step /= 2;
    }
    (0..(lanes as usize).min(threads)).all(|lane| held[0][lane / 64] >> (lane % 64) & 1 == 1)
}

/// Whether a CTA of `threads` quantizes every group of a `width` row in
/// `format`: the FP8 formats give each 128-wide group a half warp and step
/// 16 groups (eight warps) at a time, so fewer warps cover only rows of at
/// most one group per half warp; the other formats load each group's
/// values eight to a lane (four lanes for MXFP8 and MXFP4's 32-wide
/// groups, two for NVFP4's 16) and reduce their maxima over the CTA, which
/// must keep those lanes (`tree_keeps`).
fn quantizer_covers(format: u32, width: u32, threads: u32) -> bool {
    match format {
        0 | 1 => threads == 256 || width / 128 <= threads / 16,
        4 => tree_keeps(threads, 16 / 8),
        _ => tree_keeps(threads, 32 / 8),
    }
}

/// Whether a `for (int d = lane; d < dim; d += step)` loop's last increment
/// stays an `int`.
fn loop_fits(dim: u32, step: u32) -> bool {
    u64::from(dim) + u64::from(step) - 1 <= i32::MAX as u64
}

/// Whether a `for (int c = lane * 8; c < width; c += step * 8)` vector
/// loop over whole vectors keeps its last increment, from `width - 8`, an
/// `int`.
fn vector_loop_fits(width: u32, step: u32) -> bool {
    u64::from(width) + 8 * u64::from(step) - 8 <= i32::MAX as u64
}

/// Whether `quantize_mxfp8_rows` covers a `width` row with a CTA of
/// `threads`: each warp's eight 32-wide groups step 64 groups (eight
/// warps) at a time, so fewer warps cover only rows of at most eight
/// groups per warp.
fn mxfp8_rows_cover(width: u32, threads: u32) -> bool {
    threads >= 256 || width / 32 <= threads / 32 * 8
}

/// Whether `experts` experts' routes, `top_k` per row of the largest row
/// bucket, padded to whole `block`-row blocks keep the permutation's
/// `int`s: an expert's `count + Block - 1`, and the padded total plus a
/// loop's last `step - 1` slots past it. The total is at most the most
/// blocks the routes can open (one route each in up to `experts` experts,
/// every further block a whole one).
fn padded_routes_fit(experts: u32, top_k: u32, block: u32, step: u32) -> bool {
    let largest = tuning().buckets.iter().copied().max().unwrap_or(0);
    let routes = u64::from(largest) * u64::from(top_k);
    let (experts, block) = (u64::from(experts), u64::from(block));
    if block == 0 {
        return false;
    }
    let blocks = if routes <= experts {
        routes
    } else {
        experts + (routes - experts) / block
    };
    routes + block - 1 <= i32::MAX as u64 && blocks * block + u64::from(step) - 1 <= i32::MAX as u64
}

/// `rows`, once each plans at every row capacity bucket: its grid within
/// CUDA's launch limits, as `RowInstantiation::plan` computes it in checked
/// arithmetic (the gathers' padded-row grid `capacity * top_k + experts *
/// (block - 1)` among them, which can pass grid.x's limit where the padded
/// routes themselves fit, and a quantizer's group count, grid.y).
fn launchable(rows: Vec<RowInstantiation>) -> Result<Vec<RowInstantiation>, Refusal> {
    for row in &rows {
        for &capacity in &tuning().buckets {
            row.plan(
                capacity,
                RuntimeCounts {
                    m: capacity,
                    batch: 0,
                    context: 0,
                    routed_rows: 0,
                    split_k: 1,
                },
            )?;
        }
    }
    Ok(rows)
}

/// Refuses with `what` unless `holds`.
fn require(holds: bool, what: &'static str) -> Result<(), Refusal> {
    if holds { Ok(()) } else { Err(Refusal(what)) }
}

/// `entry`, once its query validates (`RowInstantiation::contract`).
fn checked(entry: RowInstantiation) -> Result<RowInstantiation, Refusal> {
    entry.contract()?;
    Ok(entry)
}

/// Build the fused RMSNorm and quantization entry (`rmsnorm_quant`) of an
/// HF BF16-intermediate RMSNorm row of `width`, over the residual sum
/// where `fused_residual`. Refuses a width that is not whole scale groups
/// of the output (128 for FP8, 32 for MXFP8 and MXFP4, 16 for NVFP4), whose
/// tail the template would not quantize; a width whose ordered squares do
/// not fit the static shared memory; a row tuning whose CTA does not
/// cover every group (`quantizer_covers`); and an eps below 2^-101, where
/// its HF mean of squares rounds otherwise than the contract's
/// (`rmsnorm::HF_MIN_EPS_BITS`).
pub fn rmsnorm_quant_entry(
    profile: Profile,
    name: &str,
    width: u32,
    eps: f32,
    fused_residual: bool,
    output: OutputQuantization,
) -> Result<RowInstantiation, Refusal> {
    let (format, group) = match output {
        OutputQuantization::Fp8Group128(ScaleMode::ExactF32) => (0, 128),
        OutputQuantization::Fp8Group128(ScaleMode::PowerOfTwoF32) => (1, 128),
        OutputQuantization::Mxfp8 => (2, 32),
        OutputQuantization::Mxfp4 => (3, 32),
        OutputQuantization::Nvfp4 => (4, 16),
    };
    require(
        width > 0 && width.is_multiple_of(group),
        "rmsnorm_quant quantizes whole scale groups of the row",
    )?;
    require(
        rmsnorm_quant_shared_bytes(width, format) <= STATIC_SHARED_BYTES,
        "rmsnorm_quant's ordered squares exceed the static shared memory",
    )?;
    require(
        quantizer_covers(format, width, tuning().row_threads),
        "the row tuning's CTA does not quantize every group",
    )?;
    let entry = checked(RowInstantiation::new(
        profile,
        name,
        RowQuery::RmsNorm {
            width,
            eps,
            kind: RmsNormKind::HfBf16Intermediate,
            fused_residual,
            output_quantization: Some(output),
        },
        width,
        (Grid::Rows, 256),
        vec![
            ptr("x", "const uint16_t"),
            ptr("weight", "const uint16_t"),
            ptr("residual", "const uint16_t"),
            ptr("residual_out", "uint16_t"),
            ptr("y", "uint16_t"),
            ptr("codes", "uint8_t"),
            ptr("scales", "void"),
            ptr("global_scale", "const float"),
            ptr("live_rows", "const uint32_t"),
            u32_arg("eps_bits"),
        ],
        format!(
            "chop::row::rmsnorm_quant<{width},3,{fused_residual},{format}>(x,weight,residual,residual_out,y,codes,scales,global_scale,live_rows,__uint_as_float(eps_bits));"
        ),
    ))?;
    // The contract has refused any eps that is not positive and finite.
    require(
        eps.to_bits() >= rmsnorm::HF_MIN_EPS_BITS,
        "rmsnorm_quant's HF mean of squares serves epsilons from 2^-101",
    )?;
    Ok(entry)
}

/// Build the tiled expert permutation (`permute_tile_count`, `_prefix`,
/// `_scan` and `_scatter`, in launch order) of `experts` experts and
/// `top_k` routes per row into `block`-row expert blocks, counting 256
/// routes per tile. The kernels give one of 256 threads to each expert and
/// to each route tile, so they refuse other expert counts, a row tuning of
/// other than 256 threads, and more than 256 route tiles at the largest
/// row bucket.
pub fn permute_tile_entries(
    profile: Profile,
    name: &str,
    experts: u32,
    top_k: u32,
    block: u32,
) -> Result<Vec<RowInstantiation>, Refusal> {
    require(
        experts == PERMUTE_TILE,
        "the tiled permutation counts and scans exactly 256 experts",
    )?;
    require(
        tuning().row_threads == PERMUTE_TILE,
        "the tiled permutation runs 256 threads",
    )?;
    require(
        top_k > 0 && block > 0,
        "the tiled permutation needs routes and blocks",
    )?;
    let largest = tuning().buckets.iter().copied().max().unwrap_or(0);
    let routes = u64::from(largest) * u64::from(top_k);
    require(
        routes <= u64::from(PERMUTE_TILE * PERMUTE_TILE),
        "the tiled permutation's prefix scans at most 256 route tiles",
    )?;
    // The scan pads each expert's routes to whole blocks in `int`s, one
    // slot at a time.
    require(
        padded_routes_fit(experts, top_k, block, 1),
        "the tiled permutation's padded route offsets are 32-bit",
    )?;
    let tiles = Grid::RouteTiles {
        top_k,
        tile: PERMUTE_TILE,
    };
    [
        (
            "count",
            tiles,
            vec![
                ptr("ids", "const int32_t"),
                ptr("tile_counts", "int32_t"),
                ptr("live_rows", "const uint32_t"),
            ],
            format!(
                "chop::row::permute_tile_count<{experts},{top_k},256>(ids,tile_counts,live_rows);"
            ),
        ),
        (
            "prefix",
            Grid::Experts(experts),
            vec![
                ptr("tile_counts", "const int32_t"),
                ptr("counts", "int32_t"),
                ptr("tile_offsets", "int32_t"),
                ptr("live_rows", "const uint32_t"),
            ],
            format!(
                "chop::row::permute_tile_prefix<{experts},{top_k},256>(tile_counts,counts,tile_offsets,live_rows);"
            ),
        ),
        (
            "scan",
            Grid::One,
            vec![
                ptr("counts", "const int32_t"),
                ptr("offsets", "int32_t"),
                ptr("sorted_rows", "int32_t"),
                ptr("padded_rows", "int32_t"),
            ],
            format!(
                "chop::row::permute_tile_scan<{experts},{block}>(counts,offsets,sorted_rows,padded_rows);"
            ),
        ),
        (
            "scatter",
            tiles,
            vec![
                ptr("ids", "const int32_t"),
                ptr("tile_offsets", "const int32_t"),
                ptr("offsets", "const int32_t"),
                ptr("sorted_rows", "int32_t"),
                ptr("live_rows", "const uint32_t"),
            ],
            format!(
                "chop::row::permute_tile_scatter<{experts},{top_k},256>(ids,tile_offsets,offsets,sorted_rows,live_rows);"
            ),
        ),
    ]
    .into_iter()
    .map(|(operation, grid, params, body)| {
        checked(RowInstantiation::new(
            profile,
            &format!("permute_tile_{operation}_{name}"),
            RowQuery::ExpertPermute {
                experts,
                block_size: block,
            },
            experts,
            (grid, 256),
            params,
            body,
        ))
    })
    .collect::<Result<_, _>>()
    .and_then(launchable)
}

/// Build the per-head Q or K RMSNorm (`dflash_qk_norm`) of `heads` heads
/// of `head_dim`, one block per row and head. The template compiles in
/// 128-wide heads, so it refuses other widths, and the heads are grid.y,
/// so it refuses more than a launch holds (`launchable`).
pub fn qk_norm_entry(
    profile: Profile,
    name: &str,
    heads: u32,
    head_dim: u32,
    eps: f32,
) -> Result<RowInstantiation, Refusal> {
    require(
        heads > 0 && template_ints(&[heads]),
        "a QK norm needs heads its template counts",
    )?;
    require(
        head_dim == HEAD_DIM,
        "dflash_qk_norm normalizes 128-wide heads",
    )?;
    let entry = checked(RowInstantiation::new(
        profile,
        name,
        RowQuery::RmsNorm {
            width: head_dim,
            eps,
            kind: RmsNormKind::QkPerHead { head_dim },
            fused_residual: false,
            output_quantization: None,
        },
        head_dim,
        (Grid::RowsHeads(heads), head_dim),
        vec![
            ptr("input", "const uint16_t"),
            ptr("weight", "const uint16_t"),
            ptr("output", "uint16_t"),
            ptr("live_tokens", "const uint32_t"),
            u32_arg("input_stride"),
            u32_arg("output_stride"),
            u32_arg("eps_bits"),
        ],
        format!(
            "chop::row::dflash_qk_norm<{heads}>(input,weight,output,live_tokens,input_stride,output_stride,__uint_as_float(eps_bits));"
        ),
    ))?;
    Ok(launchable(vec![entry])?.remove(0))
}

/// Build the NeoX RoPE alone (`dflash_rope`) of `heads` heads of
/// `head_dim`, rotating `rotary_dim` dimensions: for 32-bit then 64-bit
/// positions, the computed entry (`{name}_rope_pos{bits}`) and then the
/// cos/sin table one (`{name}_rope_table_pos{bits}`). The template compiles
/// in 128-wide heads, 64 rotated dimensions and, for the computed entries,
/// theta 1e4, so it refuses other geometry; it also refuses a name starting
/// `rope_`, whose rows the planner would grid as RoPE KV writes, and more
/// heads than grid.y holds (`launchable`).
pub fn rope_entries(
    profile: Profile,
    name: &str,
    heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    theta: f32,
) -> Result<Vec<RowInstantiation>, Refusal> {
    require(
        heads > 0 && template_ints(&[heads]),
        "a RoPE needs heads its template counts",
    )?;
    require(
        head_dim == HEAD_DIM && rotary_dim == ROPE_ROTARY,
        "dflash_rope rotates 64 of 128 dimensions",
    )?;
    require(theta == ROPE_THETA, "dflash_rope computes at theta 1e4")?;
    require(
        !name.starts_with("rope_"),
        "a RoPE alone named rope_ would take the RoPE KV grid",
    )?;
    let mut rows = Vec::new();
    for pos64 in [false, true] {
        let bits = if pos64 { 64 } else { 32 };
        for table in [false, true] {
            let mut params = vec![
                ptr("input", "const uint16_t"),
                ptr("output", "uint16_t"),
                ptr("positions", "const void"),
                ptr("live_tokens", "const uint32_t"),
                u32_arg("input_stride"),
                u32_arg("output_stride"),
            ];
            if table {
                params.push(ptr("cos_sin", "const float"));
            }
            rows.push(checked(RowInstantiation::new(
                profile,
                &format!("{name}_rope{}_pos{bits}", if table { "_table" } else { "" }),
                RowQuery::Rope {
                    head_dim,
                    rotary_dim,
                    theta,
                    style: RotaryStyle::Neox,
                    source: if table {
                        RotarySource::CosSinTable
                    } else {
                        RotarySource::Computed
                    },
                    fused_paged_kv_write: false,
                },
                head_dim,
                (Grid::RowsHeads(heads), head_dim),
                params,
                if table {
                    format!(
                        "chop::row::dflash_rope<{heads},{pos64},true>{}",
                        ROPE_ARGUMENTS[1]
                    )
                } else {
                    format!(
                        "chop::row::dflash_rope<{heads},{pos64}>{}",
                        ROPE_ARGUMENTS[0]
                    )
                },
            ))?);
        }
    }
    launchable(rows)
}

/// Build the two stages of a vocabulary-sharded argmax: each rank's
/// partial over its `vocab_per_rank` logits (`{name}_argmax_partial`),
/// then the final over the `ranks` partials (`{name}_argmax_final`).
/// Refuses an empty shard or rank set, sizes past the templates' `int`
/// parameters, a vocabulary whose last token ID does not fit their 32-bit
/// IDs, and a row tuning whose CTA reduction drops a column of the shard.
pub fn sharded_argmax_entries(
    profile: Profile,
    name: &str,
    vocab_per_rank: u32,
    ranks: u32,
) -> Result<[RowInstantiation; 2], Refusal> {
    require(
        vocab_per_rank > 0 && ranks > 0,
        "a sharded argmax needs a vocabulary shard on each rank",
    )?;
    require(
        template_ints(&[vocab_per_rank, ranks]),
        "sharded argmax shapes are template ints",
    )?;
    require(
        u64::from(vocab_per_rank) * u64::from(ranks) - 1 <= i32::MAX as u64,
        "sharded argmax token IDs are 32-bit",
    )?;
    // Every lane under the shard's width holds a column's best.
    require(
        tree_keeps(tuning().row_threads, vocab_per_rank),
        "the row tuning's CTA does not reduce every column of the shard",
    )?;
    Ok([
        checked(RowInstantiation::new(
            profile,
            &format!("{name}_argmax_partial"),
            RowQuery::ShardedArgmaxPartial {
                vocab_per_rank,
                ranks,
                logits: DType::Bf16,
            },
            vocab_per_rank,
            (Grid::Rows, 256),
            vec![
                ptr("logits", "const uint16_t"),
                ptr("scores_out", "float"),
                ptr("ids_out", "int32_t"),
                ptr("live_rows", "const uint32_t"),
                ptr("live_width", "const uint32_t"),
                u32_arg("vocab_offset"),
            ],
            format!(
                "chop::row::dflash_argmax_partial<{vocab_per_rank}>(logits,scores_out,ids_out,live_rows,live_width,vocab_offset);"
            ),
        ))?,
        checked(RowInstantiation::new(
            profile,
            &format!("{name}_argmax_final"),
            RowQuery::ShardedArgmaxFinal {
                ranks,
                pairs: ArgmaxLayout::RowMajor,
            },
            ranks,
            (Grid::Rows, 128),
            vec![
                ptr("scores", "const float"),
                ptr("ids", "const int32_t"),
                ptr("output", "int32_t"),
                ptr("live_rows", "const uint32_t"),
            ],
            format!("chop::row::dflash_argmax_final<{ranks}>(scores,ids,output,live_rows);"),
        ))?,
    ])
}

/// The attention heads and cache a RoPE KV write rotates and stores.
#[derive(Clone, Copy, Debug)]
pub struct RopeKvShape {
    pub q_heads: u32,
    pub kv_heads: u32,
    pub qk_dim: u32,
    pub v_dim: u32,
    pub rotary_dim: u32,
    pub theta: f32,
    pub style: RotaryStyle,
    /// The keys and values go to a sliding-window ring rather than paged
    /// pools.
    pub ring: bool,
    /// A ring write that leaves the ring to the prefill's ring write.
    pub prefill: bool,
    /// FP8 E4M3 caches.
    pub fp8: bool,
}

/// Build the RoPE KV writes (`rope_kv`) of `shape`: for each page size
/// (32 for a ring, else 16, 32 and 64), the computed then the cos/sin
/// table rotation, each for 32-bit then 64-bit positions
/// (`rope_{name}_{computed|table}_p{page}_pos{bits}`). The template rotates
/// Q heads in groups of four with 64 lanes each and stores the K and V
/// heads the first Q heads hold, so it refuses Q heads that are not a
/// positive multiple of four, more KV heads than Q heads, a row tuning of
/// other than 256 threads, a rotated width that is odd, zero or wider than
/// the head, a prefill write to paged pools (paged writes always store),
/// sizes or a packed QKV row width past the template's `int`s, and Q heads
/// whose grid passes CUDA's limits at a row capacity bucket (`launchable`:
/// below 512 rows the planner grids every Q head, not every group of four).
pub fn rope_kv_entries(
    profile: Profile,
    name: &str,
    shape: RopeKvShape,
) -> Result<Vec<RowInstantiation>, Refusal> {
    let RopeKvShape {
        q_heads,
        kv_heads,
        qk_dim,
        v_dim,
        rotary_dim,
        theta,
        style,
        ring,
        prefill,
        fp8,
    } = shape;
    require(
        q_heads > 0 && q_heads.is_multiple_of(ROPE_KV_HEAD_GROUP),
        "rope_kv rotates Q heads in groups of four",
    )?;
    require(
        kv_heads > 0 && kv_heads <= q_heads,
        "rope_kv writes at most as many KV heads as Q heads",
    )?;
    require(
        tuning().row_threads == ROPE_KV_THREADS,
        "rope_kv groups run 256 threads",
    )?;
    require(
        rotary_dim > 0 && rotary_dim.is_multiple_of(2) && rotary_dim <= qk_dim && v_dim > 0,
        "rope_kv rotates an even width within the head",
    )?;
    require(ring || !prefill, "a paged RoPE KV write always stores")?;
    require(
        template_ints(&[q_heads, kv_heads, qk_dim, v_dim, rotary_dim]),
        "rope_kv shapes are template ints",
    )?;
    // Fields within `int` keep these products within `u64`.
    let packed = u64::from(q_heads) * u64::from(qk_dim)
        + u64::from(kv_heads) * (u64::from(qk_dim) + u64::from(v_dim));
    require(
        packed <= i32::MAX as u64,
        "rope_kv's packed QKV row is a template int",
    )?;
    // The packed row does not bound the head loops' last step: a narrow QK
    // head leaves a V width whose final `d += blockDim.x` passes INT32_MAX.
    require(
        loop_fits(qk_dim, ROPE_KV_THREADS) && loop_fits(v_dim, ROPE_KV_THREADS),
        "rope_kv's head loops step within an int",
    )?;
    let packed = packed as u32;
    let neox = style == RotaryStyle::Neox;
    let page_sizes: &[u32] = if ring { &[32] } else { &[16, 32, 64] };
    let mut rows = Vec::new();
    for &page_size in page_sizes {
        for table in [false, true] {
            for pos64 in [false, true] {
                rows.push(checked(RowInstantiation::new(
                    profile,
                    &format!(
                        "rope_{name}_{}_p{page_size}_pos{}",
                        if table { "table" } else { "computed" },
                        if pos64 { 64 } else { 32 }
                    ),
                    RowQuery::Rope {
                        head_dim: qk_dim,
                        rotary_dim,
                        theta,
                        style,
                        source: if table {
                            RotarySource::CosSinTable
                        } else {
                            RotarySource::Computed
                        },
                        fused_paged_kv_write: !ring,
                    },
                    packed,
                    (Grid::RowsHeads(q_heads / ROPE_KV_HEAD_GROUP), 256),
                    vec![
                        ptr("qkv", "const uint16_t"),
                        ptr("q", "uint16_t"),
                        ptr("k_new", "uint16_t"),
                        ptr("v_new", "uint16_t"),
                        ptr("k_pool", "void"),
                        ptr("v_pool", "void"),
                        ptr("block_tables", "const int32_t"),
                        ptr("slot_ids", "const int32_t"),
                        ptr("token_batch", "const int32_t"),
                        ptr("positions", "const void"),
                        ptr("cos_sin", "const float"),
                        ptr("live_tokens", "const uint32_t"),
                        u32_arg("max_pages"),
                        u32_arg("ring_size"),
                        u64_arg("k_page_stride_bytes"),
                        u64_arg("v_page_stride_bytes"),
                        u32_arg("theta_bits"),
                        u32_arg("v_multiplier_bits"),
                        u32_arg("k_scale_bits"),
                        u32_arg("v_scale_bits"),
                    ],
                    format!(
                        "chop::row::rope_kv<{q_heads},{kv_heads},{qk_dim},{v_dim},{rotary_dim},{neox},{table},{ring},{prefill},{fp8},{pos64},{page_size}>{ROPE_KV_ARGUMENTS}"
                    ),
                ))?);
            }
        }
    }
    launchable(rows)
}

/// Build the sliding-window ring writes (`ring_write`) of `kv_heads` heads
/// into a ring of `ring_size` slots: BF16 then FP8 E4M3, each for 32-bit
/// then 64-bit positions (`ring_write_{name}_{bf16|fp8}_pos{bits}`), then
/// the BF16 single-sequence tail with 32-bit positions, which writes only
/// the ring's last window of one sequence
/// (`ring_write_{name}_bf16_single_sequence_pos32`). Refuses empty heads,
/// an empty ring, head sizes past the template's `int`s or whose loops step
/// past one, and more KV heads than grid.y holds (`launchable`).
pub fn ring_write_entries(
    profile: Profile,
    name: &str,
    kv_heads: u32,
    qk_dim: u32,
    v_dim: u32,
    ring_size: u32,
) -> Result<Vec<RowInstantiation>, Refusal> {
    require(
        kv_heads > 0 && qk_dim > 0 && v_dim > 0 && ring_size > 0,
        "a ring write needs heads and ring slots",
    )?;
    require(
        template_ints(&[kv_heads, qk_dim, v_dim]),
        "ring write heads are template ints",
    )?;
    let threads = tuning().row_threads;
    require(
        loop_fits(qk_dim, threads) && loop_fits(v_dim, threads),
        "a ring write's head loops step past an int",
    )?;
    let entry = |label: &str, fp8: bool, pos64: bool, tail: bool| {
        checked(RowInstantiation::new(
            profile,
            &format!("ring_write_{name}_{label}"),
            RowQuery::RingWrite {
                qk_dim,
                v_dim,
                kv_heads,
                ring_size,
                kv_dtype: if fp8 { DType::Fp8E4M3 } else { DType::Bf16 },
            },
            qk_dim,
            (
                if tail {
                    Grid::RingTail {
                        ring: ring_size,
                        heads: kv_heads,
                    }
                } else {
                    Grid::RowsHeads(kv_heads)
                },
                256,
            ),
            vec![
                ptr("k_new", "const uint16_t"),
                ptr("v_new", "const uint16_t"),
                ptr("k_ring", "void"),
                ptr("v_ring", "void"),
                ptr("slot_ids", "const int32_t"),
                ptr("token_batch", "const int32_t"),
                ptr("positions", "const void"),
                ptr("last_positions", "const int64_t"),
                ptr("live_tokens", "const uint32_t"),
                u32_arg("ring_size"),
                u32_arg("k_scale_bits"),
                u32_arg("v_scale_bits"),
            ],
            if tail {
                format!(
                    "chop::row::ring_write<{kv_heads},{qk_dim},{v_dim},{fp8},{pos64},true>{RING_WRITE_ARGUMENTS}"
                )
            } else {
                format!(
                    "chop::row::ring_write<{kv_heads},{qk_dim},{v_dim},{fp8},{pos64}>{RING_WRITE_ARGUMENTS}"
                )
            },
        ))
    };
    let mut rows = Vec::new();
    for (dtype, fp8) in [("bf16", false), ("fp8", true)] {
        for pos64 in [false, true] {
            let bits = if pos64 { 64 } else { 32 };
            rows.push(entry(&format!("{dtype}_pos{bits}"), fp8, pos64, false)?);
        }
    }
    rows.push(entry("bf16_single_sequence_pos32", false, false, true)?);
    launchable(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_group_counts_match_the_generated_selection() {
        for (grouped, n_group, topk_group) in [(false, 1, 1), (true, 8, 4)] {
            let entry = router_entry(Profile::StdDenseMoe, "groups", 128, 2, grouped).unwrap();
            assert_eq!(
                entry.query,
                RowQuery::Router {
                    experts: 128,
                    top_k: 8,
                    kind: RouterKind::GroupLimitedSelectedSoftmax {
                        n_group,
                        topk_group
                    },
                }
            );
            assert!(
                entry
                    .body
                    .contains(&format!("<128,8,2,true,{n_group},{topk_group}>"))
            );
            entry.contract().unwrap();
        }
    }

    #[test]
    fn router_refuses_an_unknown_kind() {
        assert!(router_entry(Profile::StdDenseMoe, "unknown", 128, 3, false).is_err());
    }

    /// The router's scores fit the static shared memory up to the widths
    /// nvcc compiles: 12,280 softmax experts (12,281 need 49,156 bytes) and
    /// 12,272 grouped ones (12,280 need 0xc020); the sigmoid warp entry
    /// ranks 256 experts alone, and a grouped router's selected groups hold
    /// its eight routes.
    #[test]
    fn router_refuses_experts_its_templates_do_not_build() {
        let router = |experts, kind, grouped| router_entry(PROFILE, "test", experts, kind, grouped);
        assert!(router(12280, 0, false).is_ok());
        assert!(router(12281, 0, false).is_err());
        assert!(router(12288, 0, false).is_err());
        assert!(router(16384, 0, false).is_err());
        assert!(router(12272, 2, true).is_ok());
        assert!(router(12280, 2, true).is_err());
        assert!(router(256, 1, false).is_ok());
        // Four of eight groups of one expert each leave four candidates
        // for eight routes (the device repeats expert 0); groups of two
        // hold exactly eight.
        assert!(router(8, 2, true).is_err());
        assert!(router(16, 2, true).is_ok());
        assert!(router(8, 2, false).is_ok());
        for experts in [128, 384] {
            assert!(router(experts, 1, false).is_err(), "{experts}");
        }
    }

    const PROFILE: Profile = Profile::StdDenseMoe;

    /// The per-head norm and the RoPE alone build the geometry their
    /// templates compile in and refuse any other head width, rotated
    /// width or computed theta.
    #[test]
    fn head_rows_refuse_geometry_their_templates_do_not_compile() {
        assert!(qk_norm_entry(PROFILE, "test_q_norm", 2, 128, 1e-6).is_ok());
        assert!(qk_norm_entry(PROFILE, "test_q_norm", 2, 64, 1e-6).is_err());
        assert!(qk_norm_entry(PROFILE, "test_q_norm", 0, 128, 1e-6).is_err());
        let rope = rope_entries(PROFILE, "test_q", 2, 128, 64, 1e4).unwrap();
        assert_eq!(rope.len(), 4);
        for (head, rotary, theta) in [(64, 64, 1e4), (128, 32, 1e4), (128, 64, 1e3)] {
            assert!(rope_entries(PROFILE, "test_q", 2, head, rotary, theta).is_err());
        }
        assert!(rope_entries(PROFILE, "rope_test_q", 2, 128, 64, 1e4).is_err());
    }

    /// The fused RMSNorm and quantize refuses a row that is not whole
    /// scale groups of its output.
    #[test]
    fn rmsnorm_quant_refuses_partial_scale_groups() {
        let fp8 = OutputQuantization::Fp8Group128(ScaleMode::ExactF32);
        let quant =
            |width, output| rmsnorm_quant_entry(PROFILE, "test_rq", width, 1e-6, true, output);
        assert!(quant(4096, fp8).is_ok());
        assert!(quant(4160, fp8).is_err());
        assert!(quant(4160, OutputQuantization::Mxfp8).is_ok());
        assert!(quant(4112, OutputQuantization::Mxfp4).is_err());
        assert!(quant(4112, OutputQuantization::Nvfp4).is_ok());
        assert!(quant(4104, OutputQuantization::Nvfp4).is_err());
        // The ordered squares fill the static shared memory: 12,160 FP8
        // columns fit, 12,288 (`0xc010` bytes) do not; the MXFP8 group
        // maxima leave room for 12,000 but not 12,032.
        assert!(quant(12160, fp8).is_ok());
        assert!(quant(12288, fp8).is_err());
        assert!(quant(12000, OutputQuantization::Mxfp8).is_ok());
        assert!(quant(12032, OutputQuantization::Mxfp8).is_err());
        assert!(quant(32768, OutputQuantization::Nvfp4).is_err());
        // The HF mean of squares serves epsilons from 2^-101 on.
        let least = f32::from_bits(rmsnorm::HF_MIN_EPS_BITS);
        for (eps, served) in [
            (f32::from_bits(1), false),
            (f32::from_bits(rmsnorm::HF_MIN_EPS_BITS - 1), false),
            (least, true),
            (1e-6, true),
        ] {
            for output in [fp8, OutputQuantization::Nvfp4] {
                let entry = rmsnorm_quant_entry(PROFILE, "test_rq", 4096, eps, true, output);
                assert_eq!(entry.is_ok(), served, "{eps:e}");
            }
        }
    }

    /// FP8 rows need eight warps unless fewer half warps cover every
    /// group; the other formats reduce over power-of-two CTAs.
    #[test]
    fn quantizer_coverage_follows_the_cta() {
        assert!(quantizer_covers(0, 4096, 256));
        assert!(!quantizer_covers(0, 4096, 128));
        assert!(!quantizer_covers(1, 4096, 128));
        assert!(quantizer_covers(0, 1024, 128));
        assert!(!quantizer_covers(0, 2048, 96));
        assert!(quantizer_covers(2, 4096, 128));
        // NVFP4's two loading lanes survive a 96-thread reduction, MXFP8's
        // and MXFP4's four a 160-thread one; 96 threads lose MXFP8's lane 2.
        for threads in [96, 160, 192, 224] {
            assert!(quantizer_covers(4, 4096, threads), "{threads}");
        }
        assert!(quantizer_covers(2, 4096, 160));
        assert!(quantizer_covers(3, 4096, 160));
        assert!(!quantizer_covers(2, 4096, 96));
    }

    /// The legacy builders refuse geometry past their templates' `int`s,
    /// partial groups and vectors, and keep their served rows.
    #[test]
    fn legacy_rows_refuse_geometry_their_templates_do_not_cover() {
        // Every accepted entry launches at every row capacity bucket.
        let plans = |rows: &[RowInstantiation]| {
            rows.iter().all(|row| {
                tuning().buckets.iter().all(|&capacity| {
                    let live = RuntimeCounts {
                        m: capacity,
                        batch: 0,
                        context: 0,
                        routed_rows: 0,
                        split_k: 1,
                    };
                    row.plan(capacity, live).is_ok()
                })
            })
        };
        // Quantizers: whole 32-wide groups whose NVFP4 group count, grid.y,
        // stays within 65,535 (1,048,544 columns are 65,534 groups;
        // 1,048,576 are 65,536).
        for width in [4096, 8192, 1_048_544] {
            let rows = quantize_entries(PROFILE, width, "test").unwrap();
            assert_eq!(rows.len(), 6);
            assert!(plans(&rows), "{width}");
        }
        for width in [0, 4100, 4112, 1_048_576, 2_147_483_392, 1 << 31] {
            assert!(quantize_entries(PROFILE, width, "test").is_err(), "{width}");
        }
        // The `nvfp4_row_amax` loop alone stays an `int` to 2,147,483,392
        // columns at 256 threads.
        assert!(loop_fits(2_147_483_392, 256));
        assert!(!loop_fits(2_147_483_424, 256));
        assert!(nvfp4_global_scale(PROFILE).is_ok());
        // Permutation: the gathers' grid, `8192 * 8 + experts * (block -
        // 1)` at the largest bucket, stays within grid.x's 2^31 - 1: blocks
        // of 8,388,352 rows at 256 experts and 16,776,704 at 128.
        let rows = expert_permute_entries(PROFILE, "test", 256, 64).unwrap();
        assert_eq!(rows.len(), 5);
        assert!(plans(&rows));
        for (experts, block, served) in [
            (256, 8_388_352, true),
            (256, 8_388_353, false),
            (128, 16_776_704, true),
            (128, 16_776_705, false),
            (256, 1 << 23, false),
            (1 << 31, 64, false),
            (256, 1 << 31, false),
            (0, 64, false),
            (256, 0, false),
        ] {
            let rows = expert_permute_entries(PROFILE, "test", experts, block);
            assert_eq!(rows.is_ok(), served, "{experts} {block}");
            if let Ok(rows) = rows {
                assert!(plans(&rows), "{experts} {block}");
            }
        }
        // The padded offsets alone: 256 experts of one block each pad to
        // 2,147,483,392 rows and the scatter's tail steps to 2,147,483,423.
        assert!(padded_routes_fit(256, 8, 8_388_607, 32));
        assert!(!padded_routes_fit(256, 8, 1 << 23, 32));
        // Combine: whole vectors, a 256-thread vector loop ending within
        // `int`, and at least the top 8 experts.
        assert!(expert_combine_entry(PROFILE, "test", 4096, 256, true).is_ok());
        assert!(expert_combine_entry(PROFILE, "test", 2_147_481_600, 256, true).is_ok());
        for (width, experts) in [
            (2_147_481_608, 256),
            (1 << 31, 256),
            (4100, 256),
            (0, 256),
            (4096, 4),
        ] {
            assert!(
                expert_combine_entry(PROFILE, "test", width, experts, false).is_err(),
                "{width} {experts}"
            );
        }
        // SwiGLU: gate and up indices up to `2 width - 1` within `int`.
        assert_eq!(swiglu_entries(PROFILE, 1 << 30).unwrap().len(), 2);
        assert!(swiglu_entries(PROFILE, (1 << 30) + 8).is_err());
        assert!(swiglu_entries(PROFILE, 0).is_err());
    }

    /// The MXFP8 rows step eight warps of groups; vector loops end at their
    /// last whole vector.
    #[test]
    fn legacy_cta_and_loop_predicates_follow_their_templates() {
        assert!(mxfp8_rows_cover(4096, 256));
        // 128 threads cover 32 of each 64 groups: 64 of a 4096 row's 128.
        assert!(!mxfp8_rows_cover(4096, 128));
        assert!(mxfp8_rows_cover(1024, 128));
        assert!(!mxfp8_rows_cover(1024, 32));
        assert!(mxfp8_rows_cover(256, 32));
        // A 256-thread vector loop from 2,147,481,600 ends at INT32_MAX - 7.
        assert!(vector_loop_fits(2_147_481_600, 256));
        assert!(!vector_loop_fits(2_147_481_608, 256));
        // The global scale's 96-thread CTA drops lane 2's maximum; the
        // quantizers' 32-thread one keeps every loading lane.
        assert!(!tree_keeps(96, 8192));
        assert!([16, 4, 2].iter().all(|&lanes| tree_keeps(32, lanes)));
    }

    /// The tree reduction keeps every lane of a power-of-two CTA, and of
    /// others only the lanes its halving reaches.
    #[test]
    fn tree_reductions_keep_the_lanes_they_reach() {
        for threads in [32, 64, 128, 256] {
            assert!(tree_keeps(threads, threads));
        }
        assert!(!tree_keeps(96, 96));
        assert!(!tree_keeps(96, 3));
        assert!(tree_keeps(96, 2));
        assert!(!tree_keeps(160, 5));
        assert!(tree_keeps(160, 4));
    }

    /// The tiled permutation scans exactly 256 experts and at most 256
    /// route tiles at the largest row bucket.
    #[test]
    fn tiled_permutation_refuses_other_experts_and_too_many_tiles() {
        assert_eq!(
            permute_tile_entries(PROFILE, "test_b64", 256, 8, 64)
                .unwrap()
                .len(),
            4
        );
        assert!(permute_tile_entries(PROFILE, "test_b64", 128, 8, 64).is_err());
        assert!(permute_tile_entries(PROFILE, "test_b64", 256, 16, 64).is_err());
        assert!(permute_tile_entries(PROFILE, "test_b64", 256, 0, 64).is_err());
        assert!(permute_tile_entries(PROFILE, "test_b64", 256, 8, 0).is_err());
        assert!(permute_tile_entries(PROFILE, "test_b64", 256, 8, 1 << 31).is_err());
        // 256 experts of one block each pad to 256 * 8,388,607 =
        // 2,147,483,392 rows, an `int`; one more row per block is not.
        assert!(permute_tile_entries(PROFILE, "test_b64", 256, 8, 8_388_607).is_ok());
        assert!(permute_tile_entries(PROFILE, "test_b64", 256, 8, 1 << 23).is_err());
        assert!(permute_tile_entries(PROFILE, "test_b64", 256, 8, 1 << 20).is_ok());
    }

    /// The head-gridded rows refuse heads past grid.y's 65,535 at any row
    /// capacity bucket, and launch the last accepted count at every bucket.
    /// A RoPE KV write grids every Q head below 512 rows, so it stops at
    /// 65,532 Q heads (whole groups of four).
    #[test]
    fn head_rows_refuse_heads_past_the_launch_grid() {
        let plans = |rows: &[RowInstantiation]| {
            rows.iter().all(|row| {
                tuning().buckets.iter().all(|&capacity| {
                    let live = RuntimeCounts {
                        m: capacity,
                        batch: 0,
                        context: 0,
                        routed_rows: 0,
                        split_k: 1,
                    };
                    row.plan(capacity, live).is_ok()
                })
            })
        };
        let qk = |heads| qk_norm_entry(PROFILE, "test", heads, 128, 1e-6).map(|row| vec![row]);
        assert!(plans(&qk(65_535).unwrap()));
        assert!(qk(65_536).is_err());
        // A QK norm under the RoPE rows' prefix takes their four-fold
        // grid below 512 rows; heads whose four-fold count passes u32
        // refuse rather than overflow.
        let rope_named = |heads| qk_norm_entry(PROFILE, "rope_test", heads, 128, 1e-6);
        assert!(rope_named((1 << 30) + 1).is_err());
        assert!(rope_named(u32::MAX >> 1).is_err());
        let rope = |heads| rope_entries(PROFILE, "test", heads, 128, 64, 1e4);
        assert!(plans(&rope(65_535).unwrap()));
        assert!(rope(65_536).is_err());
        let ring = |heads| ring_write_entries(PROFILE, "test", heads, 128, 128, 128);
        assert!(plans(&ring(65_535).unwrap()));
        assert!(ring(65_536).is_err());
        let rope_kv = |q_heads| {
            let shape = RopeKvShape {
                q_heads,
                kv_heads: 1,
                qk_dim: 128,
                v_dim: 128,
                rotary_dim: 64,
                theta: 1e4,
                style: RotaryStyle::Neox,
                ring: false,
                prefill: false,
                fp8: false,
            };
            rope_kv_entries(PROFILE, "test", shape)
        };
        assert!(plans(&rope_kv(65_532).unwrap()));
        assert!(rope_kv(65_536).is_err());
    }

    /// RoPE KV writes refuse Q heads outside groups of four, more KV heads
    /// than Q heads, rotated widths the head does not hold and paged
    /// prefill writes; ring writes and the sharded argmax refuse empty
    /// shapes and IDs past 32 bits.
    #[test]
    fn cache_writes_and_argmax_refuse_shapes_their_templates_do_not_serve() {
        let shape = RopeKvShape {
            q_heads: 16,
            kv_heads: 2,
            qk_dim: 192,
            v_dim: 128,
            rotary_dim: 64,
            theta: 1e4,
            style: RotaryStyle::Neox,
            ring: true,
            prefill: true,
            fp8: false,
        };
        assert_eq!(rope_kv_entries(PROFILE, "test", shape).unwrap().len(), 4);
        let paged = RopeKvShape {
            ring: false,
            prefill: false,
            ..shape
        };
        assert_eq!(rope_kv_entries(PROFILE, "test", paged).unwrap().len(), 12);
        for refused in [
            RopeKvShape {
                q_heads: 6,
                ..shape
            },
            RopeKvShape {
                q_heads: 0,
                ..shape
            },
            RopeKvShape {
                kv_heads: 20,
                ..shape
            },
            RopeKvShape {
                kv_heads: 0,
                ..shape
            },
            RopeKvShape {
                rotary_dim: 63,
                ..shape
            },
            RopeKvShape {
                rotary_dim: 256,
                ..shape
            },
            RopeKvShape {
                prefill: true,
                ..paged
            },
        ] {
            assert!(
                rope_kv_entries(PROFILE, "test", refused).is_err(),
                "{refused:?}"
            );
        }
        assert_eq!(
            ring_write_entries(PROFILE, "test", 2, 192, 128, 128)
                .unwrap()
                .len(),
            5
        );
        assert!(ring_write_entries(PROFILE, "test", 2, 192, 128, 0).is_err());
        assert!(ring_write_entries(PROFILE, "test", 0, 192, 128, 128).is_err());
        assert!(ring_write_entries(PROFILE, "test", 2, 1 << 31, 128, 128).is_err());
        assert!(ring_write_entries(PROFILE, "test", 2, 192, 1 << 31, 128).is_err());
        // A 256-thread loop over 2,147,483,392 dimensions ends at INT32_MAX;
        // one more dimension overflows its increment.
        assert!(ring_write_entries(PROFILE, "test", 2, 2_147_483_392, 128, 128).is_ok());
        assert!(ring_write_entries(PROFILE, "test", 2, 2_147_483_393, 128, 128).is_err());
        assert!(ring_write_entries(PROFILE, "test", 2, 192, 2_147_483_393, 128).is_err());
        let overflowing = RopeKvShape {
            q_heads: u32::MAX - 3,
            kv_heads: u32::MAX - 3,
            qk_dim: u32::MAX,
            v_dim: u32::MAX,
            ..shape
        };
        assert!(rope_kv_entries(PROFILE, "test", overflowing).is_err());
        let huge = RopeKvShape {
            qk_dim: 1 << 31,
            rotary_dim: 64,
            ..shape
        };
        assert!(rope_kv_entries(PROFILE, "test", huge).is_err());
        let wide = RopeKvShape {
            q_heads: 1 << 24,
            kv_heads: 1,
            ..shape
        };
        assert!(rope_kv_entries(PROFILE, "test", wide).is_err());
        // A packed row within `int` whose V loop's last 256-thread step
        // would pass INT32_MAX; one dimension less ends at INT32_MAX.
        let narrow_qk = RopeKvShape {
            q_heads: 4,
            kv_heads: 1,
            qk_dim: 2,
            rotary_dim: 2,
            v_dim: 2_147_483_393,
            ..shape
        };
        assert!(rope_kv_entries(PROFILE, "test", narrow_qk).is_err());
        let v_at_limit = RopeKvShape {
            v_dim: 2_147_483_392,
            ..narrow_qk
        };
        assert!(rope_kv_entries(PROFILE, "test", v_at_limit).is_ok());
        assert!(qk_norm_entry(PROFILE, "test_q_norm", 1 << 31, 128, 1e-6).is_err());
        assert!(rope_entries(PROFILE, "test_q", 1 << 31, 128, 64, 1e4).is_err());
        assert!(sharded_argmax_entries(PROFILE, "test", 38144, 4).is_ok());
        assert!(sharded_argmax_entries(PROFILE, "test", 0, 4).is_err());
        assert!(sharded_argmax_entries(PROFILE, "test", u32::MAX / 2, 4).is_err());
        // The last ID of two 2^30-entry shards is 2^31 - 1; one more entry
        // passes it, and a single 2^31-entry shard is no template int.
        assert!(sharded_argmax_entries(PROFILE, "test", 1 << 30, 2).is_ok());
        assert!(sharded_argmax_entries(PROFILE, "test", (1 << 30) + 1, 2).is_err());
        assert!(sharded_argmax_entries(PROFILE, "test", 1 << 31, 1).is_err());
        // A shard narrower than the CTA needs only its own lanes kept.
        assert!(tree_keeps(96, 2) && !tree_keeps(96, 128));
    }
}
