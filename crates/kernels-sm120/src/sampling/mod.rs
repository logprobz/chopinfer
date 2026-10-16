//! Model-neutral SM120 device sampling entry and launch contract.

use std::path::PathBuf;

use crate::Profile;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, CompileKey, CompileKnobs, DType, InstantiationSpec, LaunchPlan,
    OpFamily, Refusal, RuntimeCounts, StaticQuery, Target, Validate,
};

pub const MAX_BATCH: u32 = 64;
pub const MAX_WIDTH: u32 = 1 << 20;
pub const SIMPLE_TILE: u32 = 1024;
pub const SIMPLE_PARTIAL_BYTES: u64 = 112;
/// Per-row filtered sampling record: a 128-byte state and 2048 16-byte bins.
pub const FILTER_ROW_BYTES: u64 = 128 + 2048 * 16;
const TUNING: &str = include_str!("../../tuning/sampling.json");

fn tuned_warps(width: u32) -> Result<u32, Refusal> {
    let key = if width <= 8192 {
        "\"width_le_8192_warps\""
    } else {
        "\"width_le_1048576_warps\""
    };
    let line = TUNING
        .lines()
        .find(|line| line.trim_start().starts_with(key))
        .ok_or(Refusal("sampling tuning bucket is missing"))?;
    let value = line
        .split_once(':')
        .and_then(|(_, value)| value.trim().trim_end_matches(',').parse::<u32>().ok())
        .ok_or(Refusal("sampling tuning warp count is invalid"))?;
    if !(1..=32).contains(&value) {
        return Err(Refusal("sampling tuning warp count is out of range"));
    }
    Ok(value)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SamplingQuery {
    pub row_capacity: u32,
    pub width_capacity: u32,
    pub dtype: DType,
}

impl Validate for SamplingQuery {
    fn validate(&self) -> Result<(), Refusal> {
        if self.row_capacity == 0 || self.row_capacity > MAX_BATCH {
            return Err(Refusal("sampling row capacity must be 1..=64"));
        }
        if self.width_capacity == 0 || self.width_capacity > MAX_WIDTH {
            return Err(Refusal("sampling width capacity must be 1..=1048576"));
        }
        if !matches!(self.dtype, DType::F32 | DType::Bf16) {
            return Err(Refusal("sampling supports F32 or BF16 logits"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SamplingBuffers {
    pub logits: u64,
    pub metadata: u64,
    pub counts: u64,
    pub biases: u64,
    pub eos: u64,
    pub adjusted: u64,
    pub order: u64,
    pub output: u64,
    /// Constraint masks, `live_width.div_ceil(32)` words per row, or none.
    pub masks: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedSampling {
    pub launch: LaunchPlan,
    pub arguments: ArgumentList,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedSimpleSampling {
    pub partials: LaunchPlan,
    pub finish: LaunchPlan,
    pub partial_args: ArgumentList,
    pub finish_args: ArgumentList,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SimpleSamplingBuffers {
    pub logits: u64,
    pub metadata: u64,
    pub partials: u64,
    pub output: u64,
}

/// `top` selects the `simple_top_partials` entries, whose top-logprob tile
/// sort needs 256 threads; the others use 64 threads above eight rows.
pub fn plan_simple(
    query: SamplingQuery,
    counts: RuntimeCounts,
    live_width: u32,
    row_stride_elements: u64,
    buffers: SimpleSamplingBuffers,
    top: bool,
) -> Result<PlannedSimpleSampling, Refusal> {
    query.validate()?;
    if counts.batch == 0 || counts.batch > query.row_capacity {
        return Err(Refusal("sampling batch exceeds prepared capacity"));
    }
    if live_width == 0 || live_width > query.width_capacity {
        return Err(Refusal("sampling live width exceeds prepared capacity"));
    }
    if row_stride_elements < u64::from(live_width) {
        return Err(Refusal("sampling row stride is narrower than live width"));
    }
    let tiles = live_width.div_ceil(SIMPLE_TILE);
    let workspace_bytes = u64::from(query.row_capacity)
        .checked_mul(u64::from(tiles))
        .and_then(|items| items.checked_mul(SIMPLE_PARTIAL_BYTES))
        .ok_or(Refusal("sampling partial workspace size overflow"))?;
    let partials = LaunchPlan {
        grid: [tiles, counts.batch, 1],
        block: [if top || counts.batch <= 8 { 256 } else { 64 }, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes,
        cluster: [1, 1, 1],
        pdl: true,
    };
    let finish = LaunchPlan {
        grid: [counts.batch, 1, 1],
        block: [32, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes,
        cluster: [1, 1, 1],
        pdl: true,
    };
    partials.validate()?;
    finish.validate()?;
    let common = vec![
        Argument::DeviceBuffer(buffers.logits),
        Argument::U64(row_stride_elements),
        Argument::U32(live_width),
        Argument::U32(counts.batch),
        Argument::DeviceBuffer(buffers.metadata),
        Argument::DeviceBuffer(buffers.partials),
    ];
    let partial_args = ArgumentList(common.clone());
    let mut last = common;
    last.push(Argument::DeviceBuffer(buffers.output));
    let finish_args = ArgumentList(last);
    partial_args.validate()?;
    finish_args.validate()?;
    Ok(PlannedSimpleSampling {
        partials,
        finish,
        partial_args,
        finish_args,
    })
}

/// The filtered path: stats, histogram, collect, resolve and Gumbel kernels
/// share one argument list; the finish kernel takes the simple finish
/// arguments.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedFilteredSampling {
    pub stats: LaunchPlan,
    pub histogram: LaunchPlan,
    pub collect: LaunchPlan,
    pub resolve: LaunchPlan,
    pub gumbel: LaunchPlan,
    pub finish: LaunchPlan,
    pub tile_args: ArgumentList,
    pub finish_args: ArgumentList,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FilteredSamplingBuffers {
    pub logits: u64,
    pub metadata: u64,
    pub counts: u64,
    pub biases: u64,
    pub eos: u64,
    pub partials: u64,
    pub filter: u64,
    pub adjusted: u64,
    pub order: u64,
    pub output: u64,
    /// Constraint masks, `live_width.div_ceil(32)` words per row, or none.
    pub masks: Option<u64>,
}

/// Histogram blocks per row: about 384 blocks in total, at most four tiles
/// per block, and never more blocks than tiles. `runtime-kernels` mirrors it.
pub fn filtered_histogram_groups(tiles: u32, rows: u32) -> u32 {
    tiles
        .div_ceil(4)
        .max(384_u32.div_ceil(rows.max(1)).min(tiles))
}

/// Plans the filtered path. The workspace covers simple partials, filter
/// records and `rows * live` candidate values (f64) and IDs (u32).
pub fn plan_filtered(
    query: SamplingQuery,
    counts: RuntimeCounts,
    live_width: u32,
    row_stride_elements: u64,
    buffers: FilteredSamplingBuffers,
) -> Result<PlannedFilteredSampling, Refusal> {
    query.validate()?;
    if counts.batch == 0 || counts.batch > query.row_capacity {
        return Err(Refusal("sampling batch exceeds prepared capacity"));
    }
    if live_width == 0 || live_width > query.width_capacity {
        return Err(Refusal("sampling live width exceeds prepared capacity"));
    }
    if row_stride_elements < u64::from(live_width) {
        return Err(Refusal("sampling row stride is narrower than live width"));
    }
    let tiles = live_width.div_ceil(SIMPLE_TILE);
    let rows = u64::from(query.row_capacity);
    let workspace_bytes = rows
        .checked_mul(u64::from(tiles))
        .and_then(|items| items.checked_mul(SIMPLE_PARTIAL_BYTES))
        .and_then(|partials| partials.checked_add(rows * FILTER_ROW_BYTES))
        .and_then(|bytes| bytes.checked_add(rows * u64::from(query.width_capacity) * 12))
        .ok_or(Refusal("sampling filtered workspace size overflow"))?;
    let tile = |width, threads| LaunchPlan {
        grid: [width, counts.batch, 1],
        block: [threads, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes,
        cluster: [1, 1, 1],
        pdl: true,
    };
    let per_row = |threads| LaunchPlan {
        grid: [counts.batch, 1, 1],
        block: [threads, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes,
        cluster: [1, 1, 1],
        pdl: true,
    };
    let planned = PlannedFilteredSampling {
        stats: tile(tiles, 256),
        histogram: tile(filtered_histogram_groups(tiles, counts.batch), 256),
        collect: tile(tiles, 256),
        resolve: per_row(1024),
        gumbel: tile(tiles, 256),
        finish: per_row(32),
        tile_args: ArgumentList(vec![
            Argument::DeviceBuffer(buffers.logits),
            Argument::U64(row_stride_elements),
            Argument::U32(live_width),
            Argument::U32(counts.batch),
            Argument::DeviceBuffer(buffers.metadata),
            Argument::DeviceBuffer(buffers.counts),
            Argument::DeviceBuffer(buffers.biases),
            Argument::DeviceBuffer(buffers.eos),
            Argument::DeviceBuffer(buffers.partials),
            Argument::DeviceBuffer(buffers.filter),
            Argument::DeviceBuffer(buffers.adjusted),
            Argument::DeviceBuffer(buffers.order),
            buffers
                .masks
                .map_or(Argument::NullDevicePointer, Argument::DeviceBuffer),
            Argument::U32(if buffers.masks.is_some() {
                live_width.div_ceil(32)
            } else {
                0
            }),
        ]),
        finish_args: ArgumentList(vec![
            Argument::DeviceBuffer(buffers.logits),
            Argument::U64(row_stride_elements),
            Argument::U32(live_width),
            Argument::U32(counts.batch),
            Argument::DeviceBuffer(buffers.metadata),
            Argument::DeviceBuffer(buffers.partials),
            Argument::DeviceBuffer(buffers.output),
        ]),
    };
    for launch in [
        planned.stats,
        planned.histogram,
        planned.collect,
        planned.resolve,
        planned.gumbel,
        planned.finish,
    ] {
        launch.validate()?;
    }
    planned.tile_args.validate()?;
    planned.finish_args.validate()?;
    Ok(planned)
}

pub fn plan(
    query: SamplingQuery,
    counts: RuntimeCounts,
    live_width: u32,
    row_stride_elements: u64,
    buffers: SamplingBuffers,
) -> Result<PlannedSampling, Refusal> {
    query.validate()?;
    if counts.batch == 0 || counts.batch > query.row_capacity {
        return Err(Refusal("sampling batch exceeds prepared capacity"));
    }
    if live_width == 0 || live_width > query.width_capacity {
        return Err(Refusal("sampling live width exceeds prepared capacity"));
    }
    if row_stride_elements < u64::from(live_width) {
        return Err(Refusal("sampling row stride is narrower than live width"));
    }
    let workspace_bytes = u64::from(query.row_capacity)
        .checked_mul(u64::from(query.width_capacity))
        .and_then(|items| items.checked_mul(12))
        .ok_or(Refusal("sampling workspace size overflow"))?;
    let launch = LaunchPlan {
        grid: [query.row_capacity, 1, 1],
        block: [32 * tuned_warps(query.width_capacity)?, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes,
        cluster: [1, 1, 1],
        pdl: true,
    };
    launch.validate()?;
    let arguments = ArgumentList(vec![
        Argument::DeviceBuffer(buffers.logits),
        Argument::U64(row_stride_elements),
        Argument::U32(live_width),
        Argument::U32(counts.batch),
        Argument::DeviceBuffer(buffers.metadata),
        Argument::DeviceBuffer(buffers.counts),
        Argument::DeviceBuffer(buffers.biases),
        Argument::DeviceBuffer(buffers.eos),
        Argument::DeviceBuffer(buffers.adjusted),
        Argument::DeviceBuffer(buffers.order),
        Argument::DeviceBuffer(buffers.output),
        buffers
            .masks
            .map_or(Argument::NullDevicePointer, Argument::DeviceBuffer),
        Argument::U32(if buffers.masks.is_some() {
            live_width.div_ceil(32)
        } else {
            0
        }),
    ]);
    arguments.validate()?;
    Ok(PlannedSampling { launch, arguments })
}

pub fn instantiation(query: SamplingQuery) -> Result<InstantiationSpec, Refusal> {
    query.validate()?;
    let format = if query.dtype == DType::F32 {
        "f32"
    } else {
        "bf16"
    };
    let spec = InstantiationSpec {
        family: OpFamily::Row,
        key: CompileKey {
            query: StaticQuery::Row {
                family: "sampling",
                width_capacity: query.width_capacity,
                dtype: query.dtype,
                traits_digest: 0,
            },
            knobs: CompileKnobs {
                tile_m: 1,
                tile_n: 32,
                tile_k: 1,
                stages: 1,
                swizzle: 1,
                warps: tuned_warps(query.width_capacity)? as u8,
                split_k_capacity: 1,
            },
            dense: None,
            target: Target::Sm120f,
        },
        symbol: format!("chop_sampling_{format}_v1"),
    };
    spec.validate()?;
    Ok(spec)
}

pub fn cubin_spec() -> CubinSpec {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda");
    let threads = 32 * tuned_warps(152_576).expect("sampling tuning is valid");
    let params = |logit_type: &str| {
        let mut values = vec![EntryParam::DevicePointer {
            name: "logits".into(),
            cxx_type: format!("const {logit_type}"),
        }];
        values.extend([
            EntryParam::U64 {
                name: "stride".into(),
            },
            EntryParam::U32 {
                name: "live".into(),
            },
            EntryParam::U32 {
                name: "rows".into(),
            },
        ]);
        for (name, cxx_type) in [
            ("meta", "const chop::sampling::RowMeta"),
            ("counts", "const chop::sampling::TokenCount"),
            ("biases", "const chop::sampling::TokenBias"),
            ("eos", "const uint32_t"),
            ("adjusted", "double"),
            ("order", "uint32_t"),
            ("output", "chop::sampling::RowOutput"),
            ("masks", "const uint32_t"),
        ] {
            values.push(EntryParam::DevicePointer {
                name: name.into(),
                cxx_type: cxx_type.into(),
            });
        }
        values.push(EntryParam::U32 {
            name: "mask_words".into(),
        });
        values
    };
    let body = |bf16: bool| {
        format!(
            "chop::sampling::run<{bf16}>(logits, stride, live, rows, meta, counts, biases, eos, adjusted, order, output, masks, mask_words);"
        )
    };
    let simple_params = |logit_type: &str, finish: bool| {
        let mut values = vec![EntryParam::DevicePointer {
            name: "logits".into(),
            cxx_type: format!("const {logit_type}"),
        }];
        values.extend([
            EntryParam::U64 {
                name: "stride".into(),
            },
            EntryParam::U32 {
                name: "live".into(),
            },
            EntryParam::U32 {
                name: "rows".into(),
            },
            EntryParam::DevicePointer {
                name: "meta".into(),
                cxx_type: "const chop::sampling::RowMeta".into(),
            },
            EntryParam::DevicePointer {
                name: "partials".into(),
                cxx_type: if finish {
                    "const chop::sampling::SimplePartial".into()
                } else {
                    "chop::sampling::SimplePartial".into()
                },
            },
        ]);
        if finish {
            values.push(EntryParam::DevicePointer {
                name: "output".into(),
                cxx_type: "chop::sampling::RowOutput".into(),
            });
        }
        values
    };
    let filtered_params = |logit_type: &str| {
        let mut values = params(logit_type);
        values.truncate(8);
        for (name, cxx_type) in [
            ("partials", "chop::sampling::SimplePartial"),
            ("filter", "chop::sampling::FilterRow"),
            ("adjusted", "double"),
            ("order", "uint32_t"),
            ("masks", "const uint32_t"),
        ] {
            values.push(EntryParam::DevicePointer {
                name: name.into(),
                cxx_type: cxx_type.into(),
            });
        }
        values.push(EntryParam::U32 {
            name: "mask_words".into(),
        });
        values
    };
    let mut entries = Vec::new();
    for (format, logit_type) in [("f32", "float"), ("bf16", "__nv_bfloat16")] {
        for (stage, body, threads) in [
            (
                "stats",
                "chop::sampling::filtered_stats(logits, stride, live, rows, meta, counts, biases, eos, partials, filter, masks, mask_words);",
                256,
            ),
            (
                "histogram",
                "chop::sampling::filtered_histogram(logits, stride, live, rows, meta, counts, biases, eos, partials, filter, masks, mask_words);",
                256,
            ),
            (
                "collect",
                "chop::sampling::filtered_collect(logits, stride, live, rows, meta, counts, biases, eos, partials, filter, adjusted, order, masks, mask_words);",
                256,
            ),
            (
                "resolve",
                "chop::sampling::filtered_resolve(logits, stride, live, rows, meta, counts, biases, eos, filter, adjusted, order, masks, mask_words);",
                1024,
            ),
            (
                "gumbel",
                "chop::sampling::filtered_gumbel(logits, stride, live, rows, meta, counts, biases, eos, partials, filter, masks, mask_words);",
                256,
            ),
        ] {
            entries.push(EntrySpec::new(
                format!("chop_sampling_filtered_{stage}_{format}_v1"),
                filtered_params(logit_type),
                body.into(),
                threads,
                1,
            ));
        }
        entries.push(EntrySpec::new(
            format!("chop_sampling_filtered_finish_{format}_v1"),
            simple_params(logit_type, true),
            format!(
                "chop::sampling::simple_finish<{logit_type}, chop::sampling::SIMPLE_TILE, true>(logits, stride, live, rows, meta, partials, output);"
            ),
            32,
            1,
        ));
    }
    CubinSpec {
        name: "sampling_sm120".into(),
        sources_root: root,
        include_dirs: vec![PathBuf::from(".")],
        headers: vec!["#include \"sampling/filtered.cuh\"".into()],
        defines: Vec::new(),
        entries: vec![
            EntrySpec::new(
                "chop_sampling_f32_v1".into(),
                params("float"),
                body(false),
                threads,
                1,
            ),
            EntrySpec::new(
                "chop_sampling_bf16_v1".into(),
                params("__nv_bfloat16"),
                body(true),
                threads,
                1,
            ),
            EntrySpec::new(
                "chop_sampling_simple_partials_f32_v1".into(),
                simple_params("float", false),
                "chop::sampling::simple_partials(logits, stride, live, rows, meta, partials);".into(),
                256,
                1,
            ),
            EntrySpec::new(
                "chop_sampling_simple_top_partials_f32_v1".into(),
                simple_params("float", false),
                "chop::sampling::simple_partials<float, true>(logits, stride, live, rows, meta, partials);".into(),
                256,
                1,
            ),
            EntrySpec::new(
                "chop_sampling_simple_finish_f32_v1".into(),
                simple_params("float", true),
                "chop::sampling::simple_finish(logits, stride, live, rows, meta, partials, output);".into(),
                32,
                1,
            ),
            EntrySpec::new(
                "chop_sampling_simple_partials_bf16_v1".into(),
                simple_params("__nv_bfloat16", false),
                "chop::sampling::simple_partials(logits, stride, live, rows, meta, partials);".into(),
                256,
                1,
            ),
            EntrySpec::new(
                "chop_sampling_simple_top_partials_bf16_v1".into(),
                simple_params("__nv_bfloat16", false),
                "chop::sampling::simple_partials<__nv_bfloat16, true>(logits, stride, live, rows, meta, partials);".into(),
                256,
                1,
            ),
            EntrySpec::new(
                "chop_sampling_simple_finish_bf16_v1".into(),
                simple_params("__nv_bfloat16", true),
                "chop::sampling::simple_finish(logits, stride, live, rows, meta, partials, output);".into(),
                32,
                1,
            ),
            EntrySpec::new(
                "chop_sampling_greedy_partials_1024_f32_v1".into(),
                simple_params("float", false),
                "chop::sampling::greedy_partials<float, 1024>(logits, stride, live, rows, meta, partials);".into(),
                256,
                1,
            ),
            EntrySpec::new(
                "chop_sampling_greedy_partials_1024_bf16_v1".into(),
                simple_params("__nv_bfloat16", false),
                "chop::sampling::greedy_partials<__nv_bfloat16, 1024>(logits, stride, live, rows, meta, partials);".into(),
                256,
                1,
            ),
        ]
        .into_iter()
        .chain(entries)
        .collect(),
    }
}

/// Sampling has one model-neutral cubin shared by both supported profiles.
pub fn cubin_specs(_profile: Option<Profile>) -> Vec<CubinSpec> {
    vec![cubin_spec()]
}
