//! Row stages in both shape forms. A launch-shape entry
//! (`ShapeBinding::Launch`) takes its widths, strides, heads, vocabulary
//! bound and routed-row divisor as launch arguments; the compiled instance of
//! the same shapes (`ShapeBinding::Compiled`) is the reference it must equal
//! bit for bit. Every pair comes from a served use, a catalog's probe of a
//! shape it does not serve yet, or a lab shape for the stage kinds no served
//! use covers. Both arms take lab-local names from their bodies, so the
//! prepared selection holds each distinct entry once; `sass_same.sh OLD=NEW`
//! ties a lab entry to a served one.

use kernel_build::{CubinSpec, EntryDeclaration, EntryParam, EntrySpec, SelectedCubin};
use kernel_core::contract::{
    ArgumentList, DType, LaunchPlan, OutputQuantization, ScaleMode, WeightFormat,
};
use kernels_sm120::precision::{Compute, KernelOffer};
use kernels_sm120::row::precision::{
    self as row, CopyRows, GatherInput, LookupBuffers, LookupIds, LookupRequest, Nvfp4Global,
    QuantizerBuffers, RowResources, RowStage, ShapeBinding,
};
use std::collections::BTreeSet;

/// Buffer handles in a pair's launch arguments; each arm binds its own
/// outputs to them.
pub mod buffer {
    pub const INPUT: u64 = 1;
    pub const OUTPUT: u64 = 2;
    pub const LIVE: u64 = 3;
    pub const CODES: u64 = 4;
    pub const SCALES: u64 = 5;
    pub const GLOBAL: u64 = 6;
    pub const AMAX: u64 = 7;
    pub const TABLE: u64 = 8;
    pub const IDS: u64 = 9;
    pub const LOCAL_SCALES: u64 = 10;
    pub const SORTED: u64 = 11;
    pub const PADDED: u64 = 12;
}

/// One row builder: its stages launch in order.
#[derive(Clone, Debug)]
pub enum Stage {
    Copy(row::LayoutCopy),
    Lookup(row::Lookup),
    Quantizer(row::Quantizer),
    Gather(row::Gather),
}

impl Stage {
    pub fn stages(&self) -> Vec<RowStage> {
        match self {
            Stage::Copy(c) => vec![c.stage().clone()],
            Stage::Lookup(l) => vec![l.stage().clone()],
            Stage::Quantizer(q) => q.stages().to_vec(),
            Stage::Gather(g) => vec![g.stage().clone()],
        }
    }
    /// The planned rows: live rows, or padded routed rows for a gather.
    pub fn capacity(&self) -> u32 {
        self.stages().last().expect("row builder stage").capacity()
    }
    fn shapes(&self) -> ShapeBinding {
        self.stages()[0].shapes()
    }
    fn with_shapes(&self, shapes: ShapeBinding, symbol: &str) -> Result<Self, String> {
        let e = |e: kernel_core::contract::Refusal| e.to_string();
        Ok(match self {
            Stage::Copy(c) => Stage::Copy(c.with_shapes(shapes, symbol).map_err(e)?),
            Stage::Lookup(l) => Stage::Lookup(l.with_shapes(shapes, symbol).map_err(e)?),
            Stage::Quantizer(q) => Stage::Quantizer(q.with_shapes(shapes, symbol).map_err(e)?),
            Stage::Gather(g) => Stage::Gather(g.with_shapes(shapes, symbol).map_err(e)?),
        })
    }
    /// Each stage with its arguments over [`buffer`] handles, in launch
    /// order. A lookup addresses table rows `row_offset..row_offset + shard_rows`.
    pub fn launches(
        &self,
        row_offset: u64,
        shard_rows: u64,
    ) -> Result<Vec<(RowStage, ArgumentList)>, String> {
        use buffer::*;
        let e = |e: kernel_core::contract::Refusal| e.to_string();
        Ok(match self {
            Stage::Copy(c) => vec![(
                c.stage().clone(),
                c.arguments(INPUT, OUTPUT, LIVE).map_err(e)?,
            )],
            Stage::Lookup(l) => {
                let r = l.request();
                vec![(
                    l.stage().clone(),
                    l.arguments(LookupBuffers {
                        table: TABLE,
                        ids: IDS,
                        local_scales: (r.weight == WeightFormat::Nvfp4).then_some(LOCAL_SCALES),
                        global_scale: r.global_scale.map(|_| GLOBAL),
                        output: OUTPUT,
                        live_rows: LIVE,
                        row_offset,
                        shard_rows,
                    })
                    .map_err(e)?,
                )]
            }
            Stage::Quantizer(q) => q
                .stages()
                .iter()
                .cloned()
                .zip(
                    q.arguments(QuantizerBuffers {
                        status_row_offset: 0,
                        status: None,
                        input: INPUT,
                        codes: CODES,
                        scales: SCALES,
                        live_rows: LIVE,
                        global_scale: (q.global() != Nvfp4Global::Absent).then_some(GLOBAL),
                        row_amax: (q.global() == Nvfp4Global::TensorAmax).then_some(AMAX),
                    })
                    .map_err(e)?,
                )
                .collect(),
            Stage::Gather(g) => vec![(
                g.stage().clone(),
                g.arguments(
                    INPUT,
                    SORTED,
                    if g.quantized() { CODES } else { OUTPUT },
                    g.quantized().then_some(SCALES),
                    PADDED,
                )
                .map_err(e)?,
            )],
        })
    }
}

/// Where a pair comes from.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum Origin {
    /// A served use of a model's inventory.
    Served,
    /// A catalog's probe of a use it does not serve, such as another topology's.
    Probe,
    /// A lab shape for a stage kind no served use covers.
    Lab,
}

/// One use in both forms under lab-local names.
#[derive(Clone, Debug)]
pub struct Pair {
    /// The served symbol, or a description of a lab shape.
    pub label: String,
    pub origin: Origin,
    /// The compiled instance of the use's shapes.
    pub reference: Stage,
    /// The launch-shape entry.
    pub candidate: Stage,
}

/// A lab-local name from a builder's last stage body and ABI.
fn lab_name(stage: &Stage) -> Result<Stage, String> {
    let placeholder = stage.with_shapes(stage.shapes(), "chop_lab_row_v1")?;
    let entry = placeholder
        .stages()
        .last()
        .expect("row stage")
        .entry()
        .clone();
    let digest = kernel_build::digest(
        format!(
            "{:?}|{}|{}|{}",
            entry.params, entry.body, entry.max_threads, entry.min_blocks
        )
        .as_bytes(),
    );
    let form = match stage.shapes() {
        ShapeBinding::Launch => "launch",
        ShapeBinding::Compiled => "compiled",
    };
    stage.with_shapes(
        stage.shapes(),
        &format!("chop_lab_row_{form}_{}_v1", &digest[..16]),
    )
}

fn pair(label: String, origin: Origin, served: Stage) -> Result<Pair, String> {
    let reference = lab_name(&served.with_shapes(ShapeBinding::Compiled, "chop_lab_row_v1")?)?;
    let candidate = lab_name(&served.with_shapes(ShapeBinding::Launch, "chop_lab_row_v1")?)?;
    Ok(Pair {
        label,
        origin,
        reference,
        candidate,
    })
}

/// The row builders of an offer: its gather, quantizer and lookup.
fn offer_stages(offer: &KernelOffer) -> Vec<Stage> {
    let handle = offer.handle();
    let mut stages = Vec::new();
    if let Some(gather) = &handle.gather {
        stages.push(Stage::Gather(gather.clone()));
    }
    if let Some(quantizer) = &handle.quantizer {
        stages.push(Stage::Quantizer(quantizer.clone()));
    }
    if let Compute::Lookup(lookup) = &handle.compute {
        stages.push(Stage::Lookup(lookup.clone()));
    }
    stages
}

fn lab_stages() -> Result<Vec<(String, Stage)>, String> {
    let e = |e: kernel_core::contract::Refusal| e.to_string();
    let r = RowResources::SM120F;
    let c = ShapeBinding::Compiled;
    let mut stages = Vec::new();
    let quantizers = [
        (OutputQuantization::Nvfp4, Nvfp4Global::Supplied, 256, 1),
        (OutputQuantization::Nvfp4, Nvfp4Global::Supplied, 256, 64),
        (OutputQuantization::Nvfp4, Nvfp4Global::TensorAmax, 256, 64),
        (
            OutputQuantization::Fp8Group128(ScaleMode::ExactF32),
            Nvfp4Global::Absent,
            2560,
            64,
        ),
        (
            OutputQuantization::Fp8Group128(ScaleMode::PowerOfTwoF32),
            Nvfp4Global::Absent,
            256,
            64,
        ),
        (OutputQuantization::Mxfp8, Nvfp4Global::Absent, 256, 64),
        (OutputQuantization::Mxfp8, Nvfp4Global::Absent, 160, 64),
    ];
    for (format, global, width, capacity) in quantizers {
        stages.push((
            format!("quantize {format:?} {global:?} width {width} capacity {capacity}"),
            Stage::Quantizer(
                row::Quantizer::new(width, capacity, format, global, "lab_v1", c, r).map_err(e)?,
            ),
        ));
    }
    // Gathers of 4 experts, top-2 routes padded to 16, and the activation
    // quantizer at each gather's padded capacity.
    for tokens in [1, 64] {
        for (input, quantized) in [
            (GatherInput::Tokens, false),
            (GatherInput::Tokens, true),
            (GatherInput::Routed, false),
        ] {
            let gather = row::Gather::new(256, tokens, 4, 2, 16, input, quantized, "lab_v1", c, r)
                .map_err(e)?;
            let capacity = gather.stage().capacity();
            stages.push((
                format!("gather {input:?} mxfp8 {quantized} width 256 tokens {tokens}"),
                Stage::Gather(gather),
            ));
            if input == GatherInput::Tokens && !quantized {
                stages.push((
                    format!("quantize Nvfp4 Supplied width 256 capacity {capacity}"),
                    Stage::Quantizer(
                        row::Quantizer::new(
                            256,
                            capacity,
                            OutputQuantization::Nvfp4,
                            Nvfp4Global::Supplied,
                            "lab_v1",
                            c,
                            r,
                        )
                        .map_err(e)?,
                    ),
                ));
            }
        }
    }
    let lookups = [
        (
            WeightFormat::Nvfp4,
            Some(DType::F32),
            37,
            1,
            LookupIds::I32,
            1,
        ),
        (
            WeightFormat::Nvfp4,
            Some(DType::F32),
            37,
            1,
            LookupIds::I32,
            64,
        ),
        (WeightFormat::Bf16, None, 37, 4, LookupIds::U64, 64),
        (
            WeightFormat::Fp8PerTensor,
            Some(DType::Bf16),
            37,
            4,
            LookupIds::U64,
            64,
        ),
        (
            WeightFormat::Fp8PerTensor,
            Some(DType::F32),
            37,
            4,
            LookupIds::U64,
            64,
        ),
    ];
    for (weight, global_scale, vocab, heads, ids, capacity) in lookups {
        stages.push((
            format!("lookup {weight:?} {global_scale:?} {ids:?} heads {heads} capacity {capacity}"),
            Stage::Lookup(
                row::Lookup::new(
                    LookupRequest {
                        vocab,
                        width: 160,
                        heads,
                        capacity,
                        weight,
                        global_scale,
                        ids,
                        shapes: c,
                    },
                    "lab_v1",
                    r,
                )
                .map_err(e)?,
            ),
        ));
    }
    stages.push((
        "copy 12 interleaved columns capacity 64".into(),
        Stage::Copy(
            row::LayoutCopy::new(
                12,
                64,
                CopyRows {
                    row_stride: 12,
                    column_stride: 1,
                    column_offset: 0,
                },
                CopyRows {
                    row_stride: 64,
                    column_stride: 2,
                    column_offset: 1,
                },
                "lab_v1",
                c,
                r,
            )
            .map_err(e)?,
        ),
    ));
    Ok(stages)
}

/// Every pair, in a fixed order: served uses, probes, then lab shapes. A
/// use repeated by several offers (several phases at one capacity) is one pair.
pub fn pairs() -> Result<Vec<Pair>, String> {
    let mut sources = Vec::new();
    let (offers, _, copies) = crate::models::projection_inventory()?;
    for offer in &offers {
        sources.extend(offer_stages(offer).into_iter().map(|s| (Origin::Served, s)));
    }
    sources.extend(copies.into_iter().map(|c| (Origin::Served, Stage::Copy(c))));
    for catalog in crate::models::CATALOGS {
        let (offers, _, copies) = (catalog.row_shape_probes)()?;
        for offer in &offers {
            sources.extend(offer_stages(offer).into_iter().map(|s| (Origin::Probe, s)));
        }
        sources.extend(copies.into_iter().map(|c| (Origin::Probe, Stage::Copy(c))));
    }
    let mut seen = BTreeSet::new();
    let mut pairs = Vec::new();
    for (origin, stage) in sources {
        let label = stage
            .stages()
            .last()
            .expect("row stage")
            .entry()
            .symbol
            .clone();
        let pair = pair(label, origin, stage)?;
        let key = format!(
            "{} {:?}",
            pair.candidate.capacity(),
            pair.candidate.launches(0, 1)?
        );
        if seen.insert(key) {
            pairs.push(pair);
        }
    }
    for (label, stage) in lab_stages()? {
        pairs.push(pair(label, Origin::Lab, stage)?);
    }
    Ok(pairs)
}

pub const FILL: &str = "chop_lab_row_shapes_fill_v1";
pub const STORE: &str = "chop_lab_row_shapes_store_v1";
pub const COUNT: &str = "chop_lab_row_shapes_count_v1";
/// Threads per block of the helpers; each launches at most 1024 blocks.
pub const THREADS: u32 = 256;

/// The helper plan over `bytes` bytes.
pub fn helper_plan(bytes: u64) -> LaunchPlan {
    LaunchPlan {
        grid: [
            bytes.div_ceil(u64::from(THREADS)).clamp(1, 1024) as u32,
            1,
            1,
        ],
        block: [THREADS, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: false,
    }
}

/// The prepared cubins: every distinct entry of both arms, and the
/// device-side fill, store and compare helpers (`cuda/row_shapes.cuh`).
pub fn selected_cubins(pairs: &[Pair]) -> Result<Vec<SelectedCubin>, String> {
    let mut stages = Vec::new();
    let mut symbols = std::collections::BTreeMap::new();
    for pair in pairs {
        for stage in pair
            .reference
            .stages()
            .into_iter()
            .chain(pair.candidate.stages())
        {
            let entry = format!("{:?}", stage.entry());
            match symbols.get(&stage.entry().symbol) {
                Some(known) if *known == entry => {}
                Some(_) => {
                    return Err(format!(
                        "lab row name {} has two bodies",
                        stage.entry().symbol
                    ));
                }
                None => {
                    symbols.insert(stage.entry().symbol.clone(), entry);
                    stages.push(stage);
                }
            }
        }
    }
    let rows = SelectedCubin::new(
        row::cubin_spec("generated_projection_row_shapes_v1", &stages)
            .map_err(|e| e.to_string())?,
        stages.iter().map(|s| s.declaration()).collect(),
    )
    .map_err(|e| e.to_string())?;
    let ptr = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    let entries = vec![
        EntrySpec::new(
            FILL.into(),
            vec![
                ptr("output", "uint8_t"),
                EntryParam::U64 {
                    name: "bytes".into(),
                },
                EntryParam::U32 {
                    name: "value".into(),
                },
            ],
            "chop::lab::row_shapes::fill(output,bytes,value);".into(),
            THREADS,
            1,
        ),
        EntrySpec::new(
            STORE.into(),
            vec![
                ptr("output", "uint32_t"),
                EntryParam::U32 {
                    name: "value".into(),
                },
            ],
            "chop::lab::row_shapes::store(output,value);".into(),
            THREADS,
            1,
        ),
        EntrySpec::new(
            COUNT.into(),
            vec![
                ptr("a", "const uint8_t"),
                ptr("b", "const uint8_t"),
                EntryParam::U64 {
                    name: "bytes".into(),
                },
                ptr("count", "unsigned long long"),
            ],
            "chop::lab::row_shapes::count_differences(a,b,bytes,count);".into(),
            THREADS,
            1,
        ),
    ];
    let declarations = entries
        .iter()
        .map(|entry| EntryDeclaration {
            symbol: entry.symbol.clone(),
            operation: "lab row-shape helper".into(),
            contract: "lab.row-shapes.bytes.v1".into(),
            implementation: entry.symbol.clone(),
            requirements: [("entry".to_owned(), entry.body.clone())].into(),
            correctness_case: None,
        })
        .collect();
    let helpers = SelectedCubin::new(
        CubinSpec {
            name: "lab_row_shape_helpers_v1".into(),
            sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
            include_dirs: vec![],
            headers: vec!["#include \"row_shapes.cuh\"".into()],
            defines: vec![],
            entries,
        },
        declarations,
    )
    .map_err(|e| e.to_string())?;
    Ok(vec![rows, helpers])
}
