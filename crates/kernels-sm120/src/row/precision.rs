//! Prepared row stages for quantized operands, independent of model catalogs.

use super::{ptr, static_traits_digest, u32_arg, u64_arg};
use kernel_build::{CubinSpec, EntryDeclaration, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, CompileKey, CompileKnobs, DType, InstantiationSpec, LaunchPlan,
    OpFamily, OutputQuantization, Refusal, ScaleMode, StaticQuery, Target, Validate, WeightFormat,
};
use kernel_core::precision::DynamicScale;

#[derive(Clone, Copy, Debug)]
pub struct RowResources {
    pub target: Target,
    pub threads: u32,
    pub shared_bytes: u32,
    pub pdl: bool,
}

impl RowResources {
    pub const SM120F: Self = Self {
        target: Target::Sm120f,
        threads: 1024,
        shared_bytes: 101_376,
        pdl: true,
    };
}

/// Whether a row stage's shapes are launch arguments or template
/// parameters. Shapes are widths, strides, heads, the vocabulary bound and
/// the routed-row divisor; formats, global-scale recipes, ID types, tiles and
/// accumulation order are compile-time either way, and both forms write the
/// same bits.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Eq,
    PartialEq,
    Ord,
    PartialOrd,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum ShapeBinding {
    /// One entry serves every shape (`row/precision_launch.cuh`); the
    /// builder appends the shapes to its launch arguments.
    Launch,
    /// The shapes' template instance: one entry per shape.
    #[default]
    Compiled,
}

/// A concrete entry, argument schema and capacity plan retained by its consumer.
#[derive(Clone, Debug)]
pub struct RowStage {
    entry: EntrySpec,
    capacity: u32,
    grid: Grid,
    width: u32,
    numerical_contract: &'static str,
    shapes: ShapeBinding,
}

/// A row stage's grid rule: x covers the planned rows or is one block; y
/// counts scale groups, heads or 1. Capacity only sizes the planned grid, so
/// it is not part of the entry's identity or declaration. A launch-shape
/// stage declares the launch argument its y extent follows.
#[derive(Clone, Copy, Debug)]
struct Grid {
    rows: bool,
    y: u32,
    launch_y: Option<&'static str>,
}

impl Grid {
    fn rows(y: u32) -> Self {
        Self {
            rows: true,
            y,
            launch_y: None,
        }
    }
    fn rows_by(y: u32, launch_y: &'static str) -> Self {
        Self {
            launch_y: Some(launch_y),
            ..Self::rows(y)
        }
    }
    const ONE: Self = Self {
        rows: false,
        y: 1,
        launch_y: None,
    };
    fn rule(self, shapes: ShapeBinding) -> serde_json::Value {
        let y = match (shapes, self.launch_y) {
            (ShapeBinding::Launch, Some(argument)) => serde_json::json!(argument),
            _ => serde_json::json!(self.y),
        };
        if self.rows {
            serde_json::json!(["rows", y, 1])
        } else {
            serde_json::json!([1, y, 1])
        }
    }
}

impl RowStage {
    pub(crate) fn rename(&mut self, names: &std::collections::BTreeMap<String, String>) {
        if let Some(name) = names.get(&self.entry.symbol) {
            self.entry.symbol.clone_from(name);
        }
    }
    /// A launch-shape stage's identity names no shape.
    pub(crate) fn identity_declaration(&self) -> serde_json::Value {
        let grid = self.grid.rule(self.shapes);
        match self.shapes {
            ShapeBinding::Compiled => {
                serde_json::json!({"grid":grid,"threads":self.entry.max_threads,"pdl":true,
                    "width":self.width,"contract":self.numerical_contract})
            }
            ShapeBinding::Launch => {
                serde_json::json!({"grid":grid,"threads":self.entry.max_threads,"pdl":true,
                    "shapes":"launch","contract":self.numerical_contract})
            }
        }
    }

    fn new(
        entry: EntrySpec,
        capacity: u32,
        grid: Grid,
        width: u32,
        numerical_contract: &'static str,
        shapes: ShapeBinding,
        resources: RowResources,
    ) -> Result<Self, Refusal> {
        if resources.target != Target::Sm120f
            || !resources.pdl
            || resources.threads < entry.max_threads
            || resources.shared_bytes < 1024
        {
            return Err(Refusal(
                "row stage requires SM120f, PDL and its declared thread/shared bounds",
            ));
        }
        if capacity == 0
            || capacity > i32::MAX as u32
            || width == 0
            || width > i32::MAX as u32 - 2048
        {
            return Err(Refusal(
                "row stage capacity or width exceeds CUDA loop bounds",
            ));
        }
        let stage = Self {
            entry,
            capacity,
            grid,
            width,
            numerical_contract,
            shapes,
        };
        stage.contract().validate()?;
        stage.plan(capacity)?;
        Ok(stage)
    }

    pub fn entry(&self) -> &EntrySpec {
        &self.entry
    }
    pub fn capacity(&self) -> u32 {
        self.capacity
    }
    pub fn numerical_contract(&self) -> &'static str {
        self.numerical_contract
    }
    pub fn shapes(&self) -> ShapeBinding {
        self.shapes
    }
    /// The width the stage was built for; a launch-shape stage passes it at launch.
    pub fn width(&self) -> u32 {
        self.width
    }
    /// A launch-shape stage declares no width: one entry serves every shape.
    pub fn declaration(&self) -> EntryDeclaration {
        let shape = match self.shapes {
            ShapeBinding::Compiled => ("width", self.width.to_string()),
            ShapeBinding::Launch => ("shapes", "launch".into()),
        };
        EntryDeclaration {
            symbol: self.entry.symbol.clone(),
            operation: "precision row".into(),
            contract: self.numerical_contract.into(),
            implementation: "sm120.precision-row.v1".into(),
            requirements: [
                ("target", "sm120f".into()),
                shape,
                ("arguments", format!("{:?}", self.entry.params)),
                ("grid", self.grid.rule(self.shapes).to_string()),
                ("block", format!("{:?}", [self.entry.max_threads, 1, 1])),
                ("pdl", "true".into()),
                ("entry", self.entry.body.clone()),
            ]
            .into_iter()
            .map(|(k, v)| (k.into(), v))
            .collect(),
            correctness_case: None,
        }
    }
    pub fn contract(&self) -> InstantiationSpec {
        InstantiationSpec {
            family: OpFamily::Row,
            key: CompileKey {
                query: StaticQuery::Row {
                    family: "precision-row",
                    width_capacity: self.width,
                    dtype: DType::Bf16,
                    traits_digest: static_traits_digest(&self.entry.body),
                },
                knobs: CompileKnobs {
                    tile_m: 1,
                    tile_n: self.entry.max_threads as u16,
                    tile_k: 1,
                    stages: 1,
                    swizzle: 0,
                    warps: (self.entry.max_threads / 32) as u8,
                    split_k_capacity: 1,
                },
                dense: None,
                target: Target::Sm120f,
            },
            symbol: self.entry.symbol.clone(),
        }
    }
    pub fn plan(&self, live_rows: u32) -> Result<LaunchPlan, Refusal> {
        if live_rows > self.capacity {
            return Err(Refusal("row count exceeds prepared capacity"));
        }
        let plan = LaunchPlan {
            grid: [
                if self.grid.rows { self.capacity } else { 1 },
                self.grid.y,
                1,
            ],
            block: [self.entry.max_threads, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }
    pub fn arguments(&self, values: Vec<Argument>) -> Result<ArgumentList, Refusal> {
        super::pack_arguments(&self.entry.params, values)
    }
}

pub fn cubin_spec(name: &str, stages: &[RowStage]) -> Result<CubinSpec, Refusal> {
    if stages.is_empty() {
        return Err(Refusal("row cubin requires an entry"));
    }
    let mut names = std::collections::BTreeSet::new();
    for stage in stages {
        if !names.insert(&stage.entry.symbol) {
            return Err(Refusal("duplicate row entry symbol"));
        }
    }
    let mut headers = vec![
        "#include \"row/quantize.cuh\"".into(),
        "#include \"row/router.cuh\"".into(),
        "#include \"row/precision.cuh\"".into(),
    ];
    // Only cubins with launch-shape stages include their header, so other
    // cubins keep their source closure.
    if stages.iter().any(|s| s.shapes == ShapeBinding::Launch) {
        headers.push("#include \"row/precision_launch.cuh\"".into());
    }
    if stages
        .iter()
        .any(|s| s.numerical_contract == "quantize.bf16-input.e4m3-ceil-ue8m0-32.codes.v1")
    {
        headers.push(format!("#include \"{}\"", super::block_quantize::HEADER));
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers,
        defines: vec![],
        entries: stages.iter().map(|s| s.entry.clone()).collect(),
    })
}

/// Supplied is a validated checkpoint scalar, already divided by 6 * 448.
/// TensorAmax is a distinct recipe with two explicit reduction launches.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Nvfp4Global {
    Absent,
    Supplied,
    TensorAmax,
}

#[derive(Clone, Copy, Debug)]
pub struct QuantizerBuffers {
    pub input: u64,
    pub codes: u64,
    pub scales: u64,
    pub live_rows: u64,
    pub global_scale: Option<u64>,
    pub row_amax: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct Quantizer {
    stages: Vec<RowStage>,
    global: Nvfp4Global,
    codes_bytes: u64,
    scales_bytes: u64,
    width: u32,
    format: OutputQuantization,
    resources: RowResources,
    scale_rule: Option<DynamicScale>,
}

impl Quantizer {
    pub(crate) fn rename(&mut self, names: &std::collections::BTreeMap<String, String>) {
        for stage in &mut self.stages {
            stage.rename(names);
        }
    }

    pub fn new(
        width: u32,
        capacity: u32,
        format: OutputQuantization,
        global: Nvfp4Global,
        symbol: &str,
        shapes: ShapeBinding,
        resources: RowResources,
    ) -> Result<Self, Refusal> {
        let launch = shapes == ShapeBinding::Launch;
        let (tag, group, packed, scale_bytes, numerical) = match format {
            OutputQuantization::Fp8Group128(ScaleMode::ExactF32) => {
                (0, 128, false, 4, "quantize.fp8.group128.exact-f32.v1")
            }
            OutputQuantization::Fp8Group128(ScaleMode::PowerOfTwoF32) => {
                (1, 128, false, 4, "quantize.fp8.group128.pow2-f32.v1")
            }
            OutputQuantization::Mxfp8 => (2, 32, false, 1, "quantize.mxfp8.group32.ue8m0.v1"),
            OutputQuantization::Nvfp4 => (
                4,
                16,
                true,
                1,
                "quantize.nvfp4.group16.ue4m3.supplied-global.v1",
            ),
            _ => {
                return Err(Refusal(
                    "this activation quantizer supports FP8, MXFP8 and NVFP4",
                ));
            }
        };
        if width == 0
            || width % group != 0
            || (format == OutputQuantization::Nvfp4) != (global != Nvfp4Global::Absent)
        {
            return Err(Refusal(
                "quantizer grouping or global-scale policy is incompatible",
            ));
        }
        let mut stages = Vec::new();
        if global == Nvfp4Global::TensorAmax {
            let stem = symbol
                .strip_suffix("_v1")
                .ok_or(Refusal("quantizer symbol must end in _v1"))?;
            let mut params = vec![
                ptr("input", "const uint16_t"),
                ptr("row_amax", "float"),
                ptr("live_rows", "const uint32_t"),
            ];
            let body = if launch {
                params.push(u32_arg("width"));
                "chop::row::precision::launch::nvfp4_row_amax(input,row_amax,live_rows,width);"
                    .into()
            } else {
                format!("chop::row::nvfp4_row_amax<{width}>(input,row_amax,live_rows);")
            };
            stages.push(RowStage::new(
                EntrySpec::new(format!("{stem}_amax_v1"), params, body, 256, 1),
                capacity,
                Grid::rows(1),
                width,
                "quantize.nvfp4.tensor-amax.v1",
                shapes,
                resources,
            )?);
            stages.push(RowStage::new(
                EntrySpec::new(
                    format!("{stem}_global_v1"),
                    vec![
                        ptr("row_amax", "const float"),
                        ptr("scale", "float"),
                        ptr("live_rows", "const uint32_t"),
                    ],
                    "chop::row::nvfp4_global_scale(row_amax,scale,live_rows);".into(),
                    256,
                    1,
                ),
                capacity,
                Grid::ONE,
                width,
                "quantize.nvfp4.tensor-amax.v1",
                shapes,
                resources,
            )?);
        }
        // The vector variant groups eight scale groups per warp. Use the
        // ordinary group kernel when the final warp would be partial.
        let vector_mx = tag == 2 && width.is_multiple_of(256);
        let mut params = vec![
            ptr("input", "const uint16_t"),
            ptr("codes", "uint8_t"),
            ptr("scales", "void"),
            ptr("global_scale", "const float"),
            ptr("live_rows", "const uint32_t"),
        ];
        let body = match (vector_mx, launch) {
            (true, false) => format!(
                "chop::row::quantize_mxfp8_rows<{width}>(input,codes,reinterpret_cast<uint8_t*>(scales),live_rows);"
            ),
            (false, false) => format!(
                "chop::row::quantize<{width},{tag}>(input,codes,scales,global_scale,live_rows);"
            ),
            (true, true) => "chop::row::precision::launch::quantize_mxfp8_rows(input,codes,reinterpret_cast<uint8_t*>(scales),live_rows,width);".into(),
            (false, true) => format!(
                "chop::row::precision::launch::quantize<{tag}>(input,codes,scales,global_scale,live_rows,width);"
            ),
        };
        if launch {
            params.push(u32_arg("width"));
        }
        stages.push(RowStage::new(
            EntrySpec::new(
                symbol.into(),
                params,
                body,
                if vector_mx { 256 } else { 32 },
                1,
            ),
            capacity,
            if vector_mx {
                Grid::rows(1)
            } else {
                Grid::rows_by(width / group, "width / group")
            },
            width,
            numerical,
            shapes,
            resources,
        )?);
        Ok(Self {
            stages,
            global,
            codes_bytes: u64::from(width) * u64::from(capacity) / if packed { 2 } else { 1 },
            scales_bytes: u64::from(width / group) * u64::from(capacity) * scale_bytes,
            width,
            format,
            resources,
            scale_rule: match format {
                OutputQuantization::Fp8Group128(mode) => Some(DynamicScale::GroupAmax(mode)),
                OutputQuantization::Mxfp8 => Some(DynamicScale::MxPowerOfTwo),
                OutputQuantization::Nvfp4 if global == Nvfp4Global::TensorAmax => {
                    Some(DynamicScale::Nvfp4TensorAmax)
                }
                _ => None,
            },
        })
    }

    /// Choose the declared dynamic scale rule, never infer a ceiling rule
    /// from the MXFP8 storage format. Existing rules retain their entries.
    pub fn with_dynamic_scale(&self, rule: DynamicScale, symbol: &str) -> Result<Self, Refusal> {
        rule.validate()
            .map_err(|_| Refusal("invalid dynamic quantizer scale rule"))?;
        let stage = self.stages.last().expect("quantizer stage");
        let mut selected = Self::new(
            self.width,
            stage.capacity,
            self.format,
            self.global,
            symbol,
            stage.shapes,
            self.resources,
        )?;
        if let (OutputQuantization::Mxfp8, DynamicScale::CeilPowerOfTwo { amax_floor_bits }) =
            (self.format, rule)
        {
            use kernel_core::precision::quantizer::{
                BlockQuantization, BlockQuantizeQuery, QuantizeOutput,
            };
            let entry = super::block_quantize::Entry::new(
                BlockQuantizeQuery {
                    width: self.width,
                    input: DType::Bf16,
                    format: BlockQuantization::E4M3Ceil32 { amax_floor_bits },
                    output: QuantizeOutput::Codes,
                },
                stage.capacity,
                symbol,
            )?;
            selected.stages = vec![RowStage::new(
                entry.spec_with_shapes(stage.shapes),
                stage.capacity,
                Grid::rows_by(self.width / 32, "width / group"),
                self.width,
                "quantize.bf16-input.e4m3-ceil-ue8m0-32.codes.v1",
                stage.shapes,
                self.resources,
            )?];
            selected.scale_rule = Some(rule);
        } else if selected.scale_rule != Some(rule) {
            return Err(Refusal(
                "dynamic scale rule does not match the quantizer format or global recipe",
            ));
        }
        Ok(selected)
    }

    pub fn scale_rule(&self) -> Option<DynamicScale> {
        self.scale_rule
    }

    /// The same quantizer with its shapes bound the other way, under `symbol`.
    pub fn with_shapes(&self, shapes: ShapeBinding, symbol: &str) -> Result<Self, Refusal> {
        let main = self.stages.last().expect("quantizer stage");
        let selected = Self::new(
            self.width,
            main.capacity,
            self.format,
            self.global,
            symbol,
            shapes,
            self.resources,
        )?;
        match self.scale_rule {
            Some(rule) => selected.with_dynamic_scale(rule, symbol),
            None => Ok(selected),
        }
    }
    pub fn stages(&self) -> &[RowStage] {
        &self.stages
    }
    pub fn global(&self) -> Nvfp4Global {
        self.global
    }
    pub fn codes_bytes(&self) -> u64 {
        self.codes_bytes
    }
    pub fn scales_bytes(&self) -> u64 {
        self.scales_bytes
    }
    pub fn scratch_bytes(&self) -> u64 {
        if self.global == Nvfp4Global::TensorAmax {
            (u64::from(self.stages[0].capacity) + 1) * 4
        } else {
            0
        }
    }
    pub fn arguments(&self, b: QuantizerBuffers) -> Result<Vec<ArgumentList>, Refusal> {
        let ptr = Argument::DeviceBuffer;
        match (self.global, b.global_scale, b.row_amax) {
            (Nvfp4Global::Absent, None, None)
            | (Nvfp4Global::Supplied, Some(_), None)
            | (Nvfp4Global::TensorAmax, Some(_), Some(_)) => {}
            _ => {
                return Err(Refusal(
                    "quantizer buffers do not match its supplied or dynamic global recipe",
                ));
            }
        }
        if matches!(self.scale_rule, Some(DynamicScale::CeilPowerOfTwo { .. })) {
            let values = [ptr(b.input), ptr(b.codes), ptr(b.scales), ptr(b.live_rows)]
                .into_iter()
                .chain(
                    (self.stages[0].shapes == ShapeBinding::Launch)
                        .then_some(Argument::U32(self.width)),
                )
                .collect();
            return Ok(vec![self.stages[0].arguments(values)?]);
        }
        let mut values = Vec::new();
        let width =
            (self.stages[0].shapes == ShapeBinding::Launch).then_some(Argument::U32(self.width));
        if self.global == Nvfp4Global::TensorAmax {
            values.push(
                [ptr(b.input), ptr(b.row_amax.unwrap()), ptr(b.live_rows)]
                    .into_iter()
                    .chain(width.clone())
                    .collect(),
            );
            values.push(vec![
                ptr(b.row_amax.unwrap()),
                ptr(b.global_scale.unwrap()),
                ptr(b.live_rows),
            ]);
        }
        values.push(
            [
                ptr(b.input),
                ptr(b.codes),
                ptr(b.scales),
                b.global_scale.map_or(Argument::NullDevicePointer, ptr),
                ptr(b.live_rows),
            ]
            .into_iter()
            .chain(width)
            .collect(),
        );
        self.stages
            .iter()
            .zip(values)
            .map(|(stage, args)| stage.arguments(args))
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GatherInput {
    Tokens,
    Routed,
}

#[derive(Clone, Debug)]
pub struct Gather {
    stage: RowStage,
    quantized: bool,
    request: GatherRequest,
}

/// A gather's construction inputs, retained to bind its shapes again.
#[derive(Clone, Copy, Debug)]
struct GatherRequest {
    tokens: u32,
    experts: u32,
    top_k: u32,
    padding: u32,
    input: GatherInput,
    divisor: u32,
    resources: RowResources,
}

impl Gather {
    pub(crate) fn rename(&mut self, names: &std::collections::BTreeMap<String, String>) {
        self.stage.rename(names);
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "routed layout and output encoding are explicit"
    )]
    pub fn new(
        width: u32,
        tokens: u32,
        experts: u32,
        top_k: u32,
        padding: u32,
        input: GatherInput,
        mxfp8: bool,
        symbol: &str,
        shapes: ShapeBinding,
        resources: RowResources,
    ) -> Result<Self, Refusal> {
        if top_k == 0 || top_k > experts || width % if mxfp8 { 32 } else { 8 } != 0 {
            return Err(Refusal(
                "gather requires valid routing and aligned complete rows",
            ));
        }
        let routed = tokens
            .checked_mul(top_k)
            .ok_or(Refusal("gather route count overflows"))?;
        let capacity = crate::moe::padded_capacity(routed, experts, padding)?;
        let divisor = if input == GatherInput::Tokens {
            top_k
        } else {
            1
        };
        let mut params = vec![
            ptr("input", "const uint16_t"),
            ptr("sorted_rows", "const int32_t"),
        ];
        let launch = shapes == ShapeBinding::Launch;
        let body = match (mxfp8, launch) {
            (true, false) => format!(
                "chop::row::gather_mxfp8<{width},{divisor}>(input,sorted_rows,codes,scales,padded_rows);"
            ),
            (false, false) => format!(
                "chop::row::gather<{width},{divisor}>(input,sorted_rows,output,padded_rows);"
            ),
            (true, true) => "chop::row::precision::launch::gather_mxfp8(input,sorted_rows,codes,scales,padded_rows,width,divisor);".into(),
            (false, true) => "chop::row::precision::launch::gather(input,sorted_rows,output,padded_rows,width,divisor);".into(),
        };
        if mxfp8 {
            params.extend([ptr("codes", "uint8_t"), ptr("scales", "uint8_t")]);
        } else {
            params.push(ptr("output", "uint16_t"));
        }
        params.push(ptr("padded_rows", "const int32_t"));
        if launch {
            params.extend([u32_arg("width"), u32_arg("divisor")]);
        }
        Ok(Self {
            stage: RowStage::new(
                EntrySpec::new(symbol.into(), params, body, 256, 1),
                capacity,
                Grid::rows(1),
                width,
                if mxfp8 {
                    "gather.mxfp8.group32.ue8m0.v1"
                } else {
                    "gather.bf16.exact.v1"
                },
                shapes,
                resources,
            )?,
            quantized: mxfp8,
            request: GatherRequest {
                tokens,
                experts,
                top_k,
                padding,
                input,
                divisor,
                resources,
            },
        })
    }
    /// The same gather with its shapes bound the other way, under `symbol`.
    pub fn with_shapes(&self, shapes: ShapeBinding, symbol: &str) -> Result<Self, Refusal> {
        let r = self.request;
        Self::new(
            self.stage.width,
            r.tokens,
            r.experts,
            r.top_k,
            r.padding,
            r.input,
            self.quantized,
            symbol,
            shapes,
            r.resources,
        )
    }
    pub fn stage(&self) -> &RowStage {
        &self.stage
    }
    pub fn quantized(&self) -> bool {
        self.quantized
    }
    /// Routed rows: a sorted row holds one of them or -1.
    pub fn routes(&self) -> u32 {
        self.request.tokens * self.request.top_k
    }
    /// The input rows the routes address.
    pub fn input_rows(&self) -> u32 {
        self.routes() / self.request.divisor
    }
    pub fn arguments(
        &self,
        input: u64,
        sorted_rows: u64,
        output: u64,
        scales: Option<u64>,
        padded_rows: u64,
    ) -> Result<ArgumentList, Refusal> {
        if self.quantized != scales.is_some() {
            return Err(Refusal(
                "gather scale buffer disagrees with output encoding",
            ));
        }
        let mut args = vec![
            Argument::DeviceBuffer(input),
            Argument::DeviceBuffer(sorted_rows),
            Argument::DeviceBuffer(output),
        ];
        if let Some(scales) = scales {
            args.push(Argument::DeviceBuffer(scales));
        }
        args.push(Argument::DeviceBuffer(padded_rows));
        if self.stage.shapes == ShapeBinding::Launch {
            args.extend([
                Argument::U32(self.stage.width),
                Argument::U32(self.request.divisor),
            ]);
        }
        self.stage.arguments(args)
    }
}

/// Element strides describe an exact BF16 copy, including interleaved columns.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CopyRows {
    pub row_stride: u32,
    pub column_stride: u32,
    pub column_offset: u32,
}

impl CopyRows {
    fn validate(self, columns: u32) -> Result<(), Refusal> {
        if columns == 0
            || self.column_stride == 0
            || self.row_stride == 0
            || (columns - 1)
                .checked_mul(self.column_stride)
                .and_then(|v| v.checked_add(self.column_offset))
                .is_none_or(|last| last >= self.row_stride)
        {
            return Err(Refusal("layout copy columns exceed their row stride"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct LayoutCopy {
    stage: RowStage,
    source: CopyRows,
    destination: CopyRows,
    source_bytes: u64,
    destination_bytes: u64,
    resources: RowResources,
}

/// A launch-shape copy's appended arguments, in order.
const COPY_SHAPES: [&str; 7] = [
    "columns",
    "source_row",
    "source_column",
    "source_offset",
    "dest_row",
    "dest_column",
    "dest_offset",
];

impl LayoutCopy {
    pub fn new(
        columns: u32,
        capacity: u32,
        source: CopyRows,
        destination: CopyRows,
        symbol: &str,
        shapes: ShapeBinding,
        resources: RowResources,
    ) -> Result<Self, Refusal> {
        source.validate(columns)?;
        destination.validate(columns)?;
        let mut params = vec![
            ptr("input", "const uint16_t"),
            ptr("output", "uint16_t"),
            ptr("live_rows", "const uint32_t"),
        ];
        let body = match shapes {
            ShapeBinding::Compiled => format!(
                "chop::row::precision::copy_rows<{columns},{},{},{},{},{},{}>(input,output,live_rows);",
                source.row_stride,
                source.column_stride,
                source.column_offset,
                destination.row_stride,
                destination.column_stride,
                destination.column_offset
            ),
            ShapeBinding::Launch => {
                params.extend(COPY_SHAPES.map(u32_arg));
                format!(
                    "chop::row::precision::launch::copy_rows(input,output,live_rows,{});",
                    COPY_SHAPES.join(",")
                )
            }
        };
        let footprint = |rows: CopyRows| {
            (u64::from(capacity.saturating_sub(1)) * u64::from(rows.row_stride)
                + u64::from(columns - 1) * u64::from(rows.column_stride)
                + u64::from(rows.column_offset)
                + 1)
            .checked_mul(2)
            .ok_or(Refusal("layout copy footprint overflows"))
        };
        Ok(Self {
            source,
            destination,
            source_bytes: footprint(source)?,
            destination_bytes: footprint(destination)?,
            stage: RowStage::new(
                EntrySpec::new(symbol.into(), params, body, 256, 1),
                capacity,
                Grid::rows(1),
                columns,
                "layout-copy.bf16.exact.v1",
                shapes,
                resources,
            )?,
            resources,
        })
    }
    /// The same copy with its shapes bound the other way, under `symbol`.
    pub fn with_shapes(&self, shapes: ShapeBinding, symbol: &str) -> Result<Self, Refusal> {
        Self::new(
            self.stage.width,
            self.stage.capacity,
            self.source,
            self.destination,
            symbol,
            shapes,
            self.resources,
        )
    }
    pub fn layout(&self) -> (u32, CopyRows, CopyRows) {
        (self.stage.width, self.source, self.destination)
    }
    pub fn source_bytes(&self) -> u64 {
        self.source_bytes
    }
    pub fn destination_bytes(&self) -> u64 {
        self.destination_bytes
    }
    pub fn stage(&self) -> &RowStage {
        &self.stage
    }
    pub fn arguments(
        &self,
        input: u64,
        output: u64,
        live_rows: u64,
    ) -> Result<ArgumentList, Refusal> {
        if input == output {
            return Err(Refusal(
                "layout copy requires nonoverlapping input and output",
            ));
        }
        let mut args = vec![
            Argument::DeviceBuffer(input),
            Argument::DeviceBuffer(output),
            Argument::DeviceBuffer(live_rows),
        ];
        if self.stage.shapes == ShapeBinding::Launch {
            let (s, d) = (self.source, self.destination);
            args.extend(
                [
                    self.stage.width,
                    s.row_stride,
                    s.column_stride,
                    s.column_offset,
                    d.row_stride,
                    d.column_stride,
                    d.column_offset,
                ]
                .map(Argument::U32),
            );
        }
        self.stage.arguments(args)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum LookupIds {
    I32,
    U64,
}

#[derive(Clone, Copy, Debug)]
pub struct LookupRequest {
    pub vocab: u64,
    pub width: u32,
    pub heads: u32,
    pub capacity: u32,
    pub weight: WeightFormat,
    pub global_scale: Option<DType>,
    pub ids: LookupIds,
    pub shapes: ShapeBinding,
}

#[derive(Clone, Copy, Debug)]
pub struct LookupBuffers {
    pub table: u64,
    pub ids: u64,
    pub local_scales: Option<u64>,
    pub global_scale: Option<u64>,
    pub output: u64,
    pub live_rows: u64,
    pub row_offset: u64,
    pub shard_rows: u64,
}

#[derive(Clone, Debug)]
pub struct Lookup {
    stage: RowStage,
    request: LookupRequest,
    resources: RowResources,
}

impl Lookup {
    pub(crate) fn rename(&mut self, names: &std::collections::BTreeMap<String, String>) {
        self.stage.rename(names);
    }

    pub fn new(
        request: LookupRequest,
        symbol: &str,
        resources: RowResources,
    ) -> Result<Self, Refusal> {
        let r = request;
        if r.vocab == 0
            || r.heads == 0
            || r.heads > 65535
            || r.vocab
                .checked_mul(u64::from(r.width))
                .is_none_or(|v| v > (1_u64 << 40))
            || r.capacity
                .checked_mul(r.heads)
                .is_none_or(|v| v > i32::MAX as u32)
            || r.ids == LookupIds::I32 && r.vocab > i32::MAX as u64
        {
            return Err(Refusal(
                "lookup table, ID width or head geometry is out of bounds",
            ));
        }
        let (format, scale, numerical) = match (r.weight, r.global_scale) {
            (WeightFormat::Bf16, None) => (0, 0, "lookup.bf16.exact.v1"),
            (WeightFormat::Fp8PerTensor, Some(DType::Bf16)) => {
                (1, 1, "lookup.fp8.bf16-scale.bf16-rne.v1")
            }
            (WeightFormat::Fp8PerTensor, Some(DType::F32)) => {
                (1, 2, "lookup.fp8.f32-scale.bf16-rne.v1")
            }
            (WeightFormat::Nvfp4, Some(DType::F32)) if r.width % 16 == 0 => {
                (2, 2, "lookup.nvfp4.exact-product.bf16-rne.v2")
            }
            _ => {
                return Err(Refusal(
                    "lookup requires BF16, scalar-scale FP8 or group-16 NVFP4 with an F32 global",
                ));
            }
        };
        let id = if r.ids == LookupIds::I32 {
            "int32_t"
        } else {
            "uint64_t"
        };
        let body = match r.shapes {
            ShapeBinding::Compiled => format!(
                "chop::row::precision::lookup<{},{},{},{format},{scale},{id}>(table,ids,local_scales,global_scale,output,live_rows,row_offset,shard_rows);",
                r.width, r.heads, r.vocab,
            ),
            ShapeBinding::Launch => format!(
                "chop::row::precision::launch::lookup<{format},{scale},{id}>(table,ids,local_scales,global_scale,output,live_rows,row_offset,shard_rows,width,heads,vocab);"
            ),
        };
        let mut params = vec![
            ptr("table", "const uint8_t"),
            ptr(
                "ids",
                if r.ids == LookupIds::I32 {
                    "const int32_t"
                } else {
                    "const uint64_t"
                },
            ),
            ptr("local_scales", "const uint8_t"),
            ptr("global_scale", "const void"),
            ptr("output", "uint16_t"),
            ptr("live_rows", "const uint32_t"),
            u64_arg("row_offset"),
            u64_arg("shard_rows"),
        ];
        if r.shapes == ShapeBinding::Launch {
            params.extend([u32_arg("width"), u32_arg("heads"), u64_arg("vocab")]);
        }
        Ok(Self {
            stage: RowStage::new(
                EntrySpec::new(symbol.into(), params, body, 256, 1),
                r.capacity,
                Grid::rows_by(r.heads, "heads"),
                r.width,
                numerical,
                r.shapes,
                resources,
            )?,
            request,
            resources,
        })
    }
    pub fn request(&self) -> LookupRequest {
        self.request
    }
    /// The same lookup with its shapes bound the other way, under `symbol`.
    pub fn with_shapes(&self, shapes: ShapeBinding, symbol: &str) -> Result<Self, Refusal> {
        Self::new(
            LookupRequest {
                shapes,
                ..self.request
            },
            symbol,
            self.resources,
        )
    }
    pub fn input_bytes(&self) -> u64 {
        u64::from(self.request.capacity)
            * u64::from(self.request.heads)
            * if self.request.ids == LookupIds::I32 {
                4
            } else {
                8
            }
    }
    pub fn output_bytes(&self) -> u64 {
        u64::from(self.request.capacity)
            * u64::from(self.request.heads)
            * u64::from(self.request.width)
            * 2
    }
    pub fn stage(&self) -> &RowStage {
        &self.stage
    }
    pub fn arguments(&self, b: LookupBuffers) -> Result<ArgumentList, Refusal> {
        if (self.request.weight == WeightFormat::Nvfp4) != b.local_scales.is_some()
            || self.request.global_scale.is_some() != b.global_scale.is_some()
            || b.shard_rows == 0
            || b.row_offset
                .checked_add(b.shard_rows)
                .is_none_or(|n| n > self.request.vocab)
        {
            return Err(Refusal(
                "lookup scale buffers or shard range disagree with the prepared contract",
            ));
        }
        let ptr = Argument::DeviceBuffer;
        let mut args = vec![
            ptr(b.table),
            ptr(b.ids),
            b.local_scales.map_or(Argument::NullDevicePointer, ptr),
            b.global_scale.map_or(Argument::NullDevicePointer, ptr),
            ptr(b.output),
            ptr(b.live_rows),
            Argument::U64(b.row_offset),
            Argument::U64(b.shard_rows),
        ];
        if self.request.shapes == ShapeBinding::Launch {
            args.extend([
                Argument::U32(self.request.width),
                Argument::U32(self.request.heads),
                Argument::U64(self.request.vocab),
            ]);
        }
        self.stage.arguments(args)
    }
}

#[cfg(test)]
mod capacity_tests {
    use super::*;

    #[test]
    fn copies_cover_the_last_strided_element_and_refuse_overflow() {
        let source = CopyRows {
            row_stride: 4,
            column_stride: 1,
            column_offset: 0,
        };
        let destination = CopyRows {
            row_stride: 16,
            column_stride: 2,
            column_offset: 1,
        };
        let copy = LayoutCopy::new(
            4,
            8,
            source,
            destination,
            "copy_capacity_v1",
            ShapeBinding::Compiled,
            RowResources::SM120F,
        )
        .unwrap();
        assert_eq!(copy.source_bytes(), 64);
        assert_eq!(copy.destination_bytes(), 240);
        let huge = CopyRows {
            row_stride: u32::MAX,
            column_stride: 1,
            column_offset: 0,
        };
        assert!(
            LayoutCopy::new(
                4,
                u32::MAX,
                huge,
                huge,
                "copy_overflow_v1",
                ShapeBinding::Compiled,
                RowResources::SM120F
            )
            .is_err()
        );
    }

    #[test]
    fn lookup_capacity_includes_every_head_and_the_id_width() {
        for (ids, bytes) in [(LookupIds::I32, 64), (LookupIds::U64, 128)] {
            let lookup = Lookup::new(
                LookupRequest {
                    vocab: 256,
                    width: 128,
                    heads: 2,
                    capacity: 8,
                    weight: WeightFormat::Bf16,
                    global_scale: None,
                    ids,
                    shapes: ShapeBinding::Compiled,
                },
                "lookup_capacity_v1",
                RowResources::SM120F,
            )
            .unwrap();
            assert_eq!(lookup.input_bytes(), bytes);
            assert_eq!(lookup.output_bytes(), 4096);
        }
    }
}
