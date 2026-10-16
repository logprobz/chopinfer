//! Model-neutral grouped construction over the shared packed-weight engine.

use super::*;

/// NVFP4 global scale addressing. Local scale bytes always use ScaleTileLayout.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum GlobalScaleLayout {
    Absent,
    Scalar,
    /// Gate/up stores [expert, projection]; down stores [expert].
    ExpertProjection,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScaleBindings {
    pub weight_globals: GlobalScaleLayout,
    pub activation_globals: GlobalScaleLayout,
    /// A4 uses one activation operand for both projections. The preparation
    /// adapter must prove equality of their recipes and calibration contents.
    pub shared_gate_up_activation: bool,
}

impl ScaleBindings {
    pub const NONE: Self = Self {
        weight_globals: GlobalScaleLayout::Absent,
        activation_globals: GlobalScaleLayout::Absent,
        shared_gate_up_activation: true,
    };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PackedContract {
    pub decode_weight_bytes: u16,
    pub scales: ScaleBindings,
}

#[derive(Clone, Debug)]
pub struct GroupedRequest {
    pub shape: Shape,
    pub projection: Projection,
    /// Where the route weight scales the expert output.
    pub route_weight: RouteWeight,
    /// The gate/up projection's SwiGLU input clamps, if any.
    pub swiglu_clamp: Option<SwigluClamp>,
    pub phase: Phase,
    pub tokens_capacity: u32,
    pub weight: WeightFormat,
    pub activation: ActivationFormat,
    pub scales: ScaleBindings,
    pub output_quantization: Option<OutputQuantization>,
    pub swiglu_columns: SwigluColumns,
    pub target: Target,
    pub symbol: String,
    pub knobs: Option<CompileKnobs>,
}

pub fn instantiate(request: GroupedRequest) -> Result<GroupedOperation, Refusal> {
    if request.target != Target::Sm120f
        || request.tokens_capacity == 0
        || request.phase == Phase::Decode && request.tokens_capacity > 32
    {
        return Err(Refusal(
            "grouped request requires SM120f and a supported phase/capacity",
        ));
    }
    if request.shape.hidden == 0
        || request.shape.intermediate == 0
        || request.shape.intermediate.checked_mul(2).is_none()
    {
        return Err(Refusal("invalid grouped projection geometry"));
    }
    let rows = request
        .tokens_capacity
        .checked_mul(request.shape.top_k)
        .ok_or(Refusal("routed row capacity overflow"))?;
    let padded = padded_capacity(rows, request.shape.experts, 16)?;
    let (n, k) = request.shape.projection(request.projection);
    let (swiglu, weighted_moe_combine) = expert_epilogue(
        request.projection,
        request.route_weight,
        request.swiglu_clamp,
    )?;
    if k > i32::MAX as u32
        || padded > i32::MAX as u32
        || request
            .shape
            .experts
            .checked_mul(n)
            .is_none_or(|rows| rows > i32::MAX as u32)
    {
        return Err(Refusal("grouped geometry exceeds signed CUDA index bounds"));
    }
    let query = GemmQuery {
        m_capacity: rows,
        n,
        k,
        weight: request.weight,
        activation: request.activation,
        output: DType::Bf16,
        epilogues: Epilogues {
            swiglu: swiglu.map(|_| request.swiglu_columns),
            output_quantization: request.output_quantization,
            weighted_moe_combine,
            ..Epilogues::default()
        },
        grouped: Some(GroupedGemm {
            experts: request.shape.experts,
            top_k: request.shape.top_k,
            capacity: RoutedCapacity::TotalRows(padded),
        }),
    };
    let knobs = request.knobs.unwrap_or(CompileKnobs {
        tile_m: if request.phase == Phase::Decode {
            16
        } else {
            64
        },
        tile_n: 16,
        tile_k: 32,
        stages: 1,
        swizzle: 0,
        warps: if request.phase == Phase::Decode { 2 } else { 8 },
        split_k_capacity: 1,
    });
    supported(&query, request.phase, knobs, WeightLayout::Packed)?;
    let packed_ring = packed::selects_ring(&query, request.phase, knobs, WeightLayout::Packed);
    if !packed_ring && knobs.tile_m != u16::from(knobs.warps) * 8 {
        return Err(Refusal(
            "packed grouped engine requires eight rows per warp",
        ));
    }
    let operation = GroupedOperation {
        layout: WeightLayout::Packed,
        projection: request.projection,
        phase: request.phase,
        tokens_capacity: request.tokens_capacity,
        path: if request.activation == ActivationFormat::Bf16 {
            InputPath::A16Gather
        } else {
            InputPath::A8Pregather
        },
        query,
        knobs,
        weight_stages: 0,
        packed_contract: Some(PackedContract {
            decode_weight_bytes: staging_bytes(&query, knobs)?,
            scales: request.scales,
        }),
        symbol: request.symbol,
    };
    operation.instantiation().validate()?;
    operation.plan(RuntimeCounts {
        m: request.tokens_capacity,
        batch: 1,
        context: 0,
        routed_rows: rows,
        split_k: knobs.split_k_capacity,
    })?;
    Ok(operation)
}

fn staging_bytes(query: &GemmQuery, knobs: CompileKnobs) -> Result<u16, Refusal> {
    let bytes = query.k
        / if matches!(query.weight, WeightFormat::Nvfp4 | WeightFormat::Mxfp4) {
            2
        } else {
            1
        };
    let split = u32::from(knobs.split_k_capacity);
    [128, 64, 32]
        .into_iter()
        .find(|width| bytes.is_multiple_of(split * u32::from(*width)))
        .ok_or(Refusal(
            "packed grouped rows require a complete 32-byte staging transaction per split",
        ))
}

pub(super) fn validate_packed(
    v: &GroupedOperation<impl AsRef<str>>,
    contract: PackedContract,
) -> Result<(), Refusal> {
    // The packed ring reads the engine's layout with its own schedule knobs.
    let packed_ring = v.packed_ring();
    if packed_ring {
        v.packed_ring_supported()?;
    }
    if v.layout != WeightLayout::Packed
        || !packed_ring && v.knobs.tile_m != u16::from(v.knobs.warps) * 8
        || contract.decode_weight_bytes != staging_bytes(&v.query, v.knobs)?
        || !packed_ring && v.knobs.stages != 1
        || !packed_ring && v.knobs.swizzle != 0
        || v.weight_stages != 0
        || (v.path == InputPath::A16Gather) != (v.query.activation == ActivationFormat::Bf16)
        || (v.projection == Projection::GateUp) != v.query.epilogues.swiglu.is_some()
    {
        return Err(Refusal("unsupported packed grouped staging contract"));
    }
    if v.query.epilogues.bias
        || v.query.epilogues.gelu.is_some()
        || v.projection == Projection::GateUp
            && v.query.n
                % if v.query.epilogues.output_quantization.is_some() {
                    64
                } else {
                    32
                }
                != 0
        || v.query.epilogues.output_quantization.is_some()
            && v.query.epilogues.swiglu == Some(SwigluColumns::Interleaved)
    {
        return Err(Refusal(
            "unsupported packed grouped fusion or incomplete output tile",
        ));
    }
    let scales = contract.scales;
    if matches!(v.query.weight, WeightFormat::Fp8Block128x128(_)) && v.query.n % 128 != 0 {
        return Err(Refusal(
            "packed grouped FP8 requires complete 128-row scale blocks",
        ));
    }
    if v.query.weight == WeightFormat::Nvfp4 {
        if v.knobs.split_k_capacity != 1 {
            return Err(Refusal("new NVFP4 grouped bindings require split-K=1"));
        }
        if !matches!(
            scales.weight_globals,
            GlobalScaleLayout::Scalar | GlobalScaleLayout::ExpertProjection
        ) {
            return Err(Refusal("NVFP4 grouped weights require global scales"));
        }
        match v.query.activation {
            ActivationFormat::Bf16 if scales.activation_globals == GlobalScaleLayout::Absent => {}
            ActivationFormat::Nvfp4
                if scales.weight_globals == GlobalScaleLayout::Scalar
                    && scales.activation_globals == GlobalScaleLayout::Scalar
                    && (v.projection == Projection::Down || scales.shared_gate_up_activation)
                    && v.query.k % 64 == 0 => {}
            _ => {
                return Err(Refusal(
                    "NVFP4 activation execution requires bank-uniform globals, shared gate/up calibration and K64; use the supported BF16 recipe for per-expert scales",
                ));
            }
        }
    } else if scales != ScaleBindings::NONE {
        return Err(Refusal(
            "NVFP4 global scales do not apply to this grouped format",
        ));
    }
    Ok(())
}

/// Consumers own cubin grouping and names. These entries use the common packed
/// engine; specialized fragment entries keep their existing owning adapters.
pub fn cubin_spec(name: &str, operations: &[GroupedOperation]) -> Result<CubinSpec, Refusal> {
    if operations.is_empty() {
        return Err(Refusal("grouped cubin requires at least one operation"));
    }
    let mut symbols = std::collections::BTreeSet::new();
    let mut entries = Vec::new();
    for operation in operations {
        operation.instantiation().validate()?;
        supported(
            &operation.query,
            operation.phase,
            operation.knobs,
            operation.layout,
        )?;
        validate_packed(
            operation,
            operation.packed_contract.ok_or(Refusal(
                "typed grouped cubin requires an explicit packed contract",
            ))?,
        )?;
        if operation.layout != WeightLayout::Packed
            || !operation.packed_ring()
                && operation.knobs.tile_m != u16::from(operation.knobs.warps) * 8
            || operation.query.epilogues.bias
            || operation.query.epilogues.gelu.is_some()
        {
            return Err(Refusal(
                "grouped cubin requires the supported packed engine and epilogues",
            ));
        }
        let entry = EntrySpec::new(
            operation.symbol.clone(),
            parameters(),
            if operation.packed_ring() {
                operation.packed_ring_body()
            } else {
                packed_body(operation)
            },
            u32::from(operation.knobs.warps) * 32,
            1,
        );
        for entry in std::iter::once(entry).chain(reduction_entry(operation)) {
            if !symbols.insert(entry.symbol.clone()) {
                return Err(Refusal("duplicate grouped entry symbol"));
            }
            entries.push(entry);
        }
    }
    // Packed ring entries add the template header; engine-only cubins keep
    // their single header.
    let mut headers = vec!["#include \"moe/moe.cuh\"".to_string()];
    if operations.iter().any(|o| o.packed_ring()) {
        headers.push("#include \"moe/grouped_packed.cuh\"".into());
    }
    if operations.iter().any(|o| o.reduction_symbol().is_some()) {
        headers.push(REDUCE_HEADER.into());
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![PathBuf::from("common"), PathBuf::from("moe")],
        headers,
        defines: vec![],
        entries,
    })
}
