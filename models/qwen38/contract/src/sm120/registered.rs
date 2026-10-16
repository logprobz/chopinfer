//! Qwen's registered legacy projections on SM120: the compiled BF16
//! fusions, FP8 fragment experts and row lookups that serve the registered
//! ports, with their capabilities, adapter stage entries and fusion groups.

use super::{dense, gemm, moe};
use crate::backend::{FusionChoice, FusionMember, LegacyStage, StageChoice};
use crate::contract::*;
use crate::precision::{DenseRole, Projection, projection_capacity};
use kernel_core::contract::{
    ActivationFormat, DType, Epilogues, GemmQuery, Refusal, StaticQuery, Target, WeightFormat,
};
use kernel_core::precision::capability::{
    ActivationCapability, GlobalScaleCapability, OperationCapability,
};
use kernel_core::precision::{
    ArithmeticSpec, DecodeOrder, ExecutionPhase, MatrixGeometry, OperationKind, OperationUse,
    Rounding, WeightLayout,
};
use projection_spec::derive::AdapterValue as Value;
use std::collections::BTreeMap;
use std::sync::OnceLock;

/// A compiled GEMM entry's contract: its GEMM and the output rows of each
/// weight, in weight order (N is their sum).
struct GemmEntry {
    query: GemmQuery,
    segments: Vec<u32>,
}

/// Why a symbol has no compiled GEMM contract.
enum NoGemm {
    /// No catalog compiles an entry of that symbol.
    Absent,
    /// The entry is not a GEMM.
    NotGemm,
}

/// The compiled GEMM entry `symbol`: an owned fused entry, else a dense
/// catalog entry.
fn gemm_entry(symbol: &str) -> Result<GemmEntry, NoGemm> {
    static FUSED: OnceLock<Vec<kernels_sm120::gemm::fused::Entry>> = OnceLock::new();
    static DENSE: OnceLock<Vec<kernels_sm120::gemm::GemmVariant>> = OnceLock::new();
    if let Some(entry) = FUSED
        .get_or_init(gemm::fused_catalog)
        .iter()
        .find(|e| e.symbol == symbol)
    {
        return Ok(GemmEntry {
            query: entry.query,
            segments: entry.segments.clone(),
        });
    }
    let entry = DENSE
        .get_or_init(gemm::catalog)
        .iter()
        .find(|e| e.symbol == symbol)
        .ok_or(NoGemm::Absent)?;
    let StaticQuery::Gemm(query) = entry.key.query else {
        return Err(NoGemm::NotGemm);
    };
    Ok(GemmEntry {
        query,
        segments: vec![query.n],
    })
}

/// The entry a registered BF16 dense `role` serves `capacity` rows of
/// `shape` through.
fn dense_symbol(shape: Shape, role: DenseRole, capacity: u32) -> String {
    let decode = capacity <= DECODE_ENTRY_ROWS;
    let (tag, operation) = if decode {
        let name = match role {
            DenseRole::LinearQkv | DenseRole::LinearZ | DenseRole::LinearA | DenseRole::LinearB => {
                "lin_in"
            }
            DenseRole::AttentionQ
            | DenseRole::AttentionK
            | DenseRole::AttentionV
            | DenseRole::IndexQk => "attn_in",
            DenseRole::Output | DenseRole::AttentionOutput => "out",
            DenseRole::SharedGateUp => "shared_gate_up",
            DenseRole::SharedDown => "shared_down",
        };
        (shape.tag(), format!("{name}_c{capacity}"))
    } else {
        let name = match role {
            DenseRole::LinearQkv => "lin_qkv_prefill",
            DenseRole::LinearZ => "lin_z_prefill",
            DenseRole::LinearA | DenseRole::LinearB => "lin_ab",
            DenseRole::AttentionQ | DenseRole::AttentionK | DenseRole::AttentionV => {
                "attn_qkv_prefill"
            }
            DenseRole::IndexQk => "index_qk_prefill",
            DenseRole::Output | DenseRole::AttentionOutput => "out_prefill",
            DenseRole::SharedGateUp => "shared_gate_up_prefill",
            DenseRole::SharedDown => "shared_down_prefill",
        };
        (
            if role == DenseRole::IndexQk {
                shape.geometry.kernel_tag
            } else {
                shape.tag()
            },
            name.into(),
        )
    };
    let (tag, operation) = match role {
        DenseRole::Output | DenseRole::AttentionOutput => {
            let role = if role == DenseRole::Output {
                dense::Role::LinearOutput
            } else {
                dense::Role::AttentionOutput
            };
            let (profile, name) = dense::alias(shape.profile(), role);
            (
                profile.tag(),
                if decode {
                    format!("{name}_c{capacity}")
                } else {
                    format!("{name}_prefill")
                },
            )
        }
        _ => (tag, operation),
    };
    format!("chop_gemm_{tag}_{operation}_v1")
}

/// The grouped expert entry the registered FP8 fragment adapter of
/// `projection` serves `capacity` rows of `shape` through: its symbol and
/// GEMM.
fn fragment_expert(
    shape: Shape,
    projection: Projection,
    capacity: u32,
) -> Result<(String, GemmQuery), Refusal> {
    static VARIANTS: OnceLock<Vec<kernels_sm120::moe::Variant>> = OnceLock::new();
    let projection = match projection {
        Projection::GateUp => kernels_sm120::moe::Projection::GateUp,
        Projection::Down => kernels_sm120::moe::Projection::Down,
    };
    let phase = if capacity <= DECODE_ENTRY_ROWS {
        kernels_sm120::moe::Phase::Decode
    } else {
        kernels_sm120::moe::Phase::Prefill
    };
    let variant = VARIANTS
        .get_or_init(moe::catalog)
        .iter()
        .find(|v| {
            v.profile == shape.profile()
                && v.projection == projection
                && v.phase == phase
                && v.tokens_capacity == capacity
        })
        .ok_or(Refusal("missing legacy Qwen expert phase/capacity"))?;
    Ok((variant.symbol.clone(), variant.query))
}

/// The capability of the compiled BF16 fusion that serves a registered
/// dense `role` at `usage`: the legacy fused entry, checked against the
/// role's declared fusion schedule.
pub fn dense_capability(
    shape: Shape,
    role: DenseRole,
    usage: OperationUse,
) -> Result<OperationCapability, Refusal> {
    let capacity = projection_capacity(usage)?;
    let decode = capacity <= DECODE_ENTRY_ROWS;
    let symbol = dense_symbol(shape, role, capacity);
    let entry = gemm_entry(&symbol).map_err(|missing| match missing {
        NoGemm::Absent => Refusal("missing Qwen legacy dense implementation"),
        NoGemm::NotGemm => Refusal("legacy dense entry has no GEMM contract"),
    })?;
    let (query, segments) = (entry.query, entry.segments);
    let geometry = role.logical_geometry(shape)?;
    let epilogues = role.epilogues(shape)?;
    let segments_expected = role.legacy_segments(shape, decode, geometry)?;
    if segments != segments_expected
        || query.k != geometry.k as u32
        || query.weight != WeightFormat::Bf16
        || query.activation != ActivationFormat::Bf16
        || query.output != DType::Bf16
        || query.epilogues != epilogues
    {
        return Err(Refusal(
            "legacy dense projection port differs from its compiled fusion",
        ));
    }
    let capability = OperationCapability {
        kind: OperationKind::Dense,
        target: Target::Sm120f,
        usage,
        geometry,
        operands: 1 + u8::from(role == DenseRole::SharedGateUp),
        weight: WeightFormat::Bf16,
        activation: Some(ActivationCapability::Bf16),
        epilogues,
        layout: WeightLayout {
            id: "bf16-fragment".into(),
            version: 1,
        },
        local_scale_dtypes: vec![],
        globals: GlobalScaleCapability::Absent,
        shared_local_scale: false,
        arithmetic: ArithmeticSpec {
            weight_operand: DType::Bf16,
            accumulator: DType::F32,
            output: DType::Bf16,
            decode_order: DecodeOrder::ScaleThenCode,
            rounding: Rounding::NearestEven,
            numerical_contract: "qwen38.bf16-projection.v1".into(),
        },
        implementation: role.legacy_implementation(shape),
        stages: vec![symbol],
    };
    capability
        .validate()
        .map_err(|_| Refusal("invalid legacy dense capability"))?;
    Ok(capability)
}

/// The existing Qwen fragment expert engine applies each block scale after
/// its FP32 block accumulation: the capability of the grouped entry that
/// serves a registered fragment expert `projection` at `usage`.
pub fn expert_capability(
    shape: Shape,
    projection: Projection,
    usage: OperationUse,
) -> Result<OperationCapability, Refusal> {
    let top_k = shape.top_k();
    let capacity = projection_capacity(usage)?;
    let (symbol, query) = fragment_expert(shape, projection, capacity)?;
    let operands = if query.epilogues.swiglu.is_some() {
        2
    } else {
        1
    };
    let numerical_contract = "qwen38.experts.fp8-fragment.bf16.block-scale-after-mma.v1";
    let result = OperationCapability {
        kind: OperationKind::Grouped {
            experts: shape.local_experts,
            top_k,
        },
        target: Target::Sm120f,
        usage,
        geometry: MatrixGeometry {
            rows: u64::from(query.n / operands),
            k: query.k.into(),
        },
        operands: operands as u8,
        weight: query.weight,
        activation: Some(ActivationCapability::Bf16),
        epilogues: query.epilogues,
        layout: WeightLayout {
            id: "fp8-fragment".into(),
            version: 1,
        },
        local_scale_dtypes: vec![DType::Bf16, DType::F32],
        globals: GlobalScaleCapability::Absent,
        shared_local_scale: false,
        arithmetic: ArithmeticSpec {
            weight_operand: DType::Bf16,
            accumulator: DType::F32,
            output: DType::Bf16,
            decode_order: DecodeOrder::BlockScaleAfterMma,
            rounding: Rounding::NearestEven,
            numerical_contract: numerical_contract.into(),
        },
        implementation: numerical_contract.into(),
        stages: vec![symbol],
    };
    result
        .validate()
        .map_err(|_| Refusal("invalid Qwen legacy expert capability"))?;
    Ok(result)
}

/// The capability of the compiled row entry that serves an embedding or
/// n-gram lookup at `usage`.
pub fn lookup_capability(
    shape: Shape,
    ngram: bool,
    usage: OperationUse,
) -> Result<OperationCapability, Refusal> {
    let hidden = shape.hidden();
    let ngram_head_width = shape.ngram_head_width();
    let vocab = shape.vocab();
    projection_capacity(usage)?;
    let rows = if ngram {
        crate::contract::ngram_tables_for(shape.geometry)
            .map_err(|_| Refusal("Qwen n-gram table geometry"))?
            .rows
    } else {
        vocab.into()
    };
    let stage = if ngram {
        match usage.phase {
            ExecutionPhase::Prefill => format!("chop_row_{}_ngram_v1", shape.tag()),
            ExecutionPhase::Decode => format!("chop_row_{}_ngram_decode_v1", shape.tag()),
            ExecutionPhase::Verify => format!("chop_{}_ngram_verify_v1", shape.tag()),
        }
    } else {
        format!("chop_row_embedding_{}_v1", shape.geometry.kernel_tag)
    };
    let result = OperationCapability {
        kind: OperationKind::Lookup,
        target: Target::Sm120f,
        usage,
        geometry: MatrixGeometry {
            rows,
            k: if ngram {
                ngram_head_width.into()
            } else {
                hidden.into()
            },
        },
        operands: 1,
        weight: if ngram {
            WeightFormat::Fp8PerTensor
        } else {
            WeightFormat::Bf16
        },
        activation: None,
        epilogues: Epilogues::default(),
        layout: WeightLayout::row_major(),
        local_scale_dtypes: if ngram { vec![DType::Bf16] } else { vec![] },
        globals: GlobalScaleCapability::Absent,
        shared_local_scale: false,
        arithmetic: ArithmeticSpec {
            weight_operand: DType::Bf16,
            accumulator: DType::F32,
            output: DType::Bf16,
            decode_order: DecodeOrder::CodeThenScale,
            rounding: Rounding::NearestEven,
            numerical_contract: "qwen38.legacy-lookup.v1".into(),
        },
        implementation: format!(
            "qwen38.legacy-lookup.{}.tp{}.v1",
            if ngram { "ngram" } else { "embedding" },
            shape.ranks()
        ),
        stages: vec![stage],
    };
    result
        .validate()
        .map_err(|_| Refusal("invalid Qwen lookup capability"))?;
    Ok(result)
}

/// The entry and entry facts of a legacy adapter `stage` at `usage`: the
/// fragment experts' route-weight point and scale order, the gated output
/// projection's compiled entry, or the stream mix's up entry (decode or
/// prefill) with its mix entries and the 16- and 32-row buckets that run
/// the precomputed mix. Fused dense entries add their native capacity.
pub fn legacy_stage(
    shape: Shape,
    stage: LegacyStage,
    usage: OperationUse,
) -> Result<StageChoice, String> {
    let mut facts = BTreeMap::new();
    let symbol = match stage {
        LegacyStage::Experts(projection) => {
            let native = expert_capability(shape, projection, usage).map_err(|e| e.to_string())?;
            facts.insert("route-weight".into(), Value::Text("after-down".into()));
            facts.insert(
                "scale-order".into(),
                Value::Text("block-scale-after-mma".into()),
            );
            return Ok(StageChoice {
                symbol: native.stages[0].clone(),
                facts,
            });
        }
        LegacyStage::GatedOutput => dense_capability(shape, DenseRole::Output, usage)
            .map_err(|e| e.to_string())?
            .stages[0]
            .clone(),
        LegacyStage::StreamMix => {
            let capacity = projection_capacity(usage).map_err(|e| e.to_string())?;
            facts.insert("mix-symbol".into(), Value::Text(gemm::HC_UP_MIX.into()));
            facts.insert(
                "precompute-capacities".into(),
                Value::List([16, 32].map(Value::U64).into()),
            );
            facts.insert(
                "precomputed-mix-symbol".into(),
                Value::Text(kernels_sm120::gemm::fused::HC_PRECOMPUTED_MIX.into()),
            );
            facts.insert(
                "padded-mix-symbol".into(),
                Value::Text(kernels_sm120::gemm::fused::HC_PADDED_MIX.into()),
            );
            let suffix = if capacity <= DECODE_ENTRY_ROWS {
                "hc_up_decode"
            } else {
                "hc_up_prefill"
            };
            format!("chop_gemm_{}_{suffix}_v1", shape.geometry.kernel_tag)
        }
    };
    let native = fusion_entry(&symbol)?;
    facts.insert(
        "native-capacity".into(),
        Value::U64(native.query.m_capacity.into()),
    );
    Ok(StageChoice { symbol, facts })
}

/// The compiled entry of the registered input fusion group led by
/// `member` at `usage`: a dense role's fused entry, or the hyper-connection
/// input entry of the decode bucket, the block inject or the prefill
/// bottleneck.
pub fn legacy_fusion(
    shape: Shape,
    member: FusionMember<'_>,
    usage: OperationUse,
) -> Result<FusionChoice, String> {
    let symbol = match member {
        FusionMember::Dense(role) => dense_capability(shape, role, usage)
            .map_err(|e| e.to_string())?
            .stages[0]
            .clone(),
        FusionMember::HyperConnection(member) => {
            let capacity = projection_capacity(usage).map_err(|e| e.to_string())?;
            let suffix = if capacity <= DECODE_ENTRY_ROWS {
                format!("hc_in_c{capacity}")
            } else if member.ends_with("block_inject_weight") {
                "hc_inject".into()
            } else if member.ends_with("input_mix_weight_down") {
                "hc_down_prefill".into()
            } else {
                return Err("unregistered HC fusion member".into());
            };
            format!("chop_gemm_{}_{suffix}_v1", shape.geometry.kernel_tag)
        }
    };
    let entry = fusion_entry(&symbol)?;
    Ok(FusionChoice {
        symbol,
        segments: entry.segments,
        k: entry.query.k,
        native_capacity: entry.query.m_capacity,
    })
}

/// The compiled GEMM entry of a legacy native `symbol`.
fn fusion_entry(symbol: &str) -> Result<GemmEntry, String> {
    gemm_entry(symbol).map_err(|missing| match missing {
        NoGemm::Absent => format!("legacy native entry `{symbol}` is absent"),
        NoGemm::NotGemm => "legacy native entry has no GEMM query".into(),
    })
}
