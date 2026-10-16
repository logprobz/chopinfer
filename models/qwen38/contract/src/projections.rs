//! Draft projection contracts. Providers decode parameters but execute no math.
use projection_spec::registry::{
    Provider, Registration, Registry, Target, canonical, require, require_ports, typed,
};
use projection_spec::{DType, Dim, Error, Geometry, Variant};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub mod layouts;
pub mod ports;
pub mod registered;
pub mod support;
// Production never derives a Qwen graph; the deriver proves the declaration
// against native packing in tests.
#[cfg(test)]
mod variants;

pub const DECLARATION: &str = include_str!("../projections.json");

/// The exact source geometry and reduction identity of a legacy dense port.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseParameters {
    pub operation: String,
    pub rows: Dim,
    pub k: Dim,
    pub scale_order: ScaleOrder,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_gate: Option<InputGate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_mix: Option<StreamMix>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ScaleOrder {
    ScaleThenCode,
    BlockScaleAfterMma,
    ExactScaledOperand,
}

/// Expert globals, scales and route weighting must retain their old order.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertParameters {
    pub experts: Dim,
    pub top_k: Dim,
    pub scale_order: ScaleOrder,
    pub route_weight: RouteWeight,
}
#[derive(Debug, Serialize, Deserialize)]
pub enum RouteWeight {
    #[serde(rename = "after-down")]
    AfterDown,
}

fn dense_output(v: &Variant, g: &Geometry, output: DType) -> Result<Value, Error> {
    let p: DenseParameters = typed(v)?;
    require(
        projection_spec::registry::valid_name(&p.operation),
        "invalid dense operation",
    )?;
    p.rows.resolve(g)?;
    p.k.resolve(g)?;
    if let Some(gate) = &p.input_gate {
        require(
            p.operation == "linear.output" && gate.projection == "linear.z",
            "invalid legacy gated input",
        )?;
    }
    if let Some(mix) = &p.stream_mix {
        require(
            p.operation.ends_with("input_mix_weight_up")
                && mix.streams.resolve(g)? == g["streams"]
                && mix.mean_divisor.resolve(g)? == g["streams"],
            "invalid HC stream arithmetic",
        )?;
    }
    require_ports(v, &[("activation", DType::Bf16)], &[("output", output)])?;
    require(
        v.inputs[0].shape == vec![Dim::Literal(1), p.k.clone()]
            && v.outputs[0].shape == vec![Dim::Literal(1), p.rows.clone()],
        "dense ports differ from parameter geometry",
    )?;
    canonical(&p)
}
fn dense(v: &Variant, g: &Geometry) -> Result<Value, Error> {
    dense_output(v, g, DType::Bf16)
}
fn experts(v: &Variant, g: &Geometry) -> Result<Value, Error> {
    let p: ExpertParameters = typed(v)?;
    require(
        p.top_k.resolve(g)? <= p.experts.resolve(g)?,
        "top-k exceeds expert count",
    )?;
    require_ports(
        v,
        &[("activation", DType::Bf16)],
        &[("output", DType::Bf16)],
    )?;
    canonical(&p)
}

/// Construct an explicitly admitted model namespace for declaration parsing.
pub fn registry() -> Result<Registry, Error> {
    let mut registry = Registry::default();
    registry.register(&PROVIDER)?;
    Ok(registry)
}

/// Parse the draft without changing any served operation or artifact.
pub fn declaration(tp: u32) -> Result<projection_spec::Declaration, Error> {
    let ranks = crate::contract::Ranks::new(tp)
        .ok_or_else(|| Error("unserved projection rank count".into()))?;
    declaration_for(crate::contract::Shape::flash(ranks))
}

/// Expand family ordinals through the owning geometry's layer selectors.
/// Sparse lists are represented by geometry keys before shared validation.
pub fn declaration_for(
    shape: crate::contract::Shape,
) -> Result<projection_spec::Declaration, Error> {
    let geometry = geometry_for(shape)?;
    let mut declaration: projection_spec::Declaration =
        serde_json::from_str(DECLARATION).map_err(|e| Error(e.to_string()))?;
    // The declared drafter is the MTP layer; a geometry without one omits it.
    if shape.mtp_layers() == 0 {
        for drafter in declaration.drafters.clone() {
            declaration.omit_drafter(&drafter.id)?;
        }
    }
    for stem in &mut declaration.layer_stems {
        if let projection_spec::LayerIndices::Range {
            start,
            end: Dim::Key(key),
        } = &stem.indices
            && matches!(key.as_str(), "linear_layers" | "full_layers" | "ple_layers")
        {
            let family = key.trim_end_matches("_layers");
            stem.indices = projection_spec::LayerIndices::List {
                indices: (start.resolve_offset(&geometry)?..geometry[key])
                    .map(|i| Dim::Key(format!("{family}-layer-{i}")))
                    .collect(),
            };
        }
    }
    projection_spec::parse(
        &serde_json::to_string(&declaration).map_err(|e| Error(e.to_string()))?,
        &geometry,
        &registry()?,
    )
}

/// Phase adapters preserve the hash's wrapping integer and history contract.
/// I32/I64 ports carry the same storage bits as CUDA uint32_t/uint64_t.
/// Hash arithmetic remains unsigned; the descriptors introduce no casts.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HashParameters {
    pub orders: u32,
    pub heads: u32,
    pub heads_per_order: u32,
    pub history_tokens: u32,
    pub eos: u32,
    pub integer_arithmetic: HashArithmetic,
    pub phase_addressing: Vec<HashAddressing>,
}
#[derive(Debug, Serialize, Deserialize)]
pub enum HashArithmetic {
    #[serde(rename = "wrapping-u64-xor")]
    WrappingU64Xor,
}
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum HashAddressing {
    PrefillContiguous,
    DecodeItemStride,
    VerifyCuRows,
}
fn hash(v: &Variant, _g: &Geometry) -> Result<Value, Error> {
    let p: HashParameters = typed(v)?;
    require(
        p.orders == 3
            && p.heads == 16
            && p.heads_per_order == 8
            && p.history_tokens == 2
            && p.eos == crate::contract::NGRAM_EOS,
        "hash parameters differ from the served integer contract",
    )?;
    require(
        p.phase_addressing
            == [
                HashAddressing::PrefillContiguous,
                HashAddressing::DecodeItemStride,
                HashAddressing::VerifyCuRows,
            ],
        "hash addressing must cover all phases in order",
    )?;
    require_ports(
        v,
        &[
            ("tokens", DType::I32),
            ("history", DType::I32),
            ("multipliers", DType::I64),
            ("sizes", DType::I64),
            ("offsets", DType::I64),
        ],
        &[("ids", DType::I64), ("history", DType::I32)],
    )?;
    require(
        v.inputs[1].state
            && v.outputs[1].state
            && v.inputs[1].shape == vec![Dim::Literal(2)]
            && v.outputs[1].shape == v.inputs[1].shape,
        "hash history must be a two-token state",
    )?;
    require(
        v.inputs
            .iter()
            .chain(&v.outputs)
            .filter(|p| matches!(p.dtype, DType::I32 | DType::I64))
            .all(|p| {
                p.interpretation == Some(projection_spec::contracts::Interpretation::Unsigned)
            }),
        "hash integer ports require unsigned interpretation",
    )?;
    canonical(&p)
}

pub static PROVIDER: Provider = Provider {
    namespace: "qwen38",
    registrations: &[
        Registration {
            id: "qwen38/legacy-input-fusion.v1",
            numerical_contract: "qwen38.bf16-projection.v1",
            target: Target::FusionGroup,
            decoder: input_fusion,
        },
        Registration {
            id: "qwen38/ngram-hash.v1",
            numerical_contract: "qwen38.ngram-hash.v1",
            target: Target::Stage,
            decoder: hash,
        },
        Registration {
            id: "qwen38/legacy-fused-dense.v1",
            numerical_contract: "qwen38.bf16-projection.v1",
            target: Target::Projection,
            decoder: dense,
        },
        Registration {
            id: "qwen38/legacy-fragment-experts.v1",
            numerical_contract: "qwen38.experts.fp8-fragment.bf16.block-scale-after-mma.v1",
            target: Target::Projection,
            decoder: experts,
        },
    ],
};

pub fn geometry(tp: u32) -> Result<Geometry, Error> {
    let ranks = crate::contract::Ranks::new(tp)
        .ok_or_else(|| Error("unserved projection rank count".into()))?;
    geometry_for(crate::contract::Shape::flash(ranks))
}

pub fn geometry_for(shape: crate::contract::Shape) -> Result<Geometry, Error> {
    let tp = shape.ranks();
    let g = shape.geometry;
    let mut values: Geometry = [
        ("tp", tp),
        ("hidden", g.hidden()),
        ("wide", g.hidden() * g.streams()),
        ("streams", g.streams()),
        ("hc_rank", g.hc_rank()),
        ("experts", g.experts()),
        ("top_k", g.top_k()),
        ("expert_intermediate", g.expert_intermediate()),
        ("shared_intermediate", g.shared_expert_intermediate()),
        ("vocab", g.vocab()),
        (
            "linear_value",
            g.linear_value_heads() * g.linear_value_head(),
        ),
        ("linear_heads", g.linear_value_heads()),
        ("ab_stride", shape.linear_ab_stride),
        (
            "ab_mxfp8_rows",
            (g.linear_value_heads() / tp).next_multiple_of(16),
        ),
        ("ngram_width", g.ngram_head_width()),
        ("ngram_parts", g.ngram_parts()),
        ("layers", g.layers()),
        ("mtp_layers", g.mtp_layers()),
        ("vision_layers", g.vision_blocks()),
        ("vision_hidden", g.vision_hidden()),
        ("vision_intermediate", g.vision_intermediate()),
        ("vision_out", g.vision_out()),
        ("vision_positions", g.vision_positions()),
        ("vision_merge", g.vision_merge()),
        ("vision_channels", g.vision_channels()),
        ("vision_temporal_patch", g.vision_temporal_patch()),
        ("vision_patch", g.vision_patch()),
        ("linear_key_heads", g.linear_key_heads()),
        ("linear_head", g.linear_head()),
        ("linear_value_head", g.linear_value_head()),
        ("attention_heads", g.attention_q_heads()),
        ("attention_kv_heads", g.attention_kv_heads()),
        ("attention_head", g.head_dim()),
        ("attention_value", g.attention_q_heads() * g.head_dim()),
        ("index_row", shape.index_row()),
        ("index_dim", g.index_dim()),
        ("conv_taps", g.conv_taps()),
        ("ple_conv_taps", g.ple_conv_taps()),
    ]
    .into_iter()
    .map(|(k, v)| (k.into(), u64::from(v)))
    .collect();
    for (family, indices) in [
        (
            "linear",
            (0..g.layers())
                .filter(|&i| !g.is_full(i))
                .collect::<Vec<_>>(),
        ),
        ("full", (0..g.layers()).filter(|&i| g.is_full(i)).collect()),
        ("ple", g.ple_layer_ids().iter().map(|i| i - 1).collect()),
    ] {
        values.insert(format!("{family}_layers"), indices.len() as u64);
        for (ordinal, layer) in indices.into_iter().enumerate() {
            values.insert(format!("{family}-layer-{ordinal}"), u64::from(layer));
        }
    }
    let table = crate::contract::ngram_tables_for(g).map_err(|e| Error(e.to_string()))?;
    values.insert("ngram_rows".into(), table.rows);
    values.insert(
        "ngram_shard_rows".into(),
        table.rows / u64::from(g.ngram_parts()),
    );
    for (index, (&offset, &size)) in table.offsets.iter().zip(&table.sizes).enumerate() {
        values.insert(format!("ngram-start-{index}"), offset);
        values.insert(format!("ngram-size-{index}"), size);
    }
    Ok(values)
}

/// Legacy fused input schedules retain their prefill/decode arithmetic order.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputFusionParameters {
    pub prefill_groups: Vec<Vec<String>>,
    pub decode_group: Vec<String>,
}
fn input_fusion(v: &Variant, _g: &Geometry) -> Result<Value, Error> {
    let p: InputFusionParameters = typed(v)?;
    let members: std::collections::BTreeSet<_> = p.decode_group.iter().collect();
    require(
        !members.is_empty() && members.len() == p.decode_group.len(),
        "fusion repeats or omits members",
    )?;
    require(
        p.prefill_groups
            .iter()
            .flatten()
            .collect::<std::collections::BTreeSet<_>>()
            == members,
        "prefill and decode membership differ",
    )?;
    require_ports(v, &[("activation", DType::Bf16)], &[]).or_else(|_| {
        require(
            v.inputs.len() == 1
                && v.inputs[0].dtype == DType::Bf16
                && !v.outputs.is_empty()
                && v.outputs.iter().all(|p| p.dtype == DType::Bf16),
            "legacy input fusion port dtypes differ",
        )
    })?;
    canonical(&p)
}

/// These are legacy input arithmetic facts, not normalization geometry.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputGate {
    pub projection: String,
    pub activation: Sigmoid,
    pub order: NormThenGate,
}
#[derive(Debug, Serialize, Deserialize)]
pub enum Sigmoid {
    #[serde(rename = "sigmoid")]
    Sigmoid,
}
#[derive(Debug, Serialize, Deserialize)]
pub enum NormThenGate {
    #[serde(rename = "norm-then-gate")]
    NormThenGate,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamMix {
    pub streams: Dim,
    pub mean_divisor: Dim,
    pub activation: Sigmoid,
}
