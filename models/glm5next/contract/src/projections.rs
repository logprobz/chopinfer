//! The projection declaration (`projections.json`): every checkpoint tensor
//! as a projection, a side tensor of the engine that reads it, or a vision
//! tensor owned by the encoder declaration. Its provider decodes the one
//! registered implementation, the clamped SwiGLU, and executes no math.
use projection_spec::registry::{Provider, Registration, Registry, Target};
use projection_spec::{DType, Dim, Error, Geometry, Variant};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::geometry::FLASH;

pub const DECLARATION: &str = include_str!("../projections.json");

/// Served tensor-parallel degrees.
pub const RANKS: &[u32] = &[4];

/// The SwiGLU input clamps of a gate-up projection, as
/// `kernel_core::gated_mlp::SwigluClamp` declares them: F32 bits of the gate
/// maximum and the symmetric up limit. Routed experts also name the point
/// where the route weight scales the output.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClampedSwiglu {
    pub gate_max_bits: Dim,
    pub up_limit_bits: Dim,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_weight: Option<RouteWeight>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub enum RouteWeight {
    #[serde(rename = "after-down")]
    AfterDown,
}

fn require(condition: bool, reason: &str) -> Result<(), Error> {
    if condition {
        Ok(())
    } else {
        Err(Error(reason.into()))
    }
}

fn clamped_swiglu(v: &Variant, g: &Geometry) -> Result<Value, Error> {
    let p: ClampedSwiglu =
        serde_json::from_value(v.parameters.clone()).map_err(|e| Error(e.to_string()))?;
    for bits in [&p.gate_max_bits, &p.up_limit_bits] {
        let limit = u32::try_from(bits.resolve(g)?)
            .map(f32::from_bits)
            .map_err(|_| Error("SwiGLU clamp is not F32 bits".into()))?;
        require(
            limit.is_finite() && limit > 0.0,
            "SwiGLU clamp limits must be finite and positive",
        )?;
    }
    require(
        v.inputs.len() == 1 && v.outputs.len() == 1,
        "clamped SwiGLU has one activation and one output",
    )?;
    let (input, output) = (&v.inputs[0], &v.outputs[0]);
    require(
        input.id == "activation"
            && input.binding == "activation"
            && output.id == "output"
            && output.binding == "output"
            && input.dtype == DType::Bf16
            && output.dtype == DType::Bf16
            && !input.state
            && !output.state,
        "clamped SwiGLU ports differ from the A16 contract",
    )?;
    require(
        input.shape.len() == 2
            && output.shape.len() == 2
            && input.shape[0] == Dim::Literal(1)
            && output.shape[0] == Dim::Literal(1),
        "clamped SwiGLU ports are single rows",
    )?;
    serde_json::to_value(&p).map_err(|e| Error(e.to_string()))
}

const fn clamped(id: &'static str, numerical_contract: &'static str) -> Registration {
    Registration {
        id,
        numerical_contract,
        target: Target::Projection,
        decoder: clamped_swiglu,
    }
}

/// One registration per source format, since a registration fixes its
/// numerical contract.
pub static PROVIDER: Provider = Provider {
    namespace: "glm5next",
    registrations: &[
        clamped(
            "glm5next/clamped-swiglu-fp8-block128.v1",
            "glm5next.fp8-block128.scale-after-block-mma.clamped-swiglu.v1",
        ),
        clamped(
            "glm5next/clamped-swiglu-bf16.v1",
            "glm5next.bf16.clamped-swiglu.v1",
        ),
        clamped(
            "glm5next/clamped-swiglu-nvfp4.v1",
            "glm5next.nvfp4.scaled-weight.clamped-swiglu.v1",
        ),
        clamped(
            "glm5next/clamped-swiglu-mxfp8.v1",
            "glm5next.mxfp8.scaled-operand.clamped-swiglu.v1",
        ),
    ],
};

/// Construct the explicitly admitted model namespace for declaration parsing.
pub fn registry() -> Result<Registry, Error> {
    let mut registry = Registry::default();
    registry.register(&PROVIDER)?;
    Ok(registry)
}

/// Parse and validate the declaration against the geometry at `tp`.
pub fn declaration(tp: u32) -> Result<projection_spec::Declaration, Error> {
    projection_spec::parse(DECLARATION, &geometry(tp)?, &registry()?)
}

/// The declaration's dimension keys, from the compiled base geometry.
pub fn geometry(tp: u32) -> Result<Geometry, Error> {
    require(RANKS.contains(&tp), "projection geometry serves TP4 only")?;
    let g = &FLASH;
    let s = &g.shape;
    let (l, x, v) = (&s.latent, &s.indexer, &s.vision);
    let dense_layers = (0..g.layers())
        .take_while(|&i| g.mlp_kind(i) == crate::geometry::MlpKind::Dense)
        .count() as u32;
    let swiglu_limit = s.swiglu_limit as f32;
    require(
        f64::from(swiglu_limit) == s.swiglu_limit,
        "the SwiGLU limit is not an F32 value",
    )?;
    let values: [(&str, u32); 44] = [
        ("tp", tp),
        ("hidden", s.hidden),
        ("vocab", s.vocab),
        ("layers", g.layers()),
        ("mtp_end", g.layers() + g.mtp_layers()),
        ("dense_layers", dense_layers),
        ("experts", s.moe.experts),
        ("top_k", s.moe.top_k),
        ("expert_intermediate", s.moe.intermediate),
        ("shared_intermediate", s.moe.shared * s.moe.intermediate),
        ("dense_intermediate", s.dense.intermediate),
        ("swiglu_limit_bits", swiglu_limit.to_bits()),
        ("latent_heads", l.heads),
        ("q_rank", l.q_rank),
        ("kv_rank", l.kv_rank),
        ("kv_a_rows", l.kv_rank + l.qk_rope),
        ("qk_head", g.qk_head()),
        ("latent_q_rows", l.heads * g.qk_head()),
        ("latent_v_width", l.heads * l.v_head),
        ("kv_up_head", l.qk_nope + l.v_head),
        ("kv_up_rows", g.kv_up_rows()),
        ("indexer_heads", x.heads),
        ("indexer_dim", x.dim),
        ("indexer_width", x.heads * x.dim),
        ("index_pool", x.pool),
        ("linear_heads", s.linear.heads),
        ("linear_head_dim", s.linear.head_dim),
        ("linear_width", g.linear_width()),
        ("conv_taps", s.linear.conv_taps),
        ("hc_mix", g.hc_mix()),
        ("hc_width", g.hc_width()),
        ("hc_scales", 3),
        ("mtp_eh_width", 2 * s.hidden),
        ("vision_depth", v.depth),
        ("vision_hidden", v.hidden),
        ("vision_qkv", 3 * v.hidden),
        ("vision_head_dim", g.vision_head_dim()),
        ("vision_intermediate", v.intermediate),
        ("vision_out", v.out),
        ("vision_projection_intermediate", v.projection_intermediate),
        ("vision_channels", v.channels),
        ("vision_temporal_patch", v.temporal_patch),
        ("vision_patch", v.patch),
        ("vision_merge", v.merge),
    ];
    Ok(values
        .into_iter()
        .map(|(key, value)| (key.into(), u64::from(value)))
        .collect())
}
