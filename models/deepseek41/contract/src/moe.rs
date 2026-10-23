//! Routed expert contracts. Every rank owns the full expert set and a
//! tensor-parallel slice of each expert's intermediate channels.

use crate::geometry::Geometry;
use kernel_core::contract::{
    ActivationFormat, DType, Epilogues, GemmQuery, GroupedGemm, OutputQuantization, RoutedCapacity,
    SwigluColumns, WeightFormat,
};
use kernel_core::gated_mlp::quantized::SiluMulQuantizeQuery;
use kernel_core::gated_mlp::{
    GatedMlpProjection, GatedMlpProjectionQuery, RouteWeightPoint, SwigluClamp,
};
use kernel_core::moe_chained::{ChainedK, ChainedMlpProjectionQuery};
use kernel_core::precision::quantizer::{BlockQuantization, BlockQuantizeQuery, QuantizeOutput};
use kernel_core::router::RowBiasRouterQuery;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Experts {
    pub count: u32,
    pub top_k: u32,
}

pub fn expert_sets(g: &Geometry) -> Vec<Experts> {
    let main = Experts {
        count: g.shape.moe.experts,
        top_k: g.shape.moe.top_k,
    };
    let draft = Experts {
        count: g.shape.dspark.experts,
        top_k: g.shape.dspark.top_k,
    };
    let mut sets = vec![main];
    if g.dspark_stages() > 0 && main != draft {
        sets.push(draft);
    }
    sets
}

pub fn router(g: &Geometry, set: Experts) -> RowBiasRouterQuery {
    RowBiasRouterQuery {
        experts: set.count,
        top_k: set.top_k,
        scaling_factor: g.shape.moe.routed_scale as f32,
    }
}

pub fn projection(
    g: &Geometry,
    ranks: u32,
    set: Experts,
    kind: GatedMlpProjection,
    tokens: u32,
) -> Result<ChainedMlpProjectionQuery, String> {
    crate::admission::check_ranks(g, ranks)?;
    if !expert_sets(g).contains(&set) {
        return Err("expert set is not part of this geometry".into());
    }
    let intermediate = g.shape.moe.intermediate / ranks;
    let gate_up = kind == GatedMlpProjection::GateUp;
    let k = if gate_up {
        g.shape.hidden
    } else {
        intermediate
    };
    let routes = tokens
        .checked_mul(set.top_k)
        .ok_or("routed capacity overflows")?;
    Ok(ChainedMlpProjectionQuery {
        projection: GatedMlpProjectionQuery {
            gemm: GemmQuery {
                m_capacity: routes,
                n: if gate_up {
                    2 * intermediate
                } else {
                    g.shape.hidden
                },
                k,
                weight: WeightFormat::Mxfp4,
                activation: ActivationFormat::Mxfp8,
                output: if gate_up { DType::Bf16 } else { DType::F32 },
                epilogues: Epilogues {
                    swiglu: gate_up.then_some(SwigluColumns::ConcatenatedHalves),
                    output_quantization: gate_up.then_some(OutputQuantization::Mxfp8),
                    ..Epilogues::default()
                },
                grouped: Some(GroupedGemm {
                    experts: set.count,
                    top_k: set.top_k,
                    capacity: RoutedCapacity::TotalRows(routes),
                }),
            },
            projection: kind,
            swiglu_clamp: gate_up.then_some(SwigluClamp::symmetric(g.shape.swiglu_limit as f32)),
            route_weight: Some(RouteWeightPoint::BeforeDown),
        },
        k: ChainedK::new(k).map_err(|e| e.to_string())?,
    })
}

/// One routed projection's local-scale bank as the weight plan places it
/// raw: a row-major UE8M0 matrix of `rows` by `groups` (one scale per 32
/// along K) per expert and operand, matrix `expert * projections +
/// projection`. A bank the plan stores encoded expands into this layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScaleBank {
    pub rows: u32,
    pub groups: u32,
    pub projections: u32,
    pub experts: u32,
}

impl ScaleBank {
    /// The raw bank's bytes.
    pub fn bytes(&self) -> u64 {
        u64::from(self.experts)
            * u64::from(self.projections)
            * u64::from(self.rows)
            * u64::from(self.groups)
    }
}

/// The raw scale bank of `set`'s `kind` projection at `ranks` ranks: the
/// gate/up's two operands of the rank's intermediate rows over the hidden
/// width, or the down's hidden rows over the rank's intermediate slice.
pub fn scale_bank(
    g: &Geometry,
    ranks: u32,
    set: Experts,
    kind: GatedMlpProjection,
) -> Result<ScaleBank, String> {
    let gemm = projection(g, ranks, set, kind, 1)?.projection.gemm;
    let projections = if kind == GatedMlpProjection::GateUp {
        2
    } else {
        1
    };
    Ok(ScaleBank {
        rows: gemm.n / projections,
        groups: gemm.k / 32,
        projections,
        experts: set.count,
    })
}

/// The shared expert's activation at `ranks` ranks: the clamped SwiGLU of
/// its F32 gate and up columns (the rank's slice of the intermediate),
/// rounded once to BF16 and quantized as the routed experts' inputs are.
pub fn shared_activation(g: &Geometry, ranks: u32) -> SiluMulQuantizeQuery {
    let BlockQuantization::E4M3Ceil32 { amax_floor_bits } = BlockQuantization::E4M3_CEIL32 else {
        unreachable!("the ceil quantizer floors its amax")
    };
    SiluMulQuantizeQuery {
        width: g.shape.moe.intermediate / ranks,
        clamp: SwigluClamp::symmetric(g.shape.swiglu_limit as f32),
        amax_floor_bits,
    }
}

/// BF16 is the input to the trained quantizer, including the gate/up
/// epilogue's final rounded output. Continuous arithmetic remains F32.
pub fn quantizer(width: u32) -> BlockQuantizeQuery {
    BlockQuantizeQuery {
        width,
        input: DType::Bf16,
        format: BlockQuantization::E4M3_CEIL32,
        output: QuantizeOutput::Codes,
    }
}
