//! MXFP4 / ceil-MXFP8 chained compute and row-bias router cases.

use kernel_build::{CubinSpec, EntryParam as P, EntrySpec};
use kernel_core::contract::{
    ActivationFormat, DType, Epilogues, GemmQuery, GroupedGemm, OutputQuantization, RoutedCapacity,
    SwigluColumns, WeightFormat,
};
use kernel_core::gated_mlp::{
    GatedMlpProjection, GatedMlpProjectionQuery, RouteWeightPoint, SwigluClamp,
};
use kernel_core::moe_chained::{ChainedK, ChainedMlpProjectionQuery};
use kernels_sm120::moe::chained::{LocalRoute, quantized, row_bias};

pub const FAMILY: &str = "moe_chained_quantized";
pub const COMPUTE: &str = "chop_moe_chain_quantized_lab";
pub const ROUTES: &str = "chop_moe_row_bias_lab";
pub const PROBES: &str = "chop_moe_chain_quantized_probes";
pub const LIVE_PRODUCER: &str = "chop_moe_chain_probe_live_v1";
pub const COPY_PRODUCER: &str = "chop_moe_chain_probe_copy_v1";

pub fn projections() -> Vec<quantized::Projection> {
    let mut projections = Vec::new();
    for row_tile in [16, 64] {
        for (experts, top_k) in [(384, 6), (128, 3)] {
            for gate_up in [false, true] {
                let k = if gate_up { 128 } else { 576 };
                let query = ChainedMlpProjectionQuery {
                    projection: GatedMlpProjectionQuery {
                        gemm: GemmQuery {
                            m_capacity: 32 * top_k,
                            n: if gate_up { 1152 } else { 64 },
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
                                experts,
                                top_k,
                                capacity: RoutedCapacity::TotalRows(32 * top_k),
                            }),
                        },
                        projection: if gate_up {
                            GatedMlpProjection::GateUp
                        } else {
                            GatedMlpProjection::Down
                        },
                        swiglu_clamp: gate_up.then_some(SwigluClamp::symmetric(10.0)),
                        route_weight: Some(RouteWeightPoint::BeforeDown),
                    },
                    k: ChainedK::new(k).unwrap(),
                };
                projections.push(quantized::Projection::new(query, row_tile, 32).unwrap());
            }
        }
    }
    projections
}

pub fn routers() -> Vec<row_bias::Router> {
    [(384, 6), (128, 3)]
        .into_iter()
        .map(|(experts, top_k)| row_bias::Router {
            experts,
            top_k,
            capacity: 32,
        })
        .collect()
}

pub fn routes() -> Vec<row_bias::Route> {
    routers()
        .into_iter()
        .flat_map(|router| {
            [32, 64].map(|capacity| row_bias::Route {
                local: LocalRoute {
                    experts: router.experts,
                    local_experts: router.experts,
                    top_k: router.top_k,
                    capacity,
                    pad: if capacity == 32 { 16 } else { 64 },
                },
            })
        })
        .collect()
}

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(
        &[COMPUTE, ROUTES, PROBES],
        vec![
            quantized::cubin_spec(COMPUTE, &projections()),
            row_bias::cubin_spec(ROUTES, &routers(), &routes()).unwrap(),
            probes(),
        ],
    )
}

fn probes() -> CubinSpec {
    let ptr = |name: &str, ty: &str| P::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    };
    let mut copy = crate::latent_attention::shapes::pdl_copy_entry();
    copy.symbol = COPY_PRODUCER.into();
    CubinSpec {
        name: PROBES.into(), sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
        include_dirs: vec![], headers: vec!["#include \"kernel-lab/cuda/probes/pdl_prefetch.cuh\"".into()], defines: vec![],
        entries: vec![EntrySpec::new(LIVE_PRODUCER.into(), vec![
            ptr("eviction", "const uint4"), ptr("checksum", "uint32_t"), P::U64 { name: "vectors".into() },
            ptr("operands", "const chop::probe::PdlOperand"), P::U32 { name: "operand_count".into() },
            ptr("live", "uint32_t"), P::U32 { name: "rows".into() }, P::U64 { name: "tail_ns".into() },
        ], "chop::probe::pdl_prefetch_producer(eviction,checksum,vectors,operands,operand_count,live,rows,tail_ns);".into(), 128, 1), copy],
    }
}
