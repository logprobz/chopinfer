//! Routed down reductions with three, four and partially stored five chains.
use kernel_build::CubinSpec;
use kernel_core::moe_chained::ChainedK;
use kernels_sm120::moe::chained::{LocalRoute, quantized, row_bias};

pub const FAMILY: &str = "moe_chained_down_rows";
pub const COMPUTE: &str = "chop_moe_chain_down_rows_lab";

pub fn projections() -> Vec<quantized::Projection> {
    let base = crate::moe_chained_quantized::projections()
        .into_iter()
        .find(|p| !p.gate_up())
        .unwrap();
    [384, 512, 576]
        .into_iter()
        .flat_map(|k| {
            [16, 64].map(|tile| {
                let mut q = base.query;
                q.projection.gemm.k = k;
                q.projection.gemm.n = 128;
                q.k = ChainedK::new(k).unwrap();
                quantized::Projection::new(q, tile, 32).unwrap()
            })
        })
        .collect()
}

pub fn routers() -> Vec<row_bias::Router> {
    vec![row_bias::Router {
        experts: 384,
        top_k: 6,
        capacity: 32,
    }]
}

pub fn routes() -> Vec<row_bias::Route> {
    [16, 64]
        .map(|pad| row_bias::Route {
            local: LocalRoute {
                experts: 384,
                local_experts: 384,
                top_k: 6,
                capacity: if pad == 16 { 32 } else { 64 },
                pad,
            },
        })
        .to_vec()
}

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(
        &[COMPUTE],
        vec![quantized::cubin_spec(COMPUTE, &projections())],
    )
}
