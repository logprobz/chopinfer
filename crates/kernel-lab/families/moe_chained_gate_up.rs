//! A multi-chain gate/up instance for the shared routed conformance fixture.
use kernel_build::CubinSpec;
use kernel_core::moe_chained::ChainedK;
use kernels_sm120::moe::chained::quantized::{self, Projection};

pub const FAMILY: &str = "moe_chained_gate_up";
pub const CUBIN: &str = "chop_moe_chain_gate_up_lab";

pub fn projections() -> Vec<Projection> {
    let original = crate::moe_chained_quantized::projections()
        .into_iter()
        .find(|p| {
            p.gate_up()
                && p.row_tile == 16
                && p.query.projection.gemm.grouped.unwrap().experts == 128
        })
        .expect("the single-chain gate/up case remains registered");
    let mut query = original.query;
    query.projection.gemm.k = 256;
    query.k = ChainedK::new(256).unwrap();
    vec![Projection::new(query, original.row_tile, original.tokens_capacity).unwrap()]
}

pub fn cubins() -> Vec<CubinSpec> {
    vec![quantized::cubin_spec(CUBIN, &projections())]
}
