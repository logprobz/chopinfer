//! Frozen old query and parallel query, including a whole-row admission control.
use kernel_build::CubinSpec;
use kernels_sm120::pooled_indexer::{position_query, position_query_parallel};
use kernels_sm120::row::block_quantize;

pub const FAMILY: &str = "position_query_parallel";
pub const CUBIN: &str = "position_query_parallel_sm120";
pub const ROWS: [u32; 7] = [1, 2, 4, 6, 8, 16, 32];
pub fn entry() -> position_query::Entry {
    let mut e = super::position_query::entries().remove(1);
    e.name = "parallel_reference".into();
    e.capacity = 64;
    e
}
pub fn query() -> kernel_core::pooled_indexer::PositionIndexerQuery {
    let mut q = super::position_query::queries().remove(1);
    q.capacity = 64;
    q
}
pub fn quantizer() -> block_quantize::Entry {
    block_quantize::Entry::new(query().query_quantization().unwrap(), 64,
        "chop_position_query_parallel_reference_mxfp4_v2").unwrap()
}
pub fn cubins() -> Vec<CubinSpec> {
    let mut spec = position_query::cubin_spec(CUBIN, &[entry()], &[quantizer()]).unwrap();
    spec.headers.push("#include \"pooled_indexer/position_query_parallel.cuh\"".into());
    spec.entries.push(position_query_parallel::spec(&entry(), false).unwrap());
    spec.entries.push(position_query_parallel::spec(&entry(), true).unwrap());
    crate::declared(&[CUBIN], vec![spec])
}
