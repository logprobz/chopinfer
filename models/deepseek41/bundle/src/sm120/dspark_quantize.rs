//! The drafter's ceil quantizers on the shared warp-parallel engine.
//!
//! Each 32-value group keeps the serial entry's arithmetic; only integer
//! status ORs cross warps, so codes, scales and status match the serial
//! `dspark_aux` entries, which stay compiled for the lab comparison.
use kernel_build::CubinSpec;
use kernels_sm120::row::block_quantize;
use model_deepseek41_contract::geometry::Geometry;

pub const FAMILY: &str = "dspark_quantize_parallel";

pub fn cubin_name(g: &Geometry) -> String {
    format!("dspark_quantize_parallel_{}_sm120", g.kernel_tag)
}

pub fn quantizers(g: &Geometry) -> Result<Vec<block_quantize::Entry>, String> {
    super::dspark_aux::quantizers(g)?
        .into_iter()
        .map(|mut entry| {
            entry.symbol = format!("{}_parallel_v1", entry.symbol.strip_suffix("_v1").unwrap());
            entry.parallel().map_err(|e| e.to_string())
        })
        .collect()
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .map(|g| {
            block_quantize::cubin_spec(&cubin_name(g), &quantizers(g)?)
                .map(|spec| (FAMILY, spec))
                .map_err(|e| e.to_string())
        })
        .collect()
}
