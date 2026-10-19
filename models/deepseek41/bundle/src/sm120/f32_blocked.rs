//! Candidate-only exact prefill family. Serving needs measured row buckets.
use kernel_build::CubinSpec;
use kernels_sm120::f32_projection::blocked;
use model_deepseek41_contract::geometry::Geometry;

pub const FAMILY: &str = "f32_blocked_prefill";
pub fn cubin_name(g: &Geometry) -> String {
    format!("f32_blocked_{}_sm120", g.kernel_tag)
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<CubinSpec>, String> {
    geometries
        .iter()
        .map(|g| {
            blocked::cubin_spec(cubin_name(g), &super::f32_prefill::baseline_entries(g)?)
                .map_err(|e| e.to_string())
        })
        .collect()
}
