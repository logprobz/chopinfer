//! Lab-only finalizer clocks and cluster transport probes.
use kernel_build::CubinSpec;
use kernels_sm120::sinkhorn_hc::{Geometry as KernelGeometry, lagged_split, mixer_probe::Entry};
use model_deepseek41_contract::{geometry::Geometry, hc, serving::DECODE_BATCH};
pub const FAMILY: &str = "sinkhorn_hc_mixer_probe";
pub fn cubin_name(g: &Geometry) -> String {
    format!("sinkhorn_hc_{}_mixer_probe_sm120", g.kernel_tag)
}
pub fn entry(g: &Geometry) -> Entry {
    Entry {
        reference: lagged_split::entry(
            kernels_sm120::Profile::synthetic(kernels_sm120::ProfileNames {
                name: "deepseek41",
                tag: g.kernel_tag,
            }),
            KernelGeometry {
                mix: hc::mix(g),
                capacity: DECODE_BATCH,
            },
            false,
        ),
        prefix: format!("chop_{}_hc_mixer_probe", g.kernel_tag),
    }
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .map(|g| {
            Ok((
                FAMILY,
                entry(g)
                    .cubin_spec(&cubin_name(g))
                    .map_err(|e| e.to_string())?,
            ))
        })
        .collect()
}
