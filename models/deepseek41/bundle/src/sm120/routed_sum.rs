//! The routed sum the bundle registers on SM120 and its build compiles
//! (`kernels_sm120::moe::chained::RoutedSum`): each rank's weighted F32
//! routed rows summed per token in F32, with or without the shared
//! expert's F32 partial added once, one cubin of each form per geometry.

use kernel_build::CubinSpec;
use kernels_sm120::backend::Sm120;
use kernels_sm120::moe::chained::{self, RoutedSum, RoutedSumOutput};
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::serving::PREFILL_TOKENS;

/// The lock family of the routed sum cubins.
pub const FAMILY: &str = "routed_sum";

/// The symbol prefix of `g`'s sums.
pub fn prefix(g: &Geometry) -> String {
    format!("chop_moe_{}", g.kernel_tag)
}

/// The names of `g`'s two cubins: the sum, and the sum that adds a row.
pub fn cubin_names(g: &Geometry) -> [String; 2] {
    let prefix = prefix(g);
    [format!("{prefix}_route"), format!("{prefix}_sum_add")]
}

/// `g`'s backbone sum: every expert local, F32 out.
pub fn sum(g: &Geometry) -> RoutedSum {
    RoutedSum {
        top_k: g.shape.moe.top_k,
        width: g.shape.hidden,
        local_experts: g.shape.moe.experts,
        capacity: PREFILL_TOKENS,
        output: RoutedSumOutput::F32,
    }
}

/// Every geometry's two routed sum cubins, with their lock family.
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let mut specs = Vec::new();
    for g in geometries {
        let prefix = prefix(g);
        let error = |e: kernel_core::contract::Refusal| e.to_string();
        specs.push((
            FAMILY,
            chained::route_cubin_spec(&prefix, &[], &[sum(g)]).map_err(error)?,
        ));
        specs.push((
            FAMILY,
            chained::routed_sum_add_cubin_spec(format!("{prefix}_sum_add"), &prefix, &[sum(g)])
                .map_err(error)?,
        ));
    }
    Ok(specs)
}

/// `sm120` with `g`'s routed sums registered.
pub fn register(sm120: Sm120, g: &Geometry) -> Sm120 {
    sm120
        .with_routed_sum(sum(g), prefix(g))
        .with_routed_sum_add(sum(g), prefix(g))
}
