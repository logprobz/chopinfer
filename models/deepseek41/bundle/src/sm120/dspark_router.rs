//! The drafter's smaller exact F32 router projection, in a separate cubin.
use kernel_build::CubinSpec;
use kernels_sm120::backend::Sm120;
use kernels_sm120::f32_projection::{Entry, cubin_spec};
use kernels_sm120::moe::chained::{self, RoutedSum};
use model_deepseek41_contract::geometry::Geometry;

pub const FAMILY: &str = "dspark_router_f32";
pub const SUM_FAMILY: &str = "dspark_routed_sum";

fn sum(g: &Geometry) -> RoutedSum {
    RoutedSum {
        top_k: g.shape.dspark.top_k,
        local_experts: g.shape.dspark.experts,
        ..super::routed_sum::sum(g)
    }
}

fn distinct_sum(g: &Geometry) -> bool {
    g.shape.dspark.experts != g.shape.moe.experts || g.shape.dspark.top_k != g.shape.moe.top_k
}

fn sum_prefix(g: &Geometry) -> String {
    format!("chop_{}_dspark_moe", g.kernel_tag)
}

pub fn sum_cubins(g: &Geometry) -> [String; 2] {
    let prefix = sum_prefix(g);
    [format!("{prefix}_route"), format!("{prefix}_sum_add")]
}

pub fn register(backend: Sm120, g: &Geometry) -> Sm120 {
    if distinct_sum(g) {
        backend
            .with_routed_sum(sum(g), sum_prefix(g))
            .with_routed_sum_add(sum(g), sum_prefix(g))
    } else {
        backend
    }
}

pub fn cubin_name(g: &Geometry) -> String {
    format!("dspark_router_f32_{}_sm120", g.kernel_tag)
}

pub fn entries(g: &Geometry) -> Result<Vec<Entry>, String> {
    let shape = (g.shape.dspark.experts, g.shape.hidden);
    if super::f32_dense::shapes(g).contains(&shape) {
        return Ok(Vec::new());
    }
    let query = super::f32_dense::query(shape.0, shape.1);
    Ok(vec![
        Entry::new(
            query,
            super::f32_head::usage(),
            &format!("chop_{}_dspark_router_f32", g.kernel_tag),
        )
        .map_err(|e| e.to_string())?,
    ])
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let mut result = Vec::new();
    for g in geometries {
        let entries = entries(g)?;
        if !entries.is_empty() {
            result.push((
                FAMILY,
                cubin_spec(cubin_name(g), &entries).map_err(|e| e.to_string())?,
            ));
        }
        if distinct_sum(g) {
            let prefix = sum_prefix(g);
            result.push((
                SUM_FAMILY,
                chained::route_cubin_spec(&prefix, &[], &[sum(g)]).map_err(|e| e.to_string())?,
            ));
            result.push((
                SUM_FAMILY,
                chained::routed_sum_add_cubin_spec(format!("{prefix}_sum_add"), &prefix, &[sum(g)])
                    .map_err(|e| e.to_string())?,
            ));
        }
    }
    Ok(result)
}
