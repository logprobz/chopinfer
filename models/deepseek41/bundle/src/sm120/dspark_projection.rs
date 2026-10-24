//! Bounded cached projections for Flash TP4 drafter forwards.
use kernel_build::CubinSpec;
use kernels_sm120::f32_projection::{
    Entry,
    bounded_reuse::{self, Candidate},
};
use model_deepseek41_contract::{
    dspark_options::SMALL_DRAFT_ROWS,
    geometry::{FLASH, Geometry},
};

pub const FAMILY: &str = "dspark_f32_reuse";
pub const CUBIN: &str = "dspark_f32_reuse_deepseek41_sm120";

pub fn enabled(g: &Geometry, ranks: u32) -> bool {
    g.id == FLASH.id && ranks == 4
}

pub fn route(g: &Geometry, ranks: u32, entry: Entry) -> Result<Entry, String> {
    let q = entry.query();
    if enabled(g, ranks)
        && q.k == g.shape.hidden
        && [g.shape.vocab / ranks, g.shape.dspark.experts].contains(&q.n)
    {
        entry
            .with_bounded_reuse((35..=SMALL_DRAFT_ROWS).step_by(5).collect())
            .map_err(|e| e.to_string())
    } else {
        Ok(entry)
    }
}

pub fn entries() -> Result<Vec<Candidate>, String> {
    super::f32_split::entries(&FLASH, 4)?
        .into_iter()
        .map(|e| route(&FLASH, 4, e))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter_map(|e| e.bounded_reuse_candidate())
        .map(Ok)
        .collect()
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(|g| g.id == FLASH.id) {
        return Ok(Vec::new());
    }
    Ok(vec![(
        FAMILY,
        bounded_reuse::cubin_spec(CUBIN, &entries()?).map_err(|e| e.to_string())?,
    )])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drafter_routes_only_head_and_router_through_twelve_items() {
        let entries = entries().unwrap();
        assert_eq!(entries.len(), 2);
        for original in super::super::f32_split::entries(&FLASH, 4).unwrap() {
            let routed = route(&FLASH, 4, original.clone()).unwrap();
            let selected = original.query().k == FLASH.shape.hidden
                && [FLASH.shape.vocab / 4, FLASH.shape.dspark.experts]
                    .contains(&original.query().n);
            for rows in 1..=160 {
                assert_eq!(
                    routed.uses_bounded_reuse(rows),
                    selected && (35..=60).contains(&rows) && rows % 5 == 0
                );
            }
            assert_eq!(route(&FLASH, 2, original.clone()).unwrap(), original);
        }
    }
}
