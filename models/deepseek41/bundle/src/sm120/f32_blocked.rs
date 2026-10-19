//! Exact prefill selections measured at SM120, 250 W, 2026-10-19.
//! M32 wins the admitted tensor, row-fallback and weight-fallback paths.
//! Select only measured existing buckets with at least a 3 percent win
//! on every path. Other rows keep their previous selection.
use kernel_build::CubinSpec;
use kernels_sm120::f32_projection::{Entry, blocked};
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub const FAMILY: &str = "f32_blocked_prefill";
pub const SERVING_FAMILY: &str = "f32_blocked_serving";

pub fn selected_rows(n: u32, k: u32) -> &'static [u32] {
    match (n, k) {
        (384 | 512, 5120) => &[512, 2048, 4096, 8191, 8192],
        (32 | 128, 5120) => &[512, 2048, 8191, 8192],
        (128, 512) => &[2048, 8191, 8192],
        _ => &[],
    }
}

pub fn serving_entry(g: &Geometry, entry: Entry) -> Result<Entry, String> {
    if g.id != FLASH.id {
        return Ok(entry);
    }
    let q = entry.query();
    entry
        .with_blocked_prefill_buckets(
            selected_rows(q.n, q.k)
                .iter()
                .map(|&rows| (rows, 32))
                .collect(),
        )
        .map_err(|e| e.to_string())
}

pub fn serving_cubin_name(g: &Geometry) -> String {
    format!("f32_blocked_{}_serving_sm120", g.kernel_tag)
}

/// Only the qualified compute entries. Their existing prepare and weight
/// admission entries remain in the original families, and keep their ABI.
pub fn serving_cubins(geometries: &[Geometry]) -> Result<Vec<CubinSpec>, String> {
    geometries
        .iter()
        .filter(|g| g.id == FLASH.id)
        .map(|g| {
            let entries = super::f32_prefill::baseline_entries(g)?;
            let mut spec =
                blocked::cubin_spec(serving_cubin_name(g), &entries).map_err(|e| e.to_string())?;
            spec.entries.retain(|e| {
                entries
                    .iter()
                    .any(|old| e.symbol == blocked::symbol(old, 32))
            });
            Ok(spec)
        })
        .collect()
}
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn measured_routes_keep_ties_and_unmeasured_rows() {
        let mut count = 0;
        for old in super::super::f32_prefill::baseline_entries(&FLASH).unwrap() {
            let routed =
                super::super::f32_prefill::serving_entry(&FLASH, old.clone().with_split_decode())
                    .unwrap();
            let q = old.query();
            assert_eq!(
                routed.blocked_prefill_buckets().len(),
                selected_rows(q.n, q.k).len()
            );
            for rows in 1..=8192 {
                let entry = routed.entry_for_rows(rows);
                assert_eq!(
                    entry.blocked_prefill_rows() != 0,
                    selected_rows(q.n, q.k).contains(&rows)
                );
            }
            count += routed.blocked_prefill_buckets().len();
        }
        assert_eq!(count, 21);
        assert!(!selected_rows(128, 512).contains(&512));
        assert!(selected_rows(32, 5120).contains(&2048));
        let cubins = serving_cubins(&[FLASH]).unwrap();
        assert_eq!(cubins.len(), 1);
        assert_eq!(cubins[0].entries.len(), 5);
        assert!(
            cubins[0]
                .entries
                .iter()
                .all(|e| e.symbol.ends_with("_blocked_m32_v1"))
        );
    }
}
