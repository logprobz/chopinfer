//! Exact N64 staging measured at 8192 rows on SM120, 250 W, 2026-10-19.
//! Each selected shape wins by at least 3 percent on both devices across
//! tensor, row-fallback and weight-fallback inputs. Other rows retain M32.
use kernel_build::CubinSpec;
use kernels_sm120::f32_projection::{Entry, wide};
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub const FAMILY: &str = "f32_wide_serving";
pub fn selected(n: u32, k: u32, rows: u32) -> bool {
    matches!(n, 128 | 384 | 512) && k == 5120 && rows == 8192
}
pub fn serving_entry(g: &Geometry, entry: Entry) -> Result<Entry, String> {
    let q = entry.query();
    if g.id != FLASH.id || !selected(q.n, q.k, q.m_capacity) {
        return Ok(entry);
    }
    entry
        .with_wide_prefill_buckets(vec![(8192, 64)])
        .map_err(|e| e.to_string())
}
pub fn cubin_name(g: &Geometry) -> String {
    format!("f32_wide_{}_serving_sm120", g.kernel_tag)
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<CubinSpec>, String> {
    geometries
        .iter()
        .filter(|g| g.id == FLASH.id)
        .map(|g| {
            let entries: Vec<_> = super::f32_prefill::baseline_entries(g)?
                .into_iter()
                .filter(|e| selected(e.query().n, e.query().k, e.query().m_capacity))
                .collect();
            let mut spec = wide::cubin_spec(cubin_name(g), &entries).map_err(|e| e.to_string())?;
            spec.entries
                .retain(|e| entries.iter().any(|old| e.symbol == wide::symbol(old, 64)));
            Ok(spec)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wide_routes_only_three_measured_full_row_shapes() {
        let mut count = 0;
        for old in super::super::f32_prefill::baseline_entries(&FLASH).unwrap() {
            let q = old.query();
            let entry =
                super::super::f32_prefill::serving_entry(&FLASH, old.with_split_decode()).unwrap();
            for rows in 1..=8192 {
                let actual = entry.entry_for_rows(rows).wide_prefill_features();
                assert_eq!(actual, if selected(q.n, q.k, rows) { 64 } else { 0 });
                count += usize::from(actual != 0);
            }
        }
        assert_eq!(count, 3);
        let specs = cubins(&[FLASH]).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].entries.len(), 3);
        assert!(
            specs[0]
                .entries
                .iter()
                .all(|e| e.symbol.ends_with("_wide_m32_n64_v1"))
        );
    }
}
