//! Packed folds, registered only for qualified Flash row buckets.
use super::mxfp8_fold as previous;
use kernel_build::CubinSpec;
use kernels_sm120::gemm::mxfp8_fold_stream::Entry;
use model_deepseek41_contract::geometry::Geometry;

#[path = "mxfp8_fold_stream_routing.rs"]
mod routing;
pub use routing::qualified;

pub const FAMILY: &str = "mxfp8_fold_stream";
pub fn catalog(g: &Geometry) -> Result<Vec<Entry>, String> {
    previous::ticket_catalog(g)?
        .into_iter()
        .map(|old| {
            let q = old.query;
            Entry::new(
                q,
                format!(
                    "chop_deepseek41_mxfp8_stream_n{}_k{}_g{}_{}_v1",
                    q.n,
                    q.k,
                    q.groups,
                    if old.paired { "pair_f32" } else { "f32" }
                ),
                old.paired,
            )
            .map_err(|e| e.to_string())
        })
        .collect()
}
pub fn deferred_entry(g: &Geometry) -> Result<Entry, String> {
    let entries = catalog(g)?;
    let mut entry = entries
        .iter()
        .find(|e| e.ticket.query.n == 512 && !e.ticket.paired)
        .or_else(|| entries.iter().find(|e| !e.ticket.paired))
        .ok_or("missing deferred fold probe")?
        .clone();
    entry.defer_admission = true;
    entry.ticket.symbol.push_str("_deferred_v1");
    Ok(entry)
}
pub fn cubin_name(g: &Geometry) -> String {
    format!("mxfp8_fold_stream_{}_sm120_v1", g.kernel_tag)
}
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    geometries
        .iter()
        .filter(|g| g.id == model_deepseek41_contract::geometry::FLASH.id)
        .map(|g| {
            Ok((
                FAMILY,
                kernels_sm120::gemm::mxfp8_fold_stream::cubin_spec(&cubin_name(g), &{
                    let mut entries = catalog(g)?;
                    entries.push(deferred_entry(g)?);
                    entries
                })
                .map_err(|e| e.to_string())?,
            ))
        })
        .collect()
}

pub fn register(
    mut backend: kernels_sm120::backend::Sm120,
    g: &Geometry,
    ranks: u32,
) -> Result<kernels_sm120::backend::Sm120, String> {
    if g.id != model_deepseek41_contract::geometry::FLASH.id {
        return Ok(backend);
    }
    for old in previous::ticket_entries(g, ranks)? {
        let entry = catalog(g)?
            .into_iter()
            .find(|e| e.ticket.query == old.query && e.ticket.paired == old.paired)
            .ok_or("missing packed fold entry")?;
        for rows in 1..=previous::SMALL_CAPACITY {
            if qualified(g, old.query, rows, old.paired) {
                let mut bounded = entry.clone();
                bounded.ticket.query = previous::routed_query(old.query, rows);
                backend = backend.with_mxfp8_fold_stream(bounded);
            }
        }
    }
    Ok(backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::backend::{Backend, Need};
    use model_deepseek41_contract::{admission::SUPPORTED_RANKS, geometry::FLASH};
    #[test]
    fn serving_stream_buckets_are_exact_and_geometry_scoped() {
        let mut covered = std::collections::BTreeSet::new();
        let mut other = FLASH;
        other.id = "unmeasured";
        for &ranks in SUPPORTED_RANKS {
            let backend = register(kernels_sm120::backend::Sm120::new(), &FLASH, ranks).unwrap();
            for entry in previous::ticket_entries(&FLASH, ranks).unwrap() {
                for rows in 1..=64 {
                    let q = previous::routed_query(entry.query, rows);
                    let enabled = qualified(&FLASH, q, rows, entry.paired);
                    let op = backend.mxfp8_fold_gemm_stream(&q, entry.paired, Need::rows(rows));
                    assert_eq!(op.is_ok(), enabled);
                    assert!(!qualified(&other, q, rows, entry.paired));
                    if enabled {
                        covered.insert((q.n, q.k, q.groups, entry.paired, rows));
                    }
                    if ![1, 2, 4, 6, 8, 16, 32].contains(&rows) {
                        assert!(!enabled);
                    }
                }
            }
        }
        assert_eq!(covered.len(), 49);
        println!("{}", serde_json::json!({"stream_serving_buckets":covered}));
    }
}
