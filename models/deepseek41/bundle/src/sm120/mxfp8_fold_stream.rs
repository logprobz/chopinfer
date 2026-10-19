//! Experimental packed folds, built for qualification without serving routing.
use super::mxfp8_fold as previous;
use kernel_build::CubinSpec;
use kernels_sm120::gemm::mxfp8_fold_stream::Entry;
use model_deepseek41_contract::geometry::Geometry;

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
