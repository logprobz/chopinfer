//! Direct folds selected only for measured Flash shape and row buckets.
use super::{mxfp8_fold as previous, mxfp8_fold_direct_entries as entries};
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;
use kernels_sm120::gemm::mxfp8_fold_direct::Schedule;
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub fn selected(g: &Geometry, q: Mxfp8FoldGemmQuery, rows: u32, paired: bool) -> Option<Schedule> {
    let row = previous::tuning_row(
        g,
        model_deepseek41_contract::tuning::Family::DenseFold,
        "dense_fold.direct.decode",
        q,
        paired,
        rows,
    )?;
    assert_eq!(row.schedule, "direct");
    let warps = (
        previous::tuning_count(row, "n_warps"),
        previous::tuning_count(row, "k_warps"),
    );
    Some(match warps {
        (4, 1) => Schedule::Block,
        (1, 4) => Schedule::LocalSplit,
        _ => panic!("unqualified direct fold warp layout {warps:?}"),
    })
}

pub fn register(
    mut backend: kernels_sm120::backend::Sm120,
    g: &Geometry,
    ranks: u32,
) -> Result<kernels_sm120::backend::Sm120, String> {
    if g.id != FLASH.id {
        return Ok(backend);
    }
    let catalog = entries::catalog()?;
    for mut old in previous::decode_entries(g, ranks)? {
        old.query = previous::routed_query(old.query, previous::SMALL_CAPACITY);
        for rows in 1..=previous::SMALL_CAPACITY {
            if let Some(schedule) = selected(g, old.query, rows, old.paired) {
                let mut entry = catalog
                    .iter()
                    .find(|e| {
                        e.packed.ticket.query == old.query
                            && e.packed.ticket.paired == old.paired
                            && e.schedule == schedule
                    })
                    .ok_or("missing direct fold winner")?
                    .clone();
                entry.packed.ticket.query = previous::routed_query(old.query, rows);
                backend = backend.with_mxfp8_fold_direct(entry);
            }
        }
    }
    Ok(backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::backend::{Backend, Need};
    #[test]
    fn direct_fold_routes_only_measured_geometry_topology_and_rows() {
        let mut covered = std::collections::BTreeSet::new();
        let mut other = FLASH;
        other.id = "unmeasured";
        for ranks in [1, 2, 4] {
            let backend = register(kernels_sm120::backend::Sm120::new(), &FLASH, ranks).unwrap();
            for old in previous::decode_entries(&FLASH, ranks).unwrap() {
                for rows in 1..=64 {
                    let q = previous::routed_query(old.query, rows);
                    let selected = selected(&FLASH, q, rows, old.paired);
                    let op = backend.mxfp8_fold_gemm_stream(&q, old.paired, Need::rows(rows));
                    assert_eq!(op.is_ok(), selected.is_some());
                    assert!(super::selected(&other, q, rows, old.paired).is_none());
                    if let Ok(op) = op {
                        assert_eq!(op.info().workspace, 0);
                        assert_eq!(op.info().symbols.len(), 2);
                        assert!(op.info().symbols[1].0.contains(selected.unwrap().name()));
                        assert!(op.info().symbols[1].1 < 49152);
                        covered.insert((ranks, q.n, q.k, q.groups, old.paired, rows));
                        let mut bad = q;
                        bad.activation_row_stride += 32;
                        assert!(
                            backend
                                .mxfp8_fold_gemm_stream(&bad, old.paired, Need::rows(rows))
                                .is_err()
                        );
                    }
                }
            }
        }
        assert_eq!(covered.len(), 12);
    }
}
