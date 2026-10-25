//! Cluster split-K folds measured for Flash TP4 decode and verify buckets
//! of the query, window KV and output projections (WQ_a, WKV, WO_a, WO_b):
//! the K splits of each output tile form one cluster, the ring keeps the
//! CTA's folds in flight, and the ranks sum the fold partials in increasing
//! K order. They read row-major weight codes and serve no instance: the
//! status folds (`fold_fused`) replaced them at every bucket and read the
//! packed codes. The lab measures against them.
use kernel_build::CubinSpec;
use kernels_sm120::gemm::{mxfp8_fold, mxfp8_fold_cluster::Entry};
use model_deepseek41_contract::geometry::{FLASH, Geometry};
pub const FAMILY: &str = "mxfp8_fold_cluster";
pub const CUBIN: &str = "mxfp8_fold_cluster_deepseek41_sm120";
// SM120, 300 W, 2026-10-22: graph replay against the served fold of each
// bucket, warm medians in us on two devices. q-a 16: 12.0/12.9 -> 9.6/8.9.
// q-a 32: 16.0/16.0 -> 11.8/11.7. kv 16: 9.9/10.0 -> 7.7/7.6. kv 32:
// 10.9/10.9 -> 9.8/9.1. o-a 16: 13.4/12.0 -> 11.7/11.6. o-a 32: 19.1/19.9
// -> 14.2/14.0. o-b 16 (the row-local ring): 13.8/13.8 -> 9.9/10.0. o-b 32:
// 17.0/16.8 -> 15.0/15.1. Each entry is bitwise equal to the fold it
// replaces with rejected rows, delayed admission and captured decode and
// verify operands, and its SASS equals the measured lab entry's.
// Other buckets keep their route; these rows come before
// `fold_local_rows::SERVED`.

fn entry(rows: u32, n: u32, k: u32, groups: u32, tile_n: u32, splits: u32, stages: u32) -> Entry {
    let base = super::mxfp8_fold::prefill_entries(&FLASH, 4)
        .unwrap()
        .into_iter()
        .find(|e| (e.query.n, e.query.k, e.query.groups) == (n, k, groups))
        .unwrap();
    let mut q = base.query;
    q.m_capacity = rows;
    Entry {
        base: mxfp8_fold::Entry::new(
            q,
            format!("chop_deepseek41_mxfp8_cluster_n{n}_k{k}_g{groups}_f32"),
        )
        .unwrap(),
        tile_n,
        tile_rows: rows,
        splits,
        stages,
        distributed: false,
    }
}

/// Each distinct cluster descriptor, in the original cubin order.
pub fn entries() -> Vec<Entry> {
    let mut entries: Vec<Entry> = Vec::new();
    for id in super::fold_fused::CLUSTER {
        let op = format!("dense_fold.{}_cluster.decode", id.replace(['.', '-'], "_"));
        for rows in super::mxfp8_fold::status_rows(&op, 32) {
            let e = served(id, rows).expect("table row has a cluster descriptor");
            if !entries.iter().any(|old| old.symbol() == e.symbol()) {
                entries.push(e);
            }
        }
    }
    entries
}

/// The measured cluster descriptor for this projection and launch row.
pub fn served(id: &str, rows: u32) -> Option<Entry> {
    let q = query(id)?;
    let op = format!("dense_fold.{}_cluster.decode", id.replace(['.', '-'], "_"));
    let tuning = super::mxfp8_fold::status_tuning(&op, 32, rows)?;
    Some(entry(
        rows,
        q.n,
        q.k,
        q.groups,
        super::mxfp8_fold::tuning_count(tuning, "tile_n"),
        super::mxfp8_fold::tuning_count(tuning, "splits"),
        super::mxfp8_fold::tuning_count(tuning, "stages"),
    ))
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(|g| g.id == FLASH.id) {
        return Ok(Vec::new());
    }
    Ok(vec![(
        FAMILY,
        kernels_sm120::gemm::mxfp8_fold_cluster::cubin_spec(CUBIN, &entries(), false)
            .map_err(|e| e.to_string())?,
    )])
}
pub fn enabled(g: &Geometry, ranks: u32) -> bool {
    g.id == FLASH.id && ranks == 4
}

fn query(id: &str) -> Option<kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery> {
    static QUERIES: std::sync::OnceLock<
        std::collections::BTreeMap<String, kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery>,
    > = std::sync::OnceLock::new();
    let declaration = match id {
        "attention.q-a" => "layer.wq_a",
        "attention.kv" => "layer.wkv",
        "attention.o-a" => "layer.wo_a",
        "attention.o-b" => "layer.wo_b",
        _ => return None,
    };
    QUERIES
        .get_or_init(|| {
            super::mxfp8_fold::registrations(&FLASH, 4, 0)
                .unwrap()
                .into_iter()
                .map(|r| (r.projection, r.entry.query))
                .collect()
        })
        .get(declaration)
        .copied()
}
