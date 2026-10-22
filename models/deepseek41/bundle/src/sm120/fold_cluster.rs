//! Cluster split-K folds selected for measured Flash TP4 decode and verify
//! buckets of the query, window KV and output projections (WQ_a, WKV, WO_a,
//! WO_b): the K splits of each output tile form one cluster, the ring keeps
//! the CTA's folds in flight, and the ranks sum the fold partials in
//! increasing K order.
use kernel_build::CubinSpec;
use kernels_sm120::gemm::{mxfp8_fold, mxfp8_fold_cluster::Entry};
use model_deepseek41_contract::geometry::{FLASH, Geometry};
pub const FAMILY: &str = "mxfp8_fold_cluster";
pub const CUBIN: &str = "mxfp8_fold_cluster_deepseek41_sm120";
/// Projection, rows, N, K, groups, column tile, splits (cluster CTAs),
/// folds in flight.
pub type Served = (&'static str, u32, u32, u32, u32, u32, u32, u32);

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
#[rustfmt::skip]
pub const SERVED: &[Served] = &[
    ("attention.q-a", 16, 1280, 5120, 1, 32, 8, 5),
    ("attention.q-a", 32, 1280, 5120, 1, 64, 8, 5),
    ("attention.kv", 16, 512, 5120, 1, 16, 8, 5),
    ("attention.kv", 32, 512, 5120, 1, 16, 8, 5),
    ("attention.o-a", 16, 1024, 4096, 2, 32, 8, 4),
    ("attention.o-a", 32, 1024, 4096, 2, 64, 8, 3),
    ("attention.o-b", 16, 5120, 2048, 1, 32, 2, 4),
    ("attention.o-b", 32, 5120, 2048, 1, 32, 2, 4),
];

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

/// Each distinct entry `SERVED` names.
pub fn entries() -> Vec<Entry> {
    let mut entries: Vec<Entry> = Vec::new();
    for &(_, rows, n, k, groups, tile_n, splits, stages) in SERVED {
        let e = entry(rows, n, k, groups, tile_n, splits, stages);
        if !entries.iter().any(|old| old.symbol() == e.symbol()) {
            entries.push(e);
        }
    }
    entries
}

/// The entry `SERVED` names for projection `id` at `rows` rows.
pub fn served(id: &str, rows: u32) -> Option<Entry> {
    SERVED.iter().find(|s| (s.0, s.1) == (id, rows)).map(
        |&(_, rows, n, k, groups, tile_n, splits, stages)| {
            entry(rows, n, k, groups, tile_n, splits, stages)
        },
    )
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
pub fn selected(g: &Geometry, ranks: u32) -> bool {
    #[cfg(test)]
    if REFERENCE.with(|f| f.get()) {
        return false;
    }
    enabled(g, ranks)
}
#[cfg(test)]
std::thread_local! { static REFERENCE:std::cell::Cell<bool>=const{std::cell::Cell::new(false)}; }
#[cfg(test)]
pub(crate) fn with_reference<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            REFERENCE.with(|f| f.set(self.0));
        }
    }
    let _restore = Restore(REFERENCE.with(|f| f.replace(true)));
    run()
}
