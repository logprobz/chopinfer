//! Row-local folds selected for measured Flash TP4 decode and verify
//! buckets: the captured C32 query pair loop and swizzled deep rings.
use kernel_build::CubinSpec;
use kernels_sm120::gemm::{mxfp8_fold, mxfp8_fold_local_rows::Entry};
use model_deepseek41_contract::geometry::{FLASH, Geometry};
pub const FAMILY: &str = "fold_local_rows";
pub const CUBIN: &str = "fold_local_rows_deepseek41_sm120";

// SM120, 250 W, 2026-10-22: graph replay against the swizzled served fold
// of each bucket, warm medians in us on two devices (cold in parentheses).
// q-b 16: 16.9/17.0 -> 11.3/11.0 (20.5 -> 14.1). q-b 32: the pair loop
// (13.0/12.9) ties every deep ring and stays. o-b 16: 15.2/15.2 ->
// 13.6/13.2 (18.4 -> 16.4). o-b 32: 24.0/23.2 -> 17.4/17.0 (28.6/26.6 ->
// 20.2/18.4). Each ring is bitwise equal to the fold it replaces, with
// rejected rows, delayed admission and captured decode and verify operands.
// Columns: projection, rows, N, K, column tile, folds in flight (0 keeps
// the pair loop; a deep ring swizzles its shared rows). Other buckets keep
// their route.
#[rustfmt::skip]
pub const SERVED: &[(&str, u32, u32, u32, u32, u32)] = &[
    ("attention.q-b", 16, 8192, 1280, 32, 3),
    ("attention.q-b", 32, 8192, 1280, 32, 0),
    ("attention.o-b", 16, 5120, 2048, 32, 5),
    ("attention.o-b", 32, 5120, 2048, 32, 5),
];

fn entry(n: u32, k: u32, tile_n: u32, stages: u32) -> Entry {
    let base = super::mxfp8_fold::prefill_entries(&FLASH, 4)
        .unwrap()
        .into_iter()
        .find(|e| (e.query.n, e.query.k, e.query.groups) == (n, k, 1))
        .unwrap();
    let mut q = base.query;
    q.m_capacity = 32;
    Entry {
        base: mxfp8_fold::Entry::new(q, format!("chop_fold_local_n{n}_k{k}_g1_f32")).unwrap(),
        tile_n,
        stages,
        swizzle: stages > 0,
    }
}

/// The captured query pair loops, then each distinct ring `SERVED` names.
pub fn entries() -> Vec<Entry> {
    let mut entries: Vec<Entry> = [16, 32]
        .into_iter()
        .map(|tile_n| entry(8192, 1280, tile_n, 0))
        .collect();
    for &(_, _, n, k, tile_n, stages) in SERVED {
        let e = entry(n, k, tile_n, stages);
        if !entries.iter().any(|old| old.symbol() == e.symbol()) {
            entries.push(e);
        }
    }
    entries
}

/// The entry `SERVED` names for projection `id` at `rows` rows.
pub fn served(id: &str, rows: u32) -> Option<Entry> {
    SERVED
        .iter()
        .find(|s| (s.0, s.1) == (id, rows))
        .map(|&(_, _, n, k, tile_n, stages)| entry(n, k, tile_n, stages))
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(|g| g.id == FLASH.id) {
        return Ok(Vec::new());
    }
    Ok(vec![(
        FAMILY,
        kernels_sm120::gemm::mxfp8_fold_local_rows::cubin_spec(CUBIN, &entries())
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
