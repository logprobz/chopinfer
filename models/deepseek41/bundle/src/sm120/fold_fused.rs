//! Folds that admit rows by status for Flash TP4's 1-, 2-, 4-, 8-, 16- and
//! 32-row decode and verify buckets: WQ_a, WKV, WO_a and WO_b on their
//! cluster schedules (`fold_cluster::SERVED`) without the admission launch,
//! WO_a quantizing WO_b's input in its merge after the row-sliced
//! 8192-column quantizer (`row/ceil_sliced.cuh`), WO_b failing rows by
//! status and WO_a's block flags, and WQ_b's row-local fold on 48-column
//! tiles without its admission (`mxfp8_fold_local_wide.cuh`). The buckets
//! below 16 rows launch the 16-row tile's folds and quantizer with their
//! rows (`tile`). The one-row decode step keeps the dual fold for WQ_a and
//! WKV (`ops::dense_dual`), which the lowering takes before any status fold,
//! so there only WQ_b and the WO chain change.
//!
//! A fold that fails exactly the rows with a nonzero status after its
//! producer wait gives the admission kernel's result, and leaves the status
//! as it would, only when its operands come from an E4M3 ceil block
//! quantizer that zeroes the codes of the rows it rejects (zero codes and
//! scale 1 are admitted; every code of a row it passes is). The producers
//! here are:
//! - WQ_a and WKV: the cached norm quantizer inside the HC finish,
//!   completion, twin and split-finish boundaries, or the input norm
//!   quantizer;
//! - WQ_b: the query norm quantizer;
//! - WO_a: the sliced quantizer, which stores computed codes for a rejected
//!   row but ORs the served quantizer's flags, so every status ends as the
//!   served chain leaves it and the fold never reads those codes;
//! - WO_b: WO_a's merge, whose flags bytes WO_b ORs into the status.
//!
//! `fake_host_tests::fold_fused` fails if a status fold's codes come from
//! any other producer or if a producer's rejected-row zeroing changes.
//!
//! SM120, 300 W, 2026-10-22 (lab fold_fused): graph replay of each chain
//! against the served one, warm medians in us per layer on two devices,
//! bit for bit equal with rejected rows and captured layers 0 and 39. WO
//! chain (quantizer, WO_a, quantizer, WO_b with both admissions) 29.7/29.7
//! -> 21.5/21.5 at 16 rows (32 slices of 8 warps) and 37.1/35.7 ->
//! 29.5/29.0 at 32 rows (16 slices of 8 warps). WQ_b 11.2/11.2 -> 9.3/9.2
//! and 13.2/13.3 -> 11.1/11.1. WQ_a 9.0/9.1 -> 7.2/7.2 and 11.3/11.2 ->
//! 11.1/11.1. WKV 7.0/7.2 -> 6.9/7.0 and 9.2/9.3 -> 8.9/9.0.
//!
//! SM120, 300 W, 1800 MHz lock (1.73-1.79 GHz read), 2026-10-23 (lab
//! fold_fused8): the 16-row tile's functions at 8 rows against the served
//! 8-row launches (the prefetch folds with their initialize launches,
//! WQ_b's stream fold, the cached and F32 input quantizers), bit for bit
//! equal at every row count up to 8 with rejected rows and captured layers
//! 0 and 39. Warm medians in us per layer on two devices: WO chain
//! 39.6/39.6 -> 23.2/23.3, WQ_b 20.0/19.2 -> 12.7/12.8, WQ_a 12.9/13.1 ->
//! 9.0/8.9, WKV 10.8/10.9 -> 8.7/8.7; over copies that keep the weights
//! out of L2, 43.0/42.5 -> 26.4/26.1, 17.6/17.2 -> 11.9/11.8, 14.4/14.3 ->
//! 8.8/8.7 and 10.4/10.2 -> 5.2/5.1 us, and 16.7/15.6 -> 12.6/11.7 mJ per
//! layer of board energy for all four.
//!
//! SM120, 300 W, 1.74-1.79 GHz read, 2026-10-23 (lab fold_fused, the 1- to
//! 7-row tiles): the 16-row tile's functions at 1, 2 and 4 rows against the
//! launches of each row count (the stream, prefetch and direct folds with
//! their one-time launches, the one-CTA cached and F32 input quantizers),
//! bit for bit equal at every row count up to 8 with rejected rows and
//! captured layers 0 and 39 and their 1-, 2- and 4-row prefixes.
//! Main-stream change per 40-layer step on two devices, the smallest of
//! warm, cold and streamed: -0.78/-0.77 ms at 1 row (WQ_b and the WO
//! chain), -0.86/-0.82 ms at 2 rows and -1.06/-1.08 ms at 4 rows (also
//! WQ_a); board energy -0.32/-0.28, -0.25/-0.23 and -0.13/-0.12 J per token
//! over four ranks. One-row decode serves its two WO input quantizers as
//! the split ceil twins (`ceil_split`), 5.6 us per layer faster than the
//! one-CTA quantizers in their lab chain (d2-0463g1), so the one-row change
//! against the served step is about -0.56/-0.55 ms, and its energy figure
//! is against the one-CTA quantizers. At one row WQ_a's status fold beat
//! the dual fold by 0.25 to 0.33 ms only with WKV beside it on the side
//! stream, which would also give up the query window twin the dual fold
//! enables (2 us per layer), so the dual fold stays.
use kernel_build::CubinSpec;
use kernels_sm120::gemm::{
    mxfp8_fold,
    mxfp8_fold_cluster_fused::{self, Entry, Quantizer},
    mxfp8_fold_local_wide,
};
use model_deepseek41_contract::geometry::{FLASH, Geometry};
pub const FAMILY: &str = "mxfp8_fold_fused";
pub const CUBIN: &str = "mxfp8_fold_fused_deepseek41_sm120";
/// The row buckets every fused fold serves.
pub const ROWS: [u32; 6] = [1, 2, 4, 8, 16, 32];
/// The row tiles the cubin holds, each with its folds and quantizer.
pub const TILES: [u32; 2] = [16, 32];
/// The projections whose served cluster fold a fused twin replaces.
pub const CLUSTER: [&str; 4] = [
    "attention.q-a",
    "attention.kv",
    "attention.o-a",
    "attention.o-b",
];
/// Tile rows, slices per row and warps per slice of WO_a's input
/// quantizer.
pub const SLICED: &[(u32, u32, u32)] = &[(16, 32, 8), (32, 16, 8)];
/// WQ_b's wide fold at every bucket: column tile and folds in flight.
pub const WIDE: (u32, u32) = (48, 4);
/// WO_a's input: the rank's heads, BF16.
const ATTENTION: u32 = 8192;

/// The row tile bucket `rows` launches: 1, 2, 4 and 8 rows run the 16-row
/// tile.
pub fn tile(rows: u32) -> Option<u32> {
    match rows {
        1 | 2 | 4 | 8 | 16 => Some(16),
        32 => Some(32),
        _ => None,
    }
}

/// The fused twin of `id`'s served cluster fold at the tile of `rows`
/// rows.
pub fn fused(id: &str, rows: u32) -> Option<Entry> {
    if !CLUSTER.contains(&id) {
        return None;
    }
    let mut cluster = super::fold_cluster::served(id, tile(rows)?)?;
    let q = cluster.base.query;
    cluster.base = mxfp8_fold::Entry::new(
        q,
        format!(
            "chop_deepseek41_mxfp8_fused_n{}_k{}_g{}_f32",
            q.n, q.k, q.groups
        ),
    )
    .unwrap();
    Some(Entry {
        cluster,
        pending: if id == "attention.o-b" {
            q.activation_row_stride / 32
        } else {
            0
        },
        quantize: id == "attention.o-a",
    })
}

/// WQ_b's wide fold, which every bucket launches.
pub fn wide(rows: u32) -> Option<mxfp8_fold_local_wide::Entry> {
    tile(rows)?;
    let base = super::mxfp8_fold::prefill_entries(&FLASH, 4)
        .unwrap()
        .into_iter()
        .find(|e| (e.query.n, e.query.k, e.query.groups) == (8192, 1280, 1))
        .unwrap();
    let mut q = base.query;
    q.m_capacity = 32;
    Some(mxfp8_fold_local_wide::Entry {
        base: mxfp8_fold::Entry::new(q, "chop_deepseek41_mxfp8_fused_n8192_k1280_g1_f32").unwrap(),
        tile_n: WIDE.0,
        stages: WIDE.1,
    })
}

/// WO_a's input quantizer at the tile of `rows` rows.
pub fn sliced(rows: u32) -> Option<Quantizer> {
    let tile = tile(rows)?;
    SLICED
        .iter()
        .find(|s| s.0 == tile)
        .map(|&(_, slices, warps)| Quantizer {
            width: ATTENTION,
            slices,
            warps,
            f32_input: false,
            capacity: 32,
            symbol: format!("chop_deepseek41_dense_ceil_w{ATTENTION}_sliced_s{slices}w{warps}_v1"),
        })
}

pub fn entries() -> Vec<Entry> {
    TILES
        .iter()
        .flat_map(|&rows| CLUSTER.iter().filter_map(move |id| fused(id, rows)))
        .collect()
}
pub fn quantizers() -> Vec<Quantizer> {
    TILES.iter().filter_map(|&rows| sliced(rows)).collect()
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(|g| g.id == FLASH.id) {
        return Ok(Vec::new());
    }
    let mut spec = mxfp8_fold_cluster_fused::cubin_spec(CUBIN, &entries(), &quantizers(), false)
        .map_err(|e| e.to_string())?;
    let wide = wide(32).unwrap();
    wide.validate().map_err(|e| e.to_string())?;
    spec.headers
        .push("#include \"gemm/mxfp8_fold_local_wide.cuh\"".into());
    spec.entries.push(wide.spec());
    Ok(vec![(FAMILY, spec)])
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
/// Run `run` with the served cluster and row-local folds and quantizers in
/// place of the fused ones.
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fused_folds_mirror_the_served_cluster_schedules() {
        for rows in TILES {
            for id in CLUSTER {
                let f = fused(id, rows).unwrap();
                let c = super::super::fold_cluster::served(id, rows).unwrap();
                assert_eq!(
                    (
                        f.cluster.tile_n,
                        f.cluster.splits,
                        f.cluster.stages,
                        f.cluster.tile_rows
                    ),
                    (c.tile_n, c.splits, c.stages, c.tile_rows)
                );
                assert_eq!(f.cluster.base.query, c.base.query);
                f.validate().unwrap();
            }
            assert!(fused("attention.q-b", rows).is_none());
            assert_eq!(sliced(rows).unwrap().capacity, 32);
            sliced(rows).unwrap().validate().unwrap();
        }
        // The buckets below 16 rows launch the 16-row tile's functions.
        for rows in [1, 2, 4, 8] {
            for id in CLUSTER {
                assert_eq!(
                    fused(id, rows).unwrap().symbol(),
                    fused(id, 16).unwrap().symbol()
                );
            }
            assert_eq!(sliced(rows).unwrap().symbol, sliced(16).unwrap().symbol);
            assert_eq!(wide(rows).unwrap().symbol(), wide(16).unwrap().symbol());
        }
        for rows in [3, 5, 6, 7, 12, 24] {
            assert!(fused("attention.o-a", rows).is_none() && sliced(rows).is_none());
            assert!(wide(rows).is_none());
        }
        let spec = cubins(&[FLASH]).unwrap().pop().unwrap().1;
        // Eight folds, two quantizers and one wide fold; no lab controls.
        assert_eq!(spec.entries.len(), 11);
        assert!(
            spec.entries
                .iter()
                .all(|e| !e.symbol.ends_with("_ignore_v1"))
        );
        assert_eq!(
            fused("attention.o-a", 32).unwrap().symbol(),
            "chop_deepseek41_mxfp8_fused_n1024_k4096_g2_f32_cluster32_n64_c8s3_statusq_v1"
        );
        assert_eq!(
            wide(16).unwrap().symbol(),
            "chop_deepseek41_mxfp8_fused_n8192_k1280_g1_f32_local32_n48_s4x_status_v1"
        );
    }
}
