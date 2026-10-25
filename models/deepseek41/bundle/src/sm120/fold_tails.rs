//! Additional Flash TP4 decode/verify status-fold rows, including 35-row
//! balanced drafter partitions and complete draft blocks through 60 rows.
//!
//! Each entry matches its current packed fallback bitwise on four logical
//! ranks with poisoned eager/captured replays, negative controls and
//! memcheck. At 300 W with unlocked clocks (2026-10-25), every row below
//! won by more than 3% and its A/A spread on both GPUs, all ranks and both
//! warm and cold inputs. WO uses the measured sliced input quantizer.
//! WKV and other projections retain their previous admission.
use kernels_sm120::gemm::mxfp8_fold_cluster_fused::Entry;
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub(crate) const ROWS: [u32; 32] = [
    3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 35, 40, 45, 50, 55, 60,
];

/// The wide modules also serve drafter rows in default verify-32 profiles.
pub(crate) fn enabled(g: &Geometry, ranks: u32) -> bool {
    g.id == FLASH.id && ranks == 4
}

pub(crate) fn tile(rows: u32) -> Option<u32> {
    ROWS.contains(&rows).then_some(match rows {
        0..=16 => 16,
        17..=32 => 32,
        _ => 64,
    })
}

pub(crate) fn wide_rows(rows: u32) -> bool {
    matches!(rows, 35 | 40 | 45 | 50 | 55 | 60)
}

pub(crate) fn cluster(id: &str, rows: u32) -> Option<Entry> {
    if !matches!(id, "attention.q-a" | "attention.o-a" | "attention.o-b") {
        return None;
    }
    match tile(rows)? {
        64 => super::fold_verify::entry(id),
        tile => super::fold_fused::fused(id, tile),
    }
}
