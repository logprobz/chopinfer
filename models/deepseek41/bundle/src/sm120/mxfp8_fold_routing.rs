//! Measured decode selection; all entries retain the same fold arithmetic.
use super::small_query;
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;
use model_deepseek41_contract::geometry::{FLASH, Geometry};
use std::ops::RangeInclusive;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodePath {
    Split,
    Ticket,
    Prefetch,
}

// Flash SM120, 250 W, 2026-10-19, TP1/TP2/TP4, two GPUs:
// each device measured five interleaved candidate/baseline timing ratios.
// Select prefetch only when BOTH device medians are below 0.97.
// The baseline is ticket, except for the existing split-fold buckets.
// Split buckets did not measure prefetch and remain split. All other
// omitted rows retain ticket, including ties and regressions.
// Columns: N, K, groups, paired, winning rows; comment: worst selected
// median ratio on either device. Empty ranges mean no measured win.
type MeasuredRows = (u32, u32, u32, bool, &'static [RangeInclusive<u32>]);
#[rustfmt::skip]
const PREFETCH_WINS: &[MeasuredRows] = &[
    (512, 5120, 1, false, &[1..=64]), // 0.917862
    (576, 5120, 1, false, &[1..=64]), // 0.923077
    (576, 5120, 1, true, &[1..=19, 21..=21, 23..=25, 27..=46, 48..=48, 50..=64]), // 0.947325
    (1024, 4096, 2, false, &[1..=64]), // 0.961515
    (1024, 4096, 4, false, &[1..=64]), // 0.926829
    (1024, 4096, 8, false, &[1..=64]), // 0.890387
    (1152, 5120, 1, false, &[1..=21, 23..=56, 59..=59, 61..=63]), // 0.948148
    (1152, 5120, 1, true, &[1..=16, 24..=24, 35..=35, 37..=37, 42..=42, 44..=44, 47..=51]), // 0.964266
    (1280, 5120, 1, false, &[1..=16, 20..=32, 38..=42, 44..=45, 47..=64]), // 0.947368
    (2304, 5120, 1, false, &[1..=17, 28..=32, 51..=51, 56..=57, 59..=64]), // 0.962942
    (2304, 5120, 1, true, &[]),
    (4096, 1280, 1, false, &[2..=2, 4..=4, 10..=11, 13..=13, 24..=24, 27..=27, 29..=29, 31..=31, 61..=62, 64..=64]), // 0.928571
    (5120, 576, 1, false, &[29..=32, 49..=56]), // 0.916667
    (5120, 1152, 1, false, &[22..=27, 29..=32, 48..=48, 64..=64]), // 0.942042
    (5120, 2048, 1, false, &[1..=10, 12..=12, 16..=16, 24..=24]), // 0.937439
    (5120, 2304, 1, false, &[1..=1, 3..=4, 7..=7, 9..=11, 14..=14]), // 0.929609
    (5120, 4096, 1, false, &[28..=28, 32..=32]), // 0.964824
    (5120, 8192, 1, false, &[34..=35, 37..=37, 39..=39, 41..=41, 44..=44]), // 0.969690
    (5120, 15360, 1, false, &[17..=21, 23..=25]), // 0.960329
    (8192, 1280, 1, false, &[3..=3]), // 0.916558
    (16384, 1280, 1, false, &[]),
    (25600, 6144, 1, false, &[]),
    (32768, 1280, 1, false, &[]),
];

pub fn decode_path(g: &Geometry, q: Mxfp8FoldGemmQuery, rows: u32, paired: bool) -> DecodePath {
    // Shape matches do not qualify a different checkpoint geometry.
    if g.id != FLASH.id {
        return DecodePath::Ticket;
    }
    if small_query(g, q, rows).m_capacity == 32 {
        return DecodePath::Split;
    }
    if PREFETCH_WINS.iter().any(|(n, k, groups, pair, ranges)| {
        (q.n, q.k, q.groups, paired) == (*n, *k, *groups, *pair)
            && ranges.iter().any(|range| range.contains(&rows))
    }) {
        DecodePath::Prefetch
    } else {
        DecodePath::Ticket
    }
}
