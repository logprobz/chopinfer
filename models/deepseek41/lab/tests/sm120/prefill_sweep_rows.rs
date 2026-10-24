//! Deterministic coverage of packed prefill counts and tile boundaries.
use std::collections::BTreeSet;

pub const PACKED: [u32; 7] = [446, 789, 1803, 1808, 3658, 4542, 7754];
pub const TIMED: [u32; 12] = [
    320, 446, 640, 789, 1024, 1803, 1808, 2497, 3658, 4542, 7754, 8192,
];

pub fn rows(first: u32) -> Vec<u32> {
    assert!((257..=320).contains(&first));
    let mut rows: BTreeSet<u32> = (first..=640).collect();
    for center in (first.div_ceil(32) * 32..=8192).step_by(32) {
        for row in [center - 1, center, center + 1] {
            if (first..=8192).contains(&row) {
                rows.insert(row);
            }
        }
    }
    rows.extend(PACKED);
    rows.extend(TIMED);
    let mut seed = 0x58a3_741bu32;
    for _ in 0..64 {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        rows.insert(first + seed % (8193 - first));
    }
    rows.into_iter().collect()
}

#[test]
fn sweep_covers_packed_counts_and_both_sides_of_every_tile_boundary() {
    for first in [257, 320] {
        let rows: BTreeSet<_> = rows(first).into_iter().collect();
        assert!(rows.len() > 1000);
        assert!(rows.contains(&first) && rows.contains(&8192));
        assert!((first..=640).all(|r| rows.contains(&r)));
        assert!(PACKED.into_iter().chain(TIMED).all(|r| rows.contains(&r)));
        for center in (first.div_ceil(32) * 32..=8192).step_by(32) {
            for row in [center - 1, center, center + 1] {
                if (first..=8192).contains(&row) {
                    assert!(rows.contains(&row));
                }
            }
        }
    }
}
