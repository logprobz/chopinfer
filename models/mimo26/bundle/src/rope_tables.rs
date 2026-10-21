//! Immutable text RoPE tables shared by all layers of one rank.
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use model_mimo26_contract::geometry::Geometry;
#[cfg(test)]
use model_mimo26_contract::geometry::V26_FLASH;

// Each rotary pair stores an F32 cosine and sine.
pub fn bytes_per_position(geometry: &Geometry) -> u64 {
    geometry.rotary() as u64 * 4
}

#[cfg(test)]
const BYTES_PER_POSITION: u64 = V26_FLASH.rotary() as u64 * 4;

pub fn valid_context_for(geometry: &Geometry, tokens: u64) -> bool {
    (1..=geometry.context_tokens() as u64).contains(&tokens)
}

pub fn positions_fit_for(geometry: &Geometry, tokens: u64, positions: &[i32]) -> bool {
    valid_context_for(geometry, tokens) && positions.iter().all(|&p| p >= 0 && (p as u64) < tokens)
}

/// At most this many host threads build one table, each at least
/// `THREAD_POSITIONS` positions: a full-context table takes seconds on one.
const BUILD_THREADS: u64 = 16;
const THREAD_POSITIONS: u64 = 1 << 16;

fn build_for(geometry: &Geometry, start: u64, tokens: u64, theta: f64) -> Vec<u8> {
    let mut bytes = vec![0; (tokens * bytes_per_position(geometry)) as usize];
    let threads = std::thread::available_parallelism().map_or(1, |cpus| cpus.get() as u64);
    let positions = tokens
        .div_ceil(threads.clamp(1, BUILD_THREADS))
        .max(THREAD_POSITIONS);
    std::thread::scope(|scope| {
        for (chunk, out) in bytes
            .chunks_mut((positions * bytes_per_position(geometry)) as usize)
            .enumerate()
        {
            scope.spawn(move || fill_for(geometry, start + chunk as u64 * positions, theta, out));
        }
    });
    bytes
}

/// Fills `out` with the rows of consecutive positions from `first`.
fn fill_for(geometry: &Geometry, first: u64, theta: f64, out: &mut [u8]) {
    for (row, p) in out
        .chunks_exact_mut(bytes_per_position(geometry) as usize)
        .zip(first..)
    {
        for (pair, cell) in row.chunks_exact_mut(8).enumerate() {
            let (cos, sin) = kernel_core::rotary::ComputedTable {
                theta,
                rule: kernel_core::rotary::AngleRule::HfF32AngleBf16CosSin,
                axes: kernel_core::rotary::Axes::Linear,
            }
            .cos_sin([p as usize; 3], pair, geometry.rotary() as usize / 2)
            .expect("validated rotary geometry");
            cell[..4].copy_from_slice(&(cos as f32).to_le_bytes());
            cell[4..].copy_from_slice(&(sin as f32).to_le_bytes());
        }
    }
}

pub fn tables_for(geometry: &'static Geometry, tokens: u64) -> Arc<[Vec<u8>; 2]> {
    assert!(valid_context_for(geometry, tokens));
    type Tables = Weak<[Vec<u8>; 2]>;
    static CACHE: OnceLock<Mutex<BTreeMap<(&'static str, u64), Tables>>> = OnceLock::new();
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    cache.retain(|_, tables| tables.strong_count() != 0);
    if let Some(tables) = cache.get(&(geometry.id, tokens)).and_then(Weak::upgrade) {
        return tables;
    }
    let tables = Arc::new(
        [geometry.rope_theta(), geometry.swa_rope_theta()]
            .map(|theta| build_for(geometry, 0, tokens, theta)),
    );
    cache.insert((geometry.id, tokens), Arc::downgrade(&tables));
    tables
}

/// The table rows of `count` positions from `first`, built afresh.
#[cfg(test)]
pub fn rows_for(geometry: &Geometry, first: u64, count: u64, theta: f64) -> Vec<u8> {
    build_for(geometry, first, count, theta)
}

#[cfg(test)]
pub fn tables(tokens: u64) -> Arc<[Vec<u8>; 2]> {
    tables_for(&V26_FLASH, tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::{DType, RotaryStyle};
    use kernel_core::rotary::{
        AngleRule, Axes, ComputedTable, FreqTable, RotaryArithmetic, RotaryPositions,
        RotaryRounding, RotarySlice, Rotation,
    };

    /// HF's NeoX rotation of 64 values: BF16 cos and sin, one F32 mate
    /// product and one F32 FMA, then BF16 storage.
    const HF_ROPE: Rotation = Rotation {
        slice: RotarySlice { offset: 0, len: 64 },
        pairing: RotaryStyle::Neox,
        inverse: false,
        rounding: RotaryRounding {
            table: Some(DType::Bf16),
            arithmetic: RotaryArithmetic::F32Fma,
            output: Some(DType::Bf16),
        },
    };

    #[test]
    fn rotary_width_and_theta_follow_the_geometry() {
        let mut geometry = V26_FLASH;
        geometry.shape.global.rotary = 32;
        geometry.shape.global.theta = 5_000_000.0;
        let bytes = build_for(&geometry, 37, 2, geometry.rope_theta());
        assert_eq!(bytes.len(), 2 * 16 * 8);
        for (row, position) in [37, 38].into_iter().enumerate() {
            for pair in 0..16 {
                let inverse = (5_000_000.0_f32).powf(2.0 * pair as f32 / 32.0).recip();
                let angle = position as f32 * inverse;
                let expected = [
                    kernel_core::hf_rope::round_bf16(angle.cos()),
                    kernel_core::hf_rope::round_bf16(angle.sin()),
                ];
                for (plane, value) in expected.into_iter().enumerate() {
                    let offset = (row * 16 + pair) * 8 + plane * 4;
                    assert_eq!(&bytes[offset..offset + 4], &value.to_le_bytes());
                }
            }
        }
    }

    #[test]
    fn host_tables_are_released_after_upload_owners_drop() {
        let owner = tables(193);
        let lifetime = Arc::downgrade(&owner);
        let rank_upload = Arc::clone(&owner);
        drop(owner);
        assert!(lifetime.upgrade().is_some());
        drop(rank_upload);
        assert!(lifetime.upgrade().is_none());
    }

    #[test]
    fn threads_build_the_rows_one_thread_does() {
        let tokens = 3 * THREAD_POSITIONS + 5;
        let mut serial = vec![0; (tokens * BYTES_PER_POSITION) as usize];
        fill_for(&V26_FLASH, 1000, 1e7, &mut serial);
        assert!(build_for(&V26_FLASH, 1000, tokens, 1e7) == serial);
    }

    #[test]
    fn context_bound_and_exact_table_bytes() {
        for limit in [1, 127, 2048, 65536, 1048576] {
            assert!(positions_fit_for(&V26_FLASH, limit, &[0, limit as i32 - 1]));
            assert!(!positions_fit_for(&V26_FLASH, limit, &[limit as i32]));
            assert!(!positions_fit_for(&V26_FLASH, limit, &[-1]));
            assert_eq!(2 * limit * BYTES_PER_POSITION, limit * 512);
        }
        assert!(!valid_context_for(&V26_FLASH, 0));
        assert!(!valid_context_for(&V26_FLASH, 1048577));
        let table = tables(128);
        for (plane, theta) in [1e7, 1e4].into_iter().enumerate() {
            for p in [0, 36, 46, 127, 511, 4095, 65535, 262143, 1048575] {
                for i in 0..32 {
                    let bytes = build_for(&V26_FLASH, p, 1, theta);
                    let mut basis = vec![0.0; 64];
                    basis[i] = 1.0;
                    let frequencies = FreqTable::Computed(ComputedTable {
                        theta,
                        rule: AngleRule::HfF32AngleBf16CosSin,
                        axes: Axes::Linear,
                    });
                    let positions = RotaryPositions::Linear(&[p as usize]);
                    let oracle = HF_ROPE.apply(&basis, 64, positions, &frequencies).unwrap();
                    let expected = [oracle[i] as f32, oracle[i + 32] as f32];
                    assert_eq!(&bytes[i * 8..i * 8 + 4], &expected[0].to_le_bytes());
                    assert_eq!(&bytes[i * 8 + 4..i * 8 + 8], &expected[1].to_le_bytes());
                    if p < 128 {
                        let at = (p as usize * 32 + i) * 8;
                        assert_eq!(&table[plane][at..at + 4], &expected[0].to_le_bytes());
                        assert_eq!(&table[plane][at + 4..at + 8], &expected[1].to_le_bytes());
                    }
                }
            }
        }
    }
}
