//! Paged pools for the KV writer conformance cases: a pool is in-place
//! state whose pages hold old contents and decoy bytes, and a writer may
//! change only the slots its live rows name.
//!
//! Every page's first `used` bytes hold old contents that differ from page
//! to page in any [`OLD_PERIOD`] consecutive pages, and the bytes past them
//! to the stride hold [`DECOY`]. The oracle returns the whole pool with each
//! live row's slot replaced, so an untouched page, a decoy byte or a slot
//! past the live rows that changes fails the state's bit comparison, and the
//! write map ([`ranges`]) requires every written byte. The old contents
//! never equal a written row, so a high-id rerun whose writes land at the
//! old IDs over the moved pool leaves other bytes than the first run's.
//!
//! Row values are exact by construction: [`e4m3_value`] gives E4M3 values
//! (3 mantissa bits), which BF16 holds too, so a BF16 copy, a division by a
//! power-of-two scale and the E4M3 conversion all leave them exact. The
//! oracle checks every converted value with `hot_fp8::e4m3` and
//! [`bf16_exact`] and refuses one it does not hold exactly.
#![allow(dead_code)] // Shared by test binaries that use different parts.

use std::collections::BTreeSet;
use std::ops::Range;

use crate::hot_keys::DECOY;
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_lab::LabResult;
use kernel_lab::conformance::Images;

/// Pages after which a pool's old contents repeat, so a wide pool's image
/// is built from a short pattern.
pub const OLD_PERIOD: usize = 61;

/// The live counts of a wide case: 0, half and the capacity. A wide case
/// exists for its high-id rerun, and each count rebuilds its pools on the
/// host; the cases over small pools run every count of
/// `poison::live_counts`.
pub fn wide_counts(capacity: u32) -> Vec<u32> {
    vec![capacity, 0, capacity / 2]
}

/// splitmix64, for deterministic sign and value patterns.
pub fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// An old byte of a pool: never [`DECOY`] or the runner's 0xff poison.
fn old_byte(salt: u64, page: usize, byte: usize) -> u8 {
    1 + (mix(salt << 56 ^ (page as u64) << 32 | byte as u64) % 125) as u8
}

/// A pool of `pages` pages at `stride` bytes: each page's first `used`
/// bytes old contents, which differ by `salt`, the rest [`DECOY`].
pub fn old_pool(pages: usize, used: usize, stride: usize, salt: u64) -> Vec<u8> {
    assert!(used <= stride, "a page overflows its stride");
    let distinct = pages.min(OLD_PERIOD);
    let mut pattern = vec![DECOY; distinct * stride];
    for page in 0..distinct {
        for byte in 0..used {
            pattern[page * stride + byte] = old_byte(salt, page, byte);
        }
    }
    let mut image = Vec::with_capacity(pages * stride);
    while image.len() < pages * stride {
        let more = (pages * stride - image.len()).min(pattern.len());
        image.extend_from_slice(&pattern[..more]);
    }
    image
}

/// An E4M3 value from `h`: a sign, a mantissa of 3 bits and an exponent of
/// -3 to 2, so magnitudes from 1/8 to 7.5.
pub fn e4m3_value(h: u64) -> f32 {
    let sign = if h & 1 == 0 { 1.0 } else { -1.0 };
    let mantissa = (8 + (h >> 1) % 8) as f32 / 8.0;
    let exponent = ((h >> 4) % 6) as i32 - 3;
    sign * mantissa * 2f32.powi(exponent)
}

/// `value` as BF16, which it must hold exactly.
pub fn bf16_exact(value: f32) -> LabResult<u16> {
    let bits = f32_to_bf16(value);
    if bf16_to_f32(bits) != value || !value.is_finite() {
        return Err(format!("{value} is not a finite BF16 value"));
    }
    Ok(bits)
}

/// One live row's destination: the page its table names, as `read` maps
/// IDs, and its slot within the page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    pub row: usize,
    pub page: usize,
    pub slot: usize,
}

/// The slots of the live rows that write: a row whose position (from
/// `positions`) is negative writes nothing. Refuses a row whose table entry
/// is a hole and two rows of one slot, which would race.
pub fn slots(
    images: &Images,
    live: u32,
    positions: &str,
    page: usize,
    max_pages: usize,
    read: fn(i32) -> i32,
) -> LabResult<Vec<Slot>> {
    let batch = images.i32s("row_batch")?;
    let positions = images.i32s(positions)?;
    let tables = images.i32s("block_tables")?;
    let mut slots = Vec::new();
    let mut seen = BTreeSet::new();
    for row in 0..live as usize {
        let Ok(position) = usize::try_from(positions[row]) else {
            continue;
        };
        let item = usize::try_from(batch[row]).map_err(|_| format!("row {row} has no item"))?;
        if position / page >= max_pages {
            return Err(format!("row {row} position {position} lies past its table"));
        }
        let id = read(tables[item * max_pages + position / page]);
        let page_id = usize::try_from(id).map_err(|_| format!("row {row} names hole page {id}"))?;
        let slot = position % page;
        if !seen.insert((page_id, slot)) {
            return Err(format!("two rows write page {page_id} slot {slot}"));
        }
        slots.push(Slot {
            row,
            page: page_id,
            slot,
        });
    }
    Ok(slots)
}

/// `bytes` written into `pool` at `offset`; refused past the pool.
pub fn put(pool: &mut [u8], offset: usize, bytes: &[u8]) -> LabResult<()> {
    pool.get_mut(offset..offset + bytes.len())
        .ok_or_else(|| format!("a write at byte {offset} lies past the pool"))?
        .copy_from_slice(bytes);
    Ok(())
}

/// The byte ranges `(offset, len)` as a sorted range list.
pub fn ranges(spans: impl IntoIterator<Item = (usize, usize)>) -> Vec<Range<usize>> {
    let mut ranges = spans
        .into_iter()
        .map(|(offset, len)| offset..offset + len)
        .collect::<Vec<_>>();
    ranges.sort_by_key(|range| range.start);
    ranges
}
