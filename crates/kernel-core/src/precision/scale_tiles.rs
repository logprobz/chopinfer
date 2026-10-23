//! `scale-row-tiles-offset-v1`: a lossless encoding of one `[rows, groups]`
//! matrix of 1-byte block scales. A warp decodes one 64-row x 4-group tile
//! in place: its offsets occupy one contiguous region, and strip and tile
//! tables give its exception records directly.
//!
//! Each row uses exactly the base and exception decisions of
//! [`super::scale_offset`] at `width` (1, 2 or 4). A scale in the window of
//! `2^width` values from the base stores its offset; any other scale stores
//! offset 0 and an exception. Every byte value round-trips.
//!
//! A matrix has `S = ceil(rows / 64)` strips and `T = ceil(groups / 4)`
//! tiles per strip, ordered by strip, then group tile. The last strip or
//! tile may be smaller, with `r_s` rows and `g_k` groups. The block holds:
//!
//! - `rows` base bytes, padded with zero bytes to a multiple of 4.
//! - One offset region per tile, row-major within the tile: local row `rl`
//!   and group `gl` start at bit `(rl * g_k + gl) * width`, low bits first.
//!   Each region occupies `align4(ceil(r_s * g_k * width / 8))` bytes,
//!   padded with zero bits. A full tile uses 32, 64 or 128 bytes.
//! - `S + 1` little-endian `u32` strip starts, in record indices, including
//!   the total record count `n` at the end.
//! - `S * T` little-endian `u16` tile starts, relative to their strip's
//!   first record, then zero padding to a multiple of 4 bytes. A tile ends
//!   at the next tile's start, or at its strip's end for the last tile.
//! - `n` little-endian `u16` records, `(local << 8) | value`, where
//!   `local = rl * 4 + gl` even in a narrow tile. Records follow tile order
//!   and strictly increasing `local` within each tile, then zero padding
//!   to a multiple of 4 bytes.
//!
//! Shapes follow v1's bounds, with at most 1024 groups. Before a strip's
//! last tile there can be at most `64 * 1020` records, so every tile start
//! fits a `u16`. A bank uses [`super::scale_offset::bank_layout`] unchanged:
//! every block occupies a multiple of 4 bytes.

use super::scale_offset::{self, WIDTHS};
use std::fmt;

pub const FORMAT: &str = "scale-row-tiles-offset-v1";
pub const STRIP_ROWS: u64 = 64;
pub const TILE_GROUPS: u64 = 4;
pub const GROUPS_MAX: u64 = 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScaleTilesError(pub String);

impl fmt::Display for ScaleTilesError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{FORMAT}: {}", self.0)
    }
}

impl std::error::Error for ScaleTilesError {}

impl From<scale_offset::ScaleOffsetError> for ScaleTilesError {
    fn from(error: scale_offset::ScaleOffsetError) -> Self {
        Self(error.0)
    }
}

fn fail<T>(reason: impl Into<String>) -> Result<T, ScaleTilesError> {
    Err(ScaleTilesError(reason.into()))
}

fn tile_bytes(rows: u64, groups: u64, width: u32) -> u64 {
    (rows * groups * u64::from(width))
        .div_ceil(8)
        .next_multiple_of(4)
}

struct Layout {
    strips: usize,
    tiles: usize,
    plane: usize,
    strip_table: usize,
    tile_table: usize,
    fixed: usize,
}

impl Layout {
    fn new(rows: u64, groups: u64, width: u32) -> Result<Self, ScaleTilesError> {
        scale_offset::check_width(width)?;
        scale_offset::check_shape(rows, groups)?;
        if groups > GROUPS_MAX {
            return fail(format!("{groups} groups exceed {GROUPS_MAX}"));
        }
        let strips = rows.div_ceil(STRIP_ROWS);
        let tiles = groups.div_ceil(TILE_GROUPS);
        let strip_bytes = |r| {
            groups / TILE_GROUPS * tile_bytes(r, TILE_GROUPS, width)
                + tile_bytes(r, groups % TILE_GROUPS, width)
        };
        let plane = rows.next_multiple_of(4);
        let strip_table =
            plane + rows / STRIP_ROWS * strip_bytes(STRIP_ROWS) + strip_bytes(rows % STRIP_ROWS);
        let tile_table = strip_table + (strips + 1) * 4;
        let fixed = tile_table + (strips * tiles * 2).next_multiple_of(4);
        Ok(Self {
            strips: strips as usize,
            tiles: tiles as usize,
            plane: plane as usize,
            strip_table: strip_table as usize,
            tile_table: tile_table as usize,
            fixed: fixed as usize,
        })
    }
}

/// Bytes of the bases, tile offset regions, strip and tile tables, and
/// padding before the exception records.
pub fn fixed_bytes(rows: u64, groups: u64, width: u32) -> Result<u64, ScaleTilesError> {
    Ok(Layout::new(rows, groups, width)?.fixed as u64)
}

/// Bytes of one encoded matrix with `exceptions` records.
pub fn encoded_bytes(
    rows: u64,
    groups: u64,
    width: u32,
    exceptions: u64,
) -> Result<u64, ScaleTilesError> {
    if exceptions > rows.saturating_mul(groups) {
        return fail("more exceptions than scales");
    }
    Ok(fixed_bytes(rows, groups, width)? + (2 * exceptions).next_multiple_of(4))
}

fn slot(width: u32) -> usize {
    WIDTHS
        .iter()
        .position(|w| *w == width)
        .expect("checked width")
}

/// Encodes a `[rows, groups]` matrix at `width` into `out` (cleared first),
/// returning its exception count.
pub fn encode_into(
    scales: &[u8],
    rows: u64,
    groups: u64,
    width: u32,
    out: &mut Vec<u8>,
) -> Result<u64, ScaleTilesError> {
    let layout = Layout::new(rows, groups, width)?;
    if scales.len() as u64 != rows * groups {
        return fail("scale bytes differ from the matrix shape");
    }
    let slot = slot(width);
    let mask = (1_u16 << width) - 1;
    out.clear();
    out.resize(layout.fixed, 0);
    for (base, row) in out.iter_mut().zip(scales.chunks_exact(groups as usize)) {
        *base = scale_offset::row_bases(row).0[slot];
    }
    let mut plane = layout.plane;
    let mut records = 0_u32;
    for s in 0..layout.strips {
        let row_start = s as u64 * STRIP_ROWS;
        let strip_rows = (rows - row_start).min(STRIP_ROWS);
        let strip_start = records;
        let at = layout.strip_table + s * 4;
        out[at..at + 4].copy_from_slice(&strip_start.to_le_bytes());
        for k in 0..layout.tiles {
            let group_start = k as u64 * TILE_GROUPS;
            let tile_groups = (groups - group_start).min(TILE_GROUPS);
            let at = layout.tile_table + (s * layout.tiles + k) * 2;
            out[at..at + 2].copy_from_slice(&((records - strip_start) as u16).to_le_bytes());
            for rl in 0..strip_rows {
                let row = row_start + rl;
                let base = out[row as usize];
                for gl in 0..tile_groups {
                    let scale = scales[(row * groups + group_start + gl) as usize];
                    let offset = u16::from(scale).wrapping_sub(u16::from(base));
                    if scale >= base && offset <= mask {
                        let bit = ((rl * tile_groups + gl) * u64::from(width)) as usize;
                        out[plane + bit / 8] |= (offset as u8) << (bit % 8);
                    } else {
                        let local = (rl * TILE_GROUPS + gl) as u16;
                        out.extend_from_slice(&((local << 8) | u16::from(scale)).to_le_bytes());
                        records += 1;
                    }
                }
            }
            plane += tile_bytes(strip_rows, tile_groups, width) as usize;
        }
    }
    let at = layout.strip_table + layout.strips * 4;
    out[at..at + 4].copy_from_slice(&records.to_le_bytes());
    out.resize(out.len().next_multiple_of(4), 0);
    Ok(u64::from(records))
}

/// Encodes a `[rows, groups]` matrix at `width`.
pub fn encode(
    scales: &[u8],
    rows: u64,
    groups: u64,
    width: u32,
) -> Result<Vec<u8>, ScaleTilesError> {
    let mut out = Vec::new();
    encode_into(scales, rows, groups, width, &mut out)?;
    Ok(out)
}

fn read_u32(block: &[u8], at: usize) -> usize {
    u32::from_le_bytes(block[at..at + 4].try_into().expect("4 bytes")) as usize
}

fn read_u16(block: &[u8], at: usize) -> usize {
    u16::from_le_bytes(block[at..at + 2].try_into().expect("2 bytes")) as usize
}

/// Decodes one encoded matrix: the CPU oracle of in-place tile decoding.
pub fn decode(
    block: &[u8],
    rows: u64,
    groups: u64,
    width: u32,
) -> Result<Vec<u8>, ScaleTilesError> {
    let layout = Layout::new(rows, groups, width)?;
    if block.len() < layout.fixed {
        return fail("block bytes do not fit the matrix shape");
    }
    let records = read_u32(block, layout.strip_table + layout.strips * 4);
    if block.len() as u64 != encoded_bytes(rows, groups, width, records as u64)? {
        return fail("block bytes do not fit the matrix shape");
    }
    if read_u32(block, layout.strip_table) != 0 {
        return fail("strip table does not start at zero");
    }
    let mut scales = vec![0; (rows * groups) as usize];
    let mask = (1_u8 << width) - 1;
    let mut plane = layout.plane;
    for s in 0..layout.strips {
        let strip_start = read_u32(block, layout.strip_table + s * 4);
        let strip_end = read_u32(block, layout.strip_table + (s + 1) * 4);
        if strip_start > strip_end || strip_end > records {
            return fail("strip table decreases or exceeds the records");
        }
        let strip_records = strip_end - strip_start;
        let row_start = s as u64 * STRIP_ROWS;
        let strip_rows = (rows - row_start).min(STRIP_ROWS);
        let table = layout.tile_table + s * layout.tiles * 2;
        if read_u16(block, table) != 0 {
            return fail("a strip's tile table does not start at zero");
        }
        for k in 0..layout.tiles {
            let start = read_u16(block, table + k * 2);
            let end = if k + 1 == layout.tiles {
                strip_records
            } else {
                read_u16(block, table + (k + 1) * 2)
            };
            if start > end || end > strip_records {
                return fail("tile table decreases or exceeds the strip's records");
            }
            let group_start = k as u64 * TILE_GROUPS;
            let tile_groups = (groups - group_start).min(TILE_GROUPS);
            for rl in 0..strip_rows {
                let row = row_start + rl;
                for gl in 0..tile_groups {
                    let bit = ((rl * tile_groups + gl) * u64::from(width)) as usize;
                    scales[(row * groups + group_start + gl) as usize] = block[row as usize]
                        .wrapping_add((block[plane + bit / 8] >> (bit % 8)) & mask);
                }
            }
            let mut previous = None;
            let first = layout.fixed + (strip_start + start) * 2;
            let last = layout.fixed + (strip_start + end) * 2;
            for record in block[first..last].chunks_exact(2) {
                let local = u64::from(record[1]);
                let (rl, gl) = (local / TILE_GROUPS, local % TILE_GROUPS);
                if rl >= strip_rows || gl >= tile_groups || previous.is_some_and(|p| p >= local) {
                    return fail("exception records out of order or outside the tile");
                }
                previous = Some(local);
                scales[((row_start + rl) * groups + group_start + gl) as usize] = record[0];
            }
            plane += tile_bytes(strip_rows, tile_groups, width) as usize;
        }
    }
    Ok(scales)
}

/// Decodes matrix `matrix` of a bank laid out by [`super::scale_offset::bank_layout`].
pub fn decode_bank_matrix(
    bank: &[u8],
    matrix: usize,
    rows: u64,
    groups: u64,
    width: u32,
) -> Result<Vec<u8>, ScaleTilesError> {
    decode(scale_offset::bank_block(bank, matrix)?, rows, groups, width)
}

/// Each tile's exception count at `width`, in strip-major tile order,
/// without encoding the matrix.
pub fn tile_exceptions(
    scales: &[u8],
    rows: u64,
    groups: u64,
    width: u32,
) -> Result<Vec<u32>, ScaleTilesError> {
    let layout = Layout::new(rows, groups, width)?;
    if scales.len() as u64 != rows * groups {
        return fail("scale bytes differ from the matrix shape");
    }
    let slot = slot(width);
    let mask = (1_u16 << width) - 1;
    let mut counts = vec![0; layout.strips * layout.tiles];
    for (r, row) in scales.chunks_exact(groups as usize).enumerate() {
        let base = scale_offset::row_bases(row).0[slot];
        for (g, &scale) in row.iter().enumerate() {
            let offset = u16::from(scale).wrapping_sub(u16::from(base));
            if scale < base || offset > mask {
                counts[r / STRIP_ROWS as usize * layout.tiles + g / TILE_GROUPS as usize] += 1;
            }
        }
    }
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic byte stream for test matrices.
    fn noise(seed: u64, count: usize) -> Vec<u64> {
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            })
            .collect()
    }

    fn round_trip(scales: &[u8], rows: u64, groups: u64) {
        let counts = scale_offset::census(scales, rows, groups).unwrap();
        for (slot, &width) in WIDTHS.iter().enumerate() {
            let block = encode(scales, rows, groups, width).unwrap();
            assert_eq!(
                block.len() as u64,
                encoded_bytes(rows, groups, width, counts[slot]).unwrap(),
                "{rows} x {groups}, width {width}"
            );
            let decoded = decode(&block, rows, groups, width).unwrap();
            assert_eq!(decoded, scales, "{rows} x {groups}, width {width}");
            let v1 = scale_offset::encode(scales, rows, groups, width).unwrap();
            assert_eq!(
                decoded,
                scale_offset::decode(&v1, rows, groups, width).unwrap()
            );
            assert_eq!(&block[..rows as usize], &v1[..rows as usize]);
            let tiles = tile_exceptions(scales, rows, groups, width).unwrap();
            assert_eq!(tiles.len() as u64, rows.div_ceil(64) * groups.div_ceil(4));
            assert_eq!(
                tiles.iter().map(|&n| u64::from(n)).sum::<u64>(),
                counts[slot]
            );
            assert!(tiles.iter().all(|&n| n <= 256));
            let mut reused = vec![255; block.len() + 8];
            assert_eq!(
                encode_into(scales, rows, groups, width, &mut reused).unwrap(),
                counts[slot]
            );
            assert_eq!(reused, block);
        }
    }

    #[test]
    fn matrix_families_round_trip_at_full_and_partial_tiles() {
        for (rows, groups) in [
            (64, 160),
            (576, 160),
            (5120, 18),
            (65, 18),
            (1, 1),
            (63, 5),
            (130, 7),
            (64, 1024),
        ] {
            let draws = noise(1, (rows * groups) as usize);
            let clustered: Vec<u8> = draws
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    let base = 118 + (i as u64 / groups % 5) as u8;
                    base + [0, 0, 1, 1, 1, 0, 2, 0][(d % 8) as usize]
                })
                .collect();
            round_trip(&clustered, rows, groups);
            let random: Vec<u8> = draws.iter().map(|d| *d as u8).collect();
            round_trip(&random, rows, groups);
            let heavy: Vec<u8> = draws
                .iter()
                .map(|d| [0, 32, 64, 96, 128, 160, 192, 255][(d % 8) as usize])
                .collect();
            round_trip(&heavy, rows, groups);
        }
    }

    #[test]
    fn every_byte_value_round_trips() {
        let scales: Vec<u8> = (0..=255).collect();
        round_trip(&scales, 1, 256);
        round_trip(&scales, 16, 16);
        round_trip(&scales, 256, 1);
        let wide: Vec<u8> = noise(2, 18 * 40).iter().map(|d| *d as u8).collect();
        round_trip(&wide, 40, 18);
    }

    #[test]
    fn constant_and_single_group_rows_have_no_exceptions() {
        let constant = vec![127; 7 * 33];
        round_trip(&constant, 7, 33);
        let single: Vec<u8> = (0..50).map(|i| (i * 5) as u8).collect();
        round_trip(&single, 50, 1);
        for width in WIDTHS {
            assert_eq!(tile_exceptions(&constant, 7, 33, width).unwrap(), [0; 9]);
            assert_eq!(tile_exceptions(&single, 50, 1, width).unwrap(), [0]);
        }
    }

    #[test]
    fn eighteen_group_slices_round_trip() {
        let scales: Vec<u8> = noise(3, 5120 * 18)
            .iter()
            .map(|d| 120 + [0, 1, 0, 1, 3, 0, 1, 7][(d % 8) as usize])
            .collect();
        round_trip(&scales, 5120, 18);
    }

    #[test]
    fn full_exception_tiles_keep_all_256_records() {
        // Neighboring tiles choose base 10 for every row.
        let scales: Vec<u8> = (0..130 * 12)
            .map(|i| if (4..8).contains(&(i % 12)) { 200 } else { 10 })
            .collect();
        for width in WIDTHS {
            assert_eq!(
                tile_exceptions(&scales, 130, 12, width).unwrap(),
                [0, 256, 0, 0, 256, 0, 0, 8, 0]
            );
        }
        round_trip(&scales, 130, 12);
    }

    #[test]
    fn more_than_32_records_stay_in_their_tile() {
        let mut scales = vec![10; 64 * 12];
        for r in 0..40 {
            scales[r * 12 + 4] = 200;
        }
        for width in WIDTHS {
            assert_eq!(tile_exceptions(&scales, 64, 12, width).unwrap(), [0, 40, 0]);
        }
        round_trip(&scales, 64, 12);
    }

    const SMALL_SCALES: [u8; 15] = [10, 11, 10, 200, 11, 20, 21, 20, 21, 90, 30, 30, 31, 31, 30];
    const SMALL_BLOCK: [u8; 28] = [
        10, 20, 30, 0, // Bases and padding.
        0xa2, 0x0c, 0, 0, 1, 0, 0, 0, // The 3 x 4 and 3 x 1 offset regions.
        0, 0, 0, 0, 2, 0, 0, 0, // Strip starts and total record count.
        0, 0, 1, 0, // Tile starts relative to the strip.
        200, 3, 90, 4, // Local indices use stride 4 in both tiles.
    ];

    #[test]
    fn small_matrix_has_the_exact_byte_layout() {
        assert_eq!(fixed_bytes(3, 5, 1).unwrap(), 24);
        assert_eq!(encode(&SMALL_SCALES, 3, 5, 1).unwrap(), SMALL_BLOCK);
        assert_eq!(decode(&SMALL_BLOCK, 3, 5, 1).unwrap(), SMALL_SCALES);
    }

    #[test]
    fn block_length_includes_exactly_the_record_padding() {
        let mut scales = SMALL_SCALES;
        scales[9] = 20;
        let block = encode(&scales, 3, 5, 1).unwrap();
        assert_eq!(block.len(), 28);
        assert_eq!(&block[24..], &[200, 3, 0, 0]);
        for end in 0..block.len() {
            assert!(decode(&block[..end], 3, 5, 1).is_err(), "length {end}");
        }
        for extra in 1..=4 {
            let mut padded = block.clone();
            padded.resize(block.len() + extra, 0);
            assert!(decode(&padded, 3, 5, 1).is_err());
        }
        let mut padded = block;
        for at in [3, 6, 7, 9, 10, 11, 26, 27] {
            padded[at] = 255;
        }
        padded[5] |= 0xf0;
        padded[8] |= 0xf8;
        assert_eq!(decode(&padded, 3, 5, 1).unwrap(), scales);
        let mut table_padding = encode(&[9], 1, 1, 1).unwrap();
        table_padding[18..20].fill(255);
        assert_eq!(decode(&table_padding, 1, 1, 1).unwrap(), [9]);
    }

    #[test]
    fn strip_tables_must_start_at_zero_and_stay_within_the_records() {
        let mut bad = SMALL_BLOCK;
        bad[12] = 1;
        assert!(decode(&bad, 3, 5, 1).is_err());
        bad = SMALL_BLOCK;
        bad[16] = 3;
        assert!(decode(&bad, 3, 5, 1).is_err());

        let mut scales = vec![10; 130 * 12];
        scales[4] = 200;
        scales[64 * 12 + 4] = 200;
        let block = encode(&scales, 130, 12, 1).unwrap();
        let layout = Layout::new(130, 12, 1).unwrap();
        let mut bad = block.clone();
        bad[layout.strip_table + 4..layout.strip_table + 8].copy_from_slice(&2_u32.to_le_bytes());
        bad[layout.strip_table + 8..layout.strip_table + 12].copy_from_slice(&1_u32.to_le_bytes());
        assert!(decode(&bad, 130, 12, 1).is_err());
        let mut bad = block;
        bad[layout.strip_table + 4..layout.strip_table + 8].copy_from_slice(&3_u32.to_le_bytes());
        assert!(decode(&bad, 130, 12, 1).is_err());
    }

    #[test]
    fn tile_tables_must_start_at_zero_and_stay_within_the_strip() {
        for (at, value) in [(20, 1), (22, 3)] {
            let mut bad = SMALL_BLOCK;
            bad[at] = value;
            assert!(decode(&bad, 3, 5, 1).is_err());
        }
        let scales = [10, 10, 10, 200, 10, 10, 10, 10, 10];
        let mut bad = encode(&scales, 1, 9, 1).unwrap();
        let layout = Layout::new(1, 9, 1).unwrap();
        bad[layout.tile_table + 4..layout.tile_table + 6].copy_from_slice(&0_u16.to_le_bytes());
        assert!(decode(&bad, 1, 9, 1).is_err());
    }

    #[test]
    fn records_must_be_strictly_ordered_and_inside_the_tile() {
        let scales = [10, 200, 10, 200, 10, 10, 10, 10];
        let block = encode(&scales, 2, 4, 1).unwrap();
        let fixed = fixed_bytes(2, 4, 1).unwrap() as usize;
        let mut bad = block.clone();
        bad[fixed..].rotate_left(2);
        assert!(decode(&bad, 2, 4, 1).is_err());
        let mut bad = block;
        bad[fixed + 3] = bad[fixed + 1];
        assert!(decode(&bad, 2, 4, 1).is_err());
        for (at, local) in [(25, 12), (27, 5)] {
            let mut bad = SMALL_BLOCK;
            bad[at] = local;
            assert!(decode(&bad, 3, 5, 1).is_err());
        }
    }

    #[test]
    fn strip_record_totals_can_exceed_u16() {
        // All 65536 positions can have records; only tile starts are u16.
        let layout = Layout::new(64, 1024, 1).unwrap();
        let mut block = vec![0; layout.fixed];
        block[layout.strip_table + 4..layout.strip_table + 8]
            .copy_from_slice(&65536_u32.to_le_bytes());
        for k in 0..256 {
            let at = layout.tile_table + k * 2;
            block[at..at + 2].copy_from_slice(&((k * 256) as u16).to_le_bytes());
            for local in 0..256_u16 {
                block.extend_from_slice(&((local << 8) | 255).to_le_bytes());
            }
        }
        assert_eq!(decode(&block, 64, 1024, 1).unwrap(), vec![255; 64 * 1024]);
    }

    #[test]
    fn shapes_widths_and_counts_are_checked() {
        for (rows, groups, width) in [
            (0, 1, 1),
            (1, 0, 1),
            (1, 1025, 1),
            (1, 1, 3),
            (1 << 24, 2, 1),
            (u64::MAX, 2, 1),
        ] {
            assert!(fixed_bytes(rows, groups, width).is_err());
            assert!(encode(&[], rows, groups, width).is_err());
            assert!(decode(&[], rows, groups, width).is_err());
            assert!(tile_exceptions(&[], rows, groups, width).is_err());
        }
        assert!(encoded_bytes(1, 1, 1, 2).is_err());
        assert!(encoded_bytes(u64::MAX, u64::MAX, 1, u64::MAX).is_err());
        assert!(encode(&[1, 2], 1, 3, 1).is_err());
        assert!(tile_exceptions(&[1, 2], 1, 3, 1).is_err());
        assert!(
            encode(&[1], 1, 1, 3)
                .unwrap_err()
                .to_string()
                .starts_with(FORMAT)
        );
    }

    #[test]
    fn banks_hold_three_different_matrix_shapes() {
        let matrices = [
            (SMALL_SCALES.to_vec(), 3, 5, 1),
            (
                noise(4, 65 * 18).iter().map(|d| *d as u8).collect(),
                65,
                18,
                2,
            ),
            (vec![255], 1, 1, 4),
        ];
        let blocks: Vec<_> = matrices
            .iter()
            .map(|(scales, rows, groups, width)| encode(scales, *rows, *groups, *width).unwrap())
            .collect();
        let sizes: Vec<_> = blocks.iter().map(|block| block.len() as u64).collect();
        let layout = scale_offset::bank_layout(&sizes).unwrap();
        let mut bank = layout.table.clone();
        for (block, &offset) in blocks.iter().zip(&layout.offsets) {
            assert_eq!(bank.len() as u64, offset);
            bank.extend_from_slice(block);
        }
        assert_eq!(bank.len() as u64, layout.bytes);
        for (matrix, (scales, rows, groups, width)) in matrices.iter().enumerate() {
            assert_eq!(
                decode_bank_matrix(&bank, matrix, *rows, *groups, *width).unwrap(),
                *scales
            );
        }
        assert!(decode_bank_matrix(&[], 0, 1, 1, 1).is_err());
        bank[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode_bank_matrix(&bank, 0, 3, 5, 1).is_err());
    }
}
