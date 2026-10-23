//! `scale-row-offset-v1`: a lossless encoding of one `[rows, groups]` matrix
//! of 1-byte block scales, for scale planes that stay compressed in device
//! memory and expand on demand.
//!
//! Each row stores one base byte and, for every scale, its offset from the
//! base in `width` bits (1, 2 or 4). A scale outside the window of `2^width`
//! values from the base stores offset 0 and an exception record,
//! `index << 8 | value` as a little-endian `u32`, with
//! `index = row * groups + group`. Records are in
//! increasing index order. The base of a row is the value that puts the most
//! of its scales in the window, ties going to the lower base.
//!
//! The encoded block of one matrix is the bases (`rows` bytes), the offset
//! plane (`rows * ceil(groups * width / 8)` bytes, each row starting on a
//! byte, offset `g` at bits `g * width` of its row, low bits first), zero
//! padding to a multiple of 4 bytes, then the exception records.
//!
//! The codec never interprets a code: every byte value round-trips. A
//! matrix's [`ScaleCode`] only records what the bytes are (UE8M0 for MXFP4
//! and MXFP8, E4M3 for NVFP4, whose non-negative codes sort in value order);
//! payload validation of the original codes stays with the operand checks
//! that run before encoding.
//!
//! A bank of matrices is laid out by [`bank_layout`]: a little-endian `u32`
//! table of each block's byte offset within the bank, plus the end, padded
//! to 16 bytes, then the blocks in matrix order.

use std::fmt;

pub const FORMAT: &str = "scale-row-offset-v1";
/// The offset widths, in the order [`census`] reports them.
pub const WIDTHS: [u32; 3] = [1, 2, 4];
/// The most scales one matrix may hold: an exception index is 24 bits.
pub const MATRIX_SCALES_MAX: u64 = 1 << 24;
/// The bank table's alignment.
pub const TABLE_ALIGNMENT: u64 = 16;

/// What the encoded bytes are. The codec treats both alike.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum ScaleCode {
    Ue8m0,
    E4m3,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScaleOffsetError(pub String);

impl fmt::Display for ScaleOffsetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{FORMAT}: {}", self.0)
    }
}

impl std::error::Error for ScaleOffsetError {}

fn fail<T>(reason: impl Into<String>) -> Result<T, ScaleOffsetError> {
    Err(ScaleOffsetError(reason.into()))
}

fn check_width(width: u32) -> Result<(), ScaleOffsetError> {
    if WIDTHS.contains(&width) {
        Ok(())
    } else {
        fail(format!("offset width {width} is not 1, 2 or 4"))
    }
}

fn check_shape(rows: u64, groups: u64) -> Result<(), ScaleOffsetError> {
    if rows == 0 || groups == 0 {
        return fail("an empty matrix");
    }
    match rows.checked_mul(groups) {
        Some(scales) if scales <= MATRIX_SCALES_MAX => Ok(()),
        _ => fail(format!(
            "a {rows} x {groups} matrix holds more than {MATRIX_SCALES_MAX} scales"
        )),
    }
}

/// Bytes of one row of the offset plane.
#[must_use]
pub const fn plane_row_bytes(groups: u64, width: u32) -> u64 {
    (groups * width as u64).div_ceil(8)
}

/// Bytes of the bases, the offset plane and the padding before the
/// exception records.
pub fn fixed_bytes(rows: u64, groups: u64, width: u32) -> Result<u64, ScaleOffsetError> {
    check_width(width)?;
    check_shape(rows, groups)?;
    Ok((rows + rows * plane_row_bytes(groups, width)).next_multiple_of(4))
}

/// Bytes of one encoded matrix with `exceptions` records.
pub fn encoded_bytes(
    rows: u64,
    groups: u64,
    width: u32,
    exceptions: u64,
) -> Result<u64, ScaleOffsetError> {
    if exceptions > rows.saturating_mul(groups) {
        return fail("more exceptions than scales");
    }
    Ok(fixed_bytes(rows, groups, width)? + 4 * exceptions)
}

/// One row's base for each width and the exceptions each leaves.
fn row_bases(row: &[u8]) -> ([u8; 3], [u64; 3]) {
    let (mut low, mut high) = (u8::MAX, u8::MIN);
    for &scale in row {
        low = low.min(scale);
        high = high.max(scale);
    }
    let span = usize::from(high - low) + 1;
    let mut bases = [low; 3];
    let mut exceptions = [0; 3];
    if span <= 2 {
        return (bases, exceptions);
    }
    // Counts of each value above the row's lowest.
    let mut counts = [0_u32; 256];
    for &scale in row {
        counts[usize::from(scale - low)] += 1;
    }
    for (slot, width) in WIDTHS.iter().enumerate() {
        let window = 1_usize << width;
        if span <= window {
            continue;
        }
        let mut inside: u32 = counts[..window].iter().sum();
        let (mut best, mut best_start) = (inside, 0);
        for start in 1..=span - window {
            inside = inside + counts[start + window - 1] - counts[start - 1];
            if inside > best {
                (best, best_start) = (inside, start);
            }
        }
        bases[slot] = low + best_start as u8;
        exceptions[slot] = (row.len() as u32 - best) as u64;
    }
    (bases, exceptions)
}

/// The exceptions each width of [`WIDTHS`] leaves in a `[rows, groups]`
/// matrix: what the encoded size needs before the matrix is encoded.
pub fn census(scales: &[u8], rows: u64, groups: u64) -> Result<[u64; 3], ScaleOffsetError> {
    check_shape(rows, groups)?;
    if scales.len() as u64 != rows * groups {
        return fail("scale bytes differ from the matrix shape");
    }
    let mut totals = [0; 3];
    for row in scales.chunks_exact(groups as usize) {
        let (_, exceptions) = row_bases(row);
        for (total, count) in totals.iter_mut().zip(exceptions) {
            *total += count;
        }
    }
    Ok(totals)
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
) -> Result<u64, ScaleOffsetError> {
    let fixed = fixed_bytes(rows, groups, width)?;
    if scales.len() as u64 != rows * groups {
        return fail("scale bytes differ from the matrix shape");
    }
    let slot = slot(width);
    let row_bytes = plane_row_bytes(groups, width) as usize;
    let mask = (1_u16 << width) - 1;
    out.clear();
    out.resize(fixed as usize, 0);
    let (bases, plane) = out.split_at_mut(rows as usize);
    let mut records = Vec::new();
    for (r, row) in scales.chunks_exact(groups as usize).enumerate() {
        let base = row_bases(row).0[slot];
        bases[r] = base;
        let plane_row = &mut plane[r * row_bytes..(r + 1) * row_bytes];
        for (g, &scale) in row.iter().enumerate() {
            let offset = u16::from(scale).wrapping_sub(u16::from(base));
            if scale >= base && offset <= mask {
                let bit = g * width as usize;
                plane_row[bit / 8] |= (offset as u8) << (bit % 8);
            } else {
                let index = (r * groups as usize + g) as u32;
                records.push((index << 8) | u32::from(scale));
            }
        }
    }
    out.reserve(records.len() * 4);
    for record in &records {
        out.extend_from_slice(&record.to_le_bytes());
    }
    Ok(records.len() as u64)
}

/// Encodes a `[rows, groups]` matrix at `width`.
pub fn encode(
    scales: &[u8],
    rows: u64,
    groups: u64,
    width: u32,
) -> Result<Vec<u8>, ScaleOffsetError> {
    let mut out = Vec::new();
    encode_into(scales, rows, groups, width, &mut out)?;
    Ok(out)
}

/// Decodes one encoded matrix: the CPU oracle of the device expansion.
pub fn decode(
    block: &[u8],
    rows: u64,
    groups: u64,
    width: u32,
) -> Result<Vec<u8>, ScaleOffsetError> {
    let fixed = fixed_bytes(rows, groups, width)? as usize;
    if block.len() < fixed || (block.len() - fixed) % 4 != 0 {
        return fail("block bytes do not fit the matrix shape");
    }
    let row_bytes = plane_row_bytes(groups, width) as usize;
    let mask = (1_u8 << width) - 1;
    let (bases, rest) = block.split_at(rows as usize);
    let mut scales = vec![0; (rows * groups) as usize];
    for (r, row) in scales.chunks_exact_mut(groups as usize).enumerate() {
        let plane_row = &rest[r * row_bytes..(r + 1) * row_bytes];
        for (g, scale) in row.iter_mut().enumerate() {
            let bit = g * width as usize;
            *scale = bases[r].wrapping_add((plane_row[bit / 8] >> (bit % 8)) & mask);
        }
    }
    let mut previous = None;
    for record in block[fixed..].chunks_exact(4) {
        let record = u32::from_le_bytes(record.try_into().expect("4 bytes"));
        let index = (record >> 8) as usize;
        if index >= scales.len() || previous.is_some_and(|p| p >= index) {
            return fail("exception records out of order or outside the matrix");
        }
        previous = Some(index);
        scales[index] = record as u8;
    }
    Ok(scales)
}

/// A bank of encoded matrices: the offset table and each block's place.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BankLayout {
    /// Little-endian `u32` offsets of each block, then the bank's end.
    pub table: Vec<u8>,
    /// Each block's byte offset within the bank.
    pub offsets: Vec<u64>,
    pub bytes: u64,
}

/// Lays out blocks of `block_bytes` each (each a multiple of 4) after the
/// offset table.
pub fn bank_layout(block_bytes: &[u64]) -> Result<BankLayout, ScaleOffsetError> {
    if block_bytes.is_empty() {
        return fail("an empty bank");
    }
    let table_bytes = ((block_bytes.len() as u64 + 1) * 4).next_multiple_of(TABLE_ALIGNMENT);
    let mut offsets = Vec::with_capacity(block_bytes.len());
    let mut end = table_bytes;
    for &bytes in block_bytes {
        if bytes == 0 || bytes % 4 != 0 {
            return fail("a block is empty or not a multiple of 4 bytes");
        }
        offsets.push(end);
        end += bytes;
    }
    if end > u64::from(u32::MAX) {
        return fail("a bank past 4 GiB");
    }
    let mut table = Vec::with_capacity(table_bytes as usize);
    for &offset in offsets.iter().chain([&end]) {
        table.extend_from_slice(&(offset as u32).to_le_bytes());
    }
    table.resize(table_bytes as usize, 0);
    Ok(BankLayout {
        table,
        offsets,
        bytes: end,
    })
}

/// Decodes matrix `matrix` of a bank laid out by [`bank_layout`].
pub fn decode_bank_matrix(
    bank: &[u8],
    matrix: usize,
    rows: u64,
    groups: u64,
    width: u32,
) -> Result<Vec<u8>, ScaleOffsetError> {
    let entry = |i: usize| -> Result<usize, ScaleOffsetError> {
        let at = i * 4;
        let bytes = bank
            .get(at..at + 4)
            .ok_or_else(|| ScaleOffsetError("bank table ends early".into()))?;
        Ok(u32::from_le_bytes(bytes.try_into().expect("4 bytes")) as usize)
    };
    let (start, end) = (entry(matrix)?, entry(matrix + 1)?);
    let block = bank
        .get(start..end)
        .filter(|_| start <= end)
        .ok_or_else(|| ScaleOffsetError("bank block outside the bank".into()))?;
    decode(block, rows, groups, width)
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
        let counts = census(scales, rows, groups).unwrap();
        for (slot, &width) in WIDTHS.iter().enumerate() {
            let block = encode(scales, rows, groups, width).unwrap();
            assert_eq!(
                block.len() as u64,
                encoded_bytes(rows, groups, width, counts[slot]).unwrap(),
                "width {width}"
            );
            assert_eq!(
                decode(&block, rows, groups, width).unwrap(),
                scales,
                "width {width}"
            );
        }
    }

    #[test]
    fn clustered_rows_round_trip_with_few_exceptions() {
        // Scales near a per-row exponent, as trained MXFP4 experts store them.
        let (rows, groups) = (64, 160);
        let draws = noise(1, (rows * groups) as usize);
        let scales: Vec<u8> = draws
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let base = 118 + (i as u64 / groups % 5) as u8;
                base + [0, 0, 1, 1, 1, 0, 2, 0][(d % 8) as usize]
            })
            .collect();
        round_trip(&scales, rows, groups);
        let counts = census(&scales, rows, groups).unwrap();
        assert!(
            counts[0] > 0 && counts[1] == 0 && counts[2] == 0,
            "{counts:?}"
        );
    }

    #[test]
    fn every_byte_value_round_trips() {
        // Includes UE8M0 253, 254 and 255 and every E4M3 code.
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
        assert_eq!(census(&constant, 7, 33).unwrap(), [0; 3]);
        let single: Vec<u8> = (0..50).map(|i| (i * 5) as u8).collect();
        round_trip(&single, 50, 1);
        assert_eq!(census(&single, 50, 1).unwrap(), [0; 3]);
    }

    #[test]
    fn eighteen_group_slices_round_trip() {
        // TP4 down projections keep 18 groups per row.
        let draws = noise(3, 5120 * 18);
        let scales: Vec<u8> = draws
            .iter()
            .map(|d| 120 + [0, 1, 0, 1, 3, 0, 1, 7][(d % 8) as usize])
            .collect();
        round_trip(&scales, 5120, 18);
    }

    #[test]
    fn bases_prefer_the_fullest_window_then_the_lower_base() {
        // Window {10, 11} holds 2, {11, 12} holds 3: the base is 11.
        let (bases, exceptions) = row_bases(&[10, 11, 12, 12, 20]);
        assert_eq!((bases[0], exceptions[0]), (11, 2));
        // {5, 6} and {9, 10} tie: the lower base wins.
        let (bases, _) = row_bases(&[5, 6, 9, 10]);
        assert_eq!(bases[0], 5);
    }

    #[test]
    fn exception_records_are_ordered_and_bounded() {
        let scales = [1, 200, 1, 1, 90, 1];
        let block = encode(&scales, 2, 3, 1).unwrap();
        let fixed = fixed_bytes(2, 3, 1).unwrap() as usize;
        let records: Vec<u32> = block[fixed..]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(records, [(1 << 8) | 200, (4 << 8) | 90]);
        let mut swapped = block.clone();
        swapped[fixed..].rotate_left(4);
        assert!(decode(&swapped, 2, 3, 1).is_err());
    }

    #[test]
    fn shapes_and_widths_are_checked() {
        assert!(encode(&[1], 1, 1, 3).is_err());
        assert!(encode(&[], 0, 1, 1).is_err());
        assert!(fixed_bytes(1 << 12, 1 << 13, 1).is_err());
        assert!(census(&[1, 2], 1, 3).is_err());
        assert!(decode(&[0; 3], 1, 1, 1).is_err());
    }

    #[test]
    fn banks_place_blocks_after_their_table() {
        let a = encode(&[5, 6, 7, 30], 2, 2, 1).unwrap();
        let b = encode(&[9; 6], 3, 2, 2).unwrap();
        let layout = bank_layout(&[a.len() as u64, b.len() as u64]).unwrap();
        assert_eq!(layout.table.len(), 16);
        assert_eq!(layout.offsets, [16, 16 + a.len() as u64]);
        let mut bank = layout.table.clone();
        bank.extend_from_slice(&a);
        bank.extend_from_slice(&b);
        assert_eq!(bank.len() as u64, layout.bytes);
        assert_eq!(
            decode_bank_matrix(&bank, 0, 2, 2, 1).unwrap(),
            [5, 6, 7, 30]
        );
        assert_eq!(decode_bank_matrix(&bank, 1, 3, 2, 1).unwrap(), [9; 6]);
        assert!(bank_layout(&[]).is_err());
        assert!(bank_layout(&[6]).is_err());
    }
}
