//! MMA fragment packing by whole 32-byte row segments.
//!
//! The `pack` of [`FragmentWeightLayout`] places one element at a time. Both
//! formats share one structure: a tile is 16 rows by 32 bytes (32 FP8 or 16
//! BF16 elements), and lane `4g + t` holds units of rows `g` and `g + 8` of
//! that tile, where a unit is two FP8 elements or two BF16 elements. Lane bytes
//! interleave `lo[t], hi[t], lo[t + 4], hi[t + 4], ...` over the units of those
//! two row segments, so the 64 bytes of lanes `4g..4g + 4` are one fixed
//! shuffle of two 32-byte segments. This packs whole strips that way; the tests
//! compare every byte with the reference.

use super::{WeightTransformError, refusal, refuse};
use crate::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};

const SEGMENT: usize = 32;
const TILE: usize = 512;

/// Lanes `4g..4g + 4` of one tile from its rows `g` (`lo`) and `g + 8` (`hi`).
/// `UNIT` is the bytes of one lane unit: 2 for FP8, 4 for BF16.
#[inline(always)]
fn shuffle<const UNIT: usize>(lo: &[u8; SEGMENT], hi: &[u8; SEGMENT], out: &mut [u8; 64]) {
    let pairs = 8 / UNIT;
    for t in 0..4 {
        for j in 0..pairs {
            let source = (t + 4 * j) * UNIT;
            let target = t * 16 + j * 2 * UNIT;
            out[target..target + UNIT].copy_from_slice(&lo[source..source + UNIT]);
            out[target + UNIT..target + 2 * UNIT].copy_from_slice(&hi[source..source + UNIT]);
        }
    }
}

fn pack_strips_with<const UNIT: usize>(
    rows: &[u8],
    live_rows: usize,
    row_bytes: usize,
    strips: usize,
    out: &mut [u8],
) {
    const ZERO: [u8; SEGMENT] = [0; SEGMENT];
    let k_tiles = row_bytes / SEGMENT;
    let segment = |row: usize, tile: usize| -> &[u8; SEGMENT] {
        if row < live_rows {
            let start = row * row_bytes + tile * SEGMENT;
            rows[start..start + SEGMENT]
                .try_into()
                .expect("32-byte segment")
        } else {
            &ZERO
        }
    };
    for (strip, output) in out
        .chunks_exact_mut(k_tiles * TILE)
        .take(strips)
        .enumerate()
    {
        for (tile, output) in output.chunks_exact_mut(TILE).enumerate() {
            for (g, lanes) in output.chunks_exact_mut(64).enumerate() {
                let row = strip * 16 + g;
                shuffle::<UNIT>(
                    segment(row, tile),
                    segment(row + 8, tile),
                    lanes.try_into().expect("64 lane bytes"),
                );
            }
        }
    }
}

/// Bytes of one 16-row strip.
pub fn strip_bytes(layout: FragmentWeightLayout) -> Result<usize, WeightTransformError> {
    layout.packed_bytes().map_err(refusal)?;
    Ok(layout.k / layout.tile_k() * TILE)
}

/// Packs strips `first..first + out.len() / strip_bytes` of `layout`.
///
/// `rows` holds the matrix rows those strips cover, row-major from row
/// `16 * first`, stopping at the matrix's last row; rows past it are zero.
pub fn pack_strips(
    layout: FragmentWeightLayout,
    rows: &[u8],
    first: usize,
    out: &mut [u8],
) -> Result<(), WeightTransformError> {
    let strip = strip_bytes(layout)?;
    let row_bytes = layout.k * layout.element_bytes();
    if !out.len().is_multiple_of(strip) {
        return refuse("fragment output is not whole strips");
    }
    let strips = out.len() / strip;
    let end = (first + strips) * 16;
    if first + strips > layout.rows.div_ceil(16) {
        return refuse("fragment strips beyond the matrix");
    }
    let live = end.min(layout.rows) - first * 16;
    if rows.len() != live * row_bytes {
        return refuse("fragment rows do not cover the strips");
    }
    match layout.format {
        FragmentWeightFormat::Fp8E4M3 => pack_strips_with::<2>(rows, live, row_bytes, strips, out),
        FragmentWeightFormat::Bf16 => pack_strips_with::<4>(rows, live, row_bytes, strips, out),
    }
    Ok(())
}

/// Packs a whole matrix; the result equals the reference packer's bytes.
pub fn pack(layout: FragmentWeightLayout, source: &[u8]) -> Result<Vec<u8>, WeightTransformError> {
    let mut out = vec![0; layout.packed_bytes().map_err(refusal)?];
    pack_strips(layout, source, 0, &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::Transform;

    fn source(bytes: usize, seed: usize) -> Vec<u8> {
        (0..bytes)
            .map(|i| ((i.wrapping_mul(2_654_435_761) >> 7) ^ seed) as u8)
            .collect()
    }

    /// Every row count through three strips and several K, both formats.
    #[test]
    fn small_shapes_equal_the_reference_packer() {
        for format in [FragmentWeightFormat::Bf16, FragmentWeightFormat::Fp8E4M3] {
            for rows in 1..=49 {
                for k_tiles in [1, 2, 3, 5] {
                    let layout = FragmentWeightLayout {
                        rows,
                        k: k_tiles
                            * FragmentWeightLayout {
                                rows,
                                k: 32,
                                format,
                            }
                            .tile_k(),
                        format,
                    };
                    let bytes = source(rows * layout.k * layout.element_bytes(), rows);
                    assert_eq!(
                        pack(layout, &bytes).unwrap(),
                        layout.pack(&bytes).unwrap().bytes,
                        "{layout:?}"
                    );
                }
            }
        }
    }

    /// Strip-by-strip packing of a part equals the same part of the whole.
    #[test]
    fn strip_ranges_equal_the_whole_matrix() {
        for format in [FragmentWeightFormat::Bf16, FragmentWeightFormat::Fp8E4M3] {
            let layout = FragmentWeightLayout {
                rows: 75,
                k: 64,
                format,
            };
            let row_bytes = layout.k * layout.element_bytes();
            let bytes = source(layout.rows * row_bytes, 3);
            let whole = layout.pack(&bytes).unwrap().bytes;
            let strip = strip_bytes(layout).unwrap();
            for first in 0..5 {
                for count in 1..=5 - first {
                    let rows = &bytes[first * 16 * row_bytes
                        ..((first + count) * 16).min(layout.rows) * row_bytes];
                    let mut out = vec![0xaa; count * strip];
                    pack_strips(layout, rows, first, &mut out).unwrap();
                    assert_eq!(out, whole[first * strip..(first + count) * strip]);
                }
            }
        }
    }

    /// The fragment matrices of the two served model families (every rank
    /// count, main and drafter layers). Each keeps its real K and its row
    /// count modulo 16; very large matrices keep two whole strips.
    #[test]
    fn model_shapes_equal_the_reference_packer() {
        use FragmentWeightFormat::{Bf16, Fp8E4M3};
        let shapes = [
            // The first family's BF16 views at 2, 4 and 8 ranks.
            (Bf16, 4, 10240),
            (Bf16, 12, 2560),
            (Bf16, 256, 2560),
            (Bf16, 320, 10240),
            (Bf16, 512, 2560),
            (Bf16, 640, 2560),
            (Bf16, 768, 2560),
            (Bf16, 1024, 2560),
            (Bf16, 1280, 2560),
            (Bf16, 1536, 2560),
            (Bf16, 2560, 128),
            (Bf16, 2560, 256),
            (Bf16, 2560, 384),
            (Bf16, 2560, 768),
            (Bf16, 2560, 1536),
            (Bf16, 2560, 2560),
            (Bf16, 2560, 3072),
            (Bf16, 3072, 2560),
            (Bf16, 5120, 2560),
            (Bf16, 6144, 2560),
            (Bf16, 10240, 320),
            (Bf16, 10240, 2560),
            (Bf16, 31040, 2560),
            (Bf16, 62080, 2560),
            (Bf16, 124_160, 2560),
            // Its FP8 expert pieces.
            (Fp8E4M3, 640, 2560),
            (Fp8E4M3, 2560, 640),
            // The second family's base and drafter matrices.
            (Bf16, 256, 4096),
            (Bf16, 2560, 4096),
            (Bf16, 4096, 2048),
            (Bf16, 4096, 4096),
            (Bf16, 4096, 5120),
            (Bf16, 8192, 4096),
            (Bf16, 38144, 4096),
            (Fp8E4M3, 16, 4096),
            (Fp8E4M3, 3392, 4096),
            (Fp8E4M3, 3712, 4096),
            (Fp8E4M3, 4096, 4096),
            (Fp8E4M3, 8192, 4096),
        ];
        for (format, rows, k) in shapes {
            let element = if format == Bf16 { 2 } else { 1 };
            let rows = if rows * k * element <= 8 << 20 {
                rows
            } else {
                32 + rows % 16
            };
            let layout = FragmentWeightLayout { rows, k, format };
            let bytes = source(rows * k * element, k);
            assert_eq!(
                pack(layout, &bytes).unwrap(),
                layout.pack(&bytes).unwrap().bytes,
                "{layout:?}"
            );
        }
    }

    #[test]
    fn refuses_mismatched_rows_and_strips() {
        let layout = FragmentWeightLayout {
            rows: 20,
            k: 32,
            format: FragmentWeightFormat::Fp8E4M3,
        };
        let strip = strip_bytes(layout).unwrap();
        assert!(pack_strips(layout, &[0; 20 * 32], 0, &mut vec![0; strip + 1]).is_err());
        assert!(pack_strips(layout, &[0; 19 * 32], 0, &mut vec![0; 2 * strip]).is_err());
        assert!(pack_strips(layout, &[0; 4 * 32], 1, &mut vec![0; 2 * strip]).is_err());
    }
}
