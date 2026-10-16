//! MXFP4 MoE fragment packing of one expert matrix by whole 16-row strips.
//!
//! [`crate::transform::Mxfp4MoeFragmentLayout`] packs a bank of experts; a
//! plan record holds one expert's rank slice of one projection, so it packs
//! with one expert. The stored matrix is `[rows, row_bytes]` of bytes: FP4
//! code pairs (two values per byte) or UE8M0 scales (one per 32 values).
//! Both layouts keep 16-row strips in order and permute bytes within each
//! strip only, so any run of whole strips packs on its own, and the packed
//! bytes equal the stored bytes in number. The tests compare every byte with
//! the reference packer.

use super::{WeightTransformError, refuse};
use crate::transform::Mxfp4MoeFragmentLayout;

/// Which half of an MXFP4 operand the record stores.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mxfp4Section {
    /// `[rows, k / 2]` code bytes, low nibble first.
    Codes,
    /// `[rows, k / 32]` UE8M0 scale bytes.
    Scales,
}

/// One expert matrix's MXFP4 fragment geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Mxfp4Strips {
    pub section: Mxfp4Section,
    pub rows: usize,
    /// Stored bytes of one row: code bytes, or scale bytes.
    pub row_bytes: usize,
}

const STRIP_ROWS: usize = 16;

impl Mxfp4Strips {
    /// The reference packer's layout of this one matrix.
    #[must_use]
    pub fn layout(self) -> Mxfp4MoeFragmentLayout {
        Mxfp4MoeFragmentLayout {
            experts: 1,
            rows_per_expert: self.rows,
            k: self.row_bytes
                * match self.section {
                    Mxfp4Section::Codes => 2,
                    Mxfp4Section::Scales => 32,
                },
        }
    }

    /// Refuses a matrix the fragment layout cannot hold: rows in whole
    /// strips and K in whole 64-value tiles.
    pub fn check(self) -> Result<(), WeightTransformError> {
        let layout = self.layout();
        if self.rows == 0
            || !self.rows.is_multiple_of(STRIP_ROWS)
            || layout.k == 0
            || !layout.k.is_multiple_of(64)
        {
            return refuse("MXFP4 MoE fragments need 16-row strips and 64-value K tiles");
        }
        self.rows
            .checked_mul(self.row_bytes)
            .ok_or_else(|| WeightTransformError("MXFP4 matrix size overflow".into()))?;
        Ok(())
    }

    /// Packed bytes, equal to the stored bytes.
    pub fn packed_bytes(self) -> Result<usize, WeightTransformError> {
        self.check()?;
        Ok(self.rows * self.row_bytes)
    }

    /// Bytes of one packed 16-row strip.
    #[must_use]
    pub const fn strip_bytes(self) -> usize {
        STRIP_ROWS * self.row_bytes
    }

    /// Packs strips `first..first + out.len() / strip_bytes` from their
    /// stored rows, row-major from row `16 * first`.
    pub fn pack_strips(
        self,
        rows: &[u8],
        first: usize,
        out: &mut [u8],
    ) -> Result<(), WeightTransformError> {
        self.check()?;
        let strip = self.strip_bytes();
        if !out.len().is_multiple_of(strip)
            || rows.len() != out.len()
            || (first + out.len() / strip) * STRIP_ROWS > self.rows
        {
            return refuse("MXFP4 strips do not match their rows or the matrix");
        }
        match self.section {
            Mxfp4Section::Codes => pack_codes(rows, self.row_bytes, out),
            Mxfp4Section::Scales => pack_scales(rows, self.row_bytes, out),
        }
        Ok(())
    }

    /// Packs the whole matrix.
    pub fn pack(self, source: &[u8]) -> Result<Vec<u8>, WeightTransformError> {
        let mut out = vec![0; self.packed_bytes()?];
        self.pack_strips(source, 0, &mut out)?;
        Ok(out)
    }
}

/// A 64-value tile is 32 code bytes of each of 16 rows. Lane `4g + t` holds,
/// for each of 4 steps, bytes `8s + t` and `8s + t + 4` of rows `g` and
/// `g + 8`, interleaved row `g` first.
fn pack_codes(rows: &[u8], row_bytes: usize, out: &mut [u8]) {
    let k_tiles = row_bytes / 32;
    for (input, output) in rows
        .chunks_exact(STRIP_ROWS * row_bytes)
        .zip(out.chunks_exact_mut(STRIP_ROWS * row_bytes))
    {
        for (tile, output) in output.chunks_exact_mut(512).enumerate().take(k_tiles) {
            for (g, lanes) in output.chunks_exact_mut(64).enumerate() {
                let lo = &input[g * row_bytes + tile * 32..][..32];
                let hi = &input[(g + 8) * row_bytes + tile * 32..][..32];
                for t in 0..4 {
                    for s in 0..4 {
                        let at = t * 16 + s * 4;
                        let byte = s * 8 + t;
                        lanes[at] = lo[byte];
                        lanes[at + 1] = hi[byte];
                        lanes[at + 2] = lo[byte + 4];
                        lanes[at + 3] = hi[byte + 4];
                    }
                }
            }
        }
    }
}

/// A tile's 32 scale bytes: word `g` holds both 32-value blocks of rows `g`
/// and `g + 8`, block first.
fn pack_scales(rows: &[u8], row_bytes: usize, out: &mut [u8]) {
    let k_tiles = row_bytes / 2;
    for (input, output) in rows
        .chunks_exact(STRIP_ROWS * row_bytes)
        .zip(out.chunks_exact_mut(STRIP_ROWS * row_bytes))
    {
        for (tile, output) in output.chunks_exact_mut(32).enumerate().take(k_tiles) {
            for g in 0..8 {
                for block in 0..2 {
                    for high in 0..2 {
                        output[g * 4 + block * 2 + high] =
                            input[(g + 8 * high) * row_bytes + tile * 2 + block];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(bytes: usize, seed: usize) -> Vec<u8> {
        (0..bytes)
            .map(|i| ((i.wrapping_mul(2_654_435_761) >> 9) ^ seed) as u8)
            .collect()
    }

    fn reference(strips: Mxfp4Strips, bytes: &[u8]) -> Vec<u8> {
        let layout = strips.layout();
        match strips.section {
            Mxfp4Section::Codes => layout.pack_codes(bytes),
            Mxfp4Section::Scales => layout.pack_scales(bytes),
        }
        .unwrap()
        .bytes
    }

    /// Small shapes, and a served model's expert slices at 2, 4 and 8 ranks.
    #[test]
    fn matrices_equal_the_reference_packer() {
        let mut shapes = Vec::new();
        for rows in [16, 32, 48, 80] {
            for k_tiles in [1, 2, 3, 5] {
                shapes.push((Mxfp4Section::Codes, rows, 32 * k_tiles));
                shapes.push((Mxfp4Section::Scales, rows, 2 * k_tiles));
            }
        }
        for rows in [2048, 1024, 512, 256] {
            shapes.push((Mxfp4Section::Codes, rows, 1024));
            shapes.push((Mxfp4Section::Scales, rows, 64));
        }
        for k in [1024, 512, 256] {
            shapes.push((Mxfp4Section::Codes, 4096, k / 2));
            shapes.push((Mxfp4Section::Scales, 4096, k / 32));
        }
        for (section, rows, row_bytes) in shapes {
            let strips = Mxfp4Strips {
                section,
                rows,
                row_bytes,
            };
            let bytes = source(rows * row_bytes, rows ^ row_bytes);
            assert_eq!(
                strips.pack(&bytes).unwrap(),
                reference(strips, &bytes),
                "{strips:?}"
            );
        }
    }

    #[test]
    fn strip_runs_equal_the_whole_matrix() {
        for (section, row_bytes) in [(Mxfp4Section::Codes, 96), (Mxfp4Section::Scales, 6)] {
            let strips = Mxfp4Strips {
                section,
                rows: 80,
                row_bytes,
            };
            let bytes = source(80 * row_bytes, 5);
            let whole = reference(strips, &bytes);
            let strip = strips.strip_bytes();
            for first in 0..5 {
                for count in 1..=5 - first {
                    let mut out = vec![0xaa; count * strip];
                    strips
                        .pack_strips(
                            &bytes[first * strip..(first + count) * strip],
                            first,
                            &mut out,
                        )
                        .unwrap();
                    assert_eq!(out, whole[first * strip..(first + count) * strip]);
                }
            }
        }
    }

    #[test]
    fn refuses_partial_strips_and_tiles() {
        for (section, rows, row_bytes) in [
            (Mxfp4Section::Codes, 24, 32),
            (Mxfp4Section::Codes, 16, 48),
            (Mxfp4Section::Scales, 16, 3),
            (Mxfp4Section::Scales, 0, 2),
        ] {
            let strips = Mxfp4Strips {
                section,
                rows,
                row_bytes,
            };
            assert!(
                strips.pack(&vec![0; rows * row_bytes]).is_err(),
                "{strips:?}"
            );
        }
        let strips = Mxfp4Strips {
            section: Mxfp4Section::Codes,
            rows: 32,
            row_bytes: 32,
        };
        assert!(strips.pack_strips(&[0; 512], 2, &mut [0; 512]).is_err());
        assert!(strips.pack_strips(&[0; 511], 0, &mut [0; 512]).is_err());
    }
}
