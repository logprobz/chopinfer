//! Exact E4M3 code permutation for eight-row, 128-K packed folds.
//!
//! Groups are consecutive matrices. Each eight-row tile stores its 128-K
//! folds in order, then 32 lanes, then four-byte code words. A lane owns
//! one row and every fourth word of that row. A final 64-K fold occupies
//! half the bytes of a full fold, with no padding or scale conversion.

use super::{WeightRecord, WeightTransformError, refuse};
use crate::precision::WeightLayout;

pub const LAYOUT_ID: &str = "mxfp8-fold-packed-8x128-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Mxfp8FoldLayout {
    pub groups: usize,
    pub n: usize,
    pub k: usize,
}

impl Mxfp8FoldLayout {
    #[must_use]
    pub fn weight_layout() -> WeightLayout {
        WeightLayout {
            id: LAYOUT_ID.into(),
            version: 1,
        }
    }

    /// The permutation preserves the input byte count. Empty shapes,
    /// partial eight-row tiles and partial 64-K groups are refused.
    pub fn packed_bytes(self) -> Result<usize, WeightTransformError> {
        if self.groups == 0 || self.n == 0 || self.k == 0 || self.n % 8 != 0 || self.k % 64 != 0 {
            return refuse(
                "packed folds require nonzero groups, N divisible by 8 and K divisible by 64",
            );
        }
        self.groups
            .checked_mul(self.n)
            .and_then(|rows| rows.checked_mul(self.k))
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .ok_or_else(|| WeightTransformError("packed fold byte count overflows".into()))
    }

    pub fn strip_bytes(self) -> Result<usize, WeightTransformError> {
        self.packed_bytes()?;
        Ok(8 * self.k)
    }

    /// Byte offset of the four codes a row-major matrix holds at
    /// `row * k + column`, for a row counted across groups and a column
    /// divisible by four. Kernels address packed codes with the same formula
    /// (`kernels-sm120/cuda/gemm/mxfp8_fold_packed.cuh`, `word_offset`).
    pub fn word_offset(self, row: usize, column: usize) -> Result<usize, WeightTransformError> {
        let bytes = self.packed_bytes()?;
        if !column.is_multiple_of(4) || column >= self.k || row >= bytes / self.k {
            return refuse("packed fold word lies outside the matrix or splits a word");
        }
        let fold = column / 128;
        let local = column % 128;
        let segment = if self.k - fold * 128 == 64 { 64 } else { 128 };
        Ok(row / 8 * (8 * self.k)
            + fold * 1024
            + row % 8 * segment
            + local % 16 / 4 * (segment / 4)
            + local / 16 * 4)
    }

    /// Pack complete eight-row strips, starting at strip `first` of the
    /// grouped matrix. `rows` contains only those strips in row-major order.
    pub fn pack_strips(
        self,
        rows: &[u8],
        first: usize,
        out: &mut [u8],
    ) -> Result<(), WeightTransformError> {
        let bytes = self.packed_bytes()?;
        let strip = self.strip_bytes()?;
        if rows.len() != out.len()
            || !out.len().is_multiple_of(strip)
            || first > bytes / strip
            || out.len() / strip > bytes / strip - first
        {
            return refuse("packed fold rows and output must cover complete in-range strips");
        }
        for (source, packed) in rows.chunks_exact(strip).zip(out.chunks_exact_mut(strip)) {
            for fold in (0..self.k).step_by(128) {
                let width = (self.k - fold).min(128);
                for row in 0..8 {
                    for lane in 0..4 {
                        let target = fold * 8 + (row * 4 + lane) * (width / 4);
                        for word in 0..width / 16 {
                            let from = row * self.k + fold + word * 16 + lane * 4;
                            let to = target + word * 4;
                            packed[to..to + 4].copy_from_slice(&source[from..from + 4]);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn pack(self, source: &[u8]) -> Result<Vec<u8>, WeightTransformError> {
        let bytes = self.packed_bytes()?;
        if source.len() != bytes {
            return refuse("packed fold source differs from its shape");
        }
        let mut out = vec![0; bytes];
        self.pack_strips(source, 0, &mut out)?;
        Ok(out)
    }

    /// Bytewise CPU oracle, traversing logical source coordinates instead
    /// of the production packer's lane words. It preserves every code bit.
    pub fn oracle(self, source: &[u8]) -> Result<Vec<u8>, WeightTransformError> {
        let bytes = self.packed_bytes()?;
        if source.len() != bytes {
            return refuse("packed fold oracle source differs from its shape");
        }
        let mut out = vec![0; bytes];
        for (from, &byte) in source.iter().enumerate() {
            let row = from / self.k;
            let column = from % self.k;
            let fold = column / 128;
            let lane_bytes = if self.k - fold * 128 == 64 { 16 } else { 32 };
            let lane = row % 8 * 4 + column % 16 / 4;
            let to = row / 8 * (8 * self.k)
                + fold * 1024
                + lane * lane_bytes
                + column % 128 / 16 * 4
                + column % 4;
            out[to] = byte;
        }
        Ok(out)
    }
}

impl WeightRecord {
    /// Last two axes are N and K; leading axes enumerate independent groups.
    pub fn mxfp8_fold_layout(&self) -> Result<Mxfp8FoldLayout, WeightTransformError> {
        let [leading @ .., n, k] = &self.stored_shape[..] else {
            return refuse("packed folds need at least two stored dimensions");
        };
        let groups = leading
            .iter()
            .try_fold(1_u64, |groups, &dim| groups.checked_mul(dim))
            .ok_or_else(|| WeightTransformError("packed fold groups overflow".into()))?;
        let dimension = |dim| {
            usize::try_from(dim)
                .map_err(|_| WeightTransformError("packed fold dimension overflows".into()))
        };
        let layout = Mxfp8FoldLayout {
            groups: dimension(groups)?,
            n: dimension(*n)?,
            k: dimension(*k)?,
        };
        layout.packed_bytes()?;
        Ok(layout)
    }
}

#[cfg(test)]
mod tests;
