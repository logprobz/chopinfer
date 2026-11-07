//! A single row-major MXFP4 scale section without materializing its codes,
//! in the plan's row tiles when it stores them so.
use super::*;
use crate::precision::layout::{ROW_TILES_64X128, ROW_TILES_64X128_LANES};
use crate::precision::packed::ScaleBytes;
use crate::weight_transform::payload::MXFP4_SCALE_MAX;
use std::sync::OnceLock;

/// One standalone UE8M0 scale tensor that several sections are produced
/// from, such as every rank's slice of one matrix. Its checks read the whole
/// tensor, so they are taken once, on first use, and each section gets what
/// checking the tensor for it alone would give.
pub struct StandaloneScales<'a> {
    bytes: &'a [u8],
    /// The scale count first checked, and its verdict.
    valid: OnceLock<(usize, Result<(), &'static str>)>,
    /// Whether a scale is above [`MXFP4_SCALE_MAX`].
    extreme: OnceLock<bool>,
}

impl<'a> StandaloneScales<'a> {
    #[must_use]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            valid: OnceLock::new(),
            extreme: OnceLock::new(),
        }
    }

    #[must_use]
    pub const fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// The bytes checked as `count` finite UE8M0 scales.
    fn validate(&self, count: usize) -> Result<(), &'static str> {
        let check = || {
            ScaleBytes {
                dtype: DType::Ue8M0,
                bytes: self.bytes,
            }
            .validate(count, false)
        };
        match self.valid.get_or_init(|| (count, check())) {
            (checked, verdict) if *checked == count => *verdict,
            _ => check(),
        }
    }

    fn extreme(&self) -> bool {
        *self.extreme.get_or_init(|| above_mxfp4_max(self.bytes))
    }
}

/// Whether a byte of `bytes` is above [`MXFP4_SCALE_MAX`]: the largest byte
/// of each chunk, a reduction the compiler vectorizes, compared once.
fn above_mxfp4_max(bytes: &[u8]) -> bool {
    bytes
        .chunks(4096)
        .any(|chunk| chunk.iter().fold(0, |high, &byte| high.max(byte)) > MXFP4_SCALE_MAX)
}

impl PackedOperand {
    /// The scale source when this section can be produced independently.
    /// Other layouts and code-dependent admission keep the whole materializer.
    pub fn independent_scale_source(&self, record: &WeightRecord) -> Option<usize> {
        let [matrix] = self.matrices.as_slice() else {
            return None;
        };
        if self.section != PackedSection::LocalScales
            || self.plan.format != WeightFormat::Mxfp4
            || !matches!(
                self.plan.layout.id.as_str(),
                "row-major" | "row-major-groups" | ROW_TILES_64X128 | ROW_TILES_64X128_LANES
            )
            || self.plan.experts != 1
            || self.plan.projections != 1
            || matrix.format != WeightFormat::Mxfp4
            || matrix.source_experts != 1
            || matrix.source_expert != 0
            || matrix.global.is_some()
            || matrix.local_dtype != Some(DType::Ue8M0)
            || (self.decode_order.has_bf16_weight_operand()
                && self.decode_order != DecodeOrder::ExactScaledOperand)
            || matrix.k == 0
            || !matrix.k.is_multiple_of(32)
            || !matrix.k_origin.is_multiple_of(32)
            || !matrix.k_kept.start.is_multiple_of(32)
            || matrix.k_kept.start >= matrix.k_kept.end
            || matrix.k_kept.end > matrix.k
            || matrix.k_kept.end - matrix.k_kept.start > u64::from(self.plan.k)
            || matrix.rows_kept.is_empty()
            || matrix
                .rows_kept
                .iter()
                .any(|r| r.start >= r.end || r.end > matrix.rows)
        {
            return None;
        }
        let rows = matrix
            .rows_kept
            .iter()
            .try_fold(0_u64, |sum, r| sum.checked_add(r.end - r.start))?;
        if rows > u64::from(self.plan.rows)
            || matrix.row_origin.checked_add(matrix.rows).is_none()
            || matrix.k_origin.checked_add(matrix.k).is_none()
        {
            return None;
        }
        // Mirror MatrixView::repack's inclusive padded-coordinate checks.
        // A valid source extent can still overflow when padding is added.
        (matrix.k_origin as usize)
            .checked_add(matrix.k_kept.start as usize)?
            .checked_add(self.plan.k as usize - 1)?;
        (matrix.row_origin as usize)
            .checked_add(matrix.rows_kept[0].start as usize)?
            .checked_add(self.plan.rows as usize - 1)?;
        let local = matrix.local? as usize;
        if local == matrix.codes as usize {
            return None;
        }
        let elements = matrix.rows.checked_mul(matrix.k)?;
        if record.sources.get(matrix.codes as usize)?.byte_length != elements / 2
            || record.sources.get(local)?.byte_length != elements / 32
        {
            return None;
        }
        Some(local)
    }

    /// `None` retains the whole path, including actual-code checks for extreme
    /// scales. Metadata and complete scale payloads are still authenticated.
    pub fn independent_scales(
        &self,
        record: &WeightRecord,
        bytes: &[u8],
        out: Vec<u8>,
    ) -> Result<Option<Vec<u8>>, WeightTransformError> {
        self.independent_scales_of(record, &StandaloneScales::new(bytes), out)
    }

    /// [`Self::independent_scales`] from a scale tensor whose whole-tensor
    /// checks the sections produced from it share.
    pub fn independent_scales_of(
        &self,
        record: &WeightRecord,
        scales: &StandaloneScales<'_>,
        mut out: Vec<u8>,
    ) -> Result<Option<Vec<u8>>, WeightTransformError> {
        let Some(source) = self.independent_scale_source(record) else {
            return Ok(None);
        };
        self.output_bytes(record)?;
        let expected = self.raw_output_bytes(record)?;
        scales
            .validate(record.sources[source].byte_length as usize)
            .map_err(|e| {
                WeightTransformError(format!("packed operand packed operand local scale: {e}"))
            })?;
        if self.decode_order.has_bf16_weight_operand() && scales.extreme() {
            return Ok(None);
        }
        let bytes = scales.bytes();
        let matrix = &self.matrices[0];
        out.resize(expected as usize, 127);
        out.fill(127);
        let source_columns = (matrix.k / 32) as usize;
        let columns = self.plan.k.div_ceil(32) as usize;
        let live = (matrix.k_kept.end - matrix.k_kept.start).div_ceil(32) as usize;
        for (output_row, row) in matrix.rows_kept.iter().flat_map(|r| r.clone()).enumerate() {
            let start = row as usize * source_columns + matrix.k_kept.start as usize / 32;
            out[output_row * columns..output_row * columns + live]
                .copy_from_slice(&bytes[start..start + live]);
        }
        let mut sections = self.stored([Vec::new(), out, Vec::new()])?;
        Ok(Some(std::mem::take(&mut sections[1])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The chunked maximum answers what a scan of every byte does, wherever
    /// the extreme byte falls relative to the chunks.
    #[test]
    fn the_chunked_maximum_finds_every_extreme_scale() {
        let naive = |bytes: &[u8]| bytes.iter().any(|b| *b > MXFP4_SCALE_MAX);
        for length in [0_usize, 1, 31, 32, 4095, 4096, 4097, 8192, 12_289] {
            let base: Vec<u8> = (0..length).map(|i| (i * 31 % 253) as u8).collect();
            assert!(!above_mxfp4_max(&base));
            assert_eq!(above_mxfp4_max(&base), naive(&base));
            for at in [0, 1, length / 2, 4095, 4096, length.saturating_sub(1)] {
                if at >= length {
                    continue;
                }
                for value in [MXFP4_SCALE_MAX, 253, 254, 255] {
                    let mut bytes = base.clone();
                    bytes[at] = value;
                    assert_eq!(
                        above_mxfp4_max(&bytes),
                        naive(&bytes),
                        "{length} {at} {value}"
                    );
                }
            }
        }
    }
}
