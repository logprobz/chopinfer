//! A single row-major MXFP4 scale section without materializing its codes,
//! in the plan's row tiles when it stores them so.
use super::*;
use crate::precision::layout::{ROW_TILES_64X128, ROW_TILES_64X128_LANES};
use crate::precision::packed::ScaleBytes;

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
        mut out: Vec<u8>,
    ) -> Result<Option<Vec<u8>>, WeightTransformError> {
        let Some(source) = self.independent_scale_source(record) else {
            return Ok(None);
        };
        self.output_bytes(record)?;
        let expected = self.raw_output_bytes(record)?;
        ScaleBytes {
            dtype: DType::Ue8M0,
            bytes,
        }
        .validate(record.sources[source].byte_length as usize, false)
        .map_err(|e| {
            WeightTransformError(format!("packed operand packed operand local scale: {e}"))
        })?;
        if self.decode_order.has_bf16_weight_operand() && bytes.iter().any(|b| *b > 252) {
            return Ok(None);
        }
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
