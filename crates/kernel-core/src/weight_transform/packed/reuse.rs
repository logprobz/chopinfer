//! Single row-major MXFP4 matrices use their final section storage directly.
use super::*;
use crate::precision::packed::{MatrixSlice, MatrixView, ScaleBytes};

impl PackedOperand {
    pub(super) fn reuses_row_major_sections(&self) -> bool {
        let [matrix] = self.matrices.as_slice() else {
            return false;
        };
        self.plan.format == WeightFormat::Mxfp4
            && matches!(
                self.plan.layout.id.as_str(),
                "row-major" | "row-major-groups"
            )
            && self.plan.experts == 1
            && self.plan.projections == 1
            && self.plan.local_scale_dtype == Some(DType::Ue8M0)
            && matrix.format == WeightFormat::Mxfp4
            && matrix.local.is_some()
            && matrix.local_dtype == Some(DType::Ue8M0)
            && matrix.global.is_none()
    }

    pub(super) fn execute_row_major_reusing(
        &self,
        _record: &WeightRecord,
        sources: &[&[u8]],
        buffers: [Vec<u8>; 3],
        expected: u64,
    ) -> Result<[Vec<u8>; 3], WeightTransformError> {
        let matrix = &self.matrices[0];
        let part = |index: u32| -> Result<&[u8], WeightTransformError> {
            let bytes = sources[index as usize];
            let size = bytes.len() / matrix.source_experts as usize;
            if size * matrix.source_experts as usize != bytes.len() {
                return refuse("bank source does not divide into its matrices");
            }
            let start = matrix.source_expert as usize * size;
            Ok(&bytes[start..start + size])
        };
        let view = MatrixView {
            tensor: "packed operand",
            format: matrix.format,
            rows: matrix.rows as usize,
            k: matrix.k as usize,
            row_origin: matrix.row_origin as usize,
            k_origin: matrix.k_origin as usize,
            nibble_order: matrix.nibble_order,
            codes: part(matrix.codes)?,
            local: Some(ScaleBytes {
                dtype: DType::Ue8M0,
                bytes: part(matrix.local.unwrap())?,
            }),
            global: None,
        };
        view.validate().map_err(precision)?;
        if self.decode_order.scales_before_bf16_mma() {
            view.validate_bf16_operands(self.decode_order)
                .map_err(precision)?;
        }
        let slice = MatrixSlice {
            rows: matrix
                .rows_kept
                .iter()
                .flat_map(|r| r.start as usize..r.end as usize)
                .collect(),
            k_start: matrix.k_kept.start as usize,
            k: (matrix.k_kept.end - matrix.k_kept.start) as usize,
            padded_rows: self.plan.rows as usize,
            padded_k: self.plan.k as usize,
            nibble_order: NibbleOrder::EvenLow,
        };
        let [codes, scales, mut globals] = buffers;
        let packed = view
            .repack_reusing(&slice, codes, scales)
            .map_err(precision)?;
        if !packed.k_origin.is_multiple_of(32) {
            return refuse("implementation requires an aligned source scale-group origin");
        }
        globals.clear();
        let sections = [packed.codes, packed.scales, globals];
        if sections[self.section.index()].len() as u64 != expected {
            return refuse("packed section differs from the prepared contract");
        }
        Ok(sections)
    }
}
