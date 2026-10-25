//! The packed matrix tile qualified for wide target Engram verification.
use kernel_core::backend::{Prepared, Refused, View};
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers;
use model_deepseek41_contract::geometry::FLASH;

impl crate::Kernels {
    pub(crate) fn wide_engram_fold(
        shape: &crate::shape::Shape,
        rows: u32,
    ) -> Result<Option<Prepared<Mxfp8FoldGemmBuffers<View>>>, Refused> {
        if shape.geometry.id != FLASH.id
            || shape.ranks() != 4
            || !(33..=shape.verify_decode_rows()).contains(&rows)
        {
            return Ok(None);
        }
        let entry = super::prefill_matrix::catalog()
            .map_err(Refused::Rule)?
            .into_iter()
            .find(|e| {
                (e.base.query.n, e.base.query.k, e.base.query.groups) == (25600, 6144, 1)
                    && (e.tile_rows, e.stages) == (64, 2)
            })
            .ok_or_else(|| Refused::Rule("missing qualified wide Engram tile".into()))?;
        let op = kernels_sm120::backend::packed_op(
            entry.prepare(rows)?,
            [entry.symbol(entry.tile_rows)],
        )?;
        Ok(Some(super::wide_routes::tagged(op, "mxfp8_prefill", 33)))
    }
}
