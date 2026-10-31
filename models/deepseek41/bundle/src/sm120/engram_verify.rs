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
            || !(33..=shape
                .verify_rows_within(model_deepseek41_contract::verify_bounds::ENGRAM_FOLD))
                .contains(&rows)
        {
            return Ok(None);
        }
        let op = tile(rows)?;
        Ok(Some(super::wide_routes::tagged(op, "mxfp8_prefill", 33)))
    }

    /// The same tile for an exact verify step of `rows` rows above 64 with
    /// wide verify buckets ([`crate::shape::Shape::wide_verify_rows`]), up
    /// to the widest of them.
    pub(crate) fn wide_verify_engram_fold(
        shape: &crate::shape::Shape,
        rows: u32,
    ) -> Result<Option<Prepared<Mxfp8FoldGemmBuffers<View>>>, Refused> {
        let first = model_deepseek41_contract::serving::VERIFY_DECODE_ROWS + 1;
        let Some(&widest) = shape.wide_verify_rows().last() else {
            return Ok(None);
        };
        if !(first..=widest).contains(&rows) {
            return Ok(None);
        }
        let op = tile(rows)?;
        let tags = op
            .info()
            .symbols
            .iter()
            .map(|(symbol, _)| {
                kernel_core::route::branch(
                    symbol,
                    kernel_core::route::RouteClass::Fast,
                    "mxfp8_prefill",
                    "wide-verify",
                    [first..=widest],
                )
            })
            .collect();
        Ok(Some(kernel_core::route::tagged(op, tags)))
    }
}

/// The qualified Engram matrix tile (64 rows, 2 stages) prepared for
/// exactly `rows` rows.
fn tile(rows: u32) -> Result<Prepared<Mxfp8FoldGemmBuffers<View>>, Refused> {
    let entry = super::prefill_matrix::catalog()
        .map_err(Refused::Rule)?
        .into_iter()
        .find(|e| {
            (e.base.query.n, e.base.query.k, e.base.query.groups) == (25600, 6144, 1)
                && (e.tile_rows, e.stages) == (64, 2)
        })
        .ok_or_else(|| Refused::Rule("missing qualified wide Engram tile".into()))?;
    Ok(kernels_sm120::backend::packed_op(
        entry.prepare(rows)?,
        [entry.symbol(entry.tile_rows)],
    )?)
}
