//! Prepare the measured row-local entries through the shared typed interface.
use kernel_core::backend::{Need, Prepared, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers, Mxfp8FoldGemmQuery};
impl crate::Kernels {
    /// The row-local fold measured for projection `id` at `need.capacity`
    /// rows, if any; `query` is the projection's fold at that bucket.
    pub(crate) fn prepare_local_fold(
        shape: &crate::shape::Shape,
        id: &str,
        query: Mxfp8FoldGemmQuery,
        need: Need,
    ) -> Result<Option<Prepared<Mxfp8FoldGemmBuffers<View>>>, Refused> {
        if !super::fold_local_rows::selected(shape.geometry, shape.ranks()) {
            return Ok(None);
        }
        let Some(entry) = super::fold_local_rows::served(id, need.capacity) else {
            return Ok(None);
        };
        if query.m_capacity != need.capacity
            || (Mxfp8FoldGemmQuery {
                m_capacity: query.m_capacity,
                ..entry.base.query
            }) != query
        {
            return Err(Refusal("a row-local fold differs from its projection").into());
        }
        entry.prepare(need).map(Some)
    }
}
