//! Prepare the measured row-local entry through the shared typed interface.
use kernel_core::backend::{Need, Prepared, Refused, View};
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers;
impl crate::Kernels {
    pub(crate) fn prepare_local_query(
        shape: &crate::shape::Shape,
        need: Need,
    ) -> Result<Option<Prepared<Mxfp8FoldGemmBuffers<View>>>, Refused> {
        if !super::fold_local_rows::selected(shape.geometry, shape.ranks()) {
            return Ok(None);
        }
        let entry = super::fold_local_rows::entries()
            .into_iter()
            .find(|e| e.tile_n == 32)
            .unwrap();
        entry.prepare(need).map(Some)
    }
}
