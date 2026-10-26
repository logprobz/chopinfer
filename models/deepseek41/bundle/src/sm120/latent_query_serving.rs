//! Bind the qualified wide query rotation without changing prefill selection.
use kernel_core::backend::{Backend, Need, Prepared, Refused, View};
use kernel_core::latent_glue::{LatentQueryRope, LatentQueryRopeBuffers};
use kernels_sm120::backend::Sm120;

impl crate::Kernels {
    pub(crate) fn wide_latent_query_rows(capacity: u32, rows: u32) -> bool {
        super::latent_scheduled::wide_query(capacity, rows)
    }

    pub(crate) fn wide_query_rope(
        shape: &crate::shape::Shape,
        op: &LatentQueryRope,
        need: Need,
    ) -> Result<Prepared<LatentQueryRopeBuffers<View>>, Refused> {
        if !super::latent_scheduled::qualified(shape.geometry, shape.ranks())
            || !super::latent_scheduled::wide_query(shape.verify_decode_rows(), need.capacity)
        {
            return Err(Refused::Unsupported);
        }
        let op = Sm120::new()
            .with_latent_glue(super::latent_scheduled::query().binding())
            .latent_query_rope(op, need)?;
        Ok(super::wide_routes::tagged(op, "latent_query", 49))
    }
}
