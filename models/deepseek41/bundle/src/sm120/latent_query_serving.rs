//! Bind the qualified wide query rotation without changing prefill selection.
use kernel_core::backend::{Backend, Need, Prepared, Refused, View};
use kernel_core::latent_glue::{LatentQueryRope, LatentQueryRopeBuffers};
use kernels_sm120::backend::Sm120;

impl crate::Kernels {
    pub(crate) fn wide_query_rope(
        shape: &crate::shape::Shape,
        op: &LatentQueryRope,
        need: Need,
    ) -> Result<Prepared<LatentQueryRopeBuffers<View>>, Refused> {
        if !super::latent_scheduled::qualified(shape.geometry, shape.ranks())
            || !(49..=shape.verify_decode_rows()).contains(&need.capacity)
        {
            return Err(Refused::Unsupported);
        }
        Sm120::new()
            .with_latent_glue(super::latent_scheduled::query().binding())
            .latent_query_rope(op, need)
    }
}
