//! The slice-widened forms of the latent lab shape's FP8 prefill attention
//! entries (`latent_shapes.rs`, through the `latent_attention` family) and
//! their lab mutants (`latent::slices::lab_cubin_specs`).

use crate::latent_attention::shapes;
use kernel_build::CubinSpec;
use kernels_sm120::latent::slices;

pub const FAMILY: &str = "latent_attention_slices";

pub fn cubins() -> Vec<CubinSpec> {
    slices::lab_cubin_specs(shapes::PROFILE, &slices::entries(&shapes::catalog()))
        .expect("lab slice-widened entries are valid")
}
