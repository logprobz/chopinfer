//! The column merges of the latent lab shape's splitting attention entries
//! (`latent_shapes.rs`, through the `latent_attention` family) and their lab
//! mutants (`latent::merge_columns::lab_cubin_specs`).

use crate::latent_attention::shapes;
use kernel_build::CubinSpec;
use kernels_sm120::latent::merge_columns;

pub const FAMILY: &str = "latent_merge_columns";

pub fn cubins() -> Vec<CubinSpec> {
    merge_columns::lab_cubin_specs(
        shapes::PROFILE,
        &merge_columns::entries(&shapes::catalog()),
    )
    .expect("lab column merges are valid")
}
