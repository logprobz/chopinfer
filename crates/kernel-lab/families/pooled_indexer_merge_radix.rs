//! The radix merge of the pooled indexer lab profile's global selection
//! (`pooled_indexer_shapes.rs`, through the `pooled_indexer` family) and its
//! lab mutants (`pooled_indexer::merge_radix::lab_cubin_specs`).

use crate::pooled_indexer::shapes;
use kernel_build::CubinSpec;
use kernels_sm120::pooled_indexer::merge_radix;

pub const FAMILY: &str = "pooled_indexer_merge_radix";

pub fn cubins() -> Vec<CubinSpec> {
    merge_radix::lab_cubin_specs(
        shapes::PROFILE,
        &merge_radix::entries(&shapes::lab_entries()),
    )
    .expect("lab radix merges are valid")
}
