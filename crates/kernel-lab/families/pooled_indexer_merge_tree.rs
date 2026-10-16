//! The tree merges of the pooled indexer lab profile's selections
//! (`pooled_indexer_shapes.rs`, through the `pooled_indexer` family) and
//! their lab mutants (`pooled_indexer::merge_tree::lab_cubin_specs`).

use crate::pooled_indexer::shapes;
use kernel_build::CubinSpec;
use kernels_sm120::pooled_indexer::merge_tree;

pub const FAMILY: &str = "pooled_indexer_merge_tree";

pub fn cubins() -> Vec<CubinSpec> {
    merge_tree::lab_cubin_specs(shapes::PROFILE, &merge_tree::entries(&shapes::lab_entries()))
        .expect("lab tree merges are valid")
}
