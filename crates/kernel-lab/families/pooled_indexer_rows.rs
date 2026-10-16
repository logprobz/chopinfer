//! The one-split selection forms of the pooled indexer lab shape's
//! selections (`pooled_indexer_shapes.rs`, through the `pooled_indexer`
//! family), one and two rows per CTA, and their lab mutants
//! (`pooled_indexer::rows::lab_cubin_specs`).

use crate::pooled_indexer::shapes;
use kernel_build::CubinSpec;
use kernels_sm120::pooled_indexer::rows;

pub const FAMILY: &str = "pooled_indexer_rows";

pub fn cubins() -> Vec<CubinSpec> {
    let selects = shapes::lab_entries();
    let forms = rows::FORMS
        .iter()
        .flat_map(|&form| rows::entries(&selects, form))
        .collect::<Vec<_>>();
    rows::lab_cubin_specs(shapes::PROFILE, &forms).expect("lab one-split entries are valid")
}
