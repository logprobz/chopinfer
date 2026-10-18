//! Kernel GPU correctness tests of the model's catalog instances, own ops
//! and encoder, on `kernel_harness`. Each engine adds its test file and its
//! module line here.

mod f32_head;
mod head_artifacts;
mod head_probes;
mod latent;
mod latent_bf16;
// The own ops: the bundle's entries, cubin specification and `OwnOps`
// implementation, and their tests.
mod compressor;
#[path = "../../../bundle/src/sm120/dspark.rs"]
mod dspark;
#[path = "../../../bundle/src/sm120/dspark_cubin.rs"]
mod dspark_cubin;
#[path = "../../../bundle/src/sm120/dspark_entries.rs"]
mod dspark_entries;
mod dspark_heads;
mod indexer;
#[path = "../../../bundle/src/sm120/indexer.rs"]
mod indexer_entries;
mod indexer_query;
#[path = "../../../bundle/src/sm120/indexer_query.rs"]
mod indexer_query_entries;
mod own;
#[path = "../../../bundle/src/sm120/own_cubin.rs"]
mod own_cubin;
#[path = "../../../bundle/src/sm120/own_entries.rs"]
mod own_entries;
#[path = "../../../bundle/src/sm120/own_ops.rs"]
mod own_ops;
// The row widening own op and the rounding rows it launches; the lab
// builds none of the all-reduce's cubins.
#[path = "../../../bundle/src/sm120/reduce.rs"]
#[allow(dead_code)]
mod reduce;
#[path = "../../../bundle/src/sm120/widen.rs"]
mod widen;

mod moe;

mod latent_glue;

mod mxfp8_fold;
mod window_stage;

// The vision encoder: the bundle's catalog and its conformance cases.
mod encoder;
#[path = "../../../bundle/src/sm120/encoder.rs"]
#[allow(dead_code, reason = "the cases read the catalog and cubin specs")]
mod encoder_catalog;

// Shared lab specs validate their names without linking the shared lab.
fn declared(names: &[&str], specs: Vec<kernel_build::CubinSpec>) -> Vec<kernel_build::CubinSpec> {
    assert_eq!(
        names,
        specs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>()
    );
    specs
}
