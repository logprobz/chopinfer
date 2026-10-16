//! Model-owned weight and package declarations shared by build and preparation.
#![forbid(unsafe_code)]
pub mod backend;
pub mod buffers;
pub mod context_parallel;
pub mod contract;
pub mod encoder;
pub mod frontend;
pub mod geometry;
pub mod norms;
pub mod note;
pub mod own_ops;
pub mod owned;
pub mod package;
pub mod precision;
pub mod prepared;
pub mod projections;
pub mod rank;
pub mod sm120;
pub mod state;
pub mod views;
pub mod weight_plan;

// The packager's catalog, placement and prepared-precision sources, compiled
// here unchanged so the bundle plans direct loads with the code that writes
// the rank images (`ci/source_dependencies.toml` records the edge). They
// address each other and `Result` from the crate root, as in the compiler.
extern crate self as model_qwen38_contract;
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
#[path = "../../compiler/src/catalog.rs"]
pub mod catalog;
#[path = "../../compiler/src/config.rs"]
pub mod config;
#[path = "../../compiler/src/placement.rs"]
pub mod placement;
#[path = "../../compiler/src/precision_packing.rs"]
pub mod precision_packing;
#[path = "../../compiler/src/precision_source.rs"]
pub mod precision_source;
