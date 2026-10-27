//! Frozen decode variants for lab comparisons. Serving uses dense_decode.
#[path = "dense_decode_reference/backend.rs"]
mod backend;
#[path = "dense_decode_reference/mxfp8_fold_decode.rs"]
pub mod mxfp8_fold_decode;
#[path = "dense_decode_reference/mxfp8_fold_reuse.rs"]
pub mod mxfp8_fold_reuse;
#[path = "dense_decode_reference/mxfp8_fold_ticket.rs"]
pub mod mxfp8_fold_ticket;
#[allow(unused_imports, reason = "build-only consumers do not bind a backend")]
pub use backend::ReferenceBackend;
use kernels_sm120::gemm::{SM_COUNT, mxfp8_fold, mxfp8_fold_packed};
use std::path::PathBuf;

pub fn sources_root() -> PathBuf {
    // Both live and frozen headers are rooted at the SDK crates directory.
    kernels_sm120::cuda_source_root().join("../..")
}
pub fn include_dirs() -> Vec<PathBuf> {
    [
        "kernels-sm120/cuda",
        "kernels-sm120/cuda/gemm",
        "kernel-lab/cuda/baselines/dense_decode",
    ]
    .map(PathBuf::from)
    .to_vec()
}
/// Add frozen headers to a mixed reference/served lab translation unit.
pub fn configure(mut spec: kernel_build::CubinSpec) -> kernel_build::CubinSpec {
    spec.sources_root = sources_root();
    spec.include_dirs = include_dirs();
    for header in &mut spec.headers {
        for kind in ["ticket", "prefetch", "decode", "reuse"] {
            *header = header.replace(
                &format!("gemm/mxfp8_fold_{kind}.cuh"),
                &format!("kernel-lab/cuda/baselines/dense_decode/mxfp8_fold_{kind}.cuh"),
            );
        }
    }
    spec
}
