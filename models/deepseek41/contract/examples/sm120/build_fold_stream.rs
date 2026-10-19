//! Build only the dense fold families, without changing existing locks.
#[path = "../../../bundle/src/sm120/mxfp8_fold.rs"]
#[allow(dead_code)]
mod mxfp8_fold;
#[path = "../../../bundle/src/sm120/mxfp8_fold_stream.rs"]
#[allow(dead_code)]
mod registration;
use kernel_build::{BuildOptions, LockDir, Toolchain};
use model_deepseek41_contract::geometry::GEOMETRIES;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: build_fold_stream OUT_DIR")?;
    std::fs::create_dir_all(&out)?;
    let out = std::fs::canonicalize(out)?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repository = std::fs::canonicalize(root.join("../../.."))?;
    if out.starts_with(&repository) {
        return Err("build outputs must stay outside the repository".into());
    }
    let mut options = BuildOptions::from_env();
    options.emit_cargo_directives = false;
    if options.profile_line_info {
        return Err("lock verification requires line info off".into());
    }
    let dir = root.join("../bundle/kernels.lock.d");
    if options.lock_update && dir.join(format!("{}.lock", registration::FAMILY)).exists() {
        return Err(
            "build_fold_stream only initializes a new family; it cannot replace one".into(),
        );
    }
    let toolchain = Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA is required for the cubin build".into());
    }
    let specs: Vec<_> = registration::cubins(GEOMETRIES)?
        .into_iter()
        .map(|(_, s)| s)
        .collect();
    let lock = LockDir::new(&dir).family(registration::FAMILY, specs.iter().map(|s| &s.name));
    let built = kernel_build::build_in_families(
        &toolchain,
        &specs,
        &out.join("stream"),
        &lock,
        options,
        None,
    )?;
    println!("verified {} packed fold cubins", built.len());
    Ok(())
}
