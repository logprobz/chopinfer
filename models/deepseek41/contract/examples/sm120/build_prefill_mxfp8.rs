//! Build the separately locked staged MXFP8 prefill.
#[path = "../../../bundle/src/sm120/mxfp8_fold.rs"]
#[allow(dead_code)]
mod registration;
use kernel_build::{BuildOptions, LockDir, Toolchain};
use model_deepseek41_contract::geometry::GEOMETRIES;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: build_prefill_mxfp8 OUT_DIR")?;
    std::fs::create_dir_all(&out)?;
    let out = std::fs::canonicalize(out)?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repository = std::fs::canonicalize(root.join("../../.."))?;
    if out.starts_with(&repository) {
        return Err("build outputs must stay outside the repository".into());
    }
    let families = registration::prefill_cubins(GEOMETRIES)?
        .into_iter()
        .collect::<Vec<_>>();
    let specs = families
        .iter()
        .map(|(_, spec)| spec.clone())
        .collect::<Vec<_>>();
    let dir = root.join("../bundle/kernels.lock.d");
    let mut options = BuildOptions::from_env();
    options.emit_cargo_directives = false;
    if options.profile_line_info {
        return Err("lock verification requires line info off".into());
    }
    if options.lock_update
        && dir
            .join(format!("{}.lock", registration::PREFILL_FAMILY))
            .exists()
    {
        return Err(
            "build_prefill_mxfp8 only initializes a new family; it cannot replace one".into(),
        );
    }
    let toolchain = Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA is required for the cubin build".into());
    }
    let lock = LockDir::new(dir).family(
        registration::PREFILL_FAMILY,
        specs.iter().map(|spec| &spec.name),
    );
    let built = kernel_build::build_in_families(&toolchain, &specs, &out, &lock, options, None)?;
    assert_eq!(built.len(), specs.len());
    if !options.lock_update {
        kernel_build::report::BuildReport::read(
            &out.join(kernel_build::report::BUILD_REPORT_FILE),
        )?
        .verify_current()?;
    }
    println!("verified {} staged MXFP8 prefill cubins", built.len());
    Ok(())
}
