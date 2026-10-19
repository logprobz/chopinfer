//! Build only the separately locked cached F32 projection family.
#[path = "../../../bundle/src/sm120/dspark_router.rs"]
#[allow(dead_code)]
mod dspark_router;
#[path = "../../../bundle/src/sm120/f32_dense.rs"]
#[allow(dead_code)]
mod f32_dense;
#[path = "../../../bundle/src/sm120/f32_head.rs"]
#[allow(dead_code)]
mod f32_head;
#[path = "../../../bundle/src/sm120/f32_decode.rs"]
#[allow(dead_code)]
mod registration;
#[path = "../../../bundle/src/sm120/routed_sum.rs"]
#[allow(dead_code)]
mod routed_sum;
use kernel_build::{BuildOptions, LockDir, Toolchain};
use model_deepseek41_contract::geometry::GEOMETRIES;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: build_f32_decode OUT_DIR")?;
    std::fs::create_dir_all(&out)?;
    let out = std::fs::canonicalize(out)?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repository = std::fs::canonicalize(root.join("../../.."))?;
    if out.starts_with(&repository) {
        return Err("build outputs must stay outside the repository".into());
    }
    let families = registration::cubins(GEOMETRIES)?
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
    if options.lock_update && dir.join(format!("{}.lock", registration::FAMILY)).exists() {
        return Err("build_f32_decode only initializes a new family; it cannot replace one".into());
    }
    let toolchain = Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA is required for the cubin build".into());
    }
    let lock = LockDir::new(dir).family(registration::FAMILY, specs.iter().map(|spec| &spec.name));
    let built = kernel_build::build_in_families(&toolchain, &specs, &out, &lock, options, None)?;
    assert_eq!(built.len(), specs.len());
    if !options.lock_update {
        kernel_build::report::BuildReport::read(
            &out.join(kernel_build::report::BUILD_REPORT_FILE),
        )?
        .verify_current()?;
    }
    println!("verified {} cached F32 cubins", built.len());
    Ok(())
}
