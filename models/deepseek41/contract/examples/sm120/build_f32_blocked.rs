//! Build only the separately locked measured blocked prefill family.
#[path = "../../../bundle/src/sm120/dspark_router.rs"]
#[allow(dead_code)]
mod dspark_router;
#[path = "../../../bundle/src/sm120/f32_blocked.rs"]
#[allow(dead_code)]
mod f32_blocked;
#[path = "../../../bundle/src/sm120/f32_dense.rs"]
#[allow(dead_code)]
mod f32_dense;
#[path = "../../../bundle/src/sm120/f32_head.rs"]
#[allow(dead_code)]
mod f32_head;
#[path = "../../../bundle/src/sm120/f32_prefill.rs"]
#[allow(dead_code)]
mod f32_prefill;
#[path = "../../../bundle/src/sm120/f32_wide.rs"]
#[allow(dead_code)]
mod f32_wide;
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
        .ok_or("usage: build_f32_blocked OUT_DIR")?;
    std::fs::create_dir_all(&out)?;
    let out = out.canonicalize()?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if out.starts_with(root.join("../../..").canonicalize()?) {
        return Err("build outputs must stay outside the repository".into());
    }
    let specs = f32_blocked::serving_cubins(GEOMETRIES)?;
    let dir = root.join("../bundle/kernels.lock.d");
    let mut options = BuildOptions::from_env();
    options.emit_cargo_directives = false;
    if options.profile_line_info {
        return Err("lock verification requires line info off".into());
    }
    if options.lock_update
        && dir
            .join(format!("{}.lock", f32_blocked::SERVING_FAMILY))
            .exists()
    {
        return Err("only a new family may be initialized".into());
    }
    let toolchain = Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA is required for the cubin build".into());
    }
    let lock = LockDir::new(dir).family(f32_blocked::SERVING_FAMILY, specs.iter().map(|s| &s.name));
    let built = kernel_build::build_in_families(&toolchain, &specs, &out, &lock, options, None)?;
    assert_eq!(built.len(), specs.len());
    println!("verified {} blocked prefill cubins", built.len());
    Ok(())
}
