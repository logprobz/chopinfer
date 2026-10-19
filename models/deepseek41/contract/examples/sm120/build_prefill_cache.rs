//! Initialize or verify the independently locked decoded prefill cache.
#[path = "../../../bundle/src/sm120/latent.rs"]
#[allow(dead_code)]
mod latent;
#[path = "../../../bundle/src/sm120/prefill_attention.rs"]
#[allow(dead_code)]
mod prefill_attention;
#[path = "../../../bundle/src/sm120/prefill_cache.rs"]
#[allow(dead_code)]
mod registration;
use kernel_build::{BuildOptions, LockDir, Toolchain};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: build_prefill_cache OUT_DIR")?;
    std::fs::create_dir_all(&out)?;
    let out = std::fs::canonicalize(out)?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if out.starts_with(std::fs::canonicalize(root.join("../../.."))?) {
        return Err("build outputs must stay outside the repository".into());
    }
    let specs = vec![registration::cubin()?];
    let dir = root.join("../bundle/kernels.lock.d");
    let mut options = BuildOptions::from_env();
    options.emit_cargo_directives = false;
    if options.profile_line_info {
        return Err("lock verification requires line info off".into());
    }
    if options.lock_update && dir.join(format!("{}.lock", registration::FAMILY)).exists() {
        return Err("cache lock initialization cannot replace an existing family".into());
    }
    let toolchain = Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA is required for the cubin build".into());
    }
    let lock = LockDir::new(dir).family(registration::FAMILY, specs.iter().map(|spec| &spec.name));
    let built = kernel_build::build_in_families(&toolchain, &specs, &out, &lock, options, None)?;
    assert_eq!(built.len(), 1);
    if !options.lock_update {
        kernel_build::report::BuildReport::read(
            &out.join(kernel_build::report::BUILD_REPORT_FILE),
        )?
        .verify_current()?;
    }
    println!("verified decoded prefill cache cubin");
    Ok(())
}
