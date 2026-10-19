//! Build the new rounded chain producer lock family before registration.
#[path = "../../../bundle/src/sm120/hc_round_expand_mix.rs"]
#[allow(dead_code)]
mod hc_round_expand_mix;
#[path = "../../../bundle/src/sm120/hc_round_chain.rs"]
mod registration;
use kernel_build::{BuildOptions, LockDir, Toolchain};
use model_deepseek41_contract::geometry::GEOMETRIES;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: build_hc_round_chain OUT_DIR")?;
    std::fs::create_dir_all(&out)?;
    let out = std::fs::canonicalize(out)?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repository = std::fs::canonicalize(root.join("../../.."))?;
    if out.starts_with(repository) {
        return Err("build outputs must stay outside the repository".into());
    }
    let specs = registration::cubins(GEOMETRIES)?
        .into_iter()
        .map(|(_, spec)| spec)
        .collect::<Vec<_>>();
    let dir = root.join("../bundle/kernels.lock.d");
    let mut options = BuildOptions::from_env();
    options.lock_update = std::env::args().any(|a| a == "--initialize");
    options.emit_cargo_directives = false;
    if options.profile_line_info {
        return Err("lock verification requires line info off".into());
    }
    if options.lock_update && dir.join(format!("{}.lock", registration::FAMILY)).exists() {
        return Err(
            "build_hc_round_chain only initializes a new family; it cannot replace one".into(),
        );
    }
    let toolchain = Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA is required for the cubin build".into());
    }
    let lock = LockDir::new(dir).family(registration::FAMILY, specs.iter().map(|s| &s.name));
    let built = kernel_build::build_in_families(&toolchain, &specs, &out, &lock, options, None)?;
    assert_eq!(built.len(), specs.len());
    println!(
        "verified {} cubins and {} entries",
        built.len(),
        built.iter().map(|c| c.entries.len()).sum::<usize>()
    );
    Ok(())
}
