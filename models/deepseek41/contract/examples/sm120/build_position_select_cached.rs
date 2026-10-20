//! Initialize only the new serving selector family, then verify it normally.
#[path = "../../../bundle/src/sm120/indexer.rs"]
#[allow(dead_code)]
mod indexer;
#[path = "../../../bundle/src/sm120/indexer_select_cached.rs"]
mod registration;
use kernel_build::{BuildOptions, LockDir, Toolchain};
use model_deepseek41_contract::geometry::GEOMETRIES;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("expected external output directory")?,
    );
    std::fs::create_dir_all(&out)?;
    let out = out.canonicalize()?;
    if out.starts_with(root.join("../../..").canonicalize()?) {
        return Err("artifacts must stay outside the repository".into());
    }
    let dir = root.join("../bundle/kernels.lock.d");
    let mut options = BuildOptions::from_env();
    options.emit_cargo_directives = false;
    if options.profile_line_info
        || options.lock_update && dir.join("position_select_cached.lock").exists()
    {
        return Err("only a new family may be initialized; line info must be off".into());
    }
    let toolchain = Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA compilation is required".into());
    }
    let specs = registration::cubins(GEOMETRIES)?
        .into_iter()
        .map(|(_, s)| s)
        .collect::<Vec<_>>();
    let lock = LockDir::new(dir).family(registration::FAMILY, specs.iter().map(|s| &s.name));
    let built = kernel_build::build_in_families(&toolchain, &specs, &out, &lock, options, None)?;
    assert_eq!(built.len(), specs.len());
    for (spec, built) in specs.iter().zip(built) {
        assert_eq!(
            kernel_build::audit_cubin(&toolchain, spec, &built.path)?.len(),
            spec.entries.len()
        );
    }
    Ok(())
}
