//! Build only the cached ceil quantizer family.
#[path = "../../tests/sm120/ceil_cached/catalog.rs"]
#[allow(dead_code)]
mod catalog;
use kernel_build::{BuildOptions, LockDir, Toolchain};
use std::path::PathBuf;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("expected output directory")?,
    );
    std::fs::create_dir_all(&out)?;
    let out = out.canonicalize()?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if out.starts_with(root.join("../../..").canonicalize()?) {
        return Err("outputs must stay outside the repository".into());
    }
    let initialize = std::env::args().any(|a| a == "--initialize");
    let dir = root.join("kernels.lock.d");
    if initialize && dir.join(format!("{}.lock", catalog::FAMILY)).exists() {
        return Err("refuse to replace an existing lock".into());
    }
    let tc = Toolchain::detect()?;
    if tc.skipped {
        return Err("CUDA compilation is required".into());
    }
    let specs = catalog::specs();
    kernel_build::build_in_families(
        &tc,
        &specs,
        &out,
        &LockDir::new(dir).family(catalog::FAMILY, specs.iter().map(|s| &s.name)),
        BuildOptions {
            lock_update: initialize,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
        None,
    )?;
    println!(
        "CEIL_CACHED_BUILD modules=1 entries={}",
        specs[0].entries.len()
    );
    Ok(())
}
