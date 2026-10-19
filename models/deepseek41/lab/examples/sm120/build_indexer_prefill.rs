//! Build the comparison probe separately from every serving lock.
#[path = "../../tests/sm120/indexer_prefill_probe.rs"]
mod probe;
use kernel_build::{BuildOptions, LockDir, Toolchain};
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
        return Err("outputs must stay outside the repository".into());
    }
    let initialize = std::env::args().nth(2).as_deref() == Some("--initialize");
    let locks = root.join("kernels.lock.d");
    if initialize && locks.join("indexer_prefill_probe.lock").exists() {
        return Err("cannot replace an existing probe lock".into());
    }
    let spec = probe::spec();
    kernel_build::build_in_families(
        &Toolchain::detect()?,
        &[spec],
        &out,
        &LockDir::new(locks).family("indexer_prefill_probe", ["indexer_prefill_probe"]),
        BuildOptions {
            lock_update: initialize,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
        None,
    )?;
    Ok(())
}
