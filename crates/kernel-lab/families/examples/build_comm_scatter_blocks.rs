//! Create or verify only the additive scatter-block family's lock.
use kernel_build::{BuildOptions, LockDir, Toolchain};
use kernel_lab_families::comm_scatter_blocks::{FAMILY, cubins};
use std::path::PathBuf;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: build_comm_scatter_blocks OUT_DIR")?;
    std::fs::create_dir_all(&out)?;
    let out = std::fs::canonicalize(out)?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if out.starts_with(std::fs::canonicalize(root.join("../../.."))?) {
        return Err("build outputs must stay outside the repository".into());
    }
    let dir = root.join("../kernels.lock.d");
    let mut options = BuildOptions::from_env();
    options.emit_cargo_directives = false;
    if options.profile_line_info
        || options.lock_update && dir.join(format!("{FAMILY}.lock")).exists()
    {
        return Err("cannot profile or replace an existing family lock".into());
    }
    let toolchain = Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA is required".into());
    }
    let specs = cubins();
    let lock = LockDir::new(dir).family(FAMILY, specs.iter().map(|s| &s.name));
    let built = kernel_build::build_in_families(&toolchain, &specs, &out, &lock, options, None)?;
    assert_eq!(built.len(), 1);
    assert_eq!(built[0].entries.len(), 3);
    Ok(())
}
