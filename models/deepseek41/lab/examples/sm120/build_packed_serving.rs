//! Build the production-shape parity cubin; initialize only a new lock.
#[allow(dead_code)]
#[path = "../../sm120/packed_serving.rs"]
mod family;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::path::PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("expected external output directory")?,
    );
    std::fs::create_dir_all(&out)?;
    let out = out.canonicalize()?;
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if out.starts_with(root.join("../../..").canonicalize()?) {
        return Err("external output required".into());
    }
    let mut options = kernel_build::BuildOptions::from_env();
    options.emit_cargo_directives = false;
    if options.profile_line_info {
        return Err("family locks require line info off".into());
    }
    let dir = root.join("kernels.lock.d");
    if options.lock_update && dir.join(format!("{}.lock", family::FAMILY)).exists() {
        return Err("cannot overwrite an existing family".into());
    }
    let specs = family::cubins();
    let locks =
        kernel_build::LockDir::new(dir).family(family::FAMILY, specs.iter().map(|s| &s.name));
    let toolchain = kernel_build::Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA compiler required".into());
    }
    let built = kernel_build::build_in_families(&toolchain, &specs, &out, &locks, options, None)?;
    assert_eq!(built.len(), 1);
    println!(
        "verified {}: {} entries",
        family::FAMILY,
        built[0].entries.len()
    );
    Ok(())
}
