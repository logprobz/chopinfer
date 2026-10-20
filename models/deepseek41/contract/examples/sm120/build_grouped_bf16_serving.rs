#[allow(dead_code)]
#[path = "../../../bundle/src/sm120/moe_grouped_entries.rs"]
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
        return Err("output must be outside repository".into());
    }
    let dir = root.join("../bundle/kernels.lock.d");
    let mut options = kernel_build::BuildOptions::from_env();
    options.emit_cargo_directives = false;
    if options.profile_line_info {
        return Err("locked build needs line info off".into());
    }
    if options.lock_update && dir.join(format!("{}.lock", family::FAMILY)).exists() {
        return Err("cannot replace an existing family".into());
    }
    let specs = vec![family::cubin()];
    let locks =
        kernel_build::LockDir::new(dir).family(family::FAMILY, specs.iter().map(|s| &s.name));
    let toolchain = kernel_build::Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA compiler required".into());
    }
    kernel_build::build_in_families(&toolchain, &specs, &out, &locks, options, None)?;
    println!("GROUPED_SERVING_BUILD entries=2");
    Ok(())
}
