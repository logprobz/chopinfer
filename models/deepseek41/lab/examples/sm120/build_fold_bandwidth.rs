#[path = "../../sm120/fold_bandwidth.rs"]
#[allow(dead_code)]
mod family;
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = std::path::PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("expected output directory")?,
    );
    std::fs::create_dir_all(&out)?;
    let out = std::fs::canonicalize(out)?;
    if out.starts_with(std::fs::canonicalize(root.join("../../.."))?) {
        return Err("kernel outputs must be outside the repository".into());
    }
    let mut options = kernel_build::BuildOptions::from_env();
    if options.lock_update || options.profile_line_info {
        return Err(
            "use --initialize for a new family; keep global lock updates and line info off".into(),
        );
    }
    options.lock_update = match std::env::args().nth(2).as_deref() {
        None => false,
        Some("--initialize") => true,
        _ => return Err("expected --initialize or no option".into()),
    };
    let dir = root.join("kernels.lock.d");
    if options.lock_update && dir.join(format!("{}.lock", family::FAMILY)).exists() {
        return Err("cannot replace an existing family lock".into());
    }
    options.emit_cargo_directives = false;
    let toolchain = kernel_build::Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA compiler is required".into());
    }
    let specs = [family::cubin()?];
    let lock =
        kernel_build::LockDir::new(dir).family(family::FAMILY, specs.iter().map(|s| &s.name));
    let built = kernel_build::build_in_families(&toolchain, &specs, &out, &lock, options, None)?;
    println!(
        "direct_fold_build entries={}",
        built.iter().map(|c| c.entries.len()).sum::<usize>()
    );
    Ok(())
}
