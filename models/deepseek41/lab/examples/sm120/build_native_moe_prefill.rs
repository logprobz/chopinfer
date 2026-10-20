//! Build the native MoE prefill lab cubin outside the repository and check
//! it against its lock (`--initialize` writes a new family's lock once).
#[allow(dead_code)]
#[path = "../../sm120/native_moe.rs"]
mod family;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::path::PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("usage: build_native_moe_prefill OUT [--initialize]")?,
    );
    std::fs::create_dir_all(&out)?;
    let out = std::fs::canonicalize(out)?;
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if out.starts_with(std::fs::canonicalize(root.join("../../.."))?) {
        return Err("output must be outside the repository".into());
    }
    let initialize = match std::env::args().nth(2).as_deref() {
        None => false,
        Some("--initialize") => true,
        _ => return Err("expected --initialize or no second argument".into()),
    };
    let mut options = kernel_build::BuildOptions::from_env();
    if options.lock_update {
        return Err("use --initialize on this builder, without KERNELS_LOCK_UPDATE".into());
    }
    if options.profile_line_info {
        return Err("locked build requires line info off".into());
    }
    options.lock_update = initialize;
    options.emit_cargo_directives = false;
    let dir = root.join("kernels.lock.d");
    if initialize && dir.join(format!("{}.lock", family::FAMILY)).exists() {
        return Err("cannot overwrite an existing family".into());
    }
    let toolchain = kernel_build::Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA compiler is required".into());
    }
    let specs = family::cubins();
    let locks =
        kernel_build::LockDir::new(dir).family(family::FAMILY, specs.iter().map(|s| &s.name));
    let built = kernel_build::build_in_families(&toolchain, &specs, &out, &locks, options, None)?;
    println!(
        "native_moe_build cubins={} entries={}",
        built.len(),
        built.iter().map(|c| c.entries.len()).sum::<usize>()
    );
    Ok(())
}
