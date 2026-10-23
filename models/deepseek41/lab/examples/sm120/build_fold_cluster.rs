//! Build the lab cluster fold family into an external directory against its
//! lock (`--initialize` creates the lock), or with `--bundle` the bundle's
//! served cluster family against the bundle lock, which only reproduces it.
#[allow(dead_code)]
#[path = "../../sm120/fold_cluster.rs"]
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
    if options.lock_update || options.profile_line_info {
        return Err("use --initialize without environment lock updates".into());
    }
    let mode = std::env::args().nth(2);
    let bundle = mode.as_deref() == Some("--bundle");
    options.lock_update = mode.as_deref() == Some("--initialize");
    let (dir, name, specs) = if bundle {
        let (name, spec) = bundle_deepseek41::sm120::fold_cluster::cubins(std::slice::from_ref(
            &model_deepseek41_contract::geometry::FLASH,
        ))?
        .pop()
        .ok_or("the bundle builds no cluster family")?;
        (root.join("../bundle/kernels.lock.d"), name, vec![spec])
    } else {
        (
            root.join("kernels.lock.d"),
            family::FAMILY,
            family::cubins(),
        )
    };
    if options.lock_update && dir.join(format!("{name}.lock")).exists() {
        return Err("cannot overwrite an existing family".into());
    }
    let locks = kernel_build::LockDir::new(dir).family(name, specs.iter().map(|s| &s.name));
    let toolchain = kernel_build::Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA compiler required".into());
    }
    let built = kernel_build::build_in_families(&toolchain, &specs, &out, &locks, options, None)?;
    println!(
        "FOLD_CLUSTER_BUILD family={name} cubins={} entries={}",
        built.len(),
        built.iter().map(|s| s.entries.len()).sum::<usize>()
    );
    Ok(())
}
