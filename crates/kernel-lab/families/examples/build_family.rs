//! Build and verify one complete kernel-lab family without building the
//! model catalogs. Lock updates initialize new families only.

use kernel_build::{BuildOptions, LockDir, Toolchain};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        return Err("usage: build_family FAMILY OUT_DIR".into());
    }
    let (family, mut specs) = kernel_lab_families::all()
        .into_iter()
        .find(|(name, _)| *name == args[0])
        .ok_or("unknown kernel-lab family")?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(&args[1]);
    std::fs::create_dir_all(&out)?;
    let out = std::fs::canonicalize(out)?;
    let repository = std::fs::canonicalize(root.join("../../.."))?;
    if out.starts_with(repository) {
        return Err("family artifacts must stay outside the repository".into());
    }
    let mut options = BuildOptions::from_env();
    options.emit_cargo_directives = false;
    if options.profile_line_info {
        return Err("family lock verification requires line info off".into());
    }
    let dir = root.join("../kernels.lock.d");
    if options.lock_update && dir.join(format!("{family}.lock")).exists() {
        return Err(
            "build_family only initializes new locks; it cannot replace an existing family".into(),
        );
    }
    let mut lock = LockDir::new(dir).family(family, specs.iter().map(|s| &s.name));
    // Standalone conformance needs its dependency probes and, for glue,
    // numerical controls. The full catalog still assigns each cubin once.
    let dependencies = match family {
        "latent_glue" => &["dependency_probes", "latent_glue_mutants"][..],
        "latent_window_stage" => &[
            "dependency_probes",
            "latent_staged_window_keys",
            "latent_window_chain",
            "latent_glue",
        ][..],
        "rmsnorm_f32" | "latent_staged_window_keys" => &["dependency_probes"][..],
        _ => &[][..],
    };
    for (name, required) in kernel_lab_families::all() {
        if dependencies.contains(&name) {
            lock = lock.family(name, required.iter().map(|s| &s.name));
            specs.extend(required);
        }
    }
    let toolchain = Toolchain::detect()?;
    if toolchain.skipped {
        return Err("family compilation requires CUDA".into());
    }
    let built = kernel_build::build_in_families(&toolchain, &specs, &out, &lock, options, None)?;
    if built.len() != specs.len() || built.is_empty() {
        return Err("family build omitted cubins".into());
    }
    println!(
        "verified {family}: {} cubins, {} entries",
        built.len(),
        built.iter().map(|c| c.entries.len()).sum::<usize>()
    );
    Ok(())
}
