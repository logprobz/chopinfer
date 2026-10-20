//! Initialize the candidate scorer's new serving family without updating locks.
#[path = "../../../bundle/src/sm120/indexer.rs"]
#[allow(dead_code)]
mod indexer;
#[path = "../../../bundle/src/sm120/indexer_mma_prefill_tile.rs"]
mod registration;
use kernel_build::{BuildOptions, LockDir, Toolchain};
use model_deepseek41_contract::geometry::GEOMETRIES;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut args = std::env::args_os().skip(1);
    let out = PathBuf::from(args.next().ok_or("expected external output directory")?);
    let initialize = match args.next() {
        None => false,
        Some(arg) if arg == "--initialize" => true,
        _ => return Err("expected --initialize".into()),
    };
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    std::fs::create_dir_all(&out)?;
    let out = out.canonicalize()?;
    if out.starts_with(root.join("../../..").canonicalize()?) {
        return Err("artifacts must stay outside the repository".into());
    }
    let dir = root.join("../bundle/kernels.lock.d");
    let mut options = BuildOptions::from_env();
    options.emit_cargo_directives = false;
    if options.profile_line_info || options.lock_update {
        return Err("use --initialize for a new family; environment overrides are refused".into());
    }
    options.lock_update = initialize && !dir.join("position_mma_prefill_tile.lock").exists();
    let toolchain = Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA compilation is required".into());
    }
    let specs = registration::cubins(GEOMETRIES)?
        .into_iter()
        .map(|(_, s)| s)
        .collect::<Vec<_>>();
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].name, registration::CUBIN);
    let locks = LockDir::new(dir).family(registration::FAMILY, specs.iter().map(|s| &s.name));
    let built = kernel_build::build_in_families(&toolchain, &specs, &out, &locks, options, None)?;
    assert_eq!(built.len(), 1);
    assert_eq!(
        kernel_build::audit_cubin(&toolchain, &specs[0], &built[0].path)?.len(),
        8
    );
    Ok(())
}
