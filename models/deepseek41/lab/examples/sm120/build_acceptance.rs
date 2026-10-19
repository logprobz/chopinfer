//! Initialize or verify only the DSpark acceptance lock family.
#[path = "../../../bundle/src/sm120/acceptance_cubin.rs"]
mod acceptance_cubin;
#[path = "../../../bundle/src/sm120/acceptance_entries.rs"]
mod acceptance_entries;

use kernel_build::{BuildOptions, LockDir, Toolchain};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let out = PathBuf::from(
        args.next()
            .ok_or("usage: build_acceptance OUT [--initialize]")?,
    );
    let initialize = match args.next() {
        None => false,
        Some(arg) if arg == "--initialize" => true,
        _ => return Err("unsupported DSpark acceptance build argument".into()),
    };
    if args.next().is_some() {
        return Err("surplus DSpark acceptance build arguments".into());
    }
    std::fs::create_dir_all(&out)?;
    let out = std::fs::canonicalize(out)?;
    let bundle = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../bundle")
        .canonicalize()?;
    if out.starts_with(bundle.join("../../..").canonicalize()?) {
        return Err("build outputs must stay outside the repository".into());
    }
    let dir = bundle.join("kernels.lock.d");
    if initialize
        && dir
            .join(format!("{}.lock", acceptance_entries::FAMILY))
            .exists()
    {
        return Err("DSpark acceptance initialization cannot replace an existing lock".into());
    }
    let built = kernel_build::build_in_families(
        &Toolchain::detect()?,
        &[acceptance_cubin::cubin_spec(&bundle)],
        &out,
        &LockDir::new(dir).family(acceptance_entries::FAMILY, [acceptance_entries::CUBIN]),
        BuildOptions {
            lock_update: initialize,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
        None,
    )?;
    println!("verified {} DSpark acceptance cubin", built.len());
    Ok(())
}
