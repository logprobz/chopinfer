//! Initialize or verify only the Engram hash lock family.
#[path = "../../../bundle/src/sm120/engram_hash_cubin.rs"]
mod engram_hash_cubin;
#[path = "../../../bundle/src/sm120/engram_hash_entries.rs"]
mod engram_hash_entries;

use kernel_build::{BuildOptions, LockDir, Toolchain};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let out = PathBuf::from(
        args.next()
            .ok_or("usage: build_engram_hash OUT [--initialize]")?,
    );
    let initialize = match args.next() {
        None => false,
        Some(arg) if arg == "--initialize" => true,
        _ => return Err("unsupported Engram hash build argument".into()),
    };
    if args.next().is_some() {
        return Err("surplus Engram hash build arguments".into());
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
            .join(format!("{}.lock", engram_hash_entries::FAMILY))
            .exists()
    {
        return Err("Engram hash initialization cannot replace an existing lock".into());
    }
    let built = kernel_build::build_in_families(
        &Toolchain::detect()?,
        &[engram_hash_cubin::cubin_spec(&bundle)],
        &out,
        &LockDir::new(dir).family(engram_hash_entries::FAMILY, [engram_hash_entries::CUBIN]),
        BuildOptions {
            lock_update: initialize,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
        None,
    )?;
    println!("verified {} Engram hash cubin", built.len());
    Ok(())
}
