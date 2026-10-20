//! Build separately locked HC prefill candidates without registering them.
#[path = "../../tests/sm120/hc_prefill_warp/catalog.rs"]
#[allow(dead_code)]
mod catalog;
use kernel_build::{BuildOptions, LockDir, Toolchain};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let out = PathBuf::from(
        args.next()
            .ok_or("usage: build_hc_prefill_warp OUT [--initialize]")?,
    );
    let initialize = match args.next() {
        None => false,
        Some(a) if a == "--initialize" => true,
        _ => return Err("expected --initialize or no second argument".into()),
    };
    if args.next().is_some() {
        return Err("unexpected build argument".into());
    }
    std::fs::create_dir_all(&out)?;
    let out = out.canonicalize()?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if out.starts_with(root.join("../../..").canonicalize()?) {
        return Err("build outputs must stay outside the repository".into());
    }
    let locks = root.join("kernels.lock.d");
    if initialize && locks.join(format!("{}.lock", catalog::FAMILY)).exists() {
        return Err("initialization cannot replace an existing family".into());
    }
    let toolchain = Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA compilation is required".into());
    }
    let specs = catalog::specs();
    let built = kernel_build::build_in_families(
        &toolchain,
        &specs,
        &out,
        &LockDir::new(locks).family(catalog::FAMILY, specs.iter().map(|s| &s.name)),
        BuildOptions {
            lock_update: initialize,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
        None,
    )?;
    assert_eq!(built.len(), specs.len());
    println!("verified {} HC prefill modules", built.len());
    Ok(())
}
