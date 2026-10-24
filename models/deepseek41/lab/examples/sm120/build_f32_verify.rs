//! Initialize only the new separate-span F32 family, then audit it without a GPU.
#[path = "../../../bundle/src/sm120/f32_verify.rs"]
#[allow(dead_code)]
mod candidate;
use kernel_build::{BuildOptions, LockDir, Toolchain};
use std::path::PathBuf;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let out = PathBuf::from(
        args.next()
            .ok_or("usage: build_f32_verify OUT [--initialize]")?,
    );
    let initialize = match args.next() {
        None => false,
        Some(v) if v == "--initialize" => true,
        _ => return Err("unexpected option".into()),
    };
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    std::fs::create_dir_all(&out)?;
    let out = out.canonicalize()?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if out.starts_with(root.join("../../..").canonicalize()?) {
        return Err("artifacts must be external".into());
    }
    let locks = root.join("../bundle/kernels.lock.d");
    if initialize && locks.join(format!("{}.lock", candidate::FAMILY)).exists() {
        return Err("cannot replace an existing family".into());
    }
    let toolchain = Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA compilation required".into());
    }
    let specs = candidate::cubins(&[model_deepseek41_contract::geometry::FLASH])?
        .into_iter()
        .map(|(_, s)| s)
        .collect::<Vec<_>>();
    let built = kernel_build::build_in_families(
        &toolchain,
        &specs,
        &out,
        &LockDir::new(locks).family(candidate::FAMILY, [candidate::CUBIN]),
        BuildOptions {
            lock_update: initialize,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
        None,
    )?;
    if built.len() != 1 || built[0].entries.len() != 15 {
        return Err("missing explicit-span stages".into());
    }
    for spec in &specs {
        kernel_build::audit_cubin(&toolchain, spec, &out.join(format!("{}.cubin", spec.name)))?;
    }
    println!("F32_VERIFY_MODULES modules=1 stages=15 PASS");
    Ok(())
}
