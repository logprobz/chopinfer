//! Build and audit the wide router family in an external qualification stage.
#[path = "../../../bundle/src/sm120/moe.rs"]
#[allow(dead_code)]
mod registration;
use kernel_build::{BuildOptions, LockDir, Toolchain};
use kernels_sm120::moe::chained::router_finish64;
use model_deepseek41_contract::geometry::FLASH;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let out = PathBuf::from(args.next().ok_or("usage: build_moe_router64 OUT")?);
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    std::fs::create_dir_all(&out)?;
    let out = out.canonicalize()?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if out.starts_with(root.join("../../..").canonicalize()?) {
        return Err("artifacts must be external".into());
    }
    let entries = registration::floor_entries(&FLASH)?
        .into_iter()
        .filter(|e| e.experts() == 384 && e.top_k() == 6)
        .collect::<Vec<_>>();
    if entries.len() != 1 {
        return Err("expected one backbone router geometry".into());
    }
    let family = "routed_moe_router_finish64";
    let spec = router_finish64::cubin_spec("router_finish_moe_deepseek41_m64_sm120", &entries);
    let locks = out.join("kernels.lock.d");
    let toolchain = Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA compilation required".into());
    }
    let initialize = !locks.join(format!("{family}.lock")).exists();
    let built = kernel_build::build_in_families(
        &toolchain,
        std::slice::from_ref(&spec),
        &out,
        &LockDir::new(locks).family(family, [&spec.name]),
        BuildOptions {
            lock_update: initialize,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
        None,
    )?;
    if built.len() != 1 || built[0].entries.len() != 2 {
        return Err("missing wide router stages".into());
    }
    kernel_build::audit_cubin(&toolchain, &spec, &out.join(format!("{}.cubin", spec.name)))?;
    println!("MOE_ROUTER64_MODULES modules=1 stages=2 PASS");
    Ok(())
}
