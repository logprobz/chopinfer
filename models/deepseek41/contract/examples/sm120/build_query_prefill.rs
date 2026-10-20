//! Initialize only the new prefill query norm lock, then build lab controls.
#[path = "../../../bundle/src/sm120/query_prefill.rs"]
#[allow(dead_code)]
mod registration;
use kernel_build::{BuildOptions, LockDir, Toolchain};
use model_deepseek41_contract::geometry::GEOMETRIES;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: build_query_prefill OUT_DIR [--initialize]")?;
    std::fs::create_dir_all(&out)?;
    let out = std::fs::canonicalize(out)?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if out.starts_with(std::fs::canonicalize(root.join("../../.."))?) {
        return Err("build outputs must stay outside the repository".into());
    }
    let specs = registration::cubins(GEOMETRIES)?
        .into_iter()
        .map(|(_, s)| s)
        .collect::<Vec<_>>();
    let dir = root.join("../bundle/kernels.lock.d");
    let mut options = BuildOptions::from_env();
    options.lock_update = std::env::args().any(|a| a == "--initialize");
    options.emit_cargo_directives = false;
    if options.profile_line_info {
        return Err("lock verification requires line info off".into());
    }
    if options.lock_update && dir.join(format!("{}.lock", registration::FAMILY)).exists() {
        return Err("refuse to replace an existing kernel lock".into());
    }
    let toolchain = Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA is required for the cubin build".into());
    }
    let lock = LockDir::new(dir).family(registration::FAMILY, specs.iter().map(|s| &s.name));
    let built = kernel_build::build_in_families(&toolchain, &specs, &out, &lock, options, None)?;
    assert_eq!(built.len(), 1);
    let control = kernels_sm120::row::rmsnorm_f32_prefill::cubin_spec(
        "query_prefill_controls_sm120",
        &registration::controls()?,
    )?;
    let locks = LockDir::new(out.join("control-locks")).family("controls", [&control.name]);
    options.lock_update = true;
    kernel_build::build_in_families(&toolchain, &[control], &out, &locks, options, None)?;
    println!("QUERY_PREFILL_BUILD cubins=2 entries=4");
    Ok(())
}
