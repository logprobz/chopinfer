//! Build the separate exact router-finish candidate without selecting it.
#[path = "../../../bundle/src/sm120/moe.rs"]
#[allow(dead_code)]
mod registration;
use kernel_build::{BuildOptions, LockDir, Toolchain};
use kernels_sm120::moe::chained::router_finish;
use model_deepseek41_contract::geometry::FLASH;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: build_moe_router_finish OUT_DIR")?;
    std::fs::create_dir_all(&out)?;
    let out = std::fs::canonicalize(out)?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if out.starts_with(std::fs::canonicalize(root.join("../../.."))?) {
        return Err("build outputs must stay outside the repository".into());
    }
    let spec = router_finish::cubin_spec(
        "router_finish_moe_deepseek41_sm120",
        &registration::floor_entries(&FLASH)?,
    );
    let family = "routed_moe_router_finish";
    let dir = root.join("../bundle/kernels.lock.d");
    let mut options = BuildOptions::from_env();
    options.emit_cargo_directives = false;
    if options.profile_line_info {
        return Err("lock verification requires line info off".into());
    }
    if options.lock_update && dir.join(format!("{family}.lock")).exists() {
        return Err("build_moe_router_finish cannot replace an existing lock".into());
    }
    let toolchain = Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA is required for the cubin build".into());
    }
    let lock = LockDir::new(dir).family(family, [&spec.name]);
    let built = kernel_build::build_in_families(&toolchain, &[spec], &out, &lock, options, None)?;
    // Export the immutable cached router and route modules for resource
    // comparison. This reads embedded cubins without creating a context.
    for (name, symbol) in [
        (
            "baseline_router",
            "chop_deepseek41_dense_f32_m8192_n384_k5120_v1_cached_reduce_v1",
        ),
        (
            "baseline_route",
            "chop_moe_sqrt_softplus_row_bias_e384k6_t64_v1",
        ),
        (
            "baseline_metadata",
            "chop_deepseek41_moe_floor_e384_k6_metadata_v1",
        ),
    ] {
        let (_, bytes) = bundle_deepseek41::sm120::fold_cubin_for_symbol(symbol)
            .ok_or("missing reference module")?;
        std::fs::write(out.join(format!("{name}.cubin")), bytes)?;
    }
    println!(
        "verified {} cubins and {} entries",
        built.len(),
        built.iter().map(|b| b.entries.len()).sum::<usize>()
    );
    Ok(())
}
