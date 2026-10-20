//! Compile separately locked prefill candidates without changing serving.
#[path = "../../../bundle/src/sm120/indexer.rs"]
#[allow(dead_code)]
mod entries;
use kernel_build::{BuildOptions, LockDir, Toolchain};
use kernels_sm120::pooled_indexer::position_mma_tile;
use model_deepseek41_contract::geometry::FLASH;
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
    let entries = entries::entries(&FLASH);
    let scores = entries
        .iter()
        .flat_map(|e| position_mma_tile::VARIANTS.map(|v| v.spec(e)))
        .collect::<Result<Vec<_>, _>>()?;
    let toolchain = Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA compilation is required".into());
    }
    let locks = root.join("kernels.lock.d");
    let (family, spec) = ("position_mma_tile", position_mma_tile::cubin_spec(scores));
    // An existing family is always checked without lock updates.
    let new_family = initialize && !locks.join(format!("{family}.lock")).exists();
    let built = kernel_build::build_in_families(
        &toolchain,
        std::slice::from_ref(&spec),
        &out,
        &LockDir::new(&locks).family(family, [&spec.name]),
        BuildOptions {
            lock_update: new_family,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
        None,
    )?;
    if built.len() != 1 {
        return Err("missing candidate cubin".into());
    }
    let audit = kernel_build::audit_cubin(&toolchain, &spec, &built[0].path)?;
    if audit.len() != spec.entries.len() {
        return Err("incomplete candidate audit".into());
    }
    let bytes = std::fs::read(&built[0].path)?;
    std::fs::write(out.join(format!("{}.cubin", spec.name)), &bytes)?;
    let records = [serde_json::json!({"family":family,"module":spec.name,
        "sha256":kernel_build::digest(&bytes),"symbols":spec.entries.iter().map(|e|&e.symbol).collect::<Vec<_>>() })];
    std::fs::write(
        out.join("candidate-modules.json"),
        serde_json::to_vec_pretty(&records)?,
    )?;
    Ok(())
}
