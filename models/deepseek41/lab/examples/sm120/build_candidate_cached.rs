//! Build the fused candidate publisher under a new lab-only lock.
use kernel_build::{BuildOptions, LockDir, Toolchain};
use kernels_sm120::pooled_indexer::{position_blocks::CandidateEntry, position_candidates_cached};
use model_deepseek41_contract::{geometry::FLASH, indexer, serving::PREFILL_TOKENS};
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
    let entry = CandidateEntry {
        name: FLASH.kernel_tag.into(),
        query: indexer::candidates(&FLASH, PREFILL_TOKENS),
    };
    let family = "position_candidates_cached";
    let spec = position_candidates_cached::cubin_spec(
        "position_candidates_cached_sm120",
        vec![position_candidates_cached::spec(&entry)?],
    );
    let toolchain = Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA compilation is required".into());
    }
    let locks = root.join("kernels.lock.d");
    let options = BuildOptions::from_env();
    if options.lock_update || options.profile_line_info {
        return Err("environment lock and line-info overrides are refused".into());
    }
    let built = kernel_build::build_in_families(
        &toolchain,
        std::slice::from_ref(&spec),
        &out,
        &LockDir::new(&locks).family(family, [&spec.name]),
        BuildOptions {
            lock_update: initialize && !locks.join(format!("{family}.lock")).exists(),
            emit_cargo_directives: false,
            profile_line_info: false,
        },
        None,
    )?;
    assert_eq!(built.len(), 1);
    assert_eq!(kernel_build::audit_cubin(&toolchain, &spec, &built[0].path)?.len(), 1);
    let bytes = std::fs::read(&built[0].path)?;
    std::fs::write(out.join(format!("{}.cubin", spec.name)), &bytes)?;
    std::fs::write(
        out.join("candidate-modules.json"),
        serde_json::to_vec_pretty(&vec![serde_json::json!({
            "family":family,"module":spec.name,"sha256":kernel_build::digest(&bytes),
            "symbols":spec.entries.iter().map(|e|&e.symbol).collect::<Vec<_>>()
        })])?,
    )?;
    Ok(())
}
