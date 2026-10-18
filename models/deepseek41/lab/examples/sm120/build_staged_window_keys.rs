//! Initialize the native bounded-list family with its dependency probes.
#[path = "../../../bundle/src/sm120/latent_window_stage.rs"]
#[allow(dead_code)]
mod registration;
use kernel_build::{BuildOptions, LockDir, Toolchain};
use model_deepseek41_contract::geometry::GEOMETRIES;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: build_staged_window_keys OUT_DIR")?;
    std::fs::create_dir_all(&out)?;
    let out = std::fs::canonicalize(out)?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repository = std::fs::canonicalize(root.join("../../.."))?;
    if out.starts_with(&repository) {
        return Err("build outputs must stay outside the repository".into());
    }
    let specs = registration::key_cubins(GEOMETRIES)?
        .into_iter()
        .map(|(_, spec)| spec)
        .collect::<Vec<_>>();
    let dir = root.join("../bundle/kernels.lock.d");
    let mut options = BuildOptions::from_env();
    options.emit_cargo_directives = false;
    if options.profile_line_info {
        return Err("lock verification requires line info off".into());
    }
    if options.lock_update
        && dir
            .join(format!("{}.lock", registration::KEY_FAMILY))
            .exists()
    {
        return Err(
            "build_staged_window_keys only initializes a new family; it cannot replace one".into(),
        );
    }
    let toolchain = Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA is required for the cubin build".into());
    }
    let lock = LockDir::new(dir).family(registration::KEY_FAMILY, specs.iter().map(|s| &s.name));
    let mut built =
        kernel_build::build_in_families(&toolchain, &specs, &out, &lock, options, None)?;
    assert_eq!(built.len(), specs.len());
    // Keep the probes under their shared lab lock. The combined report
    // retains both original lock paths and every source/cubin identity.
    let report_path = out.join(kernel_build::report::BUILD_REPORT_FILE);
    let mut report = kernel_build::report::BuildReport::read(&report_path)?;
    let probe = kernels_sm120::probes::cubin_spec();
    let probe_lock = LockDir::new(repository.join("crates/kernel-lab/kernels.lock.d"))
        .family("dependency_probes", [&probe.name]);
    let probe_options = BuildOptions {
        lock_update: false,
        ..options
    };
    built.extend(kernel_build::build_in_families(
        &toolchain,
        &[probe],
        &out,
        &probe_lock,
        probe_options,
        None,
    )?);
    let probes = kernel_build::report::BuildReport::read(&report_path)?;
    probes.verify_current()?;
    assert_eq!(report.compiler_path, probes.compiler_path);
    report.source_membership_complete &= probes.source_membership_complete;
    report.source_ownership_complete &= probes.source_ownership_complete;
    report.entry_declarations_complete &= probes.entry_declarations_complete;
    report.lock_families.extend(probes.lock_families);
    report.files.extend(probes.files);
    report.cubins.extend(probes.cubins);
    report.cubins.sort_by(|a, b| a.name.cmp(&b.name));
    std::fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;
    if !options.lock_update {
        report.verify_current()?;
    }

    println!(
        "verified {} cubins and {} entries",
        built.len(),
        built.iter().map(|c| c.entries.len()).sum::<usize>()
    );
    Ok(())
}
