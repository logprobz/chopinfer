//! Build immutable reference and candidate attention artifacts without a GPU.

#[path = "../../../bundle/src/sm120/latent_scheduled.rs"]
#[allow(dead_code)]
mod candidate;
#[path = "../../../bundle/src/sm120/latent_decode.rs"]
#[allow(dead_code)]
mod decode_candidate;
#[path = "../../../bundle/src/sm120/latent_glue.rs"]
#[allow(dead_code)]
mod glue;
#[path = "../../../bundle/src/sm120/latent.rs"]
#[allow(dead_code)]
mod latent;
#[path = "../../tests/sm120/attention_pair/probe.rs"]
mod state_probe;

use kernel_build::{BuildOptions, CubinSpec, LockDir, Toolchain};
use model_deepseek41_contract::geometry::GEOMETRIES;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn build_family(
    toolchain: &Toolchain,
    directory: &Path,
    locks: &Path,
    family: &str,
    specs: &[CubinSpec],
    initialize: bool,
) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
    if initialize && locks.join(format!("{family}.lock")).exists() {
        return Err("initialization cannot replace an existing family".into());
    }
    let lock = LockDir::new(locks).family(family, specs.iter().map(|s| &s.name));
    let built = kernel_build::build_in_families(
        toolchain,
        specs,
        directory,
        &lock,
        BuildOptions {
            lock_update: initialize,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
        None,
    )?;
    if built.len() != specs.len() {
        return Err("a required attention cubin was not built".into());
    }
    built
        .iter()
        .map(|image| {
            let spec = specs
                .iter()
                .find(|s| s.name == image.name)
                .ok_or("unknown cubin")?;
            let entries = kernel_build::audit_cubin(toolchain, spec, &image.path)?;
            if entries.len() != spec.entries.len() {
                return Err("a required entry was not audited".into());
            }
            let bytes = std::fs::read(&image.path)?;
            std::fs::write(directory.join(format!("{}.cubin", image.name)), &bytes)?;
            Ok(json!({
                "name": image.name,
                "sha256": kernel_build::digest(&bytes),
                "symbols": spec.entries.iter().map(|e| &e.symbol).collect::<Vec<_>>(),
                "family": family,
                "resources": entries,
            }))
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let directory = PathBuf::from(
        args.next()
            .ok_or("usage: build_attention_pair OUT [--initialize]")?,
    );
    let initialize = match args.next() {
        None => false,
        Some(arg) if arg == "--initialize" => true,
        _ => return Err("expected --initialize or no second argument".into()),
    };
    if args.next().is_some() {
        return Err("unexpected build argument".into());
    }
    std::fs::create_dir_all(&directory)?;
    let directory = std::fs::canonicalize(directory)?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repository = std::fs::canonicalize(root.join("../../.."))?;
    if directory.starts_with(&repository) {
        return Err("attention artifacts must stay outside the repository".into());
    }
    let toolchain = Toolchain::detect()?;
    if toolchain.skipped {
        return Err("the attention pair requires CUDA compilation".into());
    }
    let locks = root.join("../bundle/kernels.lock.d");
    let mut families: BTreeMap<&str, Vec<CubinSpec>> = BTreeMap::new();
    for (family, spec) in latent::cubins(GEOMETRIES)?
        .into_iter()
        .filter(|(family, _)| *family != "latent_sources_bf16")
        .chain(glue::cubins(GEOMETRIES)?)
    {
        families.entry(family).or_default().push(spec);
    }
    let mut modules = Vec::new();
    for (family, specs) in families {
        // The reference families are always verified, including initialization.
        modules.extend(build_family(
            &toolchain, &directory, &locks, family, &specs, false,
        )?);
    }
    modules.extend(build_family(
        &toolchain,
        &directory,
        &locks,
        candidate::FAMILY,
        &candidate::cubins(GEOMETRIES)?,
        initialize && !locks.join(format!("{}.lock", candidate::FAMILY)).exists(),
    )?);
    modules.extend(build_family(
        &toolchain,
        &directory,
        &locks,
        candidate::QUERY_FAMILY,
        &[candidate::query_cubin()],
        initialize
            && !locks
                .join(format!("{}.lock", candidate::QUERY_FAMILY))
                .exists(),
    )?);
    modules.extend(build_family(
        &toolchain,
        &directory,
        &repository.join("crates/kernel-lab/kernels.lock.d"),
        "dependency_probes",
        &[kernels_sm120::probes::cubin_spec()],
        false,
    )?);
    let probe_locks = root.join("kernels.lock.d");
    modules.extend(build_family(
        &toolchain,
        &directory,
        &probe_locks,
        state_probe::FAMILY,
        &[state_probe::cubin()?],
        initialize
            && !probe_locks
                .join(format!("{}.lock", state_probe::FAMILY))
                .exists(),
    )?);
    let g = GEOMETRIES
        .iter()
        .find(|g| g.kernel_tag == "deepseek41")
        .unwrap();
    modules.extend(build_family(
        &toolchain,
        &directory,
        &locks,
        decode_candidate::FAMILY,
        &[decode_candidate::cubin(g)?],
        initialize
            && !locks
                .join(format!("{}.lock", decode_candidate::FAMILY))
                .exists(),
    )?);
    modules.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    std::fs::write(
        directory.join("attention-modules.json"),
        serde_json::to_vec_pretty(&json!({"schema": 1, "modules": modules}))?,
    )?;
    println!(
        "built and audited {} attention modules; add the kernel-lab timing module before snapshot validation",
        modules.len()
    );
    Ok(())
}
