//! Build one locked DeepSeek family and its qualification dependencies.
#[path = "../../sm120/dense_decode_reference.rs"]
mod dense_decode_reference;
mod family;

use family::{Export, Recipe, Result, Root};
use kernel_build::{BuildOptions, LockDir, Toolchain, report::BuildReport};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

fn output(path: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(path)?;
    let out = path.canonicalize()?;
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()?;
    if out.starts_with(repository) {
        return Err("family artifacts must stay outside the repository".into());
    }
    Ok(out)
}

fn build(recipe: &Recipe, out: &Path, options: BuildOptions, toolchain: &Toolchain) -> Result<()> {
    let steps = (recipe.steps)()?;
    if steps.is_empty() {
        return Err("family has no build steps".into());
    }
    let mut records = Vec::new();
    let mut reports: BTreeMap<PathBuf, BuildReport> = BTreeMap::new();
    for step in steps {
        if step.specs.is_empty() {
            return Err("family has no cubins".into());
        }
        let dir = if recipe.family == "routed_moe_floor" && step.root == Root::Bundle {
            std::env::var_os("CHOPINFER_MOE_FLOOR_LOCK_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| step.root.path(out))
        } else {
            step.root.path(out)
        };
        let families: BTreeMap<_, Vec<_>> =
            step.specs
                .iter()
                .fold(BTreeMap::new(), |mut map, (name, spec)| {
                    map.entry(*name).or_default().push(&spec.name);
                    map
                });
        let external_lock = matches!(step.root, Root::Output(_));
        let initialize = if external_lock {
            families
                .keys()
                .all(|name| !dir.join(format!("{name}.lock")).exists())
        } else {
            step.initialize && options.lock_update
        };
        if initialize
            && families
                .keys()
                .any(|name| dir.join(format!("{name}.lock")).exists())
        {
            return Err(
                "build_family only initializes absent locks; it cannot replace a family".into(),
            );
        }
        let lock = families
            .iter()
            .fold(LockDir::new(&dir), |lock, (name, names)| {
                lock.family(*name, names.iter().copied())
            });
        let specs: Vec<_> = step.specs.iter().map(|(_, spec)| spec.clone()).collect();
        let directory = out.join(step.subdir);
        if initialize {
            kernel_build::build_in_families(
                toolchain,
                &specs,
                &directory,
                &lock,
                BuildOptions {
                    lock_update: true,
                    ..options
                },
                None,
            )?;
        }
        let built = kernel_build::build_in_families(
            toolchain,
            &specs,
            &directory,
            &lock,
            BuildOptions {
                lock_update: false,
                ..options
            },
            None,
        )?;
        if built.len() != specs.len() || built.is_empty() {
            return Err("family build omitted cubins".into());
        }
        for (name, spec) in &step.specs {
            let image = built
                .iter()
                .find(|image| image.name == spec.name)
                .ok_or("family build omitted a named cubin")?;
            let resources = kernel_build::audit_cubin(toolchain, spec, &image.path)?;
            if resources.len() != spec.entries.len() {
                return Err("family audit omitted entries".into());
            }
            let bytes = std::fs::read(&image.path)?;
            let path = directory.join(format!("{}.cubin", image.name));
            std::fs::write(&path, &bytes)?;
            records.push(json!({
                "name": image.name, "module": image.name, "family": name,
                "root": step.root.label(), "lock": dir.join(format!("{name}.lock")),
                "path": path, "sha256": kernel_build::digest(&bytes),
                "symbols": spec.entries.iter().map(|e| &e.symbol).collect::<Vec<_>>(),
                "resources": resources,
            }));
        }
        let path = directory.join(kernel_build::report::BUILD_REPORT_FILE);
        let report = BuildReport::read(&path)?;
        report.verify_current()?;
        if let Some(previous) = reports.get_mut(&path) {
            if previous.compiler_path != report.compiler_path {
                return Err("dependency compiler changed".into());
            }
            previous.source_membership_complete &= report.source_membership_complete;
            previous.source_ownership_complete &= report.source_ownership_complete;
            previous.entry_declarations_complete &= report.entry_declarations_complete;
            previous.lock_families.extend(report.lock_families);
            previous.files.extend(report.files);
            previous.cubins.extend(report.cubins);
            previous.cubins.sort_by(|a, b| a.name.cmp(&b.name));
        } else {
            reports.insert(path, report);
        }
    }
    for (path, report) in reports {
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
        report.verify_current()?;
    }
    export(recipe, out, &records)?;
    std::fs::write(
        out.join("family-build.json"),
        serde_json::to_vec_pretty(&json!({
            "family": recipe.key(), "modules": records,
        }))?,
    )?;
    println!(
        "verified {}: {} cubins, {} entries",
        recipe.key(),
        records.len(),
        records
            .iter()
            .map(|r| r["symbols"].as_array().unwrap().len())
            .sum::<usize>()
    );
    Ok(())
}

fn export(recipe: &Recipe, out: &Path, records: &[Value]) -> Result<()> {
    match recipe.export {
        Export::None => {}
        Export::Modules(file) => {
            let modules: Vec<_> = records.iter().map(|r| json!({
                "family": r["family"], "module": r["module"], "sha256": r["sha256"], "symbols": r["symbols"],
            })).collect();
            std::fs::write(out.join(file), serde_json::to_vec_pretty(&modules)?)?;
        }
        Export::Attention => {
            let mut modules: Vec<_> = records
                .iter()
                .map(|r| {
                    json!({
                        "name": r["name"], "sha256": r["sha256"], "symbols": r["symbols"],
                        "family": r["family"], "resources": r["resources"],
                    })
                })
                .collect();
            modules.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
            std::fs::write(
                out.join("attention-modules.json"),
                serde_json::to_vec_pretty(&json!({"schema": 1, "modules": modules}))?,
            )?;
        }
        Export::RouterReferences => {
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
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let recipes = family::all();
    if args == ["--list"] {
        for recipe in recipes {
            println!("{}", recipe.key());
        }
        return Ok(());
    }
    if args.len() != 2 {
        return Err("usage: build_family FAMILY OUT_DIR (or --list)".into());
    }
    let recipe = recipes
        .iter()
        .find(|r| args[0] == r.key().as_str())
        .ok_or("unknown DeepSeek family; use --list")?;
    let out = output(Path::new(&args[1]))?;
    let mut options = BuildOptions::from_env();
    options.emit_cargo_directives = false;
    if options.profile_line_info {
        return Err("family lock verification requires line info off".into());
    }
    let toolchain = Toolchain::detect()?;
    if toolchain.skipped {
        return Err("family compilation requires CUDA".into());
    }
    build(recipe, &out, options, &toolchain)
}
