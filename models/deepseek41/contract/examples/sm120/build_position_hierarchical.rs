//! Build the separately locked native MMA position scorers.
#[path = "../../../bundle/src/sm120/indexer.rs"]
#[allow(dead_code)]
mod indexer;
use kernel_build::{BuildOptions, LockDir, Toolchain};
use model_deepseek41_contract::geometry::GEOMETRIES;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: build_position_hierarchical OUT_DIR")?;
    std::fs::create_dir_all(&out)?;
    let out = std::fs::canonicalize(out)?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repository = std::fs::canonicalize(root.join("../../.."))?;
    if out.starts_with(&repository) {
        return Err("build outputs must stay outside the repository".into());
    }
    use kernels_sm120::pooled_indexer::position_hierarchical as hierarchy;
    let families = GEOMETRIES
        .iter()
        .filter(|g| g.native_indexer_supported())
        .map(|g| {
            let mut entries = Vec::new();
            for entry in indexer::entries(g) {
                let pair = hierarchy::dense_specs(&entry).unwrap();
                entries.push(hierarchy::negative(pair[0].clone()));
                entries.extend(pair);
            }
            for entry in indexer::block_entries(g) {
                let pair = hierarchy::compact_specs(&entry).unwrap();
                entries.push(hierarchy::negative(pair[0].clone()));
                entries.extend(pair);
            }
            (
                "position_hierarchical",
                hierarchy::cubin_spec(
                    format!("position_hierarchical_{}_sm120", g.kernel_tag),
                    entries,
                ),
            )
        })
        .collect::<Vec<_>>();
    let specs = families
        .iter()
        .map(|(_, spec)| spec.clone())
        .collect::<Vec<_>>();
    let dir = root.join("../lab/kernels.lock.d");
    let mut options = BuildOptions::from_env();
    options.emit_cargo_directives = false;
    if options.profile_line_info {
        return Err("lock verification requires line info off".into());
    }
    if options.lock_update
        && dir
            .join(format!("{}.lock", "position_hierarchical"))
            .exists()
    {
        return Err(
            "build_position_hierarchical only initializes a new family; it cannot replace one"
                .into(),
        );
    }
    let toolchain = Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA is required for the cubin build".into());
    }
    let lock =
        LockDir::new(dir).family("position_hierarchical", specs.iter().map(|spec| &spec.name));
    let built = kernel_build::build_in_families(&toolchain, &specs, &out, &lock, options, None)?;
    assert_eq!(built.len(), specs.len());
    if !options.lock_update {
        kernel_build::report::BuildReport::read(
            &out.join(kernel_build::report::BUILD_REPORT_FILE),
        )?
        .verify_current()?;
    }
    println!("verified {} native MMA position cubins", built.len());
    Ok(())
}
