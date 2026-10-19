//! Build the separately locked parallel position query producers.
#[path = "../../../bundle/src/sm120/indexer_query.rs"]
#[allow(dead_code)]
mod indexer_query;
#[path = "../../../bundle/src/sm120/indexer_query_parallel.rs"]
#[allow(dead_code)]
mod registration;
use kernel_build::{BuildOptions, LockDir, Toolchain};
use model_deepseek41_contract::geometry::GEOMETRIES;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: build_position_query_parallel OUT_DIR")?;
    std::fs::create_dir_all(&out)?;
    let out = std::fs::canonicalize(out)?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repository = std::fs::canonicalize(root.join("../../.."))?;
    if out.starts_with(&repository) {
        return Err("build outputs must stay outside the repository".into());
    }
    let families = registration::parallel_cubins(GEOMETRIES)?
        .into_iter()
        .collect::<Vec<_>>();
    let specs = families
        .iter()
        .map(|(_, spec)| spec.clone())
        .collect::<Vec<_>>();
    let dir = root.join("../bundle/kernels.lock.d");
    let mut options = BuildOptions::from_env();
    options.emit_cargo_directives = false;
    if options.profile_line_info {
        return Err("lock verification requires line info off".into());
    }
    if options.lock_update
        && dir
            .join(format!("{}.lock", registration::PARALLEL_FAMILY))
            .exists()
    {
        return Err(
            "build_position_query_parallel only initializes a new family; it cannot replace one"
                .into(),
        );
    }
    let toolchain = Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA is required for the cubin build".into());
    }
    let lock = LockDir::new(dir).family(
        registration::PARALLEL_FAMILY,
        specs.iter().map(|spec| &spec.name),
    );
    let built = kernel_build::build_in_families(&toolchain, &specs, &out, &lock, options, None)?;
    assert_eq!(built.len(), specs.len());
    if !options.lock_update {
        kernel_build::report::BuildReport::read(
            &out.join(kernel_build::report::BUILD_REPORT_FILE),
        )?
        .verify_current()?;
    }
    println!("verified {} parallel position query cubins", built.len());
    Ok(())
}

#[test]
fn cpu_model_query_candidate_keeps_prefill_and_serving_registration_separate() {
    use kernel_core::backend::{Backend, Need};
    use model_deepseek41_contract::{indexer, serving::PREFILL_TOKENS};
    for g in GEOMETRIES {
        let backend = registration::register_parallel(kernels_sm120::backend::Sm120::new(), g);
        let query = indexer::selection(g, g.shape.csa.index_sources[0], PREFILL_TOKENS).unwrap();
        for rows in [1, 2, 4, 6, 8, 16, 32, PREFILL_TOKENS] {
            let op = backend
                .position_indexer_query(&query, Need::rows(rows))
                .unwrap();
            assert_eq!(op.info().symbols.len(), if rows <= 32 { 1 } else { 3 });
            assert!(
                op.info()
                    .symbols
                    .iter()
                    .any(|(name, _)| name.ends_with("parallel_v1"))
            );
        }
    }
}
