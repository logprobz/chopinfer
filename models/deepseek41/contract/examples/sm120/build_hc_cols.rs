//! Initialize the column-block HC pair family before the feature-gated bundle build.
#[path = "../../../bundle/src/sm120/hc_cols.rs"]
#[allow(dead_code)]
mod hc_cols;
#[path = "../../../bundle/src/sm120/hc_completion.rs"]
#[allow(dead_code)]
mod hc_completion;
#[path = "../../../bundle/src/sm120/hc_expand_mix.rs"]
#[allow(dead_code)]
mod hc_expand_mix;
#[path = "../../../bundle/src/sm120/hc_finish_norm.rs"]
#[allow(dead_code)]
mod hc_finish_norm;
#[path = "../../../bundle/src/sm120/hc_finish_twin.rs"]
#[allow(dead_code)]
mod hc_finish_twin;
#[path = "../../../bundle/src/sm120/hc_round_chain.rs"]
#[allow(dead_code)]
mod hc_round_chain;
#[path = "../../../bundle/src/sm120/hc_round_expand_mix.rs"]
#[allow(dead_code)]
mod hc_round_expand_mix;
#[path = "../../../bundle/src/sm120/hc_round_rows.rs"]
#[allow(dead_code)]
mod hc_round_rows;
#[path = "../../../bundle/src/sm120/norm_quantize.rs"]
#[allow(dead_code)]
mod norm_quantize;
#[path = "../../../bundle/src/sm120/norm_quantize_cached.rs"]
#[allow(dead_code)]
mod norm_quantize_cached;
use kernel_build::{BuildOptions, LockDir, Toolchain};
use std::path::PathBuf;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or("expected output directory")?,
    );
    std::fs::create_dir_all(&out)?;
    let out = out.canonicalize()?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if out.starts_with(root.join("../../..").canonicalize()?) {
        return Err("outputs must stay outside repository".into());
    }
    let dir = root.join("../bundle/kernels.lock.d");
    let initialize = std::env::args().any(|a| a == "--initialize");
    if initialize && dir.join(format!("{}.lock", hc_cols::FAMILY)).exists() {
        return Err("refuse existing lock replacement".into());
    }
    let specs = hc_cols::cubins()?;
    let tc = Toolchain::detect()?;
    if tc.skipped {
        return Err("CUDA compiler required".into());
    }
    kernel_build::build_in_families(
        &tc,
        &specs,
        &out,
        &LockDir::new(dir).family(hc_cols::FAMILY, specs.iter().map(|s| s.name.as_str())),
        BuildOptions {
            lock_update: initialize,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
        None,
    )?;
    println!(
        "HC_COLS_BUILD modules={} entries={} PASS",
        specs.len(),
        specs.iter().map(|s| s.entries.len()).sum::<usize>()
    );
    Ok(())
}
