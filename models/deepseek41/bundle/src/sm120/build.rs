//! The bundle's build script, which compiles and locks the model's SM120
//! cubins (`kernels.lock.d/`), so it lives in the sm120 module (`build` in
//! Cargo.toml). Each engine adds its selection here as it registers.
use std::env;
use std::path::PathBuf;

use kernel_build::{LockDir, ModelLock, Toolchain};

mod own_cubin;
// The build compiles the entries; the bundle launches them.
#[allow(dead_code)]
mod own_entries;

fn main() {
    println!("cargo:rerun-if-changed=src/sm120/build.rs");
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("output directory"));
    // (family, cubin) pairs; one family per engine or own-op set.
    let specs = vec![(own_entries::FAMILY, own_cubin::cubin_spec(&manifest))];
    let mut families: Vec<(&str, Vec<&str>)> = Vec::new();
    for (family, spec) in &specs {
        match families.iter_mut().find(|(name, _)| name == family) {
            Some((_, cubins)) => cubins.push(&spec.name),
            None => families.push((family, vec![&spec.name])),
        }
    }
    // Model-owned: `tools.model_sdk relock` redirects it in an export.
    let model_lock =
        ModelLock::from_env(manifest.join("kernels.lock.d")).expect("deepseek41 kernel lock");
    let lock = families
        .into_iter()
        .fold(LockDir::new(model_lock.path()), |lock, (family, cubins)| {
            lock.family(family, cubins)
        });
    let specs: Vec<_> = specs.into_iter().map(|(_, spec)| spec).collect();
    let built = kernel_build::build_in_families(
        &Toolchain::detect().expect("CUDA 13.2 toolchain"),
        &specs,
        &out,
        &lock,
        model_lock.options(),
        None,
    )
    .expect("locked deepseek41 cubin build and audit");
    kernel_build::write_embedded(&out, &built, |_| 0).expect("write embedded cubin list");
}
