//! The bundle's build script, which compiles and locks the example's SM120
//! cubins, so it lives in the sm120 module (`build` in Cargo.toml).
#[cfg(feature = "cuda-probe")]
use example_contract::ops;
#[cfg(feature = "precision-exercise")]
use example_contract::precision as precision_contract;
// The build compiles the selections; the bundle prepares ops from them.
#[cfg(feature = "precision-exercise")]
#[allow(dead_code)]
mod precision_selection;
#[cfg(feature = "cuda-probe")]
#[allow(dead_code)]
mod selection;

fn main() {
    #[cfg(feature = "cuda-probe")]
    probe();
    #[cfg(feature = "precision-exercise")]
    precision();
}

#[cfg(feature = "cuda-probe")]
fn probe() {
    use kernel_build::{ModelLock, Toolchain};
    use std::{env, path::PathBuf};
    let bundle = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let selection =
        selection::selection(bundle.parent().unwrap(), &kernels_sm120::cuda_source_root())
            .expect("external kernel selection");
    // Verified in place; only `tools.model_sdk relock` rewrites it.
    let lock = ModelLock::from_env(bundle.join("kernels.lock")).expect("model kernel lock");
    selection
        .build(
            &Toolchain::detect().expect("CUDA 13.2 toolchain"),
            &PathBuf::from(env::var_os("OUT_DIR").unwrap()),
            lock.path(),
            lock.options(),
        )
        .expect("locked external kernel selection");
}

#[cfg(feature = "precision-exercise")]
fn precision() {
    use kernel_build::{ModelLock, Toolchain};
    use std::{env, path::PathBuf};
    let bundle = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let selection = precision_selection::selection(
        bundle.parent().unwrap(),
        &kernels_sm120::cuda_source_root(),
    )
    .expect("external precision selection");
    let lock = ModelLock::from_env(bundle.join("precision.lock")).expect("model precision lock");
    let built = selection
        .build(
            &Toolchain::detect().expect("CUDA 13.2 toolchain"),
            &out.join("precision"),
            lock.path(),
            lock.options(),
        )
        .expect("locked precision selection");
    kernel_build::write_embedded_with_prefix(
        &out,
        std::path::Path::new("precision"),
        &built,
        |_| 1,
    )
    .expect("embed precision cubins");
    let publication = kernels_sm120::precision::publish_capabilities(
        &precision_selection::offers().unwrap(),
        &selection,
        &built,
    )
    .expect("audited portable capabilities");
    std::fs::write(out.join("precision-capabilities.json"), &publication.bytes).unwrap();
    std::fs::write(
        out.join("precision-dependency.txt"),
        publication.binding_dependency,
    )
    .unwrap();
    let document = serde_json::json!({ "version": 1, "profiles": [{
        "profile": "precision-tp1", "tensor_parallel_size": 1,
        "weight_contract": kernel_build::digest(precision_contract::WEIGHT_CONTRACT),
        "package_contract": kernel_build::digest(precision_contract::PACKAGE_CONTRACT),
    }], "implementation_contract": {
        "selection": publication.selection, "dependencies": publication.dependencies,
        "capabilities": publication.sha256,
    }});
    let description = serde_json::to_vec(&document).unwrap();
    let note = runtime_bundle_sdk::native_note::encode_native_note(&description);
    std::fs::write(out.join("precision-note.bin"), note).unwrap();
}
