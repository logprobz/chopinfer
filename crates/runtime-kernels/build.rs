//! The host sampling cubin this crate owns, compiled on Linux only.
//!
//! Nothing here belongs to a model. Each installed bundle compiles, audits, and
//! stages the CUDA it owns, so this build names no model artifact, fixture, or
//! provider.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use kernel_build::{BuildOptions, Toolchain, build};

fn main() {
    println!("cargo:rerun-if-env-changed=CHOPINFER_CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CHOPINFER_NVCC_WRAPPER");
    println!("cargo:rerun-if-env-changed=RUNTIME_KERNELS_CUBIN_DIR");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH")
        .expect("Cargo did not provide the target architecture to runtime-kernels");
    assert_eq!(
        target_arch, "x86_64",
        "runtime-kernels supports only Linux x86_64 CUDA targets; got Linux {target_arch}"
    );

    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    // Each build of this crate stages its own cubins: a workspace build can
    // run this script twice at once (as a dependency and, through a build
    // dependency, for the host), and a shared directory lets one run delete
    // the other's stage.
    let cargo_out = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo OUT_DIR"));
    let unit = cargo_out
        .parent()
        .and_then(Path::file_name)
        .expect("Cargo build unit");
    // An explicit artifact root must also isolate host and target units.
    let artifact_dir = env::var_os("RUNTIME_KERNELS_CUBIN_DIR")
        .map(|root| PathBuf::from(root).join(unit))
        .unwrap_or_else(|| cargo_out.join("cubin"));
    fs::create_dir_all(&artifact_dir).expect("cannot create the owned kernel cubin directory");
    let toolchain = Toolchain::detect().expect("host kernels require CUDA 13.2");
    let sampling = build(
        &toolchain,
        &[kernels_sm120::sampling::cubin_spec()],
        &artifact_dir,
        &manifest.join("kernels.lock"),
        BuildOptions::from_env(),
    )
    .expect("compile and audit the host sampling cubin");
    let sampling_cubin = sampling
        .first()
        .expect("one sampling cubin is built on Linux");
    println!(
        "cargo:rustc-env=RUNTIME_SAMPLING_CUBIN={}",
        sampling_cubin.path.display()
    );
}
