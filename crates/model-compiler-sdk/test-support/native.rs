//! Native module fixtures for package tests: a cdylib compiled from a
//! model's fixture source, which cannot execute a model.
//!
//! Test support source, outside model-compiler-sdk's module tree: a model's
//! tests include it by `#[path]` and supply its dependencies, so no build
//! of the SDK, a host, a bundle or a package reads it. Each including test
//! uses part of it.
#![allow(dead_code)]

use std::path::PathBuf;
use std::process::Command;

/// The module `file` under `CHOPINFER_PACKAGE_NATIVE_ROOT`, when it is set.
pub fn native_root_module(file: &str) -> Option<PathBuf> {
    std::env::var_os("CHOPINFER_PACKAGE_NATIVE_ROOT").map(|root| PathBuf::from(root).join(file))
}

/// `source` compiled as the cdylib `file` (crate `crate_name`) in a new
/// directory `{label}-{process}-{nanoseconds}` under `CHOPINFER_SCRATCH`, or
/// the temporary directory, with `RUSTC` or `rustc`. Panics if it fails.
pub fn compile_native_fixture(label: &str, crate_name: &str, file: &str, source: &str) -> PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let base = std::env::var_os("CHOPINFER_SCRATCH").map_or_else(std::env::temp_dir, PathBuf::from);
    let root = base.join(format!("{label}-{}-{stamp}", std::process::id()));
    std::fs::create_dir(&root).expect("external fixture directory");
    let path = root.join("fixture.rs");
    std::fs::write(&path, source).expect("write native fixture source");
    let module = root.join(file);
    let output = Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()))
        .args(["--edition=2024", "--crate-type=cdylib"])
        .arg(format!("--crate-name={crate_name}"))
        .arg(&path)
        .arg("-o")
        .arg(&module)
        .output()
        .expect("start Rust compiler for native fixture");
    assert!(
        output.status.success(),
        "native fixture compilation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    module
}
