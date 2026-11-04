//! Native module fixtures for package tests: a cdylib compiled from a
//! model's fixture source, which cannot execute a model.
//!
//! Test support source, outside model-compiler-sdk's module tree: a model's
//! tests include it by `#[path]` and supply its dependencies, so no build
//! of the SDK, a host, a bundle or a package reads it. Each including test
//! uses part of it.
#![allow(dead_code)]

use std::ffi::OsStr;
use std::fs::File;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

/// The module `file` under `CHOPINFER_PACKAGE_NATIVE_ROOT`, when it is set.
pub fn native_root_module(file: &str) -> Option<PathBuf> {
    std::env::var_os("CHOPINFER_PACKAGE_NATIVE_ROOT").map(|root| PathBuf::from(root).join(file))
}

/// `source` compiled as the cdylib `file` (crate `crate_name`) in a new
/// directory `{label}-{process}-{nanoseconds}` under `CHOPINFER_SCRATCH`, or
/// the temporary directory, with `RUSTC` or `rustc`. The directory is the
/// caller's to change and remove. Panics if it fails.
pub fn compile_native_fixture(label: &str, crate_name: &str, file: &str, source: &str) -> PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = scratch().join(format!("{label}-{}-{stamp}", std::process::id()));
    std::fs::create_dir(&root).expect("external fixture directory");
    compile(&root, crate_name, file, source).unwrap_or_else(|error| panic!("{error}"))
}

/// `source` compiled once as the cdylib `file` (crate `crate_name`) into
/// `{label}-{key}` under [`fixture_cache`], `key` a hash of the three; a later
/// call, in this process or another, returns the same module. The module is
/// shared: read or copy it, never change it or write beside it. Panics if
/// compilation fails.
pub fn cached_native_fixture(label: &str, crate_name: &str, file: &str, source: &str) -> PathBuf {
    let mut hasher = DefaultHasher::new();
    (crate_name, file, source).hash(&mut hasher);
    let name = format!("{label}-{:016x}", hasher.finish());
    let cache = fixture_cache();
    std::fs::create_dir_all(&cache).expect("native fixture cache");
    let directory = cache.join(&name);
    let module = directory.join(file);
    // One process builds while the others wait, then they reuse its module.
    let lock = File::create(cache.join(format!("{name}.lock"))).expect("native fixture lock");
    lock.lock().expect("lock the native fixture");
    if std::fs::symlink_metadata(&module).is_ok_and(|metadata| metadata.is_file()) {
        return module;
    }
    let staging = cache.join(format!(".{name}.building"));
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir(&staging).expect("native fixture staging directory");
    if let Err(error) = compile(&staging, crate_name, file, source) {
        let _ = std::fs::remove_dir_all(&staging);
        panic!("{error}");
    }
    // The module appears with its directory, never half written.
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::rename(&staging, &directory).expect("publish the native fixture");
    module
}

/// `chopinfer-native-fixtures` in the Cargo profile directory of the running
/// test (the parent of its `deps`), else under `CHOPINFER_SCRATCH` or the
/// temporary directory.
pub fn fixture_cache() -> PathBuf {
    let executable = std::env::current_exe().ok();
    let profile = executable
        .as_deref()
        .and_then(Path::parent)
        .filter(|deps| deps.file_name() == Some(OsStr::new("deps")))
        .and_then(Path::parent);
    profile
        .map_or_else(scratch, Path::to_path_buf)
        .join("chopinfer-native-fixtures")
}

fn scratch() -> PathBuf {
    std::env::var_os("CHOPINFER_SCRATCH").map_or_else(std::env::temp_dir, PathBuf::from)
}

/// `source` compiled as the cdylib `file` (crate `crate_name`) in `root`.
fn compile(root: &Path, crate_name: &str, file: &str, source: &str) -> Result<PathBuf, String> {
    let path = root.join("fixture.rs");
    std::fs::write(&path, source)
        .map_err(|error| format!("write native fixture source: {error}"))?;
    let module = root.join(file);
    let output = Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()))
        .args(["--edition=2024", "--crate-type=cdylib"])
        .arg(format!("--crate-name={crate_name}"))
        .arg(&path)
        .arg("-o")
        .arg(&module)
        .output()
        .map_err(|error| format!("start Rust compiler for native fixture: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "native fixture compilation failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(module)
}
