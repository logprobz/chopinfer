#![forbid(unsafe_code)]

//! Emits this fixture's canonical bundle directory.
//!
//! The package a host installs has to contain the shared object the host will
//! open, and `cargo run --bin build-tp4-fixture` builds only this program and
//! the library it links. So this program builds the crate's `cdylib` itself,
//! with the same profile and feature selection it was built with, finds the
//! artifact beside its own executable, and hands that file to the emitter as the
//! native module. Nothing else about the package is discovered: every component,
//! path, layout, and requirement comes from this crate's own declaration.
//!
//! The program carries the fixture's name because every other bundle emits its
//! own package with a program of its own, and one target directory holds one
//! file per binary name.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use bundle_tp4_fixture::{
    TP4_FIXTURE_NAME_V1, emit_tp4_fixture_v1, write_tp4_fixture_checkpoint_v1,
};

const USAGE_V1: &str =
    "usage: build-tp4-fixture --fixture <NAME> --output DIR [--checkpoint NEW-DIR]";

/// The crate whose library is the native module of the emitted package.
const PACKAGE_V1: &str = "bundle-tp4-fixture";

fn main() -> ExitCode {
    match run_v1() {
        Ok(message) => {
            println!("{message}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("build-tp4-fixture: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run_v1() -> Result<String, String> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let request = parse_arguments_v1(&arguments)?;
    let module = build_native_module_v1()?;
    let emitted = emit_tp4_fixture_v1(&module, &request.output)
        .map_err(|error| format!("the fixture bundle was not emitted: {error}"))?;
    if let Some(checkpoint) = &request.checkpoint {
        write_tp4_fixture_checkpoint_v1(checkpoint).map_err(|error| {
            format!(
                "the synthetic checkpoint was not written to {}: {error}",
                checkpoint.display()
            )
        })?;
    }
    Ok(format!(
        "emitted bundle '{}' at {}: {} components, {} planned components, one native module from {}",
        emitted.manifest().id(),
        emitted.root().display(),
        emitted.manifest().components().len(),
        emitted.manifest().planned_components().len(),
        module.display(),
    ))
}

/// One parsed invocation.
struct RequestV1 {
    output: PathBuf,
    /// Where to write the synthetic checkpoint the package loads.
    checkpoint: Option<PathBuf>,
}

fn parse_arguments_v1(arguments: &[String]) -> Result<RequestV1, String> {
    let mut fixture: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut checkpoint: Option<PathBuf> = None;
    let mut remaining = arguments.iter();
    while let Some(flag) = remaining.next() {
        let value = remaining
            .next()
            .ok_or_else(|| format!("{flag} requires a value\n{USAGE_V1}"))?;
        match flag.as_str() {
            "--fixture" => set_once_v1(&mut fixture, value.clone(), flag)?,
            "--output" => set_once_v1(&mut output, PathBuf::from(value), flag)?,
            "--checkpoint" => set_once_v1(&mut checkpoint, PathBuf::from(value), flag)?,
            _ => return Err(format!("unsupported argument '{flag}'\n{USAGE_V1}")),
        }
    }
    let fixture = fixture.ok_or_else(|| format!("--fixture is required\n{USAGE_V1}"))?;
    if fixture != TP4_FIXTURE_NAME_V1 {
        return Err(format!(
            "this program emits only the '{TP4_FIXTURE_NAME_V1}' fixture, not '{fixture}'"
        ));
    }
    let output = output.ok_or_else(|| format!("--output is required\n{USAGE_V1}"))?;
    Ok(RequestV1 { output, checkpoint })
}

fn set_once_v1<T>(slot: &mut Option<T>, value: T, flag: &str) -> Result<(), String> {
    if slot.is_some() {
        return Err(format!("{flag} is given more than once\n{USAGE_V1}"));
    }
    *slot = Some(value);
    Ok(())
}

/// Builds this crate's shared object and returns the file that was produced.
///
/// The profile and the feature selection are taken from this program's own
/// build, so the library is compiled into the same directory this executable
/// sits in and no second unit graph is created for it.
fn build_native_module_v1() -> Result<PathBuf, String> {
    let artifacts = artifact_directory_v1()?;
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut command = Command::new(cargo);
    command.args([
        "build",
        "--locked",
        "--offline",
        "--manifest-path",
        manifest_path_v1(),
        "-p",
        PACKAGE_V1,
        "--lib",
        "--profile",
        profile_v1(&artifacts)?,
    ]);
    if cfg!(feature = "fake-host-services") {
        command.args(["--no-default-features", "--features", "fake-host-services"]);
    }
    let output = command
        .output()
        .map_err(|error| format!("cannot start cargo to build the native module: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "cargo could not build the native module:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let module = artifacts.join(native_module_file_name_v1());
    if !module.is_file() {
        return Err(format!(
            "cargo reported success but produced no native module at {}",
            module.display()
        ));
    }
    Ok(module)
}

fn manifest_path_v1() -> &'static str {
    concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml")
}

/// The directory cargo places this program's artifacts in.
fn artifact_directory_v1() -> Result<PathBuf, String> {
    let executable =
        std::env::current_exe().map_err(|error| format!("cannot locate this program: {error}"))?;
    executable
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| format!("{} has no artifact directory", executable.display()))
}

/// The cargo profile whose artifacts land in `artifacts`.
///
/// Cargo names the `dev` profile's directory `debug` and every other profile's
/// directory after the profile itself, so the directory this program was built
/// into is what says which profile to build the library with.
fn profile_v1(artifacts: &Path) -> Result<&'static str, String> {
    match artifacts.file_name().and_then(std::ffi::OsStr::to_str) {
        Some("debug") => Ok("dev"),
        Some("release") => Ok("release"),
        Some(other) => Err(format!(
            "this program was built into the '{other}' profile directory, which this program does \
             not know how to rebuild its library for"
        )),
        None => Err(format!(
            "{} is not a cargo profile directory",
            artifacts.display()
        )),
    }
}

fn native_module_file_name_v1() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "libbundle_tp4_fixture.dylib"
    }
    #[cfg(target_os = "linux")]
    {
        "libbundle_tp4_fixture.so"
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        "bundle_tp4_fixture.dll"
    }
}
