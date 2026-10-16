#![forbid(unsafe_code)]

//! The host program's own commands, run as a program.
//!
//! Everything here goes through the built binary, because the surface being
//! contracted is the one an operator types: which bundle a command acts on,
//! what it refuses, and what it prints when it works.
//!
//! The bundle these tests act on is invented here. Its identifier, roles, and
//! layouts mean nothing to the host, and no model or bundle crate is linked
//! into either this test or the program it runs.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

use runtime_bundle::{
    BundleComponentV1, BundleGraphBucketV1, BundleKvLayoutV1, BundleManifestV1, BundleOpaqueId,
    BundleRankPlacementV1, BundleRelativePathV1, BundleResourceRequirementsV1, BundleStateLayoutV1,
    BundleStateLifetimeV1, BundleTargetV1, BundleTopologyV1,
};
use runtime_weights::sha256_bytes;

const BUNDLE_ID: &str = "fixture.command.v1";
const ARCHITECTURE: &str = "sm_120f";
const COMPONENT_PATH: &str = "components/weights.bin";
const COMPONENT_BYTES: &[u8] = b"fixture-weight-bytes";

/// One verifiable bundle directory with no native module.
///
/// A module is what a serving process loads, and leaving it out is what makes
/// the difference between verifying and loading observable: the same directory
/// verifies, and refuses to be served.
struct FixtureV1 {
    root: TemporaryDirectoryV1,
}

impl FixtureV1 {
    fn new() -> Self {
        let root = TemporaryDirectoryV1::new("runtime-host-command");
        let package = root.path().join(BUNDLE_ID);
        fs::create_dir_all(package.join("components"))
            .expect("the fixture component directory is creatable");
        fs::write(package.join(COMPONENT_PATH), COMPONENT_BYTES)
            .expect("the fixture component is writable");
        fs::write(
            package.join("manifest.json"),
            manifest()
                .to_canonical_json()
                .expect("the fixture manifest serializes"),
        )
        .expect("the fixture manifest is writable");
        Self { root }
    }

    fn package(&self) -> PathBuf {
        self.root.path().join(BUNDLE_ID)
    }
}

fn id(value: &str) -> BundleOpaqueId {
    BundleOpaqueId::new(value).expect("the fixture identifier is valid")
}

fn manifest() -> BundleManifestV1 {
    BundleManifestV1::new(
        id(BUNDLE_ID),
        BundleTargetV1::new(
            vec![id(ARCHITECTURE)],
            vec![BundleTopologyV1::new(id("tp1"), 1).expect("tp1 is valid")],
        )
        .expect("the fixture target is valid"),
        vec![
            BundleComponentV1::new(
                id("weights"),
                id("fixture.weights.v1"),
                id("fixture.bytes.v1"),
                BundleRelativePathV1::new(COMPONENT_PATH).expect("the fixture path is relative"),
                COMPONENT_BYTES.len() as u64,
                sha256_bytes(COMPONENT_BYTES),
            )
            .expect("the fixture component is valid"),
        ],
        BundleResourceRequirementsV1::new(1, Vec::new())
            .expect("the fixture requirements are valid"),
        BundleKvLayoutV1::new(
            id("fixture.kv.v1"),
            64,
            64,
            1,
            64,
            1,
            1024,
            BundleRankPlacementV1::Replicated,
            BundleStateLifetimeV1::Transaction,
        )
        .expect("the fixture KV layout is valid"),
        BundleStateLayoutV1::new(
            id("fixture.state.v1"),
            64,
            64,
            BundleStateLifetimeV1::Request,
        )
        .expect("the fixture state layout is valid"),
        vec![
            BundleGraphBucketV1::new(id("bucket.one"), vec![id("weights")], Vec::new())
                .expect("the fixture bucket is valid"),
        ],
        Vec::new(),
        Vec::new(),
        None,
    )
    .expect("the fixture manifest is valid")
}

/// All command names exercise the shared implementation.
const PROGRAMS: [(&str, &str); 2] = [
    ("chopinfer", env!("CARGO_BIN_EXE_chopinfer")),
    ("chopinfer-host", env!("CARGO_BIN_EXE_chopinfer-host")),
];

fn run(program: &str, arguments: &[&str]) -> Output {
    Command::new(program)
        .args(arguments)
        .output()
        .expect("the program starts")
}

/// Runs one invocation through the operator's program.
fn host(arguments: &[&str]) -> Output {
    run(env!("CARGO_BIN_EXE_chopinfer"), arguments)
}

fn succeeds(arguments: &[&str]) -> String {
    let output = host(arguments);
    assert!(
        output.status.success(),
        "{arguments:?} failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("the receipt is UTF-8")
}

fn refuses(arguments: &[&str]) -> String {
    let output = host(arguments);
    assert!(
        !output.status.success(),
        "{arguments:?} must be refused, it printed:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    String::from_utf8(output.stderr).expect("the refusal is UTF-8")
}

fn text(path: &Path) -> String {
    path.to_str().expect("the fixture path is UTF-8").to_owned()
}

/// A directory is verified where it stands; nothing installs it.
#[test]
fn a_bundle_is_verified_where_the_caller_named_it() {
    let fixture = FixtureV1::new();
    let package = text(&fixture.package());

    let verified = succeeds(&["bundle", "verify", "--bundle", &package]);
    assert!(verified.contains(BUNDLE_ID), "{verified}");
    assert!(verified.contains("1 components"), "{verified}");
    assert!(verified.contains("no native module"), "{verified}");

    // Installation and selection by identifier are gone, and say what
    // replaced them.
    let install = refuses(&["bundle", "install", "--source", &package, "--root", "/r"]);
    assert!(
        install.contains("bundle install no longer exists"),
        "{install}"
    );
    let installed = refuses(&[
        "bundle",
        "verify",
        "--install-root",
        "/r",
        "--bundle",
        BUNDLE_ID,
    ]);
    assert!(installed.contains("--install-root is gone"), "{installed}");
}

/// Verification checks identity, so drifted bytes are refused as they are read.
#[test]
fn a_component_that_drifted_after_it_was_written_is_refused() {
    let fixture = FixtureV1::new();
    let package = text(&fixture.package());
    let mut drifted = COMPONENT_BYTES.to_vec();
    drifted[0] = drifted[0].wrapping_add(1);
    fs::write(fixture.package().join(COMPONENT_PATH), &drifted)
        .expect("the fixture component is writable");

    let verify = refuses(&["bundle", "verify", "--bundle", &package]);

    assert!(verify.contains("SHA-256"), "{verify}");
}

/// The architecture a bundle is checked against is the caller's to state.
#[test]
fn a_bundle_for_another_architecture_is_refused_rather_than_adapted() {
    let fixture = FixtureV1::new();
    let package = text(&fixture.package());

    let refusal = refuses(&[
        "bundle",
        "verify",
        "--bundle",
        &package,
        "--architecture",
        "sm_90a",
    ]);

    assert!(refusal.contains("host architecture sm_90a"), "{refusal}");
}

/// Only a running command reads what running needs, and says what is missing.
///
/// This bundle verifies: it declares components, and every one of
/// them is what its manifest says it is. What it does not declare is a module
/// to load or a frontend to serve, and each running command refuses for its
/// own reason. That is the difference between checking a bundle and running
/// one.
#[test]
fn only_a_running_command_reads_what_running_needs() {
    let fixture = FixtureV1::new();
    let package = text(&fixture.package());

    succeeds(&["bundle", "verify", "--bundle", &package]);

    let conformance = refuses(&[
        "conformance-run",
        "--bundle",
        &package,
        "--fixture-request",
        "external-bundle-host",
        "--fake-host-services",
    ]);
    assert!(
        conformance.contains("no native module to load"),
        "{conformance}"
    );

    // It declares no frontend export, which a host refuses before it reads
    // any component of a frontend it could not serve.
    let serving = refuses(&["serve", "--bundle", &package, "--fake-host-services"]);
    assert!(
        serving.contains("cannot be served")
            && serving.contains("the package does not require chop.frontend.v1"),
        "{serving}"
    );
    // An explicit KV cache dtype the package does not declare is refused at
    // startup, before anything else is read.
    let fp8 = refuses(&[
        "serve",
        "--bundle",
        &package,
        "--kv-cache-dtype",
        "fp8",
        "--fake-host-services",
    ]);
    assert!(
        fp8.contains("--kv-cache-dtype fp8")
            && fp8.contains("declares no fp8 KV layout; it supports bf16"),
        "{fp8}"
    );
}

/// Each command names what it will do, and refuses an invocation that does not.
#[test]
fn every_command_refuses_an_invocation_it_cannot_run() {
    let fixture = FixtureV1::new();
    let package = text(&fixture.package());

    for (words, expected) in [
        (
            vec!["conformance-run", "--bundle", &package],
            "conformance-run requires --fixture-request",
        ),
        (
            vec!["serve", "--listen", "127.0.0.1:0"],
            "serve requires --bundle",
        ),
        (
            vec!["serve", "--bundle", &package, "--replicas", "2"],
            "unsupported argument '--replicas'",
        ),
        (
            vec!["serve", "--bundle", &package, "--listen", "8310"],
            "--listen takes an address and port",
        ),
    ] {
        let refusal = refuses(&words);
        assert!(refusal.contains(expected), "{refusal}");
    }
}

/// The operator's program exposes every command, by name.
///
/// The conformance host runs the same four from the same implementation, and
/// differs only in reading a bare flag list as its own conformance run.
#[test]
fn both_programs_expose_the_same_commands() {
    let fixture = FixtureV1::new();
    let package = text(&fixture.package());

    for (name, program) in PROGRAMS {
        let usage = String::from_utf8(run(program, &[]).stderr).expect("the usage is UTF-8");
        for command in [
            "bundle verify",
            "serve --bundle",
            "conformance-run --bundle",
        ] {
            assert!(usage.contains(command), "{name} does not offer {command}");
        }

        // Naming a command reaches it in both programs.
        let verified = run(program, &["bundle", "verify", "--bundle", &package]);
        assert!(verified.status.success(), "{name} cannot verify a bundle");
    }

    // A bare flag list is a command only for the conformance host.
    let bare = [
        "--bundle",
        package.as_str(),
        "--fixture-request",
        "external-bundle-host",
        "--fake-host-services",
    ];
    let operator = String::from_utf8(run(PROGRAMS[0].1, &bare).stderr).expect("stderr is UTF-8");
    let conformance = String::from_utf8(run(PROGRAMS[1].1, &bare).stderr).expect("stderr is UTF-8");
    assert!(
        operator.contains("unsupported command '--bundle'"),
        "{operator}"
    );
    assert!(conformance.contains("native module"), "{conformance}");
}

/// A scratch directory this test owns and removes.
struct TemporaryDirectoryV1 {
    path: PathBuf,
}

impl TemporaryDirectoryV1 {
    fn new(prefix: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "{prefix}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("the scratch directory is creatable");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TemporaryDirectoryV1 {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
