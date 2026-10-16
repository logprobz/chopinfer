#![forbid(unsafe_code)]

//! Reading one component of the selected bundle by the role it serves.
//!
//! A deployment that wants a frontend declaration, or anything else a bundle
//! ships beside its weights, asks for it by role and gets bytes that were
//! authenticated against the manifest that named them. The host does not know
//! what any role means, so this path is the same for every one of them.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use runtime_bundle::{
    BundleComponentV1, BundleGraphBucketV1, BundleHostCompatibilityV1, BundleKvLayoutV1,
    BundleManifestV1, BundleOpaqueId, BundlePackageV1, BundleRankPlacementV1, BundleRelativePathV1,
    BundleResourceRequirementsV1, BundleStateLayoutV1, BundleStateLifetimeV1, BundleTargetV1,
    BundleTopologyV1,
};
use runtime_host::{HostLoadErrorV1, verified_component_by_role_v1};
use runtime_weights::sha256_bytes;

const ARCHITECTURE: &str = "sm_120f";
const FRONTEND_ROLE: &str = "chop.frontend.openai.v1";
const FRONTEND_BYTES: &[u8] = br#"{"schema":"chop-frontend-openai-v2","version":2}"#;
const WEIGHT_ROLE: &str = "fixture.weights.v1";
const WEIGHT_BYTES: &[u8] = b"fixture-weight-bytes";

fn host() -> BundleHostCompatibilityV1 {
    BundleHostCompatibilityV1::new(ARCHITECTURE, 1).expect("the fixture host descriptor is valid")
}

fn id(value: &str) -> BundleOpaqueId {
    BundleOpaqueId::new(value).expect("the fixture identifier is valid")
}

fn component(name: &str, role: &str, path: &str, bytes: &[u8]) -> BundleComponentV1 {
    BundleComponentV1::new(
        id(name),
        id(role),
        id("fixture.bytes.v1"),
        BundleRelativePathV1::new(path).expect("the fixture path is relative"),
        bytes.len() as u64,
        sha256_bytes(bytes),
    )
    .expect("the fixture component is valid")
}

/// One package with two components under different roles, and no native
/// module: reading a declaration never needs the module to be loadable.
struct FixtureV1 {
    root: TemporaryDirectoryV1,
}

impl FixtureV1 {
    fn new() -> Self {
        let root = TemporaryDirectoryV1::new("runtime-host-component-role");
        let package = root.path().join("bundle");
        fs::create_dir_all(package.join("components"))
            .expect("the fixture component directory is creatable");
        fs::write(package.join("components/frontend.json"), FRONTEND_BYTES)
            .expect("the fixture frontend component is writable");
        fs::write(package.join("components/weights.bin"), WEIGHT_BYTES)
            .expect("the fixture weight component is writable");
        let manifest = BundleManifestV1::new(
            id("fixture.roles.v1"),
            BundleTargetV1::new(
                vec![id(ARCHITECTURE)],
                vec![BundleTopologyV1::new(id("tp1"), 1).expect("tp1 is valid")],
            )
            .expect("the fixture target is valid"),
            vec![
                component(
                    "frontend",
                    FRONTEND_ROLE,
                    "components/frontend.json",
                    FRONTEND_BYTES,
                ),
                component(
                    "weights",
                    WEIGHT_ROLE,
                    "components/weights.bin",
                    WEIGHT_BYTES,
                ),
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
                BundleGraphBucketV1::new(
                    id("decode-1"),
                    vec![id("frontend"), id("weights")],
                    Vec::new(),
                )
                .expect("the fixture bucket is valid"),
            ],
            Vec::new(),
            Vec::new(),
            None,
        )
        .expect("the fixture manifest is valid");
        fs::write(
            package.join("manifest.json"),
            manifest
                .to_canonical_json()
                .expect("the fixture manifest serializes"),
        )
        .expect("the fixture manifest is writable");
        Self { root }
    }

    fn package(&self) -> PathBuf {
        self.root.path().join("bundle")
    }

    fn verified(&self) -> BundlePackageV1 {
        BundlePackageV1::verify_directory(self.package(), &host())
            .expect("the fixture package verifies")
    }
}

/// A role names one component, and its bytes are the authenticated ones.
#[test]
fn a_component_is_read_by_the_role_it_serves() {
    let fixture = FixtureV1::new();
    let package = fixture.verified();

    assert_eq!(
        verified_component_by_role_v1(&package, FRONTEND_ROLE).expect("the frontend role resolves"),
        FRONTEND_BYTES
    );
    assert_eq!(
        verified_component_by_role_v1(&package, WEIGHT_ROLE).expect("the weight role resolves"),
        WEIGHT_BYTES
    );
}

/// A role this bundle never declared is refused rather than guessed at.
#[test]
fn an_undeclared_role_is_refused() {
    let fixture = FixtureV1::new();
    let package = fixture.verified();

    let error = verified_component_by_role_v1(&package, "chop.frontend.absent.v1")
        .expect_err("an undeclared role resolves to nothing");
    assert!(matches!(error, HostLoadErrorV1::UnknownComponentRole(role)
        if role == "chop.frontend.absent.v1"));
}

/// The bytes served for a role are the verified bytes: a component rewritten
/// after verification reads back as it was verified, and only a new
/// verification sees the rewrite, which it refuses.
#[test]
fn bytes_served_are_the_verified_bytes() {
    let fixture = FixtureV1::new();
    let package = fixture.verified();

    // The same length with different bytes passes every check a path alone
    // can make, so only the verified copy keeps it from being served.
    let mut swapped = FRONTEND_BYTES.to_vec();
    swapped[2] = swapped[2].wrapping_add(1);
    fs::write(fixture.package().join("components/frontend.json"), &swapped)
        .expect("the fixture component is writable");

    assert_eq!(
        verified_component_by_role_v1(&package, FRONTEND_ROLE)
            .expect("the verified bytes stay readable"),
        FRONTEND_BYTES
    );
    assert!(
        BundlePackageV1::verify_directory(fixture.package(), &host()).is_err(),
        "a new verification must refuse the rewritten component"
    );
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
