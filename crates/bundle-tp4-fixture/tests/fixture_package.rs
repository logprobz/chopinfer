//! The package this fixture emits, as the canonical verifier reads it back.
//!
//! Every assertion here runs against a real emitted directory:
//! `build-tp4-fixture` compiles the native module, stages four components,
//! declares the planned program component, and publishes the closed package, and this reopens that package
//! through `runtime-bundle` rather than through anything this crate believes
//! about itself.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};

use bundle_tp4_fixture::{
    TP4_FIXTURE_ARCHITECTURE_V1, TP4_FIXTURE_BUNDLE_ID_V1, TP4_FIXTURE_CONTEXT_TOKENS_V1,
    TP4_FIXTURE_FRONTEND_ROLE_V1, TP4_FIXTURE_GRAPH_BUCKET_V1, TP4_FIXTURE_KV_LAYOUT_ID_V1,
    TP4_FIXTURE_MARKER_ROLE_V1, TP4_FIXTURE_MINIMUM_DEVICE_MEMORY_BYTES_V1, TP4_FIXTURE_NAME_V1,
    TP4_FIXTURE_PEAK_TOKEN_ID_V1, TP4_FIXTURE_PROGRAM_COMPONENT_V1, TP4_FIXTURE_RANK_COUNT_V1,
    TP4_FIXTURE_STATE_BYTES_V1, TP4_FIXTURE_STATE_LAYOUT_ID_V1,
    TP4_FIXTURE_TOKENIZER_CONFIGURATION_ROLE_V1, TP4_FIXTURE_TOKENIZER_DEFINITION_ROLE_V1,
    TP4_FIXTURE_TOPOLOGY_ID_V1, tp4_fixture_filler_token_v1, tp4_fixture_row_key_v1,
    tp4_fixture_tokenizer_configuration_v1, tp4_fixture_tokenizer_definition_v1,
    tp4_fixture_vocabulary_v1, tp4_published_markers_v1,
};
use runtime_bundle::{
    BUNDLE_NATIVE_ENTRY_SYMBOL_V1, BundleArtifactOriginV1, BundleHostCompatibilityV1,
    BundleOpaqueId, BundleOperationProvisioningV1, BundlePackageV1, BundleStateLifetimeV1,
    BundleTopologyV1, MODEL_BUNDLE_V1_ABI_VERSION, MODEL_BUNDLE_V1_SCHEMA,
};
use runtime_bundle_abi::BundleLibraryV1;
use runtime_host::{
    HostGenericCatalogV1, declaration_from_package_v1, deployment_from_package_v1,
    frontend_export_v1, resolve_declared_operations_v1,
};
use runtime_kernels::KernelImplementationTierV1;
use runtime_server::ServerConfig;
use serde_json::Value;

/// The host this fixture is emitted for.
fn host() -> BundleHostCompatibilityV1 {
    BundleHostCompatibilityV1::new(TP4_FIXTURE_ARCHITECTURE_V1, MODEL_BUNDLE_V1_ABI_VERSION)
        .expect("the test host descriptor is valid")
        .with_capabilities([
            BundleOpaqueId::new("chop.prompt-rows.v2").unwrap(),
            BundleOpaqueId::new("chop.request-retirement.v1").unwrap(),
            BundleOpaqueId::new("chop.speculative.v1").unwrap(),
            BundleOpaqueId::new("chop.media-inputs.v1").unwrap(),
            BundleOpaqueId::new("chop.prefix-cache.v1").unwrap(),
            BundleOpaqueId::new(runtime_bundle::WEIGHT_PLAN_CAPABILITY_V1).unwrap(),
            BundleOpaqueId::new(runtime_bundle_abi::frontend::FRONTEND_CAPABILITY_V1).unwrap(),
        ])
}

/// Emits the fixture package the acceptance harness emits, into `root`.
///
/// One emission at a time. `build-tp4-fixture` compiles this crate's shared
/// object before it stages anything, and two of them at once would have one
/// cargo replacing the artifact the other is measuring.
fn emit_package(root: &Path) -> BundlePackageV1 {
    static EMITTING: Mutex<()> = Mutex::new(());

    let output = root.join(TP4_FIXTURE_BUNDLE_ID_V1);
    let emitted = {
        let _one_at_a_time = EMITTING.lock().unwrap_or_else(PoisonError::into_inner);
        Command::new(env!("CARGO_BIN_EXE_build-tp4-fixture"))
            .args([
                "--fixture",
                TP4_FIXTURE_NAME_V1,
                "--output",
                output.to_str().expect("the output path is UTF-8"),
            ])
            .output()
            .expect("build-tp4-fixture starts")
    };
    assert!(
        emitted.status.success(),
        "build-tp4-fixture failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&emitted.stdout),
        String::from_utf8_lossy(&emitted.stderr)
    );
    BundlePackageV1::verify_directory(&output, &host())
        .expect("the emitted package verifies as a closed V1 package")
}

fn id(value: &str) -> BundleOpaqueId {
    BundleOpaqueId::new(value).expect("the test identifier is valid")
}

/// The emitted manifest is the declaration this fixture set out to make.
#[test]
fn the_emitted_package_declares_the_tp4_topology_with_a_paged_kv_layout() {
    let root = TestDirectory::new("tp4-fixture-package");
    let package = emit_package(root.path());
    let manifest = package.manifest();

    assert_eq!(manifest.schema(), MODEL_BUNDLE_V1_SCHEMA);
    assert_eq!(manifest.abi_version(), MODEL_BUNDLE_V1_ABI_VERSION);
    assert_eq!(manifest.id().as_str(), TP4_FIXTURE_BUNDLE_ID_V1);
    assert!(
        TP4_FIXTURE_BUNDLE_ID_V1.starts_with("fixture."),
        "the installed identifier has to say this is a fixture"
    );

    let [topology] = manifest.target().topologies() else {
        panic!("this fixture declares exactly one topology");
    };
    assert_eq!(topology.id().as_str(), TP4_FIXTURE_TOPOLOGY_ID_V1);
    assert_eq!(topology.id().as_str(), "tp4");
    assert_eq!(topology.rank_count(), TP4_FIXTURE_RANK_COUNT_V1);
    assert_eq!(topology.rank_count(), 4);
    assert_eq!(
        manifest.target().architectures(),
        [id(TP4_FIXTURE_ARCHITECTURE_V1)]
    );

    assert_eq!(
        manifest.kv_layout().id().as_str(),
        TP4_FIXTURE_KV_LAYOUT_ID_V1
    );
    assert_eq!(
        manifest.state_layout().id().as_str(),
        TP4_FIXTURE_STATE_LAYOUT_ID_V1
    );
    assert_eq!(manifest.state_layout().bytes(), TP4_FIXTURE_STATE_BYTES_V1);
    assert_eq!(
        manifest.state_layout().lifetime(),
        BundleStateLifetimeV1::Process
    );

    // The one bucket references every component and the one operation, which is
    // what makes them part of this bundle rather than declarations beside it.
    let [bucket] = manifest.graph_buckets() else {
        panic!("this fixture declares exactly one graph bucket");
    };
    assert_eq!(bucket.id().as_str(), TP4_FIXTURE_GRAPH_BUCKET_V1);
    assert_eq!(
        bucket.components().len(),
        manifest.components().len() + manifest.planned_components().len()
    );
    assert_eq!(manifest.components().len(), 4);
    assert!(
        bucket
            .operations()
            .contains(&tp4_fixture_row_key_v1().expect("the row key is valid"))
    );

    // The program image is planned: the weight plan loads its four rank
    // images from the synthetic checkpoint, and the package ships no image.
    let [program] = manifest.planned_components() else {
        panic!("this fixture plans exactly one component");
    };
    assert_eq!(program.id().as_str(), TP4_FIXTURE_PROGRAM_COMPONENT_V1);
    assert!(
        manifest
            .resource_requirements()
            .required_capabilities()
            .contains(&id(runtime_bundle::WEIGHT_PLAN_CAPABILITY_V1)),
        "a package with a planned component requires the weight plan"
    );
    assert!(
        manifest
            .resource_requirements()
            .required_capabilities()
            .contains(&id(runtime_bundle_abi::frontend::FRONTEND_CAPABILITY_V1)),
        "the package's module serves its chat frontend"
    );

    let native_module = manifest
        .native_module()
        .expect("this fixture ships a native module");
    assert_eq!(native_module.entry_symbol(), BUNDLE_NATIVE_ENTRY_SYMBOL_V1);
}

/// The one operation this package declares is private to it, and the real host
/// catalog resolves it at the bundle tier rather than offering to stand in.
#[test]
fn the_one_declared_operation_is_private_and_resolves_at_the_bundle_tier() {
    let root = TestDirectory::new("tp4-fixture-operations");
    let package = emit_package(root.path());
    let manifest = package.manifest();
    let key = tp4_fixture_row_key_v1().expect("the row key is valid");

    let [requirement] = manifest.operation_requirements() else {
        panic!("this fixture declares exactly one operation");
    };
    assert_eq!(requirement.key(), &key);
    assert_eq!(
        requirement.provisioning(),
        &BundleOperationProvisioningV1::BundleCustomOperation
    );

    let [implementation] = manifest.implementations() else {
        panic!("this fixture declares exactly one implementation");
    };
    assert_eq!(implementation.key(), &key);
    let artifact = implementation.detail().artifact();
    let native_module = manifest
        .native_module()
        .expect("this fixture ships a native module");
    assert_eq!(artifact.origin(), &BundleArtifactOriginV1::NativeModule);
    assert_eq!(artifact.byte_length(), native_module.byte_length());
    assert_eq!(artifact.sha256(), native_module.sha256());
    assert_eq!(
        implementation.detail().source().path().as_str(),
        "src.operation.rs"
    );

    // A host candidate carrying the same key never satisfies it: this operation
    // is private, and this package implements it.
    let keys = manifest.semantic_keys().expect("every contract has a key");
    let semantic = keys[&key].clone();
    let resolver = manifest
        .kernel_resolver([])
        .expect("the manifest's own implementation resolves");
    assert!(resolver.is_bundle_private(&semantic));
    assert_eq!(
        resolver
            .resolve(&semantic)
            .expect("the private operation resolves")
            .tier(),
        KernelImplementationTierV1::Bundle
    );

    // And the catalog a real host is compiled with resolves the whole package.
    let catalog = HostGenericCatalogV1::builtin_v1().expect("this host's catalog is valid");
    let resolved = resolve_declared_operations_v1(manifest, &catalog)
        .expect("a real host resolves everything this package declares");
    assert_eq!(resolved.len(), 1);
    assert_eq!(resolved[0].tier(), KernelImplementationTierV1::Bundle);
}

/// The verified package carries a readable declaration and a marker that denies
/// every claim this fixture must not make.
#[test]
fn the_verified_package_carries_a_readable_declaration_and_an_honest_marker() {
    let root = TestDirectory::new("tp4-fixture-components");
    let package = emit_package(root.path());

    // The declaration reads through the same function the serving command calls.
    let declaration = declaration_from_package_v1(&package)
        .expect("the emitted declaration reads through the shared parser");
    assert_eq!(declaration.vocabulary.size, tp4_fixture_vocabulary_v1());
    assert_eq!(declaration.vocabulary.size, 154_880);

    let component = |role: &str| {
        let component = package
            .manifest()
            .components()
            .iter()
            .find(|component| component.role().as_str() == role)
            .unwrap_or_else(|| panic!("this package declares a {role} component"));
        package
            .read_component(component)
            .unwrap_or_else(|error| panic!("the {role} component reads: {error}"))
    };

    // Both tokenizer roles are present, so a deployment finds exactly the bytes
    // the declaration pinned.
    assert_eq!(
        component(TP4_FIXTURE_TOKENIZER_DEFINITION_ROLE_V1),
        tp4_fixture_tokenizer_definition_v1(),
        "the staged definition must be the bytes the declaration pinned"
    );
    assert_eq!(
        component(TP4_FIXTURE_TOKENIZER_CONFIGURATION_ROLE_V1),
        tp4_fixture_tokenizer_configuration_v1(),
        "the staged configuration must be the bytes the declaration pinned"
    );
    assert!(!component(TP4_FIXTURE_FRONTEND_ROLE_V1).is_empty());

    let marker: Value = serde_json::from_slice(&component(TP4_FIXTURE_MARKER_ROLE_V1))
        .expect("the marker component is JSON");
    assert_eq!(marker["model_arithmetic"], false);
    assert_eq!(marker["nvfp4_qualified"], false);
    assert_eq!(marker["model_operations_implemented"], 0);
    assert_eq!(marker["authenticated_payload_execution"], false);
    assert_eq!(marker["logit_count"], 154_880);
    assert_eq!(marker["topology"], "tp4");
    assert_eq!(marker["rank_count"], 4);
}

/// The verified package builds a whole deployment.
///
/// This is the gate the served half depends on: the same functions the
/// serving command calls open the package's module and take its frontend
/// export, read the declaration, authenticate the two tokenizer components
/// against the identities that declaration pinned, build a tokenizer from
/// them, and bind every boundary identifier. A package whose artifacts did
/// not work would fail here rather than at a request.
#[test]
fn the_verified_package_builds_a_whole_deployment_with_a_working_tokenizer() {
    let root = TestDirectory::new("tp4-fixture-deployment");
    let package = emit_package(root.path());
    let module = package
        .open_native_module()
        .expect("the module reads")
        .expect("the package ships a module");
    let library = BundleLibraryV1::open_verified_image(module.bytes(), module.display())
        .expect("the module loads");
    let frontend =
        frontend_export_v1(package.manifest(), &library).expect("the module exports the frontend");
    drop(library);

    let deployment = deployment_from_package_v1(&package, &frontend, ServerConfig::default())
        .expect("the emitted package builds a deployment");

    assert_eq!(
        deployment.declaration().vocabulary.size,
        tp4_fixture_vocabulary_v1()
    );
    assert_eq!(
        deployment.frontend().maximum_total_tokens(),
        TP4_FIXTURE_CONTEXT_TOKENS_V1
    );

    // Every published TP4 marker decodes, through this package's tokenizer, to
    // the text the declaration pinned it to.
    let tokenizer = deployment.tokenizer();
    for (id, marker) in tp4_published_markers_v1() {
        assert_eq!(
            tokenizer.decode_bytes(&[id]).expect("a marker decodes"),
            marker.as_bytes()
        );
    }
    // And the identifier every emitted row peaks at decodes to visible text.
    assert_eq!(
        tokenizer
            .decode_utf8(&[TP4_FIXTURE_PEAK_TOKEN_ID_V1])
            .expect("the peak decodes"),
        tp4_fixture_filler_token_v1(TP4_FIXTURE_PEAK_TOKEN_ID_V1)
    );
}

/// A host this package does not target, or one too small for it, is refused.
#[test]
fn a_host_this_package_does_not_target_cannot_verify_it() {
    let root = TestDirectory::new("tp4-fixture-host");
    let package = emit_package(root.path());
    let directory = package.root().to_path_buf();

    for (host, reason) in [
        (
            BundleHostCompatibilityV1::new("sm_90a", MODEL_BUNDLE_V1_ABI_VERSION)
                .expect("the test host descriptor is valid"),
            "another architecture",
        ),
        (
            host().with_topology(
                BundleTopologyV1::new(id("tp1"), 1).expect("the test topology is valid"),
            ),
            "a topology this package does not declare",
        ),
        (
            host().with_device_memory_bytes(TP4_FIXTURE_MINIMUM_DEVICE_MEMORY_BYTES_V1 - 1),
            "less device memory than the package requires",
        ),
    ] {
        BundlePackageV1::verify_directory(&directory, &host)
            .expect_err(&format!("{reason} must be refused"));
    }
}

/// Tampering with one staged component is what the closed package refuses.
#[test]
fn a_component_the_manifest_did_not_measure_fails_verification() {
    let root = TestDirectory::new("tp4-fixture-tamper");
    let package = emit_package(root.path());
    let directory = package.root().to_path_buf();
    let marker = package
        .manifest()
        .components()
        .iter()
        .find(|component| component.role().as_str() == TP4_FIXTURE_MARKER_ROLE_V1)
        .expect("this package carries a marker component")
        .path()
        .as_str()
        .to_owned();

    let path = directory.join(&marker);
    let mut bytes = fs::read(&path).expect("the staged marker is readable");
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    fs::write(&path, &bytes).expect("the staged marker is writable");

    BundlePackageV1::verify_directory(&directory, &host())
        .expect_err("a component that is not the one the manifest measured is refused");
}

struct TestDirectory {
    path: PathBuf,
}

impl TestDirectory {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);

        let path = std::env::temp_dir().join(format!(
            "{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).expect("the test directory is creatable");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
