//! The package this fixture emits, as the canonical verifier and the tiered
//! kernel resolver read it back.
//!
//! Every assertion here runs against a real emitted directory:
//! `build-recurrent-fixture` compiles the native module, stages the components,
//! and publishes the closed package, and this reopens that package through
//! `runtime-bundle` rather than through anything this crate believes about
//! itself.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};

use bundle_recurrent_fixture::{
    RECURRENT_ARCHITECTURE_V1, RECURRENT_BUNDLE_ID_V1, RECURRENT_FIXTURE_NAME_V1,
    RECURRENT_GRAPH_BUCKET_V1, RECURRENT_KV_LAYOUT_ID_V1, RECURRENT_MINIMUM_DEVICE_MEMORY_BYTES_V1,
    RECURRENT_PROGRAM_COMPONENT_V1, RECURRENT_RANK_COUNT_V1, RECURRENT_STATE_BYTES_V1,
    RECURRENT_STATE_LAYOUT_ID_V1, RECURRENT_TOPOLOGY_ID_V1, RECURRENT_VOCABULARY_V1,
    recurrent_generic_copy_key_v1, recurrent_step_key_v1,
};
use runtime_bundle::{
    BUNDLE_MANIFEST_FILE, BUNDLE_NATIVE_ENTRY_SYMBOL_V1, BundleArtifactOriginV1,
    BundleHostCompatibilityV1, BundleManifestV1, BundleOpaqueId, BundleOperationProvisioningV1,
    BundlePackageV1, BundleStateLifetimeV1, BundleTopologyV1, GENERIC_COPY_F32_PRIMITIVE_V1,
    MODEL_BUNDLE_V1_ABI_VERSION, MODEL_BUNDLE_V1_SCHEMA, generic_copy_f32_contract_v1,
};
use runtime_host::{HostGenericCatalogV1, resolve_declared_operations_v1};
use runtime_kernels::{
    KernelImplementationCandidateV1, KernelImplementationTierV1, KernelOpaqueIdV1,
    KernelResolutionErrorV1, KernelSemanticKeyV1,
};

/// The host this fixture is emitted for.
fn host() -> BundleHostCompatibilityV1 {
    BundleHostCompatibilityV1::new(RECURRENT_ARCHITECTURE_V1, MODEL_BUNDLE_V1_ABI_VERSION)
        .expect("the test host descriptor is valid")
        .with_capabilities(
            [
                "chop.prompt-rows.v2",
                "chop.prefix-cache.state-slot.v1",
                "chop.prefix-cache.v1",
                "chop.serving-kv.v1",
                "chop.recurrent-state.v1",
                "chop.request-retirement.v1",
                "chop.speculative.v1",
                runtime_bundle::WEIGHT_PLAN_CAPABILITY_V1,
                runtime_bundle_abi::frontend::FRONTEND_CAPABILITY_V1,
            ]
            .map(|id| BundleOpaqueId::new(id).unwrap()),
        )
}

/// Emits the fixture package the acceptance harness emits, into `root`.
///
/// One emission at a time. `build-recurrent-fixture` compiles this crate's
/// shared object before it stages anything, and two of them at once would have
/// one cargo replacing the artifact the other is measuring.
fn emit_package(root: &Path) -> BundlePackageV1 {
    static EMITTING: Mutex<()> = Mutex::new(());

    let output = root.join(RECURRENT_BUNDLE_ID_V1);
    let emitted = {
        let _one_at_a_time = EMITTING.lock().unwrap_or_else(PoisonError::into_inner);
        Command::new(env!("CARGO_BIN_EXE_build-recurrent-fixture"))
            .args([
                "--fixture",
                RECURRENT_FIXTURE_NAME_V1,
                "--output",
                output.to_str().expect("the output path is UTF-8"),
            ])
            .output()
            .expect("build-recurrent-fixture starts")
    };
    assert!(
        emitted.status.success(),
        "build-recurrent-fixture failed:\nstdout:\n{}\nstderr:\n{}",
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
fn the_emitted_package_declares_one_tp1_topology_with_persistent_state() {
    let root = TestDirectory::new("recurrent-fixture-package");
    let package = emit_package(root.path());
    let manifest = package.manifest();

    assert_eq!(manifest.schema(), MODEL_BUNDLE_V1_SCHEMA);
    assert_eq!(manifest.abi_version(), MODEL_BUNDLE_V1_ABI_VERSION);
    assert_eq!(manifest.id().as_str(), RECURRENT_BUNDLE_ID_V1);

    // Exactly one rank: this fixture is the counterweight to a TP4 bundle.
    let [topology] = manifest.target().topologies() else {
        panic!("this fixture declares exactly one topology");
    };
    assert_eq!(topology.id().as_str(), RECURRENT_TOPOLOGY_ID_V1);
    assert_eq!(topology.rank_count(), RECURRENT_RANK_COUNT_V1);
    assert_eq!(
        manifest.target().architectures(),
        [id(RECURRENT_ARCHITECTURE_V1)]
    );

    // The model is persistent state, and the KV layout is declared and unused.
    assert_eq!(
        manifest.state_layout().id().as_str(),
        RECURRENT_STATE_LAYOUT_ID_V1
    );
    assert_eq!(manifest.state_layout().bytes(), RECURRENT_STATE_BYTES_V1);
    assert_eq!(
        manifest.state_layout().lifetime(),
        BundleStateLifetimeV1::Process
    );
    assert_eq!(
        manifest.kv_layout().id().as_str(),
        RECURRENT_KV_LAYOUT_ID_V1
    );

    // The one bucket references both components and both operations, which is
    // what makes them part of this bundle rather than declarations beside it.
    let [bucket] = manifest.graph_buckets() else {
        panic!("this fixture declares exactly one graph bucket");
    };
    assert_eq!(bucket.id().as_str(), RECURRENT_GRAPH_BUCKET_V1);
    assert_eq!(
        bucket.components().len(),
        manifest.components().len() + manifest.planned_components().len()
    );
    assert!(
        bucket
            .operations()
            .contains(&recurrent_step_key_v1().expect("the step key is valid"))
    );
    assert!(
        bucket
            .operations()
            .contains(&recurrent_generic_copy_key_v1().expect("the generic copy key is valid"))
    );

    // The program image is planned: the weight plan loads it from the
    // synthetic checkpoint, and the package ships no image.
    let [program] = manifest.planned_components() else {
        panic!("this fixture plans exactly one component");
    };
    assert_eq!(program.id().as_str(), RECURRENT_PROGRAM_COMPONENT_V1);

    let native_module = manifest
        .native_module()
        .expect("this fixture ships a native module");
    assert_eq!(native_module.entry_symbol(), BUNDLE_NATIVE_ENTRY_SYMBOL_V1);
}

/// One operation is this package's alone; the other is left to the host tier.
#[test]
fn one_operation_is_implemented_here_and_the_other_is_left_to_the_host() {
    let root = TestDirectory::new("recurrent-fixture-operations");
    let package = emit_package(root.path());
    let manifest = package.manifest();
    let step = recurrent_step_key_v1().expect("the step key is valid");
    let copy = recurrent_generic_copy_key_v1().expect("the generic copy key is valid");

    let requirements = manifest.operation_requirements();
    assert_eq!(requirements.len(), 2);
    let custom = requirements
        .iter()
        .find(|requirement| requirement.key() == &step)
        .expect("the custom operation is required");
    assert_eq!(
        custom.provisioning(),
        &BundleOperationProvisioningV1::BundleCustomOperation
    );
    let generic = requirements
        .iter()
        .find(|requirement| requirement.key() == &copy)
        .expect("the generic operation is required");
    assert_eq!(
        generic.provisioning(),
        &BundleOperationProvisioningV1::HostGenericPrimitive(id(GENERIC_COPY_F32_PRIMITIVE_V1))
    );

    // Exactly one implementation, and it is the custom operation's.
    let [implementation] = manifest.implementations() else {
        panic!("this fixture declares exactly one implementation");
    };
    assert_eq!(implementation.key(), &step);

    // The artifact is this package's native module, because that is where the
    // code that runs the step is compiled. The program component carries the
    // geometry words and the state seed the module reads, not the step, so a
    // record naming it would describe bytes the host uploads rather than the
    // callback it calls.
    let artifact = implementation.detail().artifact();
    let native_module = manifest
        .native_module()
        .expect("this fixture ships a native module");
    assert_eq!(artifact.origin(), &BundleArtifactOriginV1::NativeModule);
    assert_eq!(artifact.component(), None);
    assert_eq!(artifact.byte_length(), native_module.byte_length());
    assert_eq!(artifact.sha256(), native_module.sha256());
    // The program component is planned: no package file could stand in for
    // the module.
    assert!(
        manifest
            .components()
            .iter()
            .all(|component| component.id().as_str() != RECURRENT_PROGRAM_COMPONENT_V1)
    );

    // The source identity still names the file the callback is compiled from,
    // which is the provenance the artifact origin does not carry.
    assert_eq!(
        implementation.detail().source().path().as_str(),
        "src.operation.rs"
    );
}

/// The generic operation resolves through the host tier, and only through an
/// exact match; the custom one is this bundle's whatever a host offers.
#[test]
fn the_generic_operation_resolves_through_the_host_tier_or_stays_unsupported() {
    let root = TestDirectory::new("recurrent-fixture-resolution");
    let package = emit_package(root.path());
    let manifest = package.manifest();
    let keys = manifest.semantic_keys().expect("every contract has a key");
    let step = keys[&recurrent_step_key_v1().expect("the step key is valid")].clone();
    let copy =
        keys[&recurrent_generic_copy_key_v1().expect("the generic copy key is valid")].clone();
    // The host builds the shared declaration itself rather than reading the
    // manifest's, which is the whole point of a model-neutral primitive: two
    // parties that never met agree on one semantic key.
    assert_eq!(copy, host_copy_key(RECURRENT_VOCABULARY_V1));

    // With no host candidate at all, the generic operation is unsupported
    // rather than quietly satisfied by the bundle's own implementation.
    let alone = manifest
        .kernel_resolver([])
        .expect("the manifest's own implementations resolve");
    assert!(alone.is_bundle_private(&step));
    assert!(!alone.is_bundle_private(&copy));
    assert_eq!(
        alone
            .resolve(&step)
            .expect("the custom operation resolves")
            .tier(),
        KernelImplementationTierV1::Bundle
    );
    assert!(matches!(
        alone.resolve(&copy),
        Err(KernelResolutionErrorV1::Unsupported { .. })
    ));

    // A host that offers the exact contract, under the primitive the manifest
    // named, satisfies it generically.
    let offered = manifest
        .kernel_resolver([candidate(
            GENERIC_COPY_F32_PRIMITIVE_V1,
            host_copy_key(RECURRENT_VOCABULARY_V1),
            KernelImplementationTierV1::Generic,
        )])
        .expect("a generic host candidate is accepted");
    let resolved = offered
        .resolve(&copy)
        .expect("an exact generic candidate satisfies the operation");
    assert_eq!(resolved.tier(), KernelImplementationTierV1::Generic);
    assert_eq!(
        resolved.implementation_id().as_str(),
        GENERIC_COPY_F32_PRIMITIVE_V1
    );

    // A candidate that differs by any field is a different operation, so the
    // declared one is still unsupported rather than silently accepted. A host
    // that serves another width and a host that computes under other numerics
    // are both that.
    for (near_miss, reason) in [
        (host_copy_key(RECURRENT_VOCABULARY_V1 * 2), "another width"),
        (with_other_numerics(&copy), "other numerics"),
    ] {
        let mismatched = manifest
            .kernel_resolver([candidate(
                GENERIC_COPY_F32_PRIMITIVE_V1,
                near_miss,
                KernelImplementationTierV1::Generic,
            )])
            .expect("a near-miss host candidate is still a well-formed catalog");
        assert!(
            matches!(
                mismatched.resolve(&copy),
                Err(KernelResolutionErrorV1::Unsupported { .. })
            ),
            "a host offering {reason} must not satisfy this operation"
        );
    }

    // A host candidate carrying the custom operation's own key never satisfies
    // it: that operation is this package's, and this package implements it.
    let shadowed = manifest
        .kernel_resolver([candidate(
            "host.recurrent-step.v1",
            step.clone(),
            KernelImplementationTierV1::Generic,
        )])
        .expect("a host candidate for a private operation is accepted into the catalog");
    let resolved = shadowed
        .resolve(&step)
        .expect("the custom operation still resolves");
    assert_eq!(resolved.tier(), KernelImplementationTierV1::Bundle);
    assert_ne!(
        resolved.implementation_id().as_str(),
        "host.recurrent-step.v1"
    );
}

/// The real host catalog resolves both of this package's declared operations.
///
/// The two tests above build their candidates by hand, which proves the rules.
/// This one hands the emitted manifest to the catalog a real host is compiled
/// with, which is what says this package is loadable rather than merely
/// well-formed: the custom operation lands on this package's own
/// implementation, and the copy lands on the host's, under the primitive the
/// manifest asked for.
#[test]
fn the_real_host_catalog_resolves_every_operation_this_package_declares() {
    let root = TestDirectory::new("recurrent-fixture-host-catalog");
    let package = emit_package(root.path());
    let catalog = HostGenericCatalogV1::builtin_v1().expect("this host's catalog is valid");

    let resolved = resolve_declared_operations_v1(package.manifest(), &catalog)
        .expect("a real host resolves everything this package declares");

    assert_eq!(resolved.len(), 2);
    let step = resolved
        .iter()
        .find(|operation| operation.key() == &recurrent_step_key_v1().expect("the step key"))
        .expect("the custom operation resolved");
    assert_eq!(step.tier(), KernelImplementationTierV1::Bundle);
    let copy = resolved
        .iter()
        .find(|operation| {
            operation.key() == &recurrent_generic_copy_key_v1().expect("the copy key")
        })
        .expect("the generic operation resolved");
    assert_eq!(copy.tier(), KernelImplementationTierV1::Generic);
    assert_eq!(
        copy.provisioning(),
        &BundleOperationProvisioningV1::HostGenericPrimitive(id(GENERIC_COPY_F32_PRIMITIVE_V1))
    );
    assert_eq!(
        catalog
            .primitive_of(copy.implementation())
            .map(KernelOpaqueIdV1::as_str),
        Some(GENERIC_COPY_F32_PRIMITIVE_V1)
    );
}

/// A host the package does not target, or one too small for it, is refused.
#[test]
fn a_host_this_package_does_not_target_cannot_verify_it() {
    let root = TestDirectory::new("recurrent-fixture-host");
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
                BundleTopologyV1::new(id("tp4"), 4).expect("the test topology is valid"),
            ),
            "a topology this package does not declare",
        ),
        (
            host().with_device_memory_bytes(RECURRENT_MINIMUM_DEVICE_MEMORY_BYTES_V1 - 1),
            "less device memory than the package requires",
        ),
    ] {
        BundlePackageV1::verify_directory(&directory, &host)
            .expect_err(&format!("{reason} must be refused"));
    }
}

/// A manifest that names another schema or another ABI is refused outright.
#[test]
fn another_schema_or_abi_version_is_refused_before_the_body_is_read() {
    let root = TestDirectory::new("recurrent-fixture-schema");
    let package = emit_package(root.path());
    let canonical = fs::read(package.root().join(BUNDLE_MANIFEST_FILE))
        .expect("the emitted manifest is readable");
    let text = String::from_utf8(canonical).expect("the manifest is UTF-8");

    BundleManifestV1::from_canonical_json(text.as_bytes())
        .expect("the emitted manifest parses as canonical JSON");

    let other_schema = text.replace(MODEL_BUNDLE_V1_SCHEMA, "chop-model-bundle-v2");
    assert_ne!(other_schema, text, "the schema must appear in the manifest");
    let refusal = BundleManifestV1::from_canonical_json(other_schema.as_bytes())
        .expect_err("another schema is refused");
    assert!(refusal.to_string().contains("unsupported bundle schema"));

    let other_abi = text.replace(
        &format!("\"abi_version\":{MODEL_BUNDLE_V1_ABI_VERSION}"),
        "\"abi_version\":2",
    );
    assert_ne!(
        other_abi, text,
        "the ABI version must appear in the manifest"
    );
    let refusal = BundleManifestV1::from_canonical_json(other_abi.as_bytes())
        .expect_err("another ABI version is refused");
    assert!(
        refusal
            .to_string()
            .contains("unsupported bundle ABI version")
    );
}

/// The semantic key a host builds from the shared declaration, for one width.
fn host_copy_key(elements: u32) -> KernelSemanticKeyV1 {
    generic_copy_f32_contract_v1(elements)
        .expect("the shared copy contract is valid")
        .semantic_key()
        .expect("the shared copy contract has a semantic key")
}

/// One host-tier candidate for one exact semantic key.
fn candidate(
    implementation_id: &str,
    key: KernelSemanticKeyV1,
    tier: KernelImplementationTierV1,
) -> KernelImplementationCandidateV1 {
    KernelImplementationCandidateV1::new(
        KernelOpaqueIdV1::new(implementation_id).expect("the test implementation ID is valid"),
        key,
        tier,
    )
    .expect("the test candidate is valid")
}

/// The same operation under a numerical contract nobody declared.
fn with_other_numerics(key: &KernelSemanticKeyV1) -> KernelSemanticKeyV1 {
    let mut parts = key.parts();
    parts.numerical_contract = KernelOpaqueIdV1::new("fixture.recurrent.other-numerics.v1")
        .expect("the test numerical contract is valid");
    KernelSemanticKeyV1::new(parts).expect("the near-miss key is valid")
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
