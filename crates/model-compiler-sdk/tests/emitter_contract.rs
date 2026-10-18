#![forbid(unsafe_code)]

//! What the emitter promises about the directory it publishes.
//!
//! No device, no model, and no toolchain beyond `rustc` for the one native
//! module fixture: every test here writes bytes, emits a bundle, and reads the
//! result back through the canonical verifier.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use model_compiler_sdk::{
    BundleDraftV1, ComponentContentV1, ComponentDraftV1, FileIdentity,
    ImplementationArtifactDraftV1, ImplementationDraftV1, NativeModuleDraftV1, StagedContentV1,
    emit_bundle_directory_v1, plan_bundle_manifest_v1,
};
use runtime_bundle::{
    BUNDLE_NATIVE_MODULE_FORMAT_V1, BundleArtifactOriginV1, BundleBuildIdentityV1,
    BundleGraphBucketV1, BundleHostCompatibilityV1, BundleImplementationTierV1, BundleKvDtypeV1,
    BundleKvFormatV1, BundleKvLayoutV1, BundleLaunchContractV1, BundleNativeSymbolSequenceV1,
    BundleOpaqueId, BundleOperationKeyV1, BundleOperationProvisioningV1,
    BundleOperationRequirementV1, BundlePackageV1, BundleRankPlacementV1, BundleRelativePathV1,
    BundleResourceRequirementsV1, BundleScalarArgumentV1, BundleSemanticContractV1,
    BundleSourceIdentityV1, BundleSpecializationV1, BundleStateLayoutV1, BundleStateLifetimeV1,
    BundleTargetV1, BundleTensorAccessV1, BundleTensorAliasV1, BundleTensorRequirementV1,
    BundleTopologyV1, BundleWorkspaceContractV1, MIXED_PLANE_KV_CAPABILITY_V1,
    publish_staging_no_replace,
};
use runtime_weights::sha256_bytes;

const ARCHITECTURE: &str = "sm_120f";
const PLAN_BYTES: &[u8] = br#"{"schema":"fixture-graph-plan-v1","buckets":1}"#;
const MODULE_BYTES: &[u8] = b"fixture compiled module bytes v1";

fn id(value: &str) -> BundleOpaqueId {
    BundleOpaqueId::new(value).expect("the fixture identifier is valid")
}

fn path(value: &str) -> BundleRelativePathV1 {
    BundleRelativePathV1::new(value).expect("the fixture path is relative")
}

fn host() -> BundleHostCompatibilityV1 {
    BundleHostCompatibilityV1::new(ARCHITECTURE, 1).expect("the fixture host descriptor is valid")
}

fn weight_bytes() -> Vec<u8> {
    (0..300)
        .map(|index| (index as u8).wrapping_mul(7))
        .collect()
}

/// One draft with every content kind the emitter stages.
fn draft(module: Option<&Path>) -> BundleDraftV1 {
    release_draft(module, false)
}

fn release_draft(module: Option<&Path>, release: bool) -> BundleDraftV1 {
    let mut components = vec![
        ComponentDraftV1::new(
            id("fixture.weights"),
            id("fixture.weights.v1"),
            id("fixture-weight-bytes-v1"),
            path("components/weights.bin"),
            ComponentContentV1::Whole(StagedContentV1::bytes(weight_bytes())),
        ),
        ComponentDraftV1::new(
            id("fixture.graph-plan"),
            id("fixture.graph-plan.v1"),
            id("fixture-graph-plan-json-v1"),
            path("components/graph-plan.json"),
            ComponentContentV1::Whole(StagedContentV1::bytes(PLAN_BYTES)),
        ),
    ];
    if release {
        components.push(ComponentDraftV1::new(
            id("fixture.qualification"), id("fixture.qualification.v1"),
            id("json"), path("components/fixture.qualification.json"),
            ComponentContentV1::Whole(StagedContentV1::bytes(
                br#"{"artifact_complete":true,"executable":false,"execution_status":"unqualified","placeholder_only":false,"qualification_status":"unqualified","readiness":"artifact-complete-execution-unqualified","schema":"fixture-fp8-qualification-v1"}"#.to_vec(),
            )),
        ));
    }
    if release {
        // SDK-only fixtures. The model crate's metadata gate exercises real fixture bytes.
        for name in ["text-graph", "numerical-policy"] {
            components.push(ComponentDraftV1::new(
                id(&format!("fixture.{name}")),
                id(&format!("fixture.{name}")),
                id("json"),
                path(&format!("components/fixture.{name}.json")),
                ComponentContentV1::Whole(StagedContentV1::bytes(
                    format!(r#"{{"schema":"fixture-fp8-{name}-v1"}}"#).into_bytes(),
                )),
            ));
        }
    }
    let bucket = BundleGraphBucketV1::new(
        id("fixture-bucket"),
        components
            .iter()
            .map(|component| component.id().clone())
            .collect(),
        Vec::new(),
    )
    .expect("the fixture bucket is valid");
    let mut draft = BundleDraftV1::new(
        id(if release {
            "fixture-flash-fp8-tp4-sm120a"
        } else {
            "fixture.emitted.v1"
        }),
        BundleTargetV1::new(
            vec![id(ARCHITECTURE)],
            vec![BundleTopologyV1::new(id("tp2"), 2).expect("tp2 is valid")],
        )
        .expect("the fixture target is valid"),
        BundleResourceRequirementsV1::new(1, Vec::new()).expect("the fixture resources are valid"),
        BundleKvLayoutV1::new(
            id("fixture.kv.v1"),
            64,
            64,
            1,
            64,
            1,
            1_024,
            BundleRankPlacementV1::RankLocal,
            BundleStateLifetimeV1::Request,
        )
        .expect("the fixture KV layout is valid"),
        BundleStateLayoutV1::new(
            id("fixture.state.v1"),
            64,
            64,
            BundleStateLifetimeV1::Request,
        )
        .expect("the fixture state layout is valid"),
    )
    .with_components(components)
    .with_graph_buckets(vec![bucket]);
    if let Some(module) = module {
        draft = draft.with_native_module(NativeModuleDraftV1::new(
            path(&format!("native/{}", native_module_filename())),
            StagedContentV1::file(module),
        ));
    }
    draft
}

/// One named sequence, keeping the symbol order it was written with.
fn sequence(sequence_id: &str, symbols: &[&str]) -> BundleNativeSymbolSequenceV1 {
    BundleNativeSymbolSequenceV1::new(id(sequence_id), symbols.iter().copied().map(id).collect())
        .expect("the fixture symbol sequence is valid")
}

/// The operation key one fixture bundle requires and implements.
fn operation_key() -> BundleOperationKeyV1 {
    BundleOperationKeyV1::new(id("fixture.projection"), id("bf16.tp1"))
}

/// One complete semantic contract: what a host would have to match exactly.
fn semantic_contract() -> BundleSemanticContractV1 {
    let input = BundleTensorRequirementV1::new(
        id("input"),
        vec![32, 4096],
        id("bf16"),
        id("row-major.v1"),
        id("scale.none.v1"),
        Vec::new(),
        16,
        BundleTensorAccessV1::ReadOnly,
        BundleTensorAliasV1::Disjoint,
        Vec::new(),
    )
    .expect("the fixture input requirement is valid");
    let output = BundleTensorRequirementV1::new(
        id("output"),
        vec![32, 4096],
        id("bf16"),
        id("row-major.v1"),
        id("scale.none.v1"),
        Vec::new(),
        16,
        BundleTensorAccessV1::WriteOnly,
        BundleTensorAliasV1::Disjoint,
        Vec::new(),
    )
    .expect("the fixture output requirement is valid");
    BundleSemanticContractV1::new(
        id("fixture.projection.abi.v1"),
        1,
        vec![input, output],
        vec![
            BundleScalarArgumentV1::new(id("rows"), id("unsigned-int"), 32)
                .expect("the fixture scalar argument is valid"),
        ],
        id("fixture.numerics.v1"),
        vec![id("tensor-memory.v1")],
    )
    .expect("the fixture semantic contract is valid")
}

/// One implementation of that operation, anchored to a staged file.
fn implementation(artifact: ImplementationArtifactDraftV1) -> ImplementationDraftV1 {
    ImplementationDraftV1::new(
        id("fixture.projection.bundle"),
        operation_key(),
        artifact,
        id("fixture_projection_bf16_v1"),
        id(ARCHITECTURE),
        BundleLaunchContractV1::new([32, 1, 1], [256, 1, 1], 1024)
            .expect("the fixture launch contract is valid"),
        Some(
            BundleWorkspaceContractV1::new(
                8192,
                256,
                id("linear-workspace.v1"),
                BundleTensorAccessV1::ReadWrite,
                BundleTensorAliasV1::Disjoint,
            )
            .expect("the fixture workspace contract is valid"),
        ),
        vec![
            BundleSpecializationV1::new(id("tile-m"), id("128"))
                .expect("the fixture specialization is valid"),
        ],
        BundleSourceIdentityV1::new(
            id("fixture-source.v1"),
            id("0123456789abcdef"),
            id("kernels.projection.cu"),
            2048,
            sha256_bytes(b"fixture projection source"),
        )
        .expect("the fixture source identity is valid"),
        BundleBuildIdentityV1::new(
            id("nvcc"),
            id("13.2"),
            id("fixture-build.v1"),
            sha256_bytes(b"fixture projection recipe"),
        )
        .expect("the fixture build identity is valid"),
    )
}

/// A draft that requires one operation and implements it from a staged file.
fn operation_draft(artifact: ImplementationArtifactDraftV1) -> BundleDraftV1 {
    operation_draft_with(artifact, true)
}

/// The same draft, optionally without the requirement its implementation names.
fn operation_draft_with(
    artifact: ImplementationArtifactDraftV1,
    declare_requirement: bool,
) -> BundleDraftV1 {
    let components = vec![
        ComponentDraftV1::new(
            id("fixture.module"),
            id("fixture.module.v1"),
            id("fixture-module-bytes-v1"),
            path("modules/projection.module"),
            ComponentContentV1::Whole(StagedContentV1::bytes(MODULE_BYTES)),
        ),
        ComponentDraftV1::new(
            id("fixture.graph-plan"),
            id("fixture.graph-plan.v1"),
            id("fixture-graph-plan-json-v1"),
            path("components/graph-plan.json"),
            ComponentContentV1::Whole(StagedContentV1::bytes(PLAN_BYTES)),
        ),
    ];
    let bucket = BundleGraphBucketV1::new(
        id("fixture-bucket"),
        components
            .iter()
            .map(|component| component.id().clone())
            .collect(),
        if declare_requirement {
            vec![operation_key()]
        } else {
            Vec::new()
        },
    )
    .expect("the fixture bucket is valid");
    let requirements = if declare_requirement {
        vec![
            BundleOperationRequirementV1::new(
                operation_key(),
                BundleOperationProvisioningV1::BundleCustomOperation,
                semantic_contract(),
            )
            .expect("the fixture operation requirement is valid"),
        ]
    } else {
        Vec::new()
    };
    BundleDraftV1::new(
        id("fixture.operations.v1"),
        BundleTargetV1::new(
            vec![id(ARCHITECTURE)],
            vec![BundleTopologyV1::new(id("tp1"), 1).expect("tp1 is valid")],
        )
        .expect("the fixture target is valid"),
        BundleResourceRequirementsV1::new(1, Vec::new()).expect("the fixture resources are valid"),
        BundleKvLayoutV1::new(
            id("fixture.kv.v1"),
            64,
            64,
            1,
            64,
            1,
            1_024,
            BundleRankPlacementV1::RankLocal,
            BundleStateLifetimeV1::Request,
        )
        .expect("the fixture KV layout is valid"),
        BundleStateLayoutV1::new(
            id("fixture.state.v1"),
            64,
            64,
            BundleStateLifetimeV1::Request,
        )
        .expect("the fixture state layout is valid"),
    )
    .with_components(components)
    .with_graph_buckets(vec![bucket])
    .with_operation_requirements(requirements)
    .with_implementations(vec![implementation(artifact)])
}

/// A declared implementation is anchored to the component that was staged.
///
/// The compiler names the component; the format, length, and digest in the
/// artifact identity are the emitter's measurements of the file it wrote, so an
/// implementation can never point at bytes outside the closed file set.
#[test]
fn a_declared_implementation_is_anchored_to_the_staged_artifact() {
    let root = TemporaryDirectory::new("model-compiler-sdk-operations");
    let output = root.path().join("bundle");

    let emitted = emit_bundle_directory_v1(
        &operation_draft(ImplementationArtifactDraftV1::Component(id(
            "fixture.module",
        ))),
        &output,
    )
    .expect("it emits");

    let manifest = emitted.manifest();
    let requirement = manifest
        .operation_requirements()
        .first()
        .expect("the emitted manifest declares the required operation");
    assert_eq!(requirement.key(), &operation_key());
    assert_eq!(
        requirement.provisioning(),
        &BundleOperationProvisioningV1::BundleCustomOperation
    );
    assert_eq!(requirement.semantic_contract(), &semantic_contract());

    let declared = manifest
        .implementations()
        .first()
        .expect("the emitted manifest declares the implementation");
    assert_eq!(declared.key(), &operation_key());
    assert_eq!(declared.tier(), BundleImplementationTierV1::Bundle);

    let component = manifest
        .components()
        .iter()
        .find(|component| component.id().as_str() == "fixture.module")
        .expect("the artifact component is declared");
    let artifact = declared.detail().artifact();
    assert_eq!(
        artifact.origin(),
        &BundleArtifactOriginV1::Component(component.id().clone())
    );
    assert_eq!(artifact.component(), Some(component.id()));
    assert_eq!(artifact.format(), component.format());
    assert_eq!(artifact.byte_length(), component.byte_length());
    assert_eq!(artifact.sha256(), component.sha256());
    assert_eq!(artifact.byte_length(), MODULE_BYTES.len() as u64);
    assert_eq!(artifact.sha256(), sha256_bytes(MODULE_BYTES));

    // The whole package, requirements and implementations included, is what the
    // canonical verifier accepted from disk.
    BundlePackageV1::verify_directory(&output, &host())
        .expect("a package with operations re-verifies as it is loaded");
    let resolver = manifest
        .kernel_resolver([])
        .expect("the declared implementations resolve against their requirements");
    let keys = manifest
        .semantic_keys()
        .expect("every requirement carries a semantic key");
    let resolution = resolver
        .resolve(&keys[declared.key()])
        .expect("the emitted bundle satisfies the operation it declared");
    assert!(
        resolution.detail().is_some(),
        "a selected bundle candidate carries the detail the manifest declared"
    );
}

/// An implementation pointing at a component the draft never declared is
/// refused before a byte is written.
#[test]
fn an_implementation_without_its_artifact_component_is_refused() {
    let root = TemporaryDirectory::new("model-compiler-sdk-unanchored");
    let output = root.path().join("bundle");

    let error = emit_bundle_directory_v1(
        &operation_draft(ImplementationArtifactDraftV1::Component(id(
            "fixture.absent-module",
        ))),
        &output,
    )
    .expect_err("an unanchored implementation must be refused");

    assert!(
        error
            .to_string()
            .contains("names artifact component fixture.absent-module, which it does not declare"),
        "the refusal must name the component that is missing: {error}"
    );
    assert!(!output.exists(), "a refused emission publishes nothing");
    assert_eq!(staging_entries(root.path()), Vec::<String>::new());
}

/// A native-backed implementation is measured from the module that was staged.
///
/// The draft names no length and no digest: it says the artifact is this
/// package's native module, and the emitter fills both in from the file it
/// wrote, exactly as it does for a component. The record therefore describes
/// the code the host will call rather than a component travelling beside it.
#[test]
fn a_native_backed_implementation_is_measured_from_the_staged_module() {
    let root = TemporaryDirectory::new("model-compiler-sdk-native-backed");
    let output = root.path().join("bundle");
    let module = compile_native_module(root.path(), "chop_bundle_get_api_v1");
    let draft = operation_draft(ImplementationArtifactDraftV1::NativeModule).with_native_module(
        NativeModuleDraftV1::new(
            path(&format!("native/{}", native_module_filename())),
            StagedContentV1::file(&module),
        ),
    );

    let emitted = emit_bundle_directory_v1(&draft, &output).expect("it emits");

    let manifest = emitted.manifest();
    let native_module = manifest
        .native_module()
        .expect("the emitted package ships a native module");
    let declared = manifest
        .implementations()
        .first()
        .expect("the emitted manifest declares the implementation");
    let artifact = declared.detail().artifact();
    assert_eq!(artifact.origin(), &BundleArtifactOriginV1::NativeModule);
    assert_eq!(artifact.component(), None);
    assert_eq!(
        artifact.format().as_str(),
        BUNDLE_NATIVE_MODULE_FORMAT_V1,
        "a native module has one format"
    );
    assert_eq!(artifact.byte_length(), native_module.byte_length());
    assert_eq!(artifact.sha256(), native_module.sha256());
    assert_eq!(
        artifact.sha256(),
        sha256_bytes(&fs::read(&module).expect("the staged module is readable")),
        "the digest must be the module's own bytes"
    );
    // The components this package also staged are not what the record names.
    assert_ne!(artifact.sha256(), sha256_bytes(MODULE_BYTES));

    BundlePackageV1::verify_directory(&output, &host())
        .expect("a native-backed package re-verifies as it is loaded");
}

/// A native-backed implementation in a package with no module is refused.
#[test]
fn a_native_backed_implementation_without_a_module_is_refused() {
    let root = TemporaryDirectory::new("model-compiler-sdk-native-absent");
    let output = root.path().join("bundle");

    let error = emit_bundle_directory_v1(
        &operation_draft(ImplementationArtifactDraftV1::NativeModule),
        &output,
    )
    .expect_err("a native-backed implementation with no module must be refused");

    assert!(
        error
            .to_string()
            .contains("is backed by a native module, which it does not declare"),
        "the refusal must say the module is missing: {error}"
    );
    assert!(!output.exists(), "a refused emission publishes nothing");
    assert_eq!(staging_entries(root.path()), Vec::<String>::new());
}

/// An implementation of an operation the bundle never required is refused, and
/// the staged bytes are removed rather than published.
#[test]
fn an_implementation_without_a_requirement_is_refused() {
    let root = TemporaryDirectory::new("model-compiler-sdk-unrequired");
    let output = root.path().join("bundle");
    let draft = operation_draft_with(
        ImplementationArtifactDraftV1::Component(id("fixture.module")),
        false,
    );

    let error = emit_bundle_directory_v1(&draft, &output)
        .expect_err("an implementation of nothing must be refused");

    assert!(
        error.to_string().contains("references unknown operation"),
        "the refusal must name the operation nothing requires: {error}"
    );
    assert!(!output.exists(), "a refused emission publishes nothing");
    assert_eq!(staging_entries(root.path()), Vec::<String>::new());
}

/// The same draft emits the same bytes, wherever it is emitted.
#[test]
fn dry_run_manifest_is_exactly_the_emitted_manifest() {
    let root = TemporaryDirectory::new("model-compiler-sdk-dry-run");
    let output = root.path().join("bundle");
    let draft = draft(None);
    let planned = plan_bundle_manifest_v1(&draft).expect("the dry run closes");
    assert!(!output.exists(), "the dry run writes no package");
    let emitted = emit_bundle_directory_v1(&draft, &output).expect("the bundle emits");
    assert_eq!(planned, *emitted.manifest());
    assert_eq!(
        planned
            .to_canonical_json()
            .expect("planned manifest renders"),
        emitted
            .manifest()
            .to_canonical_json()
            .expect("emitted manifest renders")
    );
}

#[test]
fn one_draft_emits_one_canonical_directory() {
    let root = TemporaryDirectory::new("model-compiler-sdk-canonical");
    let first = root.path().join("first");
    let second = root.path().join("elsewhere/second");
    fs::create_dir_all(second.parent().expect("the second output has a parent"))
        .expect("the nested output parent is creatable");

    let left = emit_bundle_directory_v1(&draft(None), &first).expect("the first bundle emits");
    let right = emit_bundle_directory_v1(&draft(None), &second).expect("the second bundle emits");

    assert_eq!(left.manifest(), right.manifest());
    assert_eq!(
        fs::read(first.join("manifest.json")).expect("the first manifest is readable"),
        fs::read(second.join("manifest.json")).expect("the second manifest is readable"),
        "one draft must serialize to one canonical manifest"
    );
    for relative in ["components/weights.bin", "components/graph-plan.json"] {
        assert_eq!(
            fs::read(first.join(relative)).expect("the first component is readable"),
            fs::read(second.join(relative)).expect("the second component is readable")
        );
    }
}

/// Every declared length and digest describes the bytes that were written.
#[test]
fn the_manifest_measures_what_was_written() {
    let root = TemporaryDirectory::new("model-compiler-sdk-measured");
    let output = root.path().join("bundle");

    let emitted = emit_bundle_directory_v1(&draft(None), &output).expect("the bundle emits");

    let manifest = emitted.manifest();
    for component in manifest.components() {
        let bytes = fs::read(output.join(component.path().as_path()))
            .expect("the declared component is readable");
        assert_eq!(component.byte_length(), bytes.len() as u64);
        assert_eq!(component.sha256(), sha256_bytes(&bytes));
    }
    assert_eq!(
        fs::read(output.join("components/weights.bin")).expect("the weights are readable"),
        weight_bytes()
    );
}

/// The published directory holds the declared files and nothing else, and it
/// reads back through the same verifier a host loads it with.
#[test]
fn the_published_directory_is_a_closed_verified_package() {
    let root = TemporaryDirectory::new("model-compiler-sdk-closed");
    let output = root.path().join("bundle");

    let emitted = emit_bundle_directory_v1(&draft(None), &output).expect("the bundle emits");

    let declared = emitted
        .manifest()
        .components()
        .iter()
        .map(|component| component.path().as_str().to_owned())
        .chain(std::iter::once("manifest.json".to_owned()))
        .collect::<BTreeSet<_>>();
    assert_eq!(declared, files_under(&output));
    BundlePackageV1::verify_directory(&output, &host())
        .expect("an emitted package re-verifies as it is loaded");

    fs::write(output.join("components/undeclared.bin"), b"extra")
        .expect("the extra file is writable");
    assert!(
        BundlePackageV1::verify_directory(&output, &host()).is_err(),
        "a file the manifest never declared must reject the package"
    );
}

/// Bytes that change after emission are refused, at the same length.
#[test]
fn drifted_component_bytes_are_refused() {
    let root = TemporaryDirectory::new("model-compiler-sdk-drift");
    let output = root.path().join("bundle");
    emit_bundle_directory_v1(&draft(None), &output).expect("the bundle emits");

    let plan = output.join("components/graph-plan.json");
    let mut drifted = fs::read(&plan).expect("the plan component is readable");
    let last = drifted.len() - 1;
    drifted[last] = drifted[last].wrapping_add(1);
    fs::write(&plan, &drifted).expect("the plan component is writable");

    let error = BundlePackageV1::verify_directory(&output, &host())
        .expect_err("drifted bytes must not verify");
    assert!(
        error.to_string().contains("SHA-256"),
        "the refusal must name the digest that differs: {error}"
    );
}

/// Two components that share an identity are refused before anything is
/// written, because one of them would otherwise be staged over the other.
#[test]
fn duplicate_component_identities_are_refused() {
    let root = TemporaryDirectory::new("model-compiler-sdk-duplicates");

    for (label, component_id, component_role, component_path, expected) in [
        (
            "identifier",
            "fixture.graph-plan",
            "fixture.other.v1",
            "components/other.json",
            "component fixture.graph-plan more than once",
        ),
        (
            "role",
            "fixture.other",
            "fixture.graph-plan.v1",
            "components/other.json",
            "role fixture.graph-plan.v1 on more than one component",
        ),
        (
            "path",
            "fixture.other",
            "fixture.other.v1",
            "components/graph-plan.json",
            "more than one file at components/graph-plan.json",
        ),
    ] {
        let base = draft(None);
        let mut components = base.components().to_vec();
        components.push(ComponentDraftV1::new(
            id(component_id),
            id(component_role),
            id("fixture-graph-plan-json-v1"),
            path(component_path),
            ComponentContentV1::Whole(StagedContentV1::bytes(PLAN_BYTES)),
        ));
        let output = root.path().join(format!("bundle-{label}"));

        let Err(error) = emit_bundle_directory_v1(&base.with_components(components), &output)
        else {
            panic!("a duplicate {label} must be refused");
        };

        assert!(error.to_string().contains(expected), "{error}");
        assert!(!output.exists(), "a refused emission publishes nothing");
    }
}

/// A planned component is declared in the manifest and never staged; the
/// package then needs the weight-plan capability.
#[test]
fn a_planned_component_is_declared_without_a_file() {
    let root = TemporaryDirectory::new("model-compiler-sdk-planned");
    let base = draft(None);
    let mut components = base.components().to_vec();
    components.push(
        ComponentDraftV1::planned(
            id("fixture.planned-image"),
            id("fixture.planned-image.v1"),
            id("fixture-rank-image-bytes-v1"),
        )
        .expect("the planned component is valid"),
    );
    let bucket = BundleGraphBucketV1::new(
        id("fixture-bucket"),
        components.iter().map(|c| c.id().clone()).collect(),
        Vec::new(),
    )
    .expect("the bucket is valid");
    let planned = base
        .with_components(components)
        .with_graph_buckets(vec![bucket]);
    let refused = root.path().join("refused");
    assert!(emit_bundle_directory_v1(&planned, &refused).is_err());
    assert!(!refused.exists());
    let planned = planned.with_resource_requirements(
        BundleResourceRequirementsV1::new(1, vec![id(runtime_bundle::WEIGHT_PLAN_CAPABILITY_V1)])
            .expect("the requirements are valid"),
    );
    let output = root.path().join("bundle");
    let emitted = emit_bundle_directory_v1(&planned, &output).expect("a planned package emits");
    let manifest = emitted.manifest();
    assert_eq!(
        manifest
            .planned_components()
            .iter()
            .map(|c| (c.id().as_str(), c.role().as_str()))
            .collect::<Vec<_>>(),
        [("fixture.planned-image", "fixture.planned-image.v1")]
    );
    assert!(
        manifest
            .components()
            .iter()
            .all(|c| c.id().as_str() != "fixture.planned-image")
    );
    assert!(!files_under(&output).iter().any(|f| f.contains("planned")));
    assert_eq!(plan_bundle_manifest_v1(&planned).unwrap(), *manifest);
}

/// An output that already exists is never replaced, merged into, or removed.
#[test]
fn an_existing_output_is_never_replaced() {
    let root = TemporaryDirectory::new("model-compiler-sdk-no-replace");
    let output = root.path().join("bundle");
    fs::create_dir(&output).expect("the occupied output is creatable");
    fs::write(output.join("keep.txt"), b"keep").expect("the occupied file is writable");

    let error = emit_bundle_directory_v1(&draft(None), &output)
        .expect_err("an existing output must not be replaced");

    assert!(
        error.to_string().contains("refusing to replace"),
        "the refusal must say what it refused: {error}"
    );
    assert_eq!(
        files_under(&output),
        BTreeSet::from(["keep.txt".to_owned()]),
        "the existing directory must be left exactly as it was"
    );
    assert_eq!(
        staging_entries(root.path()),
        Vec::<String>::new(),
        "a refused emission stages nothing"
    );
}

/// A source that is a symlink is refused, and a path that leaves the bundle
/// cannot even be spelled.
#[cfg(unix)]
#[test]
fn hostile_inputs_and_paths_are_refused() {
    use std::os::unix::fs::symlink;

    let root = TemporaryDirectory::new("model-compiler-sdk-hostile");
    let secret = root.path().join("secret.bin");
    fs::write(&secret, b"secret").expect("the target file is writable");
    let link = root.path().join("link.bin");
    symlink(&secret, &link).expect("the fixture symlink is creatable");
    let output = root.path().join("bundle");

    let mut hostile = draft(None);
    hostile = replace_plan_content(hostile, StagedContentV1::file(&link));
    let error = emit_bundle_directory_v1(&hostile, &output)
        .expect_err("a symlinked input must not be staged");
    assert!(
        error.to_string().contains("must be a regular file"),
        "the refusal must name the input kind: {error}"
    );
    assert!(!output.exists(), "a refused emission publishes nothing");

    for hostile_path in [
        "../escape.bin",
        "/etc/passwd",
        "components/../../escape.bin",
    ] {
        assert!(
            BundleRelativePathV1::new(hostile_path).is_err(),
            "a path that leaves the bundle must not be constructible: {hostile_path}"
        );
    }
}

/// A failure part-way through leaves neither an output nor a staging directory.
#[test]
fn a_failed_emission_removes_its_staging_directory() {
    let root = TemporaryDirectory::new("model-compiler-sdk-cleanup");
    let output = root.path().join("bundle");
    let absent = root.path().join("never-written.bin");

    let missing = replace_plan_content(draft(None), StagedContentV1::file(&absent));
    let error = emit_bundle_directory_v1(&missing, &output)
        .expect_err("a missing input must fail the emission");

    assert!(
        error.to_string().contains("never-written.bin"),
        "the refusal must name the missing input: {error}"
    );
    assert!(!output.exists(), "a failed emission publishes nothing");
    assert_eq!(
        staging_entries(root.path()),
        Vec::<String>::new(),
        "a failed emission leaves no staging directory behind"
    );
}

/// A publication that fails part-way leaves nothing a host would load.
///
/// The emitter promises `output` is absent only for failures before
/// publication. Publication reserves the directory and links the manifest last,
/// so a failure after the reservation leaves a directory behind. That directory
/// never gets its commit marker, so no verifier and no loader accepts it, and
/// the emitter does not delete it, because by then it is a directory another
/// party may have written into.
#[test]
fn a_failed_publication_leaves_nothing_loadable() {
    let root = TemporaryDirectory::new("model-compiler-sdk-commit-marker");
    let source = root.path().join("source");
    let destination = root.path().join("destination");
    let emitted = emit_bundle_directory_v1(&draft(None), &source).expect("the bundle emits");

    // A declared file that is gone is what a publication failure looks like
    // from the inside: the reservation and the earlier links are already made.
    fs::remove_file(source.join("components/weights.bin"))
        .expect("the staged component is removable");

    let error = publish_staging_no_replace(&source, &destination, emitted.manifest())
        .expect_err("publication must fail on a declared file it cannot link");

    assert!(
        error.to_string().contains("weights.bin"),
        "the refusal must name the file that could not be published: {error}"
    );
    assert!(
        destination.is_dir(),
        "the reserved directory stays, because deleting it could take another party's files"
    );
    assert!(
        !destination.join("manifest.json").exists(),
        "the commit marker is linked last and never appears on a failed publication"
    );
    assert!(
        BundlePackageV1::verify_directory(&destination, &host()).is_err(),
        "a directory without its manifest must not verify"
    );
    let error = emit_bundle_directory_v1(&draft(None), &destination)
        .expect_err("a partial directory blocks the retry rather than being overwritten");
    assert!(
        error.to_string().contains("refusing to replace"),
        "retrying into the partial directory must be refused: {error}"
    );
}

/// The native module is staged, measured, and checked for its one entry symbol.
#[test]
fn a_native_module_is_staged_and_verified() {
    let root = TemporaryDirectory::new("model-compiler-sdk-native");
    let module = compile_native_module(root.path(), "chop_bundle_get_api_v1");
    let output = root.path().join("bundle");

    let emitted =
        emit_bundle_directory_v1(&draft(Some(&module)), &output).expect("the bundle emits");

    let native = emitted
        .manifest()
        .native_module()
        .expect("the emitted manifest declares the native module");
    assert_eq!(native.entry_symbol(), "chop_bundle_get_api_v1");
    let bytes = fs::read(output.join(native.path().as_path()))
        .expect("the staged native module is readable");
    assert_eq!(native.byte_length(), bytes.len() as u64);
    assert_eq!(native.sha256(), sha256_bytes(&bytes));

    let wrong = compile_native_module(root.path(), "chop_bundle_get_api_v2");
    let other = root.path().join("other");
    let error = emit_bundle_directory_v1(&draft(Some(&wrong)), &other)
        .expect_err("a module without the entry symbol must not be published");
    assert!(
        error.to_string().contains("does not export"),
        "the refusal must name the missing entry symbol: {error}"
    );
    assert!(!other.exists(), "a refused emission publishes nothing");
}

#[test]
fn declared_symbol_sequences_reach_the_manifest_in_the_order_they_name() {
    let root = TemporaryDirectory::new("model-compiler-sdk-symbol-sequences");
    let module = compile_native_module(root.path(), "chop_bundle_get_api_v1");
    let output = root.path().join("bundle");
    // Declared in the reverse of canonical order, so what comes back proves the
    // manifest sorted the sequences and left the names inside them alone.
    let sequences = vec![
        sequence("route.prefill", &["fixture.norm", "fixture.gather"]),
        sequence(
            "route.decode",
            &["fixture.gather", "fixture.norm", "fixture.gemm"],
        ),
    ];

    let emitted = emit_bundle_directory_v1(
        &draft(Some(&module)).with_native_symbol_sequences(sequences),
        &output,
    )
    .expect("the bundle emits");

    let declared = emitted.manifest().native_symbol_sequences();
    assert_eq!(
        declared
            .iter()
            .map(|value| value.id().as_str())
            .collect::<Vec<_>>(),
        ["route.decode", "route.prefill"]
    );
    assert_eq!(
        declared[0]
            .symbols()
            .iter()
            .map(BundleOpaqueId::as_str)
            .collect::<Vec<_>>(),
        ["fixture.gather", "fixture.norm", "fixture.gemm"]
    );
    assert_eq!(
        declared[1]
            .symbols()
            .iter()
            .map(BundleOpaqueId::as_str)
            .collect::<Vec<_>>(),
        ["fixture.norm", "fixture.gather"]
    );
}

#[test]
fn a_draft_that_declares_no_symbol_sequence_emits_what_it_emitted_before() {
    let root = TemporaryDirectory::new("model-compiler-sdk-no-symbol-sequences");
    let module = compile_native_module(root.path(), "chop_bundle_get_api_v1");
    let silent = root.path().join("silent");
    let empty = root.path().join("empty");

    let before =
        emit_bundle_directory_v1(&draft(Some(&module)), &silent).expect("the bundle emits");
    let after = emit_bundle_directory_v1(
        &draft(Some(&module)).with_native_symbol_sequences(Vec::new()),
        &empty,
    )
    .expect("the bundle emits");

    assert!(before.manifest().native_symbol_sequences().is_empty());
    let silent_bytes = fs::read(silent.join("manifest.json")).expect("the manifest is readable");
    let empty_bytes = fs::read(empty.join("manifest.json")).expect("the manifest is readable");
    assert_eq!(
        silent_bytes, empty_bytes,
        "an empty sequence list must not change the emitted manifest"
    );
    assert!(
        !String::from_utf8(silent_bytes)
            .expect("the manifest is UTF-8")
            .contains("native_symbol_sequences"),
        "a bundle that declares no sequence must not write the key"
    );
    assert!(after.manifest().native_symbol_sequences().is_empty());
}

#[test]
fn a_symbol_sequence_without_a_module_is_refused_before_a_file_is_staged() {
    let root = TemporaryDirectory::new("model-compiler-sdk-symbol-sequence-no-module");
    let output = root.path().join("bundle");

    let error = emit_bundle_directory_v1(
        &draft(None)
            .with_native_symbol_sequences(vec![sequence("route.decode", &["fixture.norm"])]),
        &output,
    )
    .expect_err("a sequence with no module must not be published");

    assert!(
        error.to_string().contains("no native module"),
        "the refusal must name the missing module: {error}"
    );
    assert!(!output.exists(), "a refused emission publishes nothing");
}

/// An input the compiler did not produce is staged only if it is the file the
/// pinned identity names, and the emitted manifest measures what was written
/// from it.
///
/// One input is larger than the buffer the emitter hashes through, so the file
/// crosses several reads and the digest may not depend on where those reads
/// landed.
#[test]
fn authenticated_inputs_are_staged_against_their_pinned_identities() {
    let root = TemporaryDirectory::new("model-compiler-sdk-authenticated");
    let output = root.path().join("bundle");
    let sources = [
        pinned_rank(root.path(), 0, 1_200_003),
        pinned_rank(root.path(), 1, 900_007),
    ];

    let emitted = emit_bundle_directory_v1(&authenticated_draft(&sources), &output)
        .expect("pinned inputs stage");

    let manifest = emitted.manifest();
    for (index, (source, identity)) in sources.iter().enumerate() {
        let component = manifest
            .components()
            .iter()
            .find(|component| component.id().as_str() == format!("fixture.pinned-{index}"))
            .expect("the pinned component is declared");
        let staged = fs::read(output.join(component.path().as_path()))
            .expect("the staged component is readable");
        assert_eq!(
            staged,
            fs::read(source).expect("the pinned input is readable"),
            "input {index} must be the pinned bytes, byte for byte"
        );
        assert_eq!(component.byte_length(), identity.size());
        assert_eq!(component.sha256(), identity.sha256());
    }

    BundlePackageV1::verify_directory(&output, &host())
        .expect("a package built from pinned inputs re-verifies as it is loaded");
}

/// Every way a pinned input can fail to be the file it was pinned as, and the
/// one outcome all of them have: nothing is described and nothing is published.
#[test]
fn an_input_that_is_not_its_pinned_file_is_refused() {
    for (label, prepare, expected) in [
        (
            "name",
            prepare_wrong_name as fn(&Path) -> (PathBuf, FileIdentity),
            "which its pinned identity requires",
        ),
        (
            "length",
            prepare_wrong_length,
            "has 512 bytes, its pinned identity declares 513",
        ),
        ("digest", prepare_wrong_digest, "has SHA-256 "),
        ("mutation", prepare_same_length_mutation, "has SHA-256 "),
        ("replacement", prepare_path_replacement, "has SHA-256 "),
    ] {
        let case = TemporaryDirectory::new(&format!("model-compiler-sdk-pinned-{label}"));
        let output = case.path().join("bundle");
        let first = prepare(case.path());
        let second = pinned_rank(case.path(), 1, 512);

        let Err(error) = emit_bundle_directory_v1(&authenticated_draft(&[first, second]), &output)
        else {
            panic!("a pinned input with the wrong {label} must not be staged");
        };

        assert!(
            error.to_string().contains(expected),
            "the refusal must say what did not match ({label}): {error}"
        );
        // The refusal a pinned part raises mid-stream reaches the caller in the
        // part's own words, rather than as whatever the digest or the write saw
        // when the stream stopped under them.
        assert!(
            error.to_string().starts_with("bundle input "),
            "the refusal must name the input it refused ({label}): {error}"
        );
        assert!(
            !output.exists(),
            "a refused pinned input publishes nothing ({label})"
        );
        assert_eq!(
            staging_entries(case.path()),
            Vec::<String>::new(),
            "a refused pinned input leaves no staging directory ({label})"
        );
    }
}

/// A pinned input whose path is a symlink is refused, whatever it points at.
#[cfg(unix)]
#[test]
fn a_symlinked_pinned_input_is_refused() {
    use std::os::unix::fs::symlink;

    let root = TemporaryDirectory::new("model-compiler-sdk-pinned-symlink");
    let output = root.path().join("bundle");
    let (path, identity) = pinned_rank(root.path(), 0, 512);
    // The same name, the same length, the same bytes, and a symlink where the
    // regular file was. Only the kind of the thing at that path differs.
    let elsewhere = root.path().join("elsewhere.bin");
    fs::rename(&path, &elsewhere).expect("the pinned file is movable");
    symlink(&elsewhere, &path).expect("the fixture symlink is creatable");

    let error = emit_bundle_directory_v1(
        &authenticated_draft(&[(path, identity), pinned_rank(root.path(), 1, 512)]),
        &output,
    )
    .expect_err("a symlinked pinned input must not be staged");

    assert!(
        error.to_string().contains("must be a regular file"),
        "the refusal must name the input kind: {error}"
    );
    assert!(!output.exists(), "a refused emission publishes nothing");
    assert_eq!(staging_entries(root.path()), Vec::<String>::new());
}

/// An output that already exists is not replaced by a pinned-input draft, and
/// the refusal happens before any pinned file is read.
#[test]
fn an_existing_output_is_never_replaced_by_authenticated_inputs() {
    let root = TemporaryDirectory::new("model-compiler-sdk-pinned-no-replace");
    let output = root.path().join("bundle");
    fs::create_dir(&output).expect("the occupied output is creatable");
    fs::write(output.join("keep.txt"), b"keep").expect("the occupied file is writable");
    let sources = [
        pinned_rank(root.path(), 0, 512),
        pinned_rank(root.path(), 1, 512),
    ];

    let error = emit_bundle_directory_v1(&authenticated_draft(&sources), &output)
        .expect_err("an existing output must not be replaced");

    assert!(
        error.to_string().contains("refusing to replace"),
        "the refusal must say what it refused: {error}"
    );
    assert_eq!(
        files_under(&output),
        BTreeSet::from(["keep.txt".to_owned()]),
        "the existing directory must be left exactly as it was"
    );
    assert_eq!(
        staging_entries(root.path()),
        Vec::<String>::new(),
        "a refused emission stages nothing"
    );
}

/// One pinned rank image on disk, and the identity that names it exactly.
fn pinned_rank(root: &Path, rank: u8, length: usize) -> (PathBuf, FileIdentity) {
    let bytes = pinned_bytes(rank, length);
    let name = format!("rank-{rank}.bin");
    let path = root.join(&name);
    fs::write(&path, &bytes).expect("the pinned rank image is writable");
    let identity = FileIdentity::new(&name, bytes.len() as u64, &sha256_bytes(&bytes))
        .expect("the pinned identity is valid");
    (path, identity)
}

fn pinned_bytes(rank: u8, length: usize) -> Vec<u8> {
    (0..length)
        .map(|index| (index as u8).wrapping_mul(31).wrapping_add(rank))
        .collect()
}

/// The pinned file is real and correct, but it does not live under the name the
/// identity pins.
fn prepare_wrong_name(root: &Path) -> (PathBuf, FileIdentity) {
    let (path, identity) = pinned_rank(root, 0, 512);
    let renamed = root.join("under-another-name.bin");
    fs::rename(&path, &renamed).expect("the pinned file is movable");
    (renamed, identity)
}

/// The identity declares a length the file does not have.
fn prepare_wrong_length(root: &Path) -> (PathBuf, FileIdentity) {
    let (path, identity) = pinned_rank(root, 0, 512);
    let stretched = FileIdentity::new(identity.name(), identity.size() + 1, identity.sha256())
        .expect("the stretched identity is valid");
    (path, stretched)
}

/// The identity is the right name and the right length, and the digest of some
/// other bytes entirely.
fn prepare_wrong_digest(root: &Path) -> (PathBuf, FileIdentity) {
    let (path, identity) = pinned_rank(root, 0, 512);
    let elsewhere = FileIdentity::new(
        identity.name(),
        identity.size(),
        &sha256_bytes(&pinned_bytes(9, 512)),
    )
    .expect("the mismatched identity is valid");
    (path, elsewhere)
}

/// The file was the pinned one when it was measured, and its bytes changed
/// under it afterwards without its length changing.
fn prepare_same_length_mutation(root: &Path) -> (PathBuf, FileIdentity) {
    let (path, identity) = pinned_rank(root, 0, 512);
    let mut mutated = fs::read(&path).expect("the pinned file is readable");
    let last = mutated.len() - 1;
    mutated[last] = mutated[last].wrapping_add(1);
    fs::write(&path, &mutated).expect("the pinned file is writable");
    (path, identity)
}

/// The pinned name now denotes a different file of the same length: a new
/// inode, not an edit of the old one.
fn prepare_path_replacement(root: &Path) -> (PathBuf, FileIdentity) {
    let (path, identity) = pinned_rank(root, 0, 512);
    fs::remove_file(&path).expect("the pinned file is removable");
    fs::write(&path, pinned_bytes(7, 512)).expect("the replacement file is writable");
    (path, identity)
}

/// The fixture draft, with one component staged from each pinned file on disk.
fn authenticated_draft(sources: &[(PathBuf, FileIdentity)]) -> BundleDraftV1 {
    let mut components: Vec<_> = sources
        .iter()
        .enumerate()
        .map(|(index, (source, identity))| {
            ComponentDraftV1::new(
                id(&format!("fixture.pinned-{index}")),
                id(&format!("fixture.pinned-{index}.v1")),
                id("fixture-pinned-bytes-v1"),
                path(&format!("components/pinned-{index}.bin")),
                ComponentContentV1::Whole(StagedContentV1::authenticated_file(
                    source,
                    identity.clone(),
                )),
            )
        })
        .collect();
    components.push(ComponentDraftV1::new(
        id("fixture.graph-plan"),
        id("fixture.graph-plan.v1"),
        id("fixture-graph-plan-json-v1"),
        path("components/graph-plan.json"),
        ComponentContentV1::Whole(StagedContentV1::bytes(PLAN_BYTES)),
    ));
    let bucket = BundleGraphBucketV1::new(
        id("fixture-bucket"),
        components
            .iter()
            .map(|component| component.id().clone())
            .collect(),
        Vec::new(),
    )
    .expect("the fixture bucket is valid");
    BundleDraftV1::new(
        id("fixture.authenticated.v1"),
        BundleTargetV1::new(
            vec![id(ARCHITECTURE)],
            vec![BundleTopologyV1::new(id("tp2"), 2).expect("tp2 is valid")],
        )
        .expect("the fixture target is valid"),
        BundleResourceRequirementsV1::new(1, Vec::new()).expect("the fixture resources are valid"),
        BundleKvLayoutV1::new(
            id("fixture.kv.v1"),
            64,
            64,
            1,
            64,
            1,
            1_024,
            BundleRankPlacementV1::RankLocal,
            BundleStateLifetimeV1::Request,
        )
        .expect("the fixture KV layout is valid"),
        BundleStateLayoutV1::new(
            id("fixture.state.v1"),
            64,
            64,
            BundleStateLifetimeV1::Request,
        )
        .expect("the fixture state layout is valid"),
    )
    .with_components(components)
    .with_graph_buckets(vec![bucket])
}

fn replace_plan_content(draft: BundleDraftV1, content: StagedContentV1) -> BundleDraftV1 {
    let components = draft
        .components()
        .iter()
        .map(|component| {
            if component.id().as_str() == "fixture.graph-plan" {
                ComponentDraftV1::new(
                    id("fixture.graph-plan"),
                    id("fixture.graph-plan.v1"),
                    id("fixture-graph-plan-json-v1"),
                    path("components/graph-plan.json"),
                    ComponentContentV1::Whole(content.clone()),
                )
            } else {
                component.clone()
            }
        })
        .collect();
    draft.with_components(components)
}

fn files_under(root: &Path) -> BTreeSet<String> {
    let mut files = BTreeSet::new();
    let mut pending = vec![PathBuf::new()];
    while let Some(relative) = pending.pop() {
        for entry in fs::read_dir(root.join(&relative)).expect("the emitted directory is readable")
        {
            let entry = entry.expect("the emitted entry is readable");
            let child = relative.join(entry.file_name());
            if entry.path().is_dir() {
                pending.push(child);
            } else {
                files.insert(child.to_string_lossy().into_owned());
            }
        }
    }
    files
}

fn staging_entries(root: &Path) -> Vec<String> {
    fs::read_dir(root)
        .expect("the scratch directory is readable")
        .map(|entry| {
            entry
                .expect("the scratch entry is readable")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter(|name| name.contains(".compiling-"))
        .collect()
}

fn compile_native_module(root: &Path, symbol: &str) -> PathBuf {
    let source = root.join(format!("{symbol}.rs"));
    fs::write(
        &source,
        format!(
            r#"
#[unsafe(no_mangle)]
pub extern "C" fn {symbol}() -> *const core::ffi::c_void {{
    static API: u8 = 0;
    &API as *const u8 as *const core::ffi::c_void
}}
"#
        ),
    )
    .expect("the native fixture source is writable");
    let module = root.join(format!("{symbol}-{}", native_module_filename()));
    let compiler = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let output = Command::new(compiler)
        .args(["--edition=2024", "--crate-type=cdylib"])
        .arg(&source)
        .arg("-o")
        .arg(&module)
        .output()
        .expect("the native fixture compiler starts");
    assert!(
        output.status.success(),
        "the native fixture compiler failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    module
}

fn native_module_filename() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "fixture.bundle.dylib"
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        "fixture.bundle.so"
    }
    #[cfg(windows)]
    {
        "fixture.bundle.dll"
    }
}

struct TemporaryDirectory {
    path: PathBuf,
}

impl TemporaryDirectory {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);

        let path = std::env::temp_dir().join(format!(
            "{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("the scratch directory is creatable");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// A fixture draft for one topology, whose minimum device memory is its rank
/// count, whose buckets run that many sequences and whose plan, of one role
/// in every topology, serves that topology alone.
fn topology_draft(topology: &str, ranks: u32, weights: &[u8]) -> BundleDraftV1 {
    let components = vec![
        ComponentDraftV1::new(
            id("fixture.weights"),
            id("fixture.weights.v1"),
            id("fixture-weight-bytes-v1"),
            path("components/weights.bin"),
            ComponentContentV1::Whole(StagedContentV1::bytes(weights.to_vec())),
        ),
        ComponentDraftV1::new(
            id(&format!("fixture.plan.{topology}")),
            id("fixture.graph-plan.v1"),
            id("fixture-graph-plan-json-v1"),
            path(&format!("components/plan-{topology}.json")),
            ComponentContentV1::Whole(StagedContentV1::bytes(
                format!(r#"{{"ranks":{ranks}}}"#).into_bytes(),
            )),
        )
        .with_topologies(vec![id(topology)]),
    ];
    let buckets = (0..=ranks.ilog2())
        .map(|power| {
            BundleGraphBucketV1::new(
                id(&format!("decode-b{}", 1 << power)),
                components.iter().map(|c| c.id().clone()).collect(),
                Vec::new(),
            )
            .expect("the fixture bucket is valid")
        })
        .collect();
    BundleDraftV1::new(
        id("fixture.topologies.v1"),
        BundleTargetV1::new(
            vec![id(ARCHITECTURE)],
            vec![BundleTopologyV1::new(id(topology), ranks).expect("the topology is valid")],
        )
        .expect("the fixture target is valid"),
        BundleResourceRequirementsV1::new(u64::from(ranks), Vec::new())
            .expect("the fixture resources are valid"),
        BundleKvLayoutV1::new(
            id("fixture.kv.v1"),
            64 * u64::from(ranks),
            64,
            1,
            64 * u64::from(ranks),
            1,
            1_024,
            BundleRankPlacementV1::RankLocal,
            BundleStateLifetimeV1::Request,
        )
        .expect("the fixture KV layout is valid"),
        BundleStateLayoutV1::new(
            id("fixture.state.v1"),
            64,
            64,
            BundleStateLifetimeV1::Request,
        )
        .expect("the fixture state layout is valid"),
    )
    .with_components(components)
    .with_graph_buckets(buckets)
}

/// Drafts of one bundle for several topologies join into one package: the
/// first topology in ID order keeps its resources as the manifest's own,
/// each other topology serves with its own, and drafts that differ beyond
/// their resources are refused.
#[test]
fn drafts_for_several_topologies_join_into_one_package() {
    let weights = weight_bytes();
    let joined = BundleDraftV1::for_topologies(vec![
        topology_draft("tp8", 8, &weights),
        topology_draft("tp2", 2, &weights),
        topology_draft("tp4", 4, &weights),
    ])
    .expect("the drafts join");
    let root = TemporaryDirectory::new("model-compiler-sdk-topologies");
    let output = root.path().join("bundle");
    let emitted = emit_bundle_directory_v1(&joined, &output).expect("the joined bundle emits");
    let manifest = emitted.manifest();
    assert_eq!(
        manifest
            .target()
            .topologies()
            .iter()
            .map(|topology| topology.id().as_str())
            .collect::<Vec<_>>(),
        ["tp2", "tp4", "tp8"]
    );
    for (topology, ranks) in [("tp2", 2_u32), ("tp4", 4), ("tp8", 8)] {
        let selected = manifest
            .for_topology(topology)
            .expect("each topology selects");
        let alone = plan_bundle_manifest_v1(&topology_draft(topology, ranks, &weights))
            .expect("each topology's own draft plans");
        assert_eq!(selected, alone, "{topology}");
    }
    assert_eq!(manifest.components().len(), 4);
    let package =
        BundlePackageV1::verify_directory(&output, &host()).expect("the package verifies");
    assert_eq!(package.manifest(), manifest);

    let other_weights = [weights.as_slice(), b"!"].concat();
    let error = BundleDraftV1::for_topologies(vec![
        topology_draft("tp2", 2, &weights),
        topology_draft("tp4", 4, &other_weights),
    ])
    .expect_err("drafts with other components are refused")
    .to_string();
    assert!(
        error.contains("differs from the first beyond its resources"),
        "{error}"
    );
    assert!(BundleDraftV1::for_topologies(Vec::new()).is_err());
    assert!(
        BundleDraftV1::for_topologies(vec![joined, topology_draft("tp1", 1, &weights)]).is_err()
    );
}

/// The BF16 layout of `topology_draft` with `ranks`, and a mixed-plane one
/// of the same pages: 32 FP8 values and their scale, then 32 FP4 values and
/// their two scales, in 64 bytes per token.
fn mixed_plane_formats(ranks: u32) -> Vec<BundleKvFormatV1> {
    use runtime_bundle::{BundleKvCodeV1, BundleKvPlaneV1, BundleKvScaleFormatV1, BundleKvScaleV1};
    let layout = |name: &str, bytes: u64| {
        BundleKvLayoutV1::new(
            id(name),
            bytes,
            64,
            1,
            bytes,
            1,
            1_024,
            BundleRankPlacementV1::RankLocal,
            BundleStateLifetimeV1::Request,
        )
        .expect("the fixture KV layout is valid")
    };
    let plane = |name: &str, code, offset, scale| {
        BundleKvPlaneV1::new(id(name), code, 32, offset, Some(scale))
            .expect("the fixture plane is valid")
    };
    let scale = |format, block, offset| {
        BundleKvScaleV1::new(format, block, offset).expect("the fixture scale is valid")
    };
    let planes = vec![
        plane(
            "window",
            BundleKvCodeV1::Fp8E4m3,
            0,
            scale(BundleKvScaleFormatV1::Ue8m0, 32, 32),
        ),
        plane(
            "compressed",
            BundleKvCodeV1::Fp4E2m1,
            40,
            scale(BundleKvScaleFormatV1::E4m3, 16, 56),
        ),
    ];
    vec![
        BundleKvFormatV1::new(
            BundleKvDtypeV1::Bf16,
            layout("fixture.kv.v1", 64 * u64::from(ranks)),
        ),
        BundleKvFormatV1::mixed_planes(planes, layout("fixture.kv-native.v1", 64))
            .expect("the fixture planes fit their layout"),
    ]
}

/// A draft that declares a mixed-plane KV element emits a package requiring
/// its capability: the emitter declares the two together, whether or not the
/// draft states the capability, for one topology and for each topology of a
/// joined draft, and the package verifies on a host that has it. A draft
/// that requires the capability with no such element is refused.
#[test]
fn a_mixed_plane_kv_element_is_emitted_with_its_capability() {
    let capability = id(MIXED_PLANE_KV_CAPABILITY_V1);
    let weights = weight_bytes();
    let mixed = |topology: &str, ranks: u32| {
        topology_draft(topology, ranks, &weights).with_kv_formats(mixed_plane_formats(ranks))
    };
    let stating = mixed("tp1", 1).with_resource_requirements(
        BundleResourceRequirementsV1::new(1, vec![capability.clone()])
            .expect("the fixture resources are valid"),
    );
    let root = TemporaryDirectory::new("model-compiler-sdk-mixed-planes");
    for (name, draft) in [("implied", mixed("tp1", 1)), ("stated", stating)] {
        let output = root.path().join(name);
        let emitted = emit_bundle_directory_v1(&draft, &output).expect("the draft emits");
        let manifest = emitted.manifest();
        assert_eq!(
            manifest.resource_requirements().required_capabilities(),
            std::slice::from_ref(&capability),
            "{name}"
        );
        assert_eq!(
            manifest.kv_formats()[1].dtype(),
            BundleKvDtypeV1::MixedPlanes
        );
        let error = BundlePackageV1::verify_directory(&output, &host())
            .expect_err("a host without the capability refuses the package")
            .to_string();
        assert!(error.contains(MIXED_PLANE_KV_CAPABILITY_V1), "{error}");
        let package = BundlePackageV1::verify_directory(
            &output,
            &host().with_capabilities([capability.clone()]),
        )
        .expect("a host with the capability verifies the package");
        assert_eq!(package.manifest(), manifest);
    }

    let joined = BundleDraftV1::for_topologies(vec![mixed("tp2", 2), mixed("tp4", 4)])
        .expect("the drafts join");
    let output = root.path().join("joined");
    let emitted = emit_bundle_directory_v1(&joined, &output).expect("the joined draft emits");
    for topology in ["tp2", "tp4"] {
        let selected = emitted
            .manifest()
            .for_topology(topology)
            .expect("each topology selects");
        assert_eq!(
            selected.resource_requirements().required_capabilities(),
            std::slice::from_ref(&capability),
            "{topology}"
        );
        assert_eq!(selected.kv_formats().len(), 2, "{topology}");
    }

    let stray = topology_draft("tp1", 1, &weights).with_resource_requirements(
        BundleResourceRequirementsV1::new(1, vec![capability])
            .expect("the fixture resources are valid"),
    );
    let error = emit_bundle_directory_v1(&stray, &root.path().join("stray"))
        .expect_err("the capability without its element is refused")
        .to_string();
    assert!(error.contains(MIXED_PLANE_KV_CAPABILITY_V1), "{error}");
}

fn model_options_draft(default: i64, qualified: bool) -> BundleDraftV1 {
    use runtime_bundle::model_options::{
        CAPABILITY, ModelOptionV1, ModelOptionsV1, OptionDefinitionV1,
    };
    draft(None)
        .with_model_options(ModelOptionsV1 {
            schema: CAPABILITY.into(),
            options: vec![ModelOptionV1 {
                name: "budget".into(),
                definition: OptionDefinitionV1::Integer {
                    default,
                    minimum: 1,
                    maximum: 4,
                },
                default_is_qualified: qualified,
            }],
        })
        .unwrap()
}

#[test]
fn model_options_native_declarations_refuse_conflicting_defaults_and_qualification() {
    let configured = model_options_draft(2, false);
    for native in [
        model_options_draft(1, true),
        model_options_draft(1, false),
        model_options_draft(2, true),
    ] {
        assert!(
            configured
                .clone()
                .with_native_declarations(&native.native_declarations())
                .is_err()
        );
    }
}

#[test]
fn model_options_native_declarations_refuse_missing_options_on_either_side() {
    let configured = model_options_draft(2, false);
    let legacy = draft(None);
    assert!(
        configured
            .clone()
            .with_native_declarations(&legacy.native_declarations())
            .is_err()
    );
    assert!(
        legacy
            .with_native_declarations(&configured.native_declarations())
            .is_err()
    );
}

#[test]
fn model_options_native_declarations_preserve_matching_and_legacy_values() {
    for configured in [
        model_options_draft(2, false),
        model_options_draft(1, true),
        draft(None),
    ] {
        let declaration = configured.native_declarations();
        let bound = configured.with_native_declarations(&declaration).unwrap();
        assert_eq!(bound.native_declarations(), declaration);
    }
}
