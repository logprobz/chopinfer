//! The catalog check a bundle passes before its native module is opened.
//!
//! Every manifest here is a well-formed, verifiable V1 declaration: nothing is
//! made invalid to be refused. What differs is the operation each one requires,
//! and whether this host's own catalog holds an exact implementation of it that
//! the bundle's own provisioning allows. A bundle that fails this check is
//! refused while it is still data, which is why the check runs before the
//! loader is handed a module image.

use runtime_bundle::{
    BundleArtifactIdentityV1, BundleBuildIdentityV1, BundleComponentV1, BundleError,
    BundleGraphBucketV1, BundleImplementationDetailV1, BundleImplementationTierV1,
    BundleImplementationV1, BundleKvLayoutV1, BundleLaunchContractV1, BundleManifestV1,
    BundleOpaqueId, BundleOperationKeyV1, BundleOperationProvisioningV1,
    BundleOperationRequirementV1, BundleRankPlacementV1, BundleRelativePathV1,
    BundleResourceRequirementsV1, BundleSemanticContractV1, BundleSourceIdentityV1,
    BundleStateLayoutV1, BundleStateLifetimeV1, BundleTargetV1, BundleTopologyV1,
    GENERIC_COPY_F32_PRIMITIVE_V1, generic_copy_f32_contract_v1,
    generic_copy_f32_implementation_id_v1, generic_copy_f32_key_v1,
};
use runtime_host::{
    HOST_GENERIC_COPY_F32_MAX_ELEMENTS_V1, HostGenericCatalogV1, HostLoadErrorV1,
    resolve_declared_operations_for_architecture_v1, resolve_declared_operations_v1,
};
use runtime_kernel_contract::KernelImplementationTierV1;

/// A width this host serves, and the one the recurrent fixture requires.
const SERVED_WIDTH: u32 = 64;
/// The other architecture the multi-architecture fixtures declare.
const OTHER_ARCHITECTURE: &str = "sm_90a";

const DIGEST: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const ARCHITECTURE: &str = "sm_120f";
const COMPONENT: &str = "program";
const IMPLEMENTATION: &str = "fixture.catalog.implementation.v1";

fn id(value: &str) -> BundleOpaqueId {
    BundleOpaqueId::new(value).expect("the test identifier is valid")
}

fn catalog() -> HostGenericCatalogV1 {
    HostGenericCatalogV1::builtin_v1().expect("this host's catalog is valid")
}

/// A manifest requiring one operation, with or without an implementation.
///
/// Everything else is the smallest well-formed declaration a V1 manifest can
/// carry, so the only thing any test below varies is the operation itself.
fn manifest_requiring(
    key: BundleOperationKeyV1,
    provisioning: BundleOperationProvisioningV1,
    contract: BundleSemanticContractV1,
    implement_here: bool,
) -> Result<BundleManifestV1, BundleError> {
    manifest_requiring_on(
        &[ARCHITECTURE],
        key,
        provisioning,
        contract,
        if implement_here {
            &[(IMPLEMENTATION, ARCHITECTURE, "fixture.catalog.symbol.v1")]
        } else {
            &[]
        },
    )
}

/// The same manifest on the architectures named, with one implementation per
/// (identifier, architecture, symbol) record.
fn manifest_requiring_on(
    architectures: &[&str],
    key: BundleOperationKeyV1,
    provisioning: BundleOperationProvisioningV1,
    contract: BundleSemanticContractV1,
    implementations: &[(&str, &str, &str)],
) -> Result<BundleManifestV1, BundleError> {
    let component = BundleComponentV1::new(
        id(COMPONENT),
        id("fixture.catalog.program.v1"),
        id("fixture.catalog.bytes.v1"),
        BundleRelativePathV1::new("components/program.bin")?,
        16,
        DIGEST,
    )?;
    let implementations = implementations
        .iter()
        .map(|(identifier, architecture, symbol)| {
            BundleImplementationV1::new(
                id(identifier),
                key.clone(),
                BundleImplementationTierV1::Bundle,
                BundleImplementationDetailV1::new(
                    BundleArtifactIdentityV1::from_component(
                        id(COMPONENT),
                        id("fixture.catalog.bytes.v1"),
                        16,
                        DIGEST,
                    )?,
                    id(symbol),
                    id(architecture),
                    BundleLaunchContractV1::new([1, 1, 1], [1, 1, 1], 0)?,
                    None,
                    Vec::new(),
                    BundleSourceIdentityV1::new(
                        id("fixture.catalog"),
                        id("0.1.0"),
                        id("src.catalog.rs"),
                        16,
                        DIGEST,
                    )?,
                    BundleBuildIdentityV1::new(id("rustc"), id("1.0.0"), id("cargo"), DIGEST)?,
                )?,
            )
        })
        .collect::<Result<Vec<_>, BundleError>>()?;
    BundleManifestV1::new(
        id("fixture.catalog.v1"),
        BundleTargetV1::new(
            architectures.iter().map(|value| id(value)).collect(),
            vec![BundleTopologyV1::new(id("tp1"), 1)?],
        )?,
        vec![component],
        BundleResourceRequirementsV1::new(1, Vec::new())?,
        BundleKvLayoutV1::new(
            id("fixture.catalog.kv.v1"),
            32,
            32,
            8,
            4,
            1,
            64,
            BundleRankPlacementV1::RankLocal,
            BundleStateLifetimeV1::Transaction,
        )?,
        BundleStateLayoutV1::new(
            id("fixture.catalog.state.v1"),
            64,
            64,
            BundleStateLifetimeV1::Process,
        )?,
        vec![BundleGraphBucketV1::new(
            id("bucket"),
            vec![id(COMPONENT)],
            vec![key.clone()],
        )?],
        vec![BundleOperationRequirementV1::new(
            key,
            provisioning,
            contract,
        )?],
        implementations,
        None,
    )
}

/// A bundle that leaves the copy to the host at a width this host serves.
fn generic_copy_manifest(elements: u32) -> BundleManifestV1 {
    manifest_requiring(
        generic_copy_f32_key_v1().expect("the shared copy key is valid"),
        BundleOperationProvisioningV1::HostGenericPrimitive(id(GENERIC_COPY_F32_PRIMITIVE_V1)),
        generic_copy_f32_contract_v1(elements).expect("the shared copy contract is valid"),
        false,
    )
    .expect("the test manifest is a well-formed V1 declaration")
}

/// The declared generic copy resolves through the host tier, under the very
/// identity this host offers that primitive at that width.
#[test]
fn a_declared_generic_copy_resolves_to_this_host_catalog_entry() {
    let manifest = generic_copy_manifest(SERVED_WIDTH);

    let [resolved] = resolve_declared_operations_v1(&manifest, &catalog())
        .expect("this host serves the declared width")
        .try_into()
        .expect("one declared operation resolves to one implementation");

    assert_eq!(resolved.tier(), KernelImplementationTierV1::Generic);
    assert_eq!(
        resolved.provisioning(),
        &BundleOperationProvisioningV1::HostGenericPrimitive(id(GENERIC_COPY_F32_PRIMITIVE_V1))
    );
    assert_eq!(
        resolved.implementation().as_str(),
        generic_copy_f32_implementation_id_v1(SERVED_WIDTH)
            .expect("the shared identity is valid")
            .as_str()
    );
}

/// A width this host does not serve is a different operation, and there is no
/// candidate for it, so the bundle is refused rather than served approximately.
#[test]
fn a_width_this_host_does_not_serve_is_refused_before_the_module_is_opened() {
    for unserved in [
        SERVED_WIDTH - 1,
        SERVED_WIDTH + 1,
        HOST_GENERIC_COPY_F32_MAX_ELEMENTS_V1 + SERVED_WIDTH,
    ] {
        let manifest = generic_copy_manifest(unserved);

        let refusal = resolve_declared_operations_v1(&manifest, &catalog())
            .err()
            .unwrap_or_else(|| panic!("a copy of {unserved} values must be refused"));

        assert!(
            matches!(refusal, HostLoadErrorV1::UnsupportedOperation { .. }),
            "a copy of {unserved} values was refused as {refusal}"
        );
    }
}

/// A contract that differs in any field is a near miss, not a close enough
/// match: the same width under other numerics has no candidate either.
#[test]
fn a_near_miss_contract_is_refused_rather_than_matched_loosely() {
    let contract =
        generic_copy_f32_contract_v1(SERVED_WIDTH).expect("the shared copy contract is valid");
    let near_miss = BundleSemanticContractV1::new(
        contract.operation_abi().clone(),
        contract.operation_abi_version(),
        contract.tensors().to_vec(),
        contract.scalar_arguments().to_vec(),
        id("chop.numerics.other-copy.v1"),
        Vec::new(),
    )
    .expect("the near-miss contract is a well-formed declaration");
    let manifest = manifest_requiring(
        generic_copy_f32_key_v1().expect("the shared copy key is valid"),
        BundleOperationProvisioningV1::HostGenericPrimitive(id(GENERIC_COPY_F32_PRIMITIVE_V1)),
        near_miss,
        false,
    )
    .expect("the test manifest is a well-formed V1 declaration");

    let refusal = resolve_declared_operations_v1(&manifest, &catalog())
        .expect_err("a copy under other numerics is a different operation");

    assert!(matches!(
        refusal,
        HostLoadErrorV1::UnsupportedOperation { .. }
    ));
}

/// An exact contract match under a primitive this host does not offer it as is
/// still refused: matching the contract is not the same as being what was asked
/// for.
#[test]
fn an_exact_match_under_another_primitive_name_is_still_refused() {
    let manifest = manifest_requiring(
        generic_copy_f32_key_v1().expect("the shared copy key is valid"),
        BundleOperationProvisioningV1::HostGenericPrimitive(id("chop.generic.copy-f32.v2")),
        generic_copy_f32_contract_v1(SERVED_WIDTH).expect("the shared copy contract is valid"),
        false,
    )
    .expect("the test manifest is a well-formed V1 declaration");

    let refusal = resolve_declared_operations_v1(&manifest, &catalog())
        .expect_err("this host offers no primitive by that name");

    assert!(
        matches!(refusal, HostLoadErrorV1::WrongOperationProvider { .. }),
        "an unoffered primitive was refused as {refusal}"
    );
}

/// A custom operation is the bundle's, and only the bundle's.
///
/// The contract here is one this host's catalog holds an exact candidate for,
/// so the only thing keeping the host's copy from satisfying it is that the
/// bundle declared the operation private. Without its own implementation the
/// bundle is refused; with one, it resolves at the bundle tier and the host's
/// identical candidate is never chosen.
#[test]
fn a_bundle_private_operation_is_never_satisfied_by_this_host_catalog() {
    let key = BundleOperationKeyV1::new(id("fixture.catalog.private"), id("v1"));
    let contract =
        generic_copy_f32_contract_v1(SERVED_WIDTH).expect("the shared copy contract is valid");

    let unimplemented = manifest_requiring(
        key.clone(),
        BundleOperationProvisioningV1::BundleCustomOperation,
        contract.clone(),
        false,
    )
    .expect("the test manifest is a well-formed V1 declaration");
    let refusal = resolve_declared_operations_v1(&unimplemented, &catalog())
        .expect_err("a private operation the bundle does not implement is unsupported");
    assert!(
        matches!(refusal, HostLoadErrorV1::UnsupportedOperation { .. }),
        "a private operation was refused as {refusal}"
    );

    let implemented = manifest_requiring(
        key,
        BundleOperationProvisioningV1::BundleCustomOperation,
        contract,
        true,
    )
    .expect("the test manifest is a well-formed V1 declaration");
    let [resolved] = resolve_declared_operations_v1(&implemented, &catalog())
        .expect("the bundle's own implementation satisfies its own operation")
        .try_into()
        .expect("one declared operation resolves to one implementation");
    assert_eq!(resolved.tier(), KernelImplementationTierV1::Bundle);
    assert_eq!(
        resolved.implementation().as_str(),
        "fixture.catalog.implementation.v1"
    );
}

/// A bundle that declares no operations passes the check with nothing resolved,
/// because the check is over what the bundle declared rather than a fixed list.
#[test]
fn a_bundle_that_declares_no_operations_resolves_nothing_and_still_loads() {
    let component = BundleComponentV1::new(
        id(COMPONENT),
        id("fixture.catalog.program.v1"),
        id("fixture.catalog.bytes.v1"),
        BundleRelativePathV1::new("components/program.bin").expect("the test path is relative"),
        16,
        DIGEST,
    )
    .expect("the test component is valid");
    let manifest = BundleManifestV1::new(
        id("fixture.catalog.silent.v1"),
        BundleTargetV1::new(
            vec![id(ARCHITECTURE)],
            vec![BundleTopologyV1::new(id("tp1"), 1).expect("tp1 is valid")],
        )
        .expect("the test target is valid"),
        vec![component],
        BundleResourceRequirementsV1::new(1, Vec::new()).expect("the test requirements are valid"),
        BundleKvLayoutV1::new(
            id("fixture.catalog.kv.v1"),
            32,
            32,
            8,
            4,
            1,
            64,
            BundleRankPlacementV1::RankLocal,
            BundleStateLifetimeV1::Transaction,
        )
        .expect("the test KV layout is valid"),
        BundleStateLayoutV1::new(
            id("fixture.catalog.state.v1"),
            64,
            64,
            BundleStateLifetimeV1::Process,
        )
        .expect("the test state layout is valid"),
        vec![
            BundleGraphBucketV1::new(id("bucket"), vec![id(COMPONENT)], Vec::new())
                .expect("the test bucket is valid"),
        ],
        Vec::new(),
        Vec::new(),
        None,
    )
    .expect("the test manifest is a well-formed V1 declaration");

    assert!(
        resolve_declared_operations_v1(&manifest, &catalog())
            .expect("a bundle with no declared operation has nothing to refuse")
            .is_empty()
    );
}

/// A bundle that declares several CUDA architectures is resolved one
/// architecture at a time. There is no architecture-free resolution: one that
/// ignored architectures would select an implementation compiled for a device
/// this process is not running.
#[test]
fn a_manifest_declaring_several_architectures_is_resolved_per_architecture() {
    let manifest = manifest_requiring_on(
        &[ARCHITECTURE, OTHER_ARCHITECTURE],
        generic_copy_f32_key_v1().expect("the shared copy key is valid"),
        BundleOperationProvisioningV1::BundleCustomOperation,
        generic_copy_f32_contract_v1(SERVED_WIDTH).expect("the shared copy contract is valid"),
        &[(IMPLEMENTATION, ARCHITECTURE, "fixture.catalog.symbol.v1")],
    )
    .expect("the test manifest is a well-formed V1 declaration");

    let refusal = resolve_declared_operations_v1(&manifest, &catalog()).expect_err(
        "an architecture-free resolution must not select an implementation declared for another \
         architecture",
    );
    assert!(
        matches!(refusal, HostLoadErrorV1::Bundle(_)),
        "refused as {refusal}"
    );
}

/// Each declared architecture resolves to the implementation declared for it,
/// and to nothing else.
#[test]
fn the_selected_architecture_gets_the_implementation_declared_for_it() {
    let manifest = manifest_requiring_on(
        &[ARCHITECTURE, OTHER_ARCHITECTURE],
        generic_copy_f32_key_v1().expect("the shared copy key is valid"),
        BundleOperationProvisioningV1::BundleCustomOperation,
        generic_copy_f32_contract_v1(SERVED_WIDTH).expect("the shared copy contract is valid"),
        &[
            (
                "fixture.catalog.implementation.sm120f.v1",
                ARCHITECTURE,
                "fixture.catalog.symbol.sm120f.v1",
            ),
            (
                "fixture.catalog.implementation.sm90a.v1",
                OTHER_ARCHITECTURE,
                "fixture.catalog.symbol.sm90a.v1",
            ),
        ],
    )
    .expect("the test manifest is a well-formed V1 declaration");

    for (architecture, expected) in [
        (ARCHITECTURE, "fixture.catalog.implementation.sm120f.v1"),
        (
            OTHER_ARCHITECTURE,
            "fixture.catalog.implementation.sm90a.v1",
        ),
    ] {
        let [resolved] = resolve_declared_operations_for_architecture_v1(
            &manifest,
            &catalog(),
            &id(architecture),
        )
        .expect("each declared architecture has its own implementation")
        .try_into()
        .expect("one declared operation resolves to one implementation");
        assert_eq!(resolved.tier(), KernelImplementationTierV1::Bundle);
        assert_eq!(
            resolved.implementation().as_str(),
            expected,
            "{architecture} must run its own implementation"
        );
    }
}

/// An implementation declared for another architecture is not a fallback: the
/// architecture without its own implementation is unsupported, and the error
/// says which architectures the bundle did declare it for.
#[test]
fn an_implementation_for_another_architecture_never_satisfies_this_host() {
    let manifest = manifest_requiring_on(
        &[ARCHITECTURE, OTHER_ARCHITECTURE],
        generic_copy_f32_key_v1().expect("the shared copy key is valid"),
        BundleOperationProvisioningV1::BundleCustomOperation,
        generic_copy_f32_contract_v1(SERVED_WIDTH).expect("the shared copy contract is valid"),
        &[(IMPLEMENTATION, ARCHITECTURE, "fixture.catalog.symbol.v1")],
    )
    .expect("the test manifest is a well-formed V1 declaration");

    let refusal = resolve_declared_operations_for_architecture_v1(
        &manifest,
        &catalog(),
        &id(OTHER_ARCHITECTURE),
    )
    .expect_err("the sm_90a host has no implementation of its own for this operation");
    assert!(
        matches!(&refusal, HostLoadErrorV1::UnsupportedOperation { reason, .. }
            if reason.contains(OTHER_ARCHITECTURE) && reason.contains(ARCHITECTURE)),
        "the refusal must name both architectures: {refusal}"
    );

    // The architecture that does declare it still resolves exactly.
    let [resolved] =
        resolve_declared_operations_for_architecture_v1(&manifest, &catalog(), &id(ARCHITECTURE))
            .expect("the declaring architecture resolves")
            .try_into()
            .expect("one declared operation resolves to one implementation");
    assert_eq!(resolved.implementation().as_str(), IMPLEMENTATION);
}

/// Resolving for an architecture the bundle does not declare is refused
/// outright, before any tier is resolved.
#[test]
fn an_undeclared_architecture_is_refused_before_any_tier_is_resolved() {
    let manifest = generic_copy_manifest(SERVED_WIDTH);

    let refusal = resolve_declared_operations_for_architecture_v1(
        &manifest,
        &catalog(),
        &id(OTHER_ARCHITECTURE),
    )
    .expect_err("the fixture declares only sm_120f");

    assert!(
        matches!(refusal, HostLoadErrorV1::Bundle(_)),
        "refused as {refusal}"
    );
}

/// A host primitive is offered to every declared architecture.
///
/// The host compiled its generic catalog for itself, so selecting it for
/// another declared architecture is exact resolution, not a cross-architecture
/// fallback.
#[test]
fn a_host_primitive_is_offered_to_every_declared_architecture() {
    let manifest = manifest_requiring_on(
        &[ARCHITECTURE, OTHER_ARCHITECTURE],
        generic_copy_f32_key_v1().expect("the shared copy key is valid"),
        BundleOperationProvisioningV1::HostGenericPrimitive(id(GENERIC_COPY_F32_PRIMITIVE_V1)),
        generic_copy_f32_contract_v1(SERVED_WIDTH).expect("the shared copy contract is valid"),
        &[],
    )
    .expect("the test manifest is a well-formed V1 declaration");

    for architecture in [ARCHITECTURE, OTHER_ARCHITECTURE] {
        let [resolved] = resolve_declared_operations_for_architecture_v1(
            &manifest,
            &catalog(),
            &id(architecture),
        )
        .expect("the host's own generic tier is not architecture-specific")
        .try_into()
        .expect("one declared operation resolves to one implementation");
        assert_eq!(
            resolved.tier(),
            KernelImplementationTierV1::Generic,
            "{architecture}"
        );
        assert_eq!(
            resolved.implementation().as_str(),
            generic_copy_f32_implementation_id_v1(SERVED_WIDTH)
                .expect("the shared identity is valid")
                .as_str(),
            "{architecture}"
        );
    }
}

/// Construction normalizes implementations into canonical order: operation,
/// tier, target architecture, and then identifier. Here the identifiers sort
/// opposite to the architectures, and the architecture wins.
#[test]
fn construction_orders_implementations_by_operation_tier_and_architecture() {
    let manifest = manifest_requiring_on(
        &[ARCHITECTURE, OTHER_ARCHITECTURE],
        generic_copy_f32_key_v1().expect("the shared copy key is valid"),
        BundleOperationProvisioningV1::BundleCustomOperation,
        generic_copy_f32_contract_v1(SERVED_WIDTH).expect("the shared copy contract is valid"),
        &[
            (
                "fixture.catalog.implementation.a.v1",
                OTHER_ARCHITECTURE,
                "fixture.catalog.symbol.a.v1",
            ),
            (
                "fixture.catalog.implementation.b.v1",
                ARCHITECTURE,
                "fixture.catalog.symbol.b.v1",
            ),
        ],
    )
    .expect("construction accepts any record order");

    let order: Vec<_> = manifest
        .implementations()
        .iter()
        .map(|implementation| implementation.id().as_str())
        .collect();
    assert_eq!(
        order,
        [
            "fixture.catalog.implementation.b.v1",
            "fixture.catalog.implementation.a.v1",
        ],
        "the sm_120f record precedes the sm_90a record although its identifier sorts after"
    );
    // The normalized order is what the canonical parser requires on disk.
    let bytes = manifest
        .to_canonical_json()
        .expect("the manifest serializes");
    BundleManifestV1::from_canonical_json(&bytes)
        .expect("the normalized order is the canonical one");
}

/// Global implementation identity stays unique: two records may not share one
/// identifier even when they qualify different architectures.
#[test]
fn two_implementations_may_not_share_one_identifier() {
    let error = manifest_requiring_on(
        &[ARCHITECTURE, OTHER_ARCHITECTURE],
        generic_copy_f32_key_v1().expect("the shared copy key is valid"),
        BundleOperationProvisioningV1::BundleCustomOperation,
        generic_copy_f32_contract_v1(SERVED_WIDTH).expect("the shared copy contract is valid"),
        &[
            (IMPLEMENTATION, ARCHITECTURE, "fixture.catalog.symbol.a.v1"),
            (
                IMPLEMENTATION,
                OTHER_ARCHITECTURE,
                "fixture.catalog.symbol.b.v1",
            ),
        ],
    )
    .expect_err("one identifier cannot name two implementations");

    let error = error.to_string();
    assert!(
        error.contains(IMPLEMENTATION),
        "the refusal must name the duplicated identifier: {error}"
    );
}
