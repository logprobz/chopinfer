use std::collections::BTreeSet;

use runtime_kernel_contract::{
    KernelImplementationCandidateV1, KernelImplementationTierV1, KernelOpaqueIdV1,
    KernelOperationAbiV1, KernelResolverV1, KernelScalarArgumentV1, KernelSemanticKeyPartsV1,
    KernelSemanticKeyV1, KernelTensorRequirementV1,
};

#[test]
fn bundle_implementation_wins_over_host_optimized_and_generic() {
    let key = key();
    let resolver = resolver([
        candidate(KernelImplementationTierV1::Generic, "generic", &key),
        candidate(KernelImplementationTierV1::HostOptimized, "host", &key),
        candidate(KernelImplementationTierV1::Bundle, "bundle", &key),
    ]);

    let resolution = resolver.resolve(&key).expect("fixture key resolves");
    assert_eq!(resolution.tier(), KernelImplementationTierV1::Bundle);
    assert_eq!(resolution.implementation_id().as_str(), "bundle");
}

#[test]
fn host_optimized_implementation_wins_when_the_bundle_has_no_match() {
    let key = key();
    let resolver = resolver([
        candidate(KernelImplementationTierV1::Generic, "generic", &key),
        candidate(KernelImplementationTierV1::HostOptimized, "host", &key),
    ]);

    let resolution = resolver.resolve(&key).expect("fixture key resolves");
    assert_eq!(resolution.tier(), KernelImplementationTierV1::HostOptimized);
    assert_eq!(resolution.implementation_id().as_str(), "host");
}

#[test]
fn generic_implementation_wins_when_higher_tiers_have_no_match() {
    let key = key();
    let resolver = resolver([candidate(
        KernelImplementationTierV1::Generic,
        "generic",
        &key,
    )]);

    let resolution = resolver.resolve(&key).expect("fixture key resolves");
    assert_eq!(resolution.tier(), KernelImplementationTierV1::Generic);
    assert_eq!(resolution.implementation_id().as_str(), "generic");
}

#[test]
fn unsupported_is_reported_when_no_tier_has_an_exact_implementation() {
    let resolver = resolver([]);

    assert!(resolver.resolve(&key()).is_err());
}

#[test]
fn duplicate_exact_matches_are_rejected_inside_each_tier() {
    let key = key();

    for tier in [
        KernelImplementationTierV1::Bundle,
        KernelImplementationTierV1::HostOptimized,
        KernelImplementationTierV1::Generic,
    ] {
        assert!(
            KernelResolverV1::new([
                candidate(tier, "first", &key),
                candidate(tier, "second", &key),
            ])
            .is_err(),
            "{tier:?} accepts duplicate exact implementations"
        );
    }
}

fn resolver(
    candidates: impl IntoIterator<Item = KernelImplementationCandidateV1>,
) -> KernelResolverV1 {
    KernelResolverV1::new(candidates).expect("fixture resolver is unambiguous")
}

fn candidate(
    tier: KernelImplementationTierV1,
    implementation_id: &str,
    key: &KernelSemanticKeyV1,
) -> KernelImplementationCandidateV1 {
    KernelImplementationCandidateV1::new(id(implementation_id), key.clone(), tier)
        .expect("fixture implementation candidate is valid")
}

fn key() -> KernelSemanticKeyV1 {
    KernelSemanticKeyV1::new(KernelSemanticKeyPartsV1 {
        operation_abi: KernelOperationAbiV1::new(id("fixture.operation.v1"), 1)
            .expect("fixture operation ABI is valid"),
        tensors: vec![
            KernelTensorRequirementV1::new(
                id("input"),
                vec![1, 4],
                id("bf16"),
                id("row-major.v1"),
                id("scale.none.v1"),
                16,
            )
            .expect("fixture input requirement is valid"),
            KernelTensorRequirementV1::new(
                id("output"),
                vec![1, 4],
                id("bf16"),
                id("row-major.v1"),
                id("scale.none.v1"),
                16,
            )
            .expect("fixture output requirement is valid"),
        ],
        scalar_arguments: vec![
            KernelScalarArgumentV1::new(id("rank"), id("u32"), 32)
                .expect("fixture rank scalar is valid"),
        ],
        numerical_contract: id("fixture.numerical.v1"),
        required_capabilities: BTreeSet::from([id("tensor-memory.v1")]),
    })
    .expect("fixture semantic key is valid")
}

fn id(value: &str) -> KernelOpaqueIdV1 {
    KernelOpaqueIdV1::new(value).expect("fixture opaque ID is valid")
}
