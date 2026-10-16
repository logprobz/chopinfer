use std::collections::BTreeSet;

use runtime_kernel_contract::{
    KERNEL_UNSUPPORTED_IMPLEMENTATION_ID, KernelAliasContractV1, KernelArtifactIdentityV1,
    KernelBuildIdentityV1, KernelImplementationCandidateV1, KernelImplementationDetailPartsV1,
    KernelImplementationDetailV1, KernelImplementationTierV1, KernelLaunchContractV1,
    KernelOpaqueIdV1, KernelOperationAbiV1, KernelQualificationKindV1, KernelResolutionErrorV1,
    KernelResolverV1, KernelScalarArgumentV1, KernelSemanticErrorV1, KernelSemanticKeyPartsV1,
    KernelSemanticKeyV1, KernelSha256V1, KernelSourceIdentityV1, KernelSpecializationV1,
    KernelTensorAccessV1, KernelTensorDependencyV1, KernelTensorRequirementV1,
    KernelWorkspaceContractV1,
};

type TensorMutation = (
    &'static str,
    Box<dyn Fn(KernelTensorRequirementV1) -> KernelTensorRequirementV1>,
);

fn id(value: &str) -> KernelOpaqueIdV1 {
    KernelOpaqueIdV1::new(value).unwrap()
}

fn digest(byte: char) -> KernelSha256V1 {
    KernelSha256V1::new(byte.to_string().repeat(64)).unwrap()
}

fn artifact(format: &str, byte_length: u64, digest_byte: char) -> KernelArtifactIdentityV1 {
    KernelArtifactIdentityV1::new(id(format), byte_length, digest(digest_byte)).unwrap()
}

fn source(
    source_id: &str,
    revision: &str,
    path: &str,
    byte_length: u64,
    digest_byte: char,
) -> KernelSourceIdentityV1 {
    KernelSourceIdentityV1::new(
        id(source_id),
        id(revision),
        id(path),
        byte_length,
        digest(digest_byte),
    )
    .unwrap()
}

fn build(
    toolchain_id: &str,
    version: &str,
    build_id: &str,
    digest_byte: char,
) -> KernelBuildIdentityV1 {
    KernelBuildIdentityV1::new(
        id(toolchain_id),
        id(version),
        id(build_id),
        digest(digest_byte),
    )
}

fn input() -> KernelTensorRequirementV1 {
    KernelTensorRequirementV1::new(
        id("input"),
        vec![32, 4096],
        id("bf16"),
        id("row-major.v1"),
        id("scale.none.v1"),
        16,
    )
    .unwrap()
}

fn weight() -> KernelTensorRequirementV1 {
    KernelTensorRequirementV1::new(
        id("weight"),
        vec![4096, 4096],
        id("fp8-e4m3"),
        id("column-major.v1"),
        id("ue8m0-block.v1"),
        32,
    )
    .unwrap()
    .with_scale_block_dimensions(vec![1, 32])
    .unwrap()
}

fn output() -> KernelTensorRequirementV1 {
    KernelTensorRequirementV1::new(
        id("output"),
        vec![32, 4096],
        id("bf16"),
        id("row-major.v1"),
        id("scale.none.v1"),
        16,
    )
    .unwrap()
    .with_access(
        KernelTensorAccessV1::WriteOnly,
        KernelAliasContractV1::AliasGroup(id("activation-arena.v1")),
    )
    .with_dependencies(vec![KernelTensorDependencyV1::new(0, id("reads-after.v1"))])
}

fn sample_key() -> KernelSemanticKeyV1 {
    KernelSemanticKeyV1::new(KernelSemanticKeyPartsV1 {
        operation_abi: KernelOperationAbiV1::new(id("custom.prefill.projection"), 1).unwrap(),
        tensors: vec![input(), weight(), output()],
        scalar_arguments: vec![
            KernelScalarArgumentV1::new(id("stream-rank"), id("unsigned-int"), 32).unwrap(),
        ],
        numerical_contract: id("model-supplied-numerics.opaque.v1"),
        required_capabilities: BTreeSet::from([id("cluster-launch.v1"), id("tensor-memory.v1")]),
    })
    .unwrap()
}

fn rebuilt(parts: KernelSemanticKeyPartsV1) -> KernelSemanticKeyV1 {
    KernelSemanticKeyV1::new(parts).unwrap()
}

fn detail_parts() -> KernelImplementationDetailPartsV1 {
    KernelImplementationDetailPartsV1 {
        artifact: artifact("cubin.v1", 4096, 'a'),
        symbol: id("custom_prefill_projection_v1"),
        target_architecture_id: id("sm_120f"),
        launch: KernelLaunchContractV1::new([32, 1, 1], [256, 1, 1], 1024).unwrap(),
        workspace: Some(
            KernelWorkspaceContractV1::new(
                8192,
                256,
                id("linear-workspace.v1"),
                KernelTensorAccessV1::ReadWrite,
                KernelAliasContractV1::AliasGroup(id("scratch-arena.v1")),
            )
            .unwrap(),
        ),
        specializations: vec![
            KernelSpecializationV1::new(id("tile-m"), id("128")),
            KernelSpecializationV1::new(id("tile-n"), id("64")),
        ],
        source: source(
            "user-kernel-source.v1",
            "0123456789abcdef",
            "kernels/projection.cu",
            2048,
            'b',
        ),
        build: build("nvcc", "13.2", "reproducible-build.v1", 'c'),
    }
}

fn sample_detail() -> KernelImplementationDetailV1 {
    KernelImplementationDetailV1::new(detail_parts()).unwrap()
}

fn bundle_candidate(key: KernelSemanticKeyV1) -> KernelImplementationCandidateV1 {
    KernelImplementationCandidateV1::with_detail(
        id("external.projection.impl.v1"),
        key,
        KernelImplementationTierV1::Bundle,
        sample_detail(),
    )
    .unwrap()
}

#[test]
fn every_single_semantic_field_mutation_fails_exact_resolution() {
    let baseline = sample_key();
    let resolver = KernelResolverV1::new([bundle_candidate(baseline.clone())]).unwrap();
    let mut mutations = Vec::new();

    let mut parts = baseline.parts();
    parts.operation_abi = KernelOperationAbiV1::new(id("custom.prefill.other"), 1).unwrap();
    mutations.push(("operation ABI ID", rebuilt(parts)));

    let mut parts = baseline.parts();
    parts.operation_abi = KernelOperationAbiV1::new(id("custom.prefill.projection"), 2).unwrap();
    mutations.push(("operation ABI version", rebuilt(parts)));

    let mut parts = baseline.parts();
    parts.tensors.swap(0, 1);
    mutations.push(("tensor order", rebuilt(parts)));

    let tensor_mutations: [TensorMutation; 8] = [
        (
            "tensor role",
            Box::new(|_| {
                KernelTensorRequirementV1::new(
                    id("other-input"),
                    vec![32, 4096],
                    id("bf16"),
                    id("row-major.v1"),
                    id("scale.none.v1"),
                    16,
                )
                .unwrap()
            }),
        ),
        (
            "tensor dimensions",
            Box::new(|_| {
                KernelTensorRequirementV1::new(
                    id("input"),
                    vec![31, 4096],
                    id("bf16"),
                    id("row-major.v1"),
                    id("scale.none.v1"),
                    16,
                )
                .unwrap()
            }),
        ),
        (
            "tensor dtype",
            Box::new(|_| {
                KernelTensorRequirementV1::new(
                    id("input"),
                    vec![32, 4096],
                    id("fp16"),
                    id("row-major.v1"),
                    id("scale.none.v1"),
                    16,
                )
                .unwrap()
            }),
        ),
        (
            "tensor layout",
            Box::new(|_| {
                KernelTensorRequirementV1::new(
                    id("input"),
                    vec![32, 4096],
                    id("bf16"),
                    id("strided.v1"),
                    id("scale.none.v1"),
                    16,
                )
                .unwrap()
            }),
        ),
        (
            "scale format",
            Box::new(|_| {
                KernelTensorRequirementV1::new(
                    id("input"),
                    vec![32, 4096],
                    id("bf16"),
                    id("row-major.v1"),
                    id("scale.other.v1"),
                    16,
                )
                .unwrap()
            }),
        ),
        (
            "scale block geometry",
            Box::new(|requirement| requirement.with_scale_block_dimensions(vec![32]).unwrap()),
        ),
        (
            "tensor alignment",
            Box::new(|_| {
                KernelTensorRequirementV1::new(
                    id("input"),
                    vec![32, 4096],
                    id("bf16"),
                    id("row-major.v1"),
                    id("scale.none.v1"),
                    32,
                )
                .unwrap()
            }),
        ),
        (
            "tensor access and alias",
            Box::new(|requirement| {
                requirement.with_access(
                    KernelTensorAccessV1::ReadWrite,
                    KernelAliasContractV1::AliasGroup(id("other-arena.v1")),
                )
            }),
        ),
    ];
    for (name, mutate) in tensor_mutations {
        let mut parts = baseline.parts();
        parts.tensors[0] = mutate(parts.tensors[0].clone());
        mutations.push((name, rebuilt(parts)));
    }

    let mut parts = baseline.parts();
    parts.tensors[2] =
        output().with_dependencies(vec![KernelTensorDependencyV1::new(1, id("reads-after.v1"))]);
    mutations.push(("tensor dependency index", rebuilt(parts)));

    let mut parts = baseline.parts();
    parts.tensors[2] = output().with_dependencies(vec![KernelTensorDependencyV1::new(
        0,
        id("other-dependency.v1"),
    )]);
    mutations.push(("tensor dependency ID", rebuilt(parts)));

    let mut parts = baseline.parts();
    parts.scalar_arguments[0] =
        KernelScalarArgumentV1::new(id("other-rank"), id("unsigned-int"), 32).unwrap();
    mutations.push(("scalar argument ID", rebuilt(parts)));

    let mut parts = baseline.parts();
    parts.scalar_arguments[0] =
        KernelScalarArgumentV1::new(id("stream-rank"), id("signed-int"), 32).unwrap();
    mutations.push(("scalar dtype", rebuilt(parts)));

    let mut parts = baseline.parts();
    parts.scalar_arguments[0] =
        KernelScalarArgumentV1::new(id("stream-rank"), id("unsigned-int"), 64).unwrap();
    mutations.push(("scalar width", rebuilt(parts)));

    let mut parts = baseline.parts();
    parts.scalar_arguments.clear();
    mutations.push(("scalar presence", rebuilt(parts)));

    let mut parts = baseline.parts();
    parts.numerical_contract = id("different-model-numerics.opaque.v1");
    mutations.push(("numerical contract", rebuilt(parts)));

    let mut parts = baseline.parts();
    parts.required_capabilities.remove(&id("cluster-launch.v1"));
    mutations.push(("required capabilities", rebuilt(parts)));

    for (name, mutation) in mutations {
        assert_eq!(
            resolver.resolve(&mutation).unwrap_err(),
            KernelResolutionErrorV1::Unsupported {
                operation_id: mutation.operation_abi().id().clone(),
            },
            "{name}"
        );
    }
    assert!(resolver.resolve(&baseline).is_ok());
}

#[test]
fn implementation_identity_never_changes_the_semantic_match() {
    let key = sample_key();
    let mut details = Vec::new();

    for changed_artifact in [
        artifact("fatbin.v1", 4096, 'a'),
        artifact("cubin.v1", 4097, 'a'),
        artifact("cubin.v1", 4096, 'd'),
    ] {
        let mut parts = detail_parts();
        parts.artifact = changed_artifact;
        details.push(parts);
    }

    let mut parts = detail_parts();
    parts.symbol = id("other_projection_symbol_v1");
    details.push(parts);

    let mut parts = detail_parts();
    parts.target_architecture_id = id("sm_121a");
    details.push(parts);

    for launch in [
        KernelLaunchContractV1::new([31, 1, 1], [256, 1, 1], 1024).unwrap(),
        KernelLaunchContractV1::new([32, 1, 1], [128, 1, 1], 1024).unwrap(),
        KernelLaunchContractV1::new([32, 1, 1], [256, 1, 1], 2048).unwrap(),
    ] {
        let mut parts = detail_parts();
        parts.launch = launch;
        details.push(parts);
    }

    let mut parts = detail_parts();
    parts.workspace = None;
    details.push(parts);

    let mut parts = detail_parts();
    parts.specializations = vec![KernelSpecializationV1::new(id("tile-k"), id("256"))];
    details.push(parts);

    for changed_source in [
        source(
            "other-source.v1",
            "0123456789abcdef",
            "kernels/projection.cu",
            2048,
            'b',
        ),
        source(
            "user-kernel-source.v1",
            "fedcba9876543210",
            "kernels/other.cu",
            2049,
            'd',
        ),
    ] {
        let mut parts = detail_parts();
        parts.source = changed_source;
        details.push(parts);
    }

    for changed_build in [
        build("other-compiler", "13.2", "reproducible-build.v1", 'c'),
        build("nvcc", "13.3", "other-build.v1", 'd'),
    ] {
        let mut parts = detail_parts();
        parts.build = changed_build;
        details.push(parts);
    }

    for (index, parts) in details.into_iter().enumerate() {
        let detail = KernelImplementationDetailV1::new(parts).unwrap();
        let candidate = KernelImplementationCandidateV1::with_detail(
            id(&format!("external.projection.impl.v{}", index + 1)),
            key.clone(),
            KernelImplementationTierV1::Bundle,
            detail,
        )
        .unwrap();
        let resolver = KernelResolverV1::new([candidate]).unwrap();
        let resolution = resolver
            .resolve(&key)
            .expect("implementation identity is not semantic");
        assert_eq!(resolution.tier(), KernelImplementationTierV1::Bundle);
        assert!(resolution.detail().is_some());
    }
}

#[test]
fn a_malformed_declared_implementation_is_rejected() {
    let mut parts = detail_parts();
    parts.specializations = vec![
        KernelSpecializationV1::new(id("tile-m"), id("128")),
        KernelSpecializationV1::new(id("tile-m"), id("64")),
    ];
    assert_eq!(
        KernelImplementationDetailV1::new(parts).unwrap_err(),
        KernelResolutionErrorV1::DuplicateSpecialization(id("tile-m"))
    );

    assert_eq!(
        KernelImplementationCandidateV1::new(
            id(KERNEL_UNSUPPORTED_IMPLEMENTATION_ID),
            sample_key(),
            KernelImplementationTierV1::Bundle,
        )
        .unwrap_err(),
        KernelResolutionErrorV1::ReservedImplementationId(id(KERNEL_UNSUPPORTED_IMPLEMENTATION_ID))
    );
}

#[test]
fn an_absent_optional_implementation_falls_through_to_a_lower_tier() {
    let key = sample_key();
    let resolver = KernelResolverV1::new([KernelImplementationCandidateV1::new(
        id("host.generic.projection.v1"),
        key.clone(),
        KernelImplementationTierV1::Generic,
    )
    .unwrap()])
    .unwrap();

    let resolution = resolver.resolve(&key).unwrap();
    assert_eq!(resolution.tier(), KernelImplementationTierV1::Generic);
    assert!(resolution.detail().is_none());
    assert_eq!(
        resolution.candidate().qualification().kind(),
        KernelQualificationKindV1::Unqualified
    );
}

#[test]
fn duplicate_exact_implementations_are_rejected_in_every_order() {
    let key = sample_key();
    let first = bundle_candidate(key.clone());
    let second = KernelImplementationCandidateV1::with_detail(
        id("external.projection.impl.v2"),
        key,
        KernelImplementationTierV1::Bundle,
        sample_detail(),
    )
    .unwrap();
    for candidates in [[first.clone(), second.clone()], [second, first]] {
        assert_eq!(
            KernelResolverV1::new(candidates).unwrap_err(),
            KernelResolutionErrorV1::AmbiguousTier {
                tier: KernelImplementationTierV1::Bundle,
                operation_id: id("custom.prefill.projection"),
            }
        );
    }
}

#[test]
fn a_repeated_implementation_id_is_rejected() {
    let key = sample_key();
    let mut other_parts = key.parts();
    other_parts.operation_abi = KernelOperationAbiV1::new(id("custom.prefill.other"), 1).unwrap();
    assert_eq!(
        KernelResolverV1::new([
            bundle_candidate(key),
            bundle_candidate(rebuilt(other_parts)),
        ])
        .unwrap_err(),
        KernelResolutionErrorV1::DuplicateImplementationId(id("external.projection.impl.v1"))
    );
}

#[test]
fn a_custom_operation_with_no_implementation_stays_unsupported() {
    let empty = KernelResolverV1::new([]).unwrap();
    assert_eq!(
        empty.resolve(&sample_key()).unwrap_err(),
        KernelResolutionErrorV1::Unsupported {
            operation_id: id("custom.prefill.projection"),
        }
    );
}

#[test]
fn semantic_key_parts_are_closed_over_only_the_approved_fields() {
    let KernelSemanticKeyPartsV1 {
        operation_abi,
        tensors,
        scalar_arguments,
        numerical_contract,
        required_capabilities,
    } = sample_key().parts();
    assert_eq!(operation_abi.id().as_str(), "custom.prefill.projection");
    assert_eq!(operation_abi.version(), 1);
    assert_eq!(tensors.len(), 3);
    assert_eq!(tensors[1].scale().block_dimensions(), [1, 32]);
    assert_eq!(scalar_arguments.len(), 1);
    assert_eq!(
        numerical_contract.as_str(),
        "model-supplied-numerics.opaque.v1"
    );
    assert_eq!(required_capabilities.len(), 2);
}

#[test]
fn a_dependency_must_name_an_earlier_operand() {
    let mut parts = sample_key().parts();
    // The input at index 0 has no earlier operand to depend on.
    parts.tensors[0] =
        input().with_dependencies(vec![KernelTensorDependencyV1::new(0, id("reads-after.v1"))]);
    assert_eq!(
        KernelSemanticKeyV1::new(parts).unwrap_err(),
        KernelSemanticErrorV1::InvalidTensorDependency {
            tensor_index: 0,
            depends_on: 0,
        }
    );

    for depends_on in [1_u16, 2, 3] {
        let mut parts = sample_key().parts();
        // The weight at index 1 may only depend on index 0.
        parts.tensors[1] = weight().with_dependencies(vec![KernelTensorDependencyV1::new(
            depends_on,
            id("reads-after.v1"),
        )]);
        assert_eq!(
            KernelSemanticKeyV1::new(parts).unwrap_err(),
            KernelSemanticErrorV1::InvalidTensorDependency {
                tensor_index: 1,
                depends_on,
            },
            "operand 1 must not depend on {depends_on}"
        );
    }

    let mut parts = sample_key().parts();
    parts.tensors[1] =
        weight().with_dependencies(vec![KernelTensorDependencyV1::new(0, id("reads-after.v1"))]);
    assert!(KernelSemanticKeyV1::new(parts).is_ok());
}

#[test]
fn a_bundle_private_operation_is_never_satisfied_by_a_lower_tier() {
    let key = sample_key();
    let host = KernelImplementationCandidateV1::new(
        id("host.projection.generic.v1"),
        key.clone(),
        KernelImplementationTierV1::Generic,
    )
    .unwrap();
    let optimized = KernelImplementationCandidateV1::new(
        id("host.projection.optimized.v1"),
        key.clone(),
        KernelImplementationTierV1::HostOptimized,
    )
    .unwrap();

    let open = KernelResolverV1::new([host.clone(), optimized.clone()]).unwrap();
    assert!(!open.is_bundle_private(&key));
    assert_eq!(
        open.resolve(&key).unwrap().tier(),
        KernelImplementationTierV1::HostOptimized
    );

    let private =
        KernelResolverV1::new_with_bundle_private([host, optimized], [key.clone()]).unwrap();
    assert!(private.is_bundle_private(&key));
    assert_eq!(
        private.resolve(&key).unwrap_err(),
        KernelResolutionErrorV1::Unsupported {
            operation_id: id("custom.prefill.projection"),
        }
    );

    let satisfied =
        KernelResolverV1::new_with_bundle_private([bundle_candidate(key.clone())], [key.clone()])
            .unwrap();
    let resolution = satisfied.resolve(&key).unwrap();
    assert_eq!(resolution.tier(), KernelImplementationTierV1::Bundle);
    assert!(resolution.detail().is_some());
}
