//! Kernel identities, launch contracts and resolution without native artifacts.
#![forbid(unsafe_code)]

mod artifact;
mod contract;
mod implementation;
mod semantic;

pub use artifact::{
    KernelArtifactIdentityV1, KernelBuildIdentityV1, KernelIdentityErrorV1, KernelOpaqueIdV1,
    KernelSha256V1, KernelSourceIdentityV1,
};

pub use contract::{
    KernelAliasContractV1, KernelContractErrorV1, KernelLaunchContractV1, KernelScaleContractV1,
    KernelSpecializationV1, KernelTensorAccessV1, KernelTensorDependencyV1,
    KernelWorkspaceContractV1,
};

pub use semantic::{
    KernelOperationAbiV1, KernelScalarArgumentV1, KernelSemanticErrorV1, KernelSemanticKeyPartsV1,
    KernelSemanticKeyV1, KernelTensorRequirementV1, MAX_SEMANTIC_CAPABILITIES,
    MAX_SEMANTIC_DIMENSIONS, MAX_SEMANTIC_SCALARS, MAX_SEMANTIC_TENSORS,
};

pub use implementation::{
    KERNEL_UNSUPPORTED_IMPLEMENTATION_ID, KernelImplementationCandidateV1,
    KernelImplementationDetailPartsV1, KernelImplementationDetailV1, KernelImplementationTierV1,
    KernelQualificationKindV1, KernelQualificationStateV1, KernelResolutionErrorV1,
    KernelResolutionV1, KernelResolverV1, MAX_RESOLVER_IMPLEMENTATIONS,
};

// Buffer bounds shared by host admission and the native sampler.
pub const SAMPLING_SIMPLE_TILE: usize = 1024;
pub const SAMPLING_SIMPLE_PARTIAL_BYTES: usize = 112;
pub const SAMPLING_FILTER_ROW_BYTES: usize = 128 + 2048 * 16;
