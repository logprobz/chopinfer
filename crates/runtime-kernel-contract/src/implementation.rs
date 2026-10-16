//! The implementation half of a kernel contract, and the tiered resolver.
//!
//! An implementation descriptor names who satisfies a semantic key: its
//! artifact, symbol, launch, source, build, and provider tier. Resolution
//! matches the semantic key exactly, then picks the highest tier that declared
//! an implementation. Ambiguity inside one tier is rejected outright.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use crate::artifact::{
    KernelArtifactIdentityV1, KernelBuildIdentityV1, KernelOpaqueIdV1, KernelSourceIdentityV1,
};
use crate::contract::{KernelLaunchContractV1, KernelSpecializationV1, KernelWorkspaceContractV1};
use crate::semantic::KernelSemanticKeyV1;

/// The largest number of implementations one resolver may hold.
pub const MAX_RESOLVER_IMPLEMENTATIONS: usize = 4_096;

/// The reserved identity the resolver reports when nothing matches. No
/// declared implementation may claim it.
pub const KERNEL_UNSUPPORTED_IMPLEMENTATION_ID: &str = "unsupported";

/// Who provides an implementation. Resolution visits these in declared order.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelImplementationTierV1 {
    Bundle,
    HostOptimized,
    Generic,
}

impl KernelImplementationTierV1 {
    /// Bundle first, then host-optimized, then correct generic.
    pub const RESOLUTION_ORDER: [Self; 3] = [Self::Bundle, Self::HostOptimized, Self::Generic];

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bundle => "bundle",
            Self::HostOptimized => "host-optimized",
            Self::Generic => "generic",
        }
    }
}

impl fmt::Display for KernelImplementationTierV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// The open form of one implementation's artifact and build identity.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelImplementationDetailPartsV1 {
    pub artifact: KernelArtifactIdentityV1,
    pub symbol: KernelOpaqueIdV1,
    pub target_architecture_id: KernelOpaqueIdV1,
    pub launch: KernelLaunchContractV1,
    pub workspace: Option<KernelWorkspaceContractV1>,
    pub specializations: Vec<KernelSpecializationV1>,
    pub source: KernelSourceIdentityV1,
    pub build: KernelBuildIdentityV1,
}

/// The artifact, symbol, launch, source, and build identity of one
/// implementation. None of these fields take part in semantic matching.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelImplementationDetailV1 {
    artifact: KernelArtifactIdentityV1,
    symbol: KernelOpaqueIdV1,
    target_architecture_id: KernelOpaqueIdV1,
    launch: KernelLaunchContractV1,
    workspace: Option<KernelWorkspaceContractV1>,
    specializations: Vec<KernelSpecializationV1>,
    source: KernelSourceIdentityV1,
    build: KernelBuildIdentityV1,
}

impl KernelImplementationDetailV1 {
    pub fn new(parts: KernelImplementationDetailPartsV1) -> Result<Self, KernelResolutionErrorV1> {
        let mut specialization_ids = BTreeSet::new();
        for specialization in &parts.specializations {
            if !specialization_ids.insert(specialization.id()) {
                return Err(KernelResolutionErrorV1::DuplicateSpecialization(
                    specialization.id().clone(),
                ));
            }
        }
        Ok(Self {
            artifact: parts.artifact,
            symbol: parts.symbol,
            target_architecture_id: parts.target_architecture_id,
            launch: parts.launch,
            workspace: parts.workspace,
            specializations: parts.specializations,
            source: parts.source,
            build: parts.build,
        })
    }

    #[must_use]
    pub const fn artifact(&self) -> &KernelArtifactIdentityV1 {
        &self.artifact
    }

    #[must_use]
    pub const fn symbol(&self) -> &KernelOpaqueIdV1 {
        &self.symbol
    }

    #[must_use]
    pub const fn target_architecture_id(&self) -> &KernelOpaqueIdV1 {
        &self.target_architecture_id
    }

    #[must_use]
    pub const fn launch(&self) -> &KernelLaunchContractV1 {
        &self.launch
    }

    #[must_use]
    pub const fn workspace(&self) -> Option<&KernelWorkspaceContractV1> {
        self.workspace.as_ref()
    }

    #[must_use]
    pub fn specializations(&self) -> &[KernelSpecializationV1] {
        &self.specializations
    }

    #[must_use]
    pub const fn source(&self) -> &KernelSourceIdentityV1 {
        &self.source
    }

    #[must_use]
    pub const fn build(&self) -> &KernelBuildIdentityV1 {
        &self.build
    }
}

/// How far one implementation has been qualified for execution.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum KernelQualificationKindV1 {
    Qualified,
    Unqualified,
    Rejected,
}

/// The qualification state of one implementation and why it holds.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelQualificationStateV1 {
    kind: KernelQualificationKindV1,
    detail_id: KernelOpaqueIdV1,
}

impl KernelQualificationStateV1 {
    #[must_use]
    pub const fn unqualified(blocker_id: KernelOpaqueIdV1) -> Self {
        Self {
            kind: KernelQualificationKindV1::Unqualified,
            detail_id: blocker_id,
        }
    }

    #[must_use]
    pub const fn kind(&self) -> KernelQualificationKindV1 {
        self.kind
    }

    #[must_use]
    pub const fn detail_id(&self) -> &KernelOpaqueIdV1 {
        &self.detail_id
    }
}

/// One declared implementation of one exact semantic key.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KernelImplementationCandidateV1 {
    implementation_id: KernelOpaqueIdV1,
    semantic_key: KernelSemanticKeyV1,
    tier: KernelImplementationTierV1,
    detail: Option<KernelImplementationDetailV1>,
    qualification: KernelQualificationStateV1,
}

impl KernelImplementationCandidateV1 {
    /// Declares an implementation whose artifact detail is not yet attached.
    /// An absent optional implementation is a legitimate declaration, so a
    /// lower tier may still satisfy the same key.
    pub fn new(
        implementation_id: KernelOpaqueIdV1,
        semantic_key: KernelSemanticKeyV1,
        tier: KernelImplementationTierV1,
    ) -> Result<Self, KernelResolutionErrorV1> {
        Self::declare(implementation_id, semantic_key, tier, None)
    }

    /// Declares an implementation together with its artifact detail. A
    /// malformed detail rejects the declaration, and with it the bundle.
    pub fn with_detail(
        implementation_id: KernelOpaqueIdV1,
        semantic_key: KernelSemanticKeyV1,
        tier: KernelImplementationTierV1,
        detail: KernelImplementationDetailV1,
    ) -> Result<Self, KernelResolutionErrorV1> {
        Self::declare(implementation_id, semantic_key, tier, Some(detail))
    }

    fn declare(
        implementation_id: KernelOpaqueIdV1,
        semantic_key: KernelSemanticKeyV1,
        tier: KernelImplementationTierV1,
        detail: Option<KernelImplementationDetailV1>,
    ) -> Result<Self, KernelResolutionErrorV1> {
        if implementation_id.as_str() == KERNEL_UNSUPPORTED_IMPLEMENTATION_ID {
            return Err(KernelResolutionErrorV1::ReservedImplementationId(
                implementation_id,
            ));
        }
        Ok(Self {
            implementation_id,
            semantic_key,
            tier,
            detail,
            qualification: KernelQualificationStateV1::unqualified(
                KernelOpaqueIdV1::new("kernel-execution-not-established.v1")
                    .expect("static blocker ID is a valid opaque ID"),
            ),
        })
    }

    #[must_use]
    pub const fn implementation_id(&self) -> &KernelOpaqueIdV1 {
        &self.implementation_id
    }

    #[must_use]
    pub const fn semantic_key(&self) -> &KernelSemanticKeyV1 {
        &self.semantic_key
    }

    #[must_use]
    pub const fn tier(&self) -> KernelImplementationTierV1 {
        self.tier
    }

    #[must_use]
    pub const fn detail(&self) -> Option<&KernelImplementationDetailV1> {
        self.detail.as_ref()
    }

    #[must_use]
    pub const fn qualification(&self) -> &KernelQualificationStateV1 {
        &self.qualification
    }
}

/// One exact semantic match, taken from the highest tier that declared it.
#[derive(Clone, Copy, Debug)]
pub struct KernelResolutionV1<'a> {
    candidate: &'a KernelImplementationCandidateV1,
}

impl<'a> KernelResolutionV1<'a> {
    #[must_use]
    pub const fn candidate(self) -> &'a KernelImplementationCandidateV1 {
        self.candidate
    }

    #[must_use]
    pub const fn implementation_id(self) -> &'a KernelOpaqueIdV1 {
        self.candidate.implementation_id()
    }

    #[must_use]
    pub const fn tier(self) -> KernelImplementationTierV1 {
        self.candidate.tier()
    }

    #[must_use]
    pub const fn detail(self) -> Option<&'a KernelImplementationDetailV1> {
        self.candidate.detail()
    }
}

/// Resolves semantic keys to implementations in bundle, host-optimized, then
/// correct-generic order.
#[derive(Clone, Debug, Default)]
pub struct KernelResolverV1 {
    candidates: Vec<KernelImplementationCandidateV1>,
    bundle_private: BTreeSet<KernelSemanticKeyV1>,
}

impl KernelResolverV1 {
    pub fn new(
        candidates: impl IntoIterator<Item = KernelImplementationCandidateV1>,
    ) -> Result<Self, KernelResolutionErrorV1> {
        Self::declare(candidates, BTreeSet::new())
    }

    /// Builds a resolver where the listed operations are private to the
    /// bundle that declared them.
    ///
    /// A bundle-private operation is satisfied only by a `Bundle` tier
    /// declaration. A host-optimized or generic candidate carrying the same
    /// semantic key never satisfies it, so an absent bundle implementation
    /// leaves the operation unsupported rather than silently falling through
    /// to host code the bundle never asked for.
    pub fn new_with_bundle_private(
        candidates: impl IntoIterator<Item = KernelImplementationCandidateV1>,
        bundle_private: impl IntoIterator<Item = KernelSemanticKeyV1>,
    ) -> Result<Self, KernelResolutionErrorV1> {
        Self::declare(candidates, bundle_private.into_iter().collect())
    }

    fn declare(
        candidates: impl IntoIterator<Item = KernelImplementationCandidateV1>,
        bundle_private: BTreeSet<KernelSemanticKeyV1>,
    ) -> Result<Self, KernelResolutionErrorV1> {
        let candidates = candidates.into_iter().collect::<Vec<_>>();
        if candidates.len() > MAX_RESOLVER_IMPLEMENTATIONS {
            return Err(KernelResolutionErrorV1::TooManyImplementations {
                count: candidates.len(),
            });
        }
        let mut implementation_ids = BTreeSet::new();
        let mut exact_tiers = BTreeSet::new();
        for candidate in &candidates {
            if !implementation_ids.insert(candidate.implementation_id()) {
                return Err(KernelResolutionErrorV1::DuplicateImplementationId(
                    candidate.implementation_id().clone(),
                ));
            }
            if !exact_tiers.insert((candidate.semantic_key(), candidate.tier())) {
                return Err(KernelResolutionErrorV1::AmbiguousTier {
                    tier: candidate.tier(),
                    operation_id: candidate.semantic_key().operation_abi().id().clone(),
                });
            }
        }
        Ok(Self {
            candidates,
            bundle_private,
        })
    }

    #[must_use]
    pub fn candidates(&self) -> &[KernelImplementationCandidateV1] {
        &self.candidates
    }

    /// Whether only a bundle declaration may satisfy this operation.
    #[must_use]
    pub fn is_bundle_private(&self, key: &KernelSemanticKeyV1) -> bool {
        self.bundle_private.contains(key)
    }

    pub fn resolve(
        &self,
        key: &KernelSemanticKeyV1,
    ) -> Result<KernelResolutionV1<'_>, KernelResolutionErrorV1> {
        let order: &[KernelImplementationTierV1] = if self.bundle_private.contains(key) {
            &[KernelImplementationTierV1::Bundle]
        } else {
            &KernelImplementationTierV1::RESOLUTION_ORDER
        };
        for &tier in order {
            let mut matches = self
                .candidates
                .iter()
                .filter(|candidate| candidate.tier() == tier && candidate.semantic_key() == key);
            let Some(candidate) = matches.next() else {
                continue;
            };
            if matches.next().is_some() {
                return Err(KernelResolutionErrorV1::AmbiguousTier {
                    tier,
                    operation_id: key.operation_abi().id().clone(),
                });
            }
            return Ok(KernelResolutionV1 { candidate });
        }
        Err(KernelResolutionErrorV1::Unsupported {
            operation_id: key.operation_abi().id().clone(),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelResolutionErrorV1 {
    ReservedImplementationId(KernelOpaqueIdV1),
    DuplicateSpecialization(KernelOpaqueIdV1),
    DuplicateImplementationId(KernelOpaqueIdV1),
    TooManyImplementations {
        count: usize,
    },
    AmbiguousTier {
        tier: KernelImplementationTierV1,
        operation_id: KernelOpaqueIdV1,
    },
    Unsupported {
        operation_id: KernelOpaqueIdV1,
    },
}

impl fmt::Display for KernelResolutionErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReservedImplementationId(id) => write!(
                formatter,
                "implementation ID {id} is reserved for unsupported operations"
            ),
            Self::DuplicateSpecialization(id) => {
                write!(formatter, "implementation repeats specialization {id}")
            }
            Self::DuplicateImplementationId(id) => {
                write!(formatter, "implementation ID {id} is declared twice")
            }
            Self::TooManyImplementations { count } => write!(
                formatter,
                "{count} implementations exceed {MAX_RESOLVER_IMPLEMENTATIONS}"
            ),
            Self::AmbiguousTier { tier, operation_id } => write!(
                formatter,
                "operation {operation_id} has more than one exact {tier} implementation"
            ),
            Self::Unsupported { operation_id } => write!(
                formatter,
                "operation {operation_id} has no exact implementation in any tier"
            ),
        }
    }
}

impl Error for KernelResolutionErrorV1 {}
