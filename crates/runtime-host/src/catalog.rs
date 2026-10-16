//! The generic operations this host offers an installed bundle.
//!
//! A bundle that leaves an operation to the host names a primitive and declares
//! the complete semantic contract it expects. This catalog is the other half of
//! that: every entry is a `Generic` candidate for one exact contract the shared
//! layer declares, offered under an identity that names both the primitive and
//! the width it serves.
//!
//! Nothing here is a model or a fixture vocabulary. The one operation this host
//! offers is the model-neutral exact F32 copy, and the contract behind every
//! candidate comes from [`runtime_bundle::generic_copy_f32_contract_v1`]
//! rather than from a second declaration written here. A width the host does
//! not serve produces no candidate at all, which is what makes an unserved
//! width unsupported rather than approximately satisfied.

use std::collections::BTreeMap;

use runtime_bundle::{
    BundleError, GENERIC_COPY_F32_BLOCK_ELEMENTS_V1, GENERIC_COPY_F32_PRIMITIVE_V1,
    generic_copy_f32_contract_v1, generic_copy_f32_implementation_id_v1,
};
use runtime_kernel_contract::{
    KernelImplementationCandidateV1, KernelImplementationTierV1, KernelOpaqueIdV1,
    KernelSemanticKeyV1,
};

/// The most values one exact F32 copy this host serves may move.
///
/// The copy is a whole-operand move with no arithmetic, so the bound is about
/// how large a generic fallback this host is willing to be responsible for
/// rather than about the contract. A bundle that needs a wider copy carries its
/// own implementation instead.
pub const HOST_GENERIC_COPY_F32_MAX_ELEMENTS_V1: u32 = 4096;

/// How many exact F32 copy widths this host serves.
pub const HOST_GENERIC_COPY_F32_WIDTHS_V1: u32 =
    HOST_GENERIC_COPY_F32_MAX_ELEMENTS_V1 / GENERIC_COPY_F32_BLOCK_ELEMENTS_V1;

/// Whether this host serves the exact F32 copy of `elements` values.
///
/// Both operands are aligned to whole blocks, so this host serves whole-block
/// widths up to its bound and nothing between them. Every other width is a
/// different semantic key, and this host offers no candidate for it.
#[must_use]
pub const fn host_serves_generic_copy_f32_width_v1(elements: u32) -> bool {
    elements != 0
        && elements.is_multiple_of(GENERIC_COPY_F32_BLOCK_ELEMENTS_V1)
        && elements <= HOST_GENERIC_COPY_F32_MAX_ELEMENTS_V1
}

/// Every width this host serves, in ascending order.
pub fn host_generic_copy_f32_widths_v1() -> impl Iterator<Item = u32> {
    (1..=HOST_GENERIC_COPY_F32_WIDTHS_V1).map(|block| block * GENERIC_COPY_F32_BLOCK_ELEMENTS_V1)
}

/// One generic operation this host offers, and the primitive it answers for.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostGenericImplementationV1 {
    primitive: KernelOpaqueIdV1,
    implementation: KernelOpaqueIdV1,
    semantic_key: KernelSemanticKeyV1,
}

impl HostGenericImplementationV1 {
    /// The primitive a bundle's provisioning names to ask for this.
    #[must_use]
    pub const fn primitive(&self) -> &KernelOpaqueIdV1 {
        &self.primitive
    }

    /// The identity this host offers the operation under.
    #[must_use]
    pub const fn implementation(&self) -> &KernelOpaqueIdV1 {
        &self.implementation
    }

    /// The exact contract this candidate satisfies, and only this one.
    #[must_use]
    pub const fn semantic_key(&self) -> &KernelSemanticKeyV1 {
        &self.semantic_key
    }

    /// The `Generic` candidate a resolver takes this as.
    ///
    /// A host never declares the bundle tier. That is what makes an operation a
    /// bundle keeps private unsatisfiable from here, whatever this offers.
    pub fn candidate(&self) -> Result<KernelImplementationCandidateV1, BundleError> {
        KernelImplementationCandidateV1::new(
            self.implementation.clone(),
            self.semantic_key.clone(),
            KernelImplementationTierV1::Generic,
        )
        .map_err(|error| {
            BundleError::validation(format!(
                "host generic implementation {} is malformed: {error}",
                self.implementation
            ))
        })
    }
}

/// Every generic operation this host offers, built once per process-start load.
#[derive(Clone, Debug)]
pub struct HostGenericCatalogV1 {
    implementations: Vec<HostGenericImplementationV1>,
    primitives: BTreeMap<KernelOpaqueIdV1, KernelOpaqueIdV1>,
}

impl HostGenericCatalogV1 {
    /// The catalog this host is compiled with.
    ///
    /// Only the exact F32 copy, at the widths this host serves. Building it can
    /// only fail if this host's own declarations are inconsistent, which is why
    /// nothing is quietly dropped: a width that cannot be described is an error
    /// rather than a candidate silently missing from the catalog.
    pub fn builtin_v1() -> Result<Self, BundleError> {
        let mut implementations = Vec::new();
        for elements in host_generic_copy_f32_widths_v1() {
            implementations.push(HostGenericImplementationV1 {
                primitive: kernel_id_v1(GENERIC_COPY_F32_PRIMITIVE_V1)?,
                implementation: kernel_id_v1(
                    generic_copy_f32_implementation_id_v1(elements)?.as_str(),
                )?,
                semantic_key: generic_copy_f32_contract_v1(elements)?.semantic_key()?,
            });
        }
        Self::new(implementations)
    }

    fn new(implementations: Vec<HostGenericImplementationV1>) -> Result<Self, BundleError> {
        let mut primitives = BTreeMap::new();
        for implementation in &implementations {
            if primitives
                .insert(
                    implementation.implementation.clone(),
                    implementation.primitive.clone(),
                )
                .is_some()
            {
                return Err(BundleError::validation(format!(
                    "this host offers implementation {} more than once",
                    implementation.implementation
                )));
            }
        }
        Ok(Self {
            implementations,
            primitives,
        })
    }

    /// Every operation this host offers.
    #[must_use]
    pub fn implementations(&self) -> &[HostGenericImplementationV1] {
        &self.implementations
    }

    /// The candidates a manifest resolver is built with.
    pub fn candidates(&self) -> Result<Vec<KernelImplementationCandidateV1>, BundleError> {
        self.implementations
            .iter()
            .map(HostGenericImplementationV1::candidate)
            .collect()
    }

    /// The primitive this host offers one implementation identity under.
    ///
    /// A resolution that landed on an identity this returns nothing for did not
    /// come from this catalog, whatever tier it claims.
    #[must_use]
    pub fn primitive_of(&self, implementation: &KernelOpaqueIdV1) -> Option<&KernelOpaqueIdV1> {
        self.primitives.get(implementation)
    }
}

fn kernel_id_v1(value: &str) -> Result<KernelOpaqueIdV1, BundleError> {
    KernelOpaqueIdV1::new(value).map_err(|error| {
        BundleError::validation(format!(
            "host catalog identifier '{value}' is invalid: {error}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use runtime_bundle::{
        GENERIC_COPY_F32_PRIMITIVE_V1, generic_copy_f32_contract_v1,
        generic_copy_f32_implementation_id_v1,
    };
    use runtime_kernel_contract::{KernelImplementationTierV1, KernelOpaqueIdV1, KernelResolverV1};

    use super::{
        HOST_GENERIC_COPY_F32_MAX_ELEMENTS_V1, HOST_GENERIC_COPY_F32_WIDTHS_V1,
        HostGenericCatalogV1, host_generic_copy_f32_widths_v1,
        host_serves_generic_copy_f32_width_v1,
    };

    fn key(elements: u32) -> runtime_kernel_contract::KernelSemanticKeyV1 {
        generic_copy_f32_contract_v1(elements)
            .expect("the shared copy contract is valid")
            .semantic_key()
            .expect("the shared copy contract has a semantic key")
    }

    /// Every width the catalog serves is offered exactly once, and the whole
    /// set is what the host said it serves rather than whatever survived.
    #[test]
    fn the_catalog_offers_one_generic_candidate_for_every_width_it_serves() {
        let catalog = HostGenericCatalogV1::builtin_v1().expect("this host's catalog is valid");

        assert_eq!(
            catalog.implementations().len(),
            HOST_GENERIC_COPY_F32_WIDTHS_V1 as usize
        );
        let candidates = catalog
            .candidates()
            .expect("every candidate is well formed");
        assert_eq!(candidates.len(), catalog.implementations().len());
        assert!(
            candidates
                .iter()
                .all(|candidate| candidate.tier() == KernelImplementationTierV1::Generic),
            "a host never declares the bundle tier"
        );
        // Duplicate identities or duplicate keys inside one tier would make the
        // resolver ambiguous, so the whole catalog has to declare as one.
        KernelResolverV1::new(candidates).expect("the whole catalog declares without conflict");
    }

    /// An unserved width has no candidate, so it stays a different operation.
    #[test]
    fn a_width_this_host_does_not_serve_is_offered_no_candidate() {
        let catalog = HostGenericCatalogV1::builtin_v1().expect("this host's catalog is valid");
        let resolver = KernelResolverV1::new(catalog.candidates().expect("candidates are valid"))
            .expect("the catalog declares without conflict");

        assert!(host_serves_generic_copy_f32_width_v1(64));
        assert!(resolver.resolve(&key(64)).is_ok());
        for unserved in [1, 63, 65, 100, HOST_GENERIC_COPY_F32_MAX_ELEMENTS_V1 + 64] {
            assert!(
                !host_serves_generic_copy_f32_width_v1(unserved),
                "{unserved} must not be a width this host serves"
            );
            assert!(
                resolver.resolve(&key(unserved)).is_err(),
                "{unserved} must have no candidate in this catalog"
            );
        }
        assert!(
            host_generic_copy_f32_widths_v1().all(host_serves_generic_copy_f32_width_v1),
            "every width the catalog enumerates must be one this host serves"
        );
    }

    /// This host offers one primitive and nothing else.
    ///
    /// A bundle's own custom operation is satisfied only by that bundle, and
    /// the catalog is what could contradict that by quietly offering something
    /// for a key it never should. Naming the whole set here is what keeps a
    /// later entry from being added without a decision.
    #[test]
    fn the_catalog_offers_exactly_one_primitive() {
        let catalog = HostGenericCatalogV1::builtin_v1().expect("this host's catalog is valid");

        let offered = catalog
            .implementations()
            .iter()
            .map(|implementation| implementation.primitive().as_str())
            .collect::<std::collections::BTreeSet<_>>();

        assert_eq!(
            offered,
            std::collections::BTreeSet::from([GENERIC_COPY_F32_PRIMITIVE_V1])
        );
    }

    /// A resolved identity is traceable back to the primitive it answers for,
    /// so a resolution can be checked against the provisioning that asked.
    #[test]
    fn every_offered_identity_names_the_primitive_it_answers_for() {
        let catalog = HostGenericCatalogV1::builtin_v1().expect("this host's catalog is valid");
        let expected = KernelOpaqueIdV1::new(GENERIC_COPY_F32_PRIMITIVE_V1)
            .expect("the shared primitive identifier is valid");

        let offered = KernelOpaqueIdV1::new(
            generic_copy_f32_implementation_id_v1(64)
                .expect("the 64-value identity is valid")
                .as_str(),
        )
        .expect("the 64-value identity is a valid kernel identifier");
        assert_eq!(catalog.primitive_of(&offered), Some(&expected));

        let unoffered = KernelOpaqueIdV1::new("chop.generic.copy-f32.v1.elements-65")
            .expect("the test identity is valid");
        assert_eq!(catalog.primitive_of(&unoffered), None);
    }
}
