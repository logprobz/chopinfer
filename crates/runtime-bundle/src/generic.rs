//! The model-neutral generic primitives a bundle may require from its host.
//!
//! Nothing here is about a model. This is the one place the exact F32 copy is
//! declared: `element_count` contiguous binary32 values are read from `source`
//! and written to `destination`, bit for bit, in the dense row-major layout with
//! no scale. Every identifier is in the shared `chop.` namespace, because the
//! host is what implements it and the bundle only names it.
//!
//! A bundle requires this by declaring it with
//! [`BundleOperationProvisioningV1::HostGenericPrimitive`][provisioning] naming
//! [`GENERIC_COPY_F32_PRIMITIVE_V1`], and carrying no implementation of its
//! own. Resolution is then the host's: a catalog that offers an exact `Generic`
//! candidate for [`generic_copy_f32_contract_v1`] satisfies it, and a
//! catalog that offers nothing, or a candidate differing in any field of the
//! semantic key, leaves it unsupported. Every field takes part in the match,
//! including the element count, so a host declares the widths it serves.
//!
//! This lives in the shared contract crate rather than in either party, because
//! a bundle and a host that disagree by one byte of any identifier do not have
//! the same operation. The manifest builder that requires the copy and the host
//! catalog that offers it both call the functions below; neither writes its own
//! copy of the declaration.
//!
//! [provisioning]: crate::BundleOperationProvisioningV1

use crate::{
    BundleError, BundleOpaqueId, BundleOperationKeyV1, BundleScalarArgumentV1,
    BundleSemanticContractV1, BundleTensorAccessV1, BundleTensorAliasV1, BundleTensorDependencyV1,
    BundleTensorRequirementV1,
};

/// The operation ABI of the exact F32 copy.
pub const GENERIC_COPY_F32_OPERATION_V1: &str = "chop.generic.copy-f32";

/// The variant of that operation this contract describes.
pub const GENERIC_COPY_F32_VARIANT_V1: &str = "v1";

/// The version of that operation ABI.
pub const GENERIC_COPY_F32_ABI_VERSION_V1: u32 = 1;

/// The host primitive a bundle names when it leaves this operation to the host.
pub const GENERIC_COPY_F32_PRIMITIVE_V1: &str = "chop.generic.copy-f32.v1";

/// The operand the copy reads.
pub const GENERIC_COPY_F32_SOURCE_ROLE_V1: &str = "source";

/// The operand the copy writes.
pub const GENERIC_COPY_F32_DESTINATION_ROLE_V1: &str = "destination";

/// The scalar argument that says how many values the copy moves.
pub const GENERIC_COPY_F32_ELEMENT_COUNT_V1: &str = "element-count";

/// The numerical contract an exact copy computes under: no arithmetic at all.
pub const NUMERICS_EXACT_COPY_V1: &str = "chop.numerics.exact-copy.v1";

/// The alignment both operands of the copy are required to satisfy.
pub const GENERIC_COPY_F32_ALIGNMENT_V1: u64 = 256;

/// How many bytes one binary32 value occupies.
pub const GENERIC_COPY_F32_ELEMENT_BYTES_V1: u64 = 4;

/// The shared dtype identifier for IEEE binary32.
pub const DTYPE_F32_V1: &str = "chop.dtype.f32.v1";

/// The shared dtype identifier for unsigned 32-bit integers.
pub const DTYPE_U32_V1: &str = "chop.dtype.u32.v1";

/// The shared dtype identifier for signed 32-bit integers.
pub const DTYPE_I32_V1: &str = "chop.dtype.i32.v1";

/// The shared layout identifier for a dense row-major operand.
pub const LAYOUT_DENSE_ROW_MAJOR_V1: &str = "chop.layout.dense-row-major.v1";

/// The shared scale-format identifier for an operand that carries no scale.
pub const SCALE_NONE_V1: &str = "chop.scale.none.v1";

/// How many values fill one whole aligned block of this contract's operands.
///
/// Both operands are aligned to [`GENERIC_COPY_F32_ALIGNMENT_V1`], so this
/// is the granularity a host that copies whole aligned blocks serves. It is a
/// property of the contract, not a policy: which multiples of it a given host
/// offers is that host catalog's own decision.
pub const GENERIC_COPY_F32_BLOCK_ELEMENTS_V1: u32 =
    (GENERIC_COPY_F32_ALIGNMENT_V1 / GENERIC_COPY_F32_ELEMENT_BYTES_V1) as u32;

/// How many bytes one operand of the copy holds for `elements` values.
#[must_use]
pub const fn generic_copy_f32_bytes_v1(elements: u32) -> u64 {
    elements as u64 * GENERIC_COPY_F32_ELEMENT_BYTES_V1
}

/// The operation key a bundle requires this primitive under.
pub fn generic_copy_f32_key_v1() -> Result<BundleOperationKeyV1, BundleError> {
    Ok(BundleOperationKeyV1::new(
        id(GENERIC_COPY_F32_OPERATION_V1)?,
        id(GENERIC_COPY_F32_VARIANT_V1)?,
    ))
}

/// The identifier a host catalog offers this primitive under, at one width.
///
/// The width is part of the semantic key, so a catalog that serves several
/// widths declares several implementations and each needs its own identity.
/// Naming them all after the primitive is what ties a resolved implementation
/// back to the primitive the bundle's provisioning asked for.
pub fn generic_copy_f32_implementation_id_v1(elements: u32) -> Result<BundleOpaqueId, BundleError> {
    id(&format!(
        "{GENERIC_COPY_F32_PRIMITIVE_V1}.elements-{elements}"
    ))
}

/// The complete semantic contract of the exact F32 copy of `elements` values.
///
/// A host catalog builds the same contract to declare the widths it serves, and
/// resolution is exact: the operation ABI and its version, both operand roles,
/// their dimensions, dtype, layout, scale, alignment, access and aliasing, the
/// ordering of the destination on the source, the scalar argument, and the
/// numerical contract all take part in the match.
pub fn generic_copy_f32_contract_v1(
    elements: u32,
) -> Result<BundleSemanticContractV1, BundleError> {
    BundleSemanticContractV1::new(
        id(GENERIC_COPY_F32_OPERATION_V1)?,
        GENERIC_COPY_F32_ABI_VERSION_V1,
        vec![
            copy_operand_v1(
                GENERIC_COPY_F32_SOURCE_ROLE_V1,
                elements,
                BundleTensorAccessV1::ReadOnly,
                Vec::new(),
            )?,
            copy_operand_v1(
                GENERIC_COPY_F32_DESTINATION_ROLE_V1,
                elements,
                BundleTensorAccessV1::WriteOnly,
                vec![BundleTensorDependencyV1::new(
                    0,
                    id(GENERIC_COPY_F32_SOURCE_ROLE_V1)?,
                )],
            )?,
        ],
        vec![BundleScalarArgumentV1::new(
            id(GENERIC_COPY_F32_ELEMENT_COUNT_V1)?,
            id(DTYPE_U32_V1)?,
            32,
        )?],
        id(NUMERICS_EXACT_COPY_V1)?,
        Vec::new(),
    )
}

fn copy_operand_v1(
    role: &str,
    elements: u32,
    access: BundleTensorAccessV1,
    dependencies: Vec<BundleTensorDependencyV1>,
) -> Result<BundleTensorRequirementV1, BundleError> {
    BundleTensorRequirementV1::new(
        id(role)?,
        vec![u64::from(elements)],
        id(DTYPE_F32_V1)?,
        id(LAYOUT_DENSE_ROW_MAJOR_V1)?,
        id(SCALE_NONE_V1)?,
        Vec::new(),
        GENERIC_COPY_F32_ALIGNMENT_V1,
        access,
        BundleTensorAliasV1::Disjoint,
        dependencies,
    )
}

fn id(value: &str) -> Result<BundleOpaqueId, BundleError> {
    BundleOpaqueId::new(value)
}

#[cfg(test)]
mod tests {
    use super::{
        GENERIC_COPY_F32_BLOCK_ELEMENTS_V1, GENERIC_COPY_F32_PRIMITIVE_V1,
        generic_copy_f32_bytes_v1, generic_copy_f32_contract_v1,
        generic_copy_f32_implementation_id_v1,
    };

    /// The width is part of the contract, so a host declares the widths it
    /// serves rather than one copy that matches every bundle.
    #[test]
    fn two_widths_are_two_operations() {
        let narrow = generic_copy_f32_contract_v1(64)
            .expect("the 64-value contract is valid")
            .semantic_key()
            .expect("the 64-value contract has a key");
        let wide = generic_copy_f32_contract_v1(128)
            .expect("the 128-value contract is valid")
            .semantic_key()
            .expect("the 128-value contract has a key");

        assert_ne!(narrow, wide);
        assert_eq!(
            narrow,
            generic_copy_f32_contract_v1(64)
                .expect("the 64-value contract is valid")
                .semantic_key()
                .expect("the 64-value contract has a key")
        );
        assert_eq!(generic_copy_f32_bytes_v1(64), 256);
    }

    /// A width of zero describes no operand, so there is no contract for it and
    /// therefore no candidate a catalog could offer.
    #[test]
    fn a_zero_width_is_not_a_contract() {
        assert!(generic_copy_f32_contract_v1(0).is_err());
    }

    /// One aligned block is what the declared alignment holds, and every
    /// implementation identity names both the primitive and its width.
    #[test]
    fn a_catalog_identity_names_the_primitive_and_the_width() {
        assert_eq!(GENERIC_COPY_F32_BLOCK_ELEMENTS_V1, 64);

        let identity = generic_copy_f32_implementation_id_v1(64)
            .expect("the 64-value implementation identity is valid");

        assert!(identity.as_str().starts_with(GENERIC_COPY_F32_PRIMITIVE_V1));
        assert_ne!(
            identity,
            generic_copy_f32_implementation_id_v1(128)
                .expect("the 128-value implementation identity is valid")
        );
    }
}
