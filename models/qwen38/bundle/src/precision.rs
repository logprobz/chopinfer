//! Admission of the publisher's compiled operation inventory. Precision
//! selection happens during preparation; instances retain these supplied
//! bindings instead of selecting new arithmetic during execution.
use kernel_core::precision::capability::{CapabilityDocument, CapabilityInventory};
use model_qwen38_contract::backend::{Kernels, Selected};

pub use crate::projections::Prepared;

/// Publisher-owned modules for prepared mixed projections. Legacy packages
/// have no bindings to these modules and retain their existing acquisition log.
pub(crate) fn mixed_module(name: &str) -> bool {
    <Selected as Kernels>::mixed_cubin(name)
}

pub fn inventory() -> Result<CapabilityInventory, String> {
    let document: CapabilityDocument =
        serde_json::from_slice(crate::preparation_note::capabilities())
            .map_err(|e| e.to_string())?;
    let available = crate::CUBINS
        .iter()
        .flat_map(|c| c.symbols.iter().map(|s| (*s).into()))
        .collect();
    CapabilityInventory::new(
        document,
        available,
        crate::preparation_note::binding_dependency().into(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn portable_capabilities_match_the_embedded_audited_entries() {
        let inventory = inventory().unwrap();
        let expected = model_qwen38_contract::precision::capabilities().unwrap();
        assert_eq!(inventory.document(), &expected);
        let entries: BTreeSet<_> = crate::CUBINS
            .iter()
            .flat_map(|c| c.symbols.iter().map(|s| (*s).to_owned()))
            .collect();
        assert!(expected.entries().is_subset(&entries));
        assert!(
            expected
                .operations
                .iter()
                .all(|op| op.activation.is_none_or(
                    |a| a == kernel_core::precision::capability::ActivationCapability::Bf16
                ))
        );
        assert_eq!(crate::preparation_note::binding_dependency().len(), 64);
    }
}

#[cfg(test)]
mod geometry_tests {
    #[test]
    fn mixed_modules_follow_every_geometry_tag() {
        for geometry in model_qwen38_contract::geometry::GEOMETRIES {
            assert!(super::mixed_module(&format!(
                "{}_precision_dense_v1",
                geometry.kernel_tag
            )));
            assert!(!super::mixed_module(&format!(
                "{}_bundle",
                geometry.kernel_tag
            )));
        }
        for cubin in crate::CUBINS {
            if crate::GEOMETRY_CUBINS
                .iter()
                .any(|(name, id, _)| *name == cubin.name && *id == "flash")
            {
                assert_eq!(
                    super::mixed_module(cubin.name),
                    cubin.name.starts_with("qwen38_precision_")
                );
            }
        }
    }
}
