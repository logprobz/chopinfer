//! Constraints over authenticated, fixed package precision. No resolution or
//! materialization is performed here, and process defaults cannot alter it.
use crate::{
    CompilerError,
    quantization::{ActivationPolicy, identity::report},
};
use kernel_core::precision::component::{COMPONENT_FORMAT, MAX_COMPONENT_BYTES, decode};
use runtime_bundle::BundlePackageV1;

pub fn check(
    package: &BundlePackageV1,
    policy: ActivationPolicy,
) -> Result<Option<Vec<String>>, CompilerError> {
    // One precision component per topology the package serves.
    let components: Vec<_> = package
        .manifest()
        .components()
        .iter()
        .filter(|component| component.format().as_str() == COMPONENT_FORMAT)
        .collect();
    if components.is_empty() {
        return Ok(None);
    }
    let mut choices = vec![format!(
        "fixed package satisfies activation policy {policy}"
    )];
    for component in &components {
        if component.byte_length() > MAX_COMPONENT_BYTES as u64 {
            return Err(CompilerError::validation(
                "precision component exceeds its authenticated byte bound",
            ));
        }
        let document = decode(&package.read_component(component)?)?;
        for selection in &document.operations {
            selection.require_policy(policy)?;
        }
        for line in report(&document)? {
            choices.push(if components.len() > 1 {
                format!("{}: {line}", component.id())
            } else {
                line
            });
        }
    }
    Ok(Some(choices))
}
