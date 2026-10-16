//! Checkpoint interpretation and preparation-time precision selection shared by
//! model compilers. No model catalog, device toolchain or serving runtime lives here.

pub mod bank;
pub mod declared;
pub mod format;
pub mod identity;
pub mod import;
pub mod package;
pub mod packing;
pub mod payload;
pub mod policy;
pub mod projection;
pub mod source;
pub mod table;

pub use kernel_core::precision;
pub use policy::{ActivationPolicy, PreparedOperation, resolve_operation};

/// Preserve absence for legacy entry points while applying the common explicit
/// argument, environment default and UTF-8 rules. Callers default None to auto.
pub fn requested_policy(
    explicit: Option<&str>,
) -> Result<Option<ActivationPolicy>, crate::CompilerError> {
    if let Some(value) = explicit {
        return Ok(Some(value.parse()?));
    }
    std::env::var_os(policy::ACTIVATION_POLICY_ENV)
        .map(|value| {
            let value = value
                .into_string()
                .map_err(|_| crate::CompilerError::validation("activation policy must be UTF-8"))?;
            Ok(value.parse()?)
        })
        .transpose()
}
