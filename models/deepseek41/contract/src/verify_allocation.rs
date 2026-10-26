//! Whether the host sends allocated verify steps
//! (`runtime_bundle_abi::VERIFY_ALLOCATION_CAPABILITY_V1`), as the package
//! model option, enabled by default. An instance prepares the allocation's
//! work and memory unless the option is false, and refuses allocated steps
//! when preparation is disabled.

use runtime_bundle::model_options::{ModelOptionV1, OptionDefinitionV1};
use std::collections::BTreeMap;

pub const OPTION: &str = runtime_bundle_abi::VERIFY_ALLOCATION_OPTION_V1;

pub fn declarations() -> [ModelOptionV1; 1] {
    [ModelOptionV1 {
        name: OPTION.into(),
        definition: OptionDefinitionV1::Boolean { default: true },
        default_is_qualified: true,
    }]
}

/// Whether resolved `options` enable allocated verify steps.
pub fn enabled(options: &BTreeMap<String, serde_json::Value>) -> Result<bool, String> {
    match options.get(OPTION) {
        None => Ok(true),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| format!("{OPTION} requires a boolean")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_bundle::model_options::{CAPABILITY, ModelOptionsV1};

    /// On by default, including a record without the option. An explicit
    /// false disables it; anything but a boolean is refused.
    #[test]
    fn the_option_enables_allocated_steps() {
        let resolve = |overrides: &[(&str, &str)]| {
            ModelOptionsV1 {
                schema: CAPABILITY.into(),
                options: declarations().into(),
            }
            .resolve(
                &overrides
                    .iter()
                    .map(|&(name, value)| (name.to_owned(), value.to_owned()))
                    .collect(),
            )
            .unwrap()
        };
        assert_eq!(enabled(&resolve(&[])), Ok(true));
        assert_eq!(enabled(&resolve(&[(OPTION, "true")])), Ok(true));
        assert_eq!(enabled(&resolve(&[(OPTION, "false")])), Ok(false));
        assert_eq!(enabled(&BTreeMap::new()), Ok(true));
        let bad = BTreeMap::from([(OPTION.to_owned(), serde_json::json!(1))]);
        assert!(enabled(&bad).is_err());
    }
}
