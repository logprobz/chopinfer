//! Whether the host sends allocated verify steps
//! (`runtime_bundle_abi::VERIFY_ALLOCATION_CAPABILITY_V1`), as the package
//! model option the host sets when its confidence calibration applies. An
//! instance prepares the allocation's work and memory only then, and
//! refuses allocated steps otherwise.

use runtime_bundle::model_options::{ModelOptionV1, OptionDefinitionV1};
use std::collections::BTreeMap;

pub const OPTION: &str = runtime_bundle_abi::VERIFY_ALLOCATION_OPTION_V1;

pub fn declarations() -> [ModelOptionV1; 1] {
    [ModelOptionV1 {
        name: OPTION.into(),
        definition: OptionDefinitionV1::Boolean { default: false },
        default_is_qualified: true,
    }]
}

/// Whether resolved `options` enable allocated verify steps.
pub fn enabled(options: &BTreeMap<String, serde_json::Value>) -> Result<bool, String> {
    match options.get(OPTION) {
        None => Ok(false),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| format!("{OPTION} requires a boolean")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_bundle::model_options::{CAPABILITY, ModelOptionsV1};

    /// Off by default and in a record without the option; the host's
    /// "true" enables it; anything but a boolean is refused.
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
        assert_eq!(enabled(&resolve(&[])), Ok(false));
        assert_eq!(enabled(&resolve(&[(OPTION, "true")])), Ok(true));
        assert_eq!(enabled(&BTreeMap::new()), Ok(false));
        let bad = BTreeMap::from([(OPTION.to_owned(), serde_json::json!(1))]);
        assert!(enabled(&bad).is_err());
    }
}
