//! Coupled sampled drafting as a package model option. The package default
//! couples; a create record without the option keeps the legacy greedy
//! drafter.

use runtime_bundle::model_options::{ModelOptionV1, OptionDefinitionV1};
use std::collections::BTreeMap;

/// Whether the DSpark drafter proposes for sampled requests from the target
/// sampler's own noise (`kernel_core::draft_noise`). Greedy requests draft
/// the same tokens either way, and emitted tokens never depend on it: a
/// draft is kept only when it equals the target's sample.
pub const COUPLED_DRAFTS: &str = "dspark_coupled_drafts";

pub fn declarations() -> [ModelOptionV1; 1] {
    [ModelOptionV1 {
        name: COUPLED_DRAFTS.into(),
        definition: OptionDefinitionV1::Boolean { default: true },
        default_is_qualified: true,
    }]
}

/// Whether resolved `options` couple sampled drafts.
pub fn coupled_drafts(options: &BTreeMap<String, serde_json::Value>) -> Result<bool, String> {
    match options.get(COUPLED_DRAFTS) {
        None => Ok(false),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| format!("{COUPLED_DRAFTS} must be boolean")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_bundle::model_options::{CAPABILITY, ModelOptionsV1};

    fn resolve(overrides: &[(&str, &str)]) -> BTreeMap<String, serde_json::Value> {
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
    }

    /// The package default couples through the host's resolution, a profile
    /// or flag override turns it off, and a record without the option keeps
    /// the greedy drafter.
    #[test]
    fn the_declared_default_couples_and_a_record_without_it_does_not() {
        assert_eq!(coupled_drafts(&resolve(&[])), Ok(true));
        assert_eq!(
            coupled_drafts(&resolve(&[(COUPLED_DRAFTS, "false")])),
            Ok(false)
        );
        assert_eq!(coupled_drafts(&BTreeMap::new()), Ok(false));
        for bad in [serde_json::json!(1), serde_json::json!("true")] {
            let options = BTreeMap::from([(COUPLED_DRAFTS.to_owned(), bad)]);
            assert!(coupled_drafts(&options).is_err());
        }
    }
}
