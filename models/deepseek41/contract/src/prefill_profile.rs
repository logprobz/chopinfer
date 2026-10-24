//! Exact prefill scheduling with the served profiles as package defaults.
//! Missing options preserve legacy execution for older create records.

use runtime_bundle::model_options::{ModelOptionV1, OptionDefinitionV1};
use std::collections::BTreeMap;

pub const CHUNK: &str = "prefill_chunk_tokens";
pub const SKIP: &str = "prefill_upper_skip";

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Profile {
    pub chunk: u32,
    pub skip: bool,
}

pub fn declarations() -> [ModelOptionV1; 2] {
    [
        ModelOptionV1 {
            name: CHUNK.into(),
            definition: OptionDefinitionV1::Integer {
                default: 8192,
                minimum: 0,
                maximum: 8192,
            },
            default_is_qualified: true,
        },
        ModelOptionV1 {
            name: SKIP.into(),
            definition: OptionDefinitionV1::Boolean { default: true },
            default_is_qualified: true,
        },
    ]
}

impl Profile {
    pub fn read(options: &BTreeMap<String, serde_json::Value>) -> Result<Self, String> {
        let chunk = match options.get(CHUNK) {
            None => 0,
            Some(value) => value
                .as_u64()
                .filter(|n| [0, 2048, 8192].contains(n))
                .ok_or("prefill_chunk_tokens must be 0, 2048 or 8192")?
                as u32,
        };
        let skip = match options.get(SKIP) {
            None => false,
            Some(value) => value
                .as_bool()
                .ok_or("prefill_upper_skip must be boolean")?,
        };
        if skip && chunk == 0 {
            return Err("prefill_upper_skip requires guaranteed chunk ends".into());
        }
        Ok(Self { chunk, skip })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn declared_options() -> runtime_bundle::model_options::ModelOptionsV1 {
        runtime_bundle::model_options::ModelOptionsV1 {
            schema: runtime_bundle::model_options::CAPABILITY.into(),
            options: declarations().into(),
        }
    }

    /// The profile's prefill options; its other options have their own
    /// declarations.
    fn profile_overrides(name: &str) -> BTreeMap<String, String> {
        let profiles: serde_json::Value =
            serde_json::from_str(include_str!("../../profiles.json")).unwrap();
        let all: BTreeMap<String, String> = serde_json::from_value(
            profiles["profiles"][name]
                .get("model_options")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({})),
        )
        .unwrap();
        all.into_iter()
            .filter(|(key, _)| [CHUNK, SKIP].contains(&key.as_str()))
            .collect()
    }

    #[test]
    fn package_defaults_qualify_the_default_profiles() {
        let declaration = declared_options();
        let defaults = declaration.resolve(&BTreeMap::new()).unwrap();
        assert_eq!(
            Profile::read(&defaults).unwrap(),
            Profile {
                chunk: 8192,
                skip: true
            }
        );
        for name in ["tp4-ar", "tp4-dspark"] {
            let resolved = declaration.resolve(&profile_overrides(name)).unwrap();
            // The profiles already selected these values before they became
            // package defaults. Their encoded execution options stay equal.
            assert_eq!(resolved, defaults, "{name}");
            assert!(declaration.is_qualified(&resolved), "{name}");
        }
    }

    #[test]
    fn diagnostic_profiles_keep_explicit_unqualified_overrides() {
        let declaration = declared_options();
        for mode in ["ar", "dspark"] {
            for (suffix, chunk, skip) in
                [("full-prefill", 0, false), ("upper-skip-2048", 2048, true)]
            {
                let name = format!("tp4-{mode}-{suffix}");
                let overrides = profile_overrides(&name);
                assert_eq!(overrides.get(CHUNK), Some(&chunk.to_string()), "{name}");
                assert_eq!(overrides.get(SKIP), Some(&skip.to_string()), "{name}");
                let resolved = declaration.resolve(&overrides).unwrap();
                assert_eq!(
                    Profile::read(&resolved).unwrap(),
                    Profile { chunk, skip },
                    "{name}"
                );
                assert!(!declaration.is_qualified(&resolved), "{name}");
            }
        }
    }

    #[test]
    fn absent_options_preserve_legacy_and_skip_requires_a_grid() {
        assert_eq!(Profile::read(&BTreeMap::new()).unwrap(), Profile::default());
        for chunk in [2048, 8192] {
            for skip in [false, true] {
                assert_eq!(
                    Profile::read(&BTreeMap::from([
                        (CHUNK.into(), chunk.into()),
                        (SKIP.into(), skip.into()),
                    ]))
                    .unwrap(),
                    Profile { chunk, skip }
                );
            }
        }
        for chunk in [0, 1, 64, 4096, 16384] {
            assert!(
                Profile::read(&BTreeMap::from([
                    (CHUNK.into(), chunk.into()),
                    (SKIP.into(), true.into()),
                ]))
                .is_err()
            );
        }
    }
}
