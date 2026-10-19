//! Opt-in exact prefill scheduling. Missing options preserve legacy execution.

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
                default: 0,
                minimum: 0,
                maximum: 8192,
            },
            default_is_qualified: true,
        },
        ModelOptionV1 {
            name: SKIP.into(),
            definition: OptionDefinitionV1::Boolean { default: false },
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
