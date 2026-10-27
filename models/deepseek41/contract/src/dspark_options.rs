//! The DSpark drafter's served settings as package model options. A create
//! record without them keeps the declared defaults.

use crate::serving::{DECODE_BATCH, MAX_DRAFT_TOKENS};
use runtime_bundle::model_options::{ModelOptionV1, OptionDefinitionV1};
use std::collections::BTreeMap;

/// Most drafting items one drafter forward runs; a step with more drafting
/// items splits into the fewest near-equal chunks within this limit. Target outputs never
/// depend on it, and the diagnostic build checks chunked drafts against one
/// whole forward.
pub const DRAFT_CHUNK_ITEMS: &str = "dspark_draft_chunk_items";

/// The package default and the chunk a create record without the option
/// keeps: at most twelve items of five block rows use the small-row paths.
pub const DEFAULT_DRAFT_CHUNK_ITEMS: u32 = 12;

/// Most drafting items one drafter forward keeps on the small-row paths:
/// the row-local quantizers, the HC mix split and the head's weight reuse,
/// whose per-row arithmetic does not depend on the forward's rows. A larger
/// chunk is served by the prefill-sized forms with the same bits.
pub const SMALL_DRAFT_ITEMS: u32 = 12;

/// The block rows of [`SMALL_DRAFT_ITEMS`]: five per drafting item.
pub const SMALL_DRAFT_ROWS: u32 = SMALL_DRAFT_ITEMS * MAX_DRAFT_TOKENS;

/// Default-off candidate: compact allocated vocabulary heads and mapped sampling.
pub const COMPACT_ALLOCATION: &str = "dspark_compact_allocation";

pub fn compact_allocation(options: &BTreeMap<String, serde_json::Value>) -> Result<bool, String> {
    options.get(COMPACT_ALLOCATION).map_or(Ok(false), |v| {
        v.as_bool()
            .ok_or_else(|| format!("{COMPACT_ALLOCATION} must be boolean"))
    })
}

pub fn declarations() -> [ModelOptionV1; 2] {
    [
        ModelOptionV1 {
            name: COMPACT_ALLOCATION.into(),
            definition: OptionDefinitionV1::Boolean { default: false },
            default_is_qualified: true,
        },
        ModelOptionV1 {
            name: DRAFT_CHUNK_ITEMS.into(),
            definition: OptionDefinitionV1::Integer {
                default: i64::from(DEFAULT_DRAFT_CHUNK_ITEMS),
                minimum: 1,
                maximum: i64::from(DECODE_BATCH),
            },
            default_is_qualified: true,
        },
    ]
}

/// The drafting items per drafter forward that resolved `options` name.
pub fn draft_chunk_items(options: &BTreeMap<String, serde_json::Value>) -> Result<u32, String> {
    match options.get(DRAFT_CHUNK_ITEMS) {
        None => Ok(DEFAULT_DRAFT_CHUNK_ITEMS),
        Some(value) => value
            .as_u64()
            .filter(|n| (1..=u64::from(DECODE_BATCH)).contains(n))
            .map(|n| n as u32)
            .ok_or_else(|| format!("{DRAFT_CHUNK_ITEMS} must be between 1 and {DECODE_BATCH}")),
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

    /// The package default reaches the drafter through the host's
    /// resolution, a profile or flag override replaces it, and a record
    /// without the option keeps the declared default.
    #[test]
    fn the_declared_default_is_the_resolved_chunk() {
        assert_eq!(draft_chunk_items(&resolve(&[])), Ok(12));
        assert_eq!(
            draft_chunk_items(&resolve(&[(DRAFT_CHUNK_ITEMS, "6")])),
            Ok(6)
        );
        assert_eq!(draft_chunk_items(&BTreeMap::new()), Ok(12));
        for bad in [
            serde_json::json!(0),
            serde_json::json!(33),
            serde_json::json!("8"),
        ] {
            let options = BTreeMap::from([(DRAFT_CHUNK_ITEMS.to_owned(), bad)]);
            assert!(draft_chunk_items(&options).is_err());
        }
    }

    #[test]
    fn compact_allocation_is_an_explicit_default_off_candidate() {
        assert_eq!(compact_allocation(&resolve(&[])), Ok(false));
        assert_eq!(compact_allocation(&BTreeMap::new()), Ok(false));
        assert_eq!(
            compact_allocation(&resolve(&[(COMPACT_ALLOCATION, "true")])),
            Ok(true)
        );
        assert!(
            compact_allocation(&BTreeMap::from([(COMPACT_ALLOCATION.into(), 1.into())])).is_err()
        );
    }
}
