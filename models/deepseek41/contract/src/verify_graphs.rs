//! The widest verify step captured in a graph, as the package model option
//! the host's `--speculation-graph-rows` sets. Wider verify steps, up to
//! the instance's verify rows, run without a graph. A create record without
//! the option uses the graph budget of the default DSpark profiles.

use crate::serving::{DECODE_BATCH, MAX_VERIFY_ROWS, VERIFY_DECODE_ROWS};
use runtime_bundle::model_options::{ModelOptionV1, OptionDefinitionV1};
use std::collections::BTreeMap;

pub const GRAPH_ROWS: &str = runtime_bundle_abi::VERIFY_GRAPH_ROWS_OPTION_V1;

/// The package default: every verify width of the default verify rows.
pub const DEFAULT_GRAPH_ROWS: u32 = VERIFY_DECODE_ROWS;

/// The most verify rows a step of the package carries.
pub const MAX_GRAPH_ROWS: u32 = DECODE_BATCH * MAX_VERIFY_ROWS;

pub fn declarations() -> [ModelOptionV1; 1] {
    [ModelOptionV1 {
        name: GRAPH_ROWS.into(),
        definition: OptionDefinitionV1::Integer {
            default: i64::from(DEFAULT_GRAPH_ROWS),
            minimum: 1,
            maximum: i64::from(MAX_GRAPH_ROWS),
        },
        default_is_qualified: true,
    }]
}

/// The graph rows that resolved `options` name.
pub fn graph_rows(options: &BTreeMap<String, serde_json::Value>) -> Result<u32, String> {
    match options.get(GRAPH_ROWS) {
        None => Ok(DEFAULT_GRAPH_ROWS),
        Some(value) => value
            .as_u64()
            .filter(|n| (1..=u64::from(MAX_GRAPH_ROWS)).contains(n))
            .map(|n| n as u32)
            .ok_or_else(|| format!("{GRAPH_ROWS} must be between 1 and {MAX_GRAPH_ROWS}")),
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

    /// The declared default reaches the bundle through the host's
    /// resolution, an override replaces it, and a record without the option
    /// keeps the default rows.
    #[test]
    fn the_declared_default_is_the_resolved_graph_rows() {
        assert_eq!(graph_rows(&resolve(&[])), Ok(64));
        assert_eq!(graph_rows(&resolve(&[(GRAPH_ROWS, "32")])), Ok(32));
        assert_eq!(graph_rows(&BTreeMap::new()), Ok(64));
        for bad in [
            serde_json::json!(0),
            serde_json::json!(193),
            serde_json::json!("64"),
        ] {
            let options = BTreeMap::from([(GRAPH_ROWS.to_owned(), bad)]);
            assert!(graph_rows(&options).is_err());
        }
    }

    #[test]
    fn dspark_profile_graph_rows_are_qualified_defaults() {
        let profiles: serde_json::Value =
            serde_json::from_str(include_str!("../../profiles.json")).unwrap();
        let declaration = ModelOptionsV1 {
            schema: CAPABILITY.into(),
            options: declarations().into(),
        };
        for name in [
            "tp4-dspark",
            "tp4-dspark-full-prefill",
            "tp4-dspark-upper-skip-2048",
        ] {
            let rows = profiles["profiles"][name]["speculation"]["graph_rows"]
                .as_u64()
                .unwrap();
            let resolved = declaration
                .resolve(&BTreeMap::from([(GRAPH_ROWS.into(), rows.to_string())]))
                .unwrap();
            assert!(declaration.is_qualified(&resolved), "{name}");
        }
    }
}
