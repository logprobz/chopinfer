//! Instance execution bounds and diagnostics, recorded in serving identity.
//! Older create records keep the same defaults when these options are absent.

use runtime_bundle::model_options::{ModelOptionV1, OptionDefinitionV1};
use serde_json::Value;
use std::collections::BTreeMap;

pub const SHARED_OVERLAP_MAX_ROWS: &str = "shared_overlap_max_rows";
pub const PREFILL_PLAN_CACHE_PLANS: &str = "prefill_plan_cache_plans";
pub const PREFILL_PLAN_CACHE_VERIFY: &str = "prefill_plan_cache_verify";
/// The candidate switch of upper-skip plan templates: the most templates
/// kept at once, 0 (the default) lowering every plan the short-prompt cache
/// does not keep in full.
pub const PREFILL_PLAN_TEMPLATES: &str = "prefill_plan_templates";
pub const DSPARK_GRAPH_HANDLES_PER_RANK: &str = "dspark_graph_handles_per_rank";
pub const DIAGNOSTIC_STEP_LOGS: &str = "diagnostic_step_logs";
pub const PREFILL_ANCHOR_ITEMS: &str = "prefill_anchor_items";
pub const DEFAULT_PLANS: usize = 16;
pub const MAX_PLANS: usize = 64;
pub const MAX_PREFILL_PLAN_TEMPLATES: usize = 64;
/// The served warmup fits within 512 graph handles per rank, leaving room
/// for completion tails that serving captures lazily.
pub const DEFAULT_GRAPH_HANDLES_PER_RANK: usize = 512;
/// The most prompt-ending items one upper-skip step keeps at their rounded
/// prompt end in anchor spares. Each spare holds one slot's captured state:
/// rings, tails and, with DSpark, the raw feature history.
pub const MAX_ANCHOR_ITEMS: u32 = 8;
/// Smallest anchored prompt in the measured cold single/packed shapes.
/// Short widths retain split routes until wider fused routes are qualified.
pub const MIN_ANCHOR_ROWS: u32 = 357;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Options {
    pub shared_overlap_max_rows: u32,
    pub prefill_plan_cache_plans: usize,
    pub prefill_plan_cache_verify: bool,
    pub prefill_plan_templates: usize,
    pub dspark_graph_handles_per_rank: usize,
    pub diagnostic_step_logs: bool,
    /// Anchor spares of an upper-skip instance; 0 keeps every prompt-end
    /// anchor at a chunk end.
    pub prefill_anchor_items: u32,
}

pub fn declarations() -> [ModelOptionV1; 7] {
    [
        (
            DIAGNOSTIC_STEP_LOGS,
            OptionDefinitionV1::Boolean { default: false },
        ),
        (
            DSPARK_GRAPH_HANDLES_PER_RANK,
            OptionDefinitionV1::Integer {
                default: DEFAULT_GRAPH_HANDLES_PER_RANK as i64,
                minimum: 0,
                maximum: i64::MAX,
            },
        ),
        (
            PREFILL_ANCHOR_ITEMS,
            OptionDefinitionV1::Integer {
                default: 0,
                minimum: 0,
                maximum: i64::from(MAX_ANCHOR_ITEMS),
            },
        ),
        (
            PREFILL_PLAN_CACHE_PLANS,
            OptionDefinitionV1::Integer {
                default: DEFAULT_PLANS as i64,
                minimum: 1,
                maximum: MAX_PLANS as i64,
            },
        ),
        (
            PREFILL_PLAN_CACHE_VERIFY,
            OptionDefinitionV1::Boolean { default: false },
        ),
        (
            PREFILL_PLAN_TEMPLATES,
            OptionDefinitionV1::Integer {
                default: 0,
                minimum: 0,
                maximum: MAX_PREFILL_PLAN_TEMPLATES as i64,
            },
        ),
        (
            SHARED_OVERLAP_MAX_ROWS,
            OptionDefinitionV1::Integer {
                default: i64::from(u32::MAX),
                minimum: 1,
                maximum: i64::from(u32::MAX),
            },
        ),
    ]
    .map(|(name, definition)| ModelOptionV1 {
        name: name.into(),
        definition,
        default_is_qualified: true,
    })
}

impl Options {
    pub fn read(options: &BTreeMap<String, Value>) -> Result<Self, String> {
        let values = declarations()
            .into_iter()
            .map(|option| {
                let value = options
                    .get(&option.name)
                    .cloned()
                    .unwrap_or_else(|| option.definition.default_value());
                if !option.definition.accepts(&value) {
                    return Err(format!(
                        "model option {} has the wrong type or is outside its allowed range",
                        option.name
                    ));
                }
                Ok((option.name, value))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let count = |name: &str| {
            usize::try_from(values[name].as_u64().expect("checked integer"))
                .map_err(|_| format!("model option {name} exceeds the platform's count range"))
        };
        Ok(Self {
            shared_overlap_max_rows: values[SHARED_OVERLAP_MAX_ROWS]
                .as_u64()
                .expect("checked u32") as u32,
            prefill_plan_cache_plans: count(PREFILL_PLAN_CACHE_PLANS)?,
            prefill_plan_templates: count(PREFILL_PLAN_TEMPLATES)?,
            prefill_plan_cache_verify: values[PREFILL_PLAN_CACHE_VERIFY]
                .as_bool()
                .expect("checked boolean"),
            dspark_graph_handles_per_rank: count(DSPARK_GRAPH_HANDLES_PER_RANK)?,
            diagnostic_step_logs: values[DIAGNOSTIC_STEP_LOGS]
                .as_bool()
                .expect("checked boolean"),
            prefill_anchor_items: values[PREFILL_ANCHOR_ITEMS]
                .as_u64()
                .expect("checked integer") as u32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_bundle::model_options::{CAPABILITY, ModelOptionsV1};

    fn declaration() -> ModelOptionsV1 {
        ModelOptionsV1 {
            schema: CAPABILITY.into(),
            options: declarations().into(),
        }
    }

    #[test]
    fn removed_short_prefill_graph_option_is_rejected() {
        for value in ["0", "16"] {
            let options = BTreeMap::from([("prefill_graph_programs".into(), value.into())]);
            assert!(declaration().resolve(&options).is_err());
        }
    }

    #[test]
    fn declared_and_legacy_defaults_keep_the_served_bounds_and_quiet_logs() {
        let declaration = declaration();
        let defaults = declaration.resolve(&BTreeMap::new()).unwrap();
        let expected = Options {
            shared_overlap_max_rows: u32::MAX,
            prefill_plan_cache_plans: 16,
            prefill_plan_cache_verify: false,
            prefill_plan_templates: 0,
            dspark_graph_handles_per_rank: 512,
            diagnostic_step_logs: false,
            prefill_anchor_items: 0,
        };
        assert_eq!(Options::read(&defaults).unwrap(), expected);
        assert_eq!(Options::read(&BTreeMap::new()).unwrap(), expected);
        assert!(declaration.is_qualified(&defaults));
    }

    #[test]
    fn overrides_reach_the_reader_and_make_identity_unqualified() {
        let declaration = declaration();
        let overrides = BTreeMap::from([
            (SHARED_OVERLAP_MAX_ROWS.into(), "8".into()),
            (PREFILL_PLAN_CACHE_PLANS.into(), "2".into()),
            (PREFILL_PLAN_CACHE_VERIFY.into(), "true".into()),
            (PREFILL_PLAN_TEMPLATES.into(), "4".into()),
            (DSPARK_GRAPH_HANDLES_PER_RANK.into(), "0".into()),
            (DIAGNOSTIC_STEP_LOGS.into(), "true".into()),
            (PREFILL_ANCHOR_ITEMS.into(), "2".into()),
        ]);
        let resolved = declaration.resolve(&overrides).unwrap();
        assert_eq!(
            Options::read(&resolved).unwrap(),
            Options {
                shared_overlap_max_rows: 8,
                prefill_plan_cache_plans: 2,
                prefill_plan_cache_verify: true,
                prefill_plan_templates: 4,
                dspark_graph_handles_per_rank: 0,
                diagnostic_step_logs: true,
                prefill_anchor_items: 2,
            }
        );
        assert!(!declaration.is_qualified(&resolved));
    }

    #[test]
    fn host_and_bundle_refuse_wrong_types_and_out_of_range_values() {
        let declaration = declaration();
        for (name, bad) in [
            (SHARED_OVERLAP_MAX_ROWS, "0"),
            (SHARED_OVERLAP_MAX_ROWS, "4294967296"),
            (PREFILL_PLAN_CACHE_PLANS, "0"),
            (PREFILL_PLAN_CACHE_PLANS, "65"),
            (PREFILL_PLAN_CACHE_PLANS, "1.5"),
            (PREFILL_PLAN_CACHE_VERIFY, "1"),
            (PREFILL_PLAN_TEMPLATES, "-1"),
            (PREFILL_PLAN_TEMPLATES, "65"),
            (DIAGNOSTIC_STEP_LOGS, "\"true\""),
            (DSPARK_GRAPH_HANDLES_PER_RANK, "-1"),
            (DSPARK_GRAPH_HANDLES_PER_RANK, "9223372036854775808"),
            (PREFILL_ANCHOR_ITEMS, "-1"),
            (PREFILL_ANCHOR_ITEMS, "9"),
        ] {
            assert!(
                declaration
                    .resolve(&BTreeMap::from([(name.into(), bad.into())]))
                    .is_err()
            );
            let value = serde_json::from_str(bad).unwrap();
            assert!(Options::read(&BTreeMap::from([(name.into(), value)])).is_err());
        }
        for (name, value) in [
            (SHARED_OVERLAP_MAX_ROWS, 1),
            (SHARED_OVERLAP_MAX_ROWS, i64::from(u32::MAX)),
            (PREFILL_PLAN_CACHE_PLANS, 1),
            (PREFILL_PLAN_CACHE_PLANS, 64),
            (PREFILL_PLAN_TEMPLATES, 0),
            (PREFILL_PLAN_TEMPLATES, 64),
            (DSPARK_GRAPH_HANDLES_PER_RANK, 0),
            (DSPARK_GRAPH_HANDLES_PER_RANK, i64::MAX),
            (PREFILL_ANCHOR_ITEMS, 0),
            (PREFILL_ANCHOR_ITEMS, 8),
        ] {
            assert!(Options::read(&BTreeMap::from([(name.into(), value.into())])).is_ok());
        }
    }
}
