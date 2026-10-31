//! Instance execution bounds and diagnostics, recorded in serving identity.
//! Older create records keep the same defaults when these options are absent.

use runtime_bundle::model_options::{ModelOptionV1, OptionDefinitionV1};
use serde_json::Value;
use std::collections::BTreeMap;

pub const SHARED_OVERLAP_MAX_ROWS: &str = "shared_overlap_max_rows";
pub const PREFILL_PLAN_CACHE_PLANS: &str = "prefill_plan_cache_plans";
pub const PREFILL_PLAN_CACHE_VERIFY: &str = "prefill_plan_cache_verify";
pub const DSPARK_GRAPH_HANDLES_PER_RANK: &str = "dspark_graph_handles_per_rank";
pub const DIAGNOSTIC_STEP_LOGS: &str = "diagnostic_step_logs";
/// Candidate: Flash TP4 verify steps above the 64-row decode route run the
/// wide verify buckets (`crate::verify_bounds::WIDE`).
pub const WIDE_VERIFY: &str = "wide_verify";
/// Candidate: Flash TP4 verify steps of 33 to 64 rows run the measured
/// whole-K exact F32 row tiles (`f32.rows.verify`) of the vocabulary head
/// and the compressor.
pub const F32_ROW_TILES: &str = "f32_row_tiles";
/// Candidate: Flash TP4 decode and verify all-reduces whose round no HC
/// boundary defers take the phased F32 rounding entries.
pub const PHASED_DECODE_REDUCE: &str = "phased_decode_reduce";
/// Candidate: DSpark drafter Markov steps of two or more items load each
/// thread's rank-256 weight chain into registers before the dependency
/// wait and read each position's base logits in place, so no row copy
/// precedes the step.
pub const DSPARK_MARKOV_STAGED: &str = "dspark_markov_staged";
/// Candidate: a DSpark drafter stage's sublayer boundaries, after its
/// attention and into the next stage, fuse their all-reduce round,
/// expansion, next mixer and input norm where the measured finish route
/// admits the forward's rows, as the backbone's do.
pub const DSPARK_HC_FUSION: &str = "dspark_hc_fusion";
pub const DEFAULT_PLANS: usize = 16;
pub const MAX_PLANS: usize = 64;
/// The served warmup fits within 512 graph handles per rank, leaving room
/// for completion tails that serving captures lazily.
pub const DEFAULT_GRAPH_HANDLES_PER_RANK: usize = 512;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Options {
    pub shared_overlap_max_rows: u32,
    pub prefill_plan_cache_plans: usize,
    pub prefill_plan_cache_verify: bool,
    pub dspark_graph_handles_per_rank: usize,
    pub diagnostic_step_logs: bool,
    pub wide_verify: bool,
    pub f32_row_tiles: bool,
    pub phased_decode_reduce: bool,
    pub dspark_markov_staged: bool,
    pub dspark_hc_fusion: bool,
}

pub fn declarations() -> [ModelOptionV1; 10] {
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
            DSPARK_HC_FUSION,
            OptionDefinitionV1::Boolean { default: false },
        ),
        (
            DSPARK_MARKOV_STAGED,
            OptionDefinitionV1::Boolean { default: false },
        ),
        (
            F32_ROW_TILES,
            OptionDefinitionV1::Boolean { default: false },
        ),
        (
            PHASED_DECODE_REDUCE,
            OptionDefinitionV1::Boolean { default: false },
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
            SHARED_OVERLAP_MAX_ROWS,
            OptionDefinitionV1::Integer {
                default: i64::from(u32::MAX),
                minimum: 1,
                maximum: i64::from(u32::MAX),
            },
        ),
        (WIDE_VERIFY, OptionDefinitionV1::Boolean { default: false }),
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
            prefill_plan_cache_verify: values[PREFILL_PLAN_CACHE_VERIFY]
                .as_bool()
                .expect("checked boolean"),
            dspark_graph_handles_per_rank: count(DSPARK_GRAPH_HANDLES_PER_RANK)?,
            diagnostic_step_logs: values[DIAGNOSTIC_STEP_LOGS]
                .as_bool()
                .expect("checked boolean"),
            wide_verify: values[WIDE_VERIFY].as_bool().expect("checked boolean"),
            f32_row_tiles: values[F32_ROW_TILES].as_bool().expect("checked boolean"),
            phased_decode_reduce: values[PHASED_DECODE_REDUCE]
                .as_bool()
                .expect("checked boolean"),
            dspark_hc_fusion: values[DSPARK_HC_FUSION].as_bool().expect("checked boolean"),
            dspark_markov_staged: values[DSPARK_MARKOV_STAGED]
                .as_bool()
                .expect("checked boolean"),
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
    fn removed_compact_main_option_is_rejected() {
        for value in ["false", "true"] {
            let options = BTreeMap::from([("dspark_main_compact".into(), value.into())]);
            assert!(declaration().resolve(&options).is_err());
        }
    }

    #[test]
    fn removed_prefill_anchor_option_is_rejected() {
        for value in ["0", "8"] {
            let options = BTreeMap::from([("prefill_anchor_items".into(), value.into())]);
            assert!(declaration().resolve(&options).is_err());
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
    fn removed_plan_template_option_is_rejected() {
        for value in ["0", "64"] {
            let options = BTreeMap::from([("prefill_plan_templates".into(), value.into())]);
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
            dspark_graph_handles_per_rank: 512,
            diagnostic_step_logs: false,
            wide_verify: false,
            f32_row_tiles: false,
            phased_decode_reduce: false,
            dspark_hc_fusion: false,
            dspark_markov_staged: false,
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
            (DSPARK_GRAPH_HANDLES_PER_RANK.into(), "0".into()),
            (DIAGNOSTIC_STEP_LOGS.into(), "true".into()),
            (WIDE_VERIFY.into(), "true".into()),
            (F32_ROW_TILES.into(), "true".into()),
            (PHASED_DECODE_REDUCE.into(), "true".into()),
            (DSPARK_HC_FUSION.into(), "true".into()),
            (DSPARK_MARKOV_STAGED.into(), "true".into()),
        ]);
        let resolved = declaration.resolve(&overrides).unwrap();
        assert_eq!(
            Options::read(&resolved).unwrap(),
            Options {
                shared_overlap_max_rows: 8,
                prefill_plan_cache_plans: 2,
                prefill_plan_cache_verify: true,
                dspark_graph_handles_per_rank: 0,
                diagnostic_step_logs: true,
                wide_verify: true,
                f32_row_tiles: true,
                phased_decode_reduce: true,
                dspark_hc_fusion: true,
                dspark_markov_staged: true,
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
            (DIAGNOSTIC_STEP_LOGS, "\"true\""),
            (WIDE_VERIFY, "1"),
            (F32_ROW_TILES, "1"),
            (PHASED_DECODE_REDUCE, "1"),
            (DSPARK_HC_FUSION, "1"),
            (DSPARK_MARKOV_STAGED, "1"),
            (DSPARK_GRAPH_HANDLES_PER_RANK, "-1"),
            (DSPARK_GRAPH_HANDLES_PER_RANK, "9223372036854775808"),
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
            (DSPARK_GRAPH_HANDLES_PER_RANK, 0),
            (DSPARK_GRAPH_HANDLES_PER_RANK, i64::MAX),
        ] {
            assert!(Options::read(&BTreeMap::from([(name.into(), value.into())])).is_ok());
        }
    }
}
