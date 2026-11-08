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
/// Candidate, with [`WIDE_VERIFY`]: a Flash TP4 drafter step of more than
/// twelve drafting items runs them in the fewest forwards the widest wide
/// verify bucket holds (five block rows an item), on the wide verify
/// buckets, instead of chunks of at most twelve.
pub const WIDE_DRAFT: &str = "wide_draft";
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
/// Candidate: DSpark drafter confidence heads load each thread's weight
/// chain into registers before the dependency wait and stage each row's
/// operands in shared memory with coalesced loads, keeping every chain,
/// fold and score.
pub const DSPARK_CONFIDENCE_STAGED: &str = "dspark_confidence_staged";
/// Candidate: DSpark drafter verify forwards whose rows the measured
/// drafter table admits (`moe.direct_drafter.verify`) run the routed
/// experts through the direct path instead of the expert chains.
pub const DSPARK_MOE_DIRECT: &str = "dspark_moe_direct";
/// Candidate, with `dspark_hc_fusion`: DSpark drafter forwards of one
/// to six items (5 to 30 block rows) admit the HC finish route through
/// the drafter table (`hc.draft_admit.verify`), so their stage boundaries
/// fuse as the measured rows' do.
pub const DSPARK_HC_DRAFT_ROWS: &str = "dspark_hc_draft_rows";
/// Candidate, on by default: a prefix capture or restore moves its
/// snapshot's pieces on each rank with one upload of the snapshot table and
/// one launch of the shared snapshot copy kernel, instead of a device copy
/// per piece and page fragment. The header actions, the drain of every rank
/// and the refusals stay as they are.
pub const PREFIX_SNAPSHOT_COPY_KERNEL: &str = "prefix_snapshot_copy_kernel";
/// Candidate: the Flash TP4 DSpark main projection runs the compact
/// admission and compute pair at the inner rows the candidate table admits
/// (`dense_prefill.dspark_main_short.prefill`, 65 to 128) instead of the
/// generic fold. Every row's bytes are unchanged.
pub const DSPARK_MAIN_SHORT_ROWS: &str = "dspark_main_short_rows";
/// Maximum prompt-ending items whose interior grid anchor one step retains.
pub const MAX_ANCHOR_ITEMS: u32 = 8;
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
    pub wide_draft: bool,
    pub phased_decode_reduce: bool,
    pub dspark_markov_staged: bool,
    pub dspark_hc_fusion: bool,
    pub dspark_confidence_staged: bool,
    pub dspark_moe_direct: bool,
    pub dspark_hc_draft_rows: bool,
    pub prefix_snapshot_copy_kernel: bool,
    pub dspark_main_short_rows: bool,
}

pub fn declarations() -> [ModelOptionV1; 15] {
    [
        (
            DIAGNOSTIC_STEP_LOGS,
            OptionDefinitionV1::Boolean { default: false },
        ),
        (
            DSPARK_CONFIDENCE_STAGED,
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
            DSPARK_HC_DRAFT_ROWS,
            OptionDefinitionV1::Boolean { default: false },
        ),
        (
            DSPARK_HC_FUSION,
            OptionDefinitionV1::Boolean { default: false },
        ),
        (
            DSPARK_MAIN_SHORT_ROWS,
            OptionDefinitionV1::Boolean { default: false },
        ),
        (
            DSPARK_MARKOV_STAGED,
            OptionDefinitionV1::Boolean { default: false },
        ),
        (
            DSPARK_MOE_DIRECT,
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
            PREFIX_SNAPSHOT_COPY_KERNEL,
            OptionDefinitionV1::Boolean { default: true },
        ),
        (
            SHARED_OVERLAP_MAX_ROWS,
            OptionDefinitionV1::Integer {
                default: i64::from(u32::MAX),
                minimum: 1,
                maximum: i64::from(u32::MAX),
            },
        ),
        (WIDE_DRAFT, OptionDefinitionV1::Boolean { default: false }),
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
        let flag = |name: &str| values[name].as_bool().expect("checked boolean");
        let count = |name: &str| {
            usize::try_from(values[name].as_u64().expect("checked integer"))
                .map_err(|_| format!("model option {name} exceeds the platform's count range"))
        };
        let wide_verify = flag(WIDE_VERIFY);
        let wide_draft = flag(WIDE_DRAFT);
        if wide_draft && !wide_verify {
            return Err(format!("model option {WIDE_DRAFT} requires {WIDE_VERIFY}"));
        }
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
            wide_verify,
            wide_draft,
            phased_decode_reduce: flag(PHASED_DECODE_REDUCE),
            dspark_hc_draft_rows: flag(DSPARK_HC_DRAFT_ROWS),
            dspark_moe_direct: flag(DSPARK_MOE_DIRECT),
            dspark_confidence_staged: flag(DSPARK_CONFIDENCE_STAGED),
            dspark_hc_fusion: flag(DSPARK_HC_FUSION),
            dspark_markov_staged: flag(DSPARK_MARKOV_STAGED),
            prefix_snapshot_copy_kernel: flag(PREFIX_SNAPSHOT_COPY_KERNEL),
            dspark_main_short_rows: flag(DSPARK_MAIN_SHORT_ROWS),
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
    fn removed_prefill_anchor_items_option_is_rejected() {
        for value in ["0", "8"] {
            let options = BTreeMap::from([("prefill_anchor_items".into(), value.into())]);
            assert!(declaration().resolve(&options).is_err());
            let values = BTreeMap::from([(
                "prefill_anchor_items".into(),
                serde_json::from_str(value).unwrap(),
            )]);
            let geometry = &crate::geometry::FLASH;
            assert!(
                geometry
                    .resolve_model_options(geometry.runtime_defaults(), &values)
                    .is_err()
            );
        }
    }

    #[test]
    fn removed_prefill_tail_folds_option_is_rejected() {
        for value in ["false", "true"] {
            let options = BTreeMap::from([("prefill_tail_folds".into(), value.into())]);
            assert!(declaration().resolve(&options).is_err());
            let values = BTreeMap::from([(
                "prefill_tail_folds".into(),
                serde_json::from_str(value).unwrap(),
            )]);
            let geometry = &crate::geometry::FLASH;
            assert!(
                geometry
                    .resolve_model_options(geometry.runtime_defaults(), &values)
                    .is_err()
            );
        }
    }

    #[test]
    fn removed_compact_main_option_is_rejected() {
        for value in ["false", "true"] {
            let options = BTreeMap::from([("dspark_main_compact".into(), value.into())]);
            assert!(declaration().resolve(&options).is_err());
        }
    }

    /// The direct MoE knob set is served from the tuning table; its four
    /// candidate options are gone.
    #[test]
    fn removed_moe_direct_options_are_rejected() {
        for name in [
            "moe_direct_keep",
            "moe_direct_loads",
            "moe_direct_retire",
            "moe_direct_stream",
        ] {
            for value in ["false", "true"] {
                let options = BTreeMap::from([(name.into(), value.into())]);
                assert!(declaration().resolve(&options).is_err(), "{name}={value}");
            }
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
    fn removed_row_tile_option_is_rejected() {
        for value in ["false", "true"] {
            let options = BTreeMap::from([("f32_row_tiles".into(), value.into())]);
            assert!(declaration().resolve(&options).is_err());
        }
    }

    /// The fused prefill norm-ceil is admitted from 33 rows by the tuning
    /// table; its candidate option is gone.
    #[test]
    fn removed_norm_ceil_rows_option_is_rejected() {
        for value in ["false", "true"] {
            let options = BTreeMap::from([("prefill_norm_ceil_rows".into(), value.into())]);
            assert!(declaration().resolve(&options).is_err());
        }
    }

    /// The MoE producer splits over the owner intervals at any row count by
    /// default; its candidate option is gone.
    #[test]
    fn removed_producer_intervals_option_is_rejected() {
        for value in ["false", "true"] {
            let options = BTreeMap::from([("prefill_dma_producer_intervals".into(), value.into())]);
            assert!(declaration().resolve(&options).is_err());
        }
    }

    /// The compact dense prefill ranges are served from the tuning table;
    /// their candidate option is gone.
    #[test]
    fn removed_compact_dense_rows_option_is_rejected() {
        for value in ["false", "true"] {
            let options = BTreeMap::from([("prefill_compact_dense_rows".into(), value.into())]);
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
    fn removed_mixed_prepared_engram_option_is_rejected() {
        for value in ["false", "true"] {
            let options = BTreeMap::from([("dspark_mixed_prepared_engram".into(), value.into())]);
            assert!(declaration().resolve(&options).is_err());
        }
    }

    #[test]
    fn removed_early_prefill_engram_option_is_rejected() {
        for value in ["false", "true"] {
            let options = BTreeMap::from([("dspark_early_prefill_engram".into(), value.into())]);
            assert!(declaration().resolve(&options).is_err());
        }
    }

    #[test]
    fn removed_early_prefill_plan_option_is_rejected() {
        for value in ["false", "true"] {
            let options = BTreeMap::from([("dspark_early_prefill_plan".into(), value.into())]);
            assert!(declaration().resolve(&options).is_err());
        }
    }

    #[test]
    fn removed_prefill_streamed_lowering_option_is_rejected() {
        for value in ["false", "true"] {
            let options = BTreeMap::from([("prefill_streamed_lowering".into(), value.into())]);
            assert!(declaration().resolve(&options).is_err());
        }
    }

    /// The refused successor prefill candidate's option is gone.
    #[test]
    fn removed_successor_prefill_option_is_rejected() {
        for value in ["false", "true"] {
            let options = BTreeMap::from([("dspark_successor_prefill".into(), value.into())]);
            assert!(declaration().resolve(&options).is_err());
        }
    }

    /// DSpark successors plan up to 16 items; their candidate knob is gone.
    #[test]
    fn removed_successor_items_option_is_rejected() {
        for value in ["8", "16"] {
            let options = BTreeMap::from([("dspark_successor_items".into(), value.into())]);
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
            wide_draft: false,
            phased_decode_reduce: false,
            dspark_hc_draft_rows: false,
            dspark_moe_direct: false,
            dspark_confidence_staged: false,
            dspark_hc_fusion: false,
            dspark_markov_staged: false,
            prefix_snapshot_copy_kernel: true,
            dspark_main_short_rows: false,
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
            (WIDE_DRAFT.into(), "true".into()),
            (PHASED_DECODE_REDUCE.into(), "true".into()),
            (DSPARK_HC_DRAFT_ROWS.into(), "true".into()),
            (DSPARK_MOE_DIRECT.into(), "true".into()),
            (DSPARK_CONFIDENCE_STAGED.into(), "true".into()),
            (DSPARK_HC_FUSION.into(), "true".into()),
            (DSPARK_MARKOV_STAGED.into(), "true".into()),
            (PREFIX_SNAPSHOT_COPY_KERNEL.into(), "false".into()),
            (DSPARK_MAIN_SHORT_ROWS.into(), "true".into()),
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
                wide_draft: true,
                phased_decode_reduce: true,
                dspark_hc_draft_rows: true,
                dspark_moe_direct: true,
                dspark_confidence_staged: true,
                dspark_hc_fusion: true,
                dspark_markov_staged: true,
                prefix_snapshot_copy_kernel: false,
                dspark_main_short_rows: true,
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
            (WIDE_DRAFT, "1"),
            (PHASED_DECODE_REDUCE, "1"),
            (DSPARK_HC_DRAFT_ROWS, "1"),
            (DSPARK_MOE_DIRECT, "1"),
            (DSPARK_CONFIDENCE_STAGED, "1"),
            (DSPARK_HC_FUSION, "1"),
            (DSPARK_MARKOV_STAGED, "1"),
            (PREFIX_SNAPSHOT_COPY_KERNEL, "1"),
            (DSPARK_MAIN_SHORT_ROWS, "1"),
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

    /// The wide drafter forward runs on the wide verify buckets, so the
    /// option alone is refused.
    #[test]
    fn the_wide_draft_requires_the_wide_verify_buckets() {
        let declaration = declaration();
        let alone = declaration
            .resolve(&BTreeMap::from([(WIDE_DRAFT.into(), "true".into())]))
            .unwrap();
        assert_eq!(
            Options::read(&alone).unwrap_err(),
            "model option wide_draft requires wide_verify"
        );
        let both = declaration
            .resolve(&BTreeMap::from([
                (WIDE_DRAFT.into(), "true".into()),
                (WIDE_VERIFY.into(), "true".into()),
            ]))
            .unwrap();
        let options = Options::read(&both).unwrap();
        assert!(options.wide_draft && options.wide_verify);
    }
}
