//! Prefill selection whose exact cutoff stops once its boundary bucket is
//! kept whole. The entries are the served live dense and valid compact
//! selectors with only the selection namespace changed; scores, workspace,
//! arguments and grids are unchanged.
use super::{position, position_blocks, position_live, position_valid};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::{
    backend::Launch,
    contract::{LaunchPlan, Refusal},
};

/// The served selector suffixes and their early-stop counterparts.
const SUFFIXES: [(&str, &str); 4] = [
    ("_select_v3_live_v1", "_select_v3_live_early_v1"),
    ("_select_v3_live_merge_v1", "_select_v3_live_merge_early_v1"),
    ("_select_v3_valid_v1", "_select_v3_valid_early_v1"),
    (
        "_select_v3_valid_merge_v1",
        "_select_v3_valid_merge_early_v1",
    ),
];

pub fn symbol(original: &str) -> Result<String, Refusal> {
    SUFFIXES
        .iter()
        .find_map(|(from, to)| {
            original
                .strip_suffix(from)
                .map(|stem| format!("{stem}{to}"))
        })
        .ok_or(Refusal(
            "early selection requires a live dense or valid compact selector",
        ))
}

fn adapt(specs: Vec<EntrySpec>, from: &str, to: &str) -> Result<Vec<EntrySpec>, Refusal> {
    // The first spec scores; the selector and its merge follow.
    specs
        .into_iter()
        .skip(1)
        .map(|mut spec| {
            if !spec.body.contains(from) {
                return Err(Refusal("early selection body changed"));
            }
            spec.symbol = symbol(&spec.symbol)?;
            spec.body = spec.body.replace(from, to);
            Ok(spec)
        })
        .collect()
}

/// The local selector and merge of a dense (all-position) query.
pub fn dense(entry: &position::Entry) -> Result<Vec<EntrySpec>, Refusal> {
    adapt(
        position_live::dense_specs(entry, false)?,
        "chop::position_live::dense_select",
        "chop::position_select_early::dense_select",
    )
}

/// The local selector and merge of a compact (retained-block) query.
pub fn compact(entry: &position_blocks::Entry) -> Result<Vec<EntrySpec>, Refusal> {
    adapt(
        position_valid::compact_specs(entry, false)?,
        "chop::position_valid::compact_select",
        "chop::position_select_early::compact_select",
    )
}

/// The early-stop symbol of one served selector launch. Only the symbol
/// changes, so the launch must keep the served selector's grid.
pub fn target(
    served: &str,
    plan: LaunchPlan,
    rows: u32,
    collective: bool,
) -> Result<String, Refusal> {
    let merge = served.ends_with("_merge_v1");
    if collective
        || plan.block != [256, 1, 1]
        || plan.grid != [if merge { 1 } else { position_live::SHARDS }, rows, 1]
    {
        return Err(Refusal("early selection requires the served selector grid"));
    }
    symbol(served)
}

pub fn remap(call: &mut Launch) -> Result<(), Refusal> {
    call.symbol = target(&call.symbol, call.plan, call.rows, call.collective)?;
    Ok(())
}

pub fn cubin_spec(name: impl Into<String>, entries: Vec<EntrySpec>) -> CubinSpec {
    let mut spec = position_valid::cubin_spec(name, entries);
    spec.headers = vec!["#include \"pooled_indexer/position_select_early.cuh\"".into()];
    spec
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIVE: &str = include_str!("../../cuda/pooled_indexer/position_live.cuh");
    const VALID: &str = include_str!("../../cuda/pooled_indexer/position_valid.cuh");
    const EARLY: &str = include_str!("../../cuda/pooled_indexer/position_select_early.cuh");
    const COMPACT: &str =
        "template<int Page, int Ratio, int MaxTopK, int Block, int Kept, bool Merge>";

    fn between<'a>(text: &'a str, start: &str, end: &str) -> &'a str {
        let first = text.find(start).expect("start marker");
        let last = first + text[first..].find(end).expect("end marker");
        &text[first..last]
    }

    #[test]
    fn copies_differ_from_the_served_selectors_only_in_the_cutoff() {
        let live = between(LIVE, "// A shard retains at most k keys.", COMPACT);
        let valid = between(VALID, COMPACT, "// Only live block maxima");
        assert_eq!(live.matches("cutoff(").count(), 2);
        let expected = format!("{}{valid}", live.replace("cutoff(", "stop_cutoff("));
        let copied = between(
            EARLY,
            "// A shard retains at most k keys.",
            "// End of the copied selectors.",
        );
        assert_eq!(copied, expected);
        // The stop is the only new code: one comparison against the
        // boundary bucket and one early return of its prefix.
        let cutoff = between(
            EARLY,
            "__device__ uint64_t stop_cutoff",
            "// A shard retains",
        );
        assert_eq!(cutoff.matches("stopped").count(), 2);
    }

    #[test]
    fn symbols_map_each_served_selector_once() {
        assert_eq!(
            symbol("chop_position_indexer_deepseek41_r2_all_select_v3_live_v1").unwrap(),
            "chop_position_indexer_deepseek41_r2_all_select_v3_live_early_v1"
        );
        assert_eq!(
            symbol("chop_position_blocks_deepseek41_r1_select_v3_valid_merge_v1").unwrap(),
            "chop_position_blocks_deepseek41_r1_select_v3_valid_merge_early_v1"
        );
        for refused in [
            "chop_position_indexer_deepseek41_r2_all_scores_v3_live_v1",
            "chop_position_indexer_deepseek41_r2_all_select_v3_cached4_v1",
            "chop_position_indexer_deepseek41_r2_all_select_v3_live_early_v1",
        ] {
            assert!(symbol(refused).is_err(), "{refused}");
        }
    }

    #[test]
    fn targets_require_the_served_selector_grid() {
        let local = "chop_position_indexer_deepseek41_r1_all_select_v3_live_v1";
        let merge = "chop_position_indexer_deepseek41_r1_all_select_v3_live_merge_v1";
        let plan = |x, y| LaunchPlan {
            grid: [x, y, 1],
            block: [256, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        assert!(
            target(local, plan(32, 65), 65, false)
                .unwrap()
                .ends_with("_live_early_v1")
        );
        assert!(
            target(merge, plan(1, 65), 65, false)
                .unwrap()
                .ends_with("_merge_early_v1")
        );
        assert!(target(local, plan(1, 65), 65, false).is_err());
        assert!(target(merge, plan(32, 65), 65, false).is_err());
        assert!(target(local, plan(32, 64), 65, false).is_err());
        assert!(target(local, plan(32, 65), 65, true).is_err());
        let mut wide = plan(32, 65);
        wide.block[0] = 128;
        assert!(target(local, wide, 65, false).is_err());
    }
}
