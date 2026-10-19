//! Four contiguous shard groups retain original keys before an ordered merge.
use super::{position, position_blocks, position_live, position_valid};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::Refusal;

pub const GROUPS: u32 = 4;
pub fn workspace_bytes(rows: u32, kept: u32) -> u64 {
    u64::from(rows) * u64::from(GROUPS) * (u64::from(kept) + 1) * 8
}
pub fn symbol(original: &str, final_stage: bool) -> String {
    format!(
        "{original}_hier_{}_v1",
        if final_stage { "final" } else { "group" }
    )
}
fn adapt(entry: EntrySpec) -> Vec<EntrySpec> {
    [false, true]
        .into_iter()
        .map(|final_stage| {
            let mut e = entry.clone();
            e.symbol = symbol(&e.symbol, final_stage);
            e.body = e
                .body
                .replace(
                    "chop::position_live::dense_select",
                    "chop::position_hierarchical::dense_select",
                )
                .replace(
                    "chop::position_valid::compact_select",
                    "chop::position_hierarchical::compact_select",
                );
            if !final_stage {
                e.body = e.body.replace(",true>(", ",false>(");
            }
            e.params.push(EntryParam::DevicePointer {
                name: "intermediate".into(),
                cxx_type: "uint64_t".into(),
            });
            e.body = e.body.replace(");", ",intermediate);");
            e
        })
        .collect()
}
pub fn dense_specs(entry: &position::Entry) -> Result<Vec<EntrySpec>, Refusal> {
    Ok(adapt(position_live::dense_specs(entry, false)?.remove(2)))
}
pub fn compact_specs(entry: &position_blocks::Entry) -> Result<Vec<EntrySpec>, Refusal> {
    Ok(adapt(
        position_valid::compact_specs(entry, false)?.remove(2),
    ))
}
pub fn negative(mut group: EntrySpec) -> EntrySpec {
    assert!(group.symbol.ends_with("_hier_group_v1"));
    group.symbol.push_str("_negative_v1");
    group.body = group.body.replace(",false>(", ",false,true>(");
    group
}
pub fn cubin_spec(name: impl Into<String>, entries: Vec<EntrySpec>) -> CubinSpec {
    let mut spec = position_live::cubin_spec(name, entries);
    spec.headers = vec!["#include \"pooled_indexer/position_hierarchical.cuh\"".into()];
    spec
}
