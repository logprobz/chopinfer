//! Register-cached local selection with score radix and ordered boundary ties.
use super::{position, position_live};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::{backend::Launch, contract::Refusal};

pub fn select_symbol(original: &str) -> Result<String, Refusal> {
    original
        .strip_suffix("_select_v3_live_v1")
        .map(|stem| format!("{stem}_select_v3_cached4_v1"))
        .ok_or(Refusal(
            "cached selection requires a dense live local selector",
        ))
}

pub fn select(entry: &position::Entry) -> Result<EntrySpec, Refusal> {
    let mut spec = position_live::dense_specs(entry, false)?.remove(1);
    spec.symbol = select_symbol(&spec.symbol)?;
    spec.body = spec.body.replace(
        "chop::position_live::dense_select",
        "chop::position_select_cached::dense_select",
    );
    Ok(spec)
}

pub fn remap(call: &mut Launch) -> Result<(), Refusal> {
    if call.plan.block != [256, 1, 1]
        || call.plan.grid[0] != position_live::SHARDS
        || call.plan.grid[2] != 1
    {
        return Err(Refusal(
            "cached selection requires the original local-shard grid",
        ));
    }
    call.symbol = select_symbol(&call.symbol)?;
    Ok(())
}

pub fn cubin_spec(name: impl Into<String>, entries: Vec<EntrySpec>) -> CubinSpec {
    let mut spec = position_live::cubin_spec(name, entries);
    spec.headers = vec!["#include \"pooled_indexer/position_select_cached.cuh\"".into()];
    spec
}
