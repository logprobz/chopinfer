//! Exact dense FP4 scoring with four query rows sharing each key tile.
use super::{position, position_live, position_mma_guard, position_mma_guard_fast};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::{backend::Launch, contract::Refusal};

pub const ROWS: u32 = 4;

pub fn score_symbol(coarse: &str) -> Result<String, Refusal> {
    coarse
        .strip_suffix("_mma_fp4_guard_fast_v1")
        .map(|stem| format!("{stem}_mma_fp4_prefill4_v1"))
        .ok_or(Refusal(
            "prefill scoring requires the coarse guarded source",
        ))
}

pub fn score(entry: &position::Entry) -> Result<EntrySpec, Refusal> {
    let mut spec = position_mma_guard::dense_specs(entry)?.remove(1);
    let call = "chop::position_mma_guard::scores";
    let start = spec
        .body
        .find(call)
        .ok_or(Refusal("missing guarded score call"))?;
    let end = start
        + spec.body[start..]
            .find(");")
            .ok_or(Refusal("unterminated score call"))?
        + 2;
    spec.body = spec.body[start..end].replace(call, "chop::position_mma_prefill::scores");
    spec.symbol = spec
        .symbol
        .replace("_mma_fp4_guard_v1", "_mma_fp4_prefill4_v1");
    Ok(spec)
}

/// Only the score grid changes. The device still admits each row separately.
pub fn remap(entry: &position::Entry, call: &mut Launch) -> Result<(), Refusal> {
    let expected = position_mma_guard_fast::score_symbol(&position_mma_guard::score_symbol(
        &position_live::symbol(&entry.score_symbol()),
    ));
    if call.symbol != expected
        || call.plan.block != [128, 1, 1]
        || call.plan.grid[2] != 1
        || call.plan.grid[0] == 0
        || call.plan.grid[1] == 0
        || call.plan.grid[1] > entry.query.capacity
    {
        return Err(Refusal(
            "prefill score requires a nonempty four-warp row grid",
        ));
    }
    score(entry)?;
    let symbol = score_symbol(&call.symbol)?;
    call.plan.grid[1] = call.plan.grid[1].div_ceil(ROWS);
    call.symbol = symbol;
    Ok(())
}

pub fn cubin_spec(name: impl Into<String>, entries: Vec<EntrySpec>) -> CubinSpec {
    let mut spec = position_mma_guard::cubin_spec(name, entries);
    spec.headers = vec!["#include \"pooled_indexer/position_mma_prefill.cuh\"".into()];
    spec
}
