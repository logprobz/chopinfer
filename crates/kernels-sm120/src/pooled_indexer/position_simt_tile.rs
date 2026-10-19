//! Register-local score trees with shared query and key tiles.
use super::{position, position_live};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::{
    backend::Launch,
    contract::{Argument, Refusal},
};
pub const ROWS: u32 = 4;
pub const COLUMNS: u32 = 64;
pub fn symbol(original: &str) -> String {
    format!(
        "{}_simt_tile_v1",
        original.strip_suffix("_live_v1").unwrap_or(original)
    )
}
pub fn score_spec(
    entry: &position::Entry,
    bf16: bool,
    control: bool,
) -> Result<EntrySpec, Refusal> {
    if entry.query.heads != 32
        || entry.query.head_dim != 128
        || entry.query.candidates != kernel_core::pooled_indexer::IndexerCandidates::All
    {
        return Err(Refusal(
            "register-tiled position scores require 32 heads of 128",
        ));
    }
    let mut spec = position_live::dense_specs(entry, bf16)?.remove(0);
    spec.symbol = symbol(&spec.symbol);
    if control {
        spec.symbol = spec
            .symbol
            .replace("_simt_tile_v1", "_simt_tile_wrong_tree_v1");
    }
    spec.body = spec
        .body
        .replace(
            "chop::position_live::dense_scores",
            "chop::position_simt_tile::scores",
        )
        .replace(">(q,", &format!(",{control}>(q,"))
        .replace("u));", "u),launch_rows);");
    spec.params.push(EntryParam::U32 {
        name: "launch_rows".into(),
    });
    Ok(spec)
}
pub fn adapt(call: &mut Launch, rows: u32) {
    call.symbol = symbol(&call.symbol);
    call.plan.grid = [position_live::SCORE_CTAS, rows.div_ceil(ROWS), 1];
    call.arguments.0.push(Argument::U32(rows));
}
pub fn cubin(name: &str, entries: Vec<EntrySpec>) -> CubinSpec {
    CubinSpec {
        name: name.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: vec!["#include \"pooled_indexer/position_simt_tile.cuh\"".into()],
        defines: vec![],
        entries,
    }
}
