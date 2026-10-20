//! Exact wide compact scores and bounded CTA counts for eager prefill.
use super::{position_blocks, position_mma_guard, position_mma_tile};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::{backend::Launch, contract::Refusal};

pub const CUBIN: &str = "position_mma_prefill_tile_sm120";

pub fn compact_symbol(source: &str) -> Result<String, Refusal> {
    let stem = source
        .strip_suffix("_mma_fp4_guard_fast_v1")
        .or_else(|| source.strip_suffix("_mma_fp4_guard_v1"))
        .ok_or(Refusal("prefill tile requires guarded native scores"))?;
    Ok(format!("{stem}_mma_fp4_prefill_tile_v1"))
}

pub fn compact_spec(entry: &position_blocks::Entry) -> Result<EntrySpec, Refusal> {
    let mut spec = position_mma_guard::compact_specs(entry)?.remove(1);
    spec.symbol = compact_symbol(&spec.symbol)?;
    // Preserve the SIMT pack-failure dispatch. Both paths use grid-stride loops.
    spec.body = spec.body.replace(
        "chop::position_mma_guard::compact_scores<",
        "chop::position_mma_prefill_tile::compact_scores<",
    );
    Ok(spec)
}

pub fn score_symbol(source: &str, compact: bool) -> Result<String, Refusal> {
    if compact {
        compact_symbol(source)
    } else {
        let source = source.replace("_mma_fp4_guard_v1", "_mma_fp4_guard_fast_v1");
        position_mma_tile::VARIANTS[1].symbol(&source)
    }
}

pub fn diagnostic(mut spec: EntrySpec) -> EntrySpec {
    spec.symbol.push_str("_counts_v1");
    spec.params.push(kernel_build::EntryParam::DevicePointer {
        name: "guard_counts".into(),
        cxx_type: "uint32_t".into(),
    });
    spec.body = spec.body.replace(
        "reinterpret_cast<const uint32_t*>(scratch+qstatus_offset));",
        "reinterpret_cast<const uint32_t*>(scratch+qstatus_offset),guard_counts);",
    );
    spec
}

pub fn cubin_spec(entries: Vec<EntrySpec>) -> CubinSpec {
    let mut spec = position_mma_guard::cubin_spec(CUBIN, entries);
    spec.headers = vec!["#include \"pooled_indexer/position_mma_prefill_tile.cuh\"".into()];
    spec
}

/// Bound setup work by the row batch. Visible counts remain device inputs.
/// The small-row schedule is unchanged; callers must select eager prefill.
pub fn remap(call: &mut Launch, compact: bool, ctas: u32) -> Result<(), Refusal> {
    if call.plan.block != [128, 1, 1]
        || call.plan.grid != [192, call.rows, 1]
        || !matches!(ctas, 4 | 8 | 16 | 32 | 192)
    {
        return Err(Refusal(
            "prefill tile requires the guarded four-warp row grid",
        ));
    }
    call.symbol = score_symbol(&call.symbol, compact)?;
    call.plan.grid[0] = ctas;
    Ok(())
}
