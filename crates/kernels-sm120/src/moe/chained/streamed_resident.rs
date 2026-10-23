//! Keep the small tile's arithmetic while targeting eight resident CTAs.
//!
//! The gate also has a build reading its bank in 64-row by 128-K tiles
//! (`row-tiles-64x128-groups`); each lane loads the same bytes, so its
//! chains and partials are the row-major gate's.
use super::{quantized::Projection, streamed};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal};

pub fn symbol(p: &Projection) -> String {
    format!("{}_small_resident8_v1", p.symbol.trim_end_matches("_v1"))
}
pub fn plan(p: &Projection, rows: u32) -> Result<LaunchPlan, Refusal> {
    if !p.gate_up() || p.row_tile != 64 || p.tokens_capacity != 64 || rows > 32 {
        return Err(Refusal(
            "resident gate requires the 64-row ABI and at most 32 live rows",
        ));
    }
    streamed::plan(p, streamed::Stage::Small, rows)
}
pub fn entry(p: &Projection) -> EntrySpec {
    assert!(plan(p, 32).is_ok());
    let mut e = streamed::entries(p)
        .into_iter()
        .find(|e| e.symbol == streamed::symbol(p, streamed::Stage::Small))
        .unwrap();
    e.symbol = symbol(p);
    e.min_blocks = 8;
    // The builder rejects spills. Only register allocation and instruction
    // scheduling may change; operands, chains and partial addresses do not.
    e
}
pub fn cubin_spec(name: &str, projections: &[Projection]) -> CubinSpec {
    super::spec(
        name.into(),
        "chained_streamed.cuh",
        projections.iter().map(entry).collect(),
    )
}

/// The resident gate reading its bank in row tiles.
pub fn tiled_symbol(p: &Projection) -> String {
    format!("{}_tiles_v1", symbol(p).trim_end_matches("_v1"))
}

/// The resident gate's ABI and launch with its bank in row tiles: the same
/// operands, read at the tiled addresses.
pub fn tiled_entry(p: &Projection) -> Result<EntrySpec, Refusal> {
    if p.query.projection.gemm.k % 128 != 0 || p.columns() % 64 != 0 {
        return Err(Refusal("row tiles need whole 64-row by 128-K tiles"));
    }
    let mut e = entry(p);
    assert_eq!(e.body.matches(",8>(input,").count(), 1);
    e.symbol = tiled_symbol(p);
    e.body = e.body.replace(",8>(input,", ",8,true>(input,");
    Ok(e)
}

/// [`cubin_spec`] and, after its entries, the tiled resident gate of each
/// projection in `tiled`.
pub fn tiled_cubin_spec(
    name: &str,
    projections: &[Projection],
    tiled: &[Projection],
) -> Result<CubinSpec, Refusal> {
    let mut spec = cubin_spec(name, projections);
    for p in tiled {
        spec.entries.push(tiled_entry(p)?);
    }
    Ok(spec)
}
