//! The direct gate small and wide queues, the resident8 gate, the absorbing
//! gate small queue and the streamed module's gate small and wide queues
//! reading their bank in quad-lane row tiles ([`ROW_TILES_64X128_LANES`],
//! `chained_lanes.cuh`). Each lane loads the bytes it converts and one word
//! of its row's four group scales, so no shuffle or byte shift forms the
//! MMA operands. The A operands, MMAs, chains and partials are the
//! row-tiled queues', so the partials are theirs bit for bit; each build
//! keeps its row-tiled twin's parameters, launch and residency.
//!
//! [`ROW_TILES_64X128_LANES`]: kernel_core::precision::layout::ROW_TILES_64X128_LANES
use super::{
    absorb, bandwidth,
    floor::{Entry, Stage},
    quantized::Projection,
    streamed, streamed_resident,
};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::Refusal;

/// The header of every quad-lane gate queue.
pub const HEADER: &str = "chained_lanes.cuh";

/// The direct gate small or wide queue reading its bank in quad-lane row
/// tiles.
pub fn direct_symbol(e: &Entry, s: Stage) -> String {
    bandwidth::tiled_symbol(e, s).replace("_bandwidth_tiles_", "_bandwidth_lanes_")
}

/// The row-tiled direct gate queue's ABI and launch over the quad-lane
/// bank.
pub fn direct_entry(e: &Entry, s: Stage) -> Result<EntrySpec, Refusal> {
    let rows = match s {
        Stage::GateSmall => 8,
        Stage::GateWide => 64,
        _ => return Err(Refusal("only the gate queues read a quad-lane bank")),
    };
    let mut spec = bandwidth::tiled_entry(e, s)?;
    let call = format!(",{rows},true>(input,");
    assert_eq!(
        spec.body.matches("chop::moe::streamed::compute<").count(),
        1
    );
    assert_eq!(spec.body.matches(&call).count(), 1);
    spec.symbol = direct_symbol(e, s);
    spec.body = spec
        .body
        .replace(
            "chop::moe::streamed::compute<",
            "chop::moe::lanes::compute<",
        )
        .replace(&call, &format!(",{rows}>(input,"));
    Ok(spec)
}

/// A direct small gate queue reading a validated tile-order encoded scale
/// bank. The code bank and every launch argument keep the raw entry's ABI;
/// the scale pointer names the encoded bank table instead of raw tiles.
/// This entry is unqualified and is not selected by a serving catalog.
pub fn direct_scale_tiles_entry(e: &Entry, width: u32) -> Result<EntrySpec, Refusal> {
    if !matches!(width, 1 | 2 | 4) {
        return Err(Refusal("tile scale offset width must be 1, 2 or 4"));
    }
    let mut spec = direct_entry(e, Stage::GateSmall)?;
    assert_eq!(spec.body.matches(",8>(input,").count(), 1);
    spec.symbol = format!(
        "{}_scale_tiles_w{width}_v1",
        spec.symbol.trim_end_matches("_v1")
    );
    spec.body = spec.body.replace(
        ",8>(input,",
        &format!(",8,chop::moe::lanes::EncodedScales<{width}>>(input,"),
    );
    Ok(spec)
}

/// The raw direct-small entry beside its three encoded-scale lab variants.
pub fn scale_tiles_cubin_spec(name: &str, e: &Entry) -> Result<CubinSpec, Refusal> {
    let mut entries = vec![direct_entry(e, Stage::GateSmall)?];
    for width in [1, 2, 4] {
        entries.push(direct_scale_tiles_entry(e, width)?);
    }
    Ok(super::spec(
        name.into(),
        "chained_lanes_scale_tiles.cuh",
        entries,
    ))
}

/// The resident8 gate reading its bank in quad-lane row tiles.
pub fn resident_symbol(p: &Projection) -> String {
    format!(
        "{}_lanes_v1",
        streamed_resident::symbol(p).trim_end_matches("_v1")
    )
}

/// The row-tiled resident8 gate's ABI and launch over the quad-lane bank.
pub fn resident_entry(p: &Projection) -> Result<EntrySpec, Refusal> {
    let mut spec = streamed_resident::tiled_entry(p)?;
    assert_eq!(
        spec.body.matches("chop::moe::streamed::compute<").count(),
        1
    );
    assert_eq!(spec.body.matches(",8,true>(input,").count(), 1);
    spec.symbol = resident_symbol(p);
    spec.body = spec
        .body
        .replace(
            "chop::moe::streamed::compute<",
            "chop::moe::lanes::compute<",
        )
        .replace(",8,true>(input,", ",8>(input,");
    Ok(spec)
}

/// The absorbing gate small queue reading its bank in quad-lane row tiles.
pub fn absorb_symbol(e: &Entry) -> String {
    absorb::tiled_symbol(e).replace("_absorb_tiles_", "_absorb_lanes_")
}

/// The row-tiled absorbing gate's ABI and launch over the quad-lane bank.
pub fn absorb_entry(e: &Entry) -> Result<EntrySpec, Refusal> {
    let mut spec = absorb::tiled_entry(e)?;
    assert_eq!(spec.body.matches("chop::moe::absorb::compute<").count(), 1);
    assert_eq!(spec.body.matches(",true>(input,").count(), 1);
    spec.symbol = absorb_symbol(e);
    spec.body = spec
        .body
        .replace(
            "chop::moe::absorb::compute<",
            "chop::moe::lanes::absorbing<",
        )
        .replace(",true>(input,", ">(input,");
    Ok(spec)
}

/// The streamed module's small or wide gate/up queue reading its bank in
/// quad-lane row tiles.
pub fn streamed_symbol(p: &Projection, s: streamed::Stage) -> String {
    format!(
        "{}_lanes_v1",
        streamed::symbol(p, s).trim_end_matches("_v1")
    )
}

/// The row-tiled streamed queue's ABI and launch over the quad-lane bank.
pub fn streamed_entry(p: &Projection, s: streamed::Stage) -> Result<EntrySpec, Refusal> {
    let rows = match s {
        streamed::Stage::Small => 8,
        streamed::Stage::Wide => 64,
        _ => return Err(Refusal("only the small and wide queues read weights")),
    };
    let mut spec = streamed::tiled_entry(p, s)?;
    let call = format!(",{rows},true>(input,");
    assert_eq!(
        spec.body.matches("chop::moe::streamed::compute<").count(),
        1
    );
    assert_eq!(spec.body.matches(&call).count(), 1);
    spec.symbol = streamed_symbol(p, s);
    spec.body = spec
        .body
        .replace(
            "chop::moe::streamed::compute<",
            "chop::moe::lanes::compute<",
        )
        .replace(&call, &format!(",{rows}>(input,"));
    Ok(spec)
}

/// The quad-lane small and wide queues of each streamed gate/up projection
/// in `projections`.
pub fn streamed_cubin_spec(name: &str, projections: &[Projection]) -> Result<CubinSpec, Refusal> {
    let mut specs = Vec::new();
    for p in projections {
        for s in streamed::tiled_stages(p) {
            specs.push(streamed_entry(p, s)?);
        }
    }
    Ok(super::spec(name.into(), HEADER, specs))
}

/// Every quad-lane gate queue of each bank in `entries`: direct small and
/// wide, resident8, absorbing.
pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    let mut specs = Vec::new();
    for e in entries {
        specs.push(direct_entry(e, Stage::GateSmall)?);
        specs.push(direct_entry(e, Stage::GateWide)?);
        specs.push(resident_entry(&e.gate)?);
        specs.push(absorb_entry(e)?);
    }
    Ok(super::spec(name.into(), HEADER, specs))
}
