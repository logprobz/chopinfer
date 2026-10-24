//! Unqualified down queues reading tile-order encoded scale banks.
//! Codes, route metadata, MMA order and tail handling retain the raw ABI.
use super::{
    absorb, bandwidth,
    floor::{Entry, Stage},
    quantized::Projection,
    streamed, strip2,
};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::Refusal;

fn encoded(
    mut entry: EntrySpec,
    suffix: &str,
    replacement: &str,
    width: u32,
) -> Result<EntrySpec, Refusal> {
    if !matches!(width, 1 | 2 | 4) {
        return Err(Refusal("tile scale offset width must be 1, 2 or 4"));
    }
    assert_eq!(entry.body.matches(suffix).count(), 1);
    entry.symbol = format!(
        "{}_scale_tiles_w{width}_v1",
        entry.symbol.trim_end_matches("_v1")
    );
    entry.body = entry.body.replace(suffix, replacement);
    Ok(entry)
}

pub fn direct_entry(e: &Entry, stage: Stage, width: u32) -> Result<EntrySpec, Refusal> {
    let rows = match stage {
        Stage::DownSmall => 8,
        Stage::DownWide => 64,
        _ => return Err(Refusal("encoded down scales require a down queue")),
    };
    encoded(
        bandwidth::entry(e, stage, bandwidth::Loads::Direct),
        &format!(",{rows}>(input,"),
        &format!(",{rows},false,chop::moe::encoded_down::Scales<{width}>>(input,"),
        width,
    )
}

pub fn absorb_entry(e: &Entry, width: u32) -> Result<EntrySpec, Refusal> {
    encoded(
        absorb::entry(e, Stage::DownSmall),
        ">(input,",
        &format!(",false,chop::moe::encoded_down::Scales<{width}>>(input,"),
        width,
    )
}

pub fn strip2_entry(e: &Entry, width: u32) -> Result<EntrySpec, Refusal> {
    encoded(
        strip2::entry(e),
        ">(input,",
        &format!(",chop::moe::encoded_down::Scales<{width}>>(input,"),
        width,
    )
}

fn streamed_raw(p: &Projection, stage: streamed::Stage) -> Result<EntrySpec, Refusal> {
    let symbol = streamed::symbol(p, stage);
    streamed::entries(p)
        .into_iter()
        .find(|e| e.symbol == symbol)
        .ok_or(Refusal("streamed projection has no such compute queue"))
}

pub fn streamed_entry(
    p: &Projection,
    stage: streamed::Stage,
    width: u32,
) -> Result<EntrySpec, Refusal> {
    if p.gate_up() {
        return Err(Refusal("encoded down scales require a down projection"));
    }
    let rows = match stage {
        streamed::Stage::Small => 8,
        streamed::Stage::Wide => 64,
        _ => return Err(Refusal("encoded down scales require a compute queue")),
    };
    encoded(
        streamed_raw(p, stage)?,
        &format!(",{rows}>(input,"),
        &format!(",{rows},false,chop::moe::encoded_down::Scales<{width}>>(input,"),
        width,
    )
}

pub fn cubin_spec(name: &str, e: &Entry, projections: &[Projection]) -> Result<CubinSpec, Refusal> {
    let mut entries = vec![
        bandwidth::entry(e, Stage::DownSmall, bandwidth::Loads::Direct),
        bandwidth::entry(e, Stage::DownWide, bandwidth::Loads::Direct),
        absorb::entry(e, Stage::DownSmall),
        strip2::entry(e),
    ];
    for width in [1, 2, 4] {
        entries.extend([
            direct_entry(e, Stage::DownSmall, width)?,
            direct_entry(e, Stage::DownWide, width)?,
            absorb_entry(e, width)?,
            strip2_entry(e, width)?,
        ]);
    }
    for p in projections.iter().filter(|p| !p.gate_up()) {
        for stage in [streamed::Stage::Small, streamed::Stage::Wide] {
            if stage == streamed::Stage::Wide && p.tokens_capacity <= 8 {
                continue;
            }
            entries.push(streamed_raw(p, stage)?);
            for width in [1, 2, 4] {
                entries.push(streamed_entry(p, stage, width)?);
            }
        }
    }
    Ok(super::spec(
        name.into(),
        "chained_down_scale_tiles.cuh",
        entries,
    ))
}
