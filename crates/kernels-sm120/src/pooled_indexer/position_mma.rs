//! Native FP4 score entries retain the existing dense and compact selectors.
use super::{position, position_blocks, position_live, position_valid};
use kernel_build::{CubinSpec, EntryParam as P, EntrySpec};
use kernel_core::{
    contract::Refusal,
    pooled_indexer::{IndexerCandidates, PositionIndexerQuery},
};

pub fn supported(query: &PositionIndexerQuery) -> bool {
    query.heads == 32 && query.head_dim == 128 && query.candidates == IndexerCandidates::All
}
pub fn score_symbol(original: &str) -> String {
    format!("{original}_mma_fp4_v1")
}
pub fn pack_symbol(original: &str) -> String {
    format!("{original}_pack_fp4_v1")
}

/// Packed queries follow selection scratch in the same rank's main workspace.
#[derive(Clone, Copy, Debug)]
pub struct Workspace {
    pub codes: u64,
    pub scales: u64,
    pub status: u64,
    pub bytes: u64,
}
impl Workspace {
    pub fn new(selection_bytes: u64, rows: u32) -> Result<Self, Refusal> {
        let add = |a: u64, b: u64| {
            a.checked_add(b)
                .ok_or(Refusal("native indexer workspace overflows"))
        };
        let codes = add(selection_bytes, 255)? & !255;
        let scales = add(codes, u64::from(rows) * 2048)?;
        let status = add(scales, u64::from(rows) * 128)?;
        let bytes = add(status, u64::from(rows) * 4)?;
        Ok(Self {
            codes,
            scales,
            status,
            bytes,
        })
    }
}
fn ptr(name: &str, ty: &str) -> P {
    P::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    }
}
fn offsets() -> Vec<P> {
    vec![
        ptr("scratch", "uint8_t"),
        P::U64 {
            name: "qcodes_offset".into(),
        },
        P::U64 {
            name: "qscales_offset".into(),
        },
        P::U64 {
            name: "qstatus_offset".into(),
        },
    ]
}
fn adapt(mut score: EntrySpec, compact: bool) -> Vec<EntrySpec> {
    let original = score.symbol.clone();
    let mut pack_params = vec![
        ptr("q", "const uint16_t"),
        ptr("live_rows", "const uint32_t"),
    ];
    pack_params.extend(offsets());
    let pack = EntrySpec::new(pack_symbol(&original),pack_params,
        "chop::ptx::griddep_launch_dependents(); chop::ptx::griddep_wait(); chop::position_mma_native::pack(q,live_rows,scratch+qcodes_offset,scratch+qscales_offset,reinterpret_cast<uint32_t*>(scratch+qstatus_offset));".into(),256,1);
    score.symbol = score_symbol(&original);
    score.params.extend(offsets());
    let fallback = score.body.clone();
    score.body = if compact {
        score.body.replace("chop::position_valid::compact_scores", "chop::position_mma_compact::scores").replace(",false>(", ">(")
    } else {
        score.body.replace("chop::position_live::dense_scores", "chop::position_mma_native::scores")
    }.replace(");", ",scratch+qcodes_offset,scratch+qscales_offset,reinterpret_cast<const uint32_t*>(scratch+qstatus_offset));");
    score.body = format!(
        "chop::ptx::griddep_wait(); if (blockIdx.y < *live_rows && reinterpret_cast<const uint32_t*>(scratch+qstatus_offset)[blockIdx.y] != 0) {{ {fallback} }} else {{ {} }}",
        score.body
    );
    vec![pack, score]
}
pub fn dense_specs(entry: &position::Entry) -> Result<Vec<EntrySpec>, Refusal> {
    if !supported(&entry.query) {
        return Err(Refusal("native MMA requires 32 x 128 All index heads"));
    }
    Ok(adapt(
        position_live::dense_specs(entry, false)?.remove(0),
        false,
    ))
}
pub fn compact_specs(entry: &position_blocks::Entry) -> Result<Vec<EntrySpec>, Refusal> {
    if !supported(&entry.query.indexer) || entry.query.block_size != 8 {
        return Err(Refusal(
            "native MMA requires 32 x 128 index heads and 8-position blocks",
        ));
    }
    Ok(adapt(
        position_valid::compact_specs(entry, false)?.remove(0),
        true,
    ))
}
pub fn cubin_spec(name: impl Into<String>, entries: Vec<EntrySpec>) -> CubinSpec {
    let mut spec = position_live::cubin_spec(name, entries);
    spec.headers = vec!["#include \"pooled_indexer/position_mma_compact.cuh\"".into()];
    spec
}
