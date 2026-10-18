//! BF16 comparison storage with the native Position v3 accumulation order.
use super::{position, position_blocks};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::Refusal;

pub fn dense_entry(entry: &position::Entry) -> position::Entry {
    position::Entry {
        name: format!("{}_bf16", entry.name),
        query: entry.query,
    }
}
pub fn compact_entry(entry: &position_blocks::Entry) -> position_blocks::Entry {
    position_blocks::Entry {
        name: format!("{}_bf16", entry.name),
        query: entry.query,
    }
}
pub fn dense_specs(entry: &position::Entry) -> Result<Vec<EntrySpec>, Refusal> {
    dense_control_specs(entry, false)
}
pub fn compact_specs(entry: &position_blocks::Entry) -> Result<Vec<EntrySpec>, Refusal> {
    compact_control_specs(entry, false)
}
/// Lab-only extra rounding in the accumulator, with a distinct score symbol.
pub fn dense_control_specs(
    entry: &position::Entry,
    round: bool,
) -> Result<Vec<EntrySpec>, Refusal> {
    let mut specs = dense_entry(entry).specs()?;
    let q = entry.query;
    let masked = q.candidates == kernel_core::pooled_indexer::IndexerCandidates::Mask;
    let scale = kernel_core::position_indexer_oracle::weight_scale(q.selection())?.to_bits();
    specs[0].body = format!(
        "chop::position_bf16::dense_scores<{},{},{},{},{masked},{round}>(q,weights,codes,scales,tables,batches,positions,live_rows,mask,scores,width,max_pages,code_stride,scale_stride,__uint_as_float({scale}u));",
        q.heads, q.head_dim, q.page_size, q.compression_ratio
    );
    if round {
        specs.truncate(1);
        specs[0].symbol = rounded_symbol(&specs[0].symbol);
    }
    Ok(specs)
}
/// The same lab-only rounding control for compact score slots.
pub fn compact_control_specs(
    entry: &position_blocks::Entry,
    round: bool,
) -> Result<Vec<EntrySpec>, Refusal> {
    let mut specs = compact_entry(entry).specs()?;
    let q = entry.query.indexer;
    let scale = kernel_core::position_indexer_oracle::weight_scale(q.selection())?.to_bits();
    specs[0].body = format!(
        "chop::position_bf16::compact_scores<{},{},{},{},{},{},{round}>(q,weights,codes,scales,tables,batches,positions,live_rows,ids,tags,stored_tags,stored_visible,source_status,scores,width,max_pages,code_stride,scale_stride,__uint_as_float({scale}u));",
        q.heads,
        q.head_dim,
        q.page_size,
        q.compression_ratio,
        entry.query.block_size,
        entry.query.top_k_blocks
    );
    if round {
        specs.truncate(1);
        specs[0].symbol = rounded_symbol(&specs[0].symbol);
    }
    Ok(specs)
}
pub fn rounded_symbol(symbol: &str) -> String {
    format!(
        "{}_round_chain_v3",
        symbol.strip_suffix("_v3").expect("versioned score symbol")
    )
}
pub fn cubin_spec(name: impl Into<String>, entries: Vec<EntrySpec>) -> CubinSpec {
    let mut spec = position::cubin_spec(name, entries);
    spec.headers = vec![r#"#include "pooled_indexer/position_bf16.cuh""#.into()];
    spec
}
