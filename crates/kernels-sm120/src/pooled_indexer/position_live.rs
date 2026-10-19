//! Fixed-grid position scoring and parallel, ordered exact selection.
use super::{position, position_bf16, position_blocks, position_candidates_runtime};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::Refusal;

pub const SCORE_CTAS: u32 = 192;
pub const SHARDS: u32 = 32;

pub fn symbol(original: &str) -> String {
    format!("{original}_live_v1")
}
pub fn merge_symbol(original: &str) -> String {
    format!("{original}_live_merge_v1")
}
pub fn workspace_bytes(rows: u32, kept: u32) -> u64 {
    u64::from(rows) * u64::from(SHARDS) * (u64::from(kept) + 1) * 8
}

fn parallel(mut specs: Vec<EntrySpec>) -> Vec<EntrySpec> {
    specs[0].symbol = symbol(&specs[0].symbol);
    let mut merge = specs[1].clone();
    merge.symbol = merge_symbol(&merge.symbol);
    merge.body = merge.body.replace(",false>(", ",true>(");
    specs[1].symbol = symbol(&specs[1].symbol);
    specs.push(merge);
    for spec in &mut specs[1..] {
        spec.params.push(EntryParam::DevicePointer {
            name: "scratch".into(),
            cxx_type: "uint64_t".into(),
        });
        spec.params.push(EntryParam::U64 {
            name: "scratch_offset".into(),
        });
        spec.body = spec.body.replace(");", ",scratch + scratch_offset / 8);");
    }
    specs
}

pub fn dense_specs(entry: &position::Entry, bf16: bool) -> Result<Vec<EntrySpec>, Refusal> {
    let mut specs = if bf16 {
        position_bf16::dense_specs(entry)?
    } else {
        entry.specs()?
    };
    specs[0].body = specs[0]
        .body
        .replace(
            "chop::position_indexer::scores",
            "chop::position_live::dense_scores",
        )
        .replace(
            "chop::position_bf16::dense_scores",
            "chop::position_live::dense_scores",
        )
        .replace(",0>(", ",false>(");
    if bf16 {
        specs[0].body = specs[0].body.replace(",false>(", ",true>(");
    }
    specs[1].body = specs[1]
        .body
        .replace(
            "chop::position_indexer::select",
            "chop::position_live::dense_select",
        )
        .replace(",0>(", ",false>(");
    Ok(parallel(specs))
}

pub fn compact_specs(
    entry: &position_blocks::Entry,
    bf16: bool,
) -> Result<Vec<EntrySpec>, Refusal> {
    let mut specs = if bf16 {
        position_bf16::compact_specs(entry)?
    } else {
        entry.specs()?
    };
    specs[0].body = specs[0]
        .body
        .replace(
            "chop::position_blocks::scores",
            "chop::position_live::compact_scores",
        )
        .replace(
            "chop::position_bf16::compact_scores",
            "chop::position_live::compact_scores",
        )
        .replace(",0>(", ",false>(");
    if bf16 {
        specs[0].body = specs[0].body.replace(",false>(", ",true>(");
    }
    specs[1].body = specs[1]
        .body
        .replace(
            "chop::position_blocks::select",
            "chop::position_live::compact_select",
        )
        .replace(",0>(", ",false>(");
    Ok(parallel(specs))
}

pub fn candidate_specs(
    entry: &position_blocks::CandidateEntry,
    runtime: bool,
) -> Result<Vec<EntrySpec>, Refusal> {
    let mut specs = if runtime {
        position_candidates_runtime::specs(entry)?
    } else {
        entry.specs()?
    };
    specs[0].body = specs[0]
        .body
        .replace("chop::position_blocks::", "chop::position_live::");
    specs[1].body = format!(
        "chop::position_live::gather<{},{},false>(blocks,live_rows,tags,visible,source_status,ids,out_tags,out_visible,status,width,{});",
        entry.query.block_size,
        entry.query.top_k_blocks,
        if runtime {
            "top_k_blocks".into()
        } else {
            entry.query.top_k_blocks.to_string()
        }
    );
    Ok(parallel(specs))
}

pub fn cubin_spec(name: impl Into<String>, entries: Vec<EntrySpec>) -> CubinSpec {
    let mut spec = position::cubin_spec(name, entries);
    spec.headers = vec!["#include \"pooled_indexer/position_live.cuh\"".into()];
    spec
}
