//! Exact small-row rotation and MXFP4 fake quantization with one row admission.
use super::position_query::Entry;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::Refusal;

pub const MAX_ROWS: u32 = 32;
pub const THREADS: u32 = 1024;

pub fn supports(entry: &Entry) -> bool {
    (entry.heads, entry.head_dim, entry.rotary_dim) == (32, 128, 64)
}

pub fn symbol(entry: &Entry) -> String {
    format!("chop_position_query_{}_parallel_v1", entry.name)
}

pub fn spec(entry: &Entry, ignore_status: bool) -> Result<EntrySpec, Refusal> {
    if !supports(entry) {
        return Err(Refusal(
            "parallel position query requires 32 heads of 128 with 64 rotary channels",
        ));
    }
    let mut spec = entry.spec()?;
    spec.symbol = symbol(entry);
    if ignore_status {
        spec.symbol = spec
            .symbol
            .replace("_parallel_v1", "_parallel_ignore_status_v1");
    }
    spec.params[4] = EntryParam::DevicePointer {
        name: "output".into(),
        cxx_type: "uint16_t".into(),
    };
    // Rebuild to bind the larger launch bound as well as the shared ABI.
    Ok(EntrySpec::new(
        spec.symbol,
        spec.params,
        format!(
            "chop::position_query_parallel::run<32,128,64,{ignore_status}>(input,cos_sin,positions,live_rows,output,status,status_row_offset,table_positions);"
        ),
        THREADS,
        1,
    ))
}

pub fn cubin(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    Ok(CubinSpec {
        name: name.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: vec!["#include \"pooled_indexer/position_query_parallel.cuh\"".into()],
        defines: vec![],
        entries: entries
            .iter()
            .map(|e| spec(e, false))
            .collect::<Result<_, _>>()?,
    })
}
