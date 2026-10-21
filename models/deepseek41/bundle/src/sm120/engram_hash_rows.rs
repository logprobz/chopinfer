//! The Engram hash with one CTA per row and one thread per (layer, column).
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use std::path::Path;

pub const FAMILY: &str = "engram_hash_rows";
pub const CUBIN: &str = "deepseek41_engram_hash_rows_sm120";
pub const SYMBOL: &str = "chop_deepseek41_engram_hash_rows_u64_v1";
/// The most (layer, column) tasks of one row; above it the serial entry runs.
pub const MAX_TASKS: u32 = 1024;

/// Whole warps holding every (layer, column) task of a row.
pub fn threads(layers: u32, columns: u32) -> Option<u32> {
    let tasks = layers.checked_mul(columns)?;
    (tasks > 0 && tasks <= MAX_TASKS).then(|| tasks.div_ceil(32) * 32)
}

pub fn cubin_spec(bundle: &Path) -> CubinSpec {
    let mut params: Vec<_> = [
        ("tokens", "const uint32_t"),
        ("batch", "const int32_t"),
        ("starts", "const int32_t"),
        ("history", "const int64_t"),
        ("text", "const uint8_t"),
        ("constants", "const uint64_t"),
        ("ids", "uint32_t"),
        ("status", "uint32_t"),
    ]
    .into_iter()
    .map(|(name, cxx_type)| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    })
    .collect();
    params.extend(
        [
            "rows", "vocab", "ngram", "heads", "layers", "items", "stride",
        ]
        .into_iter()
        .map(|name| EntryParam::U32 { name: name.into() }),
    );
    CubinSpec {
        name: CUBIN.into(),
        sources_root: bundle.into(),
        include_dirs: vec!["cuda".into()],
        headers: vec!["#include \"deepseek41_engram_hash_rows.cuh\"".into()],
        defines: vec![],
        entries: vec![EntrySpec::new(
            SYMBOL.into(),
            params,
            "chop::deepseek41::engram_hash_rows(tokens,batch,starts,history,text,constants,ids,status,rows,vocab,ngram,heads,layers,items,stride);".into(),
            MAX_TASKS,
            1,
        )],
    }
}
