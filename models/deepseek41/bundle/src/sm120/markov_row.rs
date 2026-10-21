//! The one-row Markov step over coalesced, early-staged weight tiles.
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use std::path::Path;

pub const FAMILY: &str = "dspark_markov_row";
pub const CUBIN: &str = "deepseek41_dspark_markov_row_sm120";
pub const SYMBOL: &str = "chop_deepseek41_dspark_markov_row_v1";
/// One thread per (logit, 128-product chain) of a CTA's 64 logits.
pub const THREADS: u32 = 256;
pub const COLUMNS: u32 = 64;
/// The most chains a CTA's threads hold per logit.
pub const MAX_RANK: u32 = THREADS / COLUMNS * 128;

/// `markov_row_shared_bytes` of the header.
pub fn shared_bytes(rank: u32) -> u32 {
    (COLUMNS * (rank / 2 + 1) + rank + COLUMNS * (THREADS / COLUMNS)) * 4
}

pub fn cubin_spec(bundle: &Path) -> CubinSpec {
    let mut params: Vec<_> = [
        ("embedding", "const uint16_t"),
        ("weight", "const uint16_t"),
        ("base", "const float"),
        ("output", "float"),
        ("live_rows", "const uint32_t"),
    ]
    .into_iter()
    .map(|(name, cxx_type)| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    })
    .collect();
    params.extend(
        ["rank", "vocab", "stride"]
            .into_iter()
            .map(|name| EntryParam::U32 { name: name.into() }),
    );
    CubinSpec {
        name: CUBIN.into(),
        sources_root: bundle.into(),
        include_dirs: vec!["cuda".into()],
        headers: vec!["#include \"deepseek41_markov_row.cuh\"".into()],
        defines: vec![],
        entries: vec![EntrySpec::new(
            SYMBOL.into(),
            params,
            "chop::deepseek41::dspark::markov_row(embedding,weight,base,output,live_rows,rank,vocab,stride);".into(),
            THREADS,
            1,
        )],
    }
}
