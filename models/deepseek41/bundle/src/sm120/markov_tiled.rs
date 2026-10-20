//! The Markov step with weight tiles shared by every drafting row.
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use std::path::Path;

pub const FAMILY: &str = "dspark_markov_tiled";
pub const CUBIN: &str = "deepseek41_dspark_markov_tiled_sm120";
pub const SYMBOL: &str = "chop_deepseek41_dspark_markov_tiled_v1";
/// One thread per vocabulary logit of a CTA's columns.
pub const THREADS: u32 = 128;

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
        headers: vec!["#include \"deepseek41_markov_tiled.cuh\"".into()],
        defines: vec![],
        entries: vec![EntrySpec::new(
            SYMBOL.into(),
            params,
            "chop::deepseek41::dspark::markov_tiled(embedding,weight,base,output,live_rows,rank,vocab,stride);".into(),
            THREADS,
            1,
        )],
    }
}
