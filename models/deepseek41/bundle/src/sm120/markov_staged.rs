//! The Markov step of two or more rows over register-staged weight chains,
//! with base rows read at their own stride.
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use std::path::Path;

pub const FAMILY: &str = "dspark_markov_staged";
pub const CUBIN: &str = "deepseek41_dspark_markov_staged_sm120";
pub const SYMBOL: &str = "chop_deepseek41_dspark_markov_staged_v1";
/// One thread per (logit, 128-product chain) of a CTA's 64 logits.
pub const THREADS: u32 = 128;
pub const COLUMNS: u32 = 64;
/// The one rank the entry's chains cover.
pub const RANK: u32 = 256;

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
        ["vocab", "base_stride", "output_stride"]
            .into_iter()
            .map(|name| EntryParam::U32 { name: name.into() }),
    );
    CubinSpec {
        name: CUBIN.into(),
        sources_root: bundle.into(),
        include_dirs: vec!["cuda".into()],
        headers: vec!["#include \"deepseek41_markov_staged.cuh\"".into()],
        defines: vec![],
        entries: vec![EntrySpec::new(
            SYMBOL.into(),
            params,
            "chop::deepseek41::dspark::markov_staged(embedding,weight,base,output,live_rows,vocab,base_stride,output_stride);".into(),
            THREADS,
            1,
        )],
    }
}
