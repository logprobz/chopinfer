//! Ordered F32 confidence chains, with one cooperative block per item.
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use std::path::Path;

pub const FAMILY: &str = "dspark_confidence_parallel";
pub const CUBIN: &str = "deepseek41_dspark_confidence_parallel_sm120";
pub const SYMBOL: &str = "chop_deepseek41_dspark_confidence_parallel_v1";
pub const THREADS: u32 = 128;

pub fn cubin_spec(bundle: &Path) -> CubinSpec {
    let mut params: Vec<_> = [
        ("hidden", "const float"),
        ("markov", "const uint16_t"),
        ("weight", "const uint16_t"),
        ("scores", "float"),
        ("live_rows", "const uint32_t"),
    ]
    .into_iter()
    .map(|(name, cxx_type)| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    })
    .collect();
    params.extend(
        ["width", "rank", "hidden_stride", "score_stride"]
            .into_iter()
            .map(|name| EntryParam::U32 { name: name.into() }),
    );
    CubinSpec {
        name: CUBIN.into(),
        sources_root: bundle.into(),
        include_dirs: vec!["cuda".into()],
        headers: vec!["#include \"deepseek41_confidence_parallel.cuh\"".into()],
        defines: vec![],
        entries: vec![EntrySpec::new(
            SYMBOL.into(),
            params,
            "chop::deepseek41::dspark::confidence_parallel(hidden,markov,weight,scores,live_rows,width,rank,hidden_stride,score_stride);".into(),
            THREADS,
            1,
        )],
    }
}
