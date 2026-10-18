//! DSpark's F32 head cubin, locked separately from the other own ops.

use super::dspark_entries::{CONFIDENCE, CUBIN, MARKOV, THREADS};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use std::path::Path;

fn ptr(name: &str, cxx_type: &str) -> EntryParam {
    EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    }
}

fn count(name: &str) -> EntryParam {
    EntryParam::U32 { name: name.into() }
}

pub fn cubin_spec(bundle: &Path) -> CubinSpec {
    CubinSpec {
        name: CUBIN.into(),
        sources_root: bundle.into(),
        include_dirs: vec!["cuda".into()],
        headers: vec!["#include \"deepseek41_dspark.cuh\"".into()],
        defines: vec![],
        entries: vec![
            EntrySpec::new(
                MARKOV.into(),
                vec![
                    ptr("embedding", "const uint16_t"),
                    ptr("weight", "const uint16_t"),
                    ptr("base", "const float"),
                    ptr("output", "float"),
                    ptr("live_rows", "const uint32_t"),
                    count("rank"),
                    count("vocab"),
                    count("stride"),
                ],
                "chop::deepseek41::dspark::markov(embedding,weight,base,output,live_rows,rank,vocab,stride);".into(),
                THREADS,
                1,
            ),
            EntrySpec::new(
                CONFIDENCE.into(),
                vec![
                    ptr("hidden", "const float"),
                    ptr("markov", "const uint16_t"),
                    ptr("weight", "const uint16_t"),
                    ptr("scores", "float"),
                    ptr("live_rows", "const uint32_t"),
                    count("width"),
                    count("rank"),
                    count("hidden_stride"),
                    count("score_stride"),
                ],
                "chop::deepseek41::dspark::confidence(hidden,markov,weight,scores,live_rows,width,rank,hidden_stride,score_stride);".into(),
                THREADS,
                1,
            ),
        ],
    }
}
