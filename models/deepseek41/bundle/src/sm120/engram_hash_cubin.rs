use super::engram_hash_entries::{CUBIN, SYMBOL, THREADS};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use std::path::Path;

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
    CubinSpec { name: CUBIN.into(), sources_root: bundle.into(), include_dirs: vec!["cuda".into()],
        headers: vec!["#include \"deepseek41_engram_hash.cuh\"".into()], defines: vec![],
        entries: vec![EntrySpec::new(SYMBOL.into(), params,
            "chop::deepseek41::engram_hash(tokens,batch,starts,history,text,constants,ids,status,rows,vocab,ngram,heads,layers,items,stride);".into(), THREADS, 1)] }
}
