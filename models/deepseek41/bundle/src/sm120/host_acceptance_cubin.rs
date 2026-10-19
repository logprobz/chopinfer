use super::host_acceptance_entries::{CUBIN, SYMBOL, THREADS};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use std::path::Path;

pub fn cubin_spec(bundle: &Path) -> CubinSpec {
    let mut params: Vec<_> = [
        ("items", "const uint32_t"),
        ("proposals", "const uint32_t"),
        ("samples", "const uint32_t"),
        ("stops", "const uint32_t"),
        ("results", "uint32_t"),
    ]
    .into_iter()
    .map(|(name, cxx_type)| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    })
    .collect();
    params.extend(
        [
            "count",
            "vocab",
            "sample_capacity",
            "draft_capacity",
            "stop_capacity",
        ]
        .into_iter()
        .map(|name| EntryParam::U32 { name: name.into() }),
    );
    CubinSpec { name:CUBIN.into(),sources_root:bundle.into(),include_dirs:vec!["cuda".into()],
        headers:vec!["#include \"deepseek41_host_acceptance.cuh\"".into()],defines:vec![],
        entries:vec![EntrySpec::new(SYMBOL.into(),params,
            "chop::deepseek41::host_acceptance(items,proposals,samples,stops,results,count,vocab,sample_capacity,draft_capacity,stop_capacity);".into(),THREADS,1)] }
}
