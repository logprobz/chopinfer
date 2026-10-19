//! Isolated compact native FP8 candidates. No serving catalog selects them.
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;
use kernels_sm120::gemm::mxfp8_prefill_compact::Entry;
use model_deepseek41_contract::geometry::FLASH;
pub const FAMILY: &str = "compact_dense_prefill";
pub const CUBIN: &str = "compact_dense_prefill_sm120";
pub const ORACLE: &str = "compact_dense_oracle_sm120";
pub fn entries() -> Vec<Entry> {
    bundle_deepseek41::sm120::mxfp8_fold::prefill_catalog(&FLASH)
        .unwrap()
        .into_iter()
        .flat_map(|base| [2, 3].map(|stages| Entry::new(base.clone(), stages).unwrap()))
        .collect()
}
pub fn oracle_symbol(q: Mxfp8FoldGemmQuery) -> String {
    format!(
        "chop_compact_dense_oracle_n{}_k{}_g{}_v1",
        q.n, q.k, q.groups
    )
}
pub fn cubins() -> Vec<CubinSpec> {
    let entries = entries();
    let compact = kernels_sm120::gemm::mxfp8_prefill_compact::cubin_spec(CUBIN, &entries).unwrap();
    let mut oracle = compact.clone();
    oracle.name = ORACLE.into();
    oracle.sources_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    oracle
        .include_dirs
        .push("models/deepseek41/lab/cuda".into());
    oracle.headers = vec!["#include \"compact_dense_oracle.cuh\"".into()];
    oracle.entries=entries.iter().filter(|e|e.stages==2).map(|e|{
        let q=e.base.query;let ptr=|name:&str,ty:&str|EntryParam::DevicePointer{name:name.into(),cxx_type:ty.into()};
        EntrySpec::new(oracle_symbol(q),vec![ptr("input","const uint8_t"),ptr("as","const uint8_t"),ptr("weight","const uint8_t"),ptr("ws","const uint8_t"),ptr("rows","const uint32_t"),ptr("output","double"),ptr("absolute","double"),EntryParam::U32{name:"count".into()}],format!("chop::compact_dense_oracle::run<{},{},{},{},{}>(input,as,weight,ws,rows,output,absolute,count);",q.n,q.k,q.groups,q.activation_row_stride,q.activation_group_stride),128,1)
    }).collect();
    vec![compact, oracle]
}
