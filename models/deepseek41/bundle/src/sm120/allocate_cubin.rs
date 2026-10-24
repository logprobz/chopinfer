use super::allocate_entries::{CUBIN, SYMBOL, THREADS};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use std::path::Path;

pub fn cubin_spec(bundle: &Path) -> CubinSpec {
    let pointer = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    let mut params = vec![
        pointer("items", "const uint32_t"),
        pointer("scores", "const float"),
        pointer("rule", "const uint32_t"),
        pointer("live_items", "const uint32_t"),
        pointer("status", "uint32_t"),
        pointer("verified", "uint32_t"),
        pointer("starts", "uint32_t"),
        pointer("rows", "uint32_t"),
        pointer("expand", "uint32_t"),
        pointer("token_ids", "uint32_t"),
        pointer("positions", "uint32_t"),
        pointer("token_batch", "uint32_t"),
        pointer("window_rows", "uint32_t"),
        pointer("index_visible", "uint32_t"),
        pointer("q_start", "uint32_t"),
        pointer("text", "uint8_t"),
    ];
    params.extend(
        [
            "item_capacity",
            "score_rows",
            "reserved_capacity",
            "width_capacity",
            "text_rows",
        ]
        .into_iter()
        .map(|name| EntryParam::U32 { name: name.into() }),
    );
    CubinSpec {
        name: CUBIN.into(),
        sources_root: bundle.into(),
        include_dirs: vec!["cuda".into()],
        headers: vec!["#include \"deepseek41_allocate.cuh\"".into()],
        defines: vec![],
        entries: vec![EntrySpec::new(
            SYMBOL.into(),
            params,
            "chop::deepseek41::verify_allocate(items,scores,rule,live_items,status,verified,\
             starts,rows,expand,token_ids,positions,token_batch,window_rows,index_visible,\
             q_start,text,item_capacity,score_rows,reserved_capacity,width_capacity,text_rows);"
                .into(),
            THREADS,
            1,
        )],
    }
}
