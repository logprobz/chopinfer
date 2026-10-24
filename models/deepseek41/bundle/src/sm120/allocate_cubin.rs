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
    ];
    params.extend(
        [
            "item_capacity",
            "score_rows",
            "reserved_capacity",
            "width_capacity",
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
             starts,rows,expand,item_capacity,score_rows,reserved_capacity,width_capacity);"
                .into(),
            THREADS,
            1,
        )],
    }
}
