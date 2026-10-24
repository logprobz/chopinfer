use super::retain_rows_entries::{CUBIN, SYMBOL, THREADS};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use std::path::Path;

pub fn cubin_spec(bundle: &Path) -> CubinSpec {
    let mut params: Vec<_> = [
        ("source", "const uint8_t"),
        ("segments", "const uint32_t"),
        ("destination", "uint8_t"),
    ]
    .into_iter()
    .map(|(name, cxx_type)| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    })
    .collect();
    params.extend(
        ["row_bytes", "source_rows", "destination_rows"]
            .into_iter()
            .map(|name| EntryParam::U32 { name: name.into() }),
    );
    CubinSpec {
        name: CUBIN.into(), sources_root: bundle.into(), include_dirs: vec!["cuda".into()],
        headers: vec!["#include \"deepseek41_retain_rows.cuh\"".into()], defines: vec![],
        entries: vec![EntrySpec::new(SYMBOL.into(), params,
            "chop::deepseek41::retain_rows(source,segments,destination,row_bytes,source_rows,destination_rows);".into(), THREADS, 1)],
    }
}
