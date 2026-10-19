use super::ring_gather_entries::{CUBIN, SYMBOL, THREADS};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use std::path::Path;

pub fn cubin_spec(bundle: &Path) -> CubinSpec {
    let mut params: Vec<_> = [
        ("source", "const uint8_t"),
        ("slots", "const uint32_t"),
        ("live", "const uint32_t"),
        ("destination", "uint8_t"),
    ]
    .into_iter()
    .map(|(name, cxx_type)| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    })
    .collect();
    params.extend(
        [
            "bytes",
            "source_stride",
            "destination_stride",
            "source_slots",
        ]
        .into_iter()
        .map(|name| EntryParam::U32 { name: name.into() }),
    );
    CubinSpec {
        name: CUBIN.into(), sources_root: bundle.into(), include_dirs: vec!["cuda".into()],
        headers: vec!["#include \"deepseek41_ring_gather.cuh\"".into()], defines: vec![],
        entries: vec![EntrySpec::new(SYMBOL.into(), params,
            "chop::deepseek41::ring_gather(source,slots,live,destination,bytes,source_stride,destination_stride,source_slots);".into(),
            THREADS, 1)],
    }
}
