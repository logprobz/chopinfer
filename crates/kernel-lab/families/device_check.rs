//! The device-side poison and compare probe (`cuda/probes/device_check.cuh`)
//! that kernel-lab's `device_check` runs exhaustive live-count
//! qualification with.

use kernel_build::{CubinSpec, EntryParam as P, EntrySpec};
use std::path::PathBuf;

pub const FAMILY: &str = "device_check";
pub const CUBIN: &str = "probe_device_check_sm120";
pub const FILL: &str = "chop_probe_device_check_fill_v1";
pub const COMPARE: &str = "chop_probe_device_check_compare_v1";
/// Threads of a fill or compare CTA.
pub const THREADS: u32 = 256;

pub fn cubins() -> Vec<CubinSpec> {
    let pointer = |name: &str, cxx_type: &str| P::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    let u64 = |name: &str| P::U64 { name: name.into() };
    let u32 = |name: &str| P::U32 { name: name.into() };
    let spec = CubinSpec {
        name: CUBIN.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
        include_dirs: vec![],
        headers: vec!["#include \"kernel-lab/cuda/probes/device_check.cuh\"".into()],
        defines: vec![],
        entries: vec![
            EntrySpec::new(
                FILL.into(),
                vec![pointer("buffer", "uint8_t"), u64("bytes"), u32("word")],
                "chop::probe::device_check::fill(buffer, bytes, word);".into(),
                THREADS,
                1,
            ),
            EntrySpec::new(
                COMPARE.into(),
                vec![
                    pointer("want", "const uint32_t"),
                    pointer("got", "const uint4"),
                    u64("live_words"),
                    u64("vectors"),
                    u32("poison"),
                    u32("element_bytes"),
                    pointer("counts", "uint32_t"),
                    u32("slot"),
                ],
                "chop::probe::device_check::compare(want, got, live_words, vectors, poison, element_bytes, counts, slot);"
                    .into(),
                THREADS,
                1,
            ),
        ],
    };
    crate::declared(&[CUBIN], vec![spec])
}
