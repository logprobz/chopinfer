//! The existing lab timing gate and eviction bodies, without model probes.
use kernel_build::{CubinSpec, EntryParam as P, EntrySpec};
use std::path::PathBuf;

pub fn spec() -> CubinSpec {
    let ptr = |name: &str, ty: &str| P::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    };
    let u64p = |name: &str| P::U64 { name: name.into() };
    let u32p = |name: &str| P::U32 { name: name.into() };
    let entries = vec![
        EntrySpec::new(
            "chop_probe_gate_v1".into(),
            vec![ptr("flag", "const unsigned int")],
            "chop::probe::gate(flag);".into(),
            32,
            1,
        ),
        EntrySpec::new(
            "chop_probe_gate_release_v1".into(),
            vec![
                ptr("flag", "const unsigned int"),
                P::PointerArray {
                    name: "peers".into(),
                    slots: 8,
                    cxx_type: "unsigned int".into(),
                },
                u32p("count"),
            ],
            "chop::probe::gate_release(flag, peers, count);".into(),
            32,
            1,
        ),
        EntrySpec::new(
            "chop_probe_flush_v1".into(),
            vec![ptr("buffer", "unsigned int"), u64p("words")],
            "chop::probe::flush(buffer, words);".into(),
            256,
            1,
        ),
        EntrySpec::new(
            "chop_probe_read_v1".into(),
            vec![
                ptr("input", "const unsigned int"),
                ptr("checksum", "unsigned int"),
                u64p("words"),
                u32p("repeats"),
            ],
            "chop::probe::bandwidth_read(input, checksum, words, repeats);".into(),
            256,
            1,
        ),
    ];
    CubinSpec {
        name: "probe_sm120".into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."),
        include_dirs: vec![],
        headers: vec!["#include \"crates/kernel-lab/cuda/probes/probe.cuh\"".into()],
        defines: vec![],
        entries,
    }
}
