//! The Sinkhorn mHC comm epilogue split over CTAs
//! (`comm::sinkhorn_split::lab_cubin_specs`), and the PDL producer its
//! late-wait mutation runs behind.

use kernel_build::{CubinSpec, EntryParam as P, EntrySpec};
use std::path::PathBuf;

pub const FAMILY: &str = "sinkhorn_epilogue_split";

pub fn cubins() -> Vec<CubinSpec> {
    let mut specs = kernels_sm120::comm::sinkhorn_split::lab_cubin_specs();
    specs.push(pdl_change_watch_spec());
    specs
}

/// The PDL producer that holds its write until its dependent wrote a word
/// it writes only from that output
/// (`kernel-lab/cuda/probes/pdl_change_watch.cuh`): the split epilogue's
/// late-wait mutation runs behind it.
fn pdl_change_watch_spec() -> CubinSpec {
    let ptr = |name: &str, ty: &str| P::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    };
    CubinSpec {
        name: "probe_pdl_change_watch_sm120".into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
        include_dirs: vec![],
        headers: vec!["#include \"kernel-lab/cuda/probes/pdl_change_watch.cuh\"".into()],
        defines: vec![],
        entries: vec![EntrySpec::new(
            "chop_probe_pdl_change_watch_producer_v1".into(),
            vec![
                ptr("fresh", "const uint32_t"),
                ptr("output", "uint32_t"),
                P::U64 {
                    name: "words".into(),
                },
                ptr("watch", "const unsigned int"),
                P::U64 {
                    name: "hold_ns".into(),
                },
                ptr("record", "uint32_t"),
            ],
            "chop::probe::pdl_change_watch_producer(fresh,output,words,watch,hold_ns,record);"
                .into(),
            128,
            1,
        )],
    }
}
